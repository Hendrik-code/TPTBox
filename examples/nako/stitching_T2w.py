# noqa: INP001
import argparse
import time
from pathlib import Path

import TPTBox.stitching.stitching_tools as st
from TPTBox import BIDS_FILE, BIDS_Global_info, Print_Logger

logger = Print_Logger()
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i", "--inputfolder", help="input folder (where the rawdata folder is located)", required=True)
arg_parser.add_argument("-p", "--outparant", help="input folder (where the rawdata folder is located)", default="rawdata_stitched")
arg_parser.add_argument("-s", "--sleep", type=float, default=0, help="sleep after each save")
arg_parser.add_argument("-r", "--rawdata", type=str, default="rawdata", help="the rawdata folder to be searched")
args = arg_parser.parse_args()
print("args=%s" % args)
print("args.inputfolder=%s" % args.inputfolder)
print("args.outparant=%s" % args.outparant)
print("args.rawdata=%s" % args.rawdata)

bgi = BIDS_Global_info(datasets=[Path(args.inputfolder)], parents=[args.rawdata])
print()
skipped = []
already_stitched = 0
new_stitched = 0
for name, subj in bgi.enumerate_subjects(sort=True):
    q = subj.new_query()
    q.filter_format("T2w")
    q.filter("chunk", lambda x: str(x) in ["HWS", "BWS", "LWS"])
    q.flatten()
    files: dict[str, BIDS_FILE] = {}
    to_many = False
    c = 0
    for chunk in ["HWS", "BWS", "LWS"]:
        q_tmp = q.copy()
        q_tmp.filter("chunk", chunk)
        l_t2w = list(q_tmp.loop_list())
        c += len(l_t2w)
        if len(l_t2w) != 1:
            to_many = True
            continue
        files[chunk] = l_t2w[0]
    if to_many:
        if c != 0:
            continue
        skipped.append((name, c))
        continue
    out = files["HWS"].get_changed_path(info={"chunk": None, "sequ": "stitched"}, parent=args.outparant)
    try:
        if out.exists():
            print(name, "exist", end="\r")
            already_stitched += 1
            continue
        print("Stich", out)
        nii = st.GNC_stitch_T2w(files["HWS"], files["BWS"], files["LWS"])
        crop = nii.compute_crop()
        nii.apply_crop_(crop)
        nii.save(out)
        if args.sleep != 0:
            logger.print(f"Sleepy time {args.sleep} s; stitched {new_stitched}")
            time.sleep(args.sleep)
        new_stitched += 1
    except BaseException:
        out.unlink(missing_ok=True)
        skipped.append(name + "_FAIL")
        # raise
print("These subject where skipped, because there are to many or to few files:", skipped)
print("Subject skipped:", len(skipped))
print("Subject already stitched:", already_stitched)
print("Subject stitched:", new_stitched)
