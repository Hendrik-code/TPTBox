# noqa: INP001
import argparse
from pathlib import Path

import TPTBox.stitching.stitching_tools as st
from TPTBox import BIDS_FILE, BIDS_Global_info

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i", "--inputfolder", help="input folder (where the rawdata folder is located)", required=True)
arg_parser.add_argument("-p", "--outparant", help="input folder (where the rawdata folder is located)", default="rawdata_stitched")

args = arg_parser.parse_args()
print("args=%s" % args)
print("args.inputfolder=%s" % args.inputfolder)
BIDS_Global_info.remove_splitting_key("chunk")
bgi = BIDS_Global_info(datasets=[Path(args.inputfolder)])
print()
skipped = []
already_stitched = 0
new_stitched = 0
for name, subj in bgi.enumerate_subjects():
    q = subj.new_query()
    q.filter_format("vibe")
    q.filter("chunk", lambda _: True)  # chunk key must be present. Stiched images do not have a chunk key, so they are skipped
    files: dict[str, BIDS_FILE] = {}
    for fam in q.loop_dict(key_addendum=["part"]):
        for v in fam.values():
            part = v[0].get("part")
            out = v[0].get_changed_path(parent=args.outparant, info={"chunk": None, "sequ": "stitched"})
            try:
                if out.exists():
                    print(name, "exist", end="\r")
                    already_stitched += 1
                    continue
                st.stitching(*v, out=out)
                new_stitched += 1
            except BaseException:
                out.unlink(missing_ok=True)
                raise
print("These subject where skipped, because there are to many or to few files:", skipped)
print("Subject already stitched:", already_stitched)
print("Subject stitched:", new_stitched)
