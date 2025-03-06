from __future__ import annotations

# noqa: INP001
import argparse
import random
from pathlib import Path

from TPTBox.core.compat import zip_strict
import TPTBox.stitching.stitching_tools as st
from TPTBox import BIDS_FILE, BIDS_Global_info, Print_Logger
from TPTBox.core.bids_constants import sequence_splitting_keys

logger = Print_Logger()
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i", "--inputfolder", help="input folder (where the rawdata folder is located)", required=True)
arg_parser.add_argument("-p", "--outparant", help="input folder (where the rawdata folder is located)", default="rawdata_stitched")
arg_parser.add_argument("-r", "--rawdata", type=str, default="rawdata", help="the rawdata folder to be searched")
args = arg_parser.parse_args()
print(f"args={args}")
print(f"args.inputfolder={args.inputfolder}")
print(f"args.rawdata={args.rawdata}")

sequence_splitting_keys.remove("chunk")
bgi = BIDS_Global_info(datasets=[Path(args.inputfolder)], parents=[args.rawdata], sequence_splitting_keys=sequence_splitting_keys)
print()
skipped = []
skipped_single = []
already_stitched = 0
new_stitched = 0
l = list(bgi.enumerate_subjects())
random.shuffle(l)


def split_multi_scans(v: list[BIDS_FILE], out: BIDS_FILE):
    jsons = [x.open_json() for x in v]
    ids = [(j["SeriesNumber"], bids) for j, bids in zip_strict(jsons, v)]
    ids.sort()
    curr = []
    curr_id = []
    splits = []
    splits_c = []
    for _, bids in ids:
        chunk = bids.get("chunk")
        if chunk in curr_id:
            splits.append(curr)
            splits_c.append(curr_id)
            curr = []
            curr_id = []
        curr.append(bids)
        curr_id.append(chunk)
    splits.append(curr)
    splits_c.append(curr_id)

    # test if those are not patched scans
    multiple_full_scans = True
    for c in splits_c:
        if c != [str(s + 1) for s in list(range(len(c)))]:
            multiple_full_scans = False
            break
    if not multiple_full_scans:
        logger.on_warning(
            "[",
            v[0],
            "]; is patched with multiple partial scans. Some scans will probably have movement errors. Delete those duplicated files",
            splits_c,
        )
        return False
    else:
        for number, v_list in enumerate(splits[::-1], start=1):
            out_new = out.get_changed_path(parent=out.get_parent(), info={"sequ": "stitched", "nameconflict": None, "run": str(number)})
            if out_new.exists():
                continue
            st.stitching(*v_list, out=out_new)
        return True


for name, subj in l:
    q = subj.new_query()
    q.filter_format("vibe")
    q.filter("chunk", lambda _: True)  # chunk key must be present. Stiched images do not have a chunk key, so they are skipped
    files: dict[str, BIDS_FILE] = {}
    for fam in q.loop_dict(key_addendum=["part"]):
        for v in fam.values():
            part = v[0].get("part")
            out = v[0].get_changed_bids(parent=args.outparant, info={"chunk": None, "sequ": "stitched"})
            # Check if there are multiple scans
            ids = {}
            skip = False
            for b in v:
                key = str(b.get("chunk")) + "-" + str(b.get("part"))
                ids.setdefault(key, 0)
                ids[key] += 1
                if ids[key] >= 2:
                    skip = True
            if skip:
                succ = split_multi_scans(v, out)
                if not succ:
                    skipped.append(name)

                continue
            try:
                if out.exists():
                    print(name, "exist", end="\r")
                    already_stitched += 1
                    continue
                if len(v) == 1:
                    skipped_single.append(name)
                    continue
                st.stitching(v, out=out)
                new_stitched += 1
            except BaseException:
                out.unlink(missing_ok=True)
                raise


skipped = set(skipped)
skipped_single = set(skipped_single)
c = len(skipped) + len(skipped_single)
logger.on_warning("These subject where skipped, because there multiple scans:", list(skipped)) if len(skipped) != 0 else None
if len(skipped_single) != 0:
    logger.on_warning("These subject where skipped, because there ins only a single scans:", list(skipped_single))
logger.on_warning("Subject skipped:", c) if c != 0 else None
print("Images already stitched:", already_stitched) if already_stitched != 0 else None
print("Images stitched:", new_stitched)
