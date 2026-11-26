"""Save or compare every POI the workload computes, to check that a speed change leaves results bit-identical.

Run ``--save ref.json`` on the reference code, then ``--compare ref.json`` on the changed code
(both as ``python -m benchmarks.fast_anything.check_pois`` from the repo root).
"""

from __future__ import annotations

import argparse
import json

from TPTBox import calc_poi_from_subreg_vert, to_nii

from .workload import SAMPLES, all_poi_functions


def compute() -> dict[str, list[list]]:
    subreg_ids = list(range(40, 51)) + sorted(all_poi_functions)
    out = {}
    for name, (vert_path, subreg_path) in SAMPLES.items():
        poi = calc_poi_from_subreg_vert(to_nii(vert_path, True), to_nii(subreg_path, True), subreg_id=subreg_ids, verbose=False)
        out[name] = sorted([int(r), int(s), [float(c) for c in xyz]] for r, s, xyz in poi.items())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save")
    group.add_argument("--compare")
    args = parser.parse_args()
    got = compute()
    if args.save:
        with open(args.save, "w") as f:
            json.dump(got, f)
        print(f"saved {sum(len(v) for v in got.values())} points to {args.save}")
        return
    with open(args.compare) as f:
        ref = json.load(f)
    bad = 0
    for name in ref:
        a, b = {(r, s): xyz for r, s, xyz in ref[name]}, {(r, s): xyz for r, s, xyz in got[name]}
        for key in sorted(a.keys() | b.keys()):
            if a.get(key) != b.get(key):
                bad += 1
                print(f"{name} {key}: ref={a.get(key)} got={b.get(key)}")
    n = sum(len(v) for v in ref.values())
    print(f"{n - bad}/{n} points bit-identical" if not bad else f"MISMATCH: {bad} of {n} points differ")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
