if __name__ == "__main__":
    # speed test Searchquery.filter (flatten mode): candidates.copy() + list.remove (O(n^2))
    # vs a single list comprehension (O(n)). Built on real BIDS_FILE candidates.
    import tempfile
    from pathlib import Path

    from TPTBox.core.bids_files import BIDS_FILE
    from TPTBox.tests.speedtests.speedtest import speed_test

    # build N real BIDS_FILE candidates once (touches a tempdir; not part of the timed comparison)
    tmp = Path(tempfile.mkdtemp())
    formats = ["T1w", "T2w", "ct", "msk", "dixon"]
    N = 100
    CANDIDATES = []
    for i in range(N):
        fmt = formats[i % len(formats)]
        sub = f"sub-{i:04d}"
        p = tmp / sub / f"{sub}_ses-01_sequ-{i}_{fmt}.nii.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        CANDIDATES.append(BIDS_FILE(p, tmp, verbose=False))

    keep = ["T1w"]  # keep 1/5, so the old loop removes ~80% of candidates (worst case for list.remove)

    def old_filter(cands):
        cands = list(cands)
        for bids_file in cands.copy():
            if not bids_file.do_filter("format", keep, required=True):
                cands.remove(bids_file)
        return cands

    def new_filter(cands):
        return [bids_file for bids_file in cands if bids_file.do_filter("format", keep, required=True)]

    def get_input():
        return (CANDIDATES,)  # 1-tuple so the harness passes the list as a single argument

    print(f"\n=== Searchquery.filter flatten mode ({N} candidates, keep {keep}) ===")
    speed_test(
        repeats=25,
        get_input_func=get_input,
        functions=[old_filter, new_filter],
        assert_equal_function=lambda x, y: [f.BIDS_key for f in x] == [f.BIDS_key for f in y],
    )
