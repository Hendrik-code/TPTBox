if __name__ == "__main__":
    # speed test np.isin(arr, labels) for uint segmentation masks:
    #   default vs kind='table' vs explicit boolean lookup-table (LUT).
    # numpy auto-selects kind='table' for small integer ranges, so the explicit LUT may only
    # help for wide-range uint16 (labels near dtype max) where numpy declines the table.
    import numpy as np

    from TPTBox.tests.speedtests.speedtest import speed_test

    def isin_default(arr, labels):
        return np.isin(arr, labels)

    def isin_table(arr, labels):
        return np.isin(arr, labels, kind="table")

    def lut_explicit(arr, labels):
        if len(labels) == 0:
            return np.zeros(arr.shape, dtype=bool)
        m = max(int(arr.max()), int(max(labels))) + 1
        lut = np.zeros(m, dtype=bool)
        lut[labels] = True
        return lut[arr]

    functions = [isin_default, isin_table, lut_explicit]
    eq = lambda x, y: np.array_equal(x, y)  # noqa: E731

    def make(dtype, maxval, n_labels, shape=(256, 256, 256)):
        def _f():
            arr = np.random.randint(0, maxval, size=shape).astype(dtype)
            labels = list(np.random.choice(np.arange(1, maxval), size=min(n_labels, maxval - 1), replace=False))
            labels = [int(x) for x in labels]
            return (arr, labels)

        return _f

    regimes = [
        ("uint8  range<=30   n_labels=5", np.uint8, 30, 5),
        ("uint8  range<=200  n_labels=50", np.uint8, 200, 50),
        ("uint16 range<=30   n_labels=5", np.uint16, 30, 5),
        ("uint16 wide<=60000 n_labels=1", np.uint16, 60000, 1),
        ("uint16 wide<=60000 n_labels=5", np.uint16, 60000, 5),
        ("uint16 wide<=60000 n_labels=50", np.uint16, 60000, 50),
    ]
    for name, dtype, maxval, n_labels in regimes:
        print(f"\n=== np.isin | {name} (256^3) ===")
        speed_test(repeats=12, get_input_func=make(dtype, maxval, n_labels), functions=functions, assert_equal_function=eq)
