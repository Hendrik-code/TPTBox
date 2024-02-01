from time import perf_counter
from tqdm import tqdm
from typing import Callable
from copy import deepcopy
import numpy as np


def speed_test_input(input, functions: list[Callable], assert_equal_function: Callable | None = None, *args, **kwargs):
    time_measures = {}
    outs = {}
    for f in functions:
        start = perf_counter()
        input_copy = deepcopy(input)
        out = f(input_copy, *args, **kwargs)
        time = perf_counter() - start
        outs[f.__name__] = out
        time_measures[f.__name__] = time

    if assert_equal_function is not None:
        for oname, o in outs.items():
            assertion = assert_equal_function(o, outs[functions[0].__name__])
            assert assertion, "speed_test: nonequal results given the assert_equal_function"
    return time_measures


def speed_test(
    get_input_func: Callable, functions: list[Callable], repeats: int = 20, assert_equal_function: Callable | None = None, *args, **kwargs
):
    time_sums = {}
    for i in tqdm(range(repeats)):
        input = get_input_func()
        time_measures = speed_test_input(input, functions=functions, assert_equal_function=assert_equal_function, *args, **kwargs)
        for k, v in time_measures.items():
            if k not in time_sums:
                time_sums[k] = 0
            time_sums[k] += v

    for k, v in time_sums.items():
        print(k, "\t", round(v / repeats, ndigits=6), "+-", round(np.std(v), ndigits=6))


if __name__ == "__main__":
    # speed test dilation
    from TPTBox.unit_tests.test_centroids import get_nii
    from TPTBox.core.np_utils import (
        generate_binary_structure,
        _binary_erosion,
        binary_erosion,
        _unpad,
        binary_dilation,
        _binary_dilation,
        np_erode_msk,
        np_dilate_msk,
        np_erode_msknew,
    )
    from time import perf_counter
    import numpy as np
    import random
    from tqdm import tqdm

    def get_nii_array():
        num_points = 0 if random.random() < 0.01 else 5
        nii, points, orientation, sizes = get_nii(x=(300, 300, 50), num_point=num_points)
        arr = nii.get_seg_array()
        arr_r = arr.copy()
        return arr_r

    speed_test(
        repeats=100,
        get_input_func=get_nii_array,
        functions=[np_erode_msk, np_erode_msknew],
        assert_equal_function=lambda x, y: np.count_nonzero(x) == np.count_nonzero(y),
        mm=10,
    )
    # print(time_measures)
