from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from time import perf_counter

import numpy as np
from tqdm import tqdm


def speed_test_input(inp, functions: list[Callable], assert_equal_function: Callable | None = None, *args, **kwargs):
    time_measures = {}
    outs = {}
    for f in functions:
        start = perf_counter()
        input_copy = deepcopy(inp)
        out = f(*input_copy, *args, **kwargs) if isinstance(input_copy, (tuple, list)) else f(input_copy, *args, **kwargs)
        time = perf_counter() - start
        outs[f.__name__] = out
        time_measures[f.__name__] = time

    if assert_equal_function is not None and len(functions) > 1:
        for o in outs.values():
            assertion = assert_equal_function(o, outs[functions[0].__name__])
            assert assertion, f"speed_test: nonequal results given the assert_equal_function, got {o, outs[functions[0].__name__]}"
    return time_measures


def speed_test(
    get_input_func: Callable,
    functions: list[Callable],
    repeats: int = 20,
    assert_equal_function: Callable | None = None,
    *args,
    **kwargs,
):
    time_sums: dict[str, list[float]] = {"input_function": []}
    # print first iteration
    print()
    print("Print first speed test")
    start = perf_counter()
    first_input = get_input_func()
    time = perf_counter() - start
    time_sums["input_function"].append(time)
    for f in functions:
        input_copy = deepcopy(first_input)
        out = f(*input_copy, *args, **kwargs) if isinstance(input_copy, (tuple, list)) else f(input_copy, *args, **kwargs)
        print(f.__name__, out)

    for _ in tqdm(range(repeats)):
        inp = get_input_func()
        time_measures = speed_test_input(inp, *args, functions=functions, assert_equal_function=assert_equal_function, **kwargs)
        for k, v in time_measures.items():
            if k not in time_sums:
                time_sums[k] = []
            time_sums[k].append(v)

    for k, v in time_sums.items():
        print(k, "\t", round(sum(v) / repeats, ndigits=6), "+-", round(np.std(v), ndigits=6))
