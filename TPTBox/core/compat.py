from __future__ import annotations

from collections.abc import Iterator


def zip_strict(*iterables) -> zip:
    """A strict version of zip that raises a ValueError if the input iterables have different lengths.

    Converts each iterable to a list to check lengths. This assumes all iterables are finite.

    Args:
        *iterables: Finite iterables to be zipped together.

    Returns:
        An iterator of tuples, where the i-th tuple contains the i-th element from each iterable.

    Raises:
        ValueError: If the input iterables have different lengths.
    """
    lists = [list(it) for it in iterables]
    lengths = [len(lst) for lst in lists]
    if len(set(lengths)) != 1:
        raise ValueError(f"Length mismatch: {lengths}")
    return zip(*lists)


def pairwise(iterable) -> Iterator:
    """Yield successive overlapping pairs from ``iterable``.

    Backport of :func:`itertools.pairwise`, which is only available from Python
    3.10 onwards while this package still supports 3.9. On 3.10+ the stdlib
    implementation is used directly.

    Args:
        iterable: Any iterable. It is consumed lazily, so generators are fine.

    Returns:
        An iterator over ``(s[0], s[1]), (s[1], s[2]), ...``. Empty for inputs
        with fewer than two elements.
    """
    try:
        from itertools import pairwise as _pairwise  # noqa: PLC0415
    except ImportError:
        return _pairwise_py39(iterable)
    return _pairwise(iterable)


def _pairwise_py39(iterable):
    """Pure-python ``pairwise`` used on Python 3.9."""
    it = iter(iterable)
    try:
        prev = next(it)
    except StopIteration:
        return
    for cur in it:
        yield prev, cur
        prev = cur
