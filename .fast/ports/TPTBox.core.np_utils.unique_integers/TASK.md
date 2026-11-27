# Port `TPTBox.core.np_utils.unique_integers` to Rust

You're in a Rust crate fast-anything generated for this: `/DATA/NAS/ongoing_projects/hendrik/TPTBox/.fast/ports/TPTBox.core.np_utils.unique_integers`.
Write `unique_integers` in `src/lib.rs` so it behaves **exactly** like the Python below, and runs fast.
Only edit files in this folder. Don't touch the Python project.

## The original (TPTBox/core/np_utils.py:250)

```python
def unique_integers(arr: np.ndarray) -> list[int]:
    """Sorted distinct values of an integer array, as Python ints.

    Unsigned arrays whose maximum is below 2**20 are counted with ``np.bincount``, which is several
    times faster than ``np.unique`` on label maps; everything else uses ``np.unique``.

    Args:
        arr (np.ndarray): Integer array of any shape.

    Returns:
        list[int]: The distinct values, ascending.

    Raises:
        TypeError: ``arr`` is not an integer array.
        ValueError: ``arr`` is an empty unsigned array (an empty signed array gives ``[]``).

    A Rust port of this function, verified call for call by fast-anything, replaces it when built
    (see the blocks at the end of this module and ``benchmarks/fast_anything/README.md``).
    """
    if arr.dtype.kind not in "iu":
        raise TypeError(f"expected an integer array, got {arr.dtype}")
    if arr.dtype.kind == "u":
        max_val = int(arr.max())
        if max_val < 2**20:  # ~1M labels threshold — bincount stays fast
            counts = np.bincount(arr.ravel())
            return np.where(counts > 0)[0].tolist()
    return np.unique(arr).tolist()
```

## Real calls

fast-anything recorded 207 real calls (out of 258 while the app ran). A few:

```
unique_integers(array(shape=(73, 47, 73), dtype=uint8))  ->  [0, 21, 22, 23]
unique_integers(array(shape=(73, 47, 73), dtype=uint8))  ->  [0, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
unique_integers(array(shape=(73, 47, 73), dtype=uint8))  ->  [0, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
unique_integers(array(shape=(73, 47, 73), dtype=uint8))  ->  [0, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
unique_integers(array(shape=(73, 47, 73), dtype=uint8))  ->  [0, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
unique_integers(array(shape=(73, 47, 73), dtype=uint8))  ->  [0, 21, 22, 23]
unique_integers(array(shape=(73, 47, 73), dtype=uint8))  ->  [0, 21, 22, 23]
unique_integers(array(shape=(73, 47, 73), dtype=int8))  ->  [0, 21, 22, 23]
```

51 more recorded calls are held out: verify checks them too and reports a mismatch there without the values.

## Rules

- Same result for every input, edge cases included: empty input, zero, negatives, NaN and infinity, non-ASCII text, very long input. verify tests these too.
- Python `str` indexes by Unicode code point: use `.chars()` (or bytes only after checking `is_ascii()`).
- Python ints don't overflow. `i64` is fine unless the recorded values are bigger.
- Floats must match within 1e-9 of their size (or of the largest value in the same list), and zeros keep their sign: -0.0 is not 0.0. Keep the same order of float operations.
- Where Python raises, raise too (`PyValueError::new_err(...)`, `PyKeyError`, ...).
- Make it fast: no needless allocation or cloning, borrow inputs, `with_capacity`, fast paths for common cases.
- std only, unless a crate clearly pays off. Keep the function and module names as they are.
- Note: already uses np, less to gain.

## Check

Run `cargo check` here. Then fast-anything builds the crate in release mode and replays all 258 recorded calls against it
(`fast-anything verify TPTBox.core.np_utils.unique_integers`).
