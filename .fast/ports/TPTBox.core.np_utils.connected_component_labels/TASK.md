# Port `TPTBox.core.np_utils.connected_component_labels` to Rust

You're in a Rust crate fast-anything generated for this: `/DATA/NAS/ongoing_projects/hendrik/TPTBox/.fast/ports/TPTBox.core.np_utils.connected_component_labels`.
Write `connected_component_labels` in `src/lib.rs` so it behaves **exactly** like the Python below, and runs fast.
Only edit files in this folder. Don't touch the Python project.

## The original (TPTBox/core/np_utils.py:1489)

```python
def connected_component_labels(arr: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    """Connected components of a 2D or 3D label array, as ``cc3d.connected_components(..., return_N=True)``.

    Neighbouring voxels belong to the same component when they hold the same non-zero value (floats
    compare exactly, so NaN voxels never connect).

    Args:
        arr (np.ndarray): 2D or 3D bool, integer, float32 or float64 array.
        connectivity (int): 6, 18 or 26 in 3D; 4 or 8 in 2D, where 6 counts as 4 and 18 and 26 as 8.

    Returns:
        tuple[np.ndarray, int]: The components numbered 1..N in memory order (C order, or Fortran
        order for a Fortran-contiguous ``arr``) and N. The map is uint16 while cc3d's estimate of
        provisional labels (runs of equal non-zero values along the fastest axis) is below 65535, and
        uint32 or uint64 above. An empty ``arr`` gives an empty 1D array of its own dtype.

    Raises:
        TypeError: ``arr`` has an unsupported dtype.
        ValueError: ``arr`` is not 2D or 3D, or ``connectivity`` is not supported for its dimensionality.

    A Rust port of this function, verified call for call by fast-anything, replaces it when built
    (see the blocks at the end of this module and ``benchmarks/fast_anything/README.md``).
    """
    if arr.dtype.kind not in "biu" and arr.dtype not in (np.float32, np.float64):
        raise TypeError(f"expected a bool, integer, float32 or float64 array, got {arr.dtype}")
    if arr.ndim not in (2, 3):
        raise ValueError(f"expected 2D or 3D, but got {arr.ndim}")
    return _connected_components(arr, connectivity=connectivity, return_N=True)  # type: ignore[return-value]
```

## Real calls

fast-anything recorded 63 real calls (out of 78 while the app ran). A few:

```
connected_component_labels(array(shape=(73, 47, 73), dtype=uint8), 26)  ->  (array(shape=(73, 47, 73), dtype=uint16), 3)
connected_component_labels(array(shape=(73, 47, 73), dtype=uint8), 26)  ->  (array(shape=(73, 47, 73), dtype=uint16), 3)
connected_component_labels(array(shape=(69, 24, 45), dtype=uint8), 26)  ->  (array(shape=(69, 24, 45), dtype=uint16), 13)
connected_component_labels(array(shape=(69, 24, 45), dtype=uint8), 26)  ->  (array(shape=(69, 24, 45), dtype=uint16), 13)
connected_component_labels(array(shape=(71, 45, 71), dtype=uint8), 26)  ->  (array(shape=(71, 45, 71), dtype=uint16), 16)
connected_component_labels(array(shape=(72, 25, 71), dtype=uint8), 26)  ->  (array(shape=(72, 25, 71), dtype=uint16), 15)
connected_component_labels(array(shape=(72, 25, 71), dtype=uint8), 26)  ->  (array(shape=(72, 25, 71), dtype=uint16), 15)
connected_component_labels(array(shape=(73, 47, 73), dtype=uint8), 26)  ->  (array(shape=(73, 47, 73), dtype=uint16), 3)
```

15 more recorded calls are held out: verify checks them too and reports a mismatch there without the values.

## Rules

- Same result for every input, edge cases included: empty input, zero, negatives, NaN and infinity, non-ASCII text, very long input. verify tests these too.
- Python `str` indexes by Unicode code point: use `.chars()` (or bytes only after checking `is_ascii()`).
- Python ints don't overflow. `i64` is fine unless the recorded values are bigger.
- Floats must match within 1e-9 of their size (or of the largest value in the same list), and zeros keep their sign: -0.0 is not 0.0. Keep the same order of float operations.
- Where Python raises, raise too (`PyValueError::new_err(...)`, `PyKeyError`, ...).
- Make it fast: no needless allocation or cloning, borrow inputs, `with_capacity`, fast paths for common cases.
- std only, unless a crate clearly pays off. Keep the function and module names as they are.

## Check

Run `cargo check` here. Then fast-anything builds the crate in release mode and replays all 78 recorded calls against it
(`fast-anything verify TPTBox.core.np_utils.connected_component_labels`).
