# Port `TPTBox.core.np_utils.euclidean_distance_transform` to Rust

You're in a Rust crate fast-anything generated for this: `/DATA/NAS/ongoing_projects/hendrik/TPTBox/.fast/ports/TPTBox.core.np_utils.euclidean_distance_transform`.
Write `euclidean_distance_transform` in `src/lib.rs` so it behaves **exactly** like the Python below, and runs fast.
Only edit files in this folder. Don't touch the Python project.

## The original (TPTBox/core/np_utils.py:531)

```python
def euclidean_distance_transform(
    mask: NDArray[np.bool_],
    sampling: Sequence[float] | None = None,
    return_distances: bool = True,
    return_indices: bool = False,
) -> tuple[NDArray[np.float64] | None, NDArray[np.int32] | None]:
    """Exact Euclidean distance transform of a boolean mask, as ``scipy.ndimage.distance_transform_edt``.

    For every voxel, the distance to the nearest ``False`` voxel and/or that voxel's index (the
    feature transform). Ties go to the voxel scipy picks.

    Args:
        mask (NDArray[np.bool_]): Mask of at least one dimension; distances are measured to its ``False`` voxels.
        sampling (Sequence[float] | None, optional): Voxel spacing per axis, or a single float for all
            axes. Defaults to None (isotropic unit spacing).
        return_distances (bool, optional): Compute the distances. Defaults to True.
        return_indices (bool, optional): Compute the feature transform. Defaults to False.

    Returns:
        tuple: ``(distances, indices)``: float64 distances shaped like ``mask`` and int32 indices of
        shape ``(mask.ndim, *mask.shape)``, each None when not requested.

    Raises:
        TypeError: ``mask`` is not a boolean array.
        ValueError: ``mask`` is 0-dimensional.
        RuntimeError: Neither output was requested, or ``sampling`` does not have one entry per axis.

    A Rust port of this function, verified call for call by fast-anything, replaces it when built
    (see the blocks at the end of this module and ``benchmarks/fast_anything/README.md``).
    """
    if mask.dtype != bool:
        raise TypeError(f"expected a boolean mask, got {mask.dtype}")
    if mask.ndim == 0:
        raise ValueError("expected a mask with at least one dimension")
    result = distance_transform_edt(mask, sampling=sampling, return_distances=return_distances, return_indices=return_indices)
    if return_distances and return_indices:
        return result  # type: ignore[return-value]
    return (result, None) if return_distances else (None, result)  # type: ignore[return-value]
```

## Real calls

fast-anything recorded 63 real calls (out of 117 while the app ran). A few:

```
euclidean_distance_transform(array(shape=(67, 23, 43), dtype=bool), return_indices=True)  ->  (array(shape=(67, 23, 43), dtype=float64), array(shape=(3, …
euclidean_distance_transform(array(shape=(65, 22, 41), dtype=bool))  ->  (array(shape=(65, 22, 41), dtype=float64), None)
euclidean_distance_transform(array(shape=(69, 24, 45), dtype=bool), return_distances=True, return_indices=Tru…  ->  (array(shape=(69, 24, 45), dtype=float64), array(shape=(3, …
euclidean_distance_transform(array(shape=(70, 23, 67), dtype=bool))  ->  (array(shape=(70, 23, 67), dtype=float64), None)
euclidean_distance_transform(array(shape=(65, 22, 41), dtype=bool))  ->  (array(shape=(65, 22, 41), dtype=float64), None)
euclidean_distance_transform(array(shape=(69, 24, 45), dtype=bool), return_distances=True, return_indices=Tru…  ->  (array(shape=(69, 24, 45), dtype=float64), array(shape=(3, …
euclidean_distance_transform(array(shape=(70, 23, 67), dtype=bool))  ->  (array(shape=(70, 23, 67), dtype=float64), None)
euclidean_distance_transform(array(shape=(67, 23, 43), dtype=bool), return_indices=True)  ->  (array(shape=(67, 23, 43), dtype=float64), array(shape=(3, …
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
- Note: NDArray parameters arrive as `numpy::PyReadonlyArrayDyn` (`.as_array()` gives an ndarray view, `.as_slice()` a slice when contiguous); return arrays as `Py<numpy::PyArrayDyn<T>>`, e.g. `numpy::ndarray::Array::from(vec).into_dyn().into_pyarray(py).unbind()` with `use numpy::IntoPyArray`. verify compares shape, dtype and values.

## Check

Run `cargo check` here. Then fast-anything builds the crate in release mode and replays all 78 recorded calls against it
(`fast-anything verify TPTBox.core.np_utils.euclidean_distance_transform`).
