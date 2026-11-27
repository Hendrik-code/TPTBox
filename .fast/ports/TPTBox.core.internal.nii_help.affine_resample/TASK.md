# Port `TPTBox.core.internal.nii_help.affine_resample` to Rust

You're in a Rust crate fast-anything generated for this: `/DATA/NAS/ongoing_projects/hendrik/TPTBox/.fast/ports/TPTBox.core.internal.nii_help.affine_resample`.
Write `affine_resample` in `src/lib.rs` so it behaves **exactly** like the Python below, and runs fast.
Only edit files in this folder. Don't touch the Python project.

## The original (TPTBox/core/internal/nii_help.py:339)

```python
def affine_resample(
    arr: np.ndarray,
    matrix: NDArray[np.float64],
    offset: NDArray[np.float64],
    output_shape: tuple[int, ...],
    order: int,
    mode: str,
    cval: float,
    output_dtype: str | None,
) -> np.ndarray:
    """Nearest-neighbour or linear resampling through an affine map, as ``scipy.ndimage.affine_transform``.

    Output voxel ``o`` takes the value of ``arr`` at the voxel coordinate ``matrix @ o + offset``.

    Args:
        arr: Image to sample from, with a dtype from ``_RESAMPLE_DTYPES``.
        matrix: Square float64 matrix, ``arr.ndim`` on each side.
        offset: Float64 offset of length ``arr.ndim``.
        output_shape: Shape of the result, with ``arr.ndim`` entries.
        order: 0 (nearest neighbour) or 1 (linear).
        mode: ``"nearest"`` takes interpolation taps outside the image from the nearest edge voxel,
            ``"constant"`` gives ``cval`` for every coordinate outside ``[0, n - 1]``.
        cval: Value outside the image for ``mode="constant"``.
        output_dtype: Name of the result dtype from ``_RESAMPLE_DTYPES``, or None for the dtype of
            ``arr``. Integer results are rounded half away from zero and saturate at the dtype's
            limits; NaN and values beyond the range of the 32/64-bit types follow scipy's C casts.

    Returns:
        np.ndarray: The resampled image.

    Raises:
        TypeError: A dtype outside ``_RESAMPLE_DTYPES``, or ``matrix``/``offset`` not float64.
        ValueError: An ``order``, ``mode`` or shape outside the cases above, or an empty ``arr`` with
            ``mode="nearest"`` (scipy reads outside the array there).

    A Rust port of this function, verified call for call by fast-anything, replaces it when built
    (see the block at the end of this module and ``benchmarks/fast_anything/README.md``).
    """
    import scipy.ndimage as scipy_img

    if order not in (0, 1):
        raise ValueError(f"order must be 0 or 1, got {order}")
    if mode not in ("nearest", "constant"):
        raise ValueError(f"mode must be 'nearest' or 'constant', got {mode!r}")
    if arr.dtype.name not in _RESAMPLE_DTYPES or (output_dtype is not None and output_dtype not in _RESAMPLE_DTYPES):
        raise TypeError(f"unsupported dtypes {arr.dtype} -> {output_dtype}")
    if matrix.dtype != np.float64 or offset.dtype != np.float64:
        raise TypeError(f"matrix and offset must be float64, got {matrix.dtype} and {offset.dtype}")
    n = arr.ndim
    if n == 0 or matrix.shape != (n, n) or offset.shape != (n,) or len(output_shape) != n:
        raise ValueError(f"shapes do not fit a {n}D image: matrix {matrix.shape}, offset {offset.shape}, output {output_shape}")
    if arr.size == 0 and mode == "nearest":
        raise ValueError("mode='nearest' needs a non-empty image")
    output = None if output_dtype is None else np.dtype(output_dtype)
    return scipy_img.affine_transform(arr, matrix, offset, output_shape, order=order, mode=mode, cval=cval, output=output)  # type: ignore[return-value]
```

## Real calls

fast-anything recorded 10 real calls (out of 36 while the app ran). A few:

```
affine_resample(array(shape=(73, 47, 73), dtype=uint8), array(shape=(3, 3), dtype=float64), array([0.0, 0.0, …  ->  array(shape=(110, 70, 110), dtype=uint8)
affine_resample(array(shape=(73, 47, 73), dtype=uint8), array(shape=(3, 3), dtype=float64), array([0.0, 7.031…  ->  array(shape=(110, 70, 110), dtype=uint8)
affine_resample(array(shape=(73, 47, 73), dtype=uint8), array(shape=(3, 3), dtype=float64), array([0.0, 0.0, …  ->  array(shape=(110, 70, 110), dtype=uint8)
affine_resample(array(shape=(73, 47, 73), dtype=uint8), array(shape=(3, 3), dtype=float64), array([0.0, 7.031…  ->  array(shape=(110, 70, 110), dtype=uint8)
affine_resample(array(shape=(73, 47, 73), dtype=uint8), array(shape=(3, 3), dtype=float64), array([0.0, 7.031…  ->  array(shape=(110, 70, 110), dtype=uint8)
affine_resample(array(shape=(68, 52, 67), dtype=uint8), array(shape=(3, 3), dtype=float64), array([0.0, 0.0, …  ->  array(shape=(102, 78, 100), dtype=uint8)
affine_resample(array(shape=(68, 52, 67), dtype=uint8), array(shape=(3, 3), dtype=float64), array([-7.0312466…  ->  array(shape=(102, 78, 100), dtype=uint8)
affine_resample(array(shape=(68, 52, 67), dtype=uint8), array(shape=(3, 3), dtype=float64), array([0.0, 0.0, …  ->  array(shape=(102, 78, 100), dtype=uint8)
```

2 more recorded calls are held out: verify checks them too and reports a mismatch there without the values.

## Rules

- Same result for every input, edge cases included: empty input, zero, negatives, NaN and infinity, non-ASCII text, very long input. verify tests these too.
- Python `str` indexes by Unicode code point: use `.chars()` (or bytes only after checking `is_ascii()`).
- Python ints don't overflow. `i64` is fine unless the recorded values are bigger.
- Floats must match within 1e-9 of their size (or of the largest value in the same list), and zeros keep their sign: -0.0 is not 0.0. Keep the same order of float operations.
- Where Python raises, raise too (`PyValueError::new_err(...)`, `PyKeyError`, ...).
- Make it fast: no needless allocation or cloning, borrow inputs, `with_capacity`, fast paths for common cases.
- std only, unless a crate clearly pays off. Keep the function and module names as they are.
- Note: already uses np, less to gain.
- Note: NDArray parameters arrive as `numpy::PyReadonlyArrayDyn` (`.as_array()` gives an ndarray view, `.as_slice()` a slice when contiguous); return arrays as `Py<numpy::PyArrayDyn<T>>`, e.g. `numpy::ndarray::Array::from(vec).into_dyn().into_pyarray(py).unbind()` with `use numpy::IntoPyArray`. verify compares shape, dtype and values.

## Check

Run `cargo check` here. Then fast-anything builds the crate in release mode and replays all 12 recorded calls against it
(`fast-anything verify TPTBox.core.internal.nii_help.affine_resample`).
