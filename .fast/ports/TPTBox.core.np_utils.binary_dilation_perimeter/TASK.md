# Port `TPTBox.core.np_utils.binary_dilation_perimeter` to Rust

You're in a Rust crate fast-anything generated for this: `/DATA/NAS/ongoing_projects/hendrik/TPTBox/.fast/ports/TPTBox.core.np_utils.binary_dilation_perimeter`.
Write `binary_dilation_perimeter` in `src/lib.rs` so it behaves **exactly** like the Python below, and runs fast.
Only edit files in this folder. Don't touch the Python project.

## The original (TPTBox/core/np_utils.py:2659)

```python
def binary_dilation_perimeter(image: NDArray[np.bool_], selem: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Binary dilation of a 1D, 2D or 3D mask, stamping ``selem`` (centred) onto every perimeter voxel.

    Only voxels with an unset face neighbour inside the image (see :func:`_get_perimeter_image`) are
    stamped; for structuring elements that contain every voxel between their centre and any set
    voxel (all ``generate_binary_structure`` elements) this equals a full dilation.

    Args:
        image (NDArray[np.bool_]): Mask to dilate.
        selem (NDArray[np.bool_]): Structuring element with as many dimensions as ``image`` and an
            odd size along every axis.

    Returns:
        NDArray[np.bool_]: The dilated mask.

    Raises:
        TypeError: ``image`` or ``selem`` is not boolean.
        RuntimeError: ``image`` is not 1D, 2D or 3D.
        ValueError: ``selem`` differs from ``image`` in dimensionality or has an even size.

    A Rust port of this function, verified call for call by fast-anything, replaces it when built
    (see the blocks at the end of this module and ``benchmarks/fast_anything/README.md``).
    """
    if image.dtype != bool or selem.dtype != bool:
        raise TypeError(f"expected boolean arrays, got {image.dtype} and {selem.dtype}")
    dim = image.ndim
    if not 1 <= dim <= 3:
        raise RuntimeError("Only 1D, 2D and 3D images are supported.")
    if selem.ndim != dim:
        raise ValueError(f"structure element must have {dim} dimensions, got {selem.ndim}")
    if any(num_pixels % 2 == 0 for num_pixels in selem.shape):
        raise ValueError("Only structure element of odd dimension in each direction is supported.")
    perimeter_image = _get_perimeter_image(image)
    perimeter_coords = np.where(perimeter_image)
    out = image.copy()
    if dim == 1:
        sx = selem.shape[0]
        rx = sx // 2
        lx = image.shape[0]
        for ix in perimeter_coords[0]:
            (jx_b, jx_e), (kx_b, kx_e) = _generate_array_indices(ix, rx, sx, lx)
            out[jx_b:jx_e] |= selem[kx_b:kx_e]

    if dim == 2:
        rx, ry = (n // 2 for n in selem.shape)
        lx = image.shape
        sx, sy = selem.shape
        lx, ly = image.shape
        for ix, iy in zip(perimeter_coords[0], perimeter_coords[1]):
            (jx_b, jx_e), (kx_b, kx_e) = _generate_array_indices(ix, rx, sx, lx)
            (jy_b, jy_e), (ky_b, ky_e) = _generate_array_indices(iy, ry, sy, ly)
            out[jx_b:jx_e, jy_b:jy_e] |= selem[kx_b:kx_e, ky_b:ky_e]

    if dim == 3:
        rx, ry, rz = (n // 2 for n in selem.shape)
        sx, sy, sz = selem.shape
        lx, ly, lz = image.shape
        for ix, iy, iz in zip(perimeter_coords[0], perimeter_coords[1], perimeter_coords[2]):
            (jx_b, jx_e), (kx_b, kx_e) = _generate_array_indices(ix, rx, sx, lx)
            (jy_b, jy_e), (ky_b, ky_e) = _generate_array_indices(iy, ry, sy, ly)
            (jz_b, jz_e), (kz_b, kz_e) = _generate_array_indices(iz, rz, sz, lz)
            out[jx_b:jx_e, jy_b:jy_e, jz_b:jz_e] |= selem[kx_b:kx_e, ky_b:ky_e, kz_b:kz_e]
    return out
```

## Helpers it calls

Port these as private Rust functions in the same crate (they stay in Python for everyone else).

```python
def _generate_array_indices(selem_center, selem_radius, selem_length, result_length):
    """Return the correct indices for slicing considering near-edge regions.

    Args:
        selem_center (int): The index of the structuring element's center
        selem_radius (int): The radius of the structuring element
        selem_length (int): The length of the structuring element
        result_length (int): The length of the operating image

    Returns:
        (int, int): The range begin and end indices for the operating image
        (int, int): The range begin and end indices for the structuring element image
    """
    # First index for the result array
    result_begin = selem_center - selem_radius
    # Last index for the result array
    result_end = selem_center + selem_radius + 1
    # First index for the structuring element array
    selem_begin = -result_begin if result_begin < 0 else 0
    result_begin = max(0, result_begin)
    # Last index for the structuring element array
    selem_end = selem_length - (result_end - result_length) if result_end > result_length else selem_length
    return (result_begin, result_end), (selem_begin, selem_end)
```

```python
def _get_perimeter_image(image):
    """Return the image of the perimeter structure of the input image.

    Args:
        image (Numpy array): Image data as an array

    Returns:
        Numpy array: Perimeter image
    """
    dim = image.ndim
    if dim > 3:
        raise RuntimeError("Binary image in 4D or above is not supported.")
    count = np.zeros_like(image, dtype=np.uint8)
    inner = np.zeros_like(image, dtype=bool)

    count[1:] += image[:-1]
    count[:-1] += image[1:]

    if dim == 1:
        inner |= image == 2
        for i in [0, -1]:
            inner[i] |= count[i] == 1
        return image & (~inner)

    count[:, 1:] += image[:, :-1]
    count[:, :-1] += image[:, 1:]
    if dim == 2:
        inner |= count == 4
        for i in [0, -1]:
            inner[i] |= count[i] == 3
            inner[:, i] |= count[:, i] == 3
        for i in [0, -1]:
            for j in [0, -1]:
                inner[i, j] |= count[i, j] == 2
        return image & (~inner)

    count[:, :, 1:] += image[:, :, :-1]
    count[:, :, :-1] += image[:, :, 1:]

    if dim == 3:
        inner |= count == 6
        for i in [0, -1]:
            inner[i] |= count[i] == 5
            inner[:, i] |= count[:, i] == 5
            inner[:, :, i] |= count[:, :, i] == 5
        for i in [0, -1]:
            for j in [0, -1]:
                inner[i, j] |= count[i, j] == 4
                inner[:, i, j] |= count[:, i, j] == 4
                inner[:, i, j] |= count[:, i, j] == 4
                inner[i, :, j] |= count[i, :, j] == 4
                inner[i, :, j] |= count[i, :, j] == 4
        for i in [0, -1]:
            for j in [0, -1]:
                for k in [0, -1]:
                    inner[i, j, k] |= count[i, j, k] == 3
        return image & (~inner)
    raise RuntimeError("This line should not be reached.")
```

## Real calls

fast-anything recorded 399 real calls (out of 498 while the app ran). A few:

```
binary_dilation_perimeter(array(shape=(33, 47, 41), dtype=bool), array(shape=(3, 3, 3), dtype=bool))  ->  array(shape=(33, 47, 41), dtype=bool)
binary_dilation_perimeter(array(shape=(33, 47, 16), dtype=bool), array(shape=(3, 3, 3), dtype=bool))  ->  array(shape=(33, 47, 16), dtype=bool)
binary_dilation_perimeter(array(shape=(22, 47, 26), dtype=bool), array(shape=(3, 3, 3), dtype=bool))  ->  array(shape=(22, 47, 26), dtype=bool)
binary_dilation_perimeter(array(shape=(22, 47, 25), dtype=bool), array(shape=(3, 3, 3), dtype=bool))  ->  array(shape=(22, 47, 25), dtype=bool)
binary_dilation_perimeter(array(shape=(23, 44, 18), dtype=bool), array(shape=(3, 3, 3), dtype=bool))  ->  array(shape=(23, 44, 18), dtype=bool)
binary_dilation_perimeter(array(shape=(20, 47, 18), dtype=bool), array(shape=(3, 3, 3), dtype=bool))  ->  array(shape=(20, 47, 18), dtype=bool)
binary_dilation_perimeter(array(shape=(21, 47, 20), dtype=bool), array(shape=(3, 3, 3), dtype=bool))  ->  array(shape=(21, 47, 20), dtype=bool)
binary_dilation_perimeter(array(shape=(34, 47, 43), dtype=bool), array(shape=(3, 3, 3), dtype=bool))  ->  array(shape=(34, 47, 43), dtype=bool)
```

99 more recorded calls are held out: verify checks them too and reports a mismatch there without the values.

## Rules

- Same result for every input, edge cases included: empty input, zero, negatives, NaN and infinity, non-ASCII text, very long input. verify tests these too.
- Python `str` indexes by Unicode code point: use `.chars()` (or bytes only after checking `is_ascii()`).
- Python ints don't overflow. `i64` is fine unless the recorded values are bigger.
- Floats must match within 1e-9 of their size (or of the largest value in the same list), and zeros keep their sign: -0.0 is not 0.0. Keep the same order of float operations.
- Where Python raises, raise too (`PyValueError::new_err(...)`, `PyKeyError`, ...).
- Make it fast: no needless allocation or cloning, borrow inputs, `with_capacity`, fast paths for common cases.
- std only, unless a crate clearly pays off. Keep the function and module names as they are.
- Note: already uses np, less to gain.
- Note: calls _generate_array_indices, _get_perimeter_image, they get ported together with it.
- Note: NDArray parameters arrive as `numpy::PyReadonlyArrayDyn` (`.as_array()` gives an ndarray view, `.as_slice()` a slice when contiguous); return arrays as `Py<numpy::PyArrayDyn<T>>`, e.g. `numpy::ndarray::Array::from(vec).into_dyn().into_pyarray(py).unbind()` with `use numpy::IntoPyArray`. verify compares shape, dtype and values.

## Check

Run `cargo check` here. Then fast-anything builds the crate in release mode and replays all 498 recorded calls against it
(`fast-anything verify TPTBox.core.np_utils.binary_dilation_perimeter`).
