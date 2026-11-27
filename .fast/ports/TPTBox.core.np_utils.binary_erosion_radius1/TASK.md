# Port `TPTBox.core.np_utils.binary_erosion_radius1` to Rust

You're in a Rust crate fast-anything generated for this: `/DATA/NAS/ongoing_projects/hendrik/TPTBox/.fast/ports/TPTBox.core.np_utils.binary_erosion_radius1`.
Write `binary_erosion_radius1` in `src/lib.rs` so it behaves **exactly** like the Python below, and runs fast.
Only edit files in this folder. Don't touch the Python project.

## The original (TPTBox/core/np_utils.py:938)

```python
def binary_erosion_radius1(image: NDArray[np.bool_], struct: NDArray[np.bool_], iterations: int, border_value: int) -> NDArray[np.bool_]:
    """Binary erosion with a structuring element of size 1 or 3 along every axis, as ``scipy.ndimage.binary_erosion``.

    A voxel stays set when every voxel the structuring element (centred on it) covers is set.
    Voxels outside the image count as set when ``border_value`` is truthy.

    Only sizes 1 and 3 are accepted, which covers every ``generate_binary_structure`` element
    (also with an axis expanded for slice-wise erosion). For wider elements scipy's iterated erosion
    is not plain repeated erosion next to the border when ``border_value`` is set.

    Args:
        image (NDArray[np.bool_]): Mask to erode, at least one dimension.
        struct (NDArray[np.bool_]): Structuring element with as many dimensions as ``image``.
        iterations (int): Number of erosions; below 1, erode until nothing changes.
        border_value (int): Value of the voxels outside the image.

    Returns:
        NDArray[np.bool_]: The eroded mask.

    Raises:
        TypeError: ``image`` or ``struct`` is not boolean.
        RuntimeError: ``struct`` and ``image`` differ in dimensionality.
        ValueError: ``image`` is 0-dimensional or ``struct`` has an axis of a size other than 1 or 3.

    A Rust port of this function, verified call for call by fast-anything, replaces it when built
    (see the blocks at the end of this module and ``benchmarks/fast_anything/README.md``).
    """
    if image.dtype != bool or struct.dtype != bool:
        raise TypeError(f"expected boolean arrays, got {image.dtype} and {struct.dtype}")
    if image.ndim == 0:
        raise ValueError("expected an image with at least one dimension")
    if struct.ndim != image.ndim:
        raise RuntimeError("structure and input must have same dimensionality")
    if any(s not in (1, 3) for s in struct.shape):
        raise ValueError(f"structuring element must have size 1 or 3 along every axis, got {struct.shape}")
    return binary_erosion(image, structure=struct, iterations=iterations, border_value=border_value)
```

## Real calls

fast-anything recorded 255 real calls (out of 318 while the app ran). A few:

```
binary_erosion_radius1(array(shape=(29, 47, 37), dtype=bool), array(shape=(3, 3, 3), dtype=bool), 1, 0)  ->  array(shape=(29, 47, 37), dtype=bool)
binary_erosion_radius1(array(shape=(30, 47, 12), dtype=bool), array(shape=(3, 3, 3), dtype=bool), 1, 0)  ->  array(shape=(30, 47, 12), dtype=bool)
binary_erosion_radius1(array(shape=(18, 47, 23), dtype=bool), array(shape=(3, 3, 3), dtype=bool), 1, 0)  ->  array(shape=(18, 47, 23), dtype=bool)
binary_erosion_radius1(array(shape=(18, 45, 22), dtype=bool), array(shape=(3, 3, 3), dtype=bool), 1, 0)  ->  array(shape=(18, 45, 22), dtype=bool)
binary_erosion_radius1(array(shape=(19, 41, 14), dtype=bool), array(shape=(3, 3, 3), dtype=bool), 1, 0)  ->  array(shape=(19, 41, 14), dtype=bool)
binary_erosion_radius1(array(shape=(16, 46, 14), dtype=bool), array(shape=(3, 3, 3), dtype=bool), 1, 0)  ->  array(shape=(16, 46, 14), dtype=bool)
binary_erosion_radius1(array(shape=(17, 46, 16), dtype=bool), array(shape=(3, 3, 3), dtype=bool), 1, 0)  ->  array(shape=(17, 46, 16), dtype=bool)
binary_erosion_radius1(array(shape=(31, 47, 39), dtype=bool), array(shape=(3, 3, 3), dtype=bool), 1, 0)  ->  array(shape=(31, 47, 39), dtype=bool)
```

63 more recorded calls are held out: verify checks them too and reports a mismatch there without the values.

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

Run `cargo check` here. Then fast-anything builds the crate in release mode and replays all 318 recorded calls against it
(`fast-anything verify TPTBox.core.np_utils.binary_erosion_radius1`).
