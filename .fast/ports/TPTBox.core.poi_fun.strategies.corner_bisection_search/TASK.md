# Port `TPTBox.core.poi_fun.strategies.corner_bisection_search` to Rust

You're in a Rust crate fast-anything generated for this: `/DATA/NAS/ongoing_projects/hendrik/TPTBox/.fast/ports/TPTBox.core.poi_fun.strategies.corner_bisection_search`.
Write `corner_bisection_search` in `src/lib.rs` so it behaves **exactly** like the Python below, and runs fast.
Only edit files in this folder. Don't touch the Python project.

## The original (TPTBox/core/poi_fun/strategies.py:209)

```python
def corner_bisection_search(
    region: NDArray[np.uint8],
    start: NDArray[np.float64],
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
    delta: float,
) -> NDArray[np.float64] | None:
    """Furthest point ``start + a * v1 + b * v2`` inside a binary mask, by bisection on ``a`` and ``b``.

    Inside means the trilinear interpolation of ``region`` at that point is above 0.5. First shrinks
    ``a`` and ``b`` from 1 until the point is inside, then grows them in halving steps until the
    mean step is below ``delta``.

    Args:
        region: 3D binary mask (0/1).
        start: Local voxel coordinate the search starts from, shape ``(3,)``.
        v1: First search direction, shape ``(3,)``.
        v2: Second search direction, shape ``(3,)``.
        delta: Stop once the mean step size falls below this.

    Returns:
        The point, shape ``(3,)``, or None when no point along the shrinking path is inside (``start``
        outside the mask). Where the search can no longer change state (``delta`` at or below the
        floating-point resolution) it returns the point reached.

    Raises:
        ValueError: A coordinate is NaN.

    A Rust port of this function, verified call for call by fast-anything, replaces it when built
    (see the block at the end of this module and ``benchmarks/fast_anything/README.md``).
    """
    factor1 = factor2 = 1
    interpolator = RegularGridInterpolator([np.arange(region.shape[i]) for i in range(3)], region)

    def is_inside(f1=0.0, f2=0.0):
        coords = start + (factor1 + f1) * v1 + (factor2 + f2) * v2
        if any(i < 0 for i in coords) or any(coords[i] > region.shape[i] - 1 for i in range(len(coords))):
            return False
        return interpolator(coords) > 0.5

    # Adjust factors until inside region
    while not is_inside():
        shrunk1, shrunk2 = factor1 * 0.9, factor2 * 0.7
        if shrunk1 == factor1 and shrunk2 == factor2:  # stuck at the smallest subnormal: start is outside
            return None
        factor1, factor2 = shrunk1, shrunk2
    v1_n = 1.0
    v2_n = 0.4
    f1 = f2 = 0

    # Refine factors using delta
    while delta < (v1_n + v2_n) / 2:
        state = (f1, f2, v1_n, v2_n)
        changed = False
        if is_inside(v1_n + f1, f2):
            f1 += v1_n
            changed = True
        if is_inside(f1, f2 + v2_n):
            f2 += v2_n
            changed = True
        if not changed:
            v1_n /= 2
            v2_n /= 2
        if (f1, f2, v1_n, v2_n) == state:  # steps absorbed by rounding: no further progress possible
            break

    return start + (factor1 + f1) * v1 + (factor2 + f2) * v2
```

## Real calls

fast-anything recorded 173 real calls (out of 216 while the app ran). A few:

```
corner_bisection_search(array(shape=(61, 20, 37), dtype=uint8), array([14.463032490305828, 1.657230771278585,…  ->  array([11.738330130934603, 1.1490980833528397e-8, 10.450007…
corner_bisection_search(array(shape=(61, 20, 37), dtype=uint8), array([14.463032490305828, 1.657230771278585,…  ->  array([24.49999998633734, 2.2438977368838513, 11.4955746833…
corner_bisection_search(array(shape=(61, 20, 37), dtype=uint8), array([14.463032490305828, 1.657230771278585,…  ->  array([1.736487943300678, 3.304613586296588, 8.976288691133…
corner_bisection_search(array(shape=(61, 20, 37), dtype=uint8), array([14.463032490305828, 1.657230771278585,…  ->  array([22.939868358125626, 7.26803654169105, 10.68309431992…
corner_bisection_search(array(shape=(61, 20, 37), dtype=uint8), array([13.016967509694169, 3.4427692287214144…  ->  array([24.499999948139813, 1.481984849815233, 25.7940738987…
corner_bisection_search(array(shape=(61, 20, 37), dtype=uint8), array([13.016967509694169, 3.4427692287214144…  ->  array([0.8575149848081458, 3.416919806871409, 23.0698374049…
corner_bisection_search(array(shape=(61, 20, 37), dtype=uint8), array([13.016967509694169, 3.4427692287214144…  ->  array([21.2469135996667, 7.499999979463585, 24.675997192703…
corner_bisection_search(array(shape=(61, 20, 37), dtype=uint8), array([13.739999999999998, 2.55, 17.43], dtyp…  ->  array([9.547456411631934, 9.9595790725715e-8, 17.3207534588…
```

43 more recorded calls are held out: verify checks them too and reports a mismatch there without the values.

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

Run `cargo check` here. Then fast-anything builds the crate in release mode and replays all 216 recorded calls against it
(`fast-anything verify TPTBox.core.poi_fun.strategies.corner_bisection_search`).
