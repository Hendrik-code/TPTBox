from __future__ import annotations

from functools import wraps
from time import time

import numpy as np

from TPTBox import POI, Location, Logger_Interface, Print_Logger
from TPTBox.core.compat import zip_strict
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.vert_constants import Vertebra_Instance

_log = Print_Logger()
sacrum_w_o_arcus = (Vertebra_Instance.COCC.value, Vertebra_Instance.S6.value, Vertebra_Instance.S5.value, Vertebra_Instance.S4.value)
sacrum_w_o_direction = (Vertebra_Instance.COCC.value,)


def to_local_np(
    loc: Location,
    bb: tuple[slice, slice, slice] | None,
    poi: POI,
    label: int,
    log: Logger_Interface,
    verbose: bool = True,
) -> np.ndarray | None:
    """Retrieve a POI coordinate in local (bounding-box) voxel space.

    Looks up ``(label, loc.value)`` in ``poi``.  When a bounding box ``bb`` is
    provided the global coordinate is shifted to the local sub-volume origin by
    subtracting each slice's start.

    Args:
        loc: ``Location`` enum member identifying the subregion.
        bb: Bounding-box slices used to convert global to local coordinates.
            Pass ``None`` to return the raw global coordinate.
        poi: ``POI`` object containing the landmark data.
        label: Region (vertebra) integer label.
        log: Logger used for failure messages.
        verbose: Emit a failure log message when the POI is missing.
            Defaults to ``True``.

    Returns:
        1-D numpy array ``(x, y, z)`` in local coordinates, or ``None`` when
        the requested POI is not present in ``poi``.
    """
    if (label, loc.value) in poi:
        if bb is None:
            return np.asarray(poi[label, loc.value])
        return np.asarray([a - b.start for a, b in zip_strict(poi[label, loc.value], bb)])
    if verbose:
        log.on_fail(f"region={label},subregion={loc.value} is missing")
        # raise KeyError(f"region={label},subregion={loc.value} is missing.")
    return None


def paint_into_NII(
    poi: POI,
    a: NII,
    l: list[Location] | None = None,
    idxs: list[int] | None = None,
    rays: list[tuple[Location, Location]] | None = None,
) -> NII:
    """Paint POI landmarks and connecting rays into a segmentation NII image.

    Draws centroid markers for each location in ``l`` and (optionally) rays
    connecting pairs of POI locations onto ``a``.

    Args:
        poi: ``POI`` object with the landmarks to draw.
        a: Target ``NII`` image that will be modified in place.
        l: List of ``Location`` members whose centroids are drawn.  Defaults to
            ``[Location.Vertebra_Disc_Inferior, Location.Vertebra_Disc_Superior]``.
        idxs: Region (vertebra) labels to include.  Defaults to all regions
            in ``poi``, sorted.
        rays: List of ``(start_loc, end_loc)`` pairs that define direction rays
            to paint.  Defaults to IVD disc start/end connections.

    Returns:
        The modified ``NII`` image ``a`` with landmarks and rays painted in.
    """
    from TPTBox.core.poi_fun.ray_casting import add_ray_to_img

    if l is None:
        l = [Location.Vertebra_Disc_Inferior, Location.Vertebra_Disc_Superior]
    if rays is None:
        rays = [(Location.Vertebra_Disc, Location.Vertebra_Disc_Inferior), (Location.Vertebra_Disc, Location.Vertebra_Disc_Superior)]
    if idxs is None:
        idxs = poi.keys_region(sort=True)

    assert a is not None
    spline = a.copy() * 0
    spline.rescale_()
    poi_r = poi.rescale()
    for loc in l:
        for idx in idxs:
            if (idx, loc) not in poi_r:
                continue
            x, y, z = poi_r[idx, loc]
            spline[round(x), round(y), round(z)] = loc.value
    spline.dilate_msk_(2)
    spline.resample_from_to_(a)
    a[spline != 0] = spline[spline != 0]
    for start, goal in rays:
        for idx in idxs:
            try:
                assert a is not None
                direction = np.array(poi[idx, goal]) - np.array(poi[idx, start])
                if abs(direction.sum().item()) < 0.000000000001:
                    print("skip", idx, goal, "-", start)
                    continue
                a = add_ray_to_img(poi[idx, start], direction, a, True, value=199, dilate=2)  # type: ignore
            except KeyError:
                pass
    return a


def timing(f):
    """Decorator that logs the wall-clock execution time of the wrapped function."""

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        _log.on_neutral(f"func:{f.__name__!r} took: {te - ts:2.4f} sec")
        return result

    return wrap


def make_spine_plot(pois: POI, body_spline: np.ndarray, vert_nii: NII, filenames: str) -> None:
    """Render a 2-D maximum-intensity-projection plot of the spine with the fitted spline.

    Projects the vertebra mask onto the sagittal plane and overlays the POI
    centroid positions and the spline curve.  The figure is saved to ``filenames``.

    Args:
        pois: ``POI`` object with vertebra centroids.
        body_spline: Array of shape ``(N, 3)`` with the spline sample points in
            isotropic reoriented space.
        vert_nii: Vertebra segmentation ``NII`` used as the background image.
        filenames: Output file path passed to ``matplotlib.pyplot.savefig``.
    """
    from matplotlib import pyplot as plt

    pois = pois.reorient()
    vert_nii = vert_nii.reorient().rescale(pois.zoom)
    body_center_list = list(np.array(pois.values()))
    # fitting a curve to the poi and getting it's first derivative
    plt.figure(figsize=(10, 10))
    plt.imshow(
        np.swapaxes(np.max(vert_nii.get_array(), axis=vert_nii.get_axis(direction="R")), 0, 1),
        cmap=plt.cm.gray,  # type: ignore
    )
    plt.plot(np.asarray(body_center_list)[:, 0], np.asarray(body_center_list)[:, 1])
    plt.plot(np.asarray(body_spline[:, 0]), np.asarray(body_spline[:, 1]), "-")
    plt.savefig(filenames)
