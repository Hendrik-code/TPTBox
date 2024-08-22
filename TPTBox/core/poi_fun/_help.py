from functools import wraps
from time import time

import numpy as np

from TPTBox import POI, Location, Logger_Interface, Print_Logger
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.vert_constants import Vertebra_Instance

_log = Print_Logger()
sacrum_w_o_arcus = (Vertebra_Instance.COCC.value, Vertebra_Instance.S6.value, Vertebra_Instance.S5.value, Vertebra_Instance.S4.value)
sacrum_w_o_direction = (Vertebra_Instance.COCC.value,)


def to_local_np(loc: Location, bb: tuple[slice, slice, slice], poi: POI, label, log: Logger_Interface, verbose=True):
    if (label, loc.value) in poi:
        return np.asarray([a - b.start for a, b in zip(poi[label, loc.value], bb, strict=True)])
    if verbose:
        log.on_fail(f"region={label},subregion={loc.value} is missing")
        # raise KeyError(f"region={label},subregion={loc.value} is missing.")
    return None


def paint_into_NII(poi: POI, a: NII, l=None, idxs=None, rays: None | list[tuple[Location, Location]] = None):
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
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        _log.on_neutral(f"func:{f.__name__!r} took: {te-ts:2.4f} sec")
        return result

    return wrap


def make_spine_plot(pois: POI, body_spline, vert_nii: NII, filenames):
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
