import json
from collections.abc import Callable, MutableMapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from scipy import interpolate
from typing_extensions import Self

from TPTBox.core import vert_constants
from TPTBox.core.nii_poi_abstract import Has_Affine
from TPTBox.core.vert_constants import COORDINATE, POI_DICT, Abstract_lvl, Location, Vertebra_Instance, log, log_file, logging

ROUNDING_LVL = 7
POI_ID = (
    tuple[int, int]
    | slice
    | tuple[Location, Location]
    | tuple[Location, int]
    | tuple[int, Location]
    | tuple[Vertebra_Instance, Location]
    | tuple[Vertebra_Instance, int]
)

MAPPING = dict[int | str, int | str] | dict[int, int] | dict[int, int | None] | dict[int, None] | dict[int | str, int | str | None] | None
DIMENSIONS = 3


class Abstract_POI_Definition:
    def __init__(
        self,
        path: str | Path | None = None,
        region: dict[int, str] | None = None,
        subregion: dict[int, str] | None = None,
    ) -> None:
        """Place holder class to move string names to integer with multiple definitions"""
        if path is not None:
            with open(path) as f:
                info = json.load(f)
                region = info["region"]
                subregion = info["subregion"]
        assert region is not None
        assert subregion is not None

        self.region_idx2name = region
        self.region_name2idx = {value: key for key, value in region.items()}
        self.subregion_idx2name = subregion
        self.subregion_name2idx = {value: key for key, value in subregion.items()}


def unpack_poi_id(key: POI_ID, definition: Abstract_POI_Definition) -> tuple[int, int]:
    if isinstance(key, int | np.integer):
        region = int(key % vert_constants.LABEL_MAX)
        subregion = int(key // vert_constants.LABEL_MAX)
    elif isinstance(key, slice):
        region = key.start
        subregion = key.stop
    else:
        region, subregion = key
    if isinstance(region, str):
        try:
            region = int(region)
        except ValueError:
            region = definition.region_name2idx[region]
    if isinstance(region, Enum):
        region = region.value
    if isinstance(subregion, str):
        try:
            subregion = int(subregion)
        except ValueError:
            subregion = definition.subregion_name2idx[subregion]
    if isinstance(subregion, Enum):
        subregion = subregion.value
    return region, subregion


class POI_Descriptor(AbstractSet, MutableMapping):
    """A Object that holds the coordinate data and the two step dictionary. The implementation storage of the dictionary is the sole responsibility of this class."""

    def __init__(
        self,
        *,
        default: POI_DICT | None = None,
        definition: Abstract_POI_Definition | None = None,
    ):
        if definition is None:
            definition = Abstract_POI_Definition(
                region=vert_constants.v_idx2name,
                subregion=vert_constants.subreg_idx2name,
            )
        if default is None:
            default = {}
        self.pois = default
        self.definition = definition
        self._len: int | None = None

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, _):
        if obj is None:
            return self.pois

        return getattr(obj, self._name, self.pois)

    def __set__(self, obj, value):
        self._len = None
        setattr(obj, self._name, int(value))

    def copy(self):
        from copy import deepcopy

        return POI_Descriptor(default=deepcopy(self.pois))

    def _sort(self: Self, inplace=True, order_dict: dict | None = None):
        """Sort vertebra dictionary by sorting_list"""
        if order_dict is None:
            order_dict = {}

        def order_cardinal(elem: tuple[int, ...]):
            key1, key2 = elem[0], elem[1]
            return order_dict.get(key1, key1) * vert_constants.LABEL_MAX * 64 + key2

        poi_new = {}
        for k1, k2, v in sorted(self.items(), key=order_cardinal):  # type: ignore
            if k1 not in poi_new:
                poi_new[k1] = {}
            poi_new[k1][k2] = v
        if inplace:
            self.pois = poi_new
            return self
        return POI_Descriptor(default=poi_new)

    def items(self):
        i = 0
        for region, sub in self.pois.items():
            for subregion, coords in sub.items():
                i += 1
                yield region, subregion, coords
        self._len = i

    def items_2d(self):
        return self.pois.copy().items()

    def _apply_all(self, fun: Callable[[float, float, float], COORDINATE], inplace=False):
        out = self if inplace else POI_Descriptor()
        for region, subregion, cord in self.items():
            out[region:subregion] = fun(*cord)

        return out

    def values(self):
        return [y for x1, x2, y in self.items()]

    def keys(self):
        return [(x1, x2) for x1, x2, y in self.items()]

    def keys_region(self):
        return list(self.pois.keys())

    def keys_subregion(self):
        return {x2 for x1, x2, y in self.items()}

    def __getitem__(self, key: POI_ID) -> COORDINATE:
        region, subregion = unpack_poi_id(key, self.definition)
        return self.pois[region][subregion]

    def get(self, key: POI_ID):
        return np.array(self[key])

    def __setitem__(self, key: POI_ID, value: COORDINATE):
        if isinstance(value, np.ndarray):
            value = tuple(value.tolist())
        self._len = None
        region, subregion = unpack_poi_id(key, self.definition)
        if region not in self.pois:
            self.pois[region] = {}

        self.pois[int(region)][int(subregion)] = tuple(float(v) for v in value)  # type: ignore

    def __delitem__(self, key: POI_ID):
        self._len = None
        region, subregion = unpack_poi_id(key, self.definition)
        if region not in self.pois:
            self.pois[region] = {}

        del self.pois[region][subregion]
        if len(self.pois[region]) == 0:
            del self.pois[region]

    def __contains__(self, key: POI_ID) -> bool:
        assert not isinstance(key, int), f"Must be a (vert,subreg) or vert:subreg not {key}"
        region, subregion = unpack_poi_id(key, self.definition)
        if region in self.pois:
            return subregion in self.pois[region]
        return False

    def str_to_int(self, key: str, subregion: bool):
        try:
            return self.definition.subregion_name2idx[key] if subregion else self.definition.region_name2idx[key]
        except Exception:
            try:
                return int(key)
            except Exception:
                pass
            log.print(
                f"{key} is not in ",
                self.definition.subregion_name2idx if subregion else self.definition.region_name2idx,
                subregion,
                ltype=log_file.Log_Type.FAIL,
            )
            raise

    def str_to_int_list(self, *keys: int | str, subregion=False):
        out: list[int] = []
        for k in keys:
            if isinstance(k, str):
                k = self.str_to_int(k, subregion)  # noqa: PLW2901
            out.append(k)
        return out

    def str_to_int_dict(self, d: MAPPING, subregion=False):
        out: dict[int, int | None] = {}
        if d is None:
            return out
        for k, v in d.items():
            if isinstance(k, str):
                k = self.str_to_int(k, subregion)  # noqa: PLW2901
            if isinstance(v, str):
                v = self.str_to_int(v, subregion)  # noqa: PLW2901
            out[k] = v
        return out

    def __str__(self) -> str:
        return str(self.pois)

    def __repr__(self) -> str:
        return str(self.pois)

    def __eq__(self, x):
        if isinstance(x, POI_Descriptor):
            return x.pois == self.pois
        if isinstance(x, dict):
            return self.normalize_input_data(x).pois == self.pois
        return False

    def __len__(self) -> int:
        if self._len is None:
            self._len = len(list(self.items()))
        return self._len

    def __iter__(self):
        return iter(self.keys())

    @classmethod
    def normalize_input_data(cls, dic: dict):
        _centroids = cls()
        for k, v in dic.items():
            if isinstance(k, (tuple, list)) and isinstance(v, (tuple, list)) and len(v) == DIMENSIONS:
                _centroids[k[0], k[1]] = tuple(v)  # type: ignore
            elif isinstance(k, (int, float)) and isinstance(v, tuple) and len(v) == DIMENSIONS:
                _centroids[0, int(k)] = v  # type: ignore
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    _centroids[int(k), int(k2)] = v2  # type: ignore
            else:
                raise ValueError(dic, type(dic))
        return _centroids

    def pop(self, key: POI_ID, default):
        region, subregion = unpack_poi_id(key, self.definition)
        if region not in self.pois:
            return default
        subregs = self.pois[region]
        out = subregs.pop(subregion, default)
        if len(subregs) == 0:
            self.pois.pop(region)
        return out


@dataclass
class Abstract_POI(Has_Affine):
    _centroids: POI_Descriptor = field(default_factory=lambda: POI_Descriptor(), repr=False, kw_only=True)
    centroids: POI_Descriptor = field(repr=False, hash=False, compare=False, default=None)  # type: ignore
    format: int | None = field(default=None, repr=False, compare=False)
    info: dict = field(default_factory=dict, compare=False)  # additional info (key,value pairs)
    level_one_info: type[Abstract_lvl] = Vertebra_Instance  # Must be Enum and must has order_dict
    level_two_info: type[Abstract_lvl] = Location

    @property
    def centroids(self) -> POI_Descriptor:
        return self._centroids  # type: ignore

    @centroids.setter
    def centroids(self, value):
        if isinstance(value, property):
            return
        if isinstance(value, POI_Descriptor):
            self._centroids = value
        elif isinstance(value, dict):
            self._centroids = POI_Descriptor.normalize_input_data(value)
        else:
            raise TypeError(value, type(value), "Expected: POI_Descriptor")

    def _get_centroids(self) -> POI_Descriptor:
        return self.centroids  # type: ignore

    def apply_all(self, fun: Callable[[float, float, float], COORDINATE], inplace=False):
        ctd = self._get_centroids()._apply_all(fun, inplace)
        if inplace:
            return self
        return self.copy(ctd)

    @property
    def is_global(self) -> bool: ...

    def clone(self, **qargs):
        return self.copy(**qargs)

    def copy(self, centroids: POI_Descriptor | None = None, **qargs) -> Self: ...

    def map_labels(
        self,
        label_map_full: dict[tuple[int, int], tuple[int, int]] | None = None,
        label_map_region: MAPPING = None,
        label_map_subregion: MAPPING = None,
        verbose: logging = False,
        inplace=False,
    ) -> Self:
        """
        Maps regions and subregions to new regions and subregions based on a label map dictionary.

        Args:
            label_map_full (dict[tuple[int, int], tuple[int, int]] | None, optional): A dictionary that maps individual points (regions and subregions) to new values. Defaults to None.
            label_map_region (dict[int | str, int | str] | dict[int, int], optional): A dictionary that maps regions to new regions. Defaults to {}.
            label_map_subregion (dict[int | str, int | str] | dict[int, int], optional): A dictionary that maps subregions to new subregions. Defaults to {}.
            verbose (bool, optional): A boolean flag indicating whether to print the label map dictionaries. Defaults to False.
            inplace (bool, optional): A boolean flag indicating whether to modify the centroids in place. Defaults to False.

        Returns:
            Centroids: A new Centroids object with the mapped labels.

        Example Usage:s
            centroid_obj = Centroids(centroids)
            new_centroids = centroid_obj.map_labels(label_map_full = {(20,50):(19,48)}, label_map_region = {10:5}, label_map_subregion = {10:5})

        Outputs:
            Centroids: A new Centroids object with the mapped labels. If `inplace` is True, it modifies the current centroids object in place.
        """
        if label_map_subregion is None:
            label_map_subregion = {}
        if label_map_region is None:
            label_map_region = {}
        log.print(label_map_full, "map individual points", verbose=verbose) if label_map_full is not None else None

        if label_map_full is not None:
            if not all(isinstance(a, tuple) for a in label_map_full):
                raise ValueError(f"{label_map_full} must contain only tuples. Use label_map_subregion or label_map_region instead.")
            poi_new = POI_Descriptor()
            for region, subreg, value in self.items():
                if (region, subreg) in label_map_full:
                    poi_new[label_map_full[(region, subreg)]] = value
                else:
                    poi_new[region:subreg] = value
            gen = poi_new.items()
            new_values = poi_new
        else:
            gen = self.items()
            new_values = None
        label_map_region_ = self.centroids.str_to_int_dict(label_map_region, subregion=False)
        label_map_subregion_ = self.centroids.str_to_int_dict(label_map_subregion, subregion=True)
        log.print(label_map_region_, "map_label regions", verbose=verbose) if len(label_map_region) != 0 else None
        log.print(label_map_subregion_, "map_label subregions", verbose=verbose) if len(label_map_subregion) != 0 else None

        if len(label_map_region) + len(label_map_subregion) != 0:
            poi_new = POI_Descriptor()
            for region, subreg, value in gen:
                if subreg in label_map_subregion_:
                    subreg = label_map_subregion_[subreg]  # noqa: PLW2901
                if region in label_map_region_:
                    region = label_map_region_[region]  # noqa: PLW2901
                if subreg is None or region == 0 or region is None:
                    continue
                poi_new[region:subreg] = value
            new_values = poi_new
        if new_values is None:
            return self if inplace else self.copy()
        if inplace:
            self.centroids = new_values
            return self
        return self.copy(centroids=new_values)

    def map_labels_(
        self,
        label_map_full: dict[tuple[int, int], tuple[int, int]] | None = None,
        label_map_region: MAPPING = None,
        label_map_subregion: MAPPING = None,
        verbose: logging = False,
    ):
        if label_map_subregion is None:
            label_map_subregion = {}
        if label_map_region is None:
            label_map_region = {}
        return self.map_labels(
            label_map_full=label_map_full,
            label_map_region=label_map_region,
            label_map_subregion=label_map_subregion,
            verbose=verbose,
            inplace=True,
        )

    def sort(self, inplace=True, order_dict: dict | None = None) -> Self:
        """Sort vertebra dictionary by sorting_list"""
        if self.level_one_info is not None:
            order_dict = self.level_one_info.order_dict()
        poi = self.centroids._sort(inplace=inplace, order_dict=order_dict)
        if inplace:
            self.centroids = poi
            return self
        return self.copy(centroids=poi)

    def fit_spline(
        self,
        smoothness: int = 10,
        samples_per_poi=20,
        location: int | Location = Location.Vertebra_Corpus,
        vertebra=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fits a spline interpolation through a set of centroids and calculates the first derivative of the spline curve.

        Args:
            centroids (POI): A set of centroids to interpolate.
            smoothness (int, optional): Smoothing parameter for the spline interpolation. Default is 10.
            samples_per_poi (int, optional): Number of sample points to generate per centroid. Default is 20.
            location (int, optional): Location parameter for subregion extraction. Default is 50.
            vertebra (bool, optional): Indicates whether to perform VertebraCentroids sorting. Default is True.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                - spline_points: A 2D NumPy array representing the points on the interpolated spline curve.
                - spline_1st_derivative: A 2D NumPy array representing the first derivative of the spline curve.
                shape: first dimension to select a cord, second dimension to select all X/Y/Z
        """
        if isinstance(location, Location):
            location = location.value
        if location not in self.keys_subregion() and not isinstance(location, Sequence):
            raise ValueError(f"The location {location} is not computed in this POI class")
        # Extract subregion based on the provided location
        poi = self.extract_subregion(*location) if isinstance(location, Sequence) else self.extract_subregion(location)
        # If vertebra sorting is requested, perform it
        poi = poi.sort(inplace=False, order_dict=Vertebra_Instance.order_dict() if vertebra else None)
        # Convert centroids to NumPy array for processing
        centroids_coords = np.asarray(list(poi.values()))

        # Calculate the number of sample points to generate
        num_sample_pts = len(centroids_coords) * samples_per_poi

        # Extract coordinates for interpolation
        x_sample, y_sample, z_sample = (
            centroids_coords[:, 0],
            centroids_coords[:, 1],
            centroids_coords[:, 2],
        )
        assert len(x_sample) != 0, x_sample.shape
        # Perform cubic spline interpolation
        tck, u = interpolate.splprep([x_sample, y_sample, z_sample], k=3 if len(x_sample) > 3 else len(x_sample) - 1, s=smoothness)
        u_fine = np.linspace(0, 1, num_sample_pts)
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

        # Calculate the first derivative of the spline
        xp_fine, yp_fine, zp_fine = interpolate.splev(u_fine, tck, der=1)

        # Calculate the derivatives of the spline with respect to the arc length
        xp_fine = np.diff(x_fine) / np.diff(u_fine)
        xp_fine = np.concatenate((xp_fine, np.asarray([xp_fine[-1]])))
        yp_fine = np.diff(y_fine) / np.diff(u_fine)
        yp_fine = np.concatenate((yp_fine, np.asarray([yp_fine[-1]])))
        zp_fine = np.diff(z_fine) / np.diff(u_fine)
        zp_fine = np.concatenate((zp_fine, np.asarray([zp_fine[-1]])))

        return (
            np.asarray([x_fine, y_fine, z_fine]).T,
            np.asarray([xp_fine, yp_fine, zp_fine]).T,
        )

    def __iter__(self):
        return iter(self.centroids.keys())

    def __contains__(self, key: POI_ID) -> bool:
        key = unpack_poi_id(key, self.centroids.definition)
        return key in self.centroids

    def __getitem__(self, key: POI_ID) -> COORDINATE:
        return tuple(self.centroids[key])

    def __setitem__(self, key: POI_ID, value: tuple[float, float, float] | Sequence[float]):
        if len(value) != DIMENSIONS:
            raise ValueError(value)
        self.centroids[key] = tuple(value)

    def __len__(self) -> int:
        return self.centroids.__len__()

    def items(self, sort=True):
        if sort:
            self.sort()
        return self.centroids.items()

    def items_2D(self, sort=True):
        if sort:
            self.sort()
        return self.centroids.items_2d()

    def items_flatten(self, sort=True):
        if sort:
            self.sort()
        for x1, x2, y in self.centroids.items():
            yield x2 * vert_constants.LABEL_MAX + x1, y

    def keys(self, sort=False):
        if sort:
            self.sort()
        return self.centroids.keys()

    def keys_region(self, sort=False):
        if sort:
            self.sort()
        return self.centroids.keys_region()

    def keys_subregion(self, sort=False):
        if sort:
            self.sort()
        return list(self.centroids.keys_subregion())

    def values(self, sort=False) -> list[COORDINATE]:
        if sort:
            self.sort()
        return self.centroids.values()

    def remove_centroid_(self, *label: tuple[int, int]):
        return self.remove_centroid(*label, inplace=True)

    def remove_centroid(self, *label: tuple[int, int], inplace=False):
        import warnings

        warnings.warn("remove_centroid id deprecated use remove instead", stacklevel=5)  # TODO remove in version 1.0

        obj: Self = self.copy() if inplace is False else self
        for loc in label:
            if isinstance(loc, Location):
                loc = loc.value  # noqa: PLW2901
            obj.centroids.pop(loc, None)
        return obj

    def remove_(self, *label: tuple[int, int]):
        return self.remove(*label, inplace=True)

    def remove(self, *label: tuple[int, int], inplace=False):
        obj: Self = self.copy() if inplace is False else self
        for loc in label:
            if isinstance(loc, Location):
                loc = loc.value  # noqa: PLW2901
            obj.centroids.pop(loc, None)
        return obj

    def extract_subregion(self, *location: Location | int, inplace=False):
        location_values = tuple(l if isinstance(l, int) else l.value for l in location)
        extracted_centroids = POI_Descriptor()
        for x1, x2, y in self.centroids.items():
            if x2 in location_values:
                extracted_centroids[x1, x2] = y
        if inplace:
            self.centroids = extracted_centroids
            return self
        return self.copy(centroids=extracted_centroids)

    def extract_subregion_(self, *location: Location | int):
        return self.extract_subregion(*location, inplace=True)

    def extract_vert(self, *vert_label: int, inplace=False):
        vert_labels = tuple(vert_label)
        extracted_centroids = POI_Descriptor()
        for x1, x2, y in self.centroids.items():
            if x1 in vert_labels:
                extracted_centroids[x1, x2] = y
        if inplace:
            self.centroids = extracted_centroids
            return self
        return self.copy(centroids=extracted_centroids)

    def extract_vert_(self, *vert_label: int):
        return self.extract_vert(*vert_label, inplace=True)

    def round(self, ndigits, inplace=False):
        """Round the centroid coordinates to a specified number of digits.

        This method rounds the x, y, and z coordinates of all centroids to the given number of digits.

        Args:
            ndigits (int): The number of digits to round to.
            inplace (bool, optional): If True, modify the centroid object in place.
                If False, return a new centroid object with rounded coordinates. Defaults to False.

        Returns:
            Centroids: If inplace=True, returns the modified Centroids instance.
                Otherwise, returns a new Centroids instance with rounded coordinates.

        Examples:
            >>> centroid_obj = Centroids(...)
            >>> num_digits = 3
            >>> rounded_centroid_obj = centroid_obj.round(ndigits=num_digits, inplace=False)
        """
        out = POI_Descriptor()
        for re, sre, (x, y, z) in self.items():
            out[re:sre] = (
                round(x, ndigits=ndigits),
                round(y, ndigits=ndigits),
                round(z, ndigits=ndigits),
            )
        if inplace:
            self.centroids = out
            return self
        return self.copy(out)

    def round_(self, ndigits):
        return self.round(ndigits=ndigits, inplace=True)

    # def assert_affine(self, nii: NII | Self):
    #    assert not isinstance(nii, NII)
    #    assert self.is_global == nii.is_global

    def calculate_distances_cord(self, target_point: tuple[float, float, float] | Sequence[float]) -> dict[tuple[int, int], float]:
        """Calculate the distances between the target point and each centroid.

        Args:
            target_point (Tuple[float, float, float]): The target point represented as a tuple of x, y, and z coordinates.

        Returns:
            Dict[Tuple[int, int], float]: A dictionary containing the distances between the target point and each centroid.
            The keys are tuples of two integers representing the region and subregion labels of the centroids,
            and the values are the distances (in millimeters) between the target point and each centroid.
        """
        distances = {}
        for region, subregion, (x, y, z) in self.centroids.items():
            distance = ((target_point[0] - x) ** 2 + (target_point[1] - y) ** 2 + (target_point[2] - z) ** 2) ** 0.5
            distances[(region, subregion)] = distance
        return distances

    def calculate_distances_poi(self, target_point: Self, keep_zoom=False) -> dict[tuple[int, int], float]:
        """Calculate the distances between all points and each centroid in local spacing of the first POI.

        Args:
            target_point (Tuple[float, float, float]): The target point represented as a tuple of x, y, and z coordinates.

        Returns:
            Dict[Tuple[int, int], float]: A dictionary containing the distances between the target point and each centroid.
            The keys are tuples of two integers representing the region and subregion labels of the centroids,
            and the values are the distances (in millimeters) between the target point and each centroid.
        """

        assert self.is_global == target_point.is_global
        if not keep_zoom and not self.is_global:
            self = self.rescale((1, 1, 1), verbose=False)  # type: ignore  # noqa: PLW0642
        if not self.is_global:
            if target_point.zoom != self.zoom or target_point.shape != self.zoom:
                target_point = target_point.resample_from_to(self)  # type: ignore
            else:
                target_point.assert_affine(self)

        distances = {}
        for region, subregion, (x, y, z) in target_point.intersect(self).items():
            if (region, subregion) not in self:
                continue
            c = self[region, subregion]
            distance = ((c[0] - x) ** 2 + (c[1] - y) ** 2 + (c[2] - z) ** 2) ** 0.5
            distances[(region, subregion)] = float(distance)
        return distances

    def calculate_distances_poi_two_locations(self, a: Location | int, b: Location | int, keep_zoom=False) -> dict[int, float]:
        if isinstance(a, Enum):
            a = a.value
        if isinstance(b, Enum):
            b = b.value
        p1 = self.extract_subregion(a)
        p2 = self.extract_subregion(b).map_labels(label_map_subregion={b: a})
        out = p2.calculate_distances_poi(p1, keep_zoom=keep_zoom)
        return {a: c for (a, _), c in out.items()}

    def join_left(self, pois: Self, inplace=False, _right_join=False) -> Self:
        """
        Left join operation to combine the centroids from another set of points into the current set.
        Existing values are NOT overwritten

        Args:
            pois (Self): Another set of points (centroids) to be combined.
            inplace (bool, optional): If True, the operation is performed in-place on the current set.
                                    If False, a new set is created. Default is True.

        Returns:
            Self: The combined set of centroids, either in-place or as a new set, depending on the 'inplace' parameter.
        """
        ctd_list = self.centroids
        if not inplace:
            ctd_list = ctd_list.copy()
        for x, y, c in pois.items():
            if (x, y) in self and not _right_join:
                continue
            ctd_list[x, y] = c
        if inplace:
            return self
        return self.copy(ctd_list)

    def join_left_(self, pois: Self):
        return self.join_left(pois, inplace=True)

    def __lshift__(self, other):
        return self.join_left(other)

    def __add__(self, other):
        return self.join_left(other)

    def __rshift__(self, other):
        return self.join_right(other)

    def join_right_(self, pois: Self):
        return self.join_left(pois, inplace=True, _right_join=True)

    def join_right(self, *args, **qargs):
        """
        Rights join operation to combine the centroids from another set of points into the current set.
        Existing values are overwritten.

        Args:
            pois (Self): Another set of points (centroids) to be combined.
            inplace (bool, optional): If True, the operation is performed in-place on the current set.
                                    If False, a new set is created. Default is True.

        Returns:
            Self: The combined set of centroids, either in-place or as a new set, depending on the 'inplace' parameter.
        """
        return self.join_left(*args, **qargs, _right_join=True)

    def intersect(self, pois: Self, inplace=False):
        """
        Intersect operation to find common centroids between two sets of points and update the current set.

        Args:
            pois (Self): Another set of points (centroids) to find the intersection with.
            inplace (bool, optional): If True, the operation is performed in-place on the current set.
                                    If False, a new set is created. Default is False.

        """
        filtered_centroids = POI_Descriptor()

        for x, y, c in self.items():
            if (x, y) in pois:
                filtered_centroids[x, y] = c
        if inplace:
            self.centroids = filtered_centroids
            return self
        return self.copy(filtered_centroids)

    def intersect_(self, pois: Self):
        return self.intersect(pois, inplace=True)

    def __and__(self, poi):
        return self.intersect(poi)

    def subtract(self, pois: Self, inplace=False):
        """
        Intersect operation to find common centroids between two sets of points and update the current set.

        Args:
            pois (Self): Another set of points (centroids) to find the intersection with.
            inplace (bool, optional): If True, the operation is performed in-place on the current set.
                                    If False, a new set is created. Default is False.

        """
        filtered_centroids = POI_Descriptor()

        for x, y, c in self.items():
            if (x, y) not in pois:
                filtered_centroids[x, y] = c
        if inplace:
            self.centroids = filtered_centroids
            return self
        return self.copy(filtered_centroids)

    def __sub__(self, poi):
        return self.subtract(poi)
