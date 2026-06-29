from __future__ import annotations

import json
from collections.abc import Callable, ItemsView, Iterator, MutableMapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
from scipy import interpolate
from typing_extensions import Self

from TPTBox.core import vert_constants
from TPTBox.core.vert_constants import COORDINATE, POI_DICT, Abstract_lvl, Any, Location, Vertebra_Instance, log, log_file, logging

POI_ID = Union[
    tuple[int, int],
    slice,
    tuple[Union[Abstract_lvl, int], Union[Abstract_lvl, int]],
    tuple[Abstract_lvl, Abstract_lvl],
    tuple[Abstract_lvl, int],
    tuple[int, Abstract_lvl],
    tuple[Vertebra_Instance, Location],
    tuple[Vertebra_Instance, int],
]

MAPPING = Union[
    dict[Union[int, str], Union[int, str]],
    dict[int, int],
    dict[int, Union[int, None]],
    dict[int, None],
    dict[Union[int, str], Union[int, str, None]],
    None,
]

DIMENSIONS = 3


def _flatten(vert_label) -> list:
    """Flatten a (possibly nested) sequence of labels into a flat list of ints.

    Enum members are replaced by their ``.value``.
    """
    return [
        item.value if isinstance(item, Enum) else item
        for sublist in vert_label
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]  # type: ignore


class _Abstract_POI_Definition:
    """Bidirectional name-to-integer mapping for region and subregion labels.

    Supports loading the mapping from a JSON file or from pre-built dicts.
    Used internally by :class:`POI_Descriptor` to convert human-readable label
    names to integer IDs and back.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        region: dict[int, str] | None = None,
        subregion: dict[int, str] | None = None,
    ) -> None:
        """Placeholder class to move string names to integers with multiple definitions."""
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


def unpack_poi_id(key: POI_ID, definition: _Abstract_POI_Definition) -> tuple[int, int]:
    """Convert any supported POI key type to a ``(region, subregion)`` integer pair.

    Accepted key forms: plain integer (packed label), ``slice(region, subregion)``,
    2-tuple of ints, 2-tuple of ``Abstract_lvl`` / ``Enum`` members, or mixed tuples.
    String values are resolved via ``definition``'s name-to-index mappings.

    Args:
        key: POI identifier in any of the supported formats.
        definition: Name-to-integer mapping used to resolve string labels.

    Returns:
        ``(region, subregion)`` tuple of plain Python integers.
    """
    if isinstance(key, (int, np.integer)):
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


# ---------------------------------------------------------------------------
# label_name (custom per-point / per-region names stored in POI.info["label_name"])
# In-memory (new) format: {region:int -> {subregion:int -> name:str, "name": group_name:str}}
# ---------------------------------------------------------------------------
LABEL_NAME = "label_name"
_GROUP_NAME_KEY = "name"


def normalize_label_name(d: dict | None) -> dict[int, dict]:
    """Normalize any ``label_name`` mapping to the nested form (migration helper).

    Target: ``{region:int -> {subregion:int -> name:str, "name": group_name:str}}``.
    Accepts and migrates: the old flat ``{"(1, 2)": "C2"}`` form, a JSON-loaded nested
    form with string keys ``{"1": {"2": "C2", "name": "Spine"}}``, and the already-nested
    form (idempotent).
    """
    import ast

    if not d:
        return {}
    out: dict[int, dict] = {}
    # old flat format: every key is a "(region, subregion)" tuple string
    if all(isinstance(k, str) and k.strip().startswith("(") for k in d):
        for k, name in d.items():
            region, subregion = ast.literal_eval(k)
            out.setdefault(int(region), {})[int(subregion)] = name
        return out
    # nested format (region keys may be JSON strings)
    for region, sub in d.items():
        target = out.setdefault(int(region), {})
        if isinstance(sub, dict):
            for s, name in sub.items():
                target[_GROUP_NAME_KEY if s == _GROUP_NAME_KEY else int(s)] = name
        else:  # degenerate {region: name} -> treat as the region group name
            target[_GROUP_NAME_KEY] = sub
    return out


def label_name_dict(info: dict) -> dict[int, dict]:
    """Return the normalized nested ``label_name`` dict from ``info`` (normalizing in place)."""
    ln = info.get(LABEL_NAME)
    if not ln:
        return {}
    norm = normalize_label_name(ln)
    info[LABEL_NAME] = norm  # cache the normalized form back into info
    return norm


def _id_of(x) -> int:
    """Resolve an Enum member / int / numeric string to its integer id."""
    if isinstance(x, Enum):
        return x.value
    return int(x)


class POI_Descriptor(AbstractSet, MutableMapping):
    """Two-level dictionary that maps ``(region, subregion)`` pairs to 3-D coordinates.

    Acts simultaneously as a ``MutableMapping`` and an ``AbstractSet`` so that
    POI keys can be tested for membership with ``in``.  The underlying storage
    is a nested plain-Python dict ``{region: {subregion: (x, y, z)}}``.
    All coordinate values are stored as ``tuple[float, float, float]``.

    Args:
        default: Initial nested coordinate dict.  Defaults to an empty dict.
        definition: Name-to-integer mapping for string label resolution.
            Defaults to the spine vertebra / subregion mapping from
            ``vert_constants``.
    """

    def __init__(
        self,
        *,
        default: POI_DICT | None = None,
        definition: _Abstract_POI_Definition | None = None,
    ):
        if definition is None:
            definition = _Abstract_POI_Definition(
                region=vert_constants.v_idx2name,
                subregion=vert_constants.subreg_idx2name,
            )
        if default is None:
            default = {}
        self.pois = default
        self.definition = definition
        self._len: int | None = None

    __hash__ = None  # explicitly mark as unhashable

    def __set_name__(self, owner, name: str) -> None:
        """Store the descriptor's attribute name so ``__get__`` / ``__set__`` can use it."""
        self._name = "_" + name

    def __get__(self, obj, _) -> POI_Descriptor | dict:
        """Return the descriptor value from the owner instance, or the raw dict when accessed on the class."""
        if obj is None:
            return self.pois

        return getattr(obj, self._name, self.pois)

    def __set__(self, obj, value) -> None:
        """Set the descriptor value on the owner instance."""
        self._len = None
        setattr(obj, self._name, int(value))

    def copy(self) -> POI_Descriptor:
        """Return a deep copy of this descriptor (keeping its name<->id ``definition``)."""
        from copy import deepcopy

        return POI_Descriptor(default=deepcopy(self.pois), definition=self.definition)

    def _sort(self: Self, inplace=True, order_dict: dict | None = None):
        """Sort vertebra dictionary by sorting_list."""
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

    def items(self) -> Iterator[tuple[int, int, COORDINATE]]:
        """Yield ``(region, subregion, coordinate)`` triples for all stored points."""
        i = 0
        for region, sub in self.pois.items():
            for subregion, coords in sub.items():
                i += 1
                yield region, subregion, coords
        self._len = i

    def items_2d(self) -> ItemsView[int, dict[int, COORDINATE]]:
        """Return a view of the top-level ``{region: {subregion: coord}}`` dict."""
        return self.pois.copy().items()

    def _apply_all(self, fun: Callable[[float, float, float], COORDINATE], inplace: bool = False) -> POI_Descriptor:
        """Apply a coordinate transform ``fun`` to every stored point."""
        out = self if inplace else POI_Descriptor()
        for region, subregion, cord in self.items():
            out[region:subregion] = fun(*cord)

        return out

    def values(self) -> list[COORDINATE]:
        """Return a list of all stored ``(x, y, z)`` coordinate tuples."""
        return [y for x1, x2, y in self.items()]

    def keys(self) -> list[tuple[int, int]]:
        """Return a list of all ``(region, subregion)`` key pairs."""
        return [(x1, x2) for x1, x2, y in self.items()]

    def keys_region(self) -> list[int]:
        """Return a list of all unique region (vertebra) IDs."""
        return list(self.pois.keys())

    def keys_subregion(self) -> set[int]:
        """Return the set of all unique subregion IDs across all regions."""
        return {x2 for x1, x2, y in self.items()}

    def __getitem__(self, key: POI_ID) -> COORDINATE:
        region, subregion = unpack_poi_id(key, self.definition)
        try:
            return self.pois[region][subregion]
        except KeyError:
            raise KeyError(region, subregion, "not in", list(self.keys()))  # noqa: B904

    def get(self, key: POI_ID) -> np.ndarray:
        """Return the coordinate for ``key`` as a numpy array."""
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

    def str_to_int(self, key: str, subregion: bool) -> int:
        """Convert a label name string to its integer ID.

        Args:
            key: Name string to look up.
            subregion: If ``True``, search in the subregion mapping; otherwise
                search in the region mapping.

        Returns:
            Integer label ID.

        Raises:
            KeyError: If ``key`` is not found in the mapping and cannot be cast
                to ``int``.
        """
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

    def str_to_int_list(self, *keys: int | str, subregion: bool = False) -> list[int]:
        """Convert a sequence of label names or IDs to a flat list of integer IDs.

        Args:
            *keys: Label names (str) or integer IDs.  Nested lists are
                flattened automatically.
            subregion: If ``True``, resolve strings as subregion names;
                otherwise as region names.  Defaults to ``False``.

        Returns:
            Flat list of integer label IDs.
        """
        keys = _flatten(keys)
        out: list[int] = []
        for k in keys:
            if isinstance(k, str):
                k = self.str_to_int(k, subregion)  # noqa: PLW2901
            out.append(k)
        return out

    def str_to_int_dict(self, d: MAPPING, subregion: bool = False) -> dict[int, int | None]:
        """Convert a label-map dict's string keys/values to integer IDs.

        Args:
            d: Mapping from label names or IDs to target names, IDs, or
                ``None``.  If ``None`` an empty dict is returned.
            subregion: If ``True``, resolve strings as subregion names;
                otherwise as region names.  Defaults to ``False``.

        Returns:
            Dict with all keys and non-``None`` values converted to integers.
        """
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
        """Iterate over all ``(region, subregion)`` key pairs."""
        return iter(self.keys())

    @classmethod
    def normalize_input_data(cls, dic: dict) -> POI_Descriptor:
        """Convert a raw dict of various formats into a ``POI_Descriptor``.

        Accepted dict formats:

        * ``{(region, subregion): (x, y, z)}``
        * ``{int_label: (x, y, z)}`` (treated as ``region=0``)
        * ``{region: {subregion: (x, y, z)}}``

        Args:
            dic: Input dict.

        Returns:
            Populated ``POI_Descriptor`` instance.

        Raises:
            ValueError: If a key/value pair cannot be interpreted.
        """
        _centroids = cls()
        for k, v in dic.items():
            if isinstance(k, (tuple, list)) and isinstance(v, (tuple, list, np.ndarray)) and len(v) == DIMENSIONS:
                _centroids[k[0], k[1]] = tuple(v)  # type: ignore
            elif isinstance(k, (int, float)) and isinstance(v, tuple) and len(v) == DIMENSIONS:
                _centroids[0, int(k)] = v  # type: ignore
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    _centroids[int(k), int(k2)] = v2  # type: ignore
            else:
                raise ValueError(k, type(k), v, tuple(v))
        return _centroids

    def pop(self, key: POI_ID, default: COORDINATE | None) -> COORDINATE | None:
        """Remove and return the coordinate for ``key``, or ``default`` if absent.

        Also removes the region entry if it becomes empty after the deletion.

        Args:
            key: ``(region, subregion)`` pair or any supported ``POI_ID`` form.
            default: Value to return when ``key`` is not present.

        Returns:
            The removed coordinate tuple, or ``default``.
        """
        region, subregion = unpack_poi_id(key, self.definition)
        if region not in self.pois:
            return default
        subregs = self.pois[region]
        out = subregs.pop(subregion, default)
        if len(subregs) == 0:
            self.pois.pop(region)
        return out


@dataclass
class Abstract_POI:
    """Abstract base for POI containers.

    Stores a two-level ``(region, subregion)`` → coordinate mapping and
    provides geometry operations (coordinate transforms, spline fitting,
    set operations) that are shared by both the local ``POI`` and the global
    ``POI_Global`` classes.

    Attributes:
        _centroids: Internal ``POI_Descriptor`` storage.
        centroids: Public descriptor property backed by ``_centroids``.
        format: Integer format tag identifying how the data was saved / loaded.
        level_one_info: Enum class mapping region integer IDs to names.
        level_two_info: Enum class mapping subregion integer IDs to names.
        info: Arbitrary metadata dict persisted alongside the coordinates.
    """

    _centroids: POI_Descriptor = field(default_factory=lambda: POI_Descriptor(), repr=False)
    centroids: POI_Descriptor = field(repr=False, hash=False, compare=False, default=None)  # type: ignore
    format: int | None = field(default=None, repr=False, compare=False)
    level_one_info: type[Abstract_lvl] = Any
    level_two_info: type[Abstract_lvl] = Any
    info: dict = field(default_factory=dict, compare=False, init=True)  # additional info (key,value pairs)

    def __post_init__(self):
        if not isinstance(self._centroids, POI_Descriptor):
            self._centroids = POI_Descriptor.normalize_input_data(self._centroids)

    @property
    def centroids(self) -> POI_Descriptor:
        """Return the underlying POI_Descriptor storing all centroid coordinates."""
        return self._centroids  # type: ignore

    @centroids.setter
    def centroids(self, value):
        """Set the underlying POI_Descriptor, accepting a dict or POI_Descriptor."""
        if isinstance(value, property):
            return
        if isinstance(value, POI_Descriptor):
            self._centroids = value
        elif isinstance(value, dict):
            self._centroids = POI_Descriptor.normalize_input_data(value)
        else:
            raise TypeError(value, type(value), "Expected: POI_Descriptor")

    def _get_centroids(self) -> POI_Descriptor:
        """Return the raw ``POI_Descriptor`` (subclasses may override for lazy loading)."""
        return self.centroids  # type: ignore

    def apply_all(self, fun: Callable[[float, float, float], COORDINATE], inplace: bool = False) -> Self:
        """Apply a coordinate transform to every stored point.

        Args:
            fun: Callable that accepts ``(x, y, z)`` and returns a new
                3-element coordinate.
            inplace: Modify this object in place.  Defaults to ``False``.

        Returns:
            The transformed POI (``self`` when ``inplace=True``, a new object
            otherwise).
        """
        ctd = self._get_centroids()._apply_all(fun, inplace)
        if inplace:
            return self
        return self.copy(ctd)

    @property
    def is_global(self) -> bool:
        """Return True if coordinates are in world/global space, False if in voxel space."""

    def clone(self, **qargs) -> Self:
        """Alias for :meth:`copy`; returns a copy of this POI."""
        return self.copy(**qargs)

    def copy(self, centroids: POI_Descriptor | None = None, **qargs) -> Self:
        """Return a shallow copy of this POI, optionally replacing the centroids."""

    def map_labels(
        self,
        label_map_full: dict[tuple[int, int], tuple[int, int]] | None = None,
        label_map_region: MAPPING = None,
        label_map_subregion: MAPPING = None,
        verbose: logging = False,
        inplace=False,
    ) -> Self:
        """Maps regions and subregions to new regions and subregions based on a label map dictionary.

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
    ) -> Self:
        """In-place alias for :meth:`map_labels`; modifies this object and returns it."""
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

    def sort(self, inplace: bool = True, order_dict: dict | None = None) -> Self:
        """Sort the internal centroid dictionary by vertebra ordering.

        Uses ``level_one_info.order_dict()`` when available, falling back to
        the provided ``order_dict`` or insertion order.

        Args:
            inplace: Sort in place.  Defaults to ``True``.
            order_dict: Custom ordering mapping ``{region_id: sort_key}``.
                When ``None`` and ``level_one_info`` is set, the info class's
                own ordering is used.

        Returns:
            The sorted POI (``self`` when ``inplace=True``, a copy otherwise).
        """
        if self.level_one_info is not None and self.level_one_info != Any:
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
        location: int | Abstract_lvl = Location.Vertebra_Corpus,
        vertebra=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fits a spline interpolation through a set of centroids and calculates the first derivative of the spline curve.

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
        if isinstance(location, Abstract_lvl):
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
        """Iterate over all ``(region, subregion)`` key pairs."""
        return iter(self.centroids.keys())

    def __contains__(self, key: POI_ID) -> bool:
        key = unpack_poi_id(key, self.centroids.definition)
        return key in self.centroids

    def __getitem__(self, key: POI_ID) -> COORDINATE:
        return tuple(self.centroids[key])

    def __setitem__(self, key: POI_ID, value: tuple[float, float, float] | Sequence[float] | np.ndarray) -> None:
        if len(value) != DIMENSIONS:
            raise ValueError(value)
        self.centroids[key] = tuple(value)

    def __len__(self) -> int:
        return self.centroids.__len__()

    def items(self, sort: bool = True) -> Iterator[tuple[int, int, COORDINATE]]:
        """Yield ``(region, subregion, coordinate)`` triples, optionally sorted.

        Args:
            sort: Sort by vertebra ordering before yielding.  Defaults to
                ``True``.
        """
        if sort:
            self.sort()
        return self.centroids.items()

    def items_2D(self, sort: bool = True) -> ItemsView[int, dict[int, COORDINATE]]:
        """Return the top-level ``{region: {subregion: coord}}`` dict view, optionally sorted.

        Args:
            sort: Sort by vertebra ordering first.  Defaults to ``True``.
        """
        if sort:
            self.sort()
        return self.centroids.items_2d()

    def items_flatten(self, sort: bool = True) -> Iterator[tuple[int, COORDINATE]]:
        """Yield ``(packed_label, coordinate)`` pairs using the legacy packed-int encoding.

        Args:
            sort: Sort by vertebra ordering before yielding.  Defaults to
                ``True``.
        """
        if sort:
            self.sort()
        for x1, x2, y in self.centroids.items():
            yield x2 * vert_constants.LABEL_MAX + x1, y

    def keys(self, sort: bool = False) -> list[tuple[int, int]]:
        """Return all ``(region, subregion)`` key pairs.

        Args:
            sort: Sort by vertebra ordering first.  Defaults to ``False``.
        """
        if sort:
            self.sort()
        return self.centroids.keys()

    def keys_region(self, sort: bool = False) -> list[int]:
        """Return all unique region (vertebra) IDs.

        Args:
            sort: Sort by vertebra ordering first.  Defaults to ``False``.
        """
        if sort:
            self.sort()
        return self.centroids.keys_region()

    def keys_subregion(self, sort: bool = False) -> list[int]:
        """Return all unique subregion IDs as a sorted list.

        Args:
            sort: Sort by vertebra ordering first.  Defaults to ``False``.
        """
        if sort:
            self.sort()
        return list(self.centroids.keys_subregion())

    def label_name(self, region, subregion) -> str | None:
        """Return the human-readable name of the point ``(region, subregion)``.

        A custom name in ``info["label_name"]`` takes priority over the ``level_two_info`` enum
        name (and finally falls back to the raw id as a string). Accepts ints or Enum members.
        A warning is raised when a custom name is set that differs from the ``level_two_info``
        name for that id (when ``level_two_info`` is given).
        """
        import warnings

        region_i, subregion_i = _id_of(region), _id_of(subregion)
        custom = label_name_dict(self.info).get(region_i, {}).get(subregion_i)
        enum_name = None
        if self.level_two_info not in (None, Any):
            n = self.level_two_info._get_name(subregion_i, no_raise=True)
            enum_name = n if n != str(subregion_i) else None
        if custom is not None:
            if enum_name is not None and custom != enum_name:
                warnings.warn(
                    f"label_name {custom!r} for subregion {subregion_i} is not the {self.level_two_info.__name__} name {enum_name!r}",
                    stacklevel=2,
                )
            return custom
        return enum_name if enum_name is not None else str(subregion_i)

    def set_label_name(self, region, subregion, name: str) -> None:
        """Set a custom name for the point ``(region, subregion)`` in ``info["label_name"]``."""
        d = label_name_dict(self.info)
        d.setdefault(_id_of(region), {})[_id_of(subregion)] = name
        self.info[LABEL_NAME] = d

    def level_one_name(self, region) -> str | None:
        """Return the group (level-one) name of ``region``.

        A custom group name in ``info["label_name"][region]["name"]`` takes priority over the
        ``level_one_info`` enum name. Accepts an int or Enum member.
        """
        region_i = _id_of(region)
        custom = label_name_dict(self.info).get(region_i, {}).get(_GROUP_NAME_KEY)
        if custom is not None:
            return custom
        if self.level_one_info not in (None, Any):
            return self.level_one_info._get_name(region_i, no_raise=True)
        return str(region_i)

    def set_level_one_name(self, region, name: str) -> None:
        """Set a custom group (level-one) name for ``region`` in ``info["label_name"]``."""
        d = label_name_dict(self.info)
        d.setdefault(_id_of(region), {})[_GROUP_NAME_KEY] = name
        self.info[LABEL_NAME] = d

    def values(self, sort: bool = False) -> list[COORDINATE]:
        """Return all stored ``(x, y, z)`` coordinate tuples.

        Args:
            sort: Sort by vertebra ordering first.  Defaults to ``False``.
        """
        if sort:
            self.sort()
        return self.centroids.values()

    def remove_centroid_(self, *label: tuple[int, int]) -> Self:
        """Deprecated in-place removal — use :meth:`remove_` instead."""
        return self.remove_centroid(*label, inplace=True)

    def remove_centroid(self, *label: tuple[int, int], inplace: bool = False) -> Self:
        """Deprecated — use :meth:`remove` instead."""
        import warnings

        warnings.warn("remove_centroid id deprecated use remove instead", stacklevel=5)  # TODO remove in version 1.0

        obj: Self = self.copy() if inplace is False else self
        for loc in label:
            if isinstance(loc, Abstract_lvl):
                loc = loc.value  # noqa: PLW2901
            obj.centroids.pop(loc, None)
        return obj

    def remove_(self, *label: tuple[int, int]) -> Self:
        """In-place alias for :meth:`remove`."""
        return self.remove(*label, inplace=True)

    def remove(self, *label: tuple[int, int], inplace: bool = False) -> Self:
        """Remove one or more ``(region, subregion)`` entries from this POI.

        Args:
            *label: One or more ``(region, subregion)`` tuples or
                ``Abstract_lvl`` members to remove.
            inplace: Modify this object in place.  Defaults to ``False``.

        Returns:
            The modified POI (``self`` when ``inplace=True``, a copy otherwise).
        """
        obj: Self = self.copy() if inplace is False else self
        for loc in label:
            if isinstance(loc, Abstract_lvl):
                loc = loc.value  # noqa: PLW2901
            obj.centroids.pop(loc, None)
        return obj

    def extract_subregion(self, *location: Abstract_lvl | int, inplace: bool = False) -> Self:
        """Return a POI containing only the specified subregion(s).

        Args:
            *location: One or more subregion IDs or ``Abstract_lvl`` members.
            inplace: Filter in place.  Defaults to ``False``.

        Returns:
            Filtered POI.
        """
        location = _flatten(location)

        location_values = tuple(l if isinstance(l, int) else l.value for l in location)
        extracted_centroids = POI_Descriptor()
        for x1, x2, y in self.centroids.items():
            if x2 in location_values:
                extracted_centroids[x1, x2] = y
        if inplace:
            self.centroids = extracted_centroids
            return self
        return self.copy(centroids=extracted_centroids)

    def extract_subregion_(self, *location: Abstract_lvl | int) -> Self:
        """In-place alias for :meth:`extract_subregion`."""
        return self.extract_subregion(*location, inplace=True)

    def extract_vert(self, *vert_label: int, inplace: bool = False) -> Self:
        """Deprecated — use :meth:`extract_region` instead."""
        import warnings

        warnings.warn("extract_vert id deprecated use extract_region instead", stacklevel=5)  # TODO remove in version 2.0
        return self.extract_region(*vert_label, inplace=inplace)

    def extract_vert_(self, *vert_label: int) -> Self:
        """Deprecated in-place alias — use :meth:`extract_region_` instead."""
        return self.extract_vert(*vert_label, inplace=True)

    def extract_region(self, *vert_label: int | list[int] | Enum, inplace: bool = False) -> Self:
        """Return a POI containing only the specified region(s) (vertebrae).

        Args:
            *vert_label: One or more region IDs, lists of IDs, or ``Enum``
                members to retain.
            inplace: Filter in place.  Defaults to ``False``.

        Returns:
            Filtered POI.
        """
        # flatten list
        vert_label = _flatten(vert_label)
        vert_labels = tuple(vert_label)
        extracted_centroids = POI_Descriptor()
        for x1, x2, y in self.centroids.items():
            if x1 in vert_labels:
                extracted_centroids[x1, x2] = y
        if inplace:
            self.centroids = extracted_centroids
            return self
        return self.copy(centroids=extracted_centroids)

    def extract_region_(self, *vert_label: int) -> Self:
        """In-place alias for :meth:`extract_region`."""
        return self.extract_region(*vert_label, inplace=True)

    def round(self, ndigits, inplace=False) -> Self:
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

    def round_(self, ndigits: int) -> Self:
        """In-place alias for :meth:`round`."""
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
            self = self.to_global()  # type: ignore  # noqa: PLW0642
            target_point = target_point.to_global()
        if not self.is_global:
            target_point.assert_affine(self)

        distances = {}
        for region, subregion, (x, y, z) in target_point.intersect(self).items():
            if (region, subregion) not in self:
                continue
            c = self[region, subregion]
            distance = ((c[0] - x) ** 2 + (c[1] - y) ** 2 + (c[2] - z) ** 2) ** 0.5
            distances[(region, subregion)] = float(distance)
        return distances

    def calculate_distances_poi_any_2_any(
        self, target_point: Self, keep_zoom: bool = False
    ) -> dict[tuple[int, int], dict[tuple[int, int], float]]:
        """Calculate pairwise Euclidean distances between every point in ``self`` and every point in ``target_point``.

        Both POIs are first converted to global (mm) space unless
        ``keep_zoom=True``.

        Args:
            target_point: Reference POI to measure distances to.
            keep_zoom: When ``True``, skip the global conversion and require both
                POIs to share the same affine.  Defaults to ``False``.

        Returns:
            Nested dict ``{(r2, s2): {(r, s): distance_mm}}`` covering every
            combination.
        """
        if not keep_zoom:
            self = self.to_global()  # type: ignore  # noqa: PLW0642
            target_point = target_point.to_global()
        else:
            target_point.assert_affine(self)

        distances: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
        for region2, subregion2, c in self.items():
            distances[(region2, subregion2)] = {}
            for region, subregion, (x, y, z) in target_point.items():
                distance = ((c[0] - x) ** 2 + (c[1] - y) ** 2 + (c[2] - z) ** 2) ** 0.5
                distances[(region2, subregion2)][(region, subregion)] = float(distance)
        return distances

    def calculate_distances_poi_across_regions(
        self,
        target_point: Self,
    ) -> dict[tuple[int, int], dict[tuple[int, int], float]]:
        """Calculate the distances between all points and each centroid in local spacing of the first POI.

        Args:
            target_point (Tuple[float, float, float]): The target point represented as a tuple of x, y, and z coordinates.

        Returns:
            Dict[Tuple[int, int], float]: A dictionary containing the distances between the target point and each centroid.
            The keys are tuples of two integers representing the region and subregion labels of the centroids,
            and the values are the distances (in millimeters) between the target point and each centroid.
        """
        from warnings import warn

        warn("calculate_distances_poi_across_regions is depredated", stacklevel=3)
        assert self.is_global == target_point.is_global
        if not self.is_global:
            if target_point.zoom != self.zoom or target_point.shape != self.shape:
                target_point = target_point.resample_from_to(self)  # type: ignore
            else:
                target_point.assert_affine(self)

        distances2target = {}
        for region, subregion, (x, y, z) in target_point.items():
            distances = {}
            for r2, s2, (x2, y2, z2) in self.items():
                if subregion != s2:
                    continue
                distance_vector = np.array([x, y, z]) - np.array([x2, y2, z2])
                distance_vector = np.multiply(distance_vector, self.zoom)
                distance = np.linalg.norm(distance_vector)
                distances[(r2, subregion)] = float(distance)
            distances2target[(region, subregion)] = distances
        return distances2target

    def join_left(self, pois: Self, inplace=False, _right_join=False) -> Self:
        """Left-join another POI set into this one without overwriting existing values.

        Args:
            pois (Self): Another set of points (centroids) to be combined.
            inplace (bool, optional): If True, the operation is performed in-place on the current set.
                                    If False, a new set is created. Default is True.

        Returns:
            Self: The combined set of centroids, either in-place or as a new set, depending on the 'inplace' parameter.
        """
        ctd_list = self.centroids
        src_ln = label_name_dict(pois.info)
        if not inplace:
            ctd_list = ctd_list.copy()
        dst_ln = label_name_dict(self.info) if src_ln else None
        if dst_ln is not None:
            self.info[LABEL_NAME] = dst_ln
        for x, y, c in pois.items():
            if (x, y) in self and not _right_join:
                continue
            ctd_list[x, y] = c
            if dst_ln is not None:
                name = src_ln.get(x, {}).get(y)
                if name is not None:
                    dst_ln.setdefault(x, {})[y] = name
                grp = src_ln.get(x, {}).get(_GROUP_NAME_KEY)
                if grp is not None:
                    dst_ln.setdefault(x, {}).setdefault(_GROUP_NAME_KEY, grp)
        if inplace:
            return self
        return self.copy(ctd_list)

    def join_left_(self, pois: Self) -> Self:
        """In-place alias for :meth:`join_left`."""
        return self.join_left(pois, inplace=True)

    def __lshift__(self, other) -> Self:
        """Operator ``<<``: left join (existing values not overwritten)."""
        return self.join_left(other)

    def __add__(self, other) -> Self:
        """Operator ``+``: left join (existing values not overwritten)."""
        return self.join_left(other)

    def __rshift__(self, other) -> Self:
        """Operator ``>>``: right join (existing values are overwritten)."""
        return self.join_right(other)

    def join_right_(self, pois: Self) -> Self:
        """In-place alias for :meth:`join_right`."""
        return self.join_left(pois, inplace=True, _right_join=True)

    def join_right(self, *args, **qargs) -> Self:
        """Right-join another POI set into this one, overwriting existing values.

        Args:
            pois (Self): Another set of points (centroids) to be combined.
            inplace (bool, optional): If True, the operation is performed in-place on the current set.
                                    If False, a new set is created. Default is True.

        Returns:
            Self: The combined set of centroids, either in-place or as a new set, depending on the 'inplace' parameter.
        """
        return self.join_left(*args, **qargs, _right_join=True)

    def intersect(self, pois: Self, inplace=False) -> Self:
        """Intersect operation to find common centroids between two sets of points and update the current set.

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

    def intersect_(self, pois: Self) -> Self:
        """In-place alias for :meth:`intersect`."""
        return self.intersect(pois, inplace=True)

    def __and__(self, poi) -> Self:
        """Operator ``&``: intersect (keep only points present in both POIs)."""
        return self.intersect(poi)

    def subtract(self, pois: Self, inplace: bool = False) -> Self:
        """Return a POI with all points that are NOT present in ``pois`` removed.

        Args:
            pois: POI whose keys act as a rejection set.
            inplace: Modify this object in place.  Defaults to ``False``.

        Returns:
            POI containing only keys absent from ``pois``.
        """
        filtered_centroids = POI_Descriptor()

        for x, y, c in self.items():
            if (x, y) not in pois:
                filtered_centroids[x, y] = c
        if inplace:
            self.centroids = filtered_centroids
            return self
        return self.copy(filtered_centroids)

    def __sub__(self, poi) -> Self:
        """Operator ``-``: subtract (keep only points NOT in ``poi``)."""
        return self.subtract(poi)
