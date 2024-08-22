import functools
import json
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, TypeGuard, TypeVar

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
from scipy.ndimage import center_of_mass
from typing_extensions import Self

from TPTBox.core import bids_files
from TPTBox.core.nii_wrapper import NII, Image_Reference, to_nii, to_nii_optional
from TPTBox.core.poi_fun.poi_abstract import ROUNDING_LVL, Abstract_POI, POI_Descriptor
from TPTBox.core.vert_constants import (
    AFFINE,
    AX_CODES,
    COORDINATE,
    LABEL_MAX,
    ORIGIN,
    POI_DICT,
    ROTATION,
    SHAPE,
    TRIPLE,
    ZOOMS,
    Abstract_lvl,
    Location,
    Sentinel,
    Vertebra_Instance,
    _register_lvl,
    conversion_poi,
    conversion_poi2text,
    log,
    logging,
    v_idx2name,
    v_name2idx,
)
from TPTBox.logger import Log_Type


### LEGACY DEFINITIONS ###
class _Point3D(TypedDict):
    X: float
    Y: float
    Z: float
    label: int


class _Orientation(TypedDict):
    direction: tuple[str, str, str]


_Centroid_DictList = Sequence[_Orientation | _Point3D]


### CURRENT TYPE DEFFINITONS
C = TypeVar("C", bound="POI")
POI_Reference = bids_files.BIDS_FILE | Path | str | tuple[Image_Reference, Image_Reference, Sequence[int]] | C
ctd_info_blacklist = [
    "zoom",
    "shape",
    "direction",
    "format",
    "rotation",
    "origin",
    "level_one_info",
    "level_two_info",
]  # "location"


@dataclass
class POI(Abstract_POI):
    """
    This class represents a collection of centroids used to define points of interest in medical imaging data.

    Attributes:
        orientation (Ax_Codes): A tuple of three string values representing the orientation of the image.
        centroids (dict): A dictionary of centroid points, where the keys are the labels for the centroid
            points, and values are tuples of three float values representing the x, y, and z coordinates
            of the centroid.
        zoom (Zooms | None): A tuple of three float values representing the zoom level of the image.
            Defaults to None if not provided.
        shape (tuple[float, float, float] | None): A tuple of three integer values representing the shape of the image.
            Defaults to None if not provided.
        format (int | None): An integer value representing the format of the image. Defaults to None if not provided.
        info (dict): Additional information stored as key-value pairs. Defaults to an empty dictionary.
        rotation (Rotation | None): A 3x3 numpy array representing the rotation matrix for the image orientation.
            Defaults to None if not provided.
        origin (Coordinate | None): A tuple of three float values representing the origin of the image in millimeters
            along the x, y, and z axes. Defaults to None if not provided.

    Properties:
        is_global (bool): Property indicating whether the POI is a global POI. Always returns False.
        zoom (Zooms | None): Property getter for the zoom level.
        affine: Property representing the affine transformation for the image.

    Examples:
        >>> # Create a POI object with 2D dictionary input
        >>> from BIDS.core.poi import POI
        >>> poi_data = {
        ...     (1, 0): (10.0, 20.0, 30.0),
        ...     (2, 1): (15.0, 25.0, 35.0),
        ... }
        >>> poi_obj = POI(centroids=poi_data, orientation=("R", "A", "S"), zoom=(1.0, 1.0, 1.0), shape=(256, 256, 100))

        >>> # Access attributes
        >>> print(poi_obj.orientation)
        ('R', 'A', 'S')
        >>> print(poi_obj.centroids)
        {1: {0: (10.0, 20.0, 30.0)}, 2: {1: (15.0, 25.0, 35.0)}}
        >>> print(poi_obj.zoom)
        (1.0, 1.0, 1.0)
        >>> print(poi_obj.shape)
        (256, 256, 100)

        >>> # Update attributes
        >>> poi_obj.rescale_((2.0, 2.0, 2.0))
        >>> poi_obj.centroids[(3, 0)] = (5.0, 15.0, 25.0)
        >>> print(poi_obj)
        POI(centroids={1: {0: (5.0, 10.0, 15.0)}, 2: {1: (7.5, 12.5, 17.5)}, 3: {0: (5.0, 15.0, 25.0)}}, orientation=('R', 'A', 'S'), zoom=(2.0, 2.0, 2.0), info={}, origin=None)

        >>> # Create a copy of the object
        >>> poi_copy = poi_obj.copy()

        >>> # Perform operations
        >>> poi_obj = poi_obj.map_labels({(1): (4), (2): (4)})
        >>> poi_obj.round_(0)
        >>> print(poi_obj)
        POI(centroids={4: {0: (5.0, 10.0, 15.0), 1: (8.0, 12.0, 18.0)}, 3: {0: (5.0, 15.0, 25.0)}}, orientation=('R', 'A', 'S'), zoom=(2.0, 2.0, 2.0), info={}, origin=None)
    """

    orientation: AX_CODES = ("R", "A", "S")
    zoom: ZOOMS = field(init=True, default=None)  # type: ignore
    shape: TRIPLE = field(default=None, repr=True, compare=False)  # type: ignore
    rotation: ROTATION = field(default=None, repr=False, compare=False)  # type: ignore
    origin: COORDINATE = None  # type: ignore
    # internal
    _rotation: ROTATION = field(init=False, default=None, repr=False, compare=False)
    _zoom: ZOOMS = field(init=False, default=(1, 1, 1), repr=False, compare=False)
    _vert_orientation_pir = {}  # Elusive; will not be saved; will not be copied. For Buffering results  # noqa: RUF012

    @property
    def is_global(self):
        return False

    @property
    def rotation(self):
        return self._rotation

    @property
    def zoom(self):
        return self._zoom

    @property
    def affine(self):
        assert self.zoom is not None, "Attribute 'zoom' must be set before calling affine."
        assert self.rotation is not None, "Attribute 'rotation' must be set before calling affine."
        assert self.origin is not None, "Attribute 'origin' must be set before calling affine."
        aff = np.eye(4)
        aff[:3, :3] = self.rotation @ np.diag(self.zoom)
        aff[:3, 3] = self.origin
        return np.round(aff, ROUNDING_LVL)

    @rotation.setter
    def rotation(self, value):
        if isinstance(value, property):
            pass
        elif value is None:
            self._rotation = None
        else:
            self._rotation = np.array(value)

    @zoom.setter
    def zoom(self, value):
        if isinstance(value, property):
            pass
        elif value is None:
            self._zoom = None
        else:
            self._zoom = tuple(round(float(v), ROUNDING_LVL) for v in value)  # type: ignore

    def clone(self, **qargs):
        return self.copy(**qargs)

    def copy(
        self,
        centroids: POI_DICT | POI_Descriptor | None = None,
        orientation: AX_CODES | None = None,
        zoom: ZOOMS | Sentinel = Sentinel(),  # noqa: B008
        shape: tuple[float, float, float] | tuple[float, ...] | Sentinel = Sentinel(),  # noqa: B008
        rotation: ROTATION | Sentinel = Sentinel(),  # noqa: B008
        origin: COORDINATE | Sentinel = Sentinel(),  # noqa: B008
    ) -> Self:
        """Create a copy of the POI object with optional attribute overrides.

        Args:
            centroids (POI_Dict | POI_Descriptor | None, optional): The centroids to use in the copied object.
                Defaults to None, in which case the original centroids will be used.
            orientation (Ax_Codes | None, optional): The orientation code to use in the copied object.
                Defaults to None, in which case the original orientation will be used.
            zoom (Zooms | None | Sentinel, optional): The zoom values to use in the copied object.
                Defaults to Sentinel(), in which case the original zoom values will be used.
            shape (tuple[float, float, float] | None | Sentinel, optional): The shape values to use in the copied object.
                Defaults to Sentinel(), in which case the original shape values will be used.
            rotation (Rotation | None | Sentinel, optional): The rotation matrix to use in the copied object.
                Defaults to Sentinel(), in which case the original rotation matrix will be used.
            origin (Coordinate | None | Sentinel, optional): The origin coordinates to use in the copied object.
                Defaults to Sentinel(), in which case the original origin coordinates will be used.
        Returns:
            POI: A new POI object with the specified attribute overrides.

        Examples:
            >>> centroid_obj = POI(...)
            >>> centroid_obj_copy = centroid_obj.copy(zoom=(2.0, 2.0, 2.0), rotation=rotation_matrix)
        """
        if isinstance(shape, tuple):
            shape = tuple(round(float(v), 7) for v in shape)  # type: ignore

        return POI(
            centroids=centroids.copy() if centroids is not None else self.centroids.copy(),
            orientation=orientation if orientation is not None else self.orientation,
            zoom=zoom if not isinstance(zoom, Sentinel) else self.zoom,
            shape=shape if not isinstance(shape, Sentinel) else self.shape,  # type: ignore
            rotation=rotation if not isinstance(rotation, Sentinel) else self.rotation,
            origin=origin if not isinstance(origin, Sentinel) else self.origin,
            info=deepcopy(self.info),
            format=self.format,
        )

    def local_to_global(self, x: COORDINATE) -> COORDINATE:
        """Converts local coordinates to global coordinates using zoom, rotation, and origin.

        Args:
            x (Coordinate | list[float]): The local coordinate(s) to convert.

        Returns:
            Coordinate: The converted global coordinate(s).

        Raises:
            AssertionError: If the attributes 'zoom', 'rotation', or 'origin' are missing.

        Notes:
            The 'zoom' and 'rotation' attributes should be set before calling this method.

        Examples:
            >>> centroid_obj = Centroids(...)
            >>> centroid_obj.zoom = (2.0, 2.0, 2.0)
            >>> centroid_obj.rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            >>> centroid_obj.origin = (10.0, 20.0, 30.0)
            >>> local_coordinate = (1.0, 2.0, 3.0)
            >>> global_coordinate = centroid_obj.local_to_global(local_coordinate)
        """
        assert self.zoom is not None, "Attribute 'zoom' must be set before calling local_to_global."
        assert self.rotation is not None, "Attribute 'rotation' must be set before calling local_to_global."
        assert self.origin is not None, "Attribute 'origin' must be set before calling local_to_global."

        a = self.rotation @ (np.array(x) * np.array(self.zoom)) + self.origin
        # return tuple(a.tolist())
        return tuple(round(float(v), ROUNDING_LVL) for v in a)

    def global_to_local(self, x: COORDINATE) -> COORDINATE:
        """Converts global coordinates to local coordinates using zoom, rotation, and origin.

        Args:
            x (Coordinate | list[float]): The global coordinate(s) to convert.

        Returns:
            Coordinate: The converted local coordinate(s).

        Raises:
            AssertionError: If the attributes 'zoom', 'rotation', or 'origin' are missing.

        Notes:
            The 'zoom' and 'rotation' attributes should be set before calling this method.

        Examples:
            >>> centroid_obj = Centroids(...)
            >>> centroid_obj.zoom = (2.0, 2.0, 2.0)
            >>> centroid_obj.rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            >>> centroid_obj.origin = (10.0, 20.0, 30.0)
            >>> global_coordinate = (20.0, 30.0, 40.0)
            >>> local_coordinate = centroid_obj.global_to_local(global_coordinate)
        """
        assert self.zoom is not None, "Attribute 'zoom' must be set before calling global_to_local."
        assert self.rotation is not None, "Attribute 'rotation' must be set before calling global_to_local."
        assert self.origin is not None, "Attribute 'origin' must be set before calling global_to_local."

        a = self.rotation.T @ (np.array(x) - self.origin) / np.array(self.zoom)
        # return tuple(a.tolist())
        return tuple(round(float(v), ROUNDING_LVL) for v in a)  # type: ignore

    def crop_centroids(self, **qargs):
        warnings.warn(
            "crop_centroids id deprecated use apply_crop instead",
            DeprecationWarning,
            stacklevel=4,
        )
        return self.apply_crop(**qargs)

    def apply_crop_reverse(
        self: Self,
        o_shift: tuple[slice, slice, slice] | Sequence[slice],
        shape: tuple[int, int, int] | Sequence[int],
        inplace=False,
    ):
        """A Poi crop can be trivially reversed with out any loss. See apply_crop for more information"""
        return self.apply_crop(
            tuple(slice(-shift.start, sh - shift.start) for shift, sh in zip(o_shift, shape, strict=True)),
            inplace=inplace,
        )

    def apply_crop(self: Self, o_shift: tuple[slice, slice, slice] | Sequence[slice], inplace=False):
        """When you crop an image, you have to also crop the centroids.
        There are actually no boundary to be moved, but the origin must be moved to the new 0,0,0
        Points outside the frame are NOT removed. See NII.compute_crop_slice()

        Args:
            o_shift (tuple[slice, slice, slice]): translation of the origin, cause by the crop
            inplace (bool, optional): inplace. Defaults to True.

        Returns:
            Self
        """
        """Crop the centroids based on the given origin shift due to the image crop.

        When you crop an image, you have to also crop the centroids.
        There are actually no boundaries to be moved, but the origin must be moved to the new 0, 0, 0.
        Points outside the frame are NOT removed. See NII.compute_crop_slice().

        Args:
            o_shift (tuple[slice, slice, slice]): Translation of the origin caused by the crop.
            inplace (bool, optional): If True, perform the operation in-place. Defaults to False.

        Returns:
            Centroids: If inplace is True, returns the modified self. Otherwise, returns a new Centroids object.

        Notes:
            The input 'o_shift' should be a tuple of slices for each dimension, specifying the crop range.
            The 'shape' and 'origin' attributes are updated based on the crop information.
        Raises:
            AttributeError: If the old deprecated format for 'o_shift' (a tuple of floats) is used.

        Examples:
            >>> centroid_obj = Centroids(...)
            >>> crop_slice = (slice(10, 20), slice(5, 15), slice(0, 8))
            >>> new_centroids = centroid_obj.crop_centroids(crop_slice)
        """
        origin = None
        shape = None
        try:

            def shift(x, y, z):
                return (
                    float(x - o_shift[0].start),
                    float(y - o_shift[1].start),
                    float(z - o_shift[2].start),
                )

            poi_out = self.apply_all(shift, inplace=inplace)
            if self.shape is not None:
                in_shape = self.shape

                def map_v(sli: slice, i):
                    end = sli.stop
                    if end is None:
                        return in_shape[i]
                    if end >= 0:
                        return end
                    else:
                        return end + in_shape[i]

                shape: TRIPLE | None = tuple(int(map_v(o_shift[i], i) - o_shift[i].start) for i in range(3))  # type: ignore
            if self.origin is not None:
                origin = self.local_to_global(tuple(float(y.start) for y in o_shift))  # type: ignore
                # origin = tuple(float(x + y.start) for x, y in zip(self.origin, o_shift))

        except AttributeError:
            warnings.warn(
                "using o_shift with only a tuple of floats is deprecated. Use tuple(slice(start,end),...) instead. end can be None for no change. Input: "
                + str(o_shift),
                DeprecationWarning,
                stacklevel=4,
            )
            o: tuple[float, float, float] = o_shift  # type: ignore

            def shift2(x, y, z):
                return x - o[0], y - o[1], z - o[2]

            poi_out = self.apply_all(shift2, inplace=inplace)
            shape = None

        if inplace:
            self.shape = shape
            self.origin = origin
            # centroids already replaced
            return self
        out = self.copy(
            centroids=poi_out.centroids,
            shape=shape,
            rotation=self.rotation,
            origin=origin,
        )
        return out

    def crop_centroids_(self, o_shift: tuple[slice, slice, slice]):
        import warnings

        warnings.warn(
            "crop_centroids_ id deprecated use apply_crop instead",
            DeprecationWarning,
            stacklevel=4,
        )  # TODO remove in version 1.0

        return self.apply_crop(o_shift, inplace=True)

    def apply_crop_(self, o_shift: tuple[slice, slice, slice] | Sequence[slice]):
        return self.apply_crop(o_shift, inplace=True)

    def shift_all_centroid_coordinates(
        self,
        translation_vector: tuple[slice, slice, slice] | Sequence[slice] | None,
        inplace=True,
        **kwargs,
    ):
        if translation_vector is None:
            return self
        return self.apply_crop(translation_vector, inplace=inplace, **kwargs)

    def reorient(
        self,
        axcodes_to: AX_CODES = ("P", "I", "R"),
        decimals=ROUNDING_LVL,
        verbose: logging = False,
        inplace=False,
        _shape=None,
    ):
        """Reorients the centroids of an image from the current orientation to the specified orientation.

        This method updates the position of the centroids, zoom level, and shape of the image accordingly.

        Args:
            axcodes_to (Ax_Codes, optional): An Ax_Codes object representing the desired orientation of the centroids.
                Defaults to ("P", "I", "R").
            decimals (int, optional): Number of decimal places to round the coordinates of the centroids after reorientation.
                Defaults to ROUNDING_LVL.
            verbose (bool, optional): If True, print a message indicating the current and new orientation of the centroids.
                Defaults to False.
            inplace (bool, optional): If True, update the current centroid object with the reoriented values.
                If False, return a new centroid object with reoriented values. Defaults to False.
            _shape (tuple[int] | None, optional): The shape of the image. Required if the shape is not already present in the centroid object.

        Returns:
            POI: If inplace is True, returns the updated centroid object.
                If inplace is False, returns a new centroid object with reoriented values.

        Raises:
            ValueError: If the given _shape is not compatible with the shape already present in the centroid object.
            AssertionError: If shape is not provided (either in the centroid object or as _shape argument).

        Examples:
            >>> centroid_obj = POI(...)
            >>> new_orientation = ("A", "P", "L")  # Desired orientation for reorientation
            >>> new_centroid_obj = centroid_obj.reorient(axcodes_to=new_orientation, decimals=4, inplace=False)
        """
        ctd_arr = np.transpose(np.asarray(list(self.centroids.values())))
        v_list = list(self.centroids.keys())
        if ctd_arr.shape[0] == 0:
            log.print(
                "No centroids present",
                verbose=verbose if not isinstance(verbose, bool) else True,
                ltype=Log_Type.WARNING,
            )
            return self if inplace else self.copy()

        ornt_fr = nio.axcodes2ornt(self.orientation)  # original centroid orientation
        ornt_to = nio.axcodes2ornt(axcodes_to)

        if (ornt_fr == ornt_to).all():
            log.print("ctd is already rotated to image with ", axcodes_to, verbose=verbose)
            return self if inplace else self.copy()
        trans = nio.ornt_transform(ornt_fr, ornt_to).astype(int)
        perm: list[int] = trans[:, 0].tolist()

        if self.shape is not None:
            shape = tuple([self.shape[perm.index(i)] for i in range(len(perm))])

            if _shape != shape and _shape is not None:
                raise ValueError(f"Different shapes {shape} <-> {_shape}, types {type(shape)} <-> {type(_shape)}")
        else:
            shape = _shape
        assert shape is not None, "Require shape information for flipping dimensions. Set self.shape or use reorient_centroids_to"
        shp = np.asarray(shape)
        ctd_arr[perm] = ctd_arr.copy()
        for ax in trans:
            if ax[1] == -1:
                size = shp[ax[0]]
                ctd_arr[ax[0]] = np.around(size - ctd_arr[ax[0]], decimals) - 1
        points = POI_Descriptor()
        ctd_arr = np.transpose(ctd_arr).tolist()
        for v, point in zip(v_list, ctd_arr, strict=True):
            points[v] = tuple(point)

        log.print(
            "[*] Centroids reoriented from",
            nio.ornt2axcodes(ornt_fr),
            "to",
            axcodes_to,
            verbose=verbose,
        )
        if self.zoom is not None:
            zoom_i = np.array(self.zoom)
            zoom_i[perm] = zoom_i.copy()
            zoom: ZOOMS | None = tuple(zoom_i)
        else:
            zoom = None
        perm2 = trans[:, 0]
        flip = trans[:, 1]
        if self.origin is not None and self.shape is not None:
            # When the axis is flipped the origin is changing by that side.
            # flip is -1 when when a side (of shape) is moved from the origin
            # if flip = -1 new local point is affine_matmul_(shape[1]-1) else 0
            change = ((-flip) + 1) / 2  # 1 if flip else 0
            change = tuple(a * (s - 1) for a, s in zip(change, self.shape, strict=True))
            origin = self.local_to_global(change)
        else:
            origin = None
        if self.rotation is not None:
            rotation_change = np.zeros((3, 3))
            rotation_change[0, perm2[0]] = flip[0]
            rotation_change[1, perm2[1]] = flip[1]
            rotation_change[2, perm2[2]] = flip[2]
            rotation = self.rotation
            rotation = rotation.copy() @ rotation_change
        else:
            rotation = None
        if inplace:
            self.orientation = axcodes_to
            self.centroids = points
            self.zoom = zoom
            self.shape = shape
            self.origin = origin
            self.rotation = rotation
            return self
        return self.copy(
            orientation=axcodes_to,
            centroids=points,
            zoom=zoom,
            shape=shape,
            origin=origin,
            rotation=rotation,
        )

    def reorient_(
        self,
        axcodes_to: AX_CODES = ("P", "I", "R"),
        decimals=3,
        verbose: logging = False,
        _shape=None,
    ):
        return self.reorient(axcodes_to, decimals=decimals, verbose=verbose, inplace=True, _shape=_shape)

    def reorient_centroids_to(
        self,
        img: Image_Reference,
        decimals=ROUNDING_LVL,
        verbose: logging = False,
        inplace=False,
    ) -> Self:
        # reorient centroids to image orientation
        if not isinstance(img, NII):
            img = to_nii(img)
        axcodes_to: AX_CODES = nio.aff2axcodes(img.affine)  # type: ignore
        return self.reorient(
            axcodes_to,
            decimals=decimals,
            verbose=verbose,
            inplace=inplace,
            _shape=img.shape,
        )

    def reorient_centroids_to_(self, img: Image_Reference, decimals=ROUNDING_LVL, verbose: logging = False) -> Self:
        return self.reorient_centroids_to(img, decimals=decimals, verbose=verbose, inplace=True)

    def rescale(
        self,
        voxel_spacing: ZOOMS = (1, 1, 1),
        decimals=ROUNDING_LVL,
        verbose: logging = True,
        inplace=False,
    ) -> Self:
        """Rescale the centroid coordinates to a new voxel spacing in the current x-y-z-orientation.

        Args:
            voxel_spacing (tuple[float, float, float], optional): New voxel spacing in millimeters. Defaults to (1, 1, 1).
            decimals (int, optional): Number of decimal places to round the rescaled coordinates to. Defaults to ROUNDING_LVL.
            verbose (bool, optional): Whether to print a message indicating that the centroid coordinates have been rescaled. Defaults to True.
            inplace (bool, optional): Whether to modify the current instance or return a new instance. Defaults to False.

        Returns:
            POI: If inplace=True, returns the modified POI instance. Otherwise, returns a new POI instance with rescaled centroid coordinates.

        Raises:
            AssertionError: If the 'zoom' attribute is not set in the Centroids instance.

        Examples:
            >>> centroid_obj = POI(...)
            >>> new_voxel_spacing = (2.0, 2.0, 2.0)  # Desired voxel spacing for rescaling
            >>> rescaled_centroid_obj = centroid_obj.rescale(voxel_spacing=new_voxel_spacing, decimals=4, inplace=False)
        """
        assert self.zoom is not None, "This Centroids instance doesn't have a zoom set. Use centroid.zoom = nii.zoom"

        zms = self.zoom
        shp = list(self.shape) if self.shape is not None else None
        ctd_arr = np.transpose(np.asarray(list(self.centroids.values())))
        v_list = list(self.centroids.keys())
        voxel_spacing = tuple([v if v != -1 else z for v, z in zip(voxel_spacing, zms, strict=True)])
        for i in range(3):
            fkt = zms[i] / voxel_spacing[i]
            if len(v_list) != 0:
                ctd_arr[i] = np.around(ctd_arr[i] * fkt, decimals=decimals)
            if shp is not None:
                shp[i] *= fkt
        points = POI_Descriptor()
        ctd_arr = np.transpose(ctd_arr).tolist()
        for v, point in zip(v_list, ctd_arr, strict=True):
            points[v] = tuple(point)
        log.print(
            "Rescaled centroid coordinates to spacing (x, y, z) =",
            voxel_spacing,
            "mm",
            verbose=verbose,
        )
        if shp is not None:
            shp = tuple(float(v) for v in shp)

        if inplace:
            self.centroids = points
            self.zoom = voxel_spacing
            self.shape = shp
            return self
        return self.copy(centroids=points, zoom=voxel_spacing, shape=shp)

    def rescale_(self, voxel_spacing: ZOOMS = (1, 1, 1), decimals=3, verbose: logging = False) -> Self:
        return self.rescale(
            voxel_spacing=voxel_spacing,
            decimals=decimals,
            verbose=verbose,
            inplace=True,
        )

    def to_global(self):
        """Converts the Centroids object to a global POI_Global object.

        This method converts the local centroid coordinates to global coordinates using the Centroids' zoom,
        rotation, and origin attributes and returns a new POI_Global object.

        Returns:
            POI_Global: A new POI_Global object with the converted global centroid coordinates.

        Examples:
            >>> centroid_obj = Centroids(...)
            >>> global_obj = centroid_obj.to_global()
        """
        import TPTBox.core.poi_fun.poi_global as pg

        return pg.POI_Global(self)

    def resample_from_to(self, ref: NII | Self):
        return self.to_global().to_other(ref)

    def save(
        self,
        out_path: Path | str,
        make_parents=False,
        additional_info: dict | None = None,
        verbose: logging = True,
        save_hint=2,
    ) -> None:
        """
        Saves the centroids to a JSON file.

        Args:
            out_path (Path | str): The path where the JSON file will be saved.
            make_parents (bool, optional): If True, create any necessary parent directories for the output file.
                Defaults to False.
            verbose (bool, optional): If True, print status messages to the console. Defaults to True.
            save_hint: 0 Default, 1 Gruber, 2 POI (readable), 10 ISO-POI (outdated)
        Returns:
            None

        Raises:
            TypeError: If any of the centroids have an invalid type.

        Example:
            >>> centroids = Centroids(...)
            >>> centroids.save("output/centroids.json")
        """
        if make_parents:
            Path(out_path).parent.mkdir(exist_ok=True, parents=True)

        self.sort()
        out_path = str(out_path)
        if len(self.centroids) == 0:
            log.print("POIs empty, not saved:", out_path, ltype=Log_Type.FAIL, verbose=verbose)
            return
        json_object, print_add = _poi_to_dict_list(self, additional_info, save_hint, verbose)

        # Problem with python 3 and int64 serialization.
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(type(o))

        with open(out_path, "w") as f:
            json.dump(json_object, f, default=convert, indent=4)
        log.print("POIs saved:", out_path, print_add, ltype=Log_Type.SAVE, verbose=verbose)

    def make_point_cloud_nii(self, affine=None, s=8, sphere=False):
        """Create point cloud NIfTI images from the centroid coordinates.

        This method generates two NIfTI images, one for the regions and another for the subregions,
        representing the point cloud with a specified neighborhood size.

        Args:
            affine (np.ndarray, optional): The affine transformation matrix for the NIfTI image.
                Defaults to None. If None, the centroid object's affine will be used.
            s (int, optional): The neighborhood size. Defaults to 8.

        Returns:
            tuple[NII, NII]: A tuple containing two NII objects representing the point cloud for regions and subregions.

        Raises:
            AssertionError: If the 'shape' or 'zoom' attributes are not set in the Centroids instance.

        Examples:
            >>> centroid_obj = Centroids(...)
            >>> neighborhood_size = 10
            >>> region_cloud, subregion_cloud = centroid_obj.make_point_cloud_nii(s=neighborhood_size)
        """

        assert self.shape is not None, "need shape information"
        assert self.zoom is not None, "need shape information"
        if affine is None:
            affine = self.affine
        arr = np.zeros(self.shape_int)
        arr2 = np.zeros(self.shape_int)
        s1 = min(s // 2, 1)
        s2 = min(s - s1, 1)
        from math import ceil, floor

        if sphere:
            from tqdm import tqdm

            for region, subregion, (x, y, z) in tqdm(self.items(), total=len(self)):
                coords = np.ogrid[: self.shape[0], : self.shape[1], : self.shape[2]]
                zoom = self.zoom
                distance = np.sqrt(
                    ((coords[0] - int(x)) * zoom[0]) ** 2 + ((coords[1] - int(y)) * zoom[1]) ** 2 + ((coords[2] - int(z)) * zoom[2]) ** 2
                )
                arr += np.asarray(region * (distance <= s / 2), dtype=np.uint16)
                arr2 += np.asarray(subregion * (distance <= s / 2), dtype=np.uint16)
        else:
            for region, subregion, (x, y, z) in self.items():
                arr[
                    max(int(floor(x - s1 / self.zoom[0])) + 1, 0) : min(int(ceil(x + s2 / self.zoom[0] + 1)), self.shape[0]),
                    max(int(floor(y - s1 / self.zoom[1])) + 1, 0) : min(int(ceil(y + s2 / self.zoom[1] + 1)), self.shape[1]),
                    max(int(floor(z - s1 / self.zoom[2])) + 1, 0) : min(int(ceil(z + s2 / self.zoom[2] + 1)), self.shape[2]),
                ] = region
                arr2[
                    max(int(floor(x - s1 / self.zoom[0])) + 1, 0) : min(int(ceil(x + s2 / self.zoom[0] + 1)), self.shape[0]),
                    max(int(floor(y - s1 / self.zoom[1])) + 1, 0) : min(int(ceil(y + s2 / self.zoom[1] + 1)), self.shape[1]),
                    max(int(floor(z - s1 / self.zoom[2])) + 1, 0) : min(int(ceil(z + s2 / self.zoom[2] + 1)), self.shape[2]),
                ] = subregion
        nii = nib.Nifti1Image(arr, affine=affine)
        nii2 = nib.Nifti1Image(arr2, affine=affine)
        return NII(nii, seg=True), NII(nii2, seg=True)

    def filter_points_inside_shape(self, inplace=False) -> Self:
        """Filter out centroid points that are outside the defined shape.

        This method checks each centroid point and removes any point whose coordinates
        are outside the defined shape.

        Returns:
            POI: A new POI object containing centroid points that are inside the defined shape.

        Examples:
            >>> centroid_obj = POI(...)
            >>> filtered_centroids = centroid_obj.filter_points_inside_shape()
        """
        if self.shape is None:
            raise ValueError("Cannot filter points outside shape as the shape attribute is not defined.")

        filtered_centroids = POI_Descriptor()
        for region, subregion, (x, y, z) in self.centroids.items():
            if 0 <= x < self.shape[0] and 0 <= y < self.shape[1] and 0 <= z < self.shape[2]:
                filtered_centroids[(region, subregion)] = (x, y, z)
        if inplace:
            self.centroids = filtered_centroids
            return self
        return self.copy(filtered_centroids)

    @classmethod
    def load(cls, poi: POI_Reference):
        """Load a Centroids object from various input sources.

        This method provides a convenient way to load a Centroids object from different sources,
        including BIDS files, file paths, image references, or existing POI objects.

        Args:
            poi (Centroid_Reference): The input source from which to load the Centroids object.
                It can be one of the following types:
                - BIDS_FILE: A BIDS file representing the Centroids object.
                - Path: The path to the file containing the Centroids object.
                - str: The string representation of the Centroids object file path.
                - Tuple[Image_Reference, Image_Reference, list[int]]: A tuple containing two Image_Reference objects
                and a list of integers representing the centroid data.
                - POI: An existing POI object to be loaded.

        Returns:
            POI: The loaded Centroids object.

        Examples:
            >>> # Load from a BIDS file
            >>> bids_file_path = BIDS_FILE("/path/to/centroids.json", "/path/to/dataset/")
            >>> loaded_poi = POI.load(bids_file_path)

            >>> # Load from a file path
            >>> file_path = "/path/to/centroids.json"
            >>> loaded_poi = POI.load(file_path)

            >>> # Load from an image reference tuple and centroid data
            >>> image_ref1 = Image_Reference(...)
            >>> image_ref2 = Image_Reference(...)
            >>> centroid_data = [1, 2, 3]
            >>> loaded_poi = POI.load((image_ref1, image_ref2, centroid_data))

            >>> # Load from an existing POI object
            >>> existing_poi = POI(...)
            >>> loaded_poi = POI.load(existing_poi)
        """
        return load_poi(poi)

    def assert_affine(
        self,
        other: Self | NII | None = None,
        ignore_missing_values: bool = False,
        affine: AFFINE | None = None,
        zoom: ZOOMS | None = None,
        orientation: AX_CODES | None = None,
        rotation: ROTATION | None = None,
        origin: ORIGIN | None = None,
        shape: SHAPE | None = None,
        shape_tolerance: float = 0.0,
        origin_tolerance: float = 0.0,
        error_tolerance: float = 1e-4,
        raise_error: bool = True,
        verbose: logging = False,
    ):
        """Checks if the different metadata is equal to some comparison entries

        Args:
            other (Self | POI | None, optional): If set, will assert each entry of that object instead. Defaults to None.
            affine (AFFINE | None, optional): Affine matrix to compare against. If none, will not assert affine. Defaults to None.
            zms (Zooms | None, optional): Zoom to compare against. If none, will not assert zoom. Defaults to None.
            orientation (Ax_Codes | None, optional): Orientation to compare against. If none, will not assert orientation. Defaults to None.
            origin (ORIGIN | None, optional): Origin to compare against. If none, will not assert origin. Defaults to None.
            shape (SHAPE | None, optional): Shape to compare against. If none, will not assert shape. Defaults to None.
            shape_tolerance (float, optional): error tolerance in shape as float, as POIs can have float shapes. Defaults to 0.0.
            error_tolerance (float, optional): Accepted error tolerance in all assertions except shape. Defaults to 1e-4.
            raise_error (bool, optional): If true, will raise AssertionError if anything is found. Defaults to True.
            verbose (logging, optional): If true, will print out each assertion mismatch. Defaults to False.

        Raises:
            AssertionError: If any of the assertions failed and raise_error is True

        Returns:
            bool: True if there are no assertion errors
        """
        found_errors: list[str] = []

        # Make Checks
        if other is not None:
            other_data = other._extract_affine()
            other_match = self.assert_affine(
                other=None,
                **other_data,
                raise_error=raise_error,
                shape_tolerance=shape_tolerance,
                error_tolerance=error_tolerance,
                origin_tolerance=origin_tolerance,
            )
            if not other_match:
                found_errors.append(f"object mismatch {self!s}, {other!s}")
        if affine is not None and (not ignore_missing_values or self.affine is not None):
            if self.affine is None:
                found_errors.append(f"affine mismatch {self.affine}, {affine}")
            else:
                affine_diff = self.affine - affine
                affine_match = np.all([abs(a) <= error_tolerance for a in affine_diff.flatten()])
                found_errors.append(f"affine mismatch {self.affine}, {affine}") if not affine_match else None
        if rotation is not None and (not ignore_missing_values or self.rotation is not None):
            if self.rotation is None:
                found_errors.append(f"rotation mismatch {self.rotation}, {rotation}")
            else:
                rotation_diff = self.rotation - rotation
                rotation_match = np.all([abs(a) <= error_tolerance for a in rotation_diff.flatten()])
                found_errors.append(f"rotation mismatch {self.rotation}, {rotation}") if not rotation_match else None
        if zoom is not None and (not ignore_missing_values or self.zoom is not None):
            if self.zoom is None:
                found_errors.append(f"zoom mismatch {self.zoom}, {zoom}")
            else:
                zms_diff = (self.zoom[i] - zoom[i] for i in range(3))
                zms_match = np.all([abs(a) <= error_tolerance for a in zms_diff])
                found_errors.append(f"zoom mismatch {self.zoom}, {zoom}") if not zms_match else None
        if orientation is not None and (not ignore_missing_values or self.affine is not None):
            if self.orientation is None:
                found_errors.append(f"orientation mismatch {self.orientation}, {orientation}")
            else:
                orientation_match = np.all([i == orientation[idx] for idx, i in enumerate(self.orientation)])
                found_errors.append(f"orientation mismatch {self.orientation}, {orientation}") if not orientation_match else None
        if origin is not None and (not ignore_missing_values or self.origin is not None):
            if self.origin is None:
                found_errors.append(f"origin mismatch {self.origin}, {origin}")
            else:
                origin_diff = (self.origin[i] - origin[i] for i in range(3))
                origin_match = np.all([abs(a) <= origin_tolerance for a in origin_diff])
                found_errors.append(f"origin mismatch {self.origin}, {origin}") if not origin_match else None
        if shape is not None and (not ignore_missing_values or self.shape is not None):
            if self.shape is None:
                found_errors.append(f"shape mismatch {self.shape}, {shape}")
            else:
                shape_diff = (float(self.shape[i]) - float(shape[i]) for i in range(3))
                shape_match = np.all([abs(a) <= shape_tolerance for a in shape_diff])
                found_errors.append(f"shape mismatch {self.shape}, {shape}") if not shape_match else None

        # Print errors
        for err in found_errors:
            log.print(err, Log_Type.FAIL, verbose=verbose)

        # Final conclusion and possible raising of AssertionError
        has_errors = len(found_errors) > 0
        if raise_error and has_errors:
            raise AssertionError(f"assert_affine failed with {found_errors}")

        return not has_errors

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, POI):
            return False
        else:
            value2: Self = value  # type: ignore
        if not self.assert_affine(value2, raise_error=False):
            return False
        return self.centroids == value2.centroids


######## Saving #######
def _is_Point3D(obj) -> TypeGuard[_Point3D]:
    return "label" in obj and "X" in obj and "Y" in obj and "Z" in obj


FORMAT_DOCKER = 0
FORMAT_GRUBER = 1
FORMAT_POI = 2
FORMAT_OLD_POI = 10
format_key = {FORMAT_DOCKER: "docker", FORMAT_GRUBER: "guber", FORMAT_POI: "POI"}
format_key2value = {value: key for key, value in format_key.items()}


def _poi_to_dict_list(ctd: POI, additional_info: dict | None, save_hint=0, verbose: logging = False):  # noqa: C901
    ori: dict[str, str | COORDINATE | AX_CODES] = {"direction": ctd.orientation}
    print_out = ""
    if ctd.zoom is not None:
        ori["zoom"] = ctd.zoom
    if ctd.origin is not None:
        ori["origin"] = ctd.origin  # type: ignore
    if ctd.rotation is not None:
        ori["rotation"] = ctd.rotation  # type: ignore
    if ctd.shape is not None:
        ori["shape"] = ctd.shape  # type: ignore
    if save_hint in format_key:
        ori["format"] = format_key[save_hint]  # type: ignore
        print_out = "in format " + format_key[save_hint]

    ori["level_one_info"] = str(ctd.level_one_info.__name__)
    ori["level_two_info"] = str(ctd.level_two_info.__name__)
    if additional_info is not None:
        for k, v in additional_info.items():
            if k not in ori:
                ori[k] = v

    for k, v in ctd.info.items():
        if k not in ori:
            ori[k] = v

    dict_list: list[_Orientation | (_Point3D | dict)] = [ori]

    if save_hint == FORMAT_OLD_POI:
        ctd = ctd.rescale((1, 1, 1), verbose=verbose).reorient_(("R", "P", "I"), verbose=verbose)
        dict_list = []

    temp_dict = {}
    ctd.sort()
    for vert_id, subreg_id, (x, y, z) in ctd.items():
        if save_hint == FORMAT_DOCKER:
            dict_list.append({"label": subreg_id * LABEL_MAX + vert_id, "X": x, "Y": y, "Z": z})
        elif save_hint == FORMAT_GRUBER:
            v = v_idx2name[vert_id].replace("T", "TH") + "_" + conversion_poi2text[subreg_id]
            dict_list.append({"label": v, "X": x, "Y": y, "Z": z})
        elif save_hint == FORMAT_POI:
            v_name = ctd.level_one_info._get_name(vert_id, no_raise=True)
            # sub_name = v_idx2name[subreg_id]
            if v_name not in temp_dict:
                temp_dict[v_name] = {}
            temp_dict[v_name][subreg_id] = (x, y, z)
        elif save_hint == FORMAT_OLD_POI:
            if vert_id not in temp_dict:
                temp_dict[vert_id] = {}
            temp_dict[vert_id][str(subreg_id)] = str((float(x), float(y), float(z)))
        else:
            raise NotImplementedError(save_hint)
    if len(temp_dict) != 0:
        if save_hint == FORMAT_OLD_POI:
            for k, v in temp_dict.items():
                out_dict = {"vert_label": str(k), **v}
                dict_list.append(out_dict)
        else:
            dict_list.append(temp_dict)
    return dict_list, print_out


######### Load #############
# Handling centroids #


def load_poi(ctd_path: POI_Reference, verbose=True) -> POI:  # noqa: ARG001
    """
    Load centroids from a file or a BIDS file object.

    Args:
        ctd_path (Centroid_Reference): Path to a file or BIDS file object from which to load centroids.
            Alternatively, it can be a tuple containing the following items:
            - vert: str, the name of the vertebra.
            - subreg: str, the name of the subregion.
            - ids: list[int | Location], a list of integers and/or Location objects used to filter the centroids.

    Returns:
        A Centroids object containing the loaded centroids.

    Raises:
        AssertionError: If `ctd_path` is not a recognized type.

    """
    if isinstance(ctd_path, POI):
        return ctd_path
    elif isinstance(ctd_path, bids_files.BIDS_FILE):
        dict_list: _Centroid_DictList = ctd_path.open_json()  # type: ignore
    elif isinstance(ctd_path, Path | str):
        with open(ctd_path) as json_data:
            dict_list: _Centroid_DictList = json.load(json_data)
            json_data.close()
    elif isinstance(ctd_path, tuple):
        vert = ctd_path[0]
        subreg = ctd_path[1]
        ids: list[int | Location] = ctd_path[2]  # type: ignore
        return calc_poi_from_subreg_vert(vert, subreg, subreg_id=ids)
    else:
        raise TypeError(f"{type(ctd_path)}\n{ctd_path}")
    ### format_POI_old has no META header
    if "direction" not in dict_list[0] and "vert_label" in dict_list[0]:
        return _load_format_POI_old(dict_list)  # This file if used in the old POI-pipeline and is deprecated

    assert "direction" in dict_list[0], f'File format error: first index must be a "Direction" but got {dict_list[0]}'
    axcode: AX_CODES = tuple(dict_list[0]["direction"])  # type: ignore
    zoom: ZOOMS = dict_list[0].get("zoom", None)  # type: ignore
    shape = dict_list[0].get("shape", None)  # type: ignore
    shape = tuple(shape) if shape is not None else None
    format_ = dict_list[0].get("format", None)
    origin = dict_list[0].get("origin", None)
    origin = tuple(origin) if origin is not None else None
    rotation: ROTATION = dict_list[0].get("rotation", None)
    level_one_info = _register_lvl[dict_list[0].get("level_one_info", Vertebra_Instance.__name__)]
    level_two_info = _register_lvl[dict_list[0].get("level_two_info", Location.__name__)]

    info = {k: v for k, v in dict_list[0].items() if k not in ctd_info_blacklist}

    format_ = format_key2value[format_] if format_ is not None else None
    centroids = POI_Descriptor()
    if format_ in (FORMAT_DOCKER, FORMAT_GRUBER) or format_ is None:
        _load_docker_centroids(dict_list, centroids, format_)
    elif format_ == FORMAT_POI:
        _load_POI_centroids(dict_list, centroids, level_one_info, level_two_info)
    else:
        raise NotImplementedError(format_)
    return POI(
        centroids,
        orientation=axcode,
        zoom=zoom,
        shape=shape,  # type: ignore
        format=format_,
        info=info,
        origin=origin,  # type: ignore
        rotation=rotation,  # type: ignore
        level_one_info=level_one_info,
        level_two_info=level_two_info,
    )  # type: ignore


def _load_docker_centroids(dict_list, centroids: POI_Descriptor, format_):  # noqa: ARG001
    for d in dict_list[1:]:
        assert "direction" not in d, f'File format error: only first index can be a "direction" but got {dict_list[0]}'
        if "nan" in str(d):  # skipping NaN centroids
            continue
        elif _is_Point3D(d):
            try:
                a = int(d["label"])
                subreg = a // LABEL_MAX
                if subreg == 0:
                    subreg = 50
                centroids[a % LABEL_MAX, subreg] = (d["X"], d["Y"], d["Z"])
            except Exception:
                try:
                    number, subreg = str(d["label"]).split("_", maxsplit=1)
                    number = number.replace("TH", "T").replace("SA", "S1")
                    vert_id = v_name2idx[number]
                    subreg_id = conversion_poi[subreg]
                    centroids[vert_id, subreg_id] = (d["X"], d["Y"], d["Z"])
                except Exception:
                    print(f'Label {d["label"]} is not an integer and cannot be converted to an int')
                    centroids[0, d["label"]] = (d["X"], d["Y"], d["Z"])
        else:
            raise ValueError(d)


def _load_format_POI_old(dict_list):
    # [
    # {
    #    "vert_label": "8",
    #    "85": "(281, 185, 274)",
    #    ...
    # }{...}
    # ...
    # ]
    centroids = POI_Descriptor()
    for d in dict_list:
        d: dict[str, str]
        vert_id = int(d["vert_label"])
        for k, v in d.items():
            if k == "vert_label":
                continue
            sub_id = int(k)
            t = v.replace("(", "").replace(")", "").replace(" ", "").split(",")
            t = tuple(float(x) for x in t)
            centroids[vert_id, sub_id] = t
    return POI(centroids, orientation=("R", "P", "I"), zoom=(1, 1, 1), shape=None, format=FORMAT_OLD_POI, rotation=None)  # type: ignore


def _load_POI_centroids(
    dict_list,
    centroids: POI_Descriptor,
    level_one_info: Abstract_lvl,
    level_two_info: Abstract_lvl,
):
    assert len(dict_list) == 2
    d: dict[int | str, dict[int | str, tuple[float, float, float]]] = dict_list[1]
    for vert_id, v in d.items():
        vert_id = level_one_info._get_id(vert_id, no_raise=True)  # noqa: PLW2901
        for sub_id, t in v.items():
            sub_id = level_two_info._get_id(sub_id)  # noqa: PLW2901
            centroids[vert_id, sub_id] = tuple(t)


def loc2int(i: int | Location):
    if isinstance(i, int):
        return i
    return i.value


def loc2int_list(i: int | Location | Sequence[int | Location]):
    if isinstance(i, Sequence):
        return [loc2int(j) for j in i]
    if isinstance(i, int):
        return [i]
    return [i.value]


def int2loc(
    i: int | Location | Sequence[int | Location] | Sequence[Location] | Sequence[int],
) -> Location | Sequence[Location]:
    if isinstance(i, Sequence):
        return [int2loc(j) for j in i]  # type: ignore
    elif isinstance(i, int):
        # try:
        return Location(i)
        # except Exception:
        #    return i
    return i


def calc_poi_labeled_buffered(
    msk_reference: Image_Reference,
    subreg_reference: Image_Reference | None,
    out_path: Path | str,
    subreg_id: int | Location | Sequence[int | Location] = 50,
    verbose=True,
    override=False,
    decimals=3,
    # additional_folder=False,
    check_every_point=True,
    # use_vertebra_special_action=True,
) -> POI:
    """
    Computes the centroids of the given mask `msk_reference` with respect to the given subregion `subreg_reference`,
    and saves them to a file at `out_path` (if `override=False` and the file already exists, the function loads and returns
    the existing centroids from the file).

    If `out_path` is None and `msk_reference` is a `BIDS.bids_files.BIDS_FILE` object, the function generates a path to
    save the centroids file based on the `label` attribute of the file and the given `subreg_id`.

    If `subreg_reference` is None, the function computes the centroids using only `msk_reference`.

    If `subreg_reference` is not None, the function computes the centroids with respect to the given `subreg_id` in the
    subregion defined by `subreg_reference`.

    Args:
        msk_reference (Image_Reference): The mask to compute the centroids from.
        subreg_reference (Image_Reference | None, optional): The subregion mask to compute the centroids relative to.
        out_path (Path | None, optional): The path to save the computed centroids to.
        subreg_id (int | Location | list[int | Location], optional): The ID of the subregion to compute centroids in.
        verbose (bool, optional): Whether to print verbose output during the computation.
        override (bool, optional): Whether to overwrite any existing centroids file at `out_path`.
        decimals (int, optional): The number of decimal places to round the computed centroid coordinates to.
        additional_folder (bool, optional): Whether to add a `/ctd/` folder to the path generated for the output file.

    Returns:
        Centroids: The computed centroids, as a `Centroids` object.
    """
    assert out_path is not None, "Automatic path generation is deprecated"
    out_path = Path(out_path)
    # assert out_path is not None or isinstance(
    #    msk_reference, BIDS.bids_files.BIDS_FILE
    # ), "Automatic path generation is only possible with a BIDS_FILE"
    # if out_path is None and isinstance(msk_reference, BIDS.bids_files.BIDS_FILE):
    #    if not isinstance(subreg_id, list) and subreg_id != -1:
    #        name = subreg_idx2name[loc2int(subreg_id)]
    #    elif subreg_reference is None:
    #        name = msk_reference.get("label", default="full")
    #    else:
    #        name = "multi"
    #    assert name is not None
    #    out_path = msk_reference.get_changed_path(
    #        file_type="json",
    #        format="ctd",
    #        info={"label": name.replace("_", "-")},
    #        parent="derivatives" if msk_reference.get_parent() == "rawdata" else msk_reference.get_parent(),
    #        additional_folder="ctd" if additional_folder else None,
    #    )
    assert out_path is not None
    if override:
        out_path.unlink()
    log.print(f"[*] Generate ctd json towards {out_path}", verbose=verbose)

    msk_nii = to_nii(msk_reference, True)
    sub_nii = to_nii_optional(subreg_reference, True)
    if (sub_nii is None or not check_every_point) and out_path.exists():
        return POI.load(out_path)
    if sub_nii is not None:
        ctd = calc_poi_from_subreg_vert(
            msk_nii,
            sub_nii,
            buffer_file=out_path,
            decimals=decimals,
            subreg_id=subreg_id,
            verbose=verbose,
        )
    else:
        assert not isinstance(subreg_id, Sequence), "Missing instance+semantic map for multiple Values"
        ctd = calc_centroids(msk_nii, subreg_id=loc2int(subreg_id), decimals=decimals)

    ctd.save(out_path, verbose=verbose)
    return ctd


def _buffer_it(func):
    """Decorator that reports the execution time."""

    @functools.wraps(func)
    def wrap(*args, **kwargs):
        buffer_file = kwargs.get("buffer_file", None)
        extend_to = kwargs.get("extend_to", None)
        save_buffer_file = kwargs.get("save_buffer_file", False)
        len_pref = 0
        if buffer_file is not None and Path(buffer_file).exists():
            assert extend_to is None
            print("load")
            extend_to = POI.load(buffer_file)
            len_pref = len(extend_to)
        kwargs["extend_to"] = extend_to
        # FUN
        poi: POI = func(*args, **kwargs)
        if save_buffer_file and len_pref < len(poi):
            assert buffer_file is not None
            poi.save(buffer_file)

        return poi

    return wrap


# @_buffer_it
# def calc_centroids_from_subreg_vert(
#    vert_msk: Image_Reference,
#    subreg: Image_Reference,
#    *,
#    buffer_file: str | Path | None = None,  # used by wrapper
#    save_buffer_file=False,  # used by wrapper
#    decimals=2,
#    subreg_id: int | Location | Sequence[int | Location] | Sequence[Location] | Sequence[int] = 50,
#    axcodes_to: Ax_Codes | None = None,
#    verbose: logging = True,
#    extend_to: POI | None = None,
#    use_vertebra_special_action=True,
#    _vert_ids=None,
# ) -> POI:
#    """
#    Calculates the centroids of a subregion within a vertebral mask.
#
#    Args:
#        vert_msk (Image_Reference): A vertebral mask image reference.
#        subreg (Image_Reference): An image reference for the subregion of interest.
#        decimals (int, optional): Number of decimal places to round the output coordinates to. Defaults to 1.
#        subreg_id (int | Location | list[int | Location], optional): The ID(s) of the subregion(s) to calculate centroids for. Defaults to 50.
#        axcodes_to (Ax_Codes | None, optional): A tuple of axis codes indicating the target orientation of the images. Defaults to None.
#        verbose (bool, optional): Whether to print progress messages. Defaults to False.
#        fixed_offset (int, optional): A fixed offset value to add to the calculated centroid coordinates. Defaults to 0.
#        extend_to (POI | None, optional): An existing POI object to extend with the new centroid values. Defaults to None.
#
#    Returns:
#        POI: A POI object containing the calculated centroid coordinates.
#    """
#
#    vert_msk = to_nii(vert_msk, seg=True)
#    subreg_msk = to_nii(subreg, seg=True)
#    assert vert_msk.origin == subreg_msk.origin, (vert_msk.origin, subreg_msk.origin)
#    if _vert_ids is None:
#        _vert_ids = vert_msk.unique()
#    log.print("Calc centroids from subregion id", int2loc(subreg_id), vert_msk.shape, verbose=verbose)
#
#    if axcodes_to is not None:
#        # Like: ("P","I","R")
#        vert_msk = vert_msk.reorient(verbose=verbose, axcodes_to=axcodes_to, inplace=False)
#        subreg_msk = subreg_msk.reorient(verbose=verbose, axcodes_to=axcodes_to, inplace=False)
#    assert vert_msk.orientation == subreg_msk.orientation
#    # Recursive call for multiple subregion ids
#    if isinstance(subreg_id, Sequence):
#        if extend_to is None:
#            poi = POI({}, **vert_msk._extract_affine(), format=FORMAT_POI)
#        else:
#            extend_to.format = FORMAT_POI
#            poi = extend_to
#            assert poi.orientation == vert_msk.orientation, (poi.orientation, vert_msk.orientation)
#
#        print("list") if verbose else None
#
#        for idx in subreg_id:
#            idx = loc2int(idx)
#            if not all((v, idx) in poi for v in _vert_ids):
#                poi = calc_centroids_from_subreg_vert(
#                    vert_msk,
#                    subreg_msk,
#                    buffer_file=None,
#                    subreg_id=loc2int(idx),
#                    verbose=verbose,
#                    extend_to=poi,
#                    decimals=decimals,
#                    _vert_ids=_vert_ids,
#                )
#        return poi
#    # Prepare mask to binary mask
#    vert_arr = vert_msk.get_seg_array()
#    subreg_arr = subreg_msk.get_seg_array()
#    assert subreg_msk.shape == vert_arr.shape, "Shape miss-match" + str(subreg_msk.shape) + str(vert_arr.shape)
#    subreg_id = loc2int(subreg_id)
#    if use_vertebra_special_action:
#        mapping_vert = {
#            Location.Vertebra_Disc.value: 100,
#            Location.Vertebral_Body_Endplate_Superior.value: 200,
#            Location.Vertebral_Body_Endplate_Inferior.value: 300,
#        }
#        if subreg_id == Location.Vertebra_Corpus.value:
#            subreg_arr[subreg_arr == 49] = Location.Vertebra_Corpus.value
#        elif subreg_id == Location.Vertebra_Full.value:
#            subreg_arr = vert_arr.copy()
#            subreg_arr[subreg_arr >= 26] = 0
#            subreg_arr[subreg_arr != 0] = Location.Vertebra_Full.value
#
#        if subreg_id in mapping_vert:  # IVD / Endplates Superior / Endplate Inferior
#            vert_arr[subreg_arr != subreg_id] = 0
#            vert_arr[vert_arr >= mapping_vert[subreg_id] + 100] = 0
#            vert_arr[vert_arr <= mapping_vert[subreg_id]] = 0
#            vert_arr[vert_arr != 0] -= mapping_vert[subreg_id]
#        else:
#            vert_arr[subreg_arr >= 100] = 0
#            subreg_arr[subreg_arr >= 100] = 0
#            if use_vertebra_special_action and ((subreg_id in range(81, 128)) or subreg_id == Location.Spinal_Canal.value):
#                from TPTBox.core.vertebra_pois_non_centroids import compute_non_centroid_pois
#
#                if extend_to is None:
#                    extend_to = POI({}, **vert_msk._extract_affine(), format=FORMAT_POI)
#                compute_non_centroid_pois(extend_to, int2loc(subreg_id), vert_msk, subreg_msk, _vert_ids=_vert_ids, log=log)
#                return extend_to
#            elif subreg_id < 40 or subreg_id > 50:
#                raise NotImplementedError(
#                    f"POI of subreg {subreg_id} - {Location(subreg_id) if subreg_id in Location else 'undefined'} is not a centroid. If you want the general Centroid computation use use_vertebra_special_action = False"
#                )
#    vert_arr[subreg_arr != subreg_id] = 0
#    vert_arr[vert_arr >= 100] = 0
#
#    msk_data = vert_arr
#    if extend_to is not None and subreg_id in extend_to.centroids.keys_subregion():
#        missing = False
#        for reg in np.unique(vert_arr):
#            if reg == 0:
#                continue
#            if (reg, subreg_id) in extend_to:
#                continue
#            missing = True
#            break
#        if not missing:
#            return extend_to
#
#    nii = nib.nifti1.Nifti1Image(msk_data, vert_msk.affine, vert_msk.header)
#    poi = calc_centroids(nii, decimals=decimals, subreg_id=subreg_id, extend_to=extend_to)
#    return poi
#


@_buffer_it
def calc_poi_from_subreg_vert(
    vert: Image_Reference,
    subreg: Image_Reference,
    *,
    buffer_file: str | Path | None = None,  # used by wrapper  # noqa: ARG001
    save_buffer_file=False,  # used by wrapper  # noqa: ARG001
    decimals=2,
    subreg_id: int | Location | Sequence[int | Location] | Sequence[Location] | Sequence[int] = 50,
    verbose: logging = True,
    extend_to: POI | None = None,
    # use_vertebra_special_action=True,
    _vert_ids=None,
    _print_phases=False,
) -> POI:
    """
    Calculates the centroids of a subregion within a vertebral mask. This function is spine opinionated, the general implementation is "calc_centroids_from_two_masks".
    Args:
        vert_msk (Image_Reference): A vertebral mask image reference.
        subreg (Image_Reference): An image reference for the subregion of interest.
        decimals (int, optional): Number of decimal places to round the output coordinates to. Defaults to 1.
        subreg_id (int | Location | list[int | Location], optional): The ID(s) of the subregion(s) to calculate centroids for. Defaults to 50.
        axcodes_to (Ax_Codes | None, optional): A tuple of axis codes indicating the target orientation of the images. Defaults to None.
        verbose (bool, optional): Whether to print progress messages. Defaults to False.
        fixed_offset (int, optional): A fixed offset value to add to the calculated centroid coordinates. Defaults to 0.
        extend_to (POI | None, optional): An existing POI object to extend with the new centroid values. Defaults to None.
    Returns:
        POI: A POI object containing the calculated centroid coordinates.
    """
    vert_msk = to_nii(vert, seg=True)
    subreg_msk = to_nii(subreg, seg=True)
    org_shape = subreg_msk.shape
    try:
        crop = vert_msk.compute_crop()
        crop = subreg_msk.compute_crop(maximum_size=crop)
    # crop = (slice(0, subreg_msk.shape[0]), slice(0, subreg_msk.shape[1]), slice(0, subreg_msk.shape[2]))
    except ValueError:
        return POI({}, **vert_msk._extract_affine(), format=FORMAT_POI) if extend_to is None else extend_to.copy()
    vert_msk.assert_affine(subreg_msk)
    vert_msk = vert_msk.apply_crop(crop)
    subreg_msk = subreg_msk.apply_crop(crop)
    extend_to = (
        POI(
            {},
            **vert_msk._extract_affine(),
            format=FORMAT_POI,
            level_one_info=Vertebra_Instance,
            level_two_info=Location,
        )
        if extend_to is None
        else extend_to.apply_crop(crop, inplace=True)
    )

    if _vert_ids is None:
        _vert_ids = vert_msk.unique()

    from TPTBox.core.poi_fun.vertebra_pois_non_centroids import add_prerequisites, compute_non_centroid_pois

    subreg_id = add_prerequisites(int2loc(subreg_id if isinstance(subreg_id, Sequence) else [subreg_id]))  # type: ignore

    log.print("Calc centroids from subregion id", subreg_id, vert_msk.shape, verbose=verbose)
    subreg_id_int = set(loc2int_list(subreg_id))
    subreg_id_int_phase_1 = tuple(
        filter(
            lambda i: i < 53 and i not in [Location.Vertebra_Full.value, Location.Dens_axis.value],
            subreg_id_int,
        )
    )
    # Step 1 get all required locations, crop vert/subreg
    # Step 2 calc centroids

    print("step 2", subreg_id_int) if _print_phases else None
    if len(subreg_id_int_phase_1) != 0:
        arr = vert_msk.get_array()
        arr[arr >= 100] = 0
        vert_only_bone = vert_msk.set_array(arr)
        arr = subreg_msk.get_array()
        # if use_vertebra_special_action:
        arr[arr == 49] = Location.Vertebra_Corpus.value
        subreg_msk = subreg_msk.set_array(arr)
        extend_to = calc_centroids_from_two_masks(
            vert_only_bone,
            subreg_msk,
            decimals=decimals,
            limit_ids_of_lvl_2=subreg_id_int_phase_1,
            verbose=verbose if isinstance(verbose, bool) else True,
            extend_to=extend_to,
        )
        [subreg_id_int.remove(i) for i in subreg_id_int_phase_1]
    # Step 3 Vertebra_Full
    print("step 3", subreg_id_int) if _print_phases else None
    if Location.Vertebra_Full.value in subreg_id_int:
        log.print("Calc centroid from subregion id", "Vertebra_Full", verbose=verbose)
        full = Location.Vertebra_Full.value
        vert_arr = vert_msk.get_seg_array()
        if _is_not_yet_computed((full,), extend_to, full):
            arr = vert_arr.copy()
            arr[arr >= v_name2idx["Cocc"]] = 0
            # arr[arr >= Location.Vertebra_Corpus.value] = 0
            # arr[arr != 0] = full
            extend_to = calc_centroids(
                vert_msk.set_array(arr),
                decimals=decimals,
                subreg_id=full,
                extend_to=extend_to,
                inplace=True,
            )
        subreg_id_int.remove(full)
    # Step 4 IVD / Endplates Superior / Endplate Inferior
    print("step 4", subreg_id_int) if _print_phases else None
    mapping_vert = {
        Location.Vertebra_Disc.value: 100,
        Location.Vertebral_Body_Endplate_Superior.value: 200,
        Location.Vertebral_Body_Endplate_Inferior.value: 300,
    }
    for loc, v in mapping_vert.items():
        if loc in subreg_id_int:
            log.print("Calc centroid from subregion id", Location(loc).name, verbose=verbose)
            vert_arr = vert_msk.get_seg_array()
            subreg_arr = subreg_msk.get_seg_array()
            # IVD / Endplates Superior / Endplate Inferior
            vert_arr[subreg_arr != loc] = 0
            # remove a off set of 100/200/300 and remove other that are not of interest
            vert_arr[vert_arr >= v + 100] = 0
            vert_arr[vert_arr < v] = 0
            vert_arr[vert_arr != 0] -= v
            extend_to = calc_centroids(
                vert_msk.set_array(vert_arr),
                decimals=decimals,
                subreg_id=v,
                extend_to=extend_to,
                inplace=True,
            )
            subreg_id_int.remove(loc)
    # Step 5 call non_centroid_pois
    # Prepare mask to binary mask
    print("step 5", subreg_id_int) if _print_phases else None
    vert_arr = vert_msk.get_seg_array()
    subreg_arr = subreg_msk.get_seg_array()
    assert subreg_msk.shape == vert_arr.shape, "Shape miss-match" + str(subreg_msk.shape) + str(vert_arr.shape)
    vert_arr[subreg_arr >= 100] = 0
    subreg_arr[subreg_arr >= 100] = 0

    if extend_to is None:
        extend_to = POI({}, **vert_msk._extract_affine(), format=FORMAT_POI)
    if len(subreg_id_int) != 0:
        # print("step 6", subreg_id_int)
        compute_non_centroid_pois(
            extend_to,
            int2loc(list(subreg_id_int)),
            vert_msk,
            subreg_msk,
            _vert_ids=_vert_ids,
            log=log,
        )
    extend_to.apply_crop_reverse(crop, org_shape, inplace=True)
    return extend_to


def calc_centroids_from_two_masks(
    lvl_1_msk: Image_Reference,
    lvl_2_msk: Image_Reference,
    decimals: int = 3,
    limit_ids_of_lvl_2: int | Sequence[int] | None = None,
    verbose: bool = True,
    extend_to: POI | None = None,
):
    """
    Calculates the centroids of two masks, representing a hierarchical relationship.

    Args:
        lvl_1_msk (Image_Reference): An image reference representing the higher-level mask (instance).
        lvl_2_msk (Image_Reference): An image reference representing the lower-level mask (subregion).
        decimals (int, optional): Number of decimal places to round the output coordinates to. Defaults to 3.
        limit_ids_of_lvl_2 (int | Sequence[int] | None, optional): The ID(s) of the lower-level mask to calculate centroids for. Defaults to None.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
        extend_to (POI | None, optional): An existing POI object to extend with the new centroid values. Defaults to None.

    Returns:
        POI: A POI object containing the calculated centroid coordinates.
    """
    vert_msk = to_nii(lvl_1_msk, seg=True)
    subreg_msk = to_nii(lvl_2_msk, seg=True)
    org_shape = subreg_msk.shape
    # crop to mask to speed up the segmentation
    crop = vert_msk.compute_crop()
    crop = subreg_msk.compute_crop(maximum_size=crop)
    # crop = (slice(0, subreg_msk.shape[0]), slice(0, subreg_msk.shape[1]), slice(0, subreg_msk.shape[2]))

    vert_msk = vert_msk.apply_crop(crop)
    subreg_msk = subreg_msk.apply_crop(crop)
    _vert_ids = vert_msk.unique()
    vert_msk.assert_affine(subreg_msk)
    if isinstance(limit_ids_of_lvl_2, int):
        limit_ids_of_lvl_2 = [limit_ids_of_lvl_2]
    elif limit_ids_of_lvl_2 is None:
        limit_ids_of_lvl_2 = subreg_msk.unique()
    # Recursive call for multiple subregion ids
    if extend_to is None:
        poi = POI({}, **vert_msk._extract_affine(), format=FORMAT_POI)
    else:
        poi = extend_to.apply_crop(crop)
        vert_msk.assert_affine(poi)
    loop = limit_ids_of_lvl_2
    if verbose:
        from tqdm import tqdm

        loop = tqdm(loop, total=len(loop), desc="calc centroids from two masks")
    for subreg_id in loop:
        subreg_id = loc2int(subreg_id)  # noqa: PLW2901
        if not all((v, subreg_id) in poi for v in _vert_ids):
            exclusive_mask = vert_msk.get_seg_array()
            exclusive_mask[subreg_msk.get_seg_array() != subreg_id] = 0
            # check if the point exists if extend_to is used
            is_not_yet_computed = _is_not_yet_computed(np.unique(exclusive_mask), extend_to, subreg_id)  # type: ignore
            if is_not_yet_computed:
                # calc poi for individual subreg
                nii = NII((exclusive_mask, vert_msk.affine, vert_msk.header), True)
                poi = calc_centroids(nii, decimals=decimals, subreg_id=subreg_id, extend_to=poi)
            else:
                continue
    # reverse crop
    poi.apply_crop_reverse(crop, org_shape, inplace=True)
    return poi


def _is_not_yet_computed(ids_in_arr: Sequence[int], extend_to: POI | None, subreg_id: int):
    is_not_yet_computed = True
    if extend_to is not None and subreg_id in extend_to.centroids.keys_subregion():
        is_not_yet_computed = False
        for reg in ids_in_arr:  # type: ignore
            if reg == 0:
                continue
            if (reg, subreg_id) in extend_to:
                continue
            is_not_yet_computed = True
            break
    return is_not_yet_computed


def calc_centroids(
    msk: Image_Reference,
    decimals=3,
    vert_id=-1,
    subreg_id: int | Location = 50,
    extend_to: POI | None = None,
    inplace: bool = False,
) -> POI:
    """
    Calculates the centroid coordinates of each region in the given mask image.

    Args:
        msk (Image_Reference): An `Image_Reference` object representing the input mask image.
        decimals (int, optional): An optional integer specifying the number of decimal places to round the centroid coordinates to (default is 3).
        vert_id (int, optional): An optional integer specifying the fixed vertical dimension for the centroids (default is -1).
        subreg_id (int, optional): An optional integer specifying the fixed subregion dimension for the centroids (default is 50).
        extend_to (POI, optional): An optional `POI` object to add the calculated centroids to (default is None).

    Returns:
        POI: A `POI` object containing the calculated centroid coordinates.

    Raises:
        AssertionError: If the `extend_to` object has a different orientation, location, or zoom than the input mask.

    Notes:
        - The function calculates the centroid coordinates of each region in the mask image.
        - The centroid coordinates are rounded to the specified number of decimal places.
        - The fixed dimensions for the centroids can be specified using `vert_id` and `subreg_id`.
        - If `extend_to` is provided, the calculated centroids will be added to the existing object and the updated object will be returned.
        - The region label is assumed to be an integer.
        - NaN values in the binary mask are ignored.
    """
    if isinstance(subreg_id, Location):
        subreg_id = subreg_id.value
    assert vert_id == -1 or subreg_id == -1, "first or second dimension must be fixed."
    msk_nii = to_nii(msk, seg=True)
    msk_data = msk_nii.get_seg_array()
    axc: AX_CODES = nio.aff2axcodes(msk_nii.affine)  # type: ignore
    if extend_to is None:
        ctd_list = POI_Descriptor()
    else:
        if not inplace:
            extend_to = extend_to.copy()
        ctd_list = extend_to.centroids
        extend_to.assert_affine(
            msk_nii,
            shape_tolerance=0.5,
            origin_tolerance=0.5,
        )
        # assert extend_to.orientation == axc, (extend_to.orientation, axc)
    for i in msk_nii.unique():
        msk_temp = np.zeros(msk_data.shape, dtype=bool)
        msk_temp[msk_data == i] = True
        ctr_mass: Sequence[float] = center_of_mass(msk_temp)  # type: ignore
        if subreg_id == -1:
            ctd_list[vert_id, int(i)] = tuple(round(x, decimals) for x in ctr_mass)
        else:
            ctd_list[int(i), subreg_id] = tuple(round(x, decimals) for x in ctr_mass)
    return POI(ctd_list, orientation=axc, **msk_nii._extract_affine(rm_key=["orientation"]))
