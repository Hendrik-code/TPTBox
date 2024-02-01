import functools
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, TypeGuard, TypeVar, Union

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
from scipy.ndimage import center_of_mass
from typing_extensions import Self

from . import bids_files
from .nii_wrapper import NII, Image_Reference, to_nii, to_nii_optional
from .poi_abstract import ROUNDING_LVL, Abstract_POI, POI_Descriptor, Sentinel
from .vert_constants import (
    LABEL_MAX,
    Ax_Codes,
    Coordinate,
    Location,
    POI_Dict,
    Rotation,
    Triple,
    Zooms,
    conversion_poi,
    conversion_poi2text,
    log,
    log_file,
    logging,
    v_idx2name,
    v_name2idx,
)


### LAGACY DEFINITONS ###
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
ctd_info_blacklist = ["zoom", "shape", "direction", "format", "rotation", "origin"]  # "location"


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
        >>> poi_obj = POI(centroids=poi_data, orientation=("R", "A", "S"), zoom=(1.0, 1.0, 1.0),
        ...               shape=(256, 256, 100))

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

    orientation: Ax_Codes = ("R", "A", "S")
    zoom: None | Zooms = field(init=True, default=None)  # type: ignore
    shape: Triple | None = field(default=None, repr=True, compare=False)
    rotation: Rotation | None = field(default=None, repr=False, compare=False)
    origin: Coordinate | None = None
    # internal
    _zoom: None | Zooms = field(init=False, default=None, repr=False, compare=False)

    @property
    def shape_int(self):
        assert self.shape is not None, "need shape information"
        return tuple(np.rint(list(self.shape)).astype(int))

    @property
    def is_global(self):
        return False

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

    def _extract_affine(self):
        return {"zoom": self.zoom, "origin": self.origin, "shape": self.shape, "rotation": self.rotation, "orientation": self.orientation}

    def copy(
        self,
        centroids: POI_Dict | POI_Descriptor | None = None,
        orientation: Ax_Codes | None = None,
        zoom: Zooms | None | Sentinel = Sentinel(),  # noqa: B008
        shape: tuple[float, float, float] | tuple[float, ...] | None | Sentinel = Sentinel(),  # noqa: B008
        rotation: Rotation | None | Sentinel = Sentinel(),  # noqa: B008
        origin: Coordinate | None | Sentinel = Sentinel(),  # noqa: B008
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
            info=self.info,
            format=self.format,
        )

    def local_to_global(self, x: Coordinate) -> Coordinate:
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

    def global_to_local(self, x: Coordinate) -> Coordinate:
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
        warnings.warn("crop_centroids id deprecated use apply_crop instead", DeprecationWarning, stacklevel=4)
        return self.apply_crop(**qargs)

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
                return float(x - o_shift[0].start), float(y - o_shift[1].start), float(z - o_shift[2].start)

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

                shape: Triple | None = tuple(int(map_v(o_shift[i], i) - o_shift[i].start) for i in range(3))  # type: ignore
            if self.origin is not None:
                origin = self.local_to_global(tuple(float(y.start) for y in o_shift))  # type: ignore
                print(origin)
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
        out = self.copy(centroids=poi_out.centroids, shape=shape, rotation=self.rotation, origin=origin)
        return out

    def crop_centroids_(self, o_shift: tuple[slice, slice, slice]):
        import warnings

        warnings.warn(
            "crop_centroids_ id deprecated use apply_crop instead", DeprecationWarning, stacklevel=4
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

    def reorient(self, axcodes_to: Ax_Codes = ("P", "I", "R"), decimals=ROUNDING_LVL, verbose: logging = False, inplace=False, _shape=None):
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
                ltype=log_file.Log_Type.WARNING,
            )
            return self

        ornt_fr = nio.axcodes2ornt(self.orientation)  # original centroid orientation
        ornt_to = nio.axcodes2ornt(axcodes_to)

        if (ornt_fr == ornt_to).all():
            log.print("ctd is already rotated to image with ", axcodes_to, verbose=verbose)
            return self
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

        log.print("[*] Centroids reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to, verbose=verbose)
        if self.zoom is not None:
            zoom_i = np.array(self.zoom)
            zoom_i[perm] = zoom_i.copy()
            zoom: Zooms | None = tuple(zoom_i)
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
        return self.copy(orientation=axcodes_to, centroids=points, zoom=zoom, shape=shape, origin=origin, rotation=rotation)

    def reorient_(self, axcodes_to: Ax_Codes = ("P", "I", "R"), decimals=3, verbose: logging = False, _shape=None):
        return self.reorient(axcodes_to, decimals=decimals, verbose=verbose, inplace=True, _shape=_shape)

    def reorient_centroids_to(self, img: Image_Reference, decimals=ROUNDING_LVL, verbose: logging = False, inplace=False) -> Self:
        # reorient centroids to image orientation
        if not isinstance(img, NII):
            img = to_nii(img)
        axcodes_to: Ax_Codes = nio.aff2axcodes(img.affine)  # type: ignore
        return self.reorient(axcodes_to, decimals=decimals, verbose=verbose, inplace=inplace, _shape=img.shape)

    def reorient_centroids_to_(self, img: Image_Reference, decimals=ROUNDING_LVL, verbose: logging = False) -> Self:
        return self.reorient_centroids_to(img, decimals=decimals, verbose=verbose, inplace=True)

    def rescale(self, voxel_spacing: Zooms = (1, 1, 1), decimals=ROUNDING_LVL, verbose: logging = True, inplace=False) -> Self:
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
        log.print("Rescaled centroid coordinates to spacing (x, y, z) =", voxel_spacing, "mm", verbose=verbose)
        if shp is not None:
            shp = tuple(float(v) for v in shp)

        if inplace:
            self.centroids = points
            self.zoom = voxel_spacing
            self.shape = shp
            return self
        return self.copy(centroids=points, zoom=voxel_spacing, shape=shp)

    def rescale_(self, voxel_spacing: Zooms = (1, 1, 1), decimals=3, verbose: logging = False) -> Self:
        return self.rescale(voxel_spacing=voxel_spacing, decimals=decimals, verbose=verbose, inplace=True)

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
        import TPTBox.core.poi_global as pg

        return pg.POI_Global(self)

    def resample_from_to(self, ref: NII | Self):
        return self.to_global().to_other(ref)

    def save(
        self, out_path: Path | str, make_parents=False, additional_info: dict | None = None, verbose: logging = True, save_hint=2
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
            log.print(
                "Centroids empty, not saved:",
                out_path,
                ltype=log_file.Log_Type.FAIL,
                verbose=verbose,
            )
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
        log.print(
            "Centroids saved:",
            out_path,
            print_add,
            ltype=log_file.Log_Type.SAVE,
            verbose=verbose,
        )

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

    def make_empty_nii(self, seg=False):
        nii = nib.Nifti1Image(np.zeros(self.shape_int), affine=self.affine)
        return NII(nii, seg=seg)

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

    def assert_affine(self, nii: NII | Self):
        return
        assert self.zoom is not None
        assert nii.zoom is not None
        assert np.isclose(self.zoom, nii.zoom, rtol=0.001).all()
        assert self.orientation == nii.orientation, (self.orientation, nii.orientation)
        s = self.shape_int
        if isinstance(nii, NII):
            assert s is not None
            s = tuple(round(x, 0) for x in s)
        atol = 10  # 0.01  # 1e-5
        assert s == nii.shape, (s, nii.shape, self, str(nii))
        assert self.rotation is not None
        assert nii.rotation is not None
        assert self.rotation is not None
        assert nii.rotation is not None
        assert np.isclose(self.rotation, nii.rotation, atol=atol).all()
        assert self.origin is not None
        assert nii.origin is not None
        assert nii.origin is not None
        assert self.origin is not None

        aff_equ = np.isclose(self.affine, nii.affine, atol=atol).all()
        assert aff_equ, f"{np.round(self.affine,2)}\n {np.round(nii.affine,2)}"
        assert np.isclose(self.origin, nii.origin, atol=atol).all(), (self.origin, nii.origin)


class VertebraCentroids(POI):
    def __init__(
        self,
        _centroids,
        _orientation: Ax_Codes,
        _shape: Triple | None,
        zoom=None,
        **qargs,
    ):
        assert isinstance(_orientation, (tuple, list)), "_orientation is not a list or a tuple. Did you swap _centroids and _orientation"
        super().__init__(orientation=_orientation, centroids=_centroids, shape=_shape, zoom=zoom, **qargs)
        self.sort()
        self.remove_centroid(*[(a, b) for a, b, c in self.items() if a >= 40 or b != 50], inplace=True)

    @classmethod
    def from_pois(cls, ctd: POI):
        a = ctd.centroids.copy()
        a = {(k, k2): v for k, k2, v in a.items() if k <= 30}  # filter all non Vertebra_Centroids
        return cls(
            a,
            ctd.orientation,
            _shape=ctd.shape,
            zoom=ctd.zoom,
            rotation=ctd.rotation,
            origin=ctd.origin,
            format=ctd.format,
            info=ctd.info,
        )

    def get_sorted_regions(self, subreg_id=50):
        """

        Args:
            index:

        Returns:
            the <index>-th vertebra in this Centroid list (if index == -1, returns the last one)
        """
        self.sort()
        centroid_keys = [k for k, k2 in list(self.keys()) if subreg_id == k2]
        return centroid_keys


######## Saving #######
def _is_Point3D(obj) -> TypeGuard[_Point3D]:
    return "label" in obj and "X" in obj and "Y" in obj and "Z" in obj


FORMAT_DOCKER = 0
FORMAT_GRUBER = 1
FORMAT_POI = 2
FORMAT_OLD_POI = 10
format_key = {FORMAT_DOCKER: "docker", FORMAT_GRUBER: "guber", FORMAT_POI: "POI"}
format_key2value = {value: key for key, value in format_key.items()}


def _poi_to_dict_list(ctd: POI, additional_info: dict | None, save_hint=0, verbose: logging = False):
    ori: _Orientation = {"direction": ctd.orientation}
    print_out = ""
    # if hasattr(ctd, "location") and ctd.location != Location.Unknown:
    #    ori["location"] = ctd.location_str  # type: ignore
    if ctd.zoom is not None:
        ori["zoom"] = ctd.zoom  # type: ignore
    if ctd.origin is not None:
        ori["origin"] = ctd.origin  # type: ignore
    if ctd.rotation is not None:
        ori["rotation"] = ctd.rotation  # type: ignore
    if ctd.shape is not None:
        ori["shape"] = ctd.shape  # type: ignore
    if save_hint in format_key:
        ori["format"] = format_key[save_hint]  # type: ignore
        print_out = "in format " + format_key[save_hint]

    if additional_info is not None:
        for k, v in additional_info.items():
            if k not in ori:
                ori[k] = v

    for k, v in ctd.info.items():
        if k not in ori:
            ori[k] = v

    dict_list: list[Union[_Orientation, _Point3D | dict]] = [ori]

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
            v_name = v_idx2name[vert_id] if vert_id in v_idx2name else str(vert_id)
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


def load_poi(ctd_path: POI_Reference, verbose=True) -> POI:
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
        return calc_centroids_from_subreg_vert(vert, subreg, subreg_id=ids)
    else:
        assert False, f"{type(ctd_path)}\n{ctd_path}"
    ### format_POI_old has no META header
    if "direction" not in dict_list[0] and "vert_label" in dict_list[0]:
        return _load_format_POI_old(dict_list)  # This file if used in the old POI-pipeline and is deprecated

    assert "direction" in dict_list[0], f'File format error: first index must be a "Direction" but got {dict_list[0]}'
    axcode: Ax_Codes = tuple(dict_list[0]["direction"])  # type: ignore
    zoom: Zooms = dict_list[0].get("zoom", None)  # type: ignore
    shape = dict_list[0].get("shape", None)  # type: ignore
    shape = tuple(shape) if shape is not None else None
    format_ = dict_list[0].get("format", None)
    origin = dict_list[0].get("origin", None)
    origin = tuple(origin) if origin is not None else None
    rotation = dict_list[0].get("rotation", None)
    info = {}
    for k, v in dict_list[0].items():
        if k not in ctd_info_blacklist:
            info[k] = v

    format_ = format_key2value[format_] if format_ is not None else None
    centroids = POI_Descriptor()
    if format_ in (FORMAT_DOCKER, FORMAT_GRUBER) or format_ is None:
        _load_docker_centroids(dict_list, centroids, format_)
    elif format_ == FORMAT_POI:
        _load_POI_centroids(dict_list, centroids)
    else:
        raise NotImplementedError(format_)
    return POI(centroids, orientation=axcode, zoom=zoom, shape=shape, format=format_, info=info, origin=origin, rotation=rotation)


def _load_docker_centroids(dict_list, centroids: POI_Descriptor, format):
    for d in dict_list[1:]:
        assert "direction" not in d, f'File format error: only first index can be a "direction" but got {dict_list[0]}'
        if "nan" in str(d):  # skipping NaN centroids
            continue
        elif _is_Point3D(d):
            try:
                a = int(d["label"])
                centroids[a % LABEL_MAX, a // LABEL_MAX] = (d["X"], d["Y"], d["Z"])
            except Exception:
                try:
                    number, subreg = str(d["label"]).split("_", maxsplit=1)
                    number = number.replace("TH", "T").replace("SA", "S1")
                    vert_id = v_name2idx[number]
                    subreg_id = conversion_poi[subreg]
                    centroids[vert_id, subreg_id] = (d["X"], d["Y"], d["Z"])
                except:
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
    return POI(
        centroids,
        orientation=("R", "P", "I"),
        zoom=(1, 1, 1),
        shape=None,
        format=FORMAT_OLD_POI,
    )


def _to_int(vert_id):
    try:
        return int(vert_id)
    except Exception:
        return v_name2idx[vert_id]


def _load_POI_centroids(dict_list, centroids: POI_Descriptor):
    assert len(dict_list) == 2
    d: dict[int | str, dict[int | str, tuple[float, float, float]]] = dict_list[1]
    for vert_id, v in d.items():
        vert_id = _to_int(vert_id)
        for sub_id, t in v.items():
            sub_id = _to_int(sub_id)
            centroids[vert_id, sub_id] = tuple(t)


def loc2int(i: int | Location):
    if isinstance(i, int):
        return i
    return i.value


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


def calc_centroids_labeled_buffered(
    msk_reference: Image_Reference,
    subreg_reference: Image_Reference | None,
    out_path: Path | str,
    subreg_id: int | Location | Sequence[int | Location] = 50,
    verbose=True,
    override=False,
    decimals=3,
    # additional_folder=False,
    check_every_point=True,
    use_vertebra_special_action=True,
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
    if sub_nii is None or not check_every_point:
        if out_path.exists():
            return POI.load(out_path)
    if sub_nii is not None:
        ctd = calc_centroids_from_subreg_vert(
            msk_nii,
            sub_nii,
            buffer_file=out_path,
            decimals=decimals,
            subreg_id=subreg_id,
            verbose=verbose,
            use_vertebra_special_action=use_vertebra_special_action,
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


@_buffer_it
def calc_centroids_from_subreg_vert(
    vert_msk: Image_Reference,
    subreg: Image_Reference,
    *,
    buffer_file: str | Path | None = None,  # used by wrapper
    save_buffer_file=False,  # used by wrapper
    decimals=2,
    subreg_id: int | Location | Sequence[int | Location] | Sequence[Location] | Sequence[int] = 50,
    axcodes_to: Ax_Codes | None = None,
    verbose: logging = True,
    extend_to: POI | None = None,
    use_vertebra_special_action=True,
    _vert_ids=None,
) -> POI:
    """
    Calculates the centroids of a subregion within a vertebral mask.

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

    vert_msk = to_nii(vert_msk, seg=True)
    subreg_msk = to_nii(subreg, seg=True)
    assert vert_msk.origin == subreg_msk.origin, (vert_msk.origin, subreg_msk.origin)
    if _vert_ids is None:
        _vert_ids = vert_msk.unique()
    log.print("Calc centroids from subregion id", int2loc(subreg_id), vert_msk.shape, verbose=verbose)

    if axcodes_to is not None:
        # Like: ("P","I","R")
        vert_msk = vert_msk.reorient(verbose=verbose, axcodes_to=axcodes_to, inplace=False)
        subreg_msk = subreg_msk.reorient(verbose=verbose, axcodes_to=axcodes_to, inplace=False)
    assert vert_msk.orientation == subreg_msk.orientation
    # Recursive call for multiple subregion ids
    if isinstance(subreg_id, Sequence):
        if extend_to is None:
            poi = POI({}, **vert_msk._extract_affine(), format=FORMAT_POI)
        else:
            extend_to.format = FORMAT_POI
            poi = extend_to
            assert poi.orientation == vert_msk.orientation, (poi.orientation, vert_msk.orientation)

        print("list") if verbose else None

        for idx in subreg_id:
            idx = loc2int(idx)
            if not all((v, idx) in poi for v in _vert_ids):
                poi = calc_centroids_from_subreg_vert(
                    vert_msk,
                    subreg_msk,
                    buffer_file=None,
                    subreg_id=loc2int(idx),
                    verbose=verbose,
                    extend_to=poi,
                    decimals=decimals,
                    _vert_ids=_vert_ids,
                )
        return poi
    # Prepare mask to binary mask
    vert_arr = vert_msk.get_seg_array()
    subreg_arr = subreg_msk.get_seg_array()
    assert subreg_msk.shape == vert_arr.shape, "Shape miss-match" + str(subreg_msk.shape) + str(vert_arr.shape)
    subreg_id = loc2int(subreg_id)
    if use_vertebra_special_action:
        mapping_vert = {
            Location.Vertebra_Disc.value: 100,
            Location.Vertebral_Body_Endplate_Superior.value: 200,
            Location.Vertebral_Body_Endplate_Inferior.value: 300,
        }
        if subreg_id == Location.Vertebra_Corpus.value:
            subreg_arr[subreg_arr == 49] = Location.Vertebra_Corpus.value
        elif subreg_id == Location.Vertebra_Full.value:
            subreg_arr = vert_arr.copy()
            subreg_arr[subreg_arr >= 26] = 0
            subreg_arr[subreg_arr != 0] = Location.Vertebra_Full.value

        if subreg_id in mapping_vert:  # IVD / Endplates Superior / Endplate Inferior
            vert_arr[subreg_arr != subreg_id] = 0
            vert_arr[vert_arr >= mapping_vert[subreg_id] + 100] = 0
            vert_arr[vert_arr <= mapping_vert[subreg_id]] = 0
            vert_arr[vert_arr != 0] -= mapping_vert[subreg_id]
        else:
            vert_arr[subreg_arr >= 100] = 0
            subreg_arr[subreg_arr >= 100] = 0
            if use_vertebra_special_action and ((subreg_id in range(81, 128)) or subreg_id == Location.Spinal_Canal.value):
                from TPTBox.core.vertebra_pois_non_centroids import compute_non_centroid_pois

                if extend_to is None:
                    extend_to = POI({}, **vert_msk._extract_affine(), format=FORMAT_POI)
                compute_non_centroid_pois(extend_to, int2loc(subreg_id), vert_msk, subreg_msk, _vert_ids=_vert_ids, log=log)
                return extend_to
            elif subreg_id < 40 or subreg_id > 50:
                raise NotImplementedError(
                    f"POI of subreg {subreg_id} - {Location(subreg_id) if subreg_id in Location else 'undefined'} is not a centroid. If you want the general Centroid computation use use_vertebra_special_action = False"
                )
    vert_arr[subreg_arr != subreg_id] = 0
    vert_arr[vert_arr >= 100] = 0

    msk_data = vert_arr
    if extend_to is not None and subreg_id in extend_to.centroids.keys_subregion():
        missing = False
        for reg in np.unique(vert_arr):
            if reg == 0:
                continue
            if (reg, subreg_id) in extend_to:
                continue
            missing = True
            break
        if not missing:
            return extend_to

    nii = nib.nifti1.Nifti1Image(msk_data, vert_msk.affine, vert_msk.header)
    poi = calc_centroids(nii, decimals=decimals, subreg_id=subreg_id, extend_to=extend_to)
    return poi


def calc_centroids(
    msk: Image_Reference, decimals=3, vert_id=-1, subreg_id: int | Location = 50, extend_to: POI | None = None, inplace: bool = False
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
    msk = to_nii(msk, seg=True)
    msk_data = msk.get_seg_array()
    axc: Ax_Codes = nio.aff2axcodes(msk.affine)  # type: ignore
    if extend_to is None:
        ctd_list = POI_Descriptor()
    else:
        ctd_list = extend_to.centroids
        if not inplace:
            ctd_list = ctd_list.copy()
        extend_to.assert_affine(msk)
    for i in msk.unique():
        msk_temp = np.zeros(msk_data.shape, dtype=bool)
        msk_temp[msk_data == i] = True
        ctr_mass: Sequence[float] = center_of_mass(msk_temp)  # type: ignore
        if subreg_id == -1:
            ctd_list[vert_id, int(i)] = tuple(round(x, decimals) for x in ctr_mass)
        else:
            ctd_list[int(i), subreg_id] = tuple(round(x, decimals) for x in ctr_mass)
    return POI(ctd_list, orientation=axc, **msk._extract_affine(rm_key=["orientation"]))
