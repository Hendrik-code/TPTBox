from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import Self

from TPTBox.core.np_utils import np_count_nonzero
from TPTBox.core.vert_constants import COORDINATE
from TPTBox.logger import Log_Type

from .vert_constants import (
    AFFINE,
    AX_CODES,
    DIRECTIONS,
    ORIGIN,
    ROTATION,
    ROUNDING_LVL,
    SHAPE,
    ZOOMS,
    _plane_dict,
    _same_direction,
    log,
    logging,
)

if TYPE_CHECKING:

    class Grid_Proxy:
        affine: AFFINE
        rotation: ROTATION
        zoom: ZOOMS
        spacing: ZOOMS
        origin: ORIGIN
        shape: SHAPE
        orientation: AX_CODES

else:

    class Grid_Proxy:
        pass


class Has_Grid(Grid_Proxy):
    """Parent class for methods that are shared by POI and NII"""

    info: dict

    def to_gird(self) -> Grid:
        return Grid(**self._extract_affine())

    @property
    def shape_int(self):
        assert self.shape is not None, "need shape information"
        return tuple(np.rint(list(self.shape)).astype(int).tolist())

    @property
    def spacing(self):
        return self.zoom

    @spacing.setter
    def spacing(self, value: ZOOMS):
        self.zoom = value

    def __str__(self) -> str:
        try:
            origin = tuple(np.around(self.origin, decimals=2).tolist())
        except Exception:
            origin = self.origin
        try:
            zoom = "(" + ",".join([f"{a:.2f}" for a in self.zoom]) + ")"
        except Exception as e:
            print(e)
            zoom = self.zoom

        return f"shape={self.shape_int},spacing={zoom}, origin={origin}, ori={self.orientation}"  # type: ignore

    @property
    def affine(self):
        assert self.zoom is not None, "Attribute 'zoom' must be set before calling affine."
        assert self.rotation is not None, "Attribute 'rotation' must be set before calling affine."
        assert self.origin is not None, "Attribute 'origin' must be set before calling affine."
        aff = np.eye(4)
        aff[:3, :3] = self.rotation @ np.diag(self.zoom)
        aff[:3, 3] = self.origin
        return np.round(aff, ROUNDING_LVL)

    @affine.setter
    def affine(self, affine: np.ndarray):
        rotation_zoom = affine[:3, :3]
        zoom = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
        rotation_zoom = affine[:3, :3]
        rotation = rotation_zoom / zoom
        origin = affine[:3, 3]
        self.zoom = zoom
        self.rotation = rotation
        self.origin = origin.tolist()

    def _extract_affine(self: Has_Grid, rm_key=(), **args):
        shape = self.shape_int if self.shape is not None else None
        out = {
            "zoom": self.spacing,
            "origin": self.origin,
            "shape": shape,
            "rotation": self.rotation,
            "orientation": self.orientation,
            **args,
        }
        for k in rm_key:
            out.pop(k)
        return out

    def change_affine(
        self,
        translation=None,
        rotation_degrees=None,
        scaling=None,
        degrees=True,
        inplace=False,
    ):
        """
        Apply a transformation (scaling, rotation, translation) to the affine matrix.

        Assumptions
        -----------
        - `self.affine` is a square homogeneous affine matrix of shape (n, n),
        where the spatial dimensionality is n-1 (typically n=4 for 3D).
        - The affine follows the convention:
            x_world = A @ x_homogeneous
        - Transformations are applied in the following order (right-multiplied):
            1. Scaling
            2. Rotation
            3. Translation
        i.e. the final update is:
            self.affine = (T @ R @ S) @ self.affine
        - Rotation is specified as Euler angles in the "xyz" convention
        (pitch, yaw, roll) using scipy.spatial.transform.Rotation.
        - Translation is specified in world units (e.g. mm) in (x, y, z)
        corresponding to the affine axes.
        - Scaling is applied along the affine axes, not object-local axes.
        - If `inplace=False`, a copy of the object is returned.
        If `inplace=True`, the object is modified in place.

        Parameters
        ----------
        translation : (n-1,) array-like, optional
            Translation vector in world coordinates.
        rotation_degrees : (n-1,) array-like, optional
            Euler angles (x, y, z) in degrees by default.
        scaling : (n-1,) array-like, optional
            Scaling factors along each axis.
        degrees : bool, default=True
            Whether rotation angles are given in degrees.
        inplace : bool, default=False
            Whether to modify the object in place.

        Returns
        -------
        self or copy of self
            Object with updated affine.
        """
        # warnings.warn("change_affine is untested", stacklevel=2)
        n = self.affine.shape[0]
        transform = np.eye(n)

        # Scaling
        if scaling is not None:
            assert len(scaling) == n - 1, f"Scaling must be a {n - 1}-element array-like."
            S = np.diag([*list(scaling), 1])
            transform = S @ transform

        # Rotation
        if rotation_degrees is not None:
            assert len(rotation_degrees) == n - 1, f"Rotation must be a {n - 1}-element array-like."
            rot = Rotation.from_euler("xyz", rotation_degrees, degrees=degrees).as_matrix()
            R_mat = np.eye(n)
            R_mat[: n - 1, : n - 1] = rot
            transform = R_mat @ transform

        # Translation
        if translation is not None:
            T = np.eye(n)
            T[: n - 1, n - 1] = translation
            transform = T @ transform
        if not inplace:
            self = self.copy()  # noqa: PLW0642
        # Update the affine
        self.affine = transform @ self.affine
        return self

    def change_affine_(self, translation=None, rotation_degrees=None, scaling=None, degrees=True):
        return self.change_affine(
            translation=translation,
            rotation_degrees=rotation_degrees,
            scaling=scaling,
            degrees=degrees,
            inplace=True,
        )

    def copy(self) -> Self:
        raise NotImplementedError(
            "The copy method must be implemented in the subclass. It should return a new instance of the same type with the same attributes."
        )

    def assert_affine(
        self,
        other: Self | Has_Grid | None = None,
        ignore_missing_values: bool = False,
        affine: AFFINE | None = None,
        zoom: ZOOMS | None = None,
        orientation: AX_CODES | None = None,
        rotation: ROTATION | None = None,
        origin: ORIGIN | None = None,
        shape: SHAPE | None = None,
        shape_tolerance: float = 0.0,
        origin_tolerance: float = 0.01,
        error_tolerance: float = 1e-4,
        raise_error: bool = True,
        verbose: logging = False,
        text: str = "",
    ):
        """Checks if the different metadata is equal to some comparison entries

        Args:
            other (Has_Grid | None, optional): If set, will assert each entry of that object instead. Defaults to None.
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
                found_errors.append(f"spacing mismatch {self.zoom}, {zoom}")
            else:
                zms_diff = (self.zoom[i] - zoom[i] for i in range(3))
                zms_match = np.all([abs(a) <= error_tolerance for a in zms_diff])
                found_errors.append(f"spacing mismatch {self.zoom}, {zoom}") if not zms_match else None
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
                shape_diff = (float(self.shape_int[i]) - float(shape[i]) for i in range(3))
                shape_match = np.all([abs(a) <= shape_tolerance for a in shape_diff])
                found_errors.append(f"shape mismatch {self.shape}, {shape}") if not shape_match else None

        # Print errors
        for err in found_errors:
            log.print(err, ltype=Log_Type.FAIL, verbose=verbose)
        # Final conclusion and possible raising of AssertionError
        has_errors = len(found_errors) > 0
        if raise_error and has_errors:
            raise AssertionError(f"{text}; assert_affine failed with {found_errors}")

        return not has_errors

    def get_plane(self, res_threshold: float | None = 1) -> str:
        """Determines the orientation plane of the NIfTI image along the x, y, or z-axis.

        Returns:
            str: The orientation plane of the image, which can be one of the following:
                - 'ax': Axial plane (along the z-axis).
                - 'cor': Coronal plane (along the y-axis).
                - 'sag': Sagittal plane (along the x-axis).
                - 'iso': Isotropic plane (if the image has equal zoom values along all axes).
        Examples:
            >>> nii = NII(nib.load("my_image.nii.gz"))
            >>> nii.get_plane()
            'ax'
        """
        # plane_dict = {"S": "ax", "I": "ax", "L": "sag", "R": "sag", "A": "cor", "P": "cor"}
        axc = np.array(nio.aff2axcodes(self.affine))
        zoom = self.zoom if res_threshold is None else tuple(max(i, res_threshold) for i in self.zoom)
        zms = np.around(zoom, 1)
        ix_max = np.array(zms == np.amax(zms))
        num_max = np_count_nonzero(ix_max)
        if num_max == 2:
            plane = _plane_dict[axc[~ix_max][0]]
        elif num_max == 1:
            plane = _plane_dict[axc[ix_max][0]]
        else:
            plane = "iso"
        return plane

    def get_axis(self, direction: DIRECTIONS = "S"):
        if direction not in self.orientation:
            direction = _same_direction[direction]
        return self.orientation.index(direction)

    def make_empty_POI(self, points: dict | None = None):
        from TPTBox import POI

        p = {} if points is None else points
        args = {}
        if isinstance(self, POI):
            args["level_one_info"] = self.level_one_info
            args["level_two_info"] = self.level_two_info

        return POI(
            p,
            orientation=self.orientation,
            zoom=self.zoom,
            shape=self.shape,
            rotation=self.rotation,
            origin=self.origin,
            **args,
        )

    def make_empty_nii(self, seg=False, _arr=None):
        from TPTBox import NII

        if _arr is None:
            _arr = np.zeros(self.shape_int)
        else:
            assert _arr.shape == self.shape_int, (
                f"Expected the correct shape for generating a image from Grid; Got {_arr.shape}, expected {self.shape_int}"
            )
        nii = nib.Nifti1Image(_arr, affine=self.affine)
        return NII(nii, seg=seg)

    def make_nii(self, arr: np.ndarray | None = None, seg=False):
        """Make a nii with the same grid as object. Shape must fit the Grid.

        Args:
            arr  np.ndarray: array. Defaults to None.
            seg (bool, optional): Is it a segmentation. Defaults to False.

        Returns:
            NII
        """
        if arr is None:
            arr = np.zeros(self.shape_int)
        return self.make_empty_nii(_arr=arr, seg=seg)

    def global_to_local(self, x: COORDINATE):
        a = self.rotation.T @ (np.array(x) - self.origin) / np.array(self.zoom)
        return tuple(round(float(v), 7) for v in a)

    def local_to_global(self, x: COORDINATE):
        a = self.rotation @ (np.array(x) * np.array(self.zoom)) + self.origin
        return tuple(round(float(v), 7) for v in a)

    def to_deepali_grid(self, align_corners: bool = True):
        try:
            from deepali.core import Grid
        except Exception:
            log.print_error()
            log.on_fail("run 'pip install hf-deepali' to install deepali")
            raise
        try:
            dim = np.asarray(self.header["dim"])
        except Exception:
            dim = [3]
        d = min(int(dim[0]), 3)
        size = self.shape  # dim[1 : D + 1]
        spacing = np.asarray(self.zoom)
        affine = np.asarray(self.affine).copy()
        origin = affine[:d, 3]
        direction = np.divide(affine[:d, :d], spacing)
        # Convert to ITK LPS convention
        origin[:2] *= -1
        direction[:2] *= -1
        # Replace small values and -0 by 0
        epsilon = sys.float_info.epsilon
        origin[np.abs(origin) < epsilon] = 0
        direction[np.abs(direction) < epsilon] = 0
        # Add leading channel dimension
        grid = Grid(size=size, origin=origin, spacing=spacing, direction=direction)  # type: ignore
        grid = grid.align_corners_(align_corners)
        return grid

    @classmethod
    def from_deepali_grid(cls, grid):
        try:
            from deepali.core import Grid as dp_Grid
        except Exception:
            log.print_error()
            log.on_fail("run 'pip install hf-deepali' to install deepali")
            raise
        grid_: dp_Grid = grid
        size = grid_.size()
        spacing = grid_.spacing().cpu().numpy()
        origin = grid_.origin().cpu().numpy()
        direction = grid_.direction().cpu().numpy()
        # Convert to ITK LPS convention
        origin[:2] *= -1
        direction[:2] *= -1
        # Replace small values and -0 by 0
        epsilon = sys.float_info.epsilon
        origin[np.abs(origin) < epsilon] = 0
        direction[np.abs(direction) < epsilon] = 0
        grid = Grid(shape=size, origin=origin, spacing=spacing, rotation=direction)  # type: ignore

        return grid

    def get_num_dims(self):
        return len(self.shape)


@dataclass
class Grid(Has_Grid):
    def __init__(self, **qargs) -> None:
        super().__init__()
        for k, v in qargs.items():
            if k == "spacing":
                k = "zoom"  # noqa: PLW2901
            if k == "direction":
                k = "rotation"  # noqa: PLW2901
            if k == "rotation":
                v = np.array(v)  # noqa: PLW2901
                if len(v.shape) == 1:
                    s = int(np.sqrt(v.shape[0]))
                    v = v.reshape(s, s)  # noqa: PLW2901
            setattr(self, k, v)

        ort = nio.io_orientation(self.affine)
        self.orientation = nio.ornt2axcodes(ort)  # type: ignore
