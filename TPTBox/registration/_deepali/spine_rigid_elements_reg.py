from __future__ import annotations

import os

# pip install hf-deepali
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim
from deepali.core import Axes, PaddingMode, Sampling
from deepali.modules import SampleImage
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree  # type: ignore

from TPTBox import NII, POI, Image_Reference, Location, calc_poi_from_subreg_vert, to_nii
from TPTBox.core.compat import zip_strict
from TPTBox.core.internal.deep_learning_utils import DEVICES, get_device
from TPTBox.core.nii_poi_abstract import Has_Grid
from TPTBox.registration._ridged_intensity.affine_deepali import Rigid_Registration_with_Tether
from TPTBox.registration._ridged_points import Point_Registration


def not_exist_or_is_younger_than(path: Path | str, other_path: Path | str | None) -> bool:
    """Check whether *path* does not exist or was modified before *other_path*.

    Args:
        path: File path to test.
        other_path: Reference file path for modification-time comparison, or
            ``None`` to always return ``True`` when *path* does not exist.

    Returns:
        ``True`` if *path* does not exist, *other_path* is ``None``, or *path*
        has an older modification time than *other_path``.  ``False`` otherwise.
    """
    path = Path(path)
    if not path.exists():
        return True
    if other_path is None:
        return True
    fileCreation = os.path.getmtime(path)
    fileCreation_ref = os.path.getmtime(other_path)
    return fileCreation < fileCreation_ref


def _load_poi(fixed_poi_file: Path | None, vert: NII, subreg: NII, save_pois: bool) -> POI:
    """Load or compute a POI object for a vertebra segmentation.

    If *fixed_poi_file* is provided, that POI is loaded from disk (and its
    affine is validated against *vert*).  The corpus, spinous process, and arcus
    centroids are always (re-)computed from *vert* and *subreg* using
    :func:`calc_poi_from_subreg_vert`, with the loaded file used as a buffer.

    Args:
        fixed_poi_file: Optional path to a pre-computed POI file.
        vert: Vertebra segmentation ``NII``.
        subreg: Subregion segmentation ``NII``.
        save_pois: Save the computed POI to *fixed_poi_file* when ``True``.

    Returns:
        Computed ``POI`` object with corpus, spinous-process, and arcus centroids.
    """
    buffer_file = None
    if fixed_poi_file is not None:
        poi = POI.load(fixed_poi_file)
        poi.assert_affine(vert, raise_error=True)
        buffer_file = fixed_poi_file
    poi_ct = calc_poi_from_subreg_vert(
        vert,
        subreg,
        subreg_id=[Location.Vertebra_Corpus, Location.Spinosus_Process, Location.Arcus_Vertebrae],
        buffer_file=buffer_file,
        save_buffer_file=save_pois and buffer_file is not None,
    )
    return poi_ct


def compute_distance(array: np.ndarray, background: np.ndarray | int = 0, i: int = 0) -> np.ndarray:
    """Compute the Euclidean distance transform for label *i* excluding background.

    Args:
        array: Label array whose distance transform is computed.
        background: Background mask; voxels where this equals 0 are excluded from
            the distance computation.
        i: Label value in *array* to treat as the foreground seed.

    Returns:
        Distance transform array of the same shape as *array*.
    """
    return distance_transform_edt(np.logical_and(array == i, background == 0))  # type: ignore


def compute_distance_cord(segmentation: np.ndarray, query_coords: np.ndarray, i: int = 0) -> np.ndarray:
    """Compute the distance from each query coordinate to the nearest voxel of label *i*.

    Uses a KD-tree for efficient nearest-neighbour lookup.

    Args:
        segmentation: Label array.
        query_coords: Array of shape ``(N, ndim)`` with query coordinate positions.
        i: Label value whose foreground voxels form the reference set.

    Returns:
        1-D array of length ``N`` with the distance from each query coordinate to
        the nearest voxel labelled *i*.
    """
    # List of (y, x) coordinates where we want distances & segmentation lookup
    # query_coords = np.array([(0.5, 0.5), (1.7, 3.2), (2.3, 2.3)])

    # Get all foreground pixel coordinates for distance calculation
    object_pixels = np.argwhere(segmentation == i)
    # Build KDTree for fast nearest-neighbor search
    tree = cKDTree(object_pixels)
    # Query the tree to get the nearest distances
    distances, _ = tree.query(query_coords)
    return distances


class Rigid_Elements_Registration:
    """Per-vertebra rigid registration using DeepALI.

    Performs a two-stage spine registration pipeline:

    1. Point-based pre-alignment using vertebra centroids (``Point_Registration``).
    2. Per-element (per-vertebra) rigid intensity registration via
       ``Rigid_Registration_with_Tether``.

    A weighted-blending field is then computed from the individual vertebra
    transformations using an inverse-distance weighting scheme and exposed
    through :meth:`transform_nii` and :meth:`compute_weightings`.

    Args:
        fixed_image: Fixed (reference) image.
        fixed_vert: Vertebra segmentation aligned to *fixed_image*.
        fixed_subreg: Subregion segmentation aligned to *fixed_image*.
        moving_image: Moving image to be registered.
        moving_vert: Vertebra segmentation aligned to *moving_image*.
        moving_subreg: Subregion segmentation aligned to *moving_image*.
        fixed_poi: Optional path to a pre-computed fixed POI file.
        moving_poi: Optional path to a pre-computed moving POI file.
        crop_to_FOV: Crop images to the moving-image field of view when ``True``.
        save_pois: Save computed POI files to disk when ``True``.
        reference_image: Optional reference grid; defaults to *fixed_vert*.
        device: PyTorch device; inferred from *ddevice*/*gpu* when ``None``.
        gpu: GPU index used when *device* is ``None``.
        ddevice: Device type string (e.g. ``"cuda"``).
        pyramid_levels: Number of multi-resolution pyramid levels (``None`` = 1).
        finest_level: Index of the finest pyramid level.
        coarsest_level: Index of the coarsest pyramid level.
        pyramid_finest_spacing: Target voxel spacing at the finest pyramid level.
        pyramid_min_size: Minimum spatial extent per level.
        dims: Spatial axes to optimise (default all three).
        align: Pre-align grids before optimisation.
        optim_name: Name of a ``torch.optim`` optimiser class.
        lr: Optimiser learning rate.
        optim_args: Additional optimiser keyword arguments (without ``lr``).
        verbose: Verbosity level for progress output.
        max_steps: Maximum number of optimisation steps per level.
        max_history: Window size for convergence monitoring.
        weights: Loss term weights.
        patience: Early-stopping patience (number of steps without improvement).
        patience_delta: Minimum loss improvement to reset patience.
        my: Exponent for the inverse-distance weighting scheme (default 1.5).
        orientation: Reorient all inputs to this orientation before registration.
    """

    def __init__(
        self,
        fixed_image: Image_Reference,
        fixed_vert: Image_Reference,
        fixed_subreg: Image_Reference,
        moving_image: Image_Reference,
        moving_vert: Image_Reference,
        moving_subreg: Image_Reference,
        fixed_poi: Path | None = None,
        moving_poi: Path | None = None,
        crop_to_FOV=True,
        save_pois=True,
        reference_image: Image_Reference | None = None,
        device: torch.device | None = None,
        gpu=0,
        ddevice: DEVICES = "cuda",
        # Pyramid
        pyramid_levels: int | None = None,  # 1/None = no pyramid; int: number of stacks, tuple from to (0 is finest)
        finest_level: int = 0,
        coarsest_level: int | None = None,
        pyramid_finest_spacing: Sequence[int] | torch.Tensor | None = None,
        pyramid_min_size=16,
        dims=("x", "y", "z"),
        align=False,
        optim_name="Adam",  # Optimizer name defined in torch.optim. or override on_optimizer finer control
        lr=0.01,  # Learning rate
        optim_args=None,  # args of Optimizer with out lr
        verbose=99,
        max_steps: int | Sequence[int] = 10000,  # Early stopping.  override on_converged finer control
        max_history: int | None = None,
        weights: list[float] | None = None,
        patience=100,
        patience_delta=0.00001,
        my=1.5,
        orientation=None,
    ) -> None:
        self.my = my
        if weights is None:
            weights = [1, 0.0001]
        if device is None:
            device = get_device(ddevice, gpu)
        self.device: torch.device = device  # type: ignore
        self.orientation = orientation
        self._resample_and_preReg(
            fixed_image,
            fixed_vert,
            fixed_subreg,
            fixed_poi,
            moving_image,
            moving_vert,
            moving_subreg,
            moving_poi,
            reference_image,
            crop_to_FOV,
            save_pois,
            orientation,
        )

        self._run_rigid_reg(
            pyramid_levels,
            finest_level,
            coarsest_level,
            pyramid_finest_spacing,
            pyramid_min_size,
            dims,
            align,
            optim_name,
            lr,
            optim_args,
            verbose,
            max_steps,
            max_history,
            weights,
            patience,
            patience_delta,
        )
        self.y_total = self.compute_weightings()

    def _resample_and_preReg(
        self,
        fixed_image_: Image_Reference,  # noqa: ARG002
        fixed_vert_: Image_Reference,
        fixed_subreg_: Image_Reference,
        fixed_poi_file: Path | None,
        moving_image_: Image_Reference,
        moving_vert_: Image_Reference,
        moving_subreg_: Image_Reference,
        moving_poi_file: Path | None,
        reference_image: Image_Reference | None,
        crop_to_FOV: bool,
        save_pois: bool,
        orientation,
    ) -> None:
        """Load inputs, compute POIs, apply point-registration, and crop to FOV."""
        # Load
        # fixed_image = to_nii(fixed_image_)
        fixed_vert = to_nii(fixed_vert_, True)
        fixed_subreg = to_nii(fixed_subreg_, True)
        moving_image = to_nii(moving_image_)
        moving_vert = to_nii(moving_vert_, True)
        moving_subreg = to_nii(moving_subreg_, True)
        # POIs
        fixed_poi = _load_poi(fixed_poi_file, fixed_vert, fixed_subreg, save_pois)
        moving_poi = _load_poi(moving_poi_file, moving_vert, moving_subreg, save_pois)
        # Resample
        ref = fixed_vert if reference_image is None else reference_image
        if not isinstance(ref, Has_Grid):
            ref = to_nii(ref)
        if orientation is not None:
            ref.reorient_(orientation)  # type: ignore
            fixed_vert.reorient_(orientation)
            fixed_subreg.reorient_(orientation)
            moving_image.reorient_(orientation)
            moving_vert.reorient_(orientation)
            moving_subreg.reorient_(orientation)
            fixed_poi.reorient_(orientation)
            moving_poi.reorient_(orientation)

        if not fixed_vert.assert_affine(fixed_subreg, raise_error=False):
            fixed_subreg.resample_from_to_(fixed_vert)
        if not fixed_vert.assert_affine(fixed_poi, raise_error=False):
            fixed_poi = fixed_poi.resample_from_to(fixed_vert)

        if not moving_image.assert_affine(moving_vert, raise_error=False):
            moving_vert.resample_from_to_(moving_image)
        if not moving_image.assert_affine(moving_subreg, raise_error=False):
            moving_subreg.resample_from_to_(moving_image)
        if not moving_image.assert_affine(moving_poi, raise_error=False):
            moving_poi = moving_poi.resample_from_to(moving_image)

        # fixed_image.resample_from_to_(ref)
        fixed_vert.resample_from_to_(ref)
        fixed_subreg.resample_from_to_(ref)
        # moving_image.resample_from_to_(ref)
        # moving_vert.resample_from_to_(ref)
        # moving_subreg.resample_from_to_(ref)
        fixed_poi = fixed_poi.resample_from_to(ref)
        # moving_poi = moving_poi.resample_from_to(ref)
        ## POI reg ###
        # verts = len([k for k in fixed_poi.keys_region() if k < 39])
        point_reg = Point_Registration(fixed_poi, moving_poi)  # leave_worst_percent_out=max(0, verts - 13)
        self.point_reg = point_reg

        moved_image = point_reg.transform_nii(moving_image)
        moved_vert = point_reg.transform_nii(moving_vert)
        moved_subreg = point_reg.transform_nii(moving_subreg)
        ## crop
        crop = moved_image.compute_crop() if crop_to_FOV else None
        self.crop = crop
        if crop is not None:
            # moved_image.apply_crop_(crop)
            ref = ref.apply_crop(crop)
            moved_vert.apply_crop_(crop)
            moved_subreg.apply_crop_(crop)
            # fixed_image = fixed_image.apply_crop(crop)
            fixed_vert = fixed_vert.apply_crop_(crop)
            fixed_subreg = fixed_subreg.apply_crop_(crop)
        ## attach to object
        self.ref = ref.to_gird()
        # self.moving_image = moved_image
        self.moving_vert = moved_vert
        self.moving_subreg = moved_subreg
        # self.fixed_image = fixed_image
        self.fixed_vert = fixed_vert
        self.fixed_subreg = fixed_subreg

    def _run_rigid_reg(
        self,
        pyramid_levels: int | None = None,  # 1/None = no pyramid; int: number of stacks, tuple from to (0 is finest)
        finest_level: int = 0,
        coarsest_level: int | None = None,
        pyramid_finest_spacing: Sequence[int] | torch.Tensor | None = None,
        pyramid_min_size: int = 16,
        dims=("x", "y", "z"),
        align: bool = False,
        optim_name: str = "Adam",  # Optimizer name defined in torch.optim. or override on_optimizer finer control
        lr: float = 0.01,  # Learning rate
        optim_args=None,  # args of Optimizer with out lr
        verbose: int = 99,
        max_steps: int | Sequence[int] = 10000,  # Early stopping.  override on_converged finer control
        max_history: int | None = None,
        weights: list[float] | None = None,
        patience: int = 100,
        patience_delta: float = 0.0,
    ) -> None:
        """Run per-vertebra rigid registration and store the fitted models."""
        self._rigid_registrations: list[Rigid_Registration_with_Tether] = []
        self._ids = []
        ids_fixed = self.fixed_vert.unique()
        ids_moving = self.moving_vert.unique()
        print(f"{ids_fixed =}")
        print(f"{ids_moving=}")
        from TPTBox.core.vert_constants import Vertebra_Instance

        for idx in ids_fixed:
            if idx not in ids_moving:
                continue
            if idx >= 100:
                continue
            # subreg_ct[subreg_ct == 20] = 49
            try:
                name = Vertebra_Instance(idx).name
            except Exception:
                name = str(idx)
            reg = Rigid_Registration_with_Tether(
                self.fixed_vert.extract_label(idx) * self.fixed_subreg,
                self.moving_vert.extract_label(idx) * self.moving_subreg,
                pyramid_levels=pyramid_levels,
                finest_level=finest_level,
                coarsest_level=coarsest_level,
                pyramid_finest_spacing=pyramid_finest_spacing,
                pyramid_min_size=pyramid_min_size,
                dims=dims,
                align=align,
                optim_name=optim_name,
                lr=lr,
                optim_args=optim_args,
                verbose=verbose,
                max_steps=max_steps,
                max_history=max_history,
                weights=weights,
                patience=patience,
                patience_delta=patience_delta,
                desc=name,
            )

            self._rigid_registrations.append(reg)
            self._ids.append(idx)

    @torch.no_grad()
    def compute_weightings(
        self,
        my: float | None = None,
        coords: np.ndarray | None = None,
        device: torch.device | None = None,
        align_corners: bool = True,
    ) -> torch.Tensor:
        """Compute the inverse-distance weighted blending field across all vertebrae.

        Implements the weighting scheme from
        https://www.sciencedirect.com/science/article/pii/S1077314297906081

        Each vertebra's rigid transformation is blended using inverse-distance
        weights computed from the Euclidean distance to the vertebra label.
        Voxels inside a label are assigned weight 1 for their own vertebra and 0
        for all others.

        Args:
            my: Exponent for the inverse-distance weighting (defaults to
                ``self.my``).
            coords: Optional array of query coordinates (shape ``(N, 3)``).
                When ``None``, the full reference-image grid is used.
            device: PyTorch device for computation; defaults to ``self.device``.
            align_corners: Whether corners are aligned in grid-to-world mapping.

        Returns:
            Blended displacement field as a ``torch.Tensor`` containing the
            weighted sum of per-vertebra transformed grid points.
        """
        if my is None:
            my = self.my
        ids = self._ids
        print(ids)
        # compute distances of relevant verts
        cc = self.moving_vert.extract_label(ids, keep_label=True)
        print(cc.unique())
        ccs = [cc.extract_label(i).get_array() for i in ids]
        print(len(ccs))
        distance_map_cc = [compute_distance(i) for i in ccs] if coords is None else [compute_distance_cord(i, coords) for i in ccs]
        print(len(distance_map_cc))
        # compute weightings
        q = np.stack([(i**self.my) for i in distance_map_cc])
        cond = q != 0
        q[cond] = 1 / q[cond]
        q_sum: np.ndarray = np.expand_dims(q.sum(0), 0)
        q_sum[q_sum == 0] = 1
        weighting_field = np.zeros_like(q)
        weighting_field = q / q_sum
        cc2 = cc.get_array() if coords is None else map_coordinates(cc.get_array(), coords, order=0)
        # weighing inside other segs is 0 and 1 in it self
        weighting_field[:, cc2 != 0] = 0
        for i, j in enumerate(ids):
            weighting_field[i, cc2 == j] = 1

        weighting_field__ = np.transpose(weighting_field, axes=tuple(reversed(range(weighting_field.ndim))))
        y_total = 0
        device = self.device if device is None else device
        if coords is None:
            x = self.ref.to_deepali_grid().coords(device=device).unsqueeze(0)  # grid_points
            # compute field
            for i, reg in enumerate(self._rigid_registrations):
                y = reg.transform(x, grid=True)
                y = (y - x) * torch.Tensor(weighting_field__[..., i]).to(device).unsqueeze(0).unsqueeze(-1)
                y_total += y
        else:
            x = torch.Tensor(coords).to(device)
            for i, reg in enumerate(self._rigid_registrations):
                y = reg.transform.points(
                    x,
                    axes=Axes.GRID,
                    to_axes=Axes.GRID,
                    grid=self.ref.to_deepali_grid(align_corners),
                    to_grid=self.ref.to_deepali_grid(align_corners),
                )
                y = (y - x) * torch.Tensor(weighting_field__[..., i]).to(device).unsqueeze(0).unsqueeze(-1)
                y_total += y
        y_total = y_total + x
        return y_total

    # def transform_nii_moved(self):
    #    y_total = self.y_total
    #    d = self.device
    #
    #    _sample_image = SampleImage(
    #        target=self.ref.to_deepali_grid(),
    #        source=self.ref.to_deepali_grid(),
    #        sampling=Sampling.LINEAR,
    #        padding=PaddingMode.ZEROS,
    #        align_centers=False,
    #    ).to(d)
    #    moved_data = _sample_image(y_total, self.moving_image.to_deepali(device=d).tensor())
    #    moved_data = np.transpose(moved_data.detach().cpu().numpy(), axes=tuple(reversed(range(moved_data.ndim)))).squeeze()
    #
    #    return self.moving_image.set_array(moved_data)

    @torch.no_grad()
    def transform_nii(
        self,
        img: NII,
        gpu: int | None = None,
        ddevice: DEVICES | None = None,
        align_corners: bool = True,
        padding: PaddingMode = PaddingMode.ZEROS,
    ) -> NII:
        """Apply the blended rigid warp to an arbitrary NII image.

        The moving image is first pre-aligned via ``Point_Registration``, then
        cropped to the field of view, and finally warped using the blended
        displacement field stored in ``self.y_total``.

        Args:
            img: Input ``NII`` to transform (may be a different modality from
                those used during registration).
            gpu: GPU index override.
            ddevice: Device type override.
            align_corners: Whether corners are aligned during warping.
            padding: Padding mode used outside the source image bounds.

        Returns:
            Transformed ``NII`` resampled to the reference grid.
        """
        device = get_device(ddevice, 0 if gpu is None else gpu) if ddevice is not None else self.device
        if self.orientation:
            img = img.reorient(self.orientation)
        img_point_reg = self.point_reg.transform_nii(img)
        if self.crop is not None:
            img_point_reg.apply_crop_(self.crop)
        _sample_image = SampleImage(
            target=self.ref.to_deepali_grid(),
            source=self.ref.to_deepali_grid(),
            sampling=Sampling.NEAREST if img.seg else Sampling.LINEAR,
            padding=padding,
            align_centers=align_corners,
        ).to(device)

        moved_data = _sample_image(self.y_total, img_point_reg.to_deepali(device=device).tensor())
        moved_data = np.transpose(moved_data.detach().cpu().numpy(), axes=tuple(reversed(range(moved_data.ndim)))).squeeze()
        return self.ref.make_empty_nii(img.seg, moved_data)

    def transform_poi(self, poi: POI, gpu: int | None = None, ddevice: DEVICES | None = None, align_corners: bool = True) -> POI:
        """Apply the blended rigid warp to a POI object (not yet tested).

        Args:
            poi: Input ``POI`` whose centroids are to be transformed.
            gpu: GPU index override.
            ddevice: Device type override.
            align_corners: Whether corners are aligned during warping.

        Returns:
            Transformed ``POI`` on the reference grid.

        Raises:
            NotImplementedError: This method has not been validated yet.
        """
        raise NotImplementedError("transform_poi is not tested")
        device = get_device(ddevice, 0 if gpu is None else gpu) if ddevice is not None else self.device
        # POINT REG
        poi_moving = self.point_reg.transform_poi(poi)
        if self.crop is not None:
            poi_moving.apply_crop_(self.crop)
        keys: list[tuple[int, int]] = []
        points = []
        for key, key2, (x, y, z) in poi_moving.items():
            keys.append((key, key2))
            points.append((x, y, z))
            print(key, key2, (x, y, z))
        # transform poi
        data = self.compute_weightings(coords=np.array(points), device=device, align_corners=align_corners)
        out_poi = self.ref.make_empty_POI()
        for (key, key2), (x, y, z) in zip_strict(keys, data.cpu()):
            out_poi[key, key2] = (x.item(), y.item(), z.item())
        return out_poi

    def __call__(self, *args, **kwds) -> NII:
        """Apply the registration by calling :meth:`transform_nii`."""
        return self.transform_nii(*args, **kwds)

    def get_dump(self) -> None:
        """Return a serialisable representation of the registration (not implemented).

        Raises:
            NotImplementedError: Serialisation is not yet supported.
        """
        raise NotImplementedError("transform_poi is not tested")

    def save(self, path: str | Path) -> None:
        """Serialise the registration to a pickle file (not implemented).

        Args:
            path: Destination file path.

        Raises:
            NotImplementedError: Serialisation is not yet supported.
        """
        with open(path, "wb") as w:
            pickle.dump(self.get_dump(), w)

    @classmethod
    def load(cls, path: str | Path, gpu: int = 0, ddevice: DEVICES = "cuda"):
        """Load a saved registration from a pickle file (not implemented).

        Args:
            path: Path to the pickle file.
            gpu: GPU index.
            ddevice: Device type string.

        Raises:
            NotImplementedError: Deserialisation is not yet supported.
        """
        with open(path, "rb") as w:
            return cls.load_(pickle.load(w), gpu, ddevice)

    @classmethod
    def load_(cls, w, gpu: int = 0, ddevice: DEVICES = "cuda"):
        """Reconstruct from a raw dump tuple (not implemented).

        Raises:
            NotImplementedError: Deserialisation is not yet supported.
        """
        raise NotImplementedError("transform_poi is not tested")


if __name__ == "__main__":
    root_d = Path("/DATA/NAS/ongoing_projects/a_exchange/cosim_MBS_FEM/data/derivatives/spinegan0106/CT")
    root_d2 = Path("/DATA/NAS/ongoing_projects/a_exchange/cosim_MBS_FEM/data/registered/derivatives/T2w")
    root_r = Path("/DATA/NAS/ongoing_projects/a_exchange/cosim_MBS_FEM/data/registered/rawdata/T2w")
    vert_ct_org = to_nii(root_d / "sub-spinegan0106_ses-20220430_sequ-11_space-aligASL_seg-vert_msk.nii.gz", True)
    subreg_ct_org = to_nii(root_d / "sub-spinegan0106_ses-20220430_sequ-11_space-aligASL_seg-spine_msk.nii.gz", True)
    ct_org = to_nii(root_d / "sub-spinegan0106_ses-20220430_sequ-11_space-aligASL_ct.nii.gz", False)

    vert_t2w_org = to_nii(root_d2 / "sub-spinegan0106_ses-20220430_sequ-11-aligASL_mod-dixon_seg-vert_msk.nii.gz", True)
    subreg_t2w_org = to_nii(root_d2 / "sub-spinegan0106_ses-20220430_sequ-11-aligASL_mod-dixon_seg-spine_msk.nii.gz", True)
    t2w_org = to_nii(root_r / "sub-spinegan0106_ses-20220430_sequ-11-aligASL_moved2_dixon.nii.gz", False)
    # poi = root_d / "CT/sub-spinegan0152_ses-20220623_sequ-201_seg-subreg_ctd.json"
    subreg_t2w_org.origin = t2w_org.origin
    vert_t2w_org.origin = t2w_org.origin

    rer = Rigid_Elements_Registration(
        ct_org,
        vert_ct_org,
        subreg_ct_org,
        t2w_org,
        vert_t2w_org,
        subreg_t2w_org,
        ddevice="cuda",
        gpu=0,
        crop_to_FOV=True,
        # reference_image=vert_ct_org.rescale((1, 1, 1)),
        lr=0.001,
    )
    root_d = Path("/DATA/NAS/ongoing_projects/a_exchange/cosim_MBS_FEM/robert_test")
    a = rer.transform_nii(t2w_org)
    a.save(root_d / "t2w.nii.gz")
    ct_org.resample_from_to(a).save(root_d / "ct.nii.gz")
    vert_ct_org.resample_from_to(a).save(root_d / "ct_vert.nii.gz")
    subreg_ct_org.resample_from_to(a).save(root_d / "ct_subreg.nii.gz")
    rer.transform_nii(vert_t2w_org).save(root_d / "vert.nii.gz")
    rer.transform_nii(subreg_t2w_org).save(root_d / "subreg.nii.gz")
