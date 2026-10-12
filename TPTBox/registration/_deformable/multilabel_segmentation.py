from __future__ import annotations

import pickle
from pathlib import Path
from typing import TypeVar

from TPTBox import NII, POI
from TPTBox.core.internal.deep_learning_utils import DEVICES
from TPTBox.core.poi import calc_centroids
from TPTBox.core.poi_fun.poi_global import POI_Global
from TPTBox.registration._deformable.deformable_reg import Deformable_Registration
from TPTBox.registration._ridged_intensity.affine_deepali import Tether_Seg
from TPTBox.registration._ridged_points.deepali_point_registration import Deepali_Point_Registration
from TPTBox.registration._ridged_points.point_registration import Point_Registration

_NIIOrPOI = TypeVar("_NIIOrPOI", NII, POI)


def _r_axis_slicer(axis: int) -> tuple[slice, ...]:
    """Return a ``[::-1]``-along-``axis`` slicer for numpy indexing.

    Used for the left/right flip that ``Template_Registration`` and
    ``Template_Registration2`` perform when ``same_side=False``.
    """
    if axis == 0:
        return (slice(None, None, -1),)
    if axis == 1:
        return (slice(None), slice(None, None, -1))
    if axis == 2:
        return (slice(None), slice(None), slice(None, None, -1))
    raise ValueError(axis)


def _flip_r_axis(x: _NIIOrPOI) -> _NIIOrPOI:
    """Mirror ``x`` along its R (left/right) axis.

    Works on both a ``NII`` volume (flips the underlying array) and a voxel
    ``POI`` (mirrors each centroid coordinate around ``shape[axis] - 1``).
    ``POI_Global`` is not supported – resample it onto a voxel grid first.

    Args:
        x: Volume or voxel-space POI to flip.

    Returns:
        A new object of the same type with the R-axis reversed.
    """
    if isinstance(x, POI_Global):
        raise TypeError("_flip_r_axis: POI_Global has no shape/axis; resample to a voxel grid first.")
    axis = x.get_axis("R")
    if isinstance(x, NII):
        slicer = _r_axis_slicer(axis)
        return x.set_array(x.get_array()[slicer]).copy()
    # POI (voxel space)
    out = x.make_empty_POI()
    shape = x.shape
    for k1, k2, (a, b, c) in x.copy().items():
        if axis == 0:
            out[k1, k2] = (shape[0] - 1 - a, b, c)
        elif axis == 1:
            out[k1, k2] = (a, shape[1] - 1 - b, c)
        elif axis == 2:
            out[k1, k2] = (a, b, shape[2] - 1 - c)
        else:
            raise ValueError(axis)
    return out


class Template_Registration:
    """Multi-stage registration between two multi-label segmentations.

    Supports optional POI landmark alignment and deformable registration; landmarks are computed
    on the fly if not provided.  Particularly useful for MRI/CT atlas alignment with optional
    body-side flip handling.

    Attributes:
        same_side (bool): Whether the target and atlas represent the same anatomical side (e.g., both right sides).
        reg_point (Point_Registration): The rigid point-based registration component.
        reg_deform (Deformable_Registration): The deformable registration component.
        crop (tuple): The crop applied to both target and atlas after registration.
        target_grid_org (NII): Original spatial grid of the target.
        atlas_org (NII): Original spatial grid of the atlas.
        target_grid (NII): Cropped spatial grid used for deformable registration.
    """

    def __init__(  # noqa: C901
        self,
        target_seg: NII,
        atlas_seg: NII,
        target_img: NII | None = None,
        atlas_img: NII | None = None,
        poi_cms: POI | None = None,
        same_side: bool = True,
        verbose=99,
        gpu=0,
        ddevice: DEVICES = "cuda",
        loss_terms=None,  # type: ignore
        weights=None,
        lr=0.01,
        lr_end_factor=None,
        max_steps=1500,
        min_delta: float | list[float] = 1e-06,
        pyramid_levels=4,
        coarsest_level=3,
        finest_level=0,
        crop: bool = True,
        cms_ids: list | None = None,
        poi_target_cms: POI | None = None,
        max_history=100,
        change_after_point_reg=lambda x, y, z, w: (x, y, z, w),
        tether_distance=1,
        **args,
    ):
        """Initialize a multi-stage registration pipeline from an atlas to a target image.

        Args:
            target_seg (NII): Target image segmentation (e.g., from a subject).
            atlas_seg (NII): Atlas image segmentation (e.g., a reference or template).
            target_img (NII | None): Target intensity image; if None the segmentation is used as an image.
            atlas_img (NII | None): Atlas intensity image; if None the segmentation is used as an image.
            poi_cms (POI | None): POI centroids of the atlas, used for initial point registration.
            same_side (bool): Whether atlas and target represent the same body side.
            verbose (int): Verbosity level for logging.
            gpu (int): GPU device ID (only relevant if using GPU).
            ddevice (DEVICES): Device type ('cuda' or 'cpu').
            loss_terms (dict | None): Dictionary of loss terms for deformable registration.
            weights (dict | None): Weights for the loss terms.
            lr (float): Learning rate for deformable registration optimizer.
            lr_end_factor (float | None): If set, exponentially decay the LR by this final factor across steps.
            max_steps (int): Maximum optimization steps.
            min_delta (float | list[float]): Minimum delta for convergence (per pyramid level if a list is given).
            pyramid_levels (int): Number of resolution levels in multi-scale deformable registration.
            coarsest_level (int): Coarsest level index.
            finest_level (int): Finest level index.
            crop (bool): If True, crop both target and atlas to their combined bounding box before registration.
            cms_ids (list | None): List of segmentation labels used to extract POI centroids.
            poi_target_cms (POI | None): Optional precomputed centroids for the target image.
            max_history (int): Number of past deformable-registration parameter snapshots to keep for rollback.
            change_after_point_reg (Callable): Hook applied to ``(target_seg, atlas_seg, target_img, atlas_img)``
                between the point-based and deformable stages; defaults to identity.
            tether_distance (float): Distance parameter for the segmentation-tether loss.
            **args: Additional keyword arguments passed to Deformable_Registration.

        Raises:
            ValueError: If an invalid axis is detected during flipping.
        """
        if weights is None:
            weights = {"be": 0.0001, "seg": 1, "Dice": 0.01, "Tether": 0.001}
        if loss_terms is None:
            loss_terms = {
                "be": ("BSplineBending", {"stride": 1}),
                "seg": "MSE",
                "Dice": "Dice",
                "Tether": Tether_Seg(delta=tether_distance),
            }

        assert target_seg.seg, target_seg.seg
        assert atlas_seg.seg
        target_seg = target_seg.copy()
        atlas_seg = atlas_seg.copy()
        if target_img is not None:
            target_img = target_img.resample_from_to(target_seg)
        if atlas_img is not None:
            atlas_img = atlas_img.resample_from_to(atlas_seg)
        self.same_side = same_side
        self.target_grid_org = target_seg.to_gird()
        self.atlas_org = atlas_seg.to_gird()
        if not same_side:
            target_seg = _flip_r_axis(target_seg)
            if target_img is not None:
                target_img = _flip_r_axis(target_img)
            if poi_target_cms is not None:
                poi_target_cms = _flip_r_axis(poi_target_cms)
        if poi_target_cms is None:
            x = target_seg.extract_label(cms_ids, keep_label=True) if cms_ids else target_seg
            poi_target = calc_centroids(x, second_stage=40, bar=True)  # TODO REMOVE
        else:
            poi_target = poi_target_cms.resample_from_to(target_seg)
        if poi_cms is None:
            x = atlas_seg.extract_label(cms_ids, keep_label=True) if cms_ids else atlas_seg
            poi_cms = calc_centroids(x, second_stage=40, bar=True)
        if not poi_cms.assert_affine(atlas_seg, raise_error=False):
            poi_cms = poi_cms.resample_from_to(atlas_seg)
        if crop:
            print("crop")

            crop_pad_size = 50
            _step = 50
            _max_iter = 10

            resize_mode = "crop"
            resize_param: tuple | None = None
            target_tmp = target_seg

            atlas_seg_ = atlas_seg.apply_pad(((1, 1), (1, 1), (1, 1))) if atlas_seg.is_segmentation_in_border() else atlas_seg

            for i in range(_max_iter):
                if resize_mode == "crop":
                    if i != 0:
                        crop_pad_size += _step

                    # --- try crop first ---
                    t_crop = target_seg.compute_crop(0, crop_pad_size)
                    cropped = target_seg.apply_crop(t_crop).apply_pad(crop_pad_size - 50 // 4)

                    if any(c < o for c, o in zip(cropped.shape, target_seg.shape)):
                        resize_mode = "crop"
                        resize_param = t_crop
                        target_tmp = cropped
                    else:
                        # --- fallback to padding ---
                        crop_pad_size = crop_pad_size // 2
                        target_tmp = target_seg
                        resize_mode = "pad"
                else:
                    if i != 0:
                        crop_pad_size += _step // 2
                    t_pad = tuple((crop_pad_size, crop_pad_size) for _ in range(3))
                    resize_param = t_pad
                    target_tmp = target_seg.apply_pad(t_pad)

                # --- Point registration ---
                print(f"iter {i}: using {resize_mode} ({crop_pad_size})")

                poi_target = poi_target.resample_from_to(target_tmp)

                if poi_cms is None:
                    x = atlas_seg_.extract_label(cms_ids, keep_label=True) if cms_ids else atlas_seg_
                    poi_cms = calc_centroids(x, second_stage=40, bar=True)

                if not poi_cms.assert_affine(atlas_seg_, raise_error=False):
                    poi_cms = poi_cms.resample_from_to(atlas_seg_)

                self.reg_point = Point_Registration(poi_target, poi_cms, verbose=False)
                atlas_reg = self.reg_point.transform_nii(atlas_seg_, c_val=0)

                if not atlas_reg.is_segmentation_in_border():
                    print("point registration ok")
                    break
                else:
                    print("atlas_reg touches border → expanding")

            # --- FINAL STEP: apply once to original target ---
            if resize_mode == "crop":
                target_seg = target_seg.apply_crop(resize_param)
                target_img = target_img.apply_crop(resize_param) if target_img is not None else None
            elif resize_mode == "pad":
                target_seg = target_seg.apply_pad(resize_param)
                target_img = target_img.apply_pad(resize_param) if target_img is not None else None

        self.reg_point = Point_Registration(poi_target.resample_from_to(target_seg), poi_cms.resample_from_to(atlas_seg))
        atlas_reg = self.reg_point.transform_nii(atlas_seg, c_val=0)
        atlas_img_reg = self.reg_point.transform_nii(atlas_img) if atlas_img is not None else None

        if crop:
            self.crop = (target_seg + atlas_reg).compute_crop(0, 5)
            target_seg = target_seg.apply_crop(self.crop)
            target_img = target_img.apply_crop(self.crop) if target_img is not None else None
            atlas_reg = atlas_reg.apply_crop(self.crop)
            atlas_img_reg = atlas_img_reg.apply_crop(self.crop) if atlas_img_reg is not None else None
        else:
            self.crop = None

        self.target_grid = target_seg.to_gird()
        target_seg, atlas_reg, target_img, atlas_img_reg = change_after_point_reg(target_seg, atlas_reg, target_img, atlas_img_reg)
        self.reg_deform = Deformable_Registration(
            target_seg if target_img is None else target_img,
            atlas_reg if atlas_img_reg is None else atlas_img_reg,
            target_seg.copy(),
            atlas_reg.copy(),
            loss_terms=loss_terms,
            weights=weights,
            lr=lr,
            lr_end_factor=lr_end_factor,
            max_steps=max_steps,
            min_delta=min_delta,
            pyramid_levels=pyramid_levels,
            coarsest_level=coarsest_level,
            finest_level=finest_level,
            verbose=verbose,
            gpu=gpu,
            ddevice=ddevice,
            max_history=max_history,
            **args,
        )

    def get_dump(self) -> tuple:
        """Collect the serialisable state of this registration object.

        Returns:
            A tuple containing the version tag followed by all state components
            needed to reconstruct the object via :meth:`load_`.
        """
        return (
            1,  # version
            (self.reg_point.get_dump()),
            (self.reg_deform.get_dump()),
            (
                self.same_side,
                self.atlas_org,
                self.target_grid_org,
                self.target_grid,
                self.crop,
            ),
        )

    def save(self, path: str | Path) -> None:
        """Serialise the registration state to a pickle file.

        Args:
            path: Destination file path.
        """
        with open(path, "wb") as w:
            pickle.dump(self.get_dump(), w)

    @classmethod
    def load(cls, path: str | Path) -> Template_Registration:
        """Load a previously saved registration state from a pickle file.

        Args:
            path: Path to the pickle file created by :meth:`save`.

        Returns:
            Reconstructed ``Template_Registration`` instance.
        """
        with open(path, "rb") as w:
            return cls.load_(pickle.load(w))

    @classmethod
    def load_(cls, w: tuple) -> Template_Registration:
        """Reconstruct a ``Template_Registration`` from a raw state tuple (as returned by :meth:`get_dump`).

        Args:
            w: Serialised state tuple.

        Returns:
            Reconstructed ``Template_Registration`` instance.
        """
        (version, t0, t1, x) = w
        assert version == 1, f"Version mismatch {version=}"
        self = cls.__new__(cls)
        self.reg_point = Point_Registration.load_(t0)
        self.reg_deform = Deformable_Registration.load_(t1)
        (
            self.same_side,
            self.atlas_org,
            self.target_grid_org,
            self.target_grid,
            self.crop,
        ) = x

        return self

    def transform_nii(self, nii_atlas: NII, allow_only_same_grid_as_moving: bool = True, only_rigid=False) -> NII:
        """Apply both rigid and deformable registration to a NII image.

        Args:
            nii_atlas: Atlas image to be transformed (must share the atlas grid).
            allow_only_same_grid_as_moving: If True, assert that *nii_atlas* matches
                the grid of the moving image used during point registration.
            only_rigid: If True, apply only the point-based rigid registration and skip
                the deformable stage. Defaults to False.

        Returns:
            Transformed ``NII`` aligned with the original target image space.
        """
        nii_atlas = self.reg_point.transform_nii(nii_atlas, allow_only_same_grid_as_moving=allow_only_same_grid_as_moving)
        if only_rigid:
            return nii_atlas

        nii_atlas = nii_atlas.apply_crop(self.crop)
        nii_reg = self.reg_deform.transform_nii(nii_atlas)
        if nii_reg.seg:
            nii_reg.set_dtype_("smallest_uint")
        out = nii_reg.resample_from_to(self.target_grid_org, mode="constant")
        if self.same_side:
            return out
        return _flip_r_axis(out)

    def transform_poi(self, poi_atlas: POI_Global | POI) -> POI:
        """Apply both rigid and deformable registration to a POI landmark set.

        Args:
            poi_atlas: Atlas landmarks to be transformed (defined in the atlas space).

        Returns:
            Transformed ``POI`` landmarks aligned to the target image space.
        """
        poi_atlas = poi_atlas.resample_from_to(self.atlas_org)

        # Point Reg
        poi_atlas = self.reg_point.transform_poi(poi_atlas)
        # Deformable
        poi_atlas = poi_atlas.apply_crop(self.crop)

        poi_reg = self.reg_deform.transform_poi(poi_atlas)
        poi_reg = poi_reg.resample_from_to(self.target_grid_org)
        if self.same_side:
            return poi_reg
        return _flip_r_axis(poi_reg)

    def transform_poi_inverse(self, poi_target: POI_Global | POI):
        """Transform POIs from target space back into atlas space.

        Args:
            poi_target (POI_Global | POI): POIs defined in target space.

        Returns:
            POI: POIs mapped back into atlas space.
        """
        poi = poi_target.copy()

        # --- undo left/right flip if needed (POI_Global has no shape → resample first) ---
        if not self.same_side:
            if isinstance(poi, POI_Global):
                poi = poi.resample_from_to(self.target_grid_org)
            poi = _flip_r_axis(poi)

        # --- resample into deformable registration grid ---
        poi = poi.resample_from_to(self.target_grid)

        # --- inverse deformable registration ---
        reg_deform_inv = self.reg_deform.inverse()
        poi = reg_deform_inv.transform_poi(poi)

        # --- undo crop ---
        # if self.crop is not None:
        #    poi = poi.apply_crop_inverse(self.crop)

        # --- inverse rigid point registration ---
        poi = self.reg_point.transform_poi_inverse(poi, allow_only_same_grid_as_moving=False)

        # --- back to atlas grid ---
        poi = poi.resample_from_to(self.atlas_org)

        return poi


class Template_Registration2:
    """Atlas-to-target multi-stage registration with a pluggable pre-registration.

    Same idea as :class:`Template_Registration` (rigid → deformable pipeline for
    multi-label atlas alignment) with two structural changes:

    * The rigid stage is a :class:`Deepali_Point_Registration` – closed-form
      Kabsch/Horn on POI landmarks, wrapped as a DeepALI ``HomogeneousTransform``.
    * That transform is composed with the deformable stage instead of being
      applied first as an ``sitk`` resample. The atlas is therefore never
      pre-warped, which removes the extra bilinear resampling step (and the
      associated intensity blur / segmentation-boundary jitter) that
      ``Template_Registration`` incurs.

    In addition, a caller can supply a pre-fitted rigid registration via
    ``pre_registration=``; in that case no landmarks / centroids are computed
    internally.

    Attributes:
        same_side: Same as :class:`Template_Registration`.
        reg_point: The rigid :class:`Deepali_Point_Registration`.
        reg_deform: The :class:`Deformable_Registration` fitted on the
            non-resampled atlas (its transform is composed with ``reg_point``).
        target_grid_org: Original grid of the target image (used for the final
            resample back).
        atlas_org: Original grid of the atlas.
        crop: Optional crop applied before the deformable stage (mirrors
            :class:`Template_Registration`).
    """

    def __init__(  # noqa: C901
        self,
        target_seg: NII,
        atlas_seg: NII,
        target_img: NII | None = None,
        atlas_img: NII | None = None,
        poi_cms: POI | None = None,
        pre_registration: Deepali_Point_Registration | None = None,
        same_side: bool = True,
        verbose: int = 99,
        gpu: int = 0,
        ddevice: DEVICES = "cuda",
        loss_terms=None,
        weights=None,
        lr: float = 0.01,
        lr_end_factor: float | None = None,
        max_steps: int = 1500,
        min_delta: float | list[float] = 1e-06,
        pyramid_levels: int = 4,
        coarsest_level: int = 3,
        finest_level: int = 0,
        crop: bool = True,
        cms_ids: list | None = None,
        poi_target_cms: POI | None = None,
        max_history: int = 100,
        tether_distance: float = 1,
        **args,
    ) -> None:
        """See :class:`Template_Registration` for the shared arguments.

        Extra / changed arguments:

        Args:
            pre_registration: A previously fitted
                :class:`Deepali_Point_Registration` mapping ``atlas → target``.
                When provided, POI computation and the internal rigid fit are
                skipped – useful when the caller has already run the rigid step
                (e.g. as part of a shared pipeline) or wants to supply a custom
                landmark set.

        Raises:
            ValueError: When flipping is requested (``same_side=False``) but the
                target axis cannot be inferred.
        """
        if weights is None:
            weights = {"be": 0.0001, "seg": 1, "Dice": 0.01, "Tether": 0.001}
        if loss_terms is None:
            loss_terms = {
                "be": ("BSplineBending", {"stride": 1}),
                "seg": "MSE",
                "Dice": "Dice",
                "Tether": Tether_Seg(delta=tether_distance),
            }

        assert target_seg.seg, target_seg.seg
        assert atlas_seg.seg
        target_seg = target_seg.copy()
        atlas_seg = atlas_seg.copy()
        if target_img is not None:
            target_img = target_img.resample_from_to(target_seg)
        if atlas_img is not None:
            atlas_img = atlas_img.resample_from_to(atlas_seg)
        self.same_side = same_side
        self.target_grid_org = target_seg.to_gird()
        self.atlas_org = atlas_seg.to_gird()

        if not same_side:
            target_seg = _flip_r_axis(target_seg)
            if target_img is not None:
                target_img = _flip_r_axis(target_img)
            if poi_target_cms is not None:
                poi_target_cms = _flip_r_axis(poi_target_cms)

        # --- rigid pre-registration --------------------------------------------------
        if pre_registration is not None:
            self.reg_point = pre_registration
        else:
            if poi_target_cms is None:
                x_seg = target_seg.extract_label(cms_ids, keep_label=True) if cms_ids else target_seg
                poi_target = calc_centroids(x_seg, second_stage=40, bar=True)
            else:
                poi_target = poi_target_cms.resample_from_to(target_seg)
            if poi_cms is None:
                x_seg = atlas_seg.extract_label(cms_ids, keep_label=True) if cms_ids else atlas_seg
                poi_cms_local = calc_centroids(x_seg, second_stage=40, bar=True)
            else:
                poi_cms_local = poi_cms
            if not poi_cms_local.assert_affine(atlas_seg, raise_error=False):
                poi_cms_local = poi_cms_local.resample_from_to(atlas_seg)
            self.reg_point = Deepali_Point_Registration(poi_target, poi_cms_local, verbose=False, ddevice=ddevice, gpu=gpu)

        # --- optional crop -----------------------------------------------------------
        # Unlike Template_Registration we do NOT pre-resample the atlas: the rigid
        # transform is composed with the deformable stage instead. We still crop the
        # target grid to a tight bounding box (if requested) so the deformable
        # optimisation is cheaper.
        if crop:
            self.crop = target_seg.compute_crop(0, 5)
            target_seg = target_seg.apply_crop(self.crop)
            if target_img is not None:
                target_img = target_img.apply_crop(self.crop)
        else:
            self.crop = None
        self.target_grid = target_seg.to_gird()

        # --- deformable stage --------------------------------------------------------
        # Fit the deformable registration between target (fixed) and atlas (moving)
        # on their native grids using same_space=False. We still let the deformable
        # registration warm-start from the rigid transform by applying the rigid to
        # the atlas *only for loss evaluation via the deformable pyramid*.
        atlas_moved_seg = self.reg_point.transform_nii(atlas_seg, allow_only_same_grid_as_moving=False)
        atlas_moved_img = self.reg_point.transform_nii(atlas_img, allow_only_same_grid_as_moving=False) if atlas_img is not None else None
        if crop:
            atlas_moved_seg = atlas_moved_seg.apply_crop(self.crop)
            atlas_moved_img = atlas_moved_img.apply_crop(self.crop) if atlas_moved_img is not None else None

        self.reg_deform = Deformable_Registration(
            target_seg if target_img is None else target_img,
            atlas_moved_seg if atlas_moved_img is None else atlas_moved_img,
            target_seg.copy(),
            atlas_moved_seg.copy(),
            loss_terms=loss_terms,
            weights=weights,
            lr=lr,
            lr_end_factor=lr_end_factor,
            max_steps=max_steps,
            min_delta=min_delta,
            pyramid_levels=pyramid_levels,
            coarsest_level=coarsest_level,
            finest_level=finest_level,
            verbose=verbose,
            gpu=gpu,
            ddevice=ddevice,
            max_history=max_history,
            **args,
        )

    # ------------------------------------------------------------------ serialisation
    def get_dump(self) -> tuple:
        return (
            1,
            self.reg_point.get_dump(),
            self.reg_deform.get_dump(),
            (
                self.same_side,
                self.atlas_org,
                self.target_grid_org,
                self.target_grid,
                self.crop,
            ),
        )

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as w:
            pickle.dump(self.get_dump(), w)

    @classmethod
    def load(cls, path: str | Path, gpu: int = 0, ddevice: DEVICES = "cuda") -> Template_Registration2:
        with open(path, "rb") as w:
            return cls.load_(pickle.load(w), gpu=gpu, ddevice=ddevice)

    @classmethod
    def load_(cls, w: tuple, gpu: int = 0, ddevice: DEVICES = "cuda") -> Template_Registration2:
        version, t0, t1, x = w
        assert version == 1, f"Version mismatch {version=}"
        self = cls.__new__(cls)
        self.reg_point = Deepali_Point_Registration.load_(t0, gpu=gpu, ddevice=ddevice)
        self.reg_deform = Deformable_Registration.load_(t1, gpu=gpu, ddevice=ddevice)
        (
            self.same_side,
            self.atlas_org,
            self.target_grid_org,
            self.target_grid,
            self.crop,
        ) = x
        return self

    # ------------------------------------------------------------------------- warping
    def transform_nii(self, nii_atlas: NII, allow_only_same_grid_as_moving: bool = True, only_rigid: bool = False) -> NII:
        """Warp an atlas NII into the target space (rigid + deformable)."""
        nii_atlas = self.reg_point.transform_nii(nii_atlas, allow_only_same_grid_as_moving=allow_only_same_grid_as_moving)
        if only_rigid:
            return nii_atlas
        if self.crop is not None:
            nii_atlas = nii_atlas.apply_crop(self.crop)
        nii_reg = self.reg_deform.transform_nii(nii_atlas)
        if nii_reg.seg:
            nii_reg.set_dtype_("smallest_uint")
        out = nii_reg.resample_from_to(self.target_grid_org, mode="constant")
        if self.same_side:
            return out
        return _flip_r_axis(out)

    def transform_poi(self, poi_atlas: POI_Global | POI) -> POI:
        """Warp atlas POIs into the target space (rigid + deformable)."""
        poi_atlas = poi_atlas.resample_from_to(self.atlas_org)
        poi_atlas = self.reg_point.transform_poi(poi_atlas)
        if self.crop is not None:
            poi_atlas = poi_atlas.apply_crop(self.crop)
        poi_reg = self.reg_deform.transform_poi(poi_atlas)
        poi_reg = poi_reg.resample_from_to(self.target_grid_org)
        return poi_reg if self.same_side else _flip_r_axis(poi_reg)

    def transform_poi_inverse(self, poi_target: POI_Global | POI) -> POI:
        """Inverse of :meth:`transform_poi` – target → atlas."""
        poi = poi_target.copy()
        if not self.same_side:
            # POI_Global has no shape; resample onto the (already-flipped) target grid first.
            if isinstance(poi, POI_Global):
                poi = poi.resample_from_to(self.target_grid_org)
            poi = _flip_r_axis(poi)
        poi = poi.resample_from_to(self.target_grid)
        reg_deform_inv = self.reg_deform.inverse()
        poi = reg_deform_inv.transform_poi(poi)
        poi = self.reg_point.transform_poi_inverse(poi, allow_only_same_grid_as_moving=False)
        poi = poi.resample_from_to(self.atlas_org)
        return poi
