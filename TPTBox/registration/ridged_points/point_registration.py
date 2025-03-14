from __future__ import annotations

import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import SimpleITK as sitk  # noqa: N813

from TPTBox import (
    AX_CODES,
    NII,
    POI,
    Has_Grid,
    Image_Reference,
    Location,
    Log_Type,
    Logger_Interface,
    No_Logger,
    POI_Reference,
    calc_poi_from_subreg_vert,
    to_nii,
)
from TPTBox.core.compat import zip_strict
from TPTBox.core.sitk_utils import nii_to_sitk, sitk_to_nii

NII_or_POI = TypeVar("NII_or_POI")


class Point_Registration:
    def __init__(
        self,
        poi_fixed: POI,
        poi_moving: POI,
        exclusion=None,
        log: Logger_Interface = No_Logger(),  # noqa: B008
        verbose=True,
        ax_code=None,
        zooms=None,
        leave_worst_percent_out=0.0,
    ):
        """Use two Centroids object to compute a ridged_points registration.

        Args:
            ctd_fixed (Centroids): _description_
            ctd_movig (Centroids): _description_
            representative_fixed (Image_Reference, optional): _description_. Defaults to None.
            representative_movig (Image_Reference, optional): _description_. Defaults to None.
            exclusion (list, optional): _description_. Defaults to [].
            log (_type_, optional): _description_. Defaults to No_Logger().
            verbose (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: Require at least two points

        Returns:
            Resample_Filter
        """
        assert leave_worst_percent_out < 1.0
        assert leave_worst_percent_out >= 0.0
        if exclusion is None:
            exclusion = []
        if ax_code is not None:
            poi_fixed.reorient_(ax_code)
        if zooms is not None and zooms != (-1, -1, -1):
            poi_fixed.rescale_(zooms)
        representative_f_sitk = nii_to_sitk(poi_fixed.make_empty_nii())
        representative_m_sitk = nii_to_sitk(poi_moving.make_empty_nii())

        # Register
        # filter points by name
        f_keys = list(filter(lambda x: x[0] not in exclusion, poi_fixed.keys()))
        m_keys = list(poi_moving.keys())
        # limit to only shared labels
        inter = [x for x in f_keys if x in m_keys]
        log.print(f_keys, verbose=verbose)
        log.print(poi_fixed.orientation, verbose=verbose)

        if len(inter) <= 2:
            log.print("[!] To few points, skip registration", Log_Type.FAIL)
            raise ValueError("[!] To few points, skip registration", inter)
        img_movig = poi_moving.make_empty_nii()
        assert img_movig.shape == poi_moving.shape_int, (img_movig, poi_moving.shape)
        assert img_movig.orientation == poi_moving.orientation
        if leave_worst_percent_out != 0.0:
            poi_fixed = poi_fixed.intersect(poi_moving)
            init_transform, error_reg, error_natural, delta_after = _compute_versor(
                inter, poi_fixed, representative_f_sitk, poi_moving, representative_m_sitk, verbose=False, log=log
            )
            delta_after = sorted(delta_after.items(), key=lambda x: -x[1])
            out_str = f"Did not use the following keys for registaiton (worst {leave_worst_percent_out * 100} %) "
            for i, key in enumerate(delta_after):
                if i >= len(delta_after) * leave_worst_percent_out:
                    break
                poi_fixed.remove_centroid_(key[0])
                out_str += f"{key}, "
            log.print(out_str, verbose=verbose)
            log.print("Error with all points", error_reg, Log_Type.STAGE, verbose=verbose)
            f_keys = list(filter(lambda x: x[0] not in exclusion, poi_fixed.keys()))
            m_keys = list(poi_moving.keys())
        # limit to only shared labels
        inter = [x for x in f_keys if x in m_keys]
        init_transform, error_reg, error_natural, _ = _compute_versor(
            inter, poi_fixed, representative_f_sitk, poi_moving, representative_m_sitk, verbose=verbose, log=log
        )
        self._transform: sitk.VersorRigid3DTransform = init_transform

        ### Point Reg
        self._img_moving: sitk.Image = representative_m_sitk
        self._img_fixed: sitk.Image = representative_f_sitk
        self.error_reg: float = error_reg
        self.error_natural: float = error_natural
        self.input_poi: Has_Grid = poi_moving.to_gird()
        self.out_poi: Has_Grid = poi_fixed.to_gird()

    def get_resampler(self, seg, c_val, output_space: NII | None = None) -> sitk.ResampleImageFilter:
        resampler: sitk.ResampleImageFilter = sitk.ResampleImageFilter()
        resampler_seg = sitk.ResampleImageFilter()
        if output_space is None:
            resampler_seg.SetReferenceImage(self._img_fixed)
        else:
            resampler.SetReferenceImage(nii_to_sitk(output_space))
        if seg:
            resampler_seg.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
        else:
            resampler.SetInterpolator(sitk.sitkBSplineResampler)
            resampler.SetDefaultPixelValue(c_val)
        resampler_seg.SetTransform(self._transform)
        ### Segmentation Resampler
        return resampler_seg

    def transform(self, x: NII_or_POI) -> NII_or_POI:
        if isinstance(x, POI):
            return self.transform_poi(x)
        if isinstance(x, NII):
            return self.transform_nii(x)
        raise ValueError

    def transform_poi(self, poi_moving: POI, allow_only_same_grid_as_moving=True, output_space=None):
        # output_space: POI | NII | None = None,
        if allow_only_same_grid_as_moving:
            text = "input image must be in the same space as moving.  If you sure that this input is in same space as the moving image you can turn of 'only_allow_grid_as_moving'"
            poi_moving.assert_affine(self.input_poi, text=text)
        move_l = []
        keys = []
        out = dict(zip_strict(keys, move_l))

        for key, key2, (x, y, z) in poi_moving.items():
            out[key, key2] = self.transform_cord((x, y, z))

        poi = self.out_poi.make_empty_POI(out)
        if output_space is not None:
            poi = poi.resample_from_to(output_space)
        return poi

    def transform_cord(self, cord: tuple[float, ...], out: sitk.Image | None = None):
        if out is None:
            out = self._img_fixed
        ctr_b = self._img_moving.TransformContinuousIndexToPhysicalPoint(cord)
        ctr_b = self._transform.GetInverse().TransformPoint(ctr_b)
        ctr_b = out.TransformPhysicalPointToContinuousIndex(ctr_b)
        return np.array(ctr_b)

    def transform_nii(
        self, moving_img_nii: NII, allow_only_same_grid_as_moving=True, output_space: NII | None = None, c_val: float | None = None
    ):
        if allow_only_same_grid_as_moving:
            text = "input image must be in the same space as moving.  If you sure that this input is in same space as the moving image you can turn of 'only_allow_grid_as_moving'"
            moving_img_nii.assert_affine(self.input_poi, text=text, shape_tolerance=0.9)
        if c_val is None:
            c_val = moving_img_nii.get_c_val()
        resampler = self.get_resampler(moving_img_nii.seg, c_val, output_space=output_space)
        img_sitk = nii_to_sitk(moving_img_nii)
        transformed_img = resampler.Execute(img_sitk)
        if moving_img_nii.seg:
            transformed_img = sitk.Round(transformed_img)
        return sitk_to_nii(transformed_img, seg=moving_img_nii.seg)

    def get_affine(self):
        # VersorRigid3DTransform
        # T(x) = A ( x - c ) + (t + c)
        # T(x) = Ax (- Ac + t + c)
        # let C = (- Ac + t + c)
        # (C^T*(Ax)^T)
        assert isinstance(self._transform, sitk.VersorRigid3DTransform)
        A = np.eye(4)  # noqa: N806
        v = self._transform.GetInverse()
        A[:3, :3] = np.array(v.GetMatrix()).reshape(3, 3)  # Rotation matrix
        c = np.array(v.GetCenter())  # Center of rotation
        t = np.array(v.GetTranslation())  # Translation vector
        trans = -A[:3, :3] @ c + c + t  # Correct translation formula
        A[:3, 3] = trans  # Set translation part
        return A

    def get_dump(self):
        return (
            1,  # version
            sitk_to_nii(self._img_moving, True).to_gird(),
            sitk_to_nii(self._img_fixed, True).to_gird(),
            self._transform,
            self.error_reg,
            self.error_natural,
            self.input_poi,
            self.out_poi,
        )

    def save(self, path: str | Path):
        with open(path, "wb") as w:
            pickle.dump(self.get_dump(), w)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as w:
            return cls.load_(pickle.load(w))

    @classmethod
    def load_(cls, w):
        self = cls.__new__(cls)
        (
            version,
            a,
            b,
            self._transform,
            self.error_reg,
            self.error_natural,
            self.input_poi,
            self.out_poi,
        ) = w
        a: Has_Grid
        b: Has_Grid
        self._img_fixed = nii_to_sitk(a.make_nii())
        self._img_moving = nii_to_sitk(b.make_nii())
        assert version == 1, f"Version mismatch {version=}"
        return self

    # def transform_nii_affine_only(self, moving_img_nii: NII, only_allow_grid_as_moving=True):
    #    if only_allow_grid_as_moving:
    #        text = (
    #            "input image must be in the same space as moving.  If you sure that this input is in same space as the moving image you can turn of 'only_allow_grid_as_moving'",
    #        )
    #        assert self.input_poi.shape == moving_img_nii.shape, (self.input_poi, moving_img_nii, text)
    #        assert self.input_poi.orientation == moving_img_nii.orientation, (self.input_poi, moving_img_nii, text)
    #        # moving_resampled = sitk.Resample(nii_to_sitk(moving_img_nii), self._img_fixed, self._transform, sitk.sitkLinear, 0.0)
    #    # moving_img_nii = moving_img_nii
    #    affine = self.get_affine() @ moving_img_nii.affine
    #    moving_img_nii = moving_img_nii.copy()
    #    moving_img_nii.affine = affine
    #    return moving_img_nii


def ridged_points_from_poi(
    poi_fixed: POI,
    poi_moving: POI,
    exclusion=None,
    log: Logger_Interface = No_Logger(),  # noqa: B008
    verbose=True,
    ax_code=None,
    zooms=None,
    c_val=None,
    leave_worst_percent_out=0.0,
) -> Point_Registration:
    if c_val is not None:
        warnings.warn(
            "c_val of ridged_points_from_poi is never used.",
            DeprecationWarning,
            stacklevel=4,
        )
    return Point_Registration(
        poi_fixed,
        poi_moving,
        exclusion=exclusion,
        log=log,  # noqa: B008
        verbose=verbose,
        ax_code=ax_code,
        zooms=zooms,
        leave_worst_percent_out=leave_worst_percent_out,
    )


def ridged_points_from_subreg_vert(
    poi_moving: POI_Reference,
    vert: Image_Reference,
    subreg: POI_Reference,
    poi_target_buffer: Path | str | None = None,
    orientation=None,
    zoom=(-1, -1, -1),
    subreg_id: int | Location | list[int | Location] | list[Location] | list[int] = 50,
    c_val=-1050,
    verbose=True,
    save_buffer_file=True,
) -> Point_Registration:
    if not isinstance(subreg_id, (list, tuple)):
        subreg_id = [subreg_id]
    instance_nii = to_nii(vert, True).copy()
    semantic_nii = to_nii(subreg, True).copy()
    target_poi = (
        calc_poi_from_subreg_vert(
            instance_nii, semantic_nii, subreg_id=subreg_id, buffer_file=poi_target_buffer, save_buffer_file=save_buffer_file
        )
        .copy()
        .extract_subregion_(*subreg_id)
    )
    if orientation is not None:
        target_poi.reorient_(orientation)
    if zoom != (-1, -1, -1):
        target_poi.rescale_(zoom)
    moving_poi = POI.load(poi_moving)
    return ridged_points_from_poi(target_poi, moving_poi, c_val=c_val, verbose=verbose)


def _compute_versor(
    inter: list[tuple[int, int]],
    ctd_f: POI,
    img_fixed: sitk.Image,
    ctd_m: POI,
    img_moving: sitk.Image,
    verbose=False,
    log: Logger_Interface = No_Logger(),  # noqa: B008
):
    assert len(inter) > 2, f"To few points: {inter}"
    # find shared points
    move_l = []
    fix_l = []
    # get real world coordinates of the corresponding vertebrae
    for k1, k2 in inter:
        ctr_mass_b = ctd_m[k1, k2]
        ctr_b = img_moving.TransformContinuousIndexToPhysicalPoint((ctr_mass_b[0], ctr_mass_b[1], ctr_mass_b[2]))
        move_l.append(ctr_b)
        ctr_mass_f = ctd_f[k1, k2]
        ctr_f = img_fixed.TransformContinuousIndexToPhysicalPoint((ctr_mass_f[0], ctr_mass_f[1], ctr_mass_f[2]))
        fix_l.append(ctr_f)
    log.print("[*] used POI:", inter, verbose=verbose)
    # Rough registration transform
    moving_image_points_flat = [c for p in move_l for c in p if not math.isnan(c)]
    fixed_image_points_flat = [c for p in fix_l for c in p if not math.isnan(c)]
    init_transform = sitk.VersorRigid3DTransform(
        sitk.LandmarkBasedTransformInitializer(sitk.VersorRigid3DTransform(), fixed_image_points_flat, moving_image_points_flat)
    )

    x_old = fix_l[0]
    y_old = move_l[0]
    error_reg = 0
    error_natural = 0
    err_count = 0
    err_count_n = 0
    log.print(
        f"{'key': <7}|{'fixed points': <23}|{'moved points after': <23}|{'moved points before': <23}|{'delta fixed/moved': <23}|{'distF': <5}|{'distM': <5}|",
        verbose=verbose,
    )
    k_old = -1000
    delta_after = {}
    for (k1, k2), x, y in zip_strict(inter, np.round(fix_l, decimals=1), np.round(move_l, decimals=1)):
        y2 = init_transform.GetInverse().TransformPoint(y)
        y = [round(m, ndigits=1) for m in y]  # noqa: PLW2901
        dif = [round(i - j, ndigits=1) for i, j in zip_strict(x, y2)]
        delta_after[(k1, k2)] = np.sum(np.array(dif) ** 2).item()
        dist = round(math.sqrt(sum([(i - j) ** 2 for i, j in zip_strict(x, x_old)])), ndigits=1)
        dist2 = round(math.sqrt(sum([(i - j) ** 2 for i, j in zip_strict(y, y_old)])), ndigits=1)
        error_reg += math.sqrt(sum([i * i for i in dif]))
        err_count += 1
        if k1 - k_old < 50:
            error_natural += abs(dist - dist2)
            err_count_n += 1
        else:
            dist = ""
            dist2 = ""
        if verbose:
            x_ = f"{x[0]:7.1f},{x[1]:7.1f},{x[2]:7.1f}"
            y_ = f"{y[0]:7.1f},{y[1]:7.1f},{y[2]:7.1f}"
            y2_ = f"{y2[0]:7.1f},{y2[1]:7.1f},{y2[2]:7.1f}"
            d_ = f"{dif[0]:7.1f},{dif[1]:7.1f},{dif[2]:7.1f}"
            log.print(f"{(k1, k2)!s: <7}|{x_: <23}|{y2_: <23}|{y_: <23}|{d_: <23}|{dist!s: <5}|{dist2!s: <5}|", verbose=verbose)

        x_old = x
        y_old = y
        k_old = k1
    error_reg /= max(err_count, 1)
    error_natural /= max(err_count_n, 1)
    log.print(f"Error avg registration error-vector length: {error_reg: 7.3f}", Log_Type.STAGE, verbose=verbose)
    log.print(f"Error avg point-distances: {error_natural: 7.3f}", Log_Type.STAGE, verbose=verbose)

    return init_transform, error_reg, error_natural, delta_after
