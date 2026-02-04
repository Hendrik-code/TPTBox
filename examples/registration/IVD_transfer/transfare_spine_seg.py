from __future__ import annotations

import os
import pickle
import random
from collections.abc import Sequence

# Step 1
from pathlib import Path
from tempfile import gettempdir
from typing import Literal

import numpy as np
import torch
from deepali.losses import LNCC, MAE, BSplineBending, LandmarkPointDistance
from torch import Tensor

from TPTBox import POI, Image_Reference, POI_Global, Vertebra_Instance, calc_centroids, calc_poi_from_subreg_vert, to_nii
from TPTBox.core.internal.deep_learning_utils import DEVICES
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import POI
from TPTBox.registration.deformable.deformable_reg import Deformable_Registration

# from TPTBox.registration.deformable.deformable_reg_old import Deformable_Registration as Deformable_Registration_old
from TPTBox.registration.ridged_points import Point_Registration
from TPTBox.spine.snapshot2D import Snapshot_Frame, create_snapshot


class LandmarkPointDistance_with_cut_off(LandmarkPointDistance):
    r"""Average distance between corresponding landmarks."""

    def __init__(self, poi: POI | NII, scale: float = 1):
        r"""Initialize point distance loss.

        Args:
            scale: Constant factor by which to scale average point distance value
                such that magnitude is in similar range to other registration loss
                terms, i.e., image similarity losses.

        """
        super().__init__()
        self.cutoff = 1 / min(np.array(poi.shape) * np.array(poi.zoom)) * 10
        self.scale = float(scale)

    def forward(self, x: Tensor, *ys: Tensor) -> Tensor:
        r"""Evaluate point set distance."""
        if not ys:
            raise ValueError(f"{type(self).__name__}.forward() requires at least two point sets")
        x = x.float()
        loss = torch.tensor(0, dtype=x.dtype, device=x.device)
        for y in ys:
            dists: Tensor = torch.linalg.norm(x - y, ord=2, dim=2)
            loss += dists.mean()
        loss = loss / len(ys)
        if loss <= self.cutoff:
            loss = loss * 0
        return self.scale * loss

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


class Register_Segmentations:
    def __init__(
        self,
        target: NII,
        atlas: NII,
        poi_cms: POI | None,
        verbose=99,
        gpu=0,
        ddevice: DEVICES = "cuda",
        loss_terms=None,
        weights=None,
        lr: float | Sequence[float] = (0.01, 0.01, 0.007, 0.005, 0.005),
        max_steps=1500,
        min_delta=0.00001,
        pyramid_levels=5,
        coarsest_level=4,
        finest_level=0,
        gaussian_sigma=0,
        **args,
    ):
        # Assumes that you have removed the other leg.
        if weights is None:
            weights = {"be": [0.01, 0.001, 0.0001, 0.00001, 0.0000001], "mse": 1, "points": 1}
        if loss_terms is None:
            loss_terms = {"be": ("BSplineBending", {"stride": 1}), "mse": "MSE"}
        assert target.seg
        assert atlas.seg
        target = target.copy()
        atlas = atlas.copy()
        self.target_grid_org = target.to_gird()
        self.atlas_org = atlas.to_gird()
        # Point Registration
        # target = target.apply_pad(((20, 20), (20, 20), (20, 20)))
        poi_target = calc_centroids(target, second_stage=40)
        if poi_cms is None:
            poi_cms = calc_centroids(atlas, second_stage=40)  # This will be needlessly computed all the time
        elif not poi_cms.assert_affine(atlas, raise_error=False):
            poi_cms = poi_cms.resample_from_to(atlas)
        self.reg_point = Point_Registration(poi_target, poi_cms)
        atlas_reg = self.reg_point.transform_nii(atlas)
        # poi_cms_reg = self.reg_point.transform_poi(poi_cms)
        # Major Bones
        self.crop = (target + atlas_reg).compute_crop(0, 20)
        target = target.apply_crop(self.crop)
        atlas_reg = atlas_reg.apply_crop(self.crop)
        if gaussian_sigma > 0:
            target.seg = False
            target.set_dtype_(np.float32)
            atlas_reg.seg = False
            atlas_reg.set_dtype_(np.float32)
            target = target.smooth_gaussian(gaussian_sigma)
            atlas_reg = atlas_reg.smooth_gaussian(gaussian_sigma)
        self.target_grid = target.to_gird()

        self.reg_deform = Deformable_Registration(
            target,
            atlas_reg,
            # target_landmarks=poi_target,
            # source_landmarks=poi_cms_reg,
            loss_terms=loss_terms,  # type: ignore
            weights=weights,
            lr=lr,
            max_steps=max_steps,
            min_delta=min_delta,
            pyramid_levels=pyramid_levels,
            coarsest_level=coarsest_level,
            finest_level=finest_level,
            verbose=verbose,
            gpu=gpu,
            ddevice=ddevice,
            **args,
        )

    def get_dump(self):
        return (
            1,  # version
            (self.reg_point.get_dump()),
            (self.reg_deform.get_dump()),
            (self.atlas_org, self.target_grid_org, self.target_grid, self.crop),
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
        (version, t0, t1, t2, x) = w
        assert version == 1, f"Version mismatch {version=}"
        self = cls.__new__(cls)
        self.reg_point = Point_Registration.load_(t0)
        self.reg_deform = Deformable_Registration.load_(t1)
        self.atlas_org, self.target_grid_org, self.target_grid, self.crop = x
        return self

    def forward_nii(self, nii_atlas: NII):
        nii_atlas = self.reg_point.transform_nii(nii_atlas)
        nii_atlas = nii_atlas.apply_crop(self.crop)
        nii_reg = self.reg_deform.transform_nii(nii_atlas)
        nii_reg = nii_reg.resample_from_to(self.target_grid_org)
        return nii_reg

    def forward_poi(self, poi_atlas: POI_Global | POI):
        poi_atlas = poi_atlas.resample_from_to(self.atlas_org)
        # Point Reg
        poi_atlas = self.reg_point.transform_poi(poi_atlas)
        # Deformable
        poi_atlas = poi_atlas.apply_crop(self.crop)
        poi_reg = self.reg_deform.transform_poi(poi_atlas)
        poi_reg = poi_reg.resample_from_to(self.target_grid_org)


def compute_fake_ivd(vert_full: NII, subreg_full: NII, poi: POI, dilate=1):
    pass


def get_BWS_HWS_NUM(v: list[int]):
    bws: Literal[11, 12, 13] | None = None
    lws: Literal[4, 5, 6] | None = None
    if Vertebra_Instance.L1.value in v:
        if Vertebra_Instance.T13.value in v:
            bws = 13
        elif Vertebra_Instance.T12.value in v:
            bws = 12
        elif Vertebra_Instance.T11.value in v:
            bws = 11
    if Vertebra_Instance.S1.value in v:
        if Vertebra_Instance.L6.value in v:
            lws = 6
        elif Vertebra_Instance.L5.value in v:
            lws = 5
        elif Vertebra_Instance.L4.value not in v:
            lws = 4
    return bws, lws


def _add_BWS13(ref):
    """ONLY WORKS ON THE FITTED SAMPLE"""
    ref2 = ref.copy()
    p1 = 134
    p2 = 141
    cut_off = 610
    sift = 34
    p_delta = abs(p1 - p2)
    ref.map_labels_({19: Vertebra_Instance.T13.value, 119: Vertebra_Instance.T13.value + 100, 219: Vertebra_Instance.T13.value + 200})
    ref.remove_labels_([18, 118, 218])
    ref[:, :566, :] = 0
    ref2.remove_labels_([20])
    arr = ref.get_array()

    a = ref2[:-p_delta, sift:cut_off, :].get_array()
    arr[p_delta:, : cut_off - sift, :][a != 0] = a[a != 0]
    ref.set_array_(arr)


def _remove_BWS12(ref: NII):
    # Remove VERT 19
    ext = ref.extract_label([19, 118, 218])
    ref[ext != 0] = 0
    ref2 = ref.copy()
    i1 = 565
    i2 = 600
    p1 = 134
    p2 = 141
    cut_off = (i1 + i2) // 2
    ref[:, cut_off:, :] = 0
    ref2[:, :cut_off, :] = 0
    i_delta = abs(i1 - i2)
    p_delta = abs(p1 - p2)
    a = ref2[:-p_delta, i1:970, :].get_array()
    arr = ref.get_array()
    arr[p_delta:, i1 - i_delta : 970 - i_delta, :][a != 0] = a[a != 0]
    arr2 = arr.copy() * 0
    arr2[:-p_delta, i_delta:, :] = arr[p_delta:, :-i_delta, :]
    ref.set_array_(arr2)
    ref.map_labels_({219: 218, 119: 118}, verbose=True)


def _remove_LWS6(ref):
    # Remove 1 LWS
    sacrum = ref.extract_label(26)
    ref.remove_labels_([26, 25, 125, 225])
    s2 = sacrum.copy() * 0
    p_delta = 25
    i_delta = 40
    s2[:-p_delta, :-i_delta, :] = sacrum[p_delta:, i_delta:, :]
    ref[np.logical_and(ref == 0, s2 == 1)] = 26


def _remove_LWS6_and_5(ref):
    sacrum = ref.extract_label(26)
    ref.remove_labels_([26, 25, 125, 225])
    s2 = sacrum.copy() * 0
    p_delta = 25
    i_delta = 40
    s2[:-p_delta, :-i_delta, :] = sacrum[p_delta:, i_delta:, :]
    ref[np.logical_and(ref == 0, s2 == 1)] = 26
    ref.remove_labels_([22, 122, 222])
    ref2 = ref.copy()
    cut_off = 700
    ref[:, cut_off:, :] = 0
    ref2[:, :cut_off, :] = 0
    p_delta = 6
    i_delta = 50
    a = ref2[:-p_delta, cut_off:970, :].get_array()
    arr = ref.get_array()
    b = arr[p_delta:, cut_off - i_delta : 970 - i_delta, :]
    arr[p_delta:, cut_off - i_delta : 970 - i_delta, :][b == 0] = a[b == 0]
    ref.set_array_(arr)


def _get_template(vert: NII):
    from TPTBox.segmentation.VibeSeg.auto_download import _download

    tmp = Path(os.path.join(gettempdir(), "spine-templates"))
    tmp.mkdir(exist_ok=True)
    sub_ref = tmp / "spine.nii.gz"
    if not sub_ref.exists():
        _download("https://github.com/Hendrik-code/TPTBox/releases/download/v0.2.4/spine.nii.gz", sub_ref, is_zip=False)
    vert_ref = tmp / "vert.nii.gz"
    if not vert_ref.exists():
        _download("https://github.com/Hendrik-code/TPTBox/releases/download/v0.2.4/vert.nii.gz", vert_ref, is_zip=False)

    ref = to_nii(vert_ref, True)
    ref_org = ref.copy()
    ref2 = to_nii(sub_ref, True)
    ref[ref2 == 53] -= 200
    ref[ref2 == 52] -= 199
    ref[ref2 >= 100] -= 0
    print(ref)
    assert ref.orientation == ("P", "I", "R"), ref.orientation
    assert ref.shape == (256, 970, 27), ref

    bws, lws = get_BWS_HWS_NUM(vert.unique())
    if bws is None:
        bws = 12
    if lws is None:
        lws = 6
    if bws == 12 and lws == 6:
        return ref, ref_org
    buffer_file = tmp / f"bws{bws}_lws{lws}.nii.gz"
    buffer_file_org = tmp / f"bws{bws}_lws{lws}_org.nii.gz"
    if buffer_file.exists() and buffer_file_org.exists():
        return to_nii(buffer_file, True), to_nii(buffer_file_org, True)

    os.makedirs(tmp, exist_ok=True)
    if bws == 13:
        _add_BWS13(ref)
        _add_BWS13(ref_org)
    elif bws == 11:
        _remove_BWS12(ref)
        _remove_BWS12(ref_org)
    if lws == 5:
        _remove_LWS6(ref)
        _remove_LWS6(ref_org)
    elif lws == 4:
        _remove_LWS6_and_5(ref)
        _remove_LWS6_and_5(ref_org)
    # Save_template
    ref.save(buffer_file)
    ref_org.save(buffer_file_org)
    return ref, ref_org


def _register_IVD_to_ct(vert_ct: NII, vert_ref: NII, vert_ref_org: NII, start=2, end=30, factor=1.0, factor2=1.0):
    ids = [*range(start, end + 1), *[i + 100 for i in range(start, end + 1)]]
    try:
        crop = vert_ct.compute_crop(0, 20)
    except ValueError:
        return vert_ct * 0
    vert_ct_ = vert_ct.apply_crop(crop)
    lr: float | Sequence[float] = tuple(i * factor2 for i in (0.01, 0.01, 0.007, 0.005, 0.005))
    weights = {"be": [i * factor for i in [0.01, 0.001, 0.0001, 0.00001, 0.0000001]], "mse": 1}
    loss_terms = {"be": ("BSplineBending", {"stride": 1}), "mse": "MSE"}
    vert = vert_ct_.extract_label(ids, keep_label=True)
    if vert.sum() == 0:
        return vert_ct * 0
    try:
        reg = Register_Segmentations(
            vert, vert_ref.extract_label(ids, keep_label=True), None, lr=lr, weights=weights, loss_terms=loss_terms, gaussian_sigma=2
        )
    except ValueError as e:
        print(e)
        return vert_ct * 0
    ids = [*range(start, end), *[i + 100 for i in range(start, end)], *[i + 200 for i in range(start, end)]]
    return reg.forward_nii(vert_ref_org.extract_label(ids, keep_label=True)).resample_from_to(vert_ct)


def register_IVD_to_ct(vert: Image_Reference):
    vert_ct = to_nii(vert, True)
    vert_ref, vert_ref_org = _get_template(vert_ct)
    # Remove non_matching
    u1 = vert_ref.unique()
    u2 = vert_ct.unique()

    vert_ref.map_labels_({a: 0 for a in u1 if a not in u2 and a + 1 not in u2}, verbose=False)
    vert_ref.map_labels_({a: a + 100 for a in u1 if a % 2 == 1}, verbose=False)
    vert_ct.map_labels_({1: 0}, verbose=False)
    vert_ct.map_labels_({a: a + 100 for a in u1 if a % 2 == 1}, verbose=False)

    print(vert_ref.unique())
    u = vert_ct.unique()
    print(u)
    a = _register_IVD_to_ct(vert_ct, vert_ref, vert_ref_org, start=2, end=10, factor=1, factor2=1)
    b = _register_IVD_to_ct(vert_ct, vert_ref, vert_ref_org, start=10)

    a[a == 0] = b[a == 0]
    return a


j = 0
if __name__ == "__main__":
    for r in [
        Path("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/source/dataset-shockroom-without-fx"),
        Path("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/source/dataset-myelom/"),
    ]:
        x = list(r.iterdir())
        random.shuffle(x)
        for root in x:
            try:
                snp_path_filter = (
                    Path("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/source/snapshots")
                    / (r.name)
                    / "checked"
                    / (root.name + "_full_bone.jpg")
                )
                if not snp_path_filter.exists():
                    continue
                # if root.name != "sub-C0AGY465_ses-20101221_sequ-8_ct":
                #    continue
                print(root.name)
                out = root / "totalspineseg-vert-postprocessed.nii.gz"
                final_out = root / "new_10_2.nii.gz"
                if (root / "totalspineseg-vert-postprocessed3.nii.gz").exists():
                    final_out.unlink(missing_ok=True)
                    (root / "totalspineseg-vert-postprocessed3.nii.gz").unlink(missing_ok=True)
                snp_path = (
                    Path("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/source/snapshots")
                    / (r.name + "-spine")
                    / (root.name + ".jpg")
                )
                if not (root / "totalspineseg-vert.nii.gz").exists():
                    continue

                if snp_path.exists():
                    continue
                j += 1
                # continue
                spine = root / "spine.nii.gz"
                if spine.exists():
                    subreg = to_nii(spine, True)
                else:
                    subreg = to_nii(root / "new_99.nii.gz", True)
                    subreg = subreg.extract_label([69, 70], keep_label=True).map_labels({69: 49, 70: 46})
                if not out.exists():
                    vert_ct = to_nii(root / "totalspineseg-vert.nii.gz", True)
                    assert subreg.shape == vert_ct.shape, (subreg.shape, vert_ct.shape)
                    vert_ct2 = vert_ct * subreg.clamp(0, 1)
                    vert_ct2[vert_ct2 >= 30] = 0
                    vert_ct2[vert_ct == 26] = 26
                    vert_ct = vert_ct2.infect(subreg, inplace=True)
                    vert_ct.save(out)
                else:
                    vert_ct = to_nii(out, True)
                out = root / "totalspineseg-vert-postprocessed2.nii.gz"
                if not out.exists():
                    base = register_IVD_to_ct(vert_ct)
                    base.save(out)
                else:
                    base = to_nii(out, True)
                base[base <= 100] = 0
                base[base >= 200] -= 100
                base2 = to_nii(root / "totalspineseg-vert.nii.gz", True)
                base[base2 == 60] = 60
                base[base2 == 61] = 61
                base2[base2 <= 100] = 0
                base2[base2 >= 200] = 0
                base[base == 0] = base2[base == 0]
                # base[subreg != 0] = 1
                # t = base.clamp(0, 1).fill_holes_(slice_wise_dim=1).fill_holes_(slice_wise_dim=2).fill_holes_(slice_wise_dim=0).fill_holes_()
                # t[subreg != 0] = 0
                base[subreg != 0] = 0
                # base = base.infect(t, inplace=True)
                fin99 = to_nii(root / "new_99.nii.gz", True)
                # base[fin99 == 69] = 0
                ct = to_nii(root / "ct_0.nii.gz")
                base[ct >= 500] = 0
                ivd = root / "totalspineseg-ivd-postprocessed.nii.gz"

                base.save(ivd)
                ivd_x = base

                a = {i + 100: 68 for i in range(1, 50)}
                base_99 = base.map_labels({60: 52, 61: 0, **a})
                fin99[base_99 != 0] = base_99[base_99 != 0]
                fin99_path = root / "new_99_2.nii.gz"

                fin99.save(fin99_path)
                b = {i + 100: 29 for i in range(1, 50)}
                base.map_labels_({60: 0, 61: 0, **a})
                try:
                    fin = to_nii(root / "new_10.nii.gz", True)
                    fin[base != 0] = base[base != 0]
                except Exception:
                    fin = None
                poi_path = root / "poi.json"
                poi = calc_poi_from_subreg_vert(root / "totalspineseg-vert-postprocessed2.nii.gz", subreg, subreg_id=50)
                poi.save(poi_path)

                snp_path.parent.mkdir(exist_ok=True, parents=True)
                if not poi_path.exists():
                    poi_path = None
                print("save", snp_path)
                create_snapshot(
                    snp_path,
                    [
                        Snapshot_Frame(ct, fin if fin is not None else root / "full_bone.nii.gz", poi_path),
                        Snapshot_Frame(ct, ivd_x, poi_path),
                        Snapshot_Frame(ct, subreg, poi_path),
                        Snapshot_Frame(ct, fin99, poi_path, coronal=True),
                    ],
                )
                if fin is not None:
                    fin.save(final_out)
                # exit()
                # vert_ct = to_nii(root / "sub-spinegan0007_ses-20210910_sequ-206_mod-ct_seg-vert_msk.nii.gz", True)
            except Exception:
                pass
                # raise

print(j)
