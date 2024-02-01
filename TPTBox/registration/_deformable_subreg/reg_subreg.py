import os
import sys
from dataclasses import dataclass
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
import numpy as np
import SimpleITK as sitk
from joblib import Parallel, delayed

from TPTBox import NII, Centroids, Image_Reference, calc_centroids, load_centroids, to_nii_seg
from TPTBox.core import sitk_utils
from TPTBox.core.bids_files import BIDS_FILE
from TPTBox.core.sitk_utils import to_str_sitk
from TPTBox.core import sitk_utils
from joblib import Parallel, delayed
import nibabel as nib
from TPTBox import (
    NII,
    POI,
    load_poi,
    load_centroids,
    calc_centroids,
    Image_Reference,
    to_nii_seg,
)
from dataclasses import dataclass

# import dcmstack

"""
@author: Amir Bayat and Robert Graf

IMPORTANT NOTE:
This file does not implement the connected component filter.
"""


@dataclass()
class Atlas:
    vert_atlas: NII
    vert_subreg_atlas: NII | None = None
    poi_atlas: NII | POI | None = None


def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f}")


def register(
    fixed: sitk.Image,
    moving: sitk.Image,
    subreg,
    poi: POI | None,
    reg_type="rigid",
    verbose=False,
):
    assert sitk.GetArrayFromImage(moving).sum() != 0
    print(f"[*] Registration with {reg_type}-type.")
    old_stdout = sys.stdout  # backup current stdout
    old_stderr = sys.stdout
    if not verbose:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    try:
        if reg_type == "affine":
            maximumNumberOfFunctionEvaluations = 1000
        elif reg_type == "rigid":
            maximumNumberOfFunctionEvaluations = 100
        elif reg_type == "deformable":
            maximumNumberOfFunctionEvaluations = 20
        else:
            raise NotImplementedError(reg_type)

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        R.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-9,  # reset to 1e-7 for other vertebrae
            numberOfIterations=200,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=maximumNumberOfFunctionEvaluations,
            costFunctionConvergenceFactor=1e7,
        )
        if reg_type == "affine":
            R.SetInitialTransform(sitk.AffineTransform(fixed.GetDimension()))
            R.SetInterpolator(sitk.sitkNearestNeighbor)
        elif reg_type == "rigid":
            R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
            R.SetInterpolator(sitk.sitkNearestNeighbor)
        elif reg_type == "deformable":
            transformDomainMeshSize = [8] * moving.GetDimension()
            tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)
            R.SetInitialTransform(tx, True)
            R.SetInterpolator(sitk.sitkLinear)

        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

        outTx: sitk.Transform = R.Execute(fixed, moving)
        if poi is not None:
            poi = sitk_utils.transform_centroid(poi, outTx, img_fixed=fixed, img_moving=moving, reg_type=reg_type)
        else:
            poi = POI()
        # outTx.TransformPoint(point)
        print("-------")
        print(outTx)
        print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
        print(f" Iteration: {R.GetOptimizerIteration()}")
        print(f" Metric value: {R.GetMetricValue()}")
        # sys.stdout = open(os.devnull, "w")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(outTx)

        print(to_str_sitk(subreg)) if subreg is not None else None
        print(to_str_sitk(fixed))
        transformed_subreg: sitk.Image | None = resampler.Execute(subreg) if subreg is not None else None
        transformed_atlas: sitk.Image = resampler.Execute(moving)
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if (transformed_subreg is None or sitk.GetArrayFromImage(transformed_subreg).sum() == 0) and sitk.GetArrayFromImage(
            moving
        ).sum() == 0:
            exit("Failed registration")
        return fixed, transformed_atlas, transformed_subreg, poi
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        raise e


def _is_border(x, y, z, arr):
    for a, b, c in [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]:
        if any(
            [
                x + a < 0,
                y + b < 0,
                z + c < 0,
                x + a == arr.shape[0],
                y + b == arr.shape[1],
                z + c == arr.shape[2],
            ]
        ):
            continue
        if arr[x + a, y + b, z + c] != 0:
            return True
    return False


def _get_max_count(x, y, z, arr):
    lst = []
    for a, b, c in [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]:
        if any(
            [
                x + a < 0,
                y + b < 0,
                z + c < 0,
                x + a == arr.shape[0],
                y + b == arr.shape[1],
                z + c == arr.shape[2],
            ]
        ):
            continue
        if arr[x + a, y + b, z + c] != 0:
            lst.append(arr[x + a, y + b, z + c])
    return max(lst, key=lst.count)


def _breitensuche(x, y, z, arr, nii_target, lst: list):
    for a, b, c in [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]:
        if any(
            [
                x + a < 0,
                y + b < 0,
                z + c < 0,
                x + a == arr.shape[0],
                y + b == arr.shape[1],
                z + c == arr.shape[2],
            ]
        ):
            continue
        v1 = arr[x + a, y + b, z + c]
        v2 = nii_target[x + a, y + b, z + c]
        if v2 != 0 and v1 == 0:
            lst.append((x + a, y + b, z + c))


def post_processing(out_nii: NII, GT: NII):
    print("[ ] post_processing")
    out_arr = out_nii.get_array()
    GT_fixed_nii = GT.get_array()
    out_arr[GT_fixed_nii == 0] = 0
    out_arr = out_arr
    # arr = post_processing(arr,nii_target)
    tmp = np.zeros_like(out_arr)
    tmp2 = np.zeros_like(out_arr)
    tmp[GT_fixed_nii != 0] = 1
    tmp2[out_arr == 0] = 1
    tmp3 = tmp + tmp2
    tmp3[tmp3 != 2] = 0
    x, y, z = np.where(tmp3 == 2)
    seeds = []
    for x, y, z in zip(x, y, z):
        if _is_border(x, y, z, out_arr):
            seeds.append((x, y, z))
    next_seeds = []

    while len(seeds) != 0:
        for x, y, z in seeds:
            out_arr[x, y, z] = _get_max_count(x, y, z, out_arr)
        for x, y, z in seeds:
            _breitensuche(x, y, z, out_arr, GT_fixed_nii, next_seeds)
        del seeds
        seeds = list(set(next_seeds))
        next_seeds = []

    return out_nii.set_array_(out_arr)


def load_atlas_individual(atlas_path, vert_nii: NII, label: int):
    # Load atlas
    moving_name = Path(atlas_path, f"ATLAS/ATLAS_bodies_ASL/ATLAS_vert_{label}.nii.gz")
    moving_atlas = NII.load(moving_name, seg=True).rescale_and_reorient_(vert_nii.orientation, voxel_spacing=vert_nii.zoom)

    subreg_name = Path(
        atlas_path,
        f"ATLAS/ATLAS_subregions_ASL/ATLAS_{label:02}_subreg_corrected.nii.gz",
    )
    subreg_atlas = NII.load(subreg_name, seg=True).rescale_and_reorient_(vert_nii.orientation, voxel_spacing=vert_nii.zoom)

    poi_path = Path(atlas_path, f"ATLAS/ATLAS_poi_ASL/ATLAS_poi_{label}.json")
    poi_path2 = Path(atlas_path, f"ATLAS/ATLAS_poi_ASL/ATLAS_poi_{label}.nii.gz")
    if poi_path2 is not None and poi_path2.exists() and not poi_path.exists():
        ctd = calc_centroids(NII.load(poi_path2, True).reorient_(vert_nii.orientation))
        ctd.save(poi_path)
    poi = load_poi(poi_path) if poi_path.exists() else calc_centroids(subreg_atlas)

    return moving_atlas, subreg_atlas, poi


def load_atlas_file(atlas_path: Atlas, vert_nii: NII, label: int):
    mask = atlas_path.vert_atlas.extract_label(label)

    if atlas_path.poi_atlas is None:
        poi = POI(orientation=mask.orientation, centroids={}, zoom=mask.zoom, shape=mask.shape)
    elif isinstance(atlas_path.poi_atlas, NII):
        poi = calc_centroids(atlas_path.poi_atlas.apply_mask(mask))
    else:
        poi = atlas_path.poi_atlas.get_subset_by_lower_key(label)
        # TODO

    if atlas_path.vert_subreg_atlas is None:
        subreg = mask.multiply(label)
    else:
        subreg = atlas_path.vert_subreg_atlas.apply_mask(mask)
    return mask, subreg, poi


def __make_subreg_by_registration(
    label: float,
    vert_nii: NII,
    atlas_path: str | Path | None = None,
    vert_atlas: Atlas | None = None,
    reg_type="deformable",
    verbose=False,
):
    # TODO Why only the biggest connected component?
    label = int(label)
    # load anc crop fixed
    fixed_nii = vert_nii.extract_label(label)
    ex_slice_fixed = fixed_nii.compute_crop(minimum=0, dist=10)
    fixed_nii.apply_crop_(ex_slice_fixed)
    if atlas_path is not None:
        moving_atlas, subreg_atlas, poi = load_atlas_individual(atlas_path, vert_nii, label)
    else:
        assert vert_atlas is not None
        moving_atlas, subreg_atlas, poi = load_atlas_file(vert_atlas, vert_nii, label)

    # Crop Atlas
    ex_slice_moving = moving_atlas.compute_crop(dist=5)
    moving_atlas.apply_crop_(ex_slice_moving)
    subreg_atlas.apply_crop_(ex_slice_moving)
    poi.apply_crop_(ex_slice_moving)

    # To sitk; Set Origin zero, so they are close
    moving_atlas = sitk_utils.nii_to_sitk(moving_atlas)
    moving_atlas.SetOrigin([0, 0, 0])
    subreg_atlas = sitk_utils.nii_to_sitk(subreg_atlas)
    subreg_atlas.SetOrigin([0, 0, 0])
    poi.apply_crop_(tuple(x for x in subreg_atlas.GetOrigin()))

    fixed_sitk = sitk_utils.nii_to_sitk(fixed_nii)
    fixed_sitk.SetOrigin([0, 0, 0])

    # 3 stage Registration (points!)
    fixed_sitk, moving_atlas, subreg_atlas, poi = register(
        fixed_sitk,
        moving_atlas,
        subreg_atlas,
        poi,
        reg_type="rigid",
        verbose=verbose,
    )
    if reg_type == "deformable" or reg_type == "affine":
        fixed_sitk, moving_atlas, subreg_atlas, poi = register(
            fixed_sitk,
            moving_atlas,
            subreg_atlas,
            poi,
            reg_type="affine",
            verbose=verbose,
        )
    if reg_type == "deformable":
        fixed_sitk, moving_atlas, subreg_atlas, poi = register(
            fixed_sitk,
            moving_atlas,
            subreg_atlas,
            poi,
            reg_type="deformable",
            verbose=verbose,
        )
    poi.apply_crop_(tuple(-x.start for x in ex_slice_fixed)) if poi is not None else None
    poi.round_(3) if poi is not None else None
    assert subreg_atlas is not None
    out_nii = sitk_utils.sitk_to_nii(subreg_atlas, True)
    post_processing(out_nii, fixed_nii)
    return out_nii, poi, ex_slice_fixed


def make_subreg_by_registration(
    vert_file: Path | str,
    out_file: Path | str,
    ax_code=("L", "A", "S"),
    zoom=(1, 1, 1),
    atlas_path="/media/data/robert/datasets/",
    n_jobs=8,
):
    vert_nii = NII.load(vert_file, seg=True)
    vert_nii.rescale_(zoom)
    vert_nii = vert_nii.reorient_(ax_code, verbose=True)
    result_volume = vert_nii.get_seg_array() * 0

    out = Parallel(n_jobs=n_jobs)(
        delayed(__make_subreg_by_registration)(int(label), vert_nii, atlas_path)
        for label in list(np.unique(vert_nii.get_seg_array()))
        if label > 3 and label < 5
    )
    assert out is not None
    for x in out:
        transformed_mask, poi, ex_slice_fixed = x
        print(poi)
        result_volume[ex_slice_fixed] += transformed_mask.get_seg_array()

    vert_nii.set_array_(result_volume)
    print(vert_nii)
    vert_nii.save(out_file)


def make_atlas_from_sample(
    vert_file: Image_Reference,
    out_file: Path | str,
    vert_atlas_file: Image_Reference,
    vert_subreg_atlas_file: Image_Reference | None = None,
    poi_atlas_file: Path | str | None = None,
    zoom=(-1, -1, -1),
    n_jobs=4,
    save_files: bool = False,
):
    vert_nii = to_nii_seg(vert_file)
    zoom_old = vert_nii.zoom

    vert_nii.rescale_(zoom)
    zoom_new = vert_nii.zoom
    ax_code = vert_nii.orientation
    print(ax_code)
    # vert_nii.reorient_(ax_code, verbose=True)
    result_volume = vert_nii.get_seg_array() * 0

    vert_atlas = to_nii_seg(vert_atlas_file)
    vert_atlas.rescale_(zoom_new)
    vert_atlas.reorient_(ax_code, verbose=True)
    if vert_subreg_atlas_file is not None:
        subreg_atlas = to_nii_seg(vert_subreg_atlas_file)
        subreg_atlas.rescale_(zoom_new)
        subreg_atlas.reorient_(ax_code, verbose=True)
        assert (
            vert_atlas.shape == subreg_atlas.shape if subreg_atlas is not None else True
        ), f"vert_atlas.shape != subreg_atlas.shape; \n{vert_atlas}\n{subreg_atlas}\n{vert_subreg_atlas_file},\n{vert_atlas_file}"

    else:
        subreg_atlas = None

    if poi_atlas_file is None:
        poi = None
    elif str(poi_atlas_file).endswith(".json"):
        poi = load_centroids(poi_atlas_file)
        poi.rescale_(zoom_new)
        poi.reorient_(ax_code, verbose=True)
    else:
        poi = NII.load(poi_atlas_file, seg=True)
        poi.rescale_(zoom_new)
        poi.reorient_(ax_code, verbose=True)
    atlas = Atlas(vert_atlas, subreg_atlas, poi)
    atlas_id = list(np.unique(vert_atlas.get_seg_array()))
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    out = Parallel(n_jobs=n_jobs)(
        delayed(__make_subreg_by_registration)(int(label), vert_nii, None, atlas)
        for label in list(np.unique(vert_nii.get_seg_array()))
        if label in atlas_id and label != 0
    )
    sitk.ProcessObject_SetGlobalWarningDisplay(True)
    assert out is not None
    poi_out: POI = None  # type:ignore
    for x in out:
        transformed_mask: NII
        transformed_mask, poi, ex_slice_fixed = x
        if poi_out is None:
            poi_out = poi.copy()
            # poi_out.orientation = vert_nii.orientation
            poi_out.shape = result_volume.shape
        else:
            for k, v in poi.items():
                poi_out[k] = v
        print(poi)
        result_volume[ex_slice_fixed] += transformed_mask.get_seg_array()

    # result_volume = post_processing(result_volume, vert_nii.get_array())

    vert_nii.set_array_(result_volume)
    vert_nii.seg = True
    # vert_nii.set_array_(vert_nii.get_array())

    vert_nii.rescale_(zoom_old)
    poi_out.rescale_(zoom_old)
    poi_out.reorient_(vert_nii.orientation)
    if save_files:
        print(vert_nii)
        vert_nii.save(str(out_file).replace(".json", ".nii.gz"))
        poi_out.save(str(out_file).replace(".nii.gz", ".json"), save_hint=2)
    return vert_nii, poi_out


def make_snap_og():
    from TPTBox import Centroids
    from TPTBox.snapshot2D.snapshot_templates import poi_snapshot
    from TPTBox import POI

    def make_snap(file):
        id = file.stem
        print(id, end="\r")
        try:
            base_path = f"/media/data/robert/datasets/dataset-poi/derivatives/WS_{id}"

            ct_file = BIDS_FILE(
                next(Path(base_path.replace("derivatives", "rawdata")).glob("ses-*/sub-WS_*_ses-*_seq-*_ct.nii.gz")),
                "/media/data/robert/datasets/dataset-poi/",
                verbose=False,
            )
            f = next(Path(base_path).glob("ses-*/sub-WS_*_ses-*_seq-*_seg-subreg_msk.nii.gz"))
            poi = POI.load(file)
            poi.rescale_(ct_file.open_nii_reorient(poi.orientation).zoom)
            poi_snapshot(
                ct_file,
                f,
                poi,
                out_path=f"/media/data/robert/datasets/dataset-poi/snapshot/{id}.png",
            )
        except StopIteration:
            print(id, "stopIteration")

    from joblib import Parallel, delayed

    out = Parallel(n_jobs=16)(delayed(make_snap)(file) for file in Path("/media/data/robert/datasets/dataset-poi/poi/").iterdir())


if __name__ == "__main__":
    name = "/media/data/robert/datasets/dataset-poi/derivatives_new/WS-52/ses-20221028/sub-WS-52_ses-20221028_seq-seriesdescription_vert-poi_ref-WS-06_poi.json"
    poi = POI.load(name)
    poi.save(name.replace("poi.", "poitest"), save_hint=1)

    # make_snap_og()
    exit()
    # f = "/media/data/robert/datasets/spine_transition_new/derivatives/spinegan0130/ses-20200817/sub-spinegan0130_ses-20200817_sequ-4_seg-vert_msk.nii.gz"
    # cdt = "/media/data/robert/datasets/spine_transition_new/derivatives/spinegan0130/ses-20200817/sub-spinegan0130_ses-20200817_sequ-4_seg-vert_label-Vertebra-Corpus_ctd.json"
    # make_subreg_by_registration(f, "test.nii.gz")
    vert_file = "/media/data/robert/datasets/dataset-poi/derivatives/WS_31/ses-20221024/sub-WS_31_ses-20221024_seq-seriesdescription_seg-vert_space-aligASL_msk.nii.gz"

    out_file = (
        "/media/data/robert/datasets/dataset-poi/derivatives/WS_31/ses-20221024/sub-WS_31_ses-20221024_test-robert_seg-subreg_msk.nii.gz"
    )
    atlas_file = "/media/data/robert/datasets/dataset-poi/derivatives/WS_63/ses-20221028/sub-WS_63_ses-20221028_seq-seriesdescription_seg-vert_msk.nii.gz"
    subreg_file = "/media/data/robert/datasets/dataset-poi/derivatives/WS_63/ses-20221028/sub-WS_63_ses-20221028_seq-seriesdescription_seg-subreg_msk.nii.gz"
    poi_file = "/media/data/robert/datasets/dataset-poi/poi/63.json"
    from time import time

    st = time()
    make_atlas_from_sample(vert_file, out_file, atlas_file, subreg_file, poi_file)
    print(f"it took {time()-st:} seconds")
    from TPTBox.snapshot2D.snapshot_templates import poi_snapshot

    ct_file = BIDS_FILE(
        str(Path(vert_file).parent) + "/sub-WS_31_ses-20221024_seq-seriesdescription_space-aligASL_ct.nii.gz",
        vert_file.replace("rawdata", "derivatives").split("derivatives", maxsplit=1)[0],
        verbose=False,
    )
    poi_snapshot(ct_file, out_file, str(out_file).replace(".nii.gz", ".json"))
