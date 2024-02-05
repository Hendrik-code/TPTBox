from __future__ import annotations

import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

import math
import os
import secrets
import sys
import traceback
from typing import List

import SimpleITK as sitk
from sitk_utils import pad_same

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

# sys.path.append("..")
import numpy as np

from TPTBox import BIDS_FILE, BIDS_Family, BIDS_Global_info, Subject_Container, calc_centroids_labeled_buffered


def crop_slice(msk, dist=20):
    shp = msk.shape
    zms = [1, 1, 1]
    d = np.around(dist / np.asarray(zms)).astype(int)

    cor_msk = np.where(msk > 0)
    c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
    c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
    x0 = c_min[0]
    y0 = c_min[1]
    z0 = c_min[2]
    x1 = c_max[0]
    y1 = c_max[1]
    z1 = c_max[2]
    ex_slice = tuple([slice(z0, z1), slice(y0, y1), slice(x0, x1)])
    origin_shift = tuple([x0, y0, z0])
    return ex_slice, origin_shift


from TPTBox.registration.ridged_points.point_registration import nii_to_iso_sitk_img, reload_centroids


def ridged_point_registration(ctds: list[Path | BIDS_FILE | dict[str, BIDS_FILE]], files: list[list[BIDS_FILE]], keys: list[list[str]]):
    for idx_i in range(len(ctds)):
        for idx_j in range(len(ctds)):
            if idx_i == idx_j:
                continue
            tmp = os.path.join(os.getcwd(), f"temp_{secrets.token_urlsafe(22)}")
            try:
                if not Path(tmp).exists():
                    Path(tmp).mkdir()

                # Load 2 representative from A and B
                img_a_sitk, img_a_org, img_a_iso = nii_to_iso_sitk_img(files[idx_i][0], tmp)
                img_b_sitk, img_b_org, img_b_iso = nii_to_iso_sitk_img(files[idx_j][0], tmp)
                # filter = sitk.ConstantPadImageFilter()
                # filter.SetPadLowerBound([50, 50, 50])
                # filter.SetPadUpperBound([50, 50, 50])
                # filter.SetConstant(-1000)
                # img_a_sitk = filter.Execute(img_a_sitk)
                # get centroid correspondence of A and B
                ctd_a_iso = reload_centroids(ctds[idx_i], img_a_org, img_a_iso)
                ctd_b_iso = reload_centroids(ctds[idx_j], img_b_org, img_b_iso)

                # filter points by name
                f_unq = list(ctd_a_iso.keys())
                b_unq = list(ctd_b_iso.keys())
                # limit to only shared labels
                inter = np.intersect1d(b_unq, f_unq)
                if len(inter) < 1:
                    # Skip if no intersection
                    continue

                # find shared points
                B_L = []
                F_L = []
                # get real world coordinates of the corresponding vertebrae
                for key in inter:
                    ctr_mass_b = ctd_b_iso[key]
                    ctr_b = img_b_sitk.TransformContinuousIndexToPhysicalPoint((ctr_mass_b[0], ctr_mass_b[1], ctr_mass_b[2]))
                    B_L.append(ctr_b)
                    ctr_mass_f = ctd_a_iso[key]
                    ctr_f = img_a_sitk.TransformContinuousIndexToPhysicalPoint((ctr_mass_f[0], ctr_mass_f[1], ctr_mass_f[2]))
                    F_L.append(ctr_f)

                # Rough registration transform
                moving_image_points_flat = [c for p in B_L for c in p if not math.isnan(c)]
                fixed_image_points_flat = [c for p in F_L for c in p if not math.isnan(c)]
                init_transform = sitk.VersorRigid3DTransform(
                    sitk.LandmarkBasedTransformInitializer(sitk.VersorRigid3DTransform(), fixed_image_points_flat, moving_image_points_flat)
                )
                initial_transform = sitk.TranslationTransform(img_a_sitk.GetDimension())
                initial_transform.SetParameters(init_transform.GetTranslation())

                resampler = sitk.ResampleImageFilter()

                resampler.SetReferenceImage(img_a_sitk)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetTransform(init_transform)

                transformed_img = resampler.Execute(img_b_sitk)
                merged_img = sitk.Add(transformed_img, img_a_sitk)
                # merged_img = resample_shared_space(transformed_img, img_a_sitk, verbose=True)
                # Crop the scans to the registered regions
                # ex_slice_m, _ = crop_slice(sitk.GetArrayFromImage(img_a_sitk))
                # ex_slice, _ = crop_slice(sitk.GetArrayFromImage(transformed_img))

                target_sequ = files[idx_i][0].info["sequ"]
                from_sequ = files[idx_j][0].info["sequ"]

                # Save registered file
                def register_and_save_file(img, file: BIDS_FILE, target_space: bool):
                    # print(f"[*] register {file.format}")
                    if not target_space:
                        img = resampler.Execute(img)
                    # img = img[ex_slice_m]
                    # img = img[ex_slice]
                    out_file: Path = file.get_changed_path(
                        file_type="nii.gz",
                        parent="stitching",
                        path="{sub}/sequ-" + from_sequ,
                        info={"reg": from_sequ if target_space else target_sequ},
                    )
                    if not out_file.exists():
                        out_file.parent.mkdir(exist_ok=True, parents=True)

                    print(f"[#] saving {file.format}: {out_file.name}")
                    sitk.WriteImage(img, str(out_file))

                # Single file from A
                register_and_save_file(merged_img, files[idx_i][0], True)
                # for bids in a_list[1:]:
                #    try:
                #        img = nii_to_iso_sitk_img(bids, return_inter=False)
                #        register_and_save_file(img, bids, True)
                #    except Exception as e:
                #        print(f"[!] Fail to register a sub_file, others will be registered \n\t{bids}\n\t {str(traceback.format_exc())}")

                # Single file from B
                # register_and_save_file(img_b_sitk, b_list[0], False, resampler)
                # for bids in b_list[1:]:
                #    img = nii_to_iso_sitk_img(bids, return_inter=False)
                #    register_and_save_file(img, bids, False)

            except BaseException:
                print(f"[!] Failed \n\t{ctds}\n\t{files}\n\t{keys}\n\t {traceback.format_exc()!s}")

            finally:
                import shutil

                shutil.rmtree(tmp)


def extract_nii(d: BIDS_Family):
    out = []
    keys = []
    for k, v in d.items():
        if isinstance(v, list):
            for i, l in enumerate(v):
                if "nii.gz" in l.file:
                    out.append(l)
                    keys.append(f"{k}_{i}")
        elif "nii.gz" in v.file:
            out.append(v)
            keys.append(k)
    return out, keys


def _parallelized_preprocess_scan(typ, subject: Subject_Container, force_override_A=False):
    query = subject.new_query()
    if typ == "dixon":
        # It must exist a dixon and a msk
        query.filter("format", "dixon")
        # A nii.gz must exist
        query.filter("Filetype", "nii.gz")
        query.filter("format", "msk")

        for dict_A in query.loop_dict():
            if "ctd" not in dict_A or ("msk" in dict_A and force_override_A):
                assert "msk" in dict_A, "No centroid file"
                assert not isinstance(dict_A["msk"], list), f"{dict_A['msk']} contains more than one file"
                msk_bids: BIDS_FILE = dict_A["msk"][0]
                cdt_file: Path = msk_bids.get_changed_path(file_type="json", format="ctd", info={"seg": "subreg"})
                print(cdt_file)
                print(msk_bids.file["nii.gz"])
                # ctd = im.replace(f"_{mod}.nii.gz", "_seg-subreg_ctd.json").replace(a, "derivatives")
                calc_centroids_labeled_buffered(msk_bids.file["nii.gz"], out_path=cdt_file)
    elif typ == "ct":
        query = subject.new_query()
        # Only files with a seg-subreg + ctd file.
        query.filter("format", "ctd")
        query.filter("seg", "subreg", required=True)
        # It must exist a ct
        query.filter("format", "ct")
        # query.filter("sub", "spinegan0042")  # TODO REMOVE ME
        # query2.filter("sequ", "203")
    ctds = []
    niis = []
    keys = []
    for dict_A in query.loop_dict():
        a_list, a_key = extract_nii(dict_A)
        a_ctd = dict_A["ctd"]
        ctds.append(a_ctd)
        niis.append(a_list)
        keys.append(a_key)
    print(keys)
    ridged_point_registration(ctds, niis, keys)


def parallel_execution(n_jobs, force_override_A=False):
    from joblib import Parallel, delayed

    global_info = BIDS_Global_info(
        ["/media/robert/Expansion/dataset-Testset"],
        ["sourcedata", "rawdata", "rawdata_ct", "rawdata_dixon", "derivatives"],
    )
    print(f"Found {len(global_info.subjects)} subjects in {global_info.datasets}")

    if n_jobs > 1:
        print(f"[*] Running {n_jobs} parallel jobs. Note that stdout will not be sequential")

    Parallel(n_jobs=n_jobs)(
        delayed(_parallelized_preprocess_scan)("dixon", subject, force_override_A)
        for subj_name, subject in global_info.enumerate_subjects()
    )


if __name__ == "__main__":
    a = ""
    # ridged_point_registration()
    parallel_execution(8)
