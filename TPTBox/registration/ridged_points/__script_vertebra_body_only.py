from __future__ import annotations

import sys
import time
from pathlib import Path

from TPTBox.core.bids_files import BIDS_Family

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

import math
import os
import secrets
import traceback
from typing import List

import numpy as np
import SimpleITK as sitk

from TPTBox import BIDS_FILE, BIDS_Global_info, POI_Reference, Searchquery, Subject_Container, calc_centroids_labeled_buffered


def crop_slice(msk):
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


from TPTBox.registration.ridged_points.point_registration import nii_to_iso_sitk_img, nii_to_iso_sitk_img2, reload_centroids


def ridged_point_registration(
    a_ctd: POI_Reference,
    a_list: list[BIDS_FILE],
    a_key: list[str],
    b_ctd: POI_Reference,
    b_list: list[BIDS_FILE],
    b_key: list[str],
    suppress_rotation_along_spine=True,
):
    # search for a t1,t2, dixon, ct file
    def find_representative(f_list, f_key):
        for i, key in enumerate(f_key):
            if "t1" in key or "t2" in key or "dixon" in key or "ct" in key:
                rep = f_list[i]
                f_list.remove(rep)
                f_key.remove(key)
                return rep
        print("No supported representative file")
        raise NotImplementedError()

    a_representative = find_representative(a_list, a_key)
    b_representative = find_representative(b_list, b_key)
    tmp = os.path.join(os.getcwd(), f"temp_{secrets.token_urlsafe(22)}")

    try:
        if not Path(tmp).exists():
            Path(tmp).mkdir()

        # Load 2 representative from A and B
        img_a_sitk, img_a_org, img_a_iso = nii_to_iso_sitk_img(a_representative, tmp)
        img_b_sitk, img_b_org, img_b_iso = nii_to_iso_sitk_img(b_representative, tmp)

        # get centroid correspondence of A and B
        # f_img is the related file

        ctd_a_iso = reload_centroids(a_ctd, img_a_org, img_a_iso)
        ctd_b_iso = reload_centroids(b_ctd, img_b_org, img_b_iso)

        # filter points by name
        f_unq = list(ctd_a_iso.keys())
        b_unq = list(ctd_b_iso.keys())
        # limit to only shared labels
        inter = np.intersect1d(b_unq, f_unq)
        if len(inter) < 1:
            # Skip if no intersection
            return

        # find shared points
        B_L = []
        F_L = []
        # get real world coordinates of the corresponding vertebrae
        for key in inter:
            ctr_mass_b = ctd_b_iso[key]
            ctr_b = img_b_sitk.TransformContinuousIndexToPhysicalPoint((ctr_mass_b[0], ctr_mass_b[1], ctr_mass_b[2]))
            B_L.append(ctr_b)
            if suppress_rotation_along_spine:
                ctr_b = img_b_sitk.TransformContinuousIndexToPhysicalPoint((ctr_mass_b[0], ctr_mass_b[1], ctr_mass_b[2] + 20))
                B_L.append(ctr_b)
            ctr_mass_f = ctd_a_iso[key]
            ctr_f = img_a_sitk.TransformContinuousIndexToPhysicalPoint((ctr_mass_f[0], ctr_mass_f[1], ctr_mass_f[2]))
            F_L.append(ctr_f)
            if suppress_rotation_along_spine:
                ctr_f = img_a_sitk.TransformContinuousIndexToPhysicalPoint((ctr_mass_f[0], ctr_mass_f[1], ctr_mass_f[2] + 20))
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
        # Crop the scans to the registered regions
        ex_slice_f, _ = crop_slice(sitk.GetArrayFromImage(img_a_sitk))
        ex_slice_m, _ = crop_slice(sitk.GetArrayFromImage(transformed_img))

        ex_slice = [slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(ex_slice_f, ex_slice_m, strict=False)]

        print(ex_slice_f, ex_slice_m, ex_slice)
        target_sequ = a_representative.info["sequ"]
        from_sequ = b_representative.info["sequ"]

        # Save registered file
        def register_and_save_file(img: sitk.Image, file: BIDS_FILE, target_space: bool):
            # print(f"[*] register {file.format}")
            if not target_space:
                img = resampler.Execute(img)
            img = img[ex_slice]
            if "e" in file.info:
                file.remove("e")

            info = file.info.copy()
            info["reg"] = from_sequ if target_space else target_sequ
            if file.format == "dixon":
                j = file.open_json()
                if "IP" in j["ImageType"]:
                    info["acq"] = "real"
                elif "W" in j["ImageType"]:
                    info["acq"] = "water"
                elif "F" in j["ImageType"]:
                    info["acq"] = "fat"
            file.info.clear()

            def pop(key):
                if key in info:
                    file.info[key] = info.pop(key)

            pop("sub")
            pop("ses")
            pop("sequ")
            pop("reg")
            pop("acq")
            for k, v in info.items():
                file.info[k] = v

            # sub-spinegan0042_ses-20220517_sequ-301_reg-None_acq-water_dixon.nii.gz

            out_file: Path = file.get_changed_path(
                file_type="nii.gz",
                parent="registration",
                path="{sub}/target-" + target_sequ + "_from-" + from_sequ,
                info=info,
                from_info=True,
            )
            if not out_file.exists():
                out_file.parent.mkdir(exist_ok=True, parents=True)
            if file.get_interpolation_order() == 0:
                img = sitk.Round(img)
            print(f"[#] saving {file.format}:\t{out_file.name}")
            sitk.WriteImage(img, str(out_file))

        # Single file from A
        register_and_save_file(img_a_sitk, a_representative, True)
        for bids in a_list:
            try:
                img = nii_to_iso_sitk_img2(bids, tmp)
                register_and_save_file(img, bids, True)
            except Exception:
                print(f"[!] Fail to register a sub_file, others will be registered \n\t{bids}\n\t {traceback.format_exc()!s}")

        # Single file from B
        register_and_save_file(img_b_sitk, b_representative, False)
        for bids in b_list:
            img = nii_to_iso_sitk_img2(bids, tmp)
            register_and_save_file(img, bids, False)

    except Exception:
        print("[!] Failed")
        print("\t", a_ctd)
        print("\t", b_ctd)
        print("\t", a_representative)
        print("\t", b_representative)

        print("\t", str(a_list).replace(", Format", "\n\tFormat"))
        print("\t", str(b_list).replace(", Format", "\n\tFormat"))
        print(str(traceback.format_exc()))
    finally:
        import shutil

        shutil.rmtree(tmp)


def _parallelized_preprocess_scan(subj_name, subject: Subject_Container, force_override_A=False):
    query1: Searchquery = subject.new_query()
    # It must exist a dixon and a msk
    query1.filter("format", "dixon")
    # A nii.gz must exist
    query1.filter("Filetype", "nii.gz")
    query1.filter("format", "msk")
    # query1.filter("sub", "spinegan0007")
    # query1.filter("sequ", "303")

    for dict_A in query1.loop_dict():
        if "ctd" not in dict_A or ("msk" in dict_A and force_override_A):
            assert "msk" in dict_A, "No centroid file"
            assert not isinstance(dict_A["msk"], list), f"{dict_A['msk']} contains more than one file"
            msk_bids: BIDS_FILE = dict_A["msk"][0]
            cdt_file: Path = msk_bids.get_changed_path(file_type="json", format="ctd", info={"seg": "subreg"}, parent="derivatives_msk")
            print(cdt_file)
            print(msk_bids.file["nii.gz"])
            # ctd = im.replace(f"_{mod}.nii.gz", "_seg-subreg_ctd.json").replace(a, "derivatives")
            calc_centroids_labeled_buffered(msk_bids.file["nii.gz"], out_path=cdt_file)
            bids_file = BIDS_FILE(cdt_file, msk_bids.dataset)
            subject.add(bids_file)
            # query1.candidates[""]["ctd"] = bids_file

    query2 = subject.new_query()
    # query2.filter("sub", "spinegan0007")

    # Only files with a seg-subreg + ctd file.
    query2.filter("format", "ctd")
    query2.filter("seg", "subreg", required=True)
    # It must exist a ct
    query2.filter("format", "ct")
    # query2.filter("sequ", lambda x: x == "303" or x == "300")
    # query2.filter("sub", lambda x: x == "303" or x == "300")
    # query2.filter("sequ", "203")
    print(query1)
    print(query2)
    for dict_A in query1.loop_dict():
        for dict_B in query2.loop_dict():

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

            a_list, a_key = extract_nii(dict_A)
            a_ctd = dict_A["ctd"]
            b_list, b_key = extract_nii(dict_B)
            b_ctd = dict_B["ctd"]

            print("##################################################################")
            print("[#] Processing")
            print(
                "[#] A: ",
                dict_A["ctd"][0].info["sub"],
                "sequ-" + dict_A["ctd"][0].info["sequ"],
                a_key,
            )
            print(
                "[#] B: ",
                dict_B["ctd"][0].info["sub"],
                "sequ-" + dict_B["ctd"][0].info["sequ"],
                b_key,
            )

            ridged_point_registration(a_ctd[0], a_list, a_key, b_ctd[0], b_list, b_key)


def parallel_execution(n_jobs, force_override_A=False):
    from joblib import Parallel, delayed

    global_info = BIDS_Global_info(
        ["/media/data/dataset-spinegan/register_test"],
        [
            "rawdata",
            "rawdata_ct",
            "rawdata_dixon",
            "derivatives",
            "derivatives_msk",
            "derivatives_spinalcord",
        ],  # "sourcedata"
    )
    print(f"Found {len(global_info.subjects)} subjects in {global_info.datasets}")

    if n_jobs > 1:
        print(f"[*] Running {n_jobs} parallel jobs. Note that stdout will not be sequential")

    Parallel(n_jobs=n_jobs)(
        delayed(_parallelized_preprocess_scan)(subj_name, subject, force_override_A)
        for subj_name, subject in global_info.enumerate_subjects()
    )


if __name__ == "__main__":
    a = ""
    # ridged_point_registration()
    parallel_execution(1)
