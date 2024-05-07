"""
This script assumes that there are aligned Sagittal data and poorly aligned axial data.
"""

import pickle
from pathlib import Path

import nibabel as nib

from TPTBox import NII, BIDS_Global_info, No_Logger, to_nii
from TPTBox.core.bids_files import Subject_Container
from TPTBox.registration.ridged_intensity.register import only_change_affine, registrate_nipy
from TPTBox.stitching import stitching


def register_ax_and_stich_both(sub: Subject_Container, out_folder, buffer_path, verbose_stitching=True):
    with open(buffer_path, "rb") as f:
        registration_buffer = pickle.load(f) if Path(buffer_path).exists() else {}
    new_reg_buffer = {}
    q = sub.new_query(flatten=True)
    q.filter_non_existence("seg")
    q.filter_non_existence("lesions")
    sag_q = q.copy()
    sag_q.filter("acq", "sag")
    sag_q.filter("chunk", lambda _: True)
    q.filter("acq", "ax")
    sag_files = list(sag_q.loop_list())
    sessions = list({s.get("ses") for s in sag_files})
    for session in sessions:
        try:
            sag_q2 = sag_q.copy()
            sag_q2.filter("ses", str(session))
            sag_files = list(sag_q2.loop_list())
            ## MAKE Sagital Stich
            out_sag = sag_files[0].get_changed_path(parent=out_folder, info={"sequ": "stitched", "chunk": None})
            if not out_sag.exists() and len(sag_files) > 1:
                stitching(*sag_files, out=out_sag, bias_field=True, verbose_stitching=verbose_stitching)
            if len(sag_files) == 1:
                sag_files[0].open_nii().save(out_sag)
                # out_sag = sag_files[0].file["nii.gz"]
            q2 = q.copy()
            q2.filter("ses", str(session))
            ax_files = list(q2.loop_list())
            ## MAKE AXIAL Stich resituated on Sagittal
            out_ax = ax_files[0].get_changed_path(parent=out_folder, info={"sequ": "stitched", "chunk": None})
            out_ax_cord = ax_files[0].get_changed_path(
                parent=out_folder,
                info={"sequ": "stitched", "seg": "spinalcord", "chunk": None},
                bids_format="msk",
            )
            out_ax_ms = ax_files[0].get_changed_path(
                parent=out_folder,
                info={"sequ": "stitched", "label": "lesions", "chunk": None},
                bids_format="msk",
            )

            # out_ax2 = ax_files[0].get_changed_path(parent=out_folder, info={"sequ": "stitchedxxx", "chunk": None})
            if not out_ax.exists():
                reg_ax_files = []
                stitched_sag = to_nii(out_sag)
                for ax_f in ax_files:
                    ax_nii = to_nii(ax_f)
                    if ax_f.file["nii.gz"] in registration_buffer:
                        transform = registration_buffer[ax_f.file["nii.gz"]]
                    else:
                        a = ax_nii.reorient().rescale() / ax_nii.max()
                        b = stitched_sag.resample_from_to(a) / stitched_sag.max()
                        if b.sum() == 0:
                            b = stitched_sag.resample_from_to(a)  # prevent error, when we do not intersect
                        aligned_img, transform, out_arr = registrate_nipy(a, b, similarity="cc", optimizer="rigid")
                    new_reg_buffer[ax_f.file["nii.gz"]] = transform

                    aligned_img = only_change_affine(ax_nii, transform)
                    reg_ax_files.append(aligned_img)

                stitching(*reg_ax_files, out=out_ax, bias_field=True, verbose_stitching=verbose_stitching)
            ## MAKE SPINAL-CORD
            if not out_ax_cord.exists():
                reg_ax_files = []
                for ax_f in ax_files:
                    seg = ax_f.get_sequence_files(key_addendum=["seg"])["T2w_seg-manual"][0]
                    ax_nii = to_nii(seg, seg=True)
                    ax_nii.seg = True
                    if ax_f.file["nii.gz"] in new_reg_buffer:
                        transform = new_reg_buffer[ax_f.file["nii.gz"]]
                        aligned_img = only_change_affine(ax_nii, transform)
                    else:
                        raise FileNotFoundError(out_ax)  # noqa: TRY301
                        aff = to_nii(ax_f).affine
                        aligned_img = NII(nib.nifti1.Nifti1Image(ax_nii.get_array(), aff), ax_nii.seg)

                    reg_ax_files.append(aligned_img)

                stitching(*reg_ax_files, out=out_ax_cord)
            ## MAKE MS STICH
            if not out_ax_ms.exists():
                reg_ax_files = []
                for ax_f in ax_files:
                    seg = ax_f.get_sequence_files(key_addendum=["lesions"])["T2w_lesions-manual"][0]
                    ax_nii = to_nii(seg, seg=True)
                    ax_nii.seg = True
                    if ax_f.file["nii.gz"] in new_reg_buffer:
                        transform = new_reg_buffer[ax_f.file["nii.gz"]]
                        aligned_img = only_change_affine(ax_nii, transform)
                    else:
                        raise FileNotFoundError(out_ax)  # noqa: TRY301
                        aff = to_nii(ax_f).affine
                        aligned_img = NII(nib.nifti1.Nifti1Image(ax_nii.get_array(), aff), ax_nii.seg)

                    reg_ax_files.append(aligned_img)

                stitching(*reg_ax_files, out=out_ax_ms)

        except Exception:
            # raise e
            No_Logger().print_error()
            try:
                print(ax_files)
            except Exception:
                pass

    return new_reg_buffer


def save_registration_buffer(buffers: dict | list[dict], registration_buffer, buffer_path):
    if not isinstance(buffers, list):
        buffers = [buffers]
    assert buffers is not None
    for new_buffer in buffers:
        for key, value in new_buffer.items():
            registration_buffer[key] = value
    with open(buffer_path, "wb") as x:
        pickle.dump(registration_buffer, x)


def run(root="/media/data/robert/datasets/dataset-McGinnes/", out_folder="rawdata_new", n_jobs=1, chunks=16):
    bgi = BIDS_Global_info([root], parents=["rawdata"])
    registration_buffer = {}
    buffer_path = Path(root, "registration_affines.pkl")
    if n_jobs == 1:
        for _, sub in bgi.enumerate_subjects():
            new_buffer = register_ax_and_stich_both(sub, out_folder, buffer_path)
            save_registration_buffer(new_buffer, registration_buffer, buffer_path)
    else:
        from joblib import Parallel, delayed

        jobs = []
        for _, sub in bgi.enumerate_subjects():
            jobs.append(delayed(register_ax_and_stich_both)(sub, out_folder, buffer_path))
            if len(jobs) == chunks:
                list_buffers = Parallel(n_jobs=n_jobs)(jobs)
                assert list_buffers is not None
                save_registration_buffer(list_buffers, registration_buffer, buffer_path)
                jobs = []

        list_buffers = Parallel(n_jobs=n_jobs)(jobs)
        assert list_buffers is not None
        save_registration_buffer(list_buffers, registration_buffer, buffer_path)


if __name__ == "__main__":
    run(n_jobs=8, chunks=999999)
    # run(n_jobs=8)
