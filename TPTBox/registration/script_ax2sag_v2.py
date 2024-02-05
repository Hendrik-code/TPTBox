"""
This script assumes that there are aligned Sagittal data and poorly aligned axial data.
"""
import pickle
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib

from TPTBox import BIDS_FILE, NII, BIDS_Global_info, Log_Type, No_Logger, to_nii
from TPTBox.core.bids_files import Subject_Container
from TPTBox.core.sitk_utils import affine_registration_transform
from TPTBox.registration.ridged_intensity.register import only_change_affine, registrate_ants, registrate_nipy
from TPTBox.stitching import stitching

logger = No_Logger()

from nipy.algorithms.registration.affine import Affine


def accuracy(current_best_img: NII, other: NII):
    current_best_img = current_best_img.resample_from_to(other, verbose=False)
    return (current_best_img * other).sum() / current_best_img.sum()


@dataclass
class current_best:
    acc_max: float
    current_best_img: NII
    current_best_spinal_cord: NII
    T: None | Affine = None

    def update(self, T: Affine | None, ax_spinalcord, T2w_sag_spinalcord, ax_img):
        if T is None:
            acc_new = accuracy(ax_spinalcord, T2w_sag_spinalcord)
            if acc_new > self.acc_max:
                # self.acc_max = acc_new
                # self.current_best_img = only_change_affine(ax_img, T)
                # self.current_best_spinal_cord = aligned_sc
                logger.print("new Best", "accuracy", self.acc_max, Log_Type.OK)
            else:
                logger.print("not new Best", "accuracy", acc_new, Log_Type.STRANGE)
        else:
            aligned_sc = only_change_affine(ax_spinalcord, T)
            acc_new = accuracy(aligned_sc, T2w_sag_spinalcord)
            if acc_new > self.acc_max:
                self.acc_max = acc_new
                self.current_best_img = only_change_affine(ax_img, T)
                self.current_best_spinal_cord = aligned_sc
                logger.print("new Best", "accuracy", self.acc_max, Log_Type.OK)
            else:
                logger.print("not new Best", "accuracy", acc_new, Log_Type.STRANGE)


def find_best_fit(
    ax_files: list[BIDS_FILE],
    T2w_sag: NII,
    T2w_sag_vert: NII,
    T2w_sag_spinalcord: NII,
    registration_area: NII,
    registration_buffer,
    new_reg_buffer,
    out_ax: Path,
    out_ax_cord: Path,
    out_ax_ms: Path,
    verbose_stitching,
    override=True,
):
    similarity = "dpmi"
    optimizer = "rigid"
    if not out_ax.exists() or override:
        reg_ax_files = []
        for ax_f in ax_files:
            # Start single registration of an axial to the sagittal
            ax_img = to_nii(ax_f)
            ax_spinalcord = to_nii(ax_f.get_sequence_files(key_addendum=["seg"])["T2w_seg-manual"][0], True)
            ax_spinalcord.seg = True
            # STEP 1 Test without a registration
            acc_max = accuracy(ax_spinalcord, T2w_sag_spinalcord)
            best = current_best(acc_max, ax_img, ax_spinalcord)
            logger.print("initial", "accuracy", best.acc_max, Log_Type.STRANGE)
            # STEP 2 Previous reg
            if ax_f.file["nii.gz"] in registration_buffer:
                logger.print("STEP 2", Log_Type.Yellow)
                T = registration_buffer[ax_f.file["nii.gz"]]
                best.update(T, ax_spinalcord, T2w_sag_spinalcord, ax_img)

            logger.print("STEP 8 ants", Log_Type.Yellow)
            import ants

            a = ax_spinalcord.reorient().rescale() / ax_spinalcord.max()
            b = T2w_sag_spinalcord.resample_from_to(a, verbose=False) / T2w_sag_spinalcord.max()
            out, trans = registrate_ants(a, b, similarity=similarity, optimizer=optimizer)
            out = ants.apply_transforms(b.to_ants(), T2w_sag_spinalcord.to_ants(), trans)

            best.update(None, NII(ants.to_nibabel(out), True), T2w_sag_spinalcord, ax_img)
            """logger.print("STEP 3", Log_Type.Yellow)
            smooth = 0.2
            order = -1
            a = ax_img.reorient().rescale().smooth_gaussian(smooth, nth_derivative=order) / ax_img.max()
            b = T2w_sag.resample_from_to(a, verbose=False).smooth_gaussian(smooth, nth_derivative=order) / T2w_sag.max()
            _, T, _ = registrate_nipy(a, b, similarity=similarity, optimizer=optimizer)
            best.update(T, ax_spinalcord, T2w_sag_spinalcord, ax_img)

            # STEP 5 Registration with limited view
            logger.print("STEP 4", Log_Type.Yellow)

            a = ax_img.reorient().rescale().smooth_gaussian(smooth, nth_derivative=order) / ax_img.max()
            scp = ax_spinalcord.resample_from_to(a, verbose=False).dilate_msk(5)
            scp.seg = False
            b = registration_area.resample_from_to(a, verbose=False).smooth_gaussian(smooth, nth_derivative=order) / T2w_sag.max()
            _, T, _ = registrate_nipy(a * scp, b, similarity=similarity, optimizer=optimizer)
            best.update(T, ax_spinalcord, T2w_sag_spinalcord, ax_img)

            # STEP 5 Registration with limited view
            logger.print("STEP 5", Log_Type.Yellow)

            a = ax_img.reorient().rescale() / ax_img.max()
            scp = ax_spinalcord.resample_from_to(a, verbose=False).dilate_msk(15)
            scp.seg = False
            b = registration_area.resample_from_to(a, verbose=False) / T2w_sag.max()
            _, T, _ = registrate_nipy(a * scp, b, similarity=similarity, optimizer=optimizer)
            best.update(T, ax_spinalcord, T2w_sag_spinalcord, ax_img)

            logger.print("STEP 6", Log_Type.Yellow)

            a = ax_img.reorient().rescale() / ax_img.max()
            scp = ax_spinalcord.resample_from_to(a, verbose=False).dilate_msk(30)
            scp.seg = False
            b = registration_area.resample_from_to(a, verbose=False) / T2w_sag.max()
            _, T, _ = registrate_nipy(a * scp, b, similarity=similarity, optimizer=optimizer)
            best.update(T, ax_spinalcord, T2w_sag_spinalcord, ax_img)

            logger.print("STEP 8", Log_Type.Yellow)

            a = ax_spinalcord.reorient().rescale()
            b = T2w_sag_spinalcord.resample_from_to(a, verbose=False)
            _, T, _ = registrate_nipy(a, b, similarity=similarity, optimizer=optimizer)
            best.update(T, ax_spinalcord, T2w_sag_spinalcord, ax_img)"""

            # else:
            #    a = ax_nii.reorient().rescale() / ax_nii.max()
            #    b = stitched_sag.resample_from_to(a) / stitched_sag.max()
            #    if b.sum() == 0:
            #        b = stitched_sag.resample_from_to(a)  # prevent error, when we do not intersect
            #    aligned_img, T, out_arr = registrate_nipy(a, b, similarity=similarity, optimizer=optimizer)
            # new_reg_buffer[ax_f.file["nii.gz"]] = T
            #
            # aligned_img = only_change_affine(ax_nii, T)
            # compute acc
            logger.print("accuracy", best.acc_max, Log_Type.STRANGE)
            # reg_ax_files.append(current_best_img)
        exit()
        stitching(*reg_ax_files, out=out_ax, bias_field=True, verbose_stitching=verbose_stitching)
    ## MAKE SPINAL-CORD
    # if not out_ax_cord.exists():
    #    reg_ax_files = []
    #    for ax_f in ax_files:
    #        seg = ax_f.get_sequence_files(key_addendum=["seg"])["T2w_seg-manual"][0]
    #        ax_nii = to_nii(seg, seg=True)
    #        ax_nii.seg = True
    #        if ax_f.file["nii.gz"] in new_reg_buffer:
    #            T = new_reg_buffer[ax_f.file["nii.gz"]]
    #            aligned_img = only_change_affine(ax_nii, T)
    #        else:
    #            assert False, out_ax
    #            aff = to_nii(ax_f).affine
    #            aligned_img = NII(nib.nifti1.Nifti1Image(ax_nii.get_array(), aff), ax_nii.seg)
    #
    #        reg_ax_files.append(aligned_img)
    #
    #    stitching(*reg_ax_files, out=out_ax_cord)
    ### MAKE MS STICH
    # if not out_ax_ms.exists():
    #    reg_ax_files = []
    #    for ax_f in ax_files:
    #        seg = ax_f.get_sequence_files(key_addendum=["lesions"])["T2w_lesions-manual"][0]
    #        ax_nii = to_nii(seg, seg=True)
    #        ax_nii.seg = True
    #        if ax_f.file["nii.gz"] in new_reg_buffer:
    #            T = new_reg_buffer[ax_f.file["nii.gz"]]
    #            aligned_img = only_change_affine(ax_nii, T)
    #        else:
    #            assert False, out_ax
    #            aff = to_nii(ax_f).affine
    #            aligned_img = NII(nib.nifti1.Nifti1Image(ax_nii.get_array(), aff), ax_nii.seg)
    #
    #        reg_ax_files.append(aligned_img)
    #
    #    stitching(*reg_ax_files, out=out_ax_ms)


def register_ax_and_stich_both(sub: Subject_Container, out_folder, buffer_path, verbose_stitching=True):
    if Path(buffer_path).exists():
        registration_buffer = pickle.load(open(buffer_path, "rb"))
    else:
        registration_buffer = {}
    new_reg_buffer = {}
    q = sub.new_query(flatten=True)
    q.filter_non_existence("seg")
    q.filter_non_existence("lesions")
    sag_q = q.copy()
    sag_q.filter("acq", "sag")
    sag_q.filter("chunk", lambda x: True)
    q.filter("acq", "ax")
    sag_files = list(sag_q.loop_list())
    sessions = list(set([s.get("ses") for s in sag_files]))
    for session in sessions:
        print(sub.name, session)
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

            # Get stitched sag T2w with seg.(generated by an other script...)
            sag_q_2 = sub.new_query(flatten=True)
            sag_q_2.filter_non_existence("lesions")
            sag_q_2.filter("ses", str(session))
            sag_q_2.filter("acq", "sag")
            sag_q_2.filter_filetype("nii.gz")
            sag_q_2.filter("sequ", "stitched")
            sag_q_2.unflatten()
            l_sag = list(sag_q_2.loop_dict())
            assert len(l_sag) == 1
            T2w_sag = l_sag[0]["T2w"][0].open_nii()
            T2w_sag_vert = l_sag[0]["msk_seg-spine"][0].open_nii()
            # T2w_sag_subreg = l_sag[0]["msk_seg-vert"][0].open_nii()
            T2w_sag_spinalcord = T2w_sag_vert.extract_label(61)
            registration_area = T2w_sag_vert.extract_label(60)
            registration_area += T2w_sag_spinalcord
            registration_area.dilate_msk_(2)
            registration_area = T2w_sag * registration_area
            # registration_area.save(l_sag[0]["msk_seg-spine"][0].get_changed_path(info={"seg": "registration-area"}))
            q2 = q.copy()
            q2.filter("ses", str(session))
            ax_files = list(q2.loop_list())
            ## MAKE AXIAL Stich resituated on Sagittal
            out_ax = ax_files[0].get_changed_path(parent=out_folder, info={"sequ": "stitched", "chunk": None})
            out_ax_cord = ax_files[0].get_changed_path(
                parent=out_folder,
                info={"sequ": "stitched", "seg": "spinalcord", "chunk": None},
                format="msk",
            )
            out_ax_ms = ax_files[0].get_changed_path(
                parent=out_folder,
                info={"sequ": "stitched", "label": "lesions", "chunk": None},
                format="msk",
            )

            find_best_fit(
                ax_files,
                T2w_sag,
                T2w_sag_vert,
                T2w_sag_spinalcord,
                registration_area,
                registration_buffer,
                new_reg_buffer,
                out_ax,
                out_ax_cord,
                out_ax_ms,
                verbose_stitching,
            )

        except Exception:
            # raise e
            No_Logger().print_error()
            try:
                print(ax_files)
            except:
                pass

    return new_reg_buffer


def save_registration_buffer(buffers: dict | list[dict], registration_buffer, buffer_path):
    if not isinstance(buffers, list):
        buffers = [buffers]
    assert buffers is not None
    for new_buffer in buffers:
        for key, value in new_buffer.items():
            registration_buffer[key] = value
    pickle.dump(registration_buffer, open(buffer_path, "wb"))


def run(root="/DATA/NAS/ongoing_projects/robert/dataset-neuropoly2/", out_folder="rawdata_new", n_jobs=1, chunks=16):
    bgi = BIDS_Global_info([root], parents=["rawdata", "rawdata_new", "derivatives_seg"])
    registration_buffer = {}
    buffer_path = Path(root, "registration_affines.pkl")
    if n_jobs == 1:
        for _, sub in bgi.enumerate_subjects():
            new_buffer = register_ax_and_stich_both(sub, out_folder, buffer_path)
            save_registration_buffer(new_buffer, registration_buffer, buffer_path)
            break
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
    run(n_jobs=1, chunks=999999)
    # run(n_jobs=8)
