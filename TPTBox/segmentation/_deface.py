from pathlib import Path

import numpy as np

from TPTBox import BIDS_FILE, BIDS_Global_info, Image_Reference, to_nii
from TPTBox.core.nii_wrapper import NII
from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file, run_VibeSeg


def _compute_deface_mask_cta(ct_img: Image_Reference, outpath: str | Path | None = None, override=False, gpu=None, **args):
    """
    Mahmutoglu, M.A., Rastogi, A., Schell, M. et al. Deep learning-based defacing tool for CT angiography: CTA-DEFACE. Eur Radiol Exp 8, 111 (2024). https://doi.org/10.1186/s41747-024-00510-9

    """
    if isinstance(outpath, str):
        outpath = Path(outpath)
    if isinstance(ct_img, BIDS_FILE) and outpath is None:
        outpath = ct_img.get_changed_path("nii.gz", "msk", parent="derivatives-defacing", info={"seg": "defacting", "mod": ct_img.format})
    if outpath is not None and not override and outpath.exists():
        return outpath
    return run_VibeSeg(ct_img, out_path=outpath, dataset_id=1, keep_size=False, override=override, gpu=gpu, **args)


def _extend_mask_anterior(mask: NII, n: int) -> NII:
    """
    Extend a binary mask n voxels further in anterior (A) direction.

    Parameters
    ----------
    mask : NII
        Binary or label mask
    n : int
        Number of voxels to extend beyond the most anterior voxel

    Returns
    -------
    NII
        Extended mask
    """

    m = mask.copy()
    arr = m.extract_label(1).get_array()

    # Axis corresponding to Anterior-Posterior
    axis = m.get_axis("A")

    # Orientation: +1 index is "A" or "P"
    # Typically values are ("A","P") or ("P","A")
    ori = m.orientation[axis]

    # Indices where mask is non-zero
    idx = np.where(arr != 0)
    if idx[0].size == 0:
        return m  # empty mask → nothing to extend

    coords = idx[axis]

    if ori == "A":
        anterior_idx = coords.max()
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(0, 1)
        slice_signal = arr[tuple(slicer)]
        for i in range(min(arr.shape[axis], anterior_idx + n)):
            slicer[axis] = slice(i, i + 1)

            slice_signal = np.max(np.stack([slice_signal, arr[tuple(slicer)]], axis=0), axis=0)
            arr[tuple(slicer)] |= slice_signal

    else:
        anterior_idx = coords.min()
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(0, 1)
        slice_signal = arr[tuple(slicer)]
        for i in range(max(0, anterior_idx - n), 0, -1):
            slicer[axis] = slice(i, i - 1)

            slice_signal = np.max(np.stack([slice_signal, arr[tuple(slicer)]], axis=0), axis=0)
            arr[tuple(slicer)] |= slice_signal
    mask[arr == 1] = 1
    return mask


def compute_deface_mask_cta(
    ct_img: Image_Reference, outpath: str | Path | None = None, override=False, gpu=None, partially_defaced=False, **args
):
    """
    Mahmutoglu, M.A., Rastogi, A., Schell, M. et al. Deep learning-based defacing tool for CT angiography: CTA-DEFACE. Eur Radiol Exp 8, 111 (2024). https://doi.org/10.1186/s41747-024-00510-9

    """
    if isinstance(outpath, str):
        outpath = Path(outpath)
    if isinstance(ct_img, BIDS_FILE) and outpath is None:
        outpath = ct_img.get_changed_path("nii.gz", "msk", parent="derivatives-defacing", info={"seg": "defacting-2", "mod": ct_img.format})
    if outpath is not None and not override and outpath.exists():
        return outpath
    tight_mask = compute_deface_mask_cta(ct_img, None, override=override, gpu=gpu, **args)
    ct = to_nii(ct_img, False)
    face = to_nii(tight_mask, True).resample_from_to(ct)
    if not partially_defaced:
        face = face.filter_connected_components(max_count_component=1, keep_label=True)
    face_org = face.copy()
    f2 = face.dilate_msk(6).smooth_gaussian_labelwise(1, 5)
    f2[ct > -600] = 0
    f2 = f2.filter_connected_components(max_count_component=1, keep_label=True)

    mask = _extend_mask_anterior(f2, 4)
    mask[face_org == 1] = 2

    m2 = mask.extract_label(1)
    mask = mask.clamp(0, 1).fill_holes(1, "S")
    mask *= mask.clamp(0, 1).erode_msk(4)
    mask[m2 * mask] = 1
    if outpath is not None:
        mask.save(outpath)
    return mask


def deface_img(ct_ref: Image_Reference, face_mask: Image_Reference, min_value=-1024, ct_out: Path | str | None = None, to_int=True):
    fm = to_nii(face_mask, True)
    ct_filled = to_nii(ct_ref, True)
    assert ct_filled.shape == fm.shape, (ct_filled.shape, fm.shape)
    fm = fm.clamp(0, 1)
    ct_filled[fm] = min_value
    if to_int:
        # save storage space
        ct_filled = ct_filled.set_dtype("smallest_int")
    if ct_out is not None:
        ct_filled.save(ct_out)
    return ct_filled


if __name__ == "__main__":
    bgi = BIDS_Global_info("/DATA/NAS/datasets_processed/CT_spine/dataset-myelom")
    snps = []
    msk = []
    l = list(bgi.iter_subjects())
    import random

    random.shuffle(l)
    for _, subj in l:
        q = subj.new_query(flatten=True)
        q.filter_format(lambda x: "ct" in str(x))
        q.filter_filetype(["nii.gz", "nii", "nrrd", "mrk"])
        for ct_img in q.loop_list():
            outpath = ct_img.get_changed_path(
                "nii.gz", "msk", parent="derivatives-defacing", info={"seg": "defacting", "mod": ct_img.format}
            )

            print(outpath.name, outpath.exists())
            compute_deface_mask_cta(ct_img, gpu=5)
            snp = (
                ct_img.dataset
                / "derivatives-defacing"
                / "snapshots"
                / ct_img.get_changed_path("jpg", "msk", parent="derivatives-defacing", info={"seg": "defacting", "mod": ct_img.format}).name
            )
            msk.append(outpath)
            snps.append(snp)
    from TPTBox.mesh3D.snapshot3D import make_snapshot3D_parallel

    make_snapshot3D_parallel(msk, snps, ["A", "R", "S"])

    snps = []
    msk = []

    bgi = BIDS_Global_info(
        "/DATA/NAS/ongoing_projects/robert/datasets/Carotis-CoW-Projekt/Carotis-CoW-Projekt/CT_Datensatz_TUM_20250827/",
        parents=["CT_CAROTIS"],
    )
    l = list(bgi.iter_subjects())
    import random

    random.shuffle(l)
    for _, subj in l:
        q = subj.new_query(flatten=True)
        q.filter_format(lambda x: "ct" in str(x))
        q.filter_filetype(["nii.gz", "nii", "nrrd", "mrk"])
        for ct_img in q.loop_list():
            try:
                _compute_deface_mask_cta(ct_img, gpu=5)
                outpath = ct_img.get_changed_path(
                    "nii.gz", "msk", parent="derivatives-defacing", info={"seg": "defacting", "mod": ct_img.format}
                )
                snp = (
                    ct_img.dataset
                    / "derivatives-defacing"
                    / "snapshots"
                    / ct_img.get_changed_path(
                        "jpg", "msk", parent="derivatives-defacing", info={"seg": "defacting", "mod": ct_img.format}
                    ).name
                )
                msk.append(outpath)
                snps.append(snp)
            except Exception:
                pass
    from TPTBox.mesh3D.snapshot3D import make_snapshot3D_parallel

    make_snapshot3D_parallel(msk, snps, ["A", "R", "S"])
