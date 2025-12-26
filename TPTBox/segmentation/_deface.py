from pathlib import Path

from TPTBox import BIDS_FILE, BIDS_Global_info, Image_Reference, to_nii
from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file, run_VibeSeg


def compute_deface_mask_cta(ct_img: Image_Reference, outpath: str | Path | None = None, override=False, gpu=None, **args):
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


if __name__ == "__main__":
    # bgi = BIDS_Global_info("/DATA/NAS/datasets_processed/CT_spine/dataset-myelom")
    # snps = []
    # msk = []
    # for sub, subj in bgi.iter_subjects():
    #    q = subj.new_query(flatten=True)
    #    q.filter_format(lambda x: "ct" in str(x))
    #    q.filter_filetype(["nii.gz", "nii", "nrrd", "mrk"])
    #    for ct_img in q.loop_list():
    #        compute_deface_mask_cta(ct_img, gpu=5)
    #        outpath = ct_img.get_changed_path(
    #            "nii.gz", "msk", parent="derivatives-defacing", info={"seg": "defacting", "mod": ct_img.format}
    #        )
    #        snp = (
    #            ct_img.dataset
    #            / "derivatives-defacing"
    #            / "snapshots"
    #            / ct_img.get_changed_path("jpg", "msk", parent="derivatives-defacing", info={"seg": "defacting", "mod": ct_img.format}).name
    #        )
    #        msk.append(outpath)
    #        snps.append(snp)
    # from TPTBox.mesh3D.snapshot3D import make_snapshot3D_parallel
    # make_snapshot3D_parallel(msk, snps, ["A", "R", "S"])
    snps = []
    msk = []

    bgi = BIDS_Global_info(
        "/DATA/NAS/ongoing_projects/robert/datasets/Carotis-CoW-Projekt/Carotis-CoW-Projekt/CT_Datensatz_TUM_20250827/",
        parents=["CT_CAROTIS"],
    )
    for _, subj in bgi.iter_subjects():
        q = subj.new_query(flatten=True)
        q.filter_format(lambda x: "ct" in str(x))
        q.filter_filetype(["nii.gz", "nii", "nrrd", "mrk"])
        for ct_img in q.loop_list():
            try:
                compute_deface_mask_cta(ct_img, gpu=5)
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
