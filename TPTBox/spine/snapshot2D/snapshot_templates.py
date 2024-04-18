from pathlib import Path

import numpy as np

from TPTBox import BIDS_FILE, NII, POI, Image_Reference, POI_Reference
from TPTBox.spine.snapshot2D.snapshot_modular import Snapshot_Frame, Visualization_Type, create_snapshot


##########################
# Snapshot_Frame Templates
##########################
def mip_shot(
    ct_ref: BIDS_FILE,
    vert_msk: Image_Reference,
    subreg_ctd: POI_Reference,
    subreg_msk: Image_Reference = None,
    out_path=None,
):
    frames = [
        Snapshot_Frame(
            image=ct_ref,
            segmentation=vert_msk,
            centroids=subreg_ctd,
            mode="CT",
            sagittal=True,
            coronal=True,
            axial=False,
            crop_msk=True,
            cor_savgol_filter=False,
            # cmap="viridis",#ListedColormap(cmap),
        ),
        Snapshot_Frame(
            image=ct_ref,
            segmentation=vert_msk,
            centroids=subreg_ctd,
            mode="MINMAX",
            sagittal=True,
            coronal=True,
            axial=False,
            crop_msk=True,
            hide_segmentation=True,
            visualization_type=Visualization_Type.Mean_Intensity,
            only_mask_area=True,
        ),
        Snapshot_Frame(
            image=ct_ref,
            segmentation=vert_msk,
            centroids=subreg_ctd,
            mode="None",
            sagittal=False,
            coronal=True,
            axial=False,
            crop_msk=True,
            hide_segmentation=True,
            visualization_type=Visualization_Type.Maximum_Intensity_Colored_Depth,
            image_threshold=100,
            denoise_threshold=150,
            only_mask_area=False,
        ),
    ]

    if subreg_msk is not None:
        frames.append(
            Snapshot_Frame(
                image=ct_ref,
                segmentation=subreg_msk,
                centroids=subreg_ctd,
                mode="CT",
                sagittal=True,
                coronal=True,
                axial=False,
                crop_msk=True,
                cor_savgol_filter=False,
                # cmap="viridis",#ListedColormap(cmap),
            )
        )

    if out_path is None:
        out_path = ct_ref.get_changed_path(file_type="png", bids_format="snp", info={"desc": "mip"})
    create_snapshot(snp_path=[out_path], frames=frames)
    # print("[ ]saved snapshot into:", out_path)


def sacrum_shot(
    ct_ref: BIDS_FILE,
    vert_msk: Image_Reference,
    subreg_ctd: POI_Reference,
    add_ctd: list[POI_Reference] | None = None,
    mip_c: bool = False,
    out_path=None,
):
    if isinstance(ct_ref, BIDS_FILE):
        ct_ref = ct_ref.open_nii().reorient_()
    frames = [
        Snapshot_Frame(
            image=ct_ref,
            segmentation=vert_msk,
            centroids=subreg_ctd,
            mode="CT",
            sagittal=True,
            coronal=True,
            axial=False,
            crop_msk=True,
            cor_savgol_filter=False,
            # cmap="viridis",#ListedColormap(cmap),
        )
    ]
    if add_ctd is not None:
        for c in add_ctd:
            frames.append(  # noqa: PERF401
                Snapshot_Frame(
                    image=ct_ref,
                    segmentation=vert_msk,
                    centroids=c,
                    mode="CT",
                    sagittal=False,
                    coronal=True,
                    axial=False,
                    crop_msk=True,
                    cor_savgol_filter=False,
                    # cmap="viridis",#ListedColormap(cmap),
                )
            )

    if mip_c:
        frames.append(
            Snapshot_Frame(
                image=ct_ref,
                segmentation=vert_msk,
                centroids=subreg_ctd,
                mode="None",
                sagittal=False,
                coronal=True,
                axial=False,
                crop_msk=True,
                hide_segmentation=True,
                visualization_type=Visualization_Type.Maximum_Intensity_Colored_Depth,
                image_threshold=100,
                denoise_threshold=150,
                only_mask_area=False,
            )
        )
    if out_path is None:
        out_path = ct_ref.get_changed_path(file_type="png", bids_format="snp", info={"desc": "sacrum"})
    create_snapshot(snp_path=[out_path], frames=frames)
    # print("[ ]saved snapshot into:", out_path)


def spline_shot(
    ct_ref: BIDS_FILE,
    vert_msk: Image_Reference,
    subreg_ctd: POI_Reference,
    spline_nii: NII,
    add_info: str = "",
    out_path=None,
):
    frames = [
        Snapshot_Frame(
            image=ct_ref,
            segmentation=vert_msk,
            centroids=subreg_ctd,
            mode="CT",
            sagittal=True,
            axial=False,
            crop_msk=False,
        ),
        Snapshot_Frame(
            image=ct_ref,
            segmentation=spline_nii,
            centroids=subreg_ctd,
            mode="CT",
            sagittal=True,
            coronal=True,
            axial=False,
            crop_msk=False,
        ),
    ]
    if out_path is None:
        out_path = ct_ref.get_changed_path(
            file_type="png",
            bids_format="snp",
            info={"desc": f"spline-interpolation-{add_info}"},
        )
    create_snapshot(snp_path=[out_path], frames=frames)
    # print("[ ]saved snapshot into:", out_path)


def mri_snapshot(
    mrt_ref: BIDS_FILE,
    vert_msk: Image_Reference,
    subreg_ctd: POI_Reference,
    subreg_msk: Image_Reference = None,
    out_path: str | Path | list[str | Path] | list[Path] | None = None,
):
    frames = [
        Snapshot_Frame(
            image=mrt_ref,
            segmentation=vert_msk,
            centroids=subreg_ctd,
            mode="MRI",
            sagittal=True,
            coronal=True,
            axial=False,
            crop_msk=False,
            hide_segmentation=True,
        ),
        Snapshot_Frame(
            image=mrt_ref,
            segmentation=vert_msk,
            centroids=subreg_ctd,
            mode="MRI",
            sagittal=True,
            coronal=True,
            axial=False,
            crop_msk=False,
        ),
    ]
    if subreg_msk is not None:
        frames.append(
            Snapshot_Frame(
                image=mrt_ref,
                segmentation=subreg_msk,
                centroids=subreg_ctd,
                mode="MRI",
                sagittal=True,
                coronal=True,
                axial=False,
                crop_msk=False,
            )
        )
    if out_path is None:
        out_path = mrt_ref.get_changed_path(file_type="png", bids_format="snp", info={"desc": "vert"})
    if not isinstance(out_path, list):
        out_path = [out_path]
    create_snapshot(snp_path=out_path, frames=frames)
    return out_path
    # print("[ ]saved snapshot into:", out_path)


def vibe_snapshot(
    mrt_ref: tuple[BIDS_FILE, BIDS_FILE, BIDS_FILE, BIDS_FILE],
    vert_msk: Image_Reference | None,
    subreg_ctd: POI_Reference,
    subreg_msk: Image_Reference,
    hide_centroids: bool = False,
    out_path=None,
    verbose: bool = False,
):
    # print(type(mrt_ref[0]))
    frames = [
        Snapshot_Frame(
            image=mrt_ref[0],
            segmentation=subreg_msk,
            centroids=subreg_ctd,
            mode="MRI",
            sagittal=True,
            coronal=False,
            axial=False,
            crop_msk=False,
            hide_centroids=hide_centroids,
        ),
        Snapshot_Frame(
            image=mrt_ref[1],
            segmentation=subreg_msk,
            centroids=subreg_ctd,
            mode="MRI",
            sagittal=True,
            coronal=False,
            axial=False,
            crop_msk=False,
            hide_centroids=hide_centroids,
        ),
        Snapshot_Frame(
            image=mrt_ref[2],
            segmentation=subreg_msk,
            centroids=subreg_ctd,
            mode="MRI",
            sagittal=True,
            coronal=False,
            axial=False,
            crop_msk=False,
            hide_centroids=hide_centroids,
        ),
        Snapshot_Frame(
            image=mrt_ref[3],
            segmentation=subreg_msk,
            centroids=subreg_ctd,
            mode="MRI",
            sagittal=True,
            coronal=False,
            axial=False,
            crop_msk=False,
            hide_centroids=hide_centroids,
        ),
    ]
    if vert_msk is not None:
        frames.append(
            Snapshot_Frame(
                image=mrt_ref[0],
                segmentation=vert_msk,
                centroids=subreg_ctd,
                mode="MRI",
                sagittal=True,
                coronal=True,
                axial=False,
                crop_msk=False,
                hide_centroids=hide_centroids,
            )
        )
    if out_path is None:
        out_path = mrt_ref[0].get_changed_path(file_type="png", bids_format="snp", info={"desc": "vibe"})
    create_snapshot(snp_path=[out_path], frames=frames, verbose=verbose)
    return out_path


def ct_mri_snapshot(
    mrt_ref: Image_Reference,
    ct_ref: Image_Reference,
    vert_msk_mrt: Image_Reference | None = None,
    subreg_ctd_mrt: POI_Reference | None = None,
    vert_msk_ct: Image_Reference | None = None,
    subreg_ctd_ct: POI_Reference | None = None,
    out_path=None,
):
    frames = [
        Snapshot_Frame(
            image=mrt_ref,
            segmentation=vert_msk_mrt,
            centroids=subreg_ctd_mrt,
            mode="MRI",
            sagittal=True,
            coronal=True,
            axial=False,
            crop_msk=False,
        ),
        Snapshot_Frame(
            image=ct_ref,
            segmentation=vert_msk_ct,
            centroids=subreg_ctd_ct,
            mode="CT",
            sagittal=True,
            coronal=True,
            axial=False,
            crop_msk=False,
        ),
    ]
    if out_path is None:
        assert isinstance(mrt_ref, BIDS_FILE)
        out_path = mrt_ref.get_changed_path(file_type="png", bids_format="snp", info={"desc": "vert-ct-mri"})
    create_snapshot(snp_path=[out_path], frames=frames)
    return out_path


def poi_snapshot(
    ct_nii: BIDS_FILE,
    vert_msk: Image_Reference | None,
    subreg_ctd: POI_Reference,
    out_path=None,
):
    # conversion_poi = {
    #    "SSL": 81,  # this POI is not included in our POI list
    #    "ALL_CR_S": 109,
    #    "ALL_CR": 101,
    #    "ALL_CR_D": 117,
    #    "ALL_CA_S": 111,
    #    "ALL_CA": 103,
    #    "ALL_CA_D": 119,
    #    "PLL_CR_S": 110,
    #    "PLL_CR": 102,
    #    "PLL_CR_D": 118,
    #    "PLL_CA_S": 112,
    #    "PLL_CA": 104,
    #    "PLL_CA_D": 120,
    #    "FL_CR_S": 149,
    #    "FL_CR": 125,
    #    "FL_CR_D": 141,
    #    "FL_CA_S": 151,
    #    "FL_CA": 127,
    #    "FL_CA_D": 143,
    #    "ISL_CR": 134,
    #    "ISL_CA": 136,
    #    "ITL_S": 142,
    #    "ITL_D": 144,
    # }
    poi_all = POI.load(subreg_ctd)
    sinister = {}
    median = {}
    dorsal = {}
    all_ = {}
    pll = {}
    fl = {}
    # isl = {}
    # itl = {}
    from TPTBox.core.vert_constants import Location, conversion_poi2text

    for k, k2, v in poi_all.items():
        s = k2
        if conversion_poi2text[s].endswith("_D"):
            dorsal[k] = v
        elif conversion_poi2text[s].endswith("_S"):
            sinister[k] = v
        else:
            median[k] = v
        if "ALL" in conversion_poi2text[s]:
            all_[k] = v
        elif "PLL" in conversion_poi2text[s]:
            pll[k] = v
        elif "FL" in conversion_poi2text[s]:
            fl[k] = v

    frames = [
        Snapshot_Frame(
            image=ct_nii,
            segmentation=vert_msk,
            centroids=poi_all.copy(dorsal),
            mode="CT",
            curve_location=Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right,
        ),
        Snapshot_Frame(
            image=ct_nii,
            segmentation=vert_msk,
            centroids=poi_all.copy(median),
            mode="CT",
            curve_location=Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median,
        ),
        Snapshot_Frame(
            image=ct_nii,
            segmentation=vert_msk,
            centroids=poi_all.copy(sinister),
            mode="CT",
            curve_location=Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left,
        ),
        Snapshot_Frame(
            image=ct_nii,
            segmentation=vert_msk,
            centroids=poi_all.copy(all_),
            mode="CT",
            coronal=True,
            curve_location=Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median,
        ),
        Snapshot_Frame(
            image=ct_nii,
            segmentation=vert_msk,
            centroids=poi_all.copy(pll),
            mode="CT",
            coronal=True,
            curve_location=Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Median,
        ),
        Snapshot_Frame(
            image=ct_nii,
            segmentation=vert_msk,
            centroids=poi_all.copy(fl),
            mode="MINMAX",
            sagittal=True,
            coronal=False,
            hide_segmentation=True,
            only_mask_area=True,
            visualization_type=Visualization_Type.Mean_Intensity,
            curve_location=Location.Ligament_Attachment_Point_Flava_Superior_Median,
        ),
        Snapshot_Frame(
            image=ct_nii,
            segmentation=vert_msk,
            centroids=poi_all.copy(fl),
            mode="CT",
            sagittal=False,
            coronal=True,
            curve_location=Location.Ligament_Attachment_Point_Flava_Superior_Median,
        ),
    ]
    if out_path is None:
        out_path = ct_nii.get_changed_path(file_type="png", bids_format="snp", info={"desc": "poi"})
    create_snapshot(snp_path=[out_path], frames=frames)
    # print("[ ]saved snapshot into:", out_path)


if __name__ == "__main__":
    # ct_file = BIDS_FILE(
    #    "/media/data/robert/datasets/dataset-poi/derivatives/WS_31/ses-20221024/sub-WS_31_ses-20221024_seq-seriesdescription_space-aligASL_ct.nii.gz",
    #    "/media/data/robert/datasets/dataset-poi/",
    # )
    # f = "/media/data/robert/datasets/dataset-poi/derivatives/WS_31/ses-20221024/sub-WS_31_ses-20221024_test-robert_seg-subreg_msk.nii.gz"
    # g = f.replace("nii.gz", "json")
    #
    # poi_snapshot(ct_file, f, g)
    from pathlib import Path

    def make_snap(file):
        idx = file.stem
        if idx != "63":
            return
        # print(id)
        try:
            base_path = f"/media/data/robert/datasets/dataset-poi/derivatives/WS_{idx}"

            ct_file = BIDS_FILE(
                next(Path(base_path).glob("ses-*/sub-WS_*_ses-*_seq-*_space-aligASL_ct.nii.gz")),
                "/media/data/robert/datasets/dataset-poi/",
                verbose=False,
            )
            f = next(Path(base_path).glob("ses-*/sub-WS_*_ses-*_seq-*_seg-subreg_space-aligASL_msk.nii.gz"))
            # g = next(Path(base_path).glob("ses-*/sub-WS_*_ses-*_seq-*_seg-subreg_space-aligASL_msk.nii.gz"))
            poi_snapshot(
                ct_file,
                f,
                file,
                out_path=f"/media/data/robert/datasets/dataset-poi/snapshot/{idx}.png",
            )
        except StopIteration:
            print(idx, "stopIteration")

    from joblib import Parallel, delayed

    out = Parallel(n_jobs=16)(delayed(make_snap)(file) for file in Path("/media/data/robert/datasets/dataset-poi/poi/").iterdir())
