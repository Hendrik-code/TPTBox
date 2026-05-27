from __future__ import annotations

from pathlib import Path

from TPTBox import BIDS_FILE, NII, POI, Image_Reference, POI_Reference
from TPTBox.spine.snapshot2D.snapshot_modular import Snapshot_Frame, Visualization_Type, create_snapshot


##########################
# Snapshot_Frame Templates
##########################
def mip_shot(
    ct_ref: BIDS_FILE,
    vert_msk: Image_Reference,
    subreg_ctd: POI_Reference,
    subreg_msk: Image_Reference | None = None,
    out_path: str | Path | None = None,
) -> None:
    """Save a multi-panel CT snapshot combining slice, mean-intensity, and coloured-depth MIP views.

    Three frames are always included:

    1. Standard CT slice view with vertebra segmentation and centroids
       (sagittal + coronal).
    2. Mean-intensity projection of the masked vertebra region only
       (sagittal + coronal, no segmentation overlay).
    3. Coloured-depth maximum-intensity projection of the bone above a 100 HU
       threshold (coronal only).

    An optional fourth frame with the subregion mask is appended when
    ``subreg_msk`` is provided.

    Args:
        ct_ref: BIDS file pointing to the CT image.
        vert_msk: Vertebra segmentation mask.
        subreg_ctd: Subregion centroids (POI reference).
        subreg_msk: Optional subregion segmentation mask for an extra frame.
        out_path: Output file path.  When ``None`` the path is derived
            automatically from ``ct_ref`` with ``desc=mip``.
    """
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
    out_path: str | Path | None = None,
) -> None:
    """Save a CT snapshot focused on the sacrum region.

    A primary sagittal + coronal frame is always created.  Additional coronal
    frames are appended for each extra centroid file in ``add_ctd``.  When
    ``mip_c`` is ``True`` a coloured-depth MIP coronal frame is added.

    Args:
        ct_ref: BIDS file pointing to the CT image.  If an instance of
            :class:`~TPTBox.BIDS_FILE`, the image is pre-loaded and
            reoriented.
        vert_msk: Vertebra segmentation mask.
        subreg_ctd: Primary subregion centroids (POI reference).
        add_ctd: Optional list of additional centroid references, each
            rendered as an extra coronal frame.
        mip_c: When ``True`` a coloured-depth maximum-intensity projection
            frame is appended.
        out_path: Output file path.  When ``None`` the path is derived
            automatically from ``ct_ref`` with ``desc=sacrum``.
    """
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
    out_path: str | Path | None = None,
) -> None:
    """Save a two-frame CT snapshot showing the spinal spline interpolation result.

    Frame 1 shows the original vertebra mask overlaid on the CT (sagittal
    only).  Frame 2 shows the spline NIfTI segmentation overlaid on the CT
    (sagittal + coronal) so that the spline quality can be assessed visually.

    Args:
        ct_ref: BIDS file pointing to the CT image.
        vert_msk: Vertebra segmentation mask.
        subreg_ctd: Subregion centroids (POI reference).
        spline_nii: NIfTI image containing the fitted spline segmentation.
        add_info: Optional string appended to the ``desc`` field of the
            auto-generated output path.
        out_path: Output file path.  When ``None`` the path is derived
            automatically from ``ct_ref``.
    """
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


def snapshot(
    ref: Image_Reference,
    vert_msk: Image_Reference,
    subreg_ctd: POI_Reference,
    subreg_msk: Image_Reference | None = None,
    out_path: str | Path | list[str | Path] | list[Path] | None = None,
    mode: str = "MINMAX",
    crop: bool = False,
) -> list[str | Path]:
    """Save a standard vertebra snapshot, auto-detecting CT vs. MRI mode.

    Delegates to :func:`mri_snapshot` after setting ``mode`` to ``"CT"`` for
    BIDS CT files or ``"MRI"`` for all other BIDS files.

    Args:
        ref: Source image reference.  A :class:`~TPTBox.BIDS_FILE` whose
            ``bids_format`` is ``"ct"`` is treated as CT; everything else
            is treated as MRI.
        vert_msk: Vertebra segmentation mask.
        subreg_ctd: Subregion centroids (POI reference).
        subreg_msk: Optional subregion segmentation mask for an extra frame.
        out_path: Output path(s).  ``None`` triggers automatic path derivation.
        mode: Intensity window mode passed to :class:`~.Snapshot_Frame` when
            ``ref`` is not a :class:`~TPTBox.BIDS_FILE`.
        crop: Whether to crop the output to the segmentation bounding box.

    Returns:
        List of saved output paths.
    """
    if isinstance(ref, BIDS_FILE):
        mode = "CT" if ref.bids_format == "ct" else "MRI"

    return mri_snapshot(ref, vert_msk, subreg_ctd, subreg_msk, out_path, mode, crop=crop)  # type: ignore


def mri_snapshot(
    mrt_ref: BIDS_FILE,
    vert_msk: Image_Reference,
    subreg_ctd: POI_Reference,
    subreg_msk: Image_Reference | None = None,
    out_path: str | Path | list[str | Path] | list[Path] | None = None,
    mode: str = "MRI",
    crop: bool = False,
) -> list[str | Path]:
    """Save a two-frame (or three-frame) MRI vertebra snapshot.

    Frame 1: image with centroids but without segmentation overlay (to
    check centroid placement without clutter).
    Frame 2: image with both centroids and segmentation overlay.
    Frame 3 (optional): image with the subregion mask overlay, added when
    ``subreg_msk`` is provided.

    Args:
        mrt_ref: BIDS file pointing to the MRI image.
        vert_msk: Vertebra segmentation mask.
        subreg_ctd: Subregion centroids (POI reference).
        subreg_msk: Optional subregion segmentation mask for an extra frame.
        out_path: Output path(s).  When ``None`` the path is derived
            automatically from ``mrt_ref`` with ``desc=vert``.
        mode: Intensity window mode; defaults to ``"MRI"``.
        crop: Whether to crop output frames to the segmentation bounding box.

    Returns:
        List of saved output paths.
    """
    frames = [
        Snapshot_Frame(
            image=mrt_ref,
            segmentation=vert_msk,
            centroids=subreg_ctd,
            mode=mode,
            sagittal=True,
            coronal=True,
            axial=False,
            crop_msk=crop,
            hide_segmentation=True,
        ),
        Snapshot_Frame(
            image=mrt_ref,
            segmentation=vert_msk,
            centroids=subreg_ctd,
            mode=mode,
            sagittal=True,
            coronal=True,
            axial=False,
            crop_msk=crop,
        ),
    ]
    if subreg_msk is not None:
        frames.append(
            Snapshot_Frame(
                image=mrt_ref,
                segmentation=subreg_msk,
                centroids=subreg_ctd,
                mode=mode,
                sagittal=True,
                coronal=True,
                axial=False,
                crop_msk=crop,
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
    out_path: str | Path | None = None,
    verbose: bool = False,
) -> str | Path:
    """Save a VIBE multi-contrast MRI snapshot with four Dixon phases.

    Creates one sagittal frame for each of the four VIBE contrast phases in
    ``mrt_ref``, all with the subregion mask overlay.  An optional fifth
    frame shows the first phase in sagittal + coronal with the vertebra mask.

    Args:
        mrt_ref: Four-tuple of BIDS files corresponding to the VIBE Dixon
            phases (e.g. in-phase, opposed-phase, water, fat).
        vert_msk: Optional vertebra segmentation mask used for an extra
            sagittal + coronal frame.  Skipped when ``None``.
        subreg_ctd: Subregion centroids (POI reference).
        subreg_msk: Subregion segmentation mask overlaid on all four phases.
        hide_centroids: When ``True`` centroid markers are hidden.
        out_path: Output file path.  When ``None`` the path is derived
            automatically from ``mrt_ref[0]`` with ``desc=vibe``.
        verbose: When ``True`` the saved path is printed to stdout.

    Returns:
        Path to the saved snapshot image.
    """
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
    out_path: str | Path | None = None,
    return_frames: bool = False,
) -> list | str | Path:
    """Save a two-frame snapshot comparing an MRI and a CT side by side.

    Frame 1: MRI in ``"MRI"`` mode with optional vertebra mask and centroids.
    Frame 2: CT in ``"CT"`` mode with optional vertebra mask and centroids.

    Args:
        mrt_ref: MRI image reference.
        ct_ref: CT image reference.
        vert_msk_mrt: Optional vertebra segmentation for the MRI frame.
        subreg_ctd_mrt: Optional subregion centroids for the MRI frame.
        vert_msk_ct: Optional vertebra segmentation for the CT frame.
        subreg_ctd_ct: Optional subregion centroids for the CT frame.
        out_path: Output file path.  When ``None`` the path is derived
            automatically from ``mrt_ref`` (must be a :class:`~TPTBox.BIDS_FILE`).
        return_frames: When ``True`` the list of
            :class:`~.Snapshot_Frame` objects is returned without saving.

    Returns:
        The list of :class:`~.Snapshot_Frame` objects when ``return_frames``
        is ``True``, otherwise the path to the saved snapshot image.
    """
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
    if return_frames:
        return frames
    if out_path is None:
        assert isinstance(mrt_ref, BIDS_FILE)
        out_path = mrt_ref.get_changed_path(file_type="png", bids_format="snp", info={"desc": "vert-ct-mri"})
    create_snapshot(snp_path=[out_path], frames=frames)
    return out_path


def poi_snapshot(
    ct_nii: BIDS_FILE,
    vert_msk: Image_Reference | None,
    subreg_ctd: POI_Reference,
    out_path: str | Path | None = None,
) -> None:
    """Save a multi-frame CT snapshot visualising ligament attachment-point POIs.

    The POIs are partitioned into lateral groups (dorsal ``_D``, sinister
    ``_S``, median) and ligament groups (ALL, PLL, FL).  Seven frames are
    produced covering different sagittal and coronal views of these groups.

    Args:
        ct_nii: BIDS file pointing to the CT image.
        vert_msk: Optional vertebra segmentation mask.
        subreg_ctd: POI reference containing the ligament attachment points.
        out_path: Output file path.  When ``None`` the path is derived
            automatically from ``ct_nii`` with ``desc=poi``.
    """
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

    def make_snap(file) -> None:
        """Generate a POI snapshot for the given file if it matches the expected subject index."""
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
