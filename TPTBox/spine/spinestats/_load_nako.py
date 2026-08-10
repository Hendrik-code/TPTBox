import os
from pathlib import Path

from TPTBox.core.bids_files import BIDS_FILE, BIDS_Family, Buffered_BIDS_Global_info
from TPTBox.core.nii_wrapper import to_nii


def _check(l: list[BIDS_FILE]):
    """Pick the preferred BIDS file from a list of candidates.

    Prefers files with a ``rec`` entity (reconstruction variant, defaulting to ``"Hamilton"``).
    If no such file is found, asserts that there is exactly one candidate and returns it.

    Args:
        l: Candidate BIDS files sharing the same BIDS query key.

    Returns:
        The chosen ``BIDS_FILE``.
    """
    for i in l:
        if i.get("rec", "Hamilton"):
            return i
    assert len(l) == 1, l
    return l[0]


def get_corrected_mevibe(fam: BIDS_Family, compute_PDFF=True):  # TODO return dict with literal
    """Collect the six mevibe echo images plus fat/water/PDFF/PDWF for one subject family.

    The ``BIDS_Global_info`` used to build ``fam`` must include ``derivatives_mevibe`` as a
    parent root, and its query key addendum must contain ``part`` and ``desc`` so the echo
    images are addressable. If reconstructed fat/water images are present they are preferred
    over the raw ones. When ``compute_PDFF`` is set and the reconstructed fat-fraction (PDFF)
    or water-fraction (PDWF) maps are missing on disk, they are computed as
    ``fat / (fat + water) * 1000`` (and the water equivalent), cast to the smallest int dtype,
    and saved next to the reconstructed water image.

    Args:
        fam: BIDS family for a single mevibe acquisition.
        compute_PDFF: If True, generate and persist missing PDFF/PDWF maps.

    Returns:
        Dict mapping mevibe part keys (``"eco0-opp1"`` … ``"eco5-arb1"``, ``"mevibe_part-fat"``)
        to the chosen ``BIDS_FILE`` entries.
    """
    # TODO figure out what to do when multiple present
    # BIDS_GLOBAL_INFO needs to have "derivatives_mevibe" as an additional root
    # additional keys must be part and desc
    # PDFF is recomputed
    out = {key: _check(fam[f"mevibe_part-{key}"]) for key in ["eco0-opp1", "eco1-pip1", "eco2-opp2", "eco3-in1", "eco4-pop1", "eco5-arb1"]}

    pdff = _check(fam["mevibe_part-fat-fraction"])
    if "mevibe_part-water_desc-reconstructed" in fam:
        # if "mevibe_part-fat-fraction_desc-reconstructed" not in fam:
        fat = _check(fam["mevibe_part-fat_desc-reconstructed"])
        water = _check(fam["mevibe_part-water_desc-reconstructed"])

    else:
        fat = _check(fam["mevibe_part-fat"])
        water = _check(fam["mevibe_part-water"])
    out["mevibe_part-fat"] = fat
    out["mevibe_part-fat"] = water
    pdff = water.get_changed_bids(
        "nii.gz", bids_format=water.bids_format, parent=water.parent, info={"part": "fat-fraction", "desc": "reconstructed"}
    )
    pdwf = water.get_changed_bids(
        "nii.gz", bids_format=water.bids_format, parent=water.parent, info={"part": "water-fraction", "desc": "reconstructed"}
    )

    if compute_PDFF and (not pdff.exists() or not pdwf.exists()):
        water_nii = to_nii(water)
        fat_nii = to_nii(fat)
        water_nii.set_dtype_()
        fat_nii.set_dtype_()
        if not pdff.exists():
            nii = fat_nii / (water_nii + fat_nii)
            nii[water_nii + fat_nii == 0] = 0
            nii *= 1000
            nii.set_dtype_("smallest_int")
            nii.save(pdff)
        if not pdwf.exists():
            nii = water_nii / (water_nii + fat_nii)
            nii[water_nii + fat_nii == 0] = 0
            nii *= 1000
            nii.set_dtype_("smallest_int")
            nii.save(pdwf)
    if pdff.exists():
        out["mevibe_part-fat"] = pdff
    if pdff.exists():
        out["mevibe_part-fat"] = pdwf
        # else:
        #    pdff = _check(fam["mevibe_part-fat-fraction_desc-reconstructed"])

    return out


def get_current_best_T2w_seg(sub, black_list_t2w=None):
    if black_list_t2w is None:
        black_list_t2w = [
            # Head missing T2w
            "106910",
            "100470",
            "105805",
            "119399",  # "Scoliosis, no head"
            "125130",
        ]
    search_folders = [
        "derivatives_spine_vert_fixed",
        "derivatives_spine_inference_combination162_148",
        # "derivatives_spine_inference_combination",
        # "derivatives_spine_inference_159_sacrumfix",
        # "derivatives_spine_inference_148_preliminary",
        # "derivatives_spine_inference_146_preliminary",  # sub-128135_sequ-stitched_acq-sag_mod-T2w_seg-vert_msk.nii.gz
    ]
    sub = str(sub).split("_")[0].replace("sub-", "")
    if sub in black_list_t2w:
        return None, "", None
    if sub in [
        # "100303",
        # "109091",
        "113612",
        # "106991",
        "102179",
        # "102263",
        "103730",
        "103704",
        "110618",
        "123393",
        "123222",
        "124365",
        "104249",
        "104000",
    ]:
        search_folders = ["archive/derivatives_spine_inference_148_preliminary"]
    for s in search_folders:
        vert_T2w = f"/DATA/NAS/datasets_processed/NAKO/dataset-nako/{s}/{sub[:3]}/{sub}/T2w/sub-{sub}_sequ-stitched_acq-sag_mod-T2w_seg-vert_msk.nii.gz"
        spine_T2w = f"/DATA/NAS/datasets_processed/NAKO/dataset-nako/{s}/{sub[:3]}/{sub}/T2w/sub-{sub}_sequ-stitched_acq-sag_mod-T2w_seg-spine_msk.nii.gz"
        poi = f"/DATA/NAS/datasets_processed/NAKO/dataset-nako/{s}/{sub[:3]}/{sub}/T2w/sub-{sub}_sequ-stitched_acq-sag_mod-T2w_seg-spine_ctd.json"

        if Path(vert_T2w).exists():
            return vert_T2w, spine_T2w, poi
    if not Path(vert_T2w).exists():
        T2w = Path(
            f"/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata_stitched/{sub[:3]}/{sub}/T2w/sub-{sub}_sequ-stitched_acq-sag_T2w.nii.gz"
        )
        if T2w.exists():
            log.on_fail(f"Segmentation missing; {T2w.exists()=}", vert_T2w)
        else:
            T2w_org = list(Path(f"/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata/{sub[:3]}/{sub}/T2w/").glob("*_T2w.nii.gz"))
            if len(T2w_org) <= 2:
                log.on_warning(f"Segmentation missing; {len(T2w_org)=}", Path(vert_T2w).name)
            else:
                log.on_debug(f"Segmentation missing; {(T2w_org)=}", Path(vert_T2w).name)
        return None, "", None
    return vert_T2w, spine_T2w, poi


def loop_over_repaired_nako(
    add_mevibe=False,
    add_vibe=True,
    compute_PDFF=False,
    raise_on_duplicate=True,
    dataset="/DATA/NAS/datasets_processed/NAKO/dataset-nako/",
    test=True,
    verbose=False,
):
    """Iterate over the repaired NAKO dataset yielding per-subject file dicts.

    Scans the NAKO BIDS dataset (including derivative roots for MEVIBE, inversion, and
    abdominal segmentation), and for each subject collects a curated set of image and mask
    files keyed by short names (e.g. ``"t2w"``, ``"MRSegmentator"``, ``"vibeseg100"``,
    ``"roi"``). Optionally augments each subject with corrected MEVIBE outputs (see
    :func:`get_corrected_mevibe`) and/or the four vibe part images (in-/out-phase, fat,
    water), preferring reconstructed vibe fat/water when available.

    Args:
        add_mevibe: Include corrected MEVIBE files (and optionally recompute PDFF/PDWF).
        add_vibe: Include vibe part images.
        compute_PDFF: Passed through to :func:`get_corrected_mevibe`.
        raise_on_duplicate: Assert that each key resolves to exactly one file per subject.
        dataset: Root path of the NAKO BIDS dataset.
        test: If True, restrict scanning to a single hard-coded subject subtree for quick runs.
        verbose: Log each subject id as it is processed.

    Yields:
        Dict mapping short keys to ``BIDS_FILE`` entries for one subject.
    """
    gbi = Buffered_BIDS_Global_info(
        datasets=dataset,
        parents=["rawdata", "rawdata_stitched", "derivatives_Abdominal-Segmentation", "derivatives_mevibe", "derivatives_inversion"],
        # sequence_splitting_keys=["sub", "ses"],
        # "/107/107472"
        filter_file=(lambda x: "/110/11089" in str(x)) if test else None,  # Figures sent to Paul "/117/117001"; "/113/113508"
    )

    mapping = {
        "T2w": "t2w",
        "msk_seg-MRSegmentator_part-inphase": "MRSegmentator",
        "msk_seg-VibeSeg-100_mod-vibe_part-inphase": "vibeseg100",
        "msk_seg-ROI_mod-vibe": "roi",
    }
    keys = ["pd", "T2haste", *mapping.keys(), "msk_seg-body-composition_mod-vibe"]
    for sub, subj in gbi.enumerate_subjects(sort=True):
        subj_dict = {"id": sub, "dataset": dataset}
        if verbose:
            log.on_log(sub)
        q = subj.new_query()
        q.filter("chunk", lambda _: False, required=False)
        for fam in q.loop_dict(key_addendum=["mod", "part", "desc"]):
            for k, v in fam.items():
                print(fam)
                if k in keys:
                    k = mapping.get(k, k)  # noqa: PLW2901
                    if raise_on_duplicate:
                        assert len(v) == 1, v
                        assert k not in subj_dict, (k, subj_dict, v, subj_dict[k])
                    subj_dict[k] = v[0]
                print()
        # exit()
        keys = ["msk_seg-body-composition_mod-mevibe"]
        if add_mevibe:
            q = subj.new_query()
            q.filter_format("mevibe")
            q.filter("sequ", "me1")
            for fam in q.loop_dict(key_addendum=["mod", "part", "desc"]):
                subj_dict = {**get_corrected_mevibe(fam, compute_PDFF=compute_PDFF), **subj_dict}
                for k, v in fam.items():
                    if k in keys:
                        k = mapping.get(k, k)  # noqa: PLW2901
                        if raise_on_duplicate:
                            assert len(v) == 1, v
                            assert k not in subj_dict, (k, subj_dict, v, subj_dict[k])
                        subj_dict[k] = v[0]

        if add_vibe:
            q = subj.new_query()
            q.filter_format("vibe")
            q.filter("chunk", lambda _: False, required=False)
            for fam in q.loop_dict(key_addendum=["mod", "part", "desc"]):
                for k, v in fam.items():
                    if k in ["vibe_part-inphase", "vibe_part-outphase", "vibe_part-fat", "vibe_part-water"]:
                        # k = mapping.get(k, k)  # noqa: PLW2901
                        if raise_on_duplicate:
                            assert len(v) == 1, v
                            assert k not in subj_dict, (k, subj_dict, v, subj_dict[k])
                        subj_dict[k] = v[0]

                mapp = {"vibe_part-water_desc-reconstructed": "vibe_part-water", "vibe_part-fat_desc-reconstructed": "vibe_part-fat"}
                for k, k2 in mapp.items():
                    if k in fam:
                        subj_dict[k2] = fam[k][0]
        vert, spine, poi = get_current_best_T2w_seg(sub)
        subj_dict["vert"] = vert
        subj_dict["spine"] = spine
        subj_dict["poi"] = poi
        yield subj_dict


if __name__ == "__main__":
    from TPTBox import Print_Logger

    log = Print_Logger()
    for d in loop_over_repaired_nako():
        print(d.keys())
        print(d["t2w"])
        exit()
