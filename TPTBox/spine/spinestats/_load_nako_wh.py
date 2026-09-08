import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from TPTBox import Print_Logger
from TPTBox.core.bids_files import BIDS_FILE, BIDS_Family, Buffered_BIDS_Global_info
from TPTBox.core.nii_wrapper import to_nii

# rawdata (stiched syn und org)
# derivative (alle mein)
# derivatives-fullbody-poi
# derivatives_inference_proc_RIB_HE_508

log = Print_Logger()

_DEFAULT_DECISION_CACHE = Path(__file__).with_name("_load_nako_wh_decisions.json")


class DecisionCache:
    """Persistent per-(subject, key) decision cache backed by a JSON file.

    A "decision" is any interactive choice the loop asks the user to resolve
    (e.g. which of several candidate files to keep, or whether to discard an
    unknown chunk). Once made, the answer is written to disk and reused on
    subsequent runs without prompting again.
    """

    def __init__(self, path: Path | str = _DEFAULT_DECISION_CACHE):
        self.path = Path(path)
        self.data: dict[str, dict[str, object]] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                log.on_warning(f"Could not parse decision cache {self.path}, starting fresh")
                self.data = {}

    def get(self, sub: str, key: str):
        return self.data.get(str(sub), {}).get(key)

    def set(self, sub: str, key: str, value):
        self.data.setdefault(str(sub), {})[key] = value
        self._flush()

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True))
        tmp.replace(self.path)


def _fmt_file(bf) -> str:
    try:
        return str(bf.file["nii.gz"]) if hasattr(bf, "file") else str(bf)
    except Exception:
        return str(bf)


DEFAULT_REASONS = ["Just duplicated", "Defect", "Missing", "Motion artifact"]


def _prompt_reason(default_reasons: list[str] = DEFAULT_REASONS) -> str:
    """Prompt for a free-text reason; user can pick a numbered default or type their own."""
    print("Reason?  Pick a number or type free text:")
    for i, r in enumerate(default_reasons):
        print(f"  [{i}] {r}")
    raw = input("reason> ").strip()
    if raw.isdigit() and 0 <= int(raw) < len(default_reasons):
        return default_reasons[int(raw)]
    return raw or "unspecified"


def _prompt_choice(sub: str, key: str, question: str, options: list[str], allow_discard: bool = True):
    """Prompt the user to pick one of ``options``.

    Returns a tuple ``(choice, reason)`` where ``choice`` is:
      - an int index into ``options`` (user picked one),
      - ``None`` (discard all),
      - or the sentinel string ``"__skip__"`` (do not save; ask again next run).
    ``reason`` is the free-text reason string, or ``None`` when skipped.
    """
    print("\n" + "=" * 72)
    print(f"[decision needed] subject={sub}  key={key}")
    print(question)
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    if allow_discard:
        print("  [d] discard all")
    print("  [s] skip (do not save; ask again next run)")
    while True:
        raw = input("> ").strip().lower()
        if raw == "s":
            return "__skip__", None
        if allow_discard and raw == "d":
            return None, _prompt_reason()
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(options):
                return idx, _prompt_reason()
        print("invalid input, try again")


def resolve_pick(
    cache: DecisionCache,
    sub: str,
    key: str,
    question: str,
    candidates: list,
    allow_discard: bool = True,
):
    """Return the single chosen candidate (or None if discarded), using cache when possible."""
    if len(candidates) == 1:
        return candidates[0]
    cached = cache.get(sub, key)
    labels = [_fmt_file(c) for c in candidates]
    cached_pick = _cached_pick(cached)
    if cached_pick is not None:
        if cached_pick == "__discard__":
            return None
        if cached_pick in labels:
            return candidates[labels.index(cached_pick)]
        log.on_warning(f"cached decision {cached_pick!r} for ({sub},{key}) no longer matches candidates; re-asking")
    choice, reason = _prompt_choice(sub, key, question, labels, allow_discard=allow_discard)
    if choice == "__skip__":
        return candidates[0] if candidates else None  # transient: do not save
    if choice is None:
        cache.set(sub, key, {"pick": "__discard__", "reason": reason})
        return None
    cache.set(sub, key, {"pick": labels[choice], "reason": reason})
    return candidates[choice]


EXPECTED_IMAGES = {
    # base image key -> dependent seg keys dropped if base is missing
    "T2w": ["vert", "spine", "poi"],
    "T2haste": [],
    "pd": [],
    "vibe_part-inphase": [
        "vibe_part-outphase",
        "vibe_part-fat",
        "vibe_part-water",
        "vibeseg100",
        "MRSegmentator",
        "msk_seg-body-composition_mod-vibe",
        "roi",
    ],
    "eco0-opp1": [
        "eco1-pip1",
        "eco2-opp2",
        "eco3-in1",
        "eco4-pop1",
        "eco5-arb1",
        "mevibe_part-fat",
        "msk_seg-body-composition_mod-mevibe",
    ],
}


def verify_missing_images(cache: DecisionCache, sub: str, subj_dict: dict) -> None:
    """For each expected base image absent from ``subj_dict``, ask the user whether it's
    really missing. If confirmed missing, drop the base and its dependent seg keys from
    ``subj_dict`` (set to None). Decisions are cached per (subject, image).
    """
    for base, deps in EXPECTED_IMAGES.items():
        present = subj_dict.get(base) is not None
        if present:
            continue
        key = f"missing:{base}"
        cached = cache.get(sub, key)
        decision = cached.get("decision") if isinstance(cached, dict) else cached
        Print_Logger().on_debug(sub, key, cached)
        if key in ["missing:T2haste", "missing:vibe_part-inphase", "missing:eco0-opp1", "missing:T2w"]:
            decision = "missing"
            cache.set(sub, key, {"decision": decision, "reason": "Missing"})

        elif decision is None:
            print("\n" + "=" * 72)
            print(f"[decision needed] subject={sub}  base image {base!r} not found.")
            print(f"Dependent keys that will also be dropped: {deps}")
            print("  [m] confirm MISSING (drop base + dependents, remember)")
            print("  [k] keep as-is (leave None, remember)")
            print("  [s] skip (do not save; ask again next run)")
            while True:
                raw = input("> ").strip().lower()
                if raw in ("m", "k", "s"):
                    break
                print("invalid input, try again")
            if raw == "s":
                decision = "keep"  # transient, don't save
            else:
                reason = _prompt_reason() if raw == "m" else "kept-as-is"
                decision = "missing" if raw == "m" else "keep"
                cache.set(sub, key, {"decision": decision, "reason": reason})
        if decision == "missing":
            subj_dict[base] = None
            for dep in deps:
                if dep in subj_dict:
                    subj_dict[dep] = None


def check_same_grid(cache: DecisionCache, sub: str, group: str, files: list) -> bool:
    """Verify all ``files`` share the same grid (spacing/shape/affine via ``bf.get_grid_info()``).

    On mismatch: print spacing per file, log a warning, and record an "issue" entry in the cache
    (once per subject/group/grid-signature) so the mismatch is surfaced but not re-prompted.
    Returns True when grids match, False otherwise.
    """
    grids: dict = {}
    for bf in files:
        if bf is None:
            continue
        try:
            g = bf.get_grid_info()
        except Exception as e:  # noqa: BLE001
            log.on_warning(f"get_grid_info failed for {_fmt_file(bf)}: {e}")
            continue
        grids.setdefault(str(g), []).append(_fmt_file(bf))
    if len(grids) <= 1:
        return True
    key = f"grid_mismatch:{group}:" + "|".join(sorted(grids.keys()))
    print("\n" + "=" * 72)
    print(f"[ISSUE] subject={sub}  group={group}  grid mismatch across {sum(len(v) for v in grids.values())} files:")
    for g, names in grids.items():
        print(f"  grid {g}")
        for n in names:
            print(f"    - {n}")
    log.on_warning(f"grid mismatch in {group} for subject {sub}")
    # if cache.get(sub, key) is None: TODO
    #    cache.set(sub, key, {"issue": "grid_mismatch", "grids": {g: n for g, n in grids.items()}})
    return False


def _cached_pick(cached):
    """Return the pick string from a cache entry (supports both legacy strings and new dicts)."""
    if cached is None:
        return None
    if isinstance(cached, dict):
        return cached.get("pick")
    return cached


def resolve_keep_chunks(
    cache: DecisionCache,
    sub: str,
    unknown_chunks: list[str],
    t2w_chunk: dict,
) -> list[str]:
    """Decide which unknown chunks to keep (default: discard all).

    Returns the list of chunks the caller should DROP from ``t2w_chunk``.
    """
    if not unknown_chunks:
        return []
    key = "unknown_chunks:" + ",".join(sorted(unknown_chunks))
    cached = cache.get(sub, key)
    if cached is not None:
        drop = cached.get("drop") if isinstance(cached, dict) else cached
        return list(drop) if isinstance(drop, list) else []
    print("\n" + "=" * 72)
    print(f"[decision needed] subject={sub}  unknown t2w chunks (not in BWS/LWS/HWS)")
    for c in unknown_chunks:
        print(f"  chunk={c!r}  ->  {[_fmt_file(x) for x in t2w_chunk.get(c, [])]}")
    print("Enter comma-separated chunk names to DISCARD, 'all' to discard all, or empty to keep all.")
    raw = input("> ").strip()
    if raw.lower() == "all":
        drop = list(unknown_chunks)
    elif raw == "":
        drop = []
    else:
        drop = [x.strip() for x in raw.split(",") if x.strip()]
    reason = _prompt_reason() if drop else "kept all"
    cache.set(sub, key, {"drop": drop, "reason": reason})
    return drop


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
    add_mevibe=True,
    add_vibe=True,
    compute_PDFF=False,
    dataset="/DATA/NAS/datasets_processed/NAKO/dataset-nako/",
    test=False,
    verbose=False,
    sort=True,
    test_key="/110/110",  # path matching. if you want on specific us a 6 digits
    decision_cache: DecisionCache | Path | str | None = None,
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
        sort: If True, iterate subjects in alphabetical order (see :meth:`BIDS_Global_info.iter_subjects`).
        test_key: Path substring passed to the BIDS scanner's ``filter_file`` when ``test=True``; only paths
            containing this substring are indexed. Defaults to a hard-coded example subject.
        baseline_metadata: Path to the NAKO baseline CSV used to look up height metadata.

    Yields:
        Dict mapping short keys to ``BIDS_FILE`` entries for one subject.
    """
    if not isinstance(decision_cache, DecisionCache):
        decision_cache = DecisionCache(decision_cache) if decision_cache is not None else DecisionCache()
    cache = decision_cache

    gbi = Buffered_BIDS_Global_info(
        datasets=dataset,
        parents=[
            "rawdata",
            "rawdata_stitched",
            "derivatives_Abdominal-Segmentation",
            # "derivatives_mevibe", #copied into "derivatives_Abdominal-Segmentation"
            "derivatives_inversion",
        ],
        filter_file=(lambda x: test_key in str(x)) if test else None,
    )

    for sub, subj in gbi.enumerate_subjects(sort=sort, shuffle=not sort):
        subj_dict = {"id": sub, "dataset": dataset}
        # Primary source: baseline CSV, height is in cm.
        if verbose:
            log.on_log(sub)

        q = subj.new_query(flatten=True)
        q.filter("chunk", lambda _: True, required=True)
        q.filter_format("T2w")
        t2w_chunk: dict[str, list] = {}
        for bf in q.loop_list():
            chunk = str(bf.get("chunk"))
            # TODO manual list to ignore things
            if chunk not in t2w_chunk:
                t2w_chunk[chunk] = []
            t2w_chunk[chunk].append(bf)
        white_list = ["BWS", "LWS", "HWS"]
        unknown = [a for a in t2w_chunk.keys() if a not in white_list]
        for c in resolve_keep_chunks(cache, sub, unknown, t2w_chunk):
            t2w_chunk.pop(c, None)
        for chunk_name, files in list(t2w_chunk.items()):
            if len(files) > 1:
                picked = resolve_pick(
                    cache,
                    sub,
                    f"t2w_chunk:{chunk_name}",
                    f"Multiple T2w files for chunk={chunk_name!r}; pick one to keep (or discard).",
                    files,
                )
                if picked is None:
                    t2w_chunk.pop(chunk_name)
                else:
                    t2w_chunk[chunk_name] = [picked]

        subj_dict["t2w_chunk"] = t2w_chunk  # type: ignore
        # T2w stiched
        # PD, "T2haste"
        q = subj.new_query()
        q.filter("chunk", lambda _: False, required=False)
        mapping = {"T2w": "T2w"}
        keys = ["pd", "T2haste", *mapping.keys()]

        for fam in q.loop_dict(key_addendum=["mod", "part", "desc"]):
            for k, v in fam.items():
                if k in keys:
                    k = mapping.get(k, k)  # noqa: PLW2901
                    if len(v) > 1:
                        picked = resolve_pick(cache, sub, f"main:{k}", f"Multiple files for {k}; pick one.", v)
                        if picked is None:
                            continue
                    else:
                        picked = v[0]
                    if k in subj_dict:
                        replace = resolve_pick(
                            cache,
                            sub,
                            f"main-conflict:{k}",
                            f"{k} already set from another family; keep existing or replace?",
                            [subj_dict[k], picked],
                            allow_discard=False,
                        )
                        subj_dict[k] = replace
                    else:
                        subj_dict[k] = picked

        keys = ["msk_seg-body-composition_mod-mevibe"]
        if add_mevibe:
            q = subj.new_query()
            q.filter_format("mevibe")
            # q.filter("sequ", "me1")
            mevibe_fams = list(q.loop_dict(key_addendum=["mod", "part", "desc"]))
            if len(mevibe_fams) > 1:
                labels = [str(f.get("mevibe_part-eco0-opp1", f)) for f in mevibe_fams]
                cached_pick = _cached_pick(cache.get(sub, "mevibe_fam"))
                if cached_pick == "__discard__":
                    mevibe_fams = []
                elif cached_pick in labels:
                    mevibe_fams = [mevibe_fams[labels.index(cached_pick)]]
                else:
                    choice, reason = _prompt_choice(sub, "mevibe_fam", "Multiple mevibe families; pick one.", labels, allow_discard=True)
                    if choice == "__skip__":
                        mevibe_fams = mevibe_fams[:1]
                    elif choice is None:
                        cache.set(sub, "mevibe_fam", {"pick": "__discard__", "reason": reason})
                        mevibe_fams = []
                    else:
                        cache.set(sub, "mevibe_fam", {"pick": labels[choice], "reason": reason})
                        mevibe_fams = [mevibe_fams[choice]]
            for fam in mevibe_fams:
                mevibe_out = get_corrected_mevibe(fam, compute_PDFF=compute_PDFF)
                check_same_grid(cache, sub, "mevibe", list(mevibe_out.values()))
                subj_dict = {**mevibe_out, **subj_dict}
                for k, v in fam.items():
                    if k in keys:
                        k = mapping.get(k, k)  # noqa: PLW2901
                        if len(v) > 1:
                            picked = resolve_pick(cache, sub, f"mevibe:{k}", f"Multiple mevibe files for {k}; pick one.", v)
                            if picked is None:
                                continue
                        else:
                            picked = v[0]
                        if k in subj_dict:
                            picked = resolve_pick(
                                cache,
                                sub,
                                f"mevibe-conflict:{k}",
                                f"{k} already set; keep existing or replace?",
                                [subj_dict[k], picked],
                                allow_discard=False,
                            )
                        subj_dict[k] = picked
        if add_vibe:
            mapping = {
                "msk_seg-MRSegmentator_part-inphase": "MRSegmentator",
                "msk_seg-VibeSeg-100_mod-vibe_part-inphase": "vibeseg100",
                "msk_seg-ROI_mod-vibe": "roi",
            }
            q = subj.new_query()
            q.filter_format("vibe")
            q.filter("chunk", lambda _: False, required=False)
            # q.filter("run", lambda x: x != "2", required=False)
            keys = [
                "vibe_part-inphase",
                "vibe_part-outphase",
                "vibe_part-fat",
                "vibe_part-water",
                "msk_seg-body-composition_mod-vibe",
                *mapping.keys(),
            ]
            vibe_fams = list(q.loop_dict(key_addendum=["mod", "part", "desc"]))
            if len(vibe_fams) > 1:
                labels = [str(f) for f in vibe_fams]
                cached_pick = _cached_pick(cache.get(sub, "vibe_fam"))
                if cached_pick == "__discard__":
                    vibe_fams = []
                elif cached_pick in labels:
                    vibe_fams = [vibe_fams[labels.index(cached_pick)]]
                else:
                    choice, reason = _prompt_choice(sub, "vibe_fam", "Multiple vibe families; pick one.", labels, allow_discard=True)
                    if choice == "__skip__":
                        vibe_fams = vibe_fams[:1]
                    elif choice is None:
                        cache.set(sub, "vibe_fam", {"pick": "__discard__", "reason": reason})
                        vibe_fams = []
                    else:
                        cache.set(sub, "vibe_fam", {"pick": labels[choice], "reason": reason})
                        vibe_fams = [vibe_fams[choice]]
            for fam in vibe_fams:
                vibe_files = []
                for _k in (
                    "vibe_part-inphase",
                    "vibe_part-outphase",
                    "vibe_part-fat",
                    "vibe_part-water",
                    "vibe_part-water_desc-reconstructed",
                    "vibe_part-fat_desc-reconstructed",
                ):
                    if _k in fam:
                        vibe_files.extend(fam[_k])
                check_same_grid(cache, sub, "vibe", vibe_files)
                for k, v in fam.items():
                    if k in keys:
                        k = mapping.get(k, k)  # noqa: PLW2901
                        if len(v) > 1:
                            picked = resolve_pick(cache, sub, f"vibe:{k}", f"Multiple vibe files for {k}; pick one.", v)
                            if picked is None:
                                continue
                        else:
                            picked = v[0]
                        if k in subj_dict:
                            picked = resolve_pick(
                                cache,
                                sub,
                                f"vibe-conflict:{k}",
                                f"{k} already set; keep existing or replace?",
                                [subj_dict[k], picked],
                                allow_discard=False,
                            )
                        subj_dict[k] = picked
                mapp = {"vibe_part-water_desc-reconstructed": "vibe_part-water", "vibe_part-fat_desc-reconstructed": "vibe_part-fat"}
                for k, k2 in mapp.items():
                    if k in fam:
                        subj_dict[k2] = fam[k][0]
        vert, spine, poi = get_current_best_T2w_seg(sub)
        subj_dict["vert"] = vert
        subj_dict["spine"] = spine
        subj_dict["poi"] = poi
        verify_missing_images(cache, sub, subj_dict)
        yield subj_dict


allowed_keys = ["sub", "sequ", "ses", "seg", "acq", "chunk", "part", "mod", "desc", "rec"]


def hard_link(
    d: dict,
    dataset="/DATA/NAS/datasets_processed/NAKO/dataset-nako/",
):
    subj = d.pop("id")
    d.pop("dataset")
    log.on_log(subj)

    for key, t2w in d.pop("t2w_chunk").items():
        bf: BIDS_FILE = t2w[0]

        assert len([k for k, v in bf.loop_keys() if k not in allowed_keys]) == 0, [k for k, v in bf.loop_keys() if k not in allowed_keys]
        assert key in ["BWS", "LWS", "HWS"]
        new_path = bf.get_changed_path(
            "nii.gz",
            bf.format,
            parent="rawdata",
            info={"ses": "baseline"},
            dataset_path="/DATA/NAS/datasets_processed/NAKO/dataset-nako-canonical",
        )
        if not new_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            bf.symlink_files(new_path, hard_link=True)  # exist_ok=True,
            print("Hard linked:", new_path)

    segs = [
        "msk_seg-body-composition_mod-mevibe",  # MEVIBE
        "vibeseg100",  # vibe
        "MRSegmentator",  # vibe
        "msk_seg-body-composition_mod-vibe",  # vibe
        "roi",  # vibe
        "vert",  # t2w (stiched)
        "spine",  # t2w (stiched)
        "poi",  # t2w (stiched)
    ]
    imgs = [
        "pd",
        "T2haste",
        "T2w",
        "eco0-opp1",
        "eco1-pip1",
        "eco2-opp2",
        "eco3-in1",
        "eco4-pop1",
        "eco5-arb1",
        "mevibe_part-fat",
        "vibe_part-outphase",
        "vibe_part-fat",
        "vibe_part-water",
        "vibe_part-inphase",
    ]
    for keys, parent in [(imgs, "rawdata"), (segs, "derivatives")]:
        for key in keys:
            bf = d.pop(key, None)
            if bf is None:
                continue
            info = {"run": None}
            if isinstance(bf, str):
                bf = BIDS_FILE(bf, dataset)
            assert len([k for k, v in bf.loop_keys() if k not in allowed_keys]) == 0, (
                [k for k, v in bf.loop_keys() if k not in allowed_keys],
                bf,
            )
            new_path = bf.get_changed_path(
                "nii.gz",
                bf.format,
                parent=parent,
                info=info,
                dataset_path="/DATA/NAS/datasets_processed/NAKO/dataset-nako-canonical",
            )
            if not new_path.exists():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                bf.symlink_files(new_path, hard_link=True)  # exist_ok=True,
                print(new_path)
    leftover = {k: v for k, v in d.items() if v is not None}
    assert len(leftover) == 0, leftover


def _grid_worker(nii_path: str) -> tuple[str, str]:
    """Worker: compute grid info for one NIfTI and cache it into its JSON sidecar.

    Mirrors the fallback logic of ``BIDS_FILE.get_grid_info`` for the sidecar path
    (strip all suffixes and append ``.json``), so the cached result is picked up on
    subsequent ``bf.get_grid_info()`` calls without any further work.
    """
    from TPTBox.core.internal.nii_help import _add_grid_info_to_json

    p = Path(nii_path)
    if not p.exists():
        return nii_path, "missing"
    sidecar = Path(str(p).split(".")[0] + ".json")
    try:
        _add_grid_info_to_json(p, sidecar, add=True)
        return nii_path, "ok"
    except Exception as e:  # noqa: BLE001
        return nii_path, f"error: {type(e).__name__}: {e}"


def _iter_grid_targets(subj_dict: dict):
    """Yield NIfTI paths from a ``subj_dict`` that will later be inspected by ``check_same_grid``.

    Covers the same set of files the interactive loop touches: every ``BIDS_FILE``
    stored under a modality/segmentation key, plus the T2w chunk lists.  ``None``
    values, plain strings (already-resolved paths) and stray non-``BIDS_FILE``
    entries are handled without raising.
    """

    def _to_path(v):
        if v is None:
            return None
        if isinstance(v, (str, Path)):
            s = str(v)
            return s if s and Path(s).exists() else None
        get_nii_file = getattr(v, "get_nii_file", None)
        if get_nii_file is None:
            return None
        try:
            p = get_nii_file()
        except Exception:  # noqa: BLE001
            return None
        return str(p) if p is not None else None

    for key, v in subj_dict.items():
        if key in ("id", "dataset", "t2w_chunk"):
            continue
        p = _to_path(v)
        if p is not None:
            yield p

    for files in (subj_dict.get("t2w_chunk") or {}).values():
        for bf in files:
            p = _to_path(bf)
            if p is not None:
                yield p


def precompute_grid_info_parallel(
    num_workers: int | None = None,
    max_inflight: int = 512,
    verbose: bool = True,
    **loop_kwargs,
) -> None:
    """Iterate over the NAKO loop and populate the ``grid`` JSON sidecar in parallel.

    ``BIDS_FILE.get_grid_info`` caches the computed grid inside the sidecar JSON on
    first call; the next call is essentially a JSON read.  This helper front-loads
    that first call across many workers so the interactive/serial consumer of
    :func:`loop_over_repaired_nako` never pays the per-file NIfTI-open cost.

    Args:
        num_workers: Worker process count; defaults to ``max(1, cpu_count() - 1)``.
        max_inflight: Cap on submitted-but-unfinished tasks; drained when exceeded
            so memory stays bounded on very large datasets.
        verbose: Log per-file failures and a final summary.
        **loop_kwargs: Forwarded to :func:`loop_over_repaired_nako`.
    """
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 1)

    seen: set[str] = set()
    ok = 0
    failed = 0

    def _drain(fs):
        nonlocal ok, failed
        for f in as_completed(fs):
            path, status = f.result()
            if status == "ok":
                ok += 1
            else:
                failed += 1
                if verbose:
                    log.on_warning(f"grid precompute {path}: {status}")

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        pending: list = []
        for subj_dict in loop_over_repaired_nako(**loop_kwargs):
            for p in _iter_grid_targets(subj_dict):
                if p in seen:
                    continue
                seen.add(p)
                pending.append(pool.submit(_grid_worker, p))
            if len(pending) >= max_inflight:
                _drain(pending)
                pending = []
        if pending:
            _drain(pending)

    if verbose:
        log.on_log(f"precompute_grid_info_parallel done: ok={ok} failed={failed} total={ok + failed}")


if __name__ == "__main__":
    import argparse

    from TPTBox import Print_Logger

    log = Print_Logger()

    parser = argparse.ArgumentParser(description="NAKO helpers: hard-link or prewarm grid info.")
    parser.add_argument(
        "--precompute-grid",
        action="store_true",
        help="Prewarm bf.get_grid_info() JSON caches in parallel processes instead of hard-linking.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: cpu_count-1).")
    # parser.add_argument("--no-test", action="store_true", help="Iterate the full dataset instead of the default test subtree.")
    args = parser.parse_args()
    test = False
    if args.precompute_grid:
        precompute_grid_info_parallel(num_workers=args.workers, test=test)
    else:
        for d in loop_over_repaired_nako(test=test):
            # pass
            hard_link(d)
            # print(d["T2w"])
            # break
        # check VIBE same shape
