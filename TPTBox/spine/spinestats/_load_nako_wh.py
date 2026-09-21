import json
import os
import tempfile
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

_NON_INTERACTIVE = False


class _non_interactive_mode:
    """Context manager that flips the module-level ``_NON_INTERACTIVE`` flag.

    While active, every prompt site in this module short-circuits to the same
    behaviour it would use when the user typed "skip" — first candidate is kept
    and nothing is written to the decision cache.
    """

    def __enter__(self):
        global _NON_INTERACTIVE  # noqa: PLW0603
        self._prev = _NON_INTERACTIVE
        _NON_INTERACTIVE = True
        return self

    def __exit__(self, *exc):
        global _NON_INTERACTIVE  # noqa: PLW0603
        _NON_INTERACTIVE = self._prev
        return False


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
    if _NON_INTERACTIVE:
        return "__skip__", None
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
    if key == "main:pd" or key.endswith(":pd"):
        # auto-accept: for duplicate PDs always prefer the higher ID (lexicographically last label).
        # Covers both the initial pick (``main:pd``) and the cross-family conflict path
        # (``main-conflict:pd``, ``mevibe-conflict:pd``, ``vibe-conflict:pd``, …).
        idx = max(range(len(labels)), key=labels.__getitem__)
        return candidates[idx]  # transient: do not save
    if key == "main:T2haste" or key.endswith(":T2haste"):
        # auto-accept: when duplicates differ only in the presence of a ``sequ`` entity
        # (one raw file without sequ, one or more with an explicit sequ number for the same
        # acquisition), prefer the sequ-numbered candidate. Skips the prompt only when this
        # split is unambiguous (exactly one sequ'd candidate); otherwise falls through.
        with_sequ = [c for c in candidates if getattr(c, "get", lambda *_: None)("sequ", None) is not None]
        if len(with_sequ) == 1 and len(with_sequ) < len(candidates):
            return with_sequ[0]  # transient: do not save
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
        if decision is None and _NON_INTERACTIVE:
            decision = "keep"  # transient: skip prompt, do not save
        elif key in ["missing:T2haste", "missing:vibe_part-inphase", "missing:eco0-opp1", "missing:T2w"]:
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


_RESAMPLE_TMP_ROOT = Path(
    os.environ.get(
        "TPTBOX_RESAMPLED_SCRATCH",
        "/DATA/NAS/datasets_processed/NAKO/_resampled_scratch",
    )
)


def _resample_to_ref(bf, ref_nii):
    """Resample ``bf`` onto ``ref_nii``'s grid and persist under a scratch root.

    Files are written to ``$TPTBOX_RESAMPLED_SCRATCH`` (default
    ``/DATA/NAS/datasets_processed/NAKO/_resampled_scratch``) mirroring the
    original BIDS layout with a ``desc-resampled`` entity added, so we neither
    dirty the read-only NAKO source tree nor cross filesystems (the scratch
    root lives on the same disk as NAKO so :func:`os.link` still works when
    :func:`hard_link` later re-links the file into the canonical dataset).
    The file is only written when it does not already exist; the returned
    :class:`BIDS_FILE` always points at the resampled path.
    """
    existing_desc = bf.get("desc", None)
    new_desc = "resampled" if not existing_desc else f"{existing_desc}Resampled"
    _RESAMPLE_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    new_bf = bf.get_changed_bids(
        file_type="nii.gz",
        bids_format=bf.bids_format,
        parent=bf.parent,
        info={"desc": new_desc},
        dataset_path=str(_RESAMPLE_TMP_ROOT),
    )
    new_path = new_bf.file["nii.gz"]
    if not new_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        src_nii = to_nii(bf)
        resampled = src_nii.resample_from_to(ref_nii)
        resampled.save(new_path)
    return new_bf


def check_same_grid(cache: DecisionCache, sub: str, group: str, files_by_key: dict, inphase_key: str) -> bool:
    """Ensure every file in ``files_by_key`` shares the grid of ``files_by_key[inphase_key]``.

    On mismatch the user is prompted (with a persistent per-(sub, group) decision):
      * ``y`` → resample each mismatched file to the inphase grid, save it beside
        the original with an added ``desc-resampled`` entity, and update
        ``files_by_key[k]`` in place to point at the resampled file.
      * ``n`` → drop the mismatched keys (``files_by_key[k] = None``).

    Under ``_NON_INTERACTIVE`` the default is ``resample`` and the decision is not
    persisted.  Returns ``True`` when every entry ended up on the reference grid.
    """
    grids: dict = {}
    for k, bf in files_by_key.items():
        if bf is None:
            continue
        # Segmentations (msk) don't participate in the image-grid check.
        if getattr(bf, "format", None) == "msk" or k.startswith("msk"):
            continue
        # Skip BIDS entries that carry no NIfTI (e.g. POI files with only .json / .mrk.json).
        get_nii_file = getattr(bf, "get_nii_file", None)
        if callable(get_nii_file) and get_nii_file() is None:
            continue
        try:
            g = bf.get_grid_info()
        except Exception as e:  # noqa: BLE001
            log.on_warning(f"get_grid_info failed for {_fmt_file(bf)}: {e}")
            files_by_key[k] = None
            continue
        grids[k] = (g, str(g))

    unique_sigs = {sig for _, sig in grids.values()}
    if len(unique_sigs) <= 1:
        return True

    if inphase_key not in grids:
        log.on_warning(f"inphase reference {inphase_key!r} missing for subject {sub}; cannot resample")
        return False

    ref_sig = grids[inphase_key][1]
    mismatched = [k for k, (_, sig) in grids.items() if sig != ref_sig]
    if not mismatched:
        return True

    if _NON_INTERACTIVE:
        # Precompute / non-interactive mode: don't touch anything, don't resample.
        # Grid info for every file was already read (populating the JSON cache), which
        # is all --precompute-grid wants; leave the actual reconciliation to a later
        # interactive run.
        log.on_warning(f"grid mismatch in {group} for subject {sub} (skipped in non-interactive mode)")
        return False

    print("\n" + "=" * 72)
    print(f"[grid mismatch] subject={sub}  group={group}  reference={inphase_key}  ({ref_sig})")
    for k in mismatched:
        print(f"  {k}: {grids[k][1]}  ->  {_fmt_file(files_by_key[k])}")

    cache_key = f"grid_mismatch:{group}"
    cached = cache.get(sub, cache_key)
    decision = cached.get("decision") if isinstance(cached, dict) else cached
    if decision not in ("resample", "remove"):
        print(f"  [y] resample mismatched files to the {inphase_key} grid")
        print("  [n] drop the mismatched keys")
        print("  [s] skip (do not save; ask again next run)")
        while True:
            raw = input("> ").strip().lower()
            if raw in ("y", "n", "s"):
                break
            print("invalid input, try again")
        if raw == "s":
            decision = "resample"  # transient
        else:
            decision = "resample" if raw == "y" else "remove"
            reason = "resample-to-inphase" if raw == "y" else _prompt_reason()
            cache.set(sub, cache_key, {"decision": decision, "reason": reason})

    if decision == "remove":
        for k in mismatched:
            files_by_key[k] = None
        return False

    ref_nii = to_nii(files_by_key[inphase_key])
    for k in mismatched:
        src_bf = files_by_key[k]
        try:
            files_by_key[k] = _resample_to_ref(src_bf, ref_nii)
        except Exception as e:  # noqa: BLE001
            log.on_warning(f"resample failed for {_fmt_file(src_bf)}: {e}; dropping key {k!r}")
            files_by_key[k] = None
    return True


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
    if _NON_INTERACTIVE:
        return []  # transient: keep all, do not save
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

    if "mevibe_part-water_desc-reconstructed" in fam:
        fat = _check(fam["mevibe_part-fat_desc-reconstructed"])
        water = _check(fam["mevibe_part-water_desc-reconstructed"])
    elif "mevibe_part-water" in fam:
        fat = _check(fam["mevibe_part-fat"])
        water = _check(fam["mevibe_part-water"])
    else:
        fat = None
        water = None

    # Anchor for BIDS derivations: prefer a raw/reconstructed water image, fall back to
    # whatever fraction file the family exposes.
    if water is not None:
        anchor = water
    elif "mevibe_part-water-fraction" in fam:
        anchor = _check(fam["mevibe_part-water-fraction"])
    elif "mevibe_part-fat-fraction" in fam:
        anchor = _check(fam["mevibe_part-fat-fraction"])
    else:
        return out

    pdff = anchor.get_changed_bids(
        "nii.gz", bids_format=anchor.bids_format, parent=anchor.parent, info={"part": "fat-fraction", "desc": "reconstructed"}
    )
    pdwf = anchor.get_changed_bids(
        "nii.gz", bids_format=anchor.bids_format, parent=anchor.parent, info={"part": "water-fraction", "desc": "reconstructed"}
    )

    if compute_PDFF and (not pdff.exists() or not pdwf.exists()):
        if fat is not None and water is not None:
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
        else:
            # No raw fat/water — derive the missing fraction from the one the scanner shipped.
            if not pdff.exists() and "mevibe_part-water-fraction" in fam:
                wf_nii = to_nii(_check(fam["mevibe_part-water-fraction"]))
                nii = 1000 - wf_nii
                nii.set_dtype_("smallest_int")
                nii.save(pdff)
            if not pdwf.exists() and "mevibe_part-fat-fraction" in fam:
                ff_nii = to_nii(_check(fam["mevibe_part-fat-fraction"]))
                nii = 1000 - ff_nii
                nii.set_dtype_("smallest_int")
                nii.save(pdwf)

    # Downstream (see _apply_corrections_to_subj_dict) expects "mevibe_part-fat" to hold PDFF.
    if pdff.exists():
        out["mevibe_part-fat"] = pdff
    if pdwf.exists():
        out["mevibe_part-fat"] = pdwf
        # else:
        #    pdff = _check(fam["mevibe_part-fat-fraction_desc-reconstructed"])

    return out


def get_current_best_VERIDAH(sub) -> Path | None:
    """Return the newest VERIDAH-label JSON for ``sub`` (V2 preferred), or ``None`` if missing.

    The file lives alongside the T2w segmentation under
    ``derivatives_spine_inference_162_sacrumfix/<pfx>/<sub>/T2w/`` and holds the
    ``orig_label -> fpath`` remapping used by :func:`compute_veridah_variants`.
    """
    sub = str(sub).split("_")[0].replace("sub-", "")
    for folder in ("derivatives_spine_inference_162_sacrumfix",):
        for suffix in ("VERIDAH-label-V2", "VERIDAH-label"):
            p = Path(
                f"/DATA/NAS/datasets_processed/NAKO/dataset-nako/{folder}/{sub[:3]}/{sub}/T2w/"
                f"sub-{sub}_sequ-stitched_acq-sag_mod-T2w_seg-vert_desc-{suffix}_stat.json"
            )
            if p.exists():
                return p
    return None


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
    test_key="/102/",  # path matching. if you want on specific us a 6 digits
    decision_cache: DecisionCache | Path | str | None = None,
    corrected_index: dict | Path | str | None = None,
    skip_subject=None,
    vibe_mismatch_snap_dir: Path | str | None = None,
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
        vibe_mismatch_snap_dir: When set, subjects whose VIBE parts don't share a shape are skipped
            (not yielded); a review snapshot is written to this directory as
            ``sub-<id>_vibe-shape-mismatch.jpg`` when that file doesn't already exist.
            Two sub-folders ``accept/`` and ``reject/`` are also created on demand: if the reviewer
            moves the jpg into ``accept/`` the next run auto-answers the grid prompt with "y"
            (resample); if moved into ``reject/`` it auto-answers "n" (drop mismatched keys).

    Yields:
        Dict mapping short keys to ``BIDS_FILE`` entries for one subject.
    """
    if not isinstance(decision_cache, DecisionCache):
        decision_cache = DecisionCache(decision_cache) if decision_cache is not None else DecisionCache()
    cache = decision_cache

    if isinstance(corrected_index, (str, Path)):
        corrected_index = load_corrected_index(Path(corrected_index))
    elif corrected_index is None:
        corrected_index = {}

    gbi = Buffered_BIDS_Global_info(
        datasets=dataset,
        parents=[
            "rawdata",
            "rawdata_stitched",
            "derivatives_Abdominal-Segmentation",
            "derivatives_mevibe",  # partial overlap with Abdominal-Segmentation, but hosts the paraspinal-muscles reconstructed masks
            "derivatives_inversion",
            "derivatives-fullbody-poi",  # fullbody / fov101 / fov102 POIs + registered segmentations on the stitched-water grid
        ],
        filter_file=(lambda x: test_key in str(x)) if test else None,
    )

    for sub, subj in gbi.enumerate_subjects(sort=sort, shuffle=not sort):
        if skip_subject is not None and skip_subject(sub):
            continue
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

        # Extra mevibe segmentations sourced from `derivatives_mevibe`. When the same
        # filename also lives in `derivatives_Abdominal-Segmentation` we auto-prefer
        # the `derivatives_mevibe` copy below to avoid an interactive resolve_pick.
        _MEVIBE_EXTRA_KEYS = (
            "msk_seg-spine_mod-mevibe_part-eco0-opp1",
            "msk_seg-vert_mod-mevibe_part-eco0-opp1",
            "msk_seg-seg-paraspinal-muscles-517-post_mod-mevibe_part-fat-fraction_desc-reconstructed-percent-20",
            "msk_seg-seg-paraspinal-muscles-517-post-figure_mod-mevibe_part-fat-fraction_desc-reconstructed-percent-20",
        )
        keys = ["msk_seg-body-composition_mod-mevibe", *_MEVIBE_EXTRA_KEYS]
        if add_mevibe:
            q = subj.new_query()
            q.filter_format("mevibe")
            # q.filter("sequ", "me1")
            mevibe_fams = list(q.loop_dict(key_addendum=["mod", "part", "desc"]))
            # Drop derivative-only families that don't carry the raw echo images.
            mevibe_fams = [f for f in mevibe_fams if "mevibe_part-eco0-opp1" in f]
            if len(mevibe_fams) > 1:
                labels = [str(f.get("mevibe_part-eco0-opp1", f)) for f in mevibe_fams]
                cached_pick = _cached_pick(cache.get(sub, "mevibe_fam"))
                if cached_pick == "__discard__":
                    mevibe_fams = []
                elif cached_pick in labels:
                    mevibe_fams = [mevibe_fams[labels.index(cached_pick)]]
                else:
                    # auto-accept: always prefer the higher-ID mevibe cluster (lexicographically last label).
                    idx = max(range(len(labels)), key=labels.__getitem__)
                    mevibe_fams = [mevibe_fams[idx]]  # transient: do not save
            for fam in mevibe_fams:
                mevibe_out = get_corrected_mevibe(fam, compute_PDFF=compute_PDFF)
                check_same_grid(cache, sub, "mevibe", mevibe_out, inphase_key="eco3-in1")
                subj_dict = {**mevibe_out, **subj_dict}
                for k, v in fam.items():
                    if k in keys:
                        # For the seg files also present under `derivatives_Abdominal-Segmentation`,
                        # auto-prefer the `derivatives_mevibe` copy (avoids an interactive prompt).
                        if k in _MEVIBE_EXTRA_KEYS and len(v) > 1:
                            preferred = [bf for bf in v if "/derivatives_mevibe/" in _fmt_file(bf)]
                            if preferred:
                                v = preferred
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
            # Fullbody-POI derivatives live on the stitched-water grid; they land in
            # the same vibe family via shared (sub, sequ-stitched, acq, part-water) entities.
            _FULLBODY_POI_KEYS = (
                "poi_seg-fullbody_part-water",
                # "poi_seg-fov101-reg_part-water",
                # "poi_seg-fov102-reg_part-water",
                # "msk_seg-fov101-reg_part-water",
                # "msk_seg-fov102-reg_part-water",
                # "msk_seg-fov101-reg-split-seg_part-water",
                # "msk_seg-fov102-reg-split-seg_part-water",
                # "msk_seg-fov102-reg-split-seg-leg_part-water",
            )
            keys = [
                "vibe_part-inphase",
                "vibe_part-outphase",
                "vibe_part-fat",
                "vibe_part-water",
                "msk_seg-body-composition_mod-vibe",
                *mapping.keys(),
                *_FULLBODY_POI_KEYS,
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
            _vibe_skip_subject = False
            for fam in vibe_fams:
                vibe_by_key: dict = {}
                for _k in (
                    "vibe_part-inphase",
                    "vibe_part-outphase",
                    "vibe_part-fat",
                    "vibe_part-water",
                    "vibe_part-water_desc-reconstructed",
                    "vibe_part-fat_desc-reconstructed",
                ):
                    if fam.get(_k):
                        vibe_by_key[_k] = fam[_k][0]
                if vibe_mismatch_snap_dir is not None:
                    sigs = _vibe_grid_sigs(vibe_by_key)
                    if len(set(sigs.values())) > 1:
                        snap_dir = Path(vibe_mismatch_snap_dir)
                        accept_dir = snap_dir / "accept"
                        reject_dir = snap_dir / "reject"
                        for d in (snap_dir, accept_dir, reject_dir):
                            d.mkdir(parents=True, exist_ok=True)
                        snap_name = f"sub-{sub}_vibe-shape-mismatch.jpg"
                        snap_path = snap_dir / snap_name
                        if (accept_dir / snap_name).exists():
                            # User verified this mismatch is fine — auto-answer "y" (resample).
                            cache.set(sub, "grid_mismatch:vibe", {"decision": "resample", "reason": "accepted-via-snap"})
                        elif (reject_dir / snap_name).exists():
                            # User rejected this subject's VIBE — auto-answer "n" (drop mismatched keys).
                            cache.set(sub, "grid_mismatch:vibe", {"decision": "remove", "reason": "rejected-via-snap"})
                        else:
                            if not snap_path.exists():
                                try:
                                    _save_vibe_shape_mismatch_snapshot(vibe_by_key, snap_path)
                                except Exception as e:  # noqa: BLE001
                                    log.on_warning(f"sub-{sub}: failed to save vibe mismatch snapshot: {e}")
                            log.on_warning(
                                f"sub-{sub}: VIBE grid mismatch {sigs} — awaiting review "
                                f"(move {snap_name} into accept/ or reject/); skipping subject"
                            )
                            _vibe_skip_subject = True
                            break
                check_same_grid(cache, sub, "vibe", vibe_by_key, inphase_key="vibe_part-inphase")
                # Propagate the check's outcome back to ``fam`` so the downstream unpack
                # loop below picks up resampled files (or skips removed keys).
                for _k, bf in vibe_by_key.items():
                    if bf is None:
                        fam.data_dict.pop(_k, None)
                    else:
                        fam[_k] = [bf]
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
            if _vibe_skip_subject:
                continue
        vert, spine, poi = get_current_best_T2w_seg(sub)
        subj_dict["vert"] = vert
        subj_dict["spine"] = spine
        subj_dict["poi"] = poi
        veridah = get_current_best_VERIDAH(sub)
        subj_dict["veridah"] = str(veridah) if veridah is not None else None
        if corrected_index:
            _apply_corrections_to_subj_dict(str(sub), subj_dict, corrected_index)
        verify_missing_images(cache, sub, subj_dict)
        yield subj_dict


allowed_keys = ["sub", "sequ", "ses", "seg", "acq", "chunk", "part", "mod", "desc", "rec"]


def _is_grid_only_json(path: Path | str) -> bool:
    """True when ``path`` is a JSON sidecar whose only key is ``"grid"``.

    Such a file was written by :func:`_add_grid_info_to_json` on a sidecar that
    did not exist before — it carries no real metadata, only cached grid info
    for the associated NIfTI.  We do not want these propagating into the
    canonical dataset alongside the .nii.gz.
    """
    try:
        content = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(content, dict) and set(content.keys()) == {"grid"}


_CANONICAL_DONE_ROOT = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako-canonical/.hardlink_done")

_VIBE_MISMATCH_SNAP_DIR = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako-canonical/snaps/vibe-missmatch")


def _vibe_grid_sigs(vibe_by_key: dict) -> dict[str, str]:
    """Return ``{key: grid-signature-string}`` for each present VIBE part.

    Mirrors ``check_same_grid``'s comparison: reads ``bf.get_grid_info()`` and
    stringifies it, skipping msk entries and files that have no NIfTI. Two
    signatures being unequal is exactly what would cause ``check_same_grid``
    to prompt the user.
    """
    sigs: dict[str, str] = {}
    for k, bf in vibe_by_key.items():
        if bf is None:
            continue
        if getattr(bf, "format", None) == "msk" or k.startswith("msk"):
            continue
        get_nii_file = getattr(bf, "get_nii_file", None)
        if callable(get_nii_file) and get_nii_file() is None:
            continue
        try:
            g = bf.get_grid_info()
        except Exception:  # noqa: BLE001
            continue
        if g is None:
            continue
        sigs[k] = str(g)
    return sigs


def _save_vibe_shape_mismatch_snapshot(vibe_by_key: dict, out_path: Path) -> None:
    """Save a review snapshot of VIBE parts whose grids don't agree.

    Renders one sagittal+coronal frame per present VIBE part (in/out/water/fat),
    titled with the part's shape, so a reviewer can eyeball what's off.
    """
    from TPTBox.spine.snapshot2D import Snapshot_Frame, create_snapshot

    frames = []
    for k in ("vibe_part-inphase", "vibe_part-outphase", "vibe_part-water", "vibe_part-fat"):
        bf = vibe_by_key.get(k)
        if bf is None:
            continue
        try:
            shape = tuple(bf.get_grid_info().shape)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            shape = None
        frames.append(
            Snapshot_Frame(
                image=bf,
                mode="MRI",
                sagittal=True,
                coronal=True,
                axial=False,
                crop_msk=False,
                title=f"{k} shape={shape}",
            )
        )
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    create_snapshot(snp_path=[out_path], frames=frames)


def _hard_link_done_marker(sub: str) -> Path:
    return _CANONICAL_DONE_ROOT / f"{sub}.done"


def is_hard_linked(sub: str) -> bool:
    """Return True when :func:`hard_link` has completed successfully for ``sub``."""
    return _hard_link_done_marker(str(sub)).exists()


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
        # MEVIBE extras from `derivatives_mevibe`
        "msk_seg-spine_mod-mevibe_part-eco0-opp1",
        "msk_seg-vert_mod-mevibe_part-eco0-opp1",
        "msk_seg-seg-paraspinal-muscles-517-post_mod-mevibe_part-fat-fraction_desc-reconstructed-percent-20",
        "msk_seg-seg-paraspinal-muscles-517-post-figure_mod-mevibe_part-fat-fraction_desc-reconstructed-percent-20",
        "vibeseg100",  # vibe
        "MRSegmentator",  # vibe
        "msk_seg-body-composition_mod-vibe",  # vibe
        "roi",  # vibe
        # Fullbody-POI derivatives (stitched-water grid)
        "poi_seg-fullbody_part-water",
        "poi_seg-fov101-reg_part-water",
        "poi_seg-fov102-reg_part-water",
        "msk_seg-fov101-reg_part-water",
        "msk_seg-fov102-reg_part-water",
        "msk_seg-fov101-reg-split-seg_part-water",
        "msk_seg-fov102-reg-split-seg_part-water",
        "msk_seg-fov102-reg-split-seg-leg_part-water",
        "vert",  # t2w (stiched)
        "spine",  # t2w (stiched)
        "poi",  # t2w (stiched)
        "veridah",  # VERIDAH enumeration-anomaly relabeling (V2 preferred)
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
            if bf is None or bf == "":
                continue
            info = {"run": None}
            if isinstance(bf, str):
                bf = BIDS_FILE(bf, dataset)
            bf.info.pop("run", None)
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
            # Build the set of source extensions we'd actually hard-link.
            srcs: dict[str, Path] = {ext: src for ext, src in bf.file.items() if Path(src).exists()}
            # For segmentations (msk), skip auto-generated grid-only JSON sidecars.
            if bf.format == "msk" and "json" in srcs and _is_grid_only_json(srcs["json"]):
                srcs.pop("json")
            if not srcs:
                continue
            # Compute the real target paths (per extension) and skip if all already exist.
            base = str(new_path)[: -len(".nii.gz")]
            targets = {ext: Path(base + "." + ext) for ext in srcs}
            if all(t.exists() for t in targets.values()):
                continue
            new_path.parent.mkdir(parents=True, exist_ok=True)
            # Temporarily restrict bf.file to just the sources we want linked, then restore.
            original_file = bf.file.copy()
            bf._file = srcs  # bypass the property's auto-discovery (already _checked=True)
            try:
                bf.symlink_files(new_path, hard_link=True)
                for t in targets.values():
                    print(t)
            finally:
                bf._file = original_file
    leftover = {k: v for k, v in d.items() if v is not None}
    assert len(leftover) == 0, leftover
    marker = _hard_link_done_marker(str(subj))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


# ---------------------------------------------------------------------------
# nako_export.py corrected-outputs integration
# ---------------------------------------------------------------------------
#
# nako_export.py writes fetswap-corrected VIBE / MEVIBE files under
# ``dataset-nako-canonical/rawdata-corrected/`` (see the docstring at the top
# of that script). Because it runs incrementally, at any given moment only
# some subjects/chunks/sequs are corrected. We track what's replaced in a
# side-JSON so downstream consumers know which files to swap for the
# corrected ones — and, for VIBE, when to re-stitch water/fat because a
# per-chunk correction breaks the pre-existing stitched raw volume.

_CANONICAL_ROOT = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako-canonical")
_CORRECTED_ROOT = _CANONICAL_ROOT / "rawdata-corrected"
_DEFAULT_CORRECTED_INDEX = Path(__file__).with_name("_load_nako_wh_corrected.json")


def build_corrected_index(
    corrected_root: Path = _CORRECTED_ROOT,
    out_json: Path = _DEFAULT_CORRECTED_INDEX,
    verbose: bool = True,
) -> dict:
    """Scan ``rawdata-corrected/`` and record which VIBE chunks / MEVIBE sequs nako_export replaced.

    Layout of the produced JSON::

        {"<sub>": {"vibe": {"corrected_chunks": [1, 2, ...]}, "mevibe": {"corrected_sequs": ["4", "5", ...]}}}

    The JSON is rewritten from scratch on every call — nako_export runs
    incrementally, so we always take disk state as the truth.
    """
    import re

    index: dict = {}
    if not corrected_root.exists():
        log.on_warning(f"corrected root {corrected_root} does not exist; writing empty index")
    else:
        for sub_dir in sorted(corrected_root.glob("*/*")):
            if not (sub_dir.is_dir() and sub_dir.name.isdigit()):
                continue
            sub = sub_dir.name
            entry: dict = {}
            vibe_dir = sub_dir / "vibe"
            if vibe_dir.exists():
                chunks: set[int] = set()
                for p in vibe_dir.glob(f"sub-{sub}_acq-ax_chunk-*_part-water_desc-corrected_vibe.nii.gz"):
                    m = re.search(r"chunk-(\d+)", p.name)
                    if m:
                        chunks.add(int(m.group(1)))
                if chunks:
                    entry["vibe"] = {"corrected_chunks": sorted(chunks)}
            mevibe_dir = sub_dir / "mevibe"
            if mevibe_dir.exists():
                sequs: set[str] = set()
                for p in mevibe_dir.glob(f"sub-{sub}_sequ-*_acq-ax_part-water_desc-corrected_mevibe.nii.gz"):
                    m = re.search(r"sequ-([^_]+)", p.name)
                    if m:
                        sequs.add(m.group(1))
                if sequs:
                    entry["mevibe"] = {"corrected_sequs": sorted(sequs)}
            if entry:
                index[sub] = entry
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True))
    tmp.replace(out_json)
    if verbose:
        n_vibe = sum(1 for v in index.values() if "vibe" in v)
        n_mevibe = sum(1 for v in index.values() if "mevibe" in v)
        log.on_log(f"corrected index: {len(index)} subjects ({n_vibe} vibe, {n_mevibe} mevibe) -> {out_json}")
    return index


def load_corrected_index(path: Path = _DEFAULT_CORRECTED_INDEX) -> dict:
    """Read the JSON produced by :func:`build_corrected_index`, or ``{}`` if missing."""
    if not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        log.on_warning(f"corrupt corrected index at {path}; ignoring")
        return {}


def _restitch_vibe_water_fat(
    sub: str,
    corrected_chunks: list[int],
    dataset: Path = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako"),
    scratch_root: Path = _RESAMPLE_TMP_ROOT,
) -> dict[str, Path]:
    """Re-stitch VIBE water & fat for ``sub`` mixing corrected chunks with raw ones.

    Any chunk listed in ``corrected_chunks`` is pulled from
    ``rawdata-corrected/…/vibe/…_desc-corrected_vibe.nii.gz``; everything else
    comes from ``rawdata/…/vibe/…_vibe.nii.gz``.  The stitched output goes
    under ``$TPTBOX_RESAMPLED_SCRATCH/rawdata_stitched/…/vibe/…``.

    The corrected-chunk set is baked into the output filename so a later
    ``nako_export`` pass that corrects more chunks produces a distinct file
    (no stale cache).  Returns ``{"water": Path, "fat": Path}`` (partial when
    stitching fails for one part).
    """
    import re

    from TPTBox.stitching import stitching as _stitching_fn

    ss = sub[:3]
    corr_dir = _CORRECTED_ROOT / ss / sub / "vibe"
    raw_dir = dataset / "rawdata" / ss / sub / "vibe"
    if not raw_dir.exists():
        log.on_warning(f"sub-{sub}: no raw vibe dir at {raw_dir}; skipping re-stitch")
        return {}

    all_chunks: set[int] = set()
    for p in raw_dir.glob(f"sub-{sub}_acq-ax_chunk-*_part-water_vibe.nii.gz"):
        m = re.search(r"chunk-(\d+)", p.name)
        if m:
            all_chunks.add(int(m.group(1)))
    if not all_chunks:
        log.on_warning(f"sub-{sub}: no raw vibe chunks found under {raw_dir}")
        return {}
    corrected_set = set(corrected_chunks) & all_chunks
    chunks_sorted = sorted(all_chunks)
    tag = "corrected" + "".join(f"C{c}" for c in sorted(corrected_set))

    out_dir = scratch_root / "rawdata_stitched" / ss / sub / "vibe"
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    for part in ("water", "fat"):
        out_path = out_dir / f"sub-{sub}_sequ-stitched_acq-ax_part-{part}_desc-{tag}_vibe.nii.gz"
        if out_path.exists():
            result[part] = out_path
            continue
        images: list[Path] = []
        for c in chunks_sorted:
            if c in corrected_set:
                p = corr_dir / f"sub-{sub}_acq-ax_chunk-{c}_part-{part}_desc-corrected_vibe.nii.gz"
                if p.exists():
                    images.append(p)
                    continue
            p = raw_dir / f"sub-{sub}_acq-ax_chunk-{c}_part-{part}_vibe.nii.gz"
            if p.exists():
                images.append(p)
        if len(images) < 2:
            log.on_warning(f"sub-{sub} part-{part}: only {len(images)} chunk(s) available; cannot stitch")
            continue
        try:
            _stitching_fn(
                [str(p) for p in images],
                str(out_path),
                is_seg=False,
                bias_field=False,
                verbose=False,
                verbose_stitching=False,
            )
        except Exception as e:  # noqa: BLE001
            log.on_warning(f"sub-{sub} part-{part}: stitching failed: {type(e).__name__}: {e}")
            continue
        result[part] = out_path
    return result


def _corrected_bids_file(path: Path, dataset_root: Path) -> BIDS_FILE:
    return BIDS_FILE(str(path), str(dataset_root), verbose=False)


def _apply_corrections_to_subj_dict(sub: str, subj_dict: dict, index: dict) -> dict:
    """Swap ``subj_dict`` entries for their nako_export-corrected counterparts, if any.

    Returns a small report ``{"mevibe": [sequs...], "vibe": [chunks...]}`` describing
    what was actually replaced (useful for logging / hard-link verification).
    """
    report: dict[str, list] = {"mevibe": [], "vibe": []}
    entry = index.get(str(sub))
    if not entry:
        return report

    # ---- MEVIBE: whole-sequ replacement (part-fat replaces the buggy `mevibe_part-fat` slot). ----
    if "mevibe" in entry:
        for sequ in entry["mevibe"]["corrected_sequs"]:
            corr_dir = _CORRECTED_ROOT / str(sub)[:3] / str(sub) / "mevibe"
            pdff = corr_dir / f"sub-{sub}_sequ-{sequ}_acq-ax_part-fat-fraction_desc-corrected_mevibe.nii.gz"
            if pdff.exists():
                subj_dict["mevibe_part-fat"] = _corrected_bids_file(pdff, _CANONICAL_ROOT)
                report["mevibe"].append(sequ)

    # ---- VIBE: re-stitch water + fat from corrected + raw chunks. ----
    if "vibe" in entry:
        stitched = _restitch_vibe_water_fat(str(sub), entry["vibe"]["corrected_chunks"])
        for part in ("water", "fat"):
            if part in stitched:
                subj_dict[f"vibe_part-{part}"] = _corrected_bids_file(stitched[part], _RESAMPLE_TMP_ROOT)
        report["vibe"] = list(entry["vibe"]["corrected_chunks"])

    return report


def verify_hardlink(sub: str, corrected_index_path: Path = _DEFAULT_CORRECTED_INDEX) -> None:
    """Run the loop for a single subject, apply corrections, then trace what
    :func:`hard_link` *would* do and check every source exists and every target
    would land on the same filesystem as its source (so :func:`os.link` won't
    hit ``EXDEV``).  Prints one line per file — no writes are performed.
    """
    index = load_corrected_index(corrected_index_path)
    sub = str(sub)
    ss = sub[:3]
    hits = 0
    for d in loop_over_repaired_nako(test=True, test_key=f"/{ss}/{sub}", corrected_index=index):
        if str(d.get("id", "")) != sub:
            continue
        hits += 1
        subj_report = _apply_corrections_to_subj_dict(sub, d, index)
        print(f"[verify-hardlink] sub-{sub}  corrections applied: {subj_report}")

        def _check(bf, parent: str, info: dict | None = None) -> None:
            if bf is None:
                return
            if isinstance(bf, str):
                bf = BIDS_FILE(bf, "/DATA/NAS/datasets_processed/NAKO/dataset-nako/", verbose=False)
            src = bf.get_nii_file()
            try:
                target = bf.get_changed_path(
                    "nii.gz",
                    bf.format,
                    parent=parent,
                    info=info or {},
                    dataset_path=str(_CANONICAL_ROOT),
                )
            except Exception as e:  # noqa: BLE001
                print(f"  ERR    {bf}: get_changed_path failed: {e}")
                return
            src_exists = src is not None and Path(src).exists()
            same_fs = src is not None and Path(src).stat().st_dev == _CANONICAL_ROOT.stat().st_dev if src_exists else False
            status = "ok" if src_exists and same_fs else ("cross-fs" if src_exists else "src-missing")
            print(f"  [{status:>10s}]  {src}  ->  {target}")

        for _key, t2w in (d.get("t2w_chunk") or {}).items():
            if t2w:
                _check(t2w[0], parent="rawdata", info={"ses": "baseline"})
        seg_keys = (
            "msk_seg-body-composition_mod-mevibe",
            "vibeseg100",
            "MRSegmentator",
            "msk_seg-body-composition_mod-vibe",
            "roi",
            "vert",
            "spine",
            "poi",
        )
        img_keys = (
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
        )
        for keys, parent in ((img_keys, "rawdata"), (seg_keys, "derivatives")):
            for k in keys:
                _check(d.get(k), parent=parent, info={"run": None})
    if hits == 0:
        print(f"[verify-hardlink] sub-{sub} was not produced by loop_over_repaired_nako (test_key filter?)")


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
        # Skip segmentations (msk-format files) — grid prewarm is only for image volumes.
        if getattr(v, "format", None) == "msk":
            return None
        get_nii_file = getattr(v, "get_nii_file", None)
        if get_nii_file is None:
            return None
        try:
            p = get_nii_file()
        except Exception:  # noqa: BLE001
            return None
        # JSON-only BIDS entries (e.g. fullbody POI) have no nii.gz — nothing to warm.
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

    with _non_interactive_mode(), ProcessPoolExecutor(max_workers=num_workers) as pool:
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

    parser = argparse.ArgumentParser(description="NAKO helpers: hard-link, prewarm grid info, or track nako_export corrections.")
    parser.add_argument(
        "--precompute-grid",
        action="store_true",
        help="Prewarm bf.get_grid_info() JSON caches in parallel processes instead of hard-linking.",
    )
    parser.add_argument(
        "--build-corrected-index",
        action="store_true",
        help="Scan dataset-nako-canonical/rawdata-corrected/ and (re)write the corrections JSON.",
    )
    parser.add_argument(
        "--verify-hardlink",
        metavar="SUB",
        default=None,
        help="Trace hard_link()'s planned links for one subject; check src exists + same fs as target.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: cpu_count-1).")
    parser.add_argument(
        "--vibe-mismatch-snaps",
        nargs="?",
        const=str(_VIBE_MISMATCH_SNAP_DIR),
        # default=None,
        default=str(_VIBE_MISMATCH_SNAP_DIR),
        metavar="DIR",
        help="Skip subjects whose VIBE parts have grid mismatches; write a review .jpg to DIR "
        f"(defaults to {_VIBE_MISMATCH_SNAP_DIR}) unless one already exists there. "
        "Pass '' to disable.",
    )
    args = parser.parse_args()
    test = False

    if args.build_corrected_index:
        build_corrected_index()
    elif args.verify_hardlink is not None:
        verify_hardlink(args.verify_hardlink)
    elif args.precompute_grid:
        precompute_grid_info_parallel(num_workers=args.workers, test=test)
    else:
        corrected = load_corrected_index()
        for d in loop_over_repaired_nako(
            test=test,
            corrected_index=corrected,
            skip_subject=is_hard_linked,
            vibe_mismatch_snap_dir=args.vibe_mismatch_snaps or None,
        ):
            hard_link(d)
