"""Post-hoc BIDS renamer for datasets already extracted from DICOM.

Walk a BIDS-style dataset root, re-run
:func:`~TPTBox.core.dicom.dicom_header_to_keys.extract_keys_from_json` on
every sidecar JSON, and rename each file family (`nii.gz` + `json` + `.txt` +
…) to the new BIDS path the current key logic produces. Two use cases:

1. **Sanitising illegal subject ids.** BIDS forbids ``_`` inside an entity
   value (it is the entity separator). Any subject imported with an id like
   ``180217_375491`` produced ``sub-180217_375491_ses-...`` filenames that
   downstream BIDS parsers reject or misinterpret. Leading ``-`` after
   ``sub-`` is the same category of problem. Sanitisation strips those.
2. **Compact numeric ids.** Long random-looking subject ids (``sub-0bCAY6ARpDo``)
   are unhandy; passing ``subject_prefix="ID"`` rewrites them to
   ``sub-ID001``, ``sub-ID002``, … The mapping is written to
   ``<dataset_root>/<info_dir>/subject_map.tsv`` so the original ids stay
   recoverable.

Additionally, any change picked up upstream in ``extract_keys_from_json``
(e.g. modality-fallback improvements, view/laterality/BodyPartExamined
extraction) becomes effective on-disk without a full DICOM re-extraction —
the JSON sidecar carries the same header dict the DICOM did.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from TPTBox import BIDS_FILE, Print_Logger
from TPTBox.core.dicom.dicom2nii_utils import load_json
from TPTBox.core.dicom.dicom_header_to_keys import extract_keys_from_json


class _FakeDicom:
    """Minimal stand-in for a pydicom Dataset.

    Supplies just the attributes ``extract_keys_from_json`` reads before the
    plane branch.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename


class _FakeDicomList:
    """Tiny list-shaped wrapper for :func:`extract_keys_from_json`.

    Steers upstream away from loading the NIfTI just to redo plane detection.
    Supports the two accesses upstream cares about:
    * ``lst[0].filename`` — used by the override-subject-name path.
    * Iteration — used by ``get_plane_dicom``; we yield nothing so it exits
      via the empty-affine `IndexError`, which our expanded silent-catch in
      ``get_plane_dicom`` turns into ``None`` without noise.
    """

    def __init__(self, json_path: Path) -> None:
        self._stub = _FakeDicom(str(json_path))

    def __getitem__(self, _i: int) -> _FakeDicom:
        return self._stub

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


def _plane_from_grid(grid: dict, hires_threshold: float = 0.8) -> str | None:
    """Compute the acquisition-plane label from a sidecar ``grid`` dict.

    Mirrors :func:`~TPTBox.core.dicom.dicom_header_to_keys.get_plane_dicom`'s
    core zoom/axcodes logic but reads ``spacing`` + ``orientation`` directly
    from the sidecar (populated by ``_add_grid_info_to_json`` during
    extraction) instead of re-loading the NIfTI. Lets the renamer stay
    header-only and process a full dataset in seconds instead of hours.
    """
    try:
        zooms = np.asarray(grid["spacing"], dtype=float)
        orient = grid["orientation"]
    except (KeyError, TypeError):
        return None
    if zooms.size < 3 or not isinstance(orient, (list, tuple)) or len(orient) < 3:
        return None
    zooms = np.where(zooms == 0, 1.0, zooms)
    if hires_threshold is not None:
        zooms = np.maximum(zooms, hires_threshold)
    zms = np.around(zooms, 1)
    plane_dict = {"S": "ax", "I": "ax", "L": "sag", "R": "sag", "A": "cor", "P": "cor"}
    ix_max = zms == np.amax(zms)
    num_max = int(np.count_nonzero(ix_max))
    axc = np.array(orient[:3])
    if num_max == 2:
        return plane_dict.get(axc[~ix_max][0])
    if num_max == 1:
        return plane_dict.get(axc[ix_max][0])
    return "iso"


logger = Print_Logger()


# BIDS entity value grammar allows [A-Za-z0-9] plus limited punctuation.
# `_` is the entity separator and is outright forbidden inside a value.
# A leading `-` inside a value is not spec-forbidden but the parser treats
# it as a new empty-value entity, which corrupts the split.
_ILLEGAL_IN_VALUE = re.compile(r"[_]+")


def _sanitize_entity_value(raw: str) -> str:
    """Return *raw* with characters that break BIDS entity parsing removed.

    ``_`` (entity separator) is dropped, and a leading ``-`` (which would look
    like an empty preceding entity) is stripped. Empty results become
    ``"unnamed"`` so callers never build ``sub-_ses-...``-style filenames.
    """
    clean = _ILLEGAL_IN_VALUE.sub("", raw).lstrip("-")
    return clean or "unnamed"


def _iter_json_sidecars(root: Path) -> Iterable[Path]:
    """Yield every sidecar JSON under *root*, skipping cache / hidden dirs."""
    for p in sorted(root.rglob("*.json")):
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        # Skip our own translation file if it happens to live under root.
        if p.name == "subject_map.tsv":
            continue
        yield p


def _list_subject_folders(root: Path) -> list[Path]:
    """Return every ``sub-*`` folder directly under *root*, sorted."""
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-"))


def _current_sub_id(folder: Path) -> str:
    """Strip the leading ``sub-`` prefix from a subject folder name."""
    return folder.name[len("sub-") :]


def _read_existing_subject_map(dataset_root: Path, info_dir: str) -> dict[str, str]:
    """Load a previously written ``subject_map.tsv`` if one exists.

    Returns ``old_sub -> new_sub`` from every row (session columns are
    ignored — subject-level identity is the only thing we need to keep
    re-runs stable). Missing file → empty dict.
    """
    path = dataset_root / info_dir / "subject_map.tsv"
    if not path.is_file():
        return {}
    saved: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            old = row.get("old_sub")
            new = row.get("new_sub")
            if old and new:
                saved[old] = new
    return saved


def _build_subject_map(
    subject_folders: list[Path],
    subject_prefix: str | None,
    subject_number_width: int,
    existing_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Map every existing subject id to its new BIDS-legal id.

    * ``subject_prefix=None`` — sanitise in place (drop ``_``, leading ``-``).
    * ``subject_prefix="ID"`` — reassign each subject to ``ID001``, ``ID002``,
      … in the natural sort order of the folder listing. Prefix itself is
      sanitised the same way so the caller can't accidentally reintroduce
      ``_`` via the prefix.

    ``existing_map`` (loaded from ``<info_dir>/subject_map.tsv``) makes
    re-runs safe. Folders whose id is already a value in ``existing_map``
    (i.e. they were renamed on a previous pass) map to themselves; folders
    whose id is still a key get their previously assigned new id back;
    only truly fresh subjects get a new number, taken from the next slot
    beyond the highest already-used one. Same semantics apply to the
    sanitise-in-place branch — an already-sanitised id stays as-is.
    """
    existing_map = existing_map or {}
    already_new: set[str] = set(existing_map.values())
    mapping: dict[str, str] = {}
    if subject_prefix is None:
        for folder in subject_folders:
            old = _current_sub_id(folder)
            if old in existing_map:
                mapping[old] = existing_map[old]
            elif old in already_new:
                mapping[old] = old
            else:
                mapping[old] = _sanitize_entity_value(old)
        return mapping
    clean_prefix = _sanitize_entity_value(subject_prefix)
    used_numbers: set[int] = set()
    id_re = re.compile(rf"^{re.escape(clean_prefix)}(\d+)$")
    for new in already_new:
        m = id_re.match(new)
        if m:
            used_numbers.add(int(m.group(1)))
    for folder in subject_folders:
        old = _current_sub_id(folder)
        if old in existing_map:
            mapping[old] = existing_map[old]
            m = id_re.match(existing_map[old])
            if m:
                used_numbers.add(int(m.group(1)))
            continue
        m = id_re.match(old)
        if m:  # folder is already numbered — keep it as identity.
            mapping[old] = old
            used_numbers.add(int(m.group(1)))
    next_free = (max(used_numbers) + 1) if used_numbers else 1
    for folder in subject_folders:
        old = _current_sub_id(folder)
        if old in mapping:
            continue
        while next_free in used_numbers:
            next_free += 1
        mapping[old] = f"{clean_prefix}{next_free:0{subject_number_width}d}"
        used_numbers.add(next_free)
        next_free += 1
    return mapping


def _write_subject_map(
    dataset_root: Path,
    info_dir: str,
    mapping: dict[str, str],
    session_mapping: dict[tuple[str, str], str] | None = None,
) -> Path:
    """Persist ``old_sub -> new_sub`` (plus optional session mapping) as TSV.

    TSV columns: ``old_sub``, ``new_sub``, ``old_ses``, ``new_ses``. When no
    session sanitisation happened the ses columns are left empty for the
    subject-level row and one row per subject is written; otherwise there is
    one row per (subject, session) pair. Written to
    ``<dataset_root>/<info_dir>/subject_map.tsv``; existing content is
    overwritten so re-running the renamer keeps the file authoritative.
    """
    out_dir = dataset_root / info_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "subject_map.tsv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["old_sub", "new_sub", "old_ses", "new_ses"])
        seen_pairs: set[tuple[str, str]] = set()
        if session_mapping:
            for (old_sub, old_ses), new_ses in sorted(session_mapping.items()):
                new_sub = mapping.get(old_sub, old_sub)
                writer.writerow([old_sub, new_sub, old_ses, new_ses])
                seen_pairs.add((old_sub, old_ses))
        for old_sub, new_sub in sorted(mapping.items()):
            if any(pair[0] == old_sub for pair in seen_pairs):
                continue
            writer.writerow([old_sub, new_sub, "", ""])
    return path


def _new_bids_path_for(
    json_path: Path,
    dataset_root: Path,
    parent: str,
    new_sub_id: str,
    session: bool,
    make_subject_chunks: int,
) -> tuple[Path, dict]:
    """Re-derive the target BIDS path from an existing sidecar JSON.

    The heavy lifting is delegated to :func:`extract_keys_from_json` +
    :func:`_generate_bids_path`, exactly as
    :func:`~TPTBox.core.dicom.dicom_extract._from_dicom_to_nii` uses them at
    extraction time. Two callback tweaks let us reuse the extract logic here:

    * ``override_subject_name`` returns *new_sub_id* verbatim, bypassing the
      DICOM-header derivation of the subject id.
    * ``dcm_data_l`` is set to the NIfTI path on disk so
      :func:`get_plane_dicom`'s ``to_nii`` branch runs — no DICOM headers are
      touched.

    Returns ``(new_json_path, new_keys)``; the caller applies the rename with
    :meth:`BIDS_FILE.rename_files`.
    """
    raw_json = load_json(json_path)
    grid = raw_json.get("grid")
    simp_json = {k: v for k, v in raw_json.items() if k != "grid"}
    # Pre-compute the plane from the sidecar `grid` block; steer
    # `extract_keys_from_json` away from re-loading the NIfTI just to redo
    # what the sidecar already recorded. We hand a tiny stub list in as
    # `dcm_data_l` — it responds to the two accesses upstream cares about
    # (``dcm_data_l[0].filename`` for the override-subject-name callback and
    # iteration for the plane detector), and the plane detector then returns
    # None silently under our expanded exception handler.
    pre_plane = _plane_from_grid(grid) if isinstance(grid, dict) else None
    dcm_stub = _FakeDicomList(json_path)
    (mri_format, keys, _ending) = extract_keys_from_json(
        simp_json,
        dcm_stub,  # type: ignore[arg-type]
        session=session,
        override_subject_name=lambda _sj, _p: new_sub_id,
    )
    if pre_plane is not None:
        keys["acq"] = pre_plane
    # Build the target path ourselves. `_generate_bids_path` composes via
    # `BIDS_FILE.get_changed_bids` which decomposes the current path to
    # infer the folder layout — on flat datasets (parent="") that swallows
    # the `sub-*` folder into the "parent" slot and drops it when we then
    # override parent. Composing `path=sub-<new>/ses-<X>` ourselves keeps
    # the layout regardless of how the source dataset happens to be laid
    # out on disk.
    sub = keys.get("sub") or new_sub_id
    ses = keys.get("ses")
    sub_folder = f"sub-{sub}"
    if make_subject_chunks:
        sub_folder = f"{sub[:make_subject_chunks]}/{sub_folder}"
    path = f"{sub_folder}/ses-{ses}" if ses else sub_folder
    src = BIDS_FILE(json_path, dataset_root, verbose=False)
    new_bids = src.get_changed_bids(
        file_type="json",
        parent=parent,
        path=path,
        additional_folder=mri_format,
        bids_format=mri_format,
        make_parent=False,
        info=keys,
        non_strict_mode=True,
    )
    return Path(new_bids.file["json"]), keys


def _rename_family(json_path: Path, new_json_path: Path, dataset_root: Path, dry_run: bool) -> list[tuple[Path, Path]]:
    """Move every sibling of *json_path* to the *new_json_path* stem.

    Extension detection reuses :meth:`BIDS_FILE.file`, which collects
    ``.json`` / ``.nii.gz`` / ``.mrk.json`` etc. as one BIDS entry.
    In addition we glob the JSON's folder for any file sharing the exact
    base stem — ``BIDS_FILE.file`` currently misses DWI companion files
    (``.bval`` / ``.bvec``) and other non-standard extensions that live
    next to the primary NIfTI. Same-stem globbing is safe because BIDS
    guarantees only truly-paired sidecars share the stem.
    """
    if json_path == new_json_path:
        return []
    bf_current = BIDS_FILE(json_path, dataset_root, verbose=False)
    ext_paths: dict[str, Path] = {ext: Path(p) for ext, p in bf_current.file.items()}
    # Add same-stem companions that BIDS_FILE didn't pick up. json_path.name
    # ends with e.g. "…_dwi.json" — strip the trailing ".json" to get the
    # stem the .bval / .bvec / other companions share.
    stem = json_path.name.removesuffix(".json")
    for sibling in json_path.parent.iterdir():
        if not sibling.is_file():
            continue
        if sibling.name == json_path.name:
            continue
        if not sibling.name.startswith(stem + "."):
            continue
        ext = sibling.name[len(stem) + 1 :]
        ext_paths.setdefault(ext, sibling)
    moves: list[tuple[Path, Path]] = []
    new_stem = str(new_json_path).removesuffix(".json")
    for ext, src in ext_paths.items():
        dst = Path(f"{new_stem}.{ext}")
        if not Path(src).exists():
            continue
        moves.append((Path(src), dst))
    if dry_run:
        return moves
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.resolve() != src.resolve():
            logger.on_warning(f"target {dst} already exists; skipping {src}")
            continue
        src.rename(dst)
    return moves


_SEQU_ENTITY_RE = re.compile(r"_sequ-([^_\s.]+)")


def _leftover_move_plan(
    scan_root: Path,
    subject_map: dict[str, str],
    parent: str | None,
) -> list[tuple[Path, Path]]:
    """Compute (src, dst) moves for files left behind after the JSON pass.

    After the JSON-driven Pass 1, an old ``sub-<OLD>/ses-<X>/`` folder can
    still hold companion files whose primary sidecar has already migrated —
    typical culprits are ``.bval`` / ``.bvec`` sitting apart from their
    ``_dwi.nii.gz`` and DWI-derived ``_dwi_ADC.nii.gz`` maps. Move them under
    ``sub-<NEW>/`` too, preferring to co-locate with their new-side twin
    (matched by ``sequ-<N>``) so BIDS stem-linkage stays intact. Files that
    can't be twinned drop at the session level of the new subject folder as
    a safe fallback.
    """
    del parent  # kept for signature symmetry; scan_root already resolves it
    moves: list[tuple[Path, Path]] = []
    for old_sub, new_sub in subject_map.items():
        if old_sub == new_sub:
            continue
        old_dir = scan_root / f"sub-{old_sub}"
        new_dir = scan_root / f"sub-{new_sub}"
        if not old_dir.is_dir():
            continue
        for src in old_dir.rglob("*"):
            if not src.is_file():
                continue
            # Skip anything a re-run of the JSON pass would handle (we don't
            # want to race the pass 1 output here).
            if src.suffix == ".json":
                continue
            m = _SEQU_ENTITY_RE.search(src.name)
            sequ = m.group(1) if m else None
            twin_stem: str | None = None
            target_dir: Path | None = None
            if sequ and new_dir.is_dir():
                for twin in new_dir.rglob(f"*sequ-{sequ}*.json"):
                    if not twin.is_file():
                        continue
                    twin_stem = twin.name.removesuffix(".json")
                    target_dir = twin.parent
                    break
            if twin_stem is not None and target_dir is not None:
                # Extract the orphan's tail after the sequ-<N> segment. That
                # tail carries the old format label plus any trailing suffix
                # (`_ADC`, extensions like `.bval` / `.bvec` / `.nii.gz`).
                # Replacing the twin's stem preserves the format-label change
                # while keeping DWI derivatives glued to the twin.
                seq_marker = f"_sequ-{sequ}"
                idx = src.name.find(seq_marker)
                if idx == -1:
                    continue
                after_sequ = src.name[idx + len(seq_marker) :]
                # after_sequ starts with either `_<oldformat>...` or `.<ext>`.
                # Strip the leading `_<oldformat>` so `_dwi.bval` becomes
                # `.bval` and `_dwi_ADC.nii.gz` becomes `_ADC.nii.gz` — both
                # then splice cleanly onto the twin stem.
                if after_sequ.startswith("_"):
                    body, sep, rest = after_sequ.partition(".")
                    old_fmt_parts = body.split("_", 2)
                    # body = "_<oldformat>" or "_<oldformat>_<suffix>"
                    if len(old_fmt_parts) >= 2:
                        suffix = ("_" + old_fmt_parts[2]) if len(old_fmt_parts) == 3 else ""
                        after_sequ = f"{suffix}.{rest}" if sep else suffix
                new_name = twin_stem + after_sequ
                dst = target_dir / new_name
            else:
                # Fallback: mirror the old ses-* folder under sub-<NEW>/ and
                # just swap the subject prefix in the filename.
                try:
                    rel = src.relative_to(old_dir)
                except ValueError:
                    continue
                new_name = src.name.replace(f"sub-{old_sub}", f"sub-{new_sub}", 1)
                dst = new_dir / rel.parent / new_name
            if src == dst or dst.exists():
                continue
            moves.append((src, dst))
    return moves


def _prune_empty_folders(root: Path) -> int:
    """Remove now-empty ``sub-*`` / ``ses-*`` subtrees left by the rename."""
    removed = 0
    for folder in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        rel = folder.relative_to(root)
        if not rel.parts:
            continue
        try:
            folder.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


def rerun_bids_naming(
    dataset_root: Path | str,
    parent: str | None = "rawdata",
    subject_prefix: str | None = None,
    subject_number_width: int = 3,
    dry_run: bool = True,
    info_dir: str = "info",
    session: bool = True,
    make_subject_chunks: int = 0,
    verbose: bool = True,
) -> dict[str, list[tuple[Path, Path]]]:
    """Rename an already-extracted BIDS dataset from its sidecar JSONs.

    Walks ``<dataset_root>/<parent>/`` (or ``<dataset_root>`` if ``parent`` is
    ``None`` / empty — matches datasets where ``sub-*`` folders sit directly
    under the dataset root), builds an ``old_sub -> new_sub`` map, then for
    every JSON sidecar re-derives its target BIDS path via
    :func:`extract_keys_from_json` + :func:`_generate_bids_path` and renames
    each file family to the new path.

    Args:
        dataset_root: Dataset root that contains the ``parent`` folder (or
            the ``sub-*`` folders themselves).
        parent: BIDS-style parent folder name (typically ``"rawdata"``). Pass
            ``None`` or ``""`` for flat datasets like ``TOF_MPRAGE/sub-*``.
        subject_prefix: If given, reassign every subject to
            ``{prefix}{n:0Xd}`` starting from 1 (see ``subject_number_width``).
            Otherwise sanitise the existing subject ids in place (strip ``_``
            and leading ``-``).
        subject_number_width: Zero-padding width for the numeric suffix in
            ``subject_prefix`` mode. Default 3 → ``ID001`` … ``ID999``.
        dry_run: When ``True`` (default) plan the moves and log them but do
            not touch disk. Flip to ``False`` to actually rename.
        info_dir: Directory (relative to ``dataset_root``) where the
            translation table lands. Defaults to ``"info"`` → written as
            ``<dataset_root>/info/subject_map.tsv``.
        session: Forwarded to :func:`extract_keys_from_json` — populate the
            ``ses`` entity from ``StudyDate`` when the sidecar lacks one.
        make_subject_chunks: Forwarded to :func:`_generate_bids_path` (adds a
            sub-folder built from the first N chars of the subject id).
        verbose: Log every planned move.

    Returns:
        Dict with keys ``"moves"`` (list of ``(src, dst)`` tuples), and
        ``"mapping_file"`` (path of the written translation table).
    """
    dataset_root = Path(dataset_root)
    parent_norm = parent or ""
    scan_root = dataset_root / parent_norm if parent_norm else dataset_root
    if not scan_root.exists():
        raise FileNotFoundError(scan_root)

    subject_folders = _list_subject_folders(scan_root)
    if not subject_folders:
        logger.on_warning(f"No sub-* folders found under {scan_root}; nothing to do.")
        return {"moves": [], "mapping_file": None}

    existing_map = _read_existing_subject_map(dataset_root, info_dir)
    subject_map = _build_subject_map(subject_folders, subject_prefix, subject_number_width, existing_map)
    logger.on_neutral(
        f"Renaming {len(subject_folders)} subject(s) "
        f"({'numeric ' + (subject_prefix or '') if subject_prefix else 'sanitising in place'}); "
        f"dry_run={dry_run}."
    )

    all_moves: list[tuple[Path, Path]] = []
    for json_path in _iter_json_sidecars(scan_root):
        # Recover the old sub id from the parent folder name — the JSON's
        # own `PatientID` field is not authoritative once we start remapping.
        try:
            sub_folder = next(p for p in json_path.parents if p.name.startswith("sub-"))
        except StopIteration:
            continue
        old_sub = _current_sub_id(sub_folder)
        new_sub = subject_map.get(old_sub, _sanitize_entity_value(old_sub))
        try:
            new_json_path, _keys = _new_bids_path_for(
                json_path,
                dataset_root,
                parent_norm,
                new_sub,
                session=session,
                make_subject_chunks=make_subject_chunks,
            )
        except Exception as e:  # noqa: BLE001
            logger.on_warning(f"Cannot re-derive BIDS name for {json_path.name}: {type(e).__name__}: {e}")
            continue
        if new_json_path == json_path:
            continue
        moves = _rename_family(json_path, new_json_path, dataset_root, dry_run=dry_run)
        if verbose:
            for src, dst in moves:
                logger.on_neutral(f"{'[dry]' if dry_run else '[mv ]'} {src.relative_to(dataset_root)}  ->  {dst.relative_to(dataset_root)}")
        all_moves.extend(moves)

    # Pass 2 — sweep orphan companions (.bval / .bvec / _ADC.nii.gz / other
    # non-sidecar files) that Pass 1 didn't see because their JSON already
    # moved on a previous run or they never had one.
    leftover = _leftover_move_plan(scan_root, subject_map, parent_norm)
    for src, dst in leftover:
        if verbose:
            logger.on_neutral(f"{'[dry]' if dry_run else '[lft]'} {src.relative_to(dataset_root)}  ->  {dst.relative_to(dataset_root)}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.resolve() != src.resolve():
                logger.on_warning(f"target {dst} already exists; skipping {src}")
                continue
            src.rename(dst)
        all_moves.append((src, dst))

    mapping_file = _write_subject_map(dataset_root, info_dir, subject_map) if not dry_run else None
    if not dry_run:
        removed = _prune_empty_folders(scan_root)
        if removed:
            logger.on_neutral(f"Pruned {removed} empty folder(s) after rename.")
    logger.on_neutral(f"Planned {len(all_moves)} file move(s); mapping table: {mapping_file}")
    return {"moves": all_moves, "mapping_file": mapping_file}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--parent", default="rawdata", help="'' / --parent '' for flat datasets")
    ap.add_argument("--subject-prefix", default=None, help='e.g. "ID" for sub-ID001, sub-ID002, ...')
    ap.add_argument("--subject-number-width", type=int, default=3)
    ap.add_argument("--info-dir", default="info")
    ap.add_argument("--make-subject-chunks", type=int, default=0)
    ap.add_argument("--no-session", dest="session", action="store_false")
    ap.add_argument("--apply", action="store_true", help="Actually perform the moves (default is dry-run).")
    args = ap.parse_args()

    rerun_bids_naming(
        args.dataset_root,
        parent=args.parent or None,
        subject_prefix=args.subject_prefix,
        subject_number_width=args.subject_number_width,
        dry_run=not args.apply,
        info_dir=args.info_dir,
        session=args.session,
        make_subject_chunks=args.make_subject_chunks,
    )
