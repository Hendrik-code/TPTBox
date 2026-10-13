"""Post-extract pairing of biplane X-ray Angiography series.

Biplane angio runs are exported as two separate DICOM series (A and B plane)
that share ``StudyInstanceUID`` and ``AcquisitionTime`` but carry different
``SeriesNumber`` values — one per plane. After :func:`extract_dicom_folder`
runs, that difference propagates into distinct ``sequ-<N>`` entities on the
two BIDS filenames, which makes the two planes look like independent runs.

This module rewires the pair so both planes carry the A-side's
``sequ-<N>`` value — they now differ only by ``acq-A`` vs ``acq-B`` and
downstream tools that group by ``(sub, ses, sequ)`` see the biplane run as
one physical acquisition.

Optionally also flips the Z-axis affine of every XA NIfTI to work around a
Siemens quirk that ships the volume upside-down (byte-identical voxel data,
just a sign flip on the third affine column). Off by default.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel

from TPTBox import BIDS_FILE, BIDS_Global_info, Print_Logger

logger = Print_Logger()

# Sidecar-JSON key we set after flipping the affine Z-column so subsequent
# runs can detect the file has already been corrected and skip it. Keeping
# state on the sidecar (rather than a separate ledger) survives file moves
# by the renamer and is what a re-extract from DICOMs would overwrite.
_Z_FLIP_MARKER = "TPTBoxXAZFlipped"

# BIDS formats that carry XA-family payload. `.filter_format` accepts these
# values verbatim as the series's `bids_format`.
_XA_FORMATS: tuple[str, ...] = ("XA", "DSA", "DSA3D", "3DRA", "fluroscopy", "subtraction")


def _norm_time(t) -> str:
    """Normalise a DICOM TM value (``HHMMSS.ffffff`` or ``HH:MM:SS.ffffff``)."""
    if t is None:
        return ""
    s = str(t)
    if ":" in s:
        try:
            hh, mm, rest = s.split(":", maxsplit=2)
        except ValueError:
            return s
    else:
        hh, mm, rest = s[:2], s[2:4], s[4:] or "0"
    try:
        return f"{int(hh):02d}:{int(mm):02d}:{float(rest):09.6f}"
    except (ValueError, TypeError):
        return s


def _bucket_key(json_obj: dict) -> tuple:
    """Group a series by (StudyInstanceUID, SeriesNumber-family, AcquisitionTime).

    Both planes of a biplane run share ``StudyInstanceUID`` and
    ``AcquisitionTime`` down to milliseconds; ``SeriesNumber`` differs by one
    (A first, B second). Bucketing on the study UID + acquisition time is
    tight enough to catch the pair without false positives.
    """
    return (
        json_obj.get("StudyInstanceUID"),
        _norm_time(json_obj.get("AcquisitionTime")),
    )


def _plane(bf: BIDS_FILE) -> str | None:
    """Return the ``acq`` entity's plane label if it is ``A`` or ``B``."""
    val = bf.get("acq")
    if val in ("A", "B"):
        return val
    return None


def _sidecar_for(nii_path: Path) -> Path:
    """Return the ``.json`` sidecar path that pairs with a ``.nii.gz`` file."""
    name = nii_path.name
    if name.endswith(".nii.gz"):
        return nii_path.with_name(name[: -len(".nii.gz")] + ".json")
    return nii_path.with_suffix(".json")


def _flip_marker_set(sidecar: Path) -> bool:
    """True when the sidecar already records that its NIfTI was Z-flipped."""
    if not sidecar.is_file():
        return False
    try:
        j = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return bool(j.get(_Z_FLIP_MARKER))


def _write_flip_marker(sidecar: Path) -> None:
    """Set ``_Z_FLIP_MARKER=true`` on the sidecar; no-op if it doesn't exist."""
    if not sidecar.is_file():
        return
    try:
        j = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return
    j[_Z_FLIP_MARKER] = True
    try:
        sidecar.write_text(json.dumps(j, indent=4), encoding="utf-8")
    except OSError as e:
        logger.on_warning(f"Could not persist flip marker in {sidecar}: {e}")


def _flip_z_affine(path: Path) -> None:
    """Negate the third affine column of the NIfTI at *path* in place.

    Pixel data is untouched — only the affine's Z-column (and its qform /
    sform copies) get sign-flipped, which fixes the upside-down display on
    Siemens XA sources. Idempotency is handled by the caller via
    :func:`_flip_marker_set` / :func:`_write_flip_marker` on the sidecar —
    a double-run of the pairing command with ``--flip-z`` would otherwise
    negate twice and land back on the original orientation.
    """
    img = nibabel.load(str(path))
    aff = img.affine.copy()
    aff[:3, 2] = -aff[:3, 2]
    out = nibabel.Nifti1Image(img.dataobj, aff, header=img.header)
    out.set_qform(aff)
    out.set_sform(aff)
    nibabel.save(out, str(path))


def pair_biplane_xa(
    dataset_root: Path | str,
    parent: str = "rawdata",
    dry_run: bool = True,
    flip_z: bool = False,
    verbose: bool = True,
) -> dict:
    """Rewrite biplane B-plane files to share A-plane's ``sequ`` value.

    Walks every XA-family series under ``<dataset_root>/<parent>/``, buckets
    by ``(StudyInstanceUID, AcquisitionTime)``, and for each bucket that
    contains both an ``acq-A`` and an ``acq-B`` member: if the two carry
    different ``sequ-<N>`` values, rename the B-side to use the A-side's
    ``sequ`` (via :meth:`BIDS_FILE.rename_files`). All extensions in the
    family follow the rename.

    When ``flip_z`` is set, every XA NIfTI seen (both A and B) also gets
    its affine's third column negated — no-op on non-Siemens data whose
    display was already right-side-up.

    Args:
        dataset_root: Dataset root that contains ``<parent>`` — same value
            ``BIDS_Global_info`` would take.
        parent: BIDS parent folder (``rawdata`` etc.).
        dry_run: Default True — plan the moves and print them, don't touch
            disk. Set False to apply.
        flip_z: If True, negate every XA NIfTI's affine Z-column. Applied
            to the file at its FINAL location (post-rename).
        verbose: Log every planned rename.

    Returns:
        Dict with ``"pairs_matched"`` (count of (A,B) buckets found),
        ``"renames"`` (list of ``(src, dst)`` tuples), and ``"flipped"``
        (list of paths whose affine was flipped).
    """
    dataset_root = Path(dataset_root)
    bgi = BIDS_Global_info(datasets=[dataset_root], parents=[parent])
    buckets_by_subject: dict[str, dict[tuple, dict[str, list[BIDS_FILE]]]] = {}
    for sub_name, subj in bgi.enumerate_subjects():
        q = subj.new_query(flatten=True)
        q.filter_format(list(_XA_FORMATS))
        subj_buckets: dict[tuple, dict[str, list[BIDS_FILE]]] = defaultdict(lambda: {"A": [], "B": []})
        for bf in q.loop_list():
            plane = _plane(bf)
            if plane is None:
                continue
            try:
                j = bf.open_json()
            except Exception:  # noqa: BLE001
                continue
            key = _bucket_key(j)
            subj_buckets[key][plane].append(bf)
        buckets_by_subject[sub_name] = subj_buckets

    renames: list[tuple[Path, Path]] = []
    flipped: list[Path] = []
    pairs_matched = 0
    for sub_name, subj_buckets in buckets_by_subject.items():
        for planes in subj_buckets.values():
            a_side = planes["A"]
            b_side = planes["B"]
            if not a_side or not b_side:
                continue
            pairs_matched += 1
            # A-side's `sequ` is authoritative — smaller SeriesNumber ships
            # first on Siemens/Philips biplane exports. Pick lowest if
            # multiple A-members exist (rare — repeated runs at same time).
            a_sequ = min({str(bf.get("sequ") or "") for bf in a_side})
            if not a_sequ:
                continue
            for bf in b_side:
                b_sequ = str(bf.get("sequ") or "")
                if b_sequ == a_sequ:
                    continue
                # Build the target path with A's sequ; keep all other entities.
                new_bids = bf.get_changed_bids(info={"sequ": a_sequ}, non_strict_mode=True)
                src_paths = {ext: Path(p) for ext, p in bf.file.items()}
                new_nii = Path(new_bids.file.get("nii.gz", ""))
                new_stem = str(new_nii).removesuffix(".nii.gz")
                for ext, src in src_paths.items():
                    dst = Path(f"{new_stem}.{ext}")
                    if src == dst or not src.exists():
                        continue
                    if verbose:
                        logger.on_neutral(
                            f"{'[dry]' if dry_run else '[mv ]'} sub={sub_name} B→A pairing  "
                            f"{src.relative_to(dataset_root)}  ->  {dst.relative_to(dataset_root)}"
                        )
                    renames.append((src, dst))
                    if not dry_run:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if dst.exists() and dst.resolve() != src.resolve():
                            logger.on_warning(f"target {dst} already exists; skipping {src}")
                            continue
                        src.rename(dst)

    if flip_z:
        # Re-scan after renames so we hit files at their final paths.
        bgi2 = BIDS_Global_info(datasets=[dataset_root], parents=[parent])
        skipped_already_flipped = 0
        for _sub, subj in bgi2.enumerate_subjects():
            q = subj.new_query(flatten=True)
            q.filter_format(list(_XA_FORMATS))
            for bf in q.loop_list():
                if _plane(bf) is None:
                    continue
                nii = bf.file.get("nii.gz")
                if nii is None or not Path(nii).exists():
                    continue
                sidecar = _sidecar_for(Path(nii))
                # Idempotency: a sidecar that already carries our marker was
                # flipped on a previous run; flipping again would undo it.
                if _flip_marker_set(sidecar):
                    skipped_already_flipped += 1
                    continue
                if verbose:
                    logger.on_neutral(f"{'[dry]' if dry_run else '[fz ]'} flip-Z {Path(nii).relative_to(dataset_root)}")
                if not dry_run:
                    _flip_z_affine(Path(nii))
                    _write_flip_marker(sidecar)
                flipped.append(Path(nii))
        if skipped_already_flipped:
            logger.on_neutral(f"flip-Z: skipped {skipped_already_flipped} file(s) that already carry the marker.")

    logger.on_neutral(
        f"Biplane pairs matched: {pairs_matched}; renames planned: {len(renames)}; flipped: {len(flipped)} (dry_run={dry_run})"
    )
    return {"pairs_matched": pairs_matched, "renames": renames, "flipped": flipped}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--parent", default="rawdata")
    ap.add_argument("--flip-z", action="store_true", help="Also negate the Z-column of every XA NIfTI's affine (Siemens quirk).")
    ap.add_argument("--apply", action="store_true", help="Perform the moves; default is dry-run.")
    args = ap.parse_args()
    pair_biplane_xa(
        args.dataset_root,
        parent=args.parent,
        dry_run=not args.apply,
        flip_z=args.flip_z,
    )
