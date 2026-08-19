"""Add rib labels to an existing vertebra instance + spine subregion segmentation.

Public entry point: :func:`add_ribs_to_vert_spine`.

Given a vertebra instance segmentation (`vert`) and a spine subregion
segmentation (`spine`), this module optionally runs VibeSeg (dataset 12) on
the source CT to obtain a whole-body rib segmentation, then delegates label
assignment to :func:`assign_ribs_to_vert_segmentation` in ``rib_chatgpt``,
which handles left/right splitting, cranial-caudal matching, and the
fallback that separates touching ribs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from TPTBox import BIDS_FILE, NII, Image_Reference, Location, No_Logger, to_nii
from TPTBox.segmentation._rib._rib_assign import assign_ribs_to_vert_segmentation

logger = No_Logger(prefix="AddRibs")

_VIBESEG_RIB_DATASET_ID = 12


def _resolve_rib_seg_path(
    rib_seg: Image_Reference | None,
    ct: Image_Reference | None,
    rib_seg_out: str | Path | None,
    dataset: str | Path | None,
    derivatives_folder: str,
) -> tuple[Image_Reference | None, Path | None]:
    """Determine the rib segmentation input and the output path used if we need to run VibeSeg.

    Returns ``(rib_seg_ref, out_path)``. ``rib_seg_ref`` is the existing segmentation
    to load (may be a path that does not exist yet — the caller is expected to write
    to it after inference). ``out_path`` is where VibeSeg should save if it runs.
    """
    if rib_seg is not None:
        return rib_seg, None

    if rib_seg_out is not None:
        p = Path(rib_seg_out)
        return (p if p.exists() else None), p

    if isinstance(ct, BIDS_FILE):
        out = ct.get_changed_path(
            "nii.gz",
            "msk",
            parent=derivatives_folder,
            info={"seg": f"VIBESeg-{_VIBESEG_RIB_DATASET_ID}"},
            dataset_path=dataset,
        )
        return (out if out.exists() else None), out

    raise ValueError(
        "rib_seg is None and no output path could be derived. Pass one of:\n"
        "  * rib_seg (existing rib segmentation), or\n"
        "  * rib_seg_out (explicit output path), or\n"
        "  * ct as a BIDS_FILE plus a `dataset` root so a BIDS path can be generated."
    )


def _ensure_rib_seg(
    rib_seg: Image_Reference | None,
    ct: Image_Reference | None,
    rib_seg_out: str | Path | None,
    dataset: str | Path | None,
    derivatives_folder: str,
    override: bool,
    vibeseg_kwargs: dict[str, Any] | None,
) -> NII:
    """Return a ready-to-use rib segmentation NII, running VibeSeg if needed."""
    resolved, out_path = _resolve_rib_seg_path(rib_seg, ct, rib_seg_out, dataset, derivatives_folder)

    if resolved is not None and not override:
        return to_nii(resolved, seg=True)

    if ct is None:
        raise ValueError("rib_seg is missing (or override=True) and no `ct` was given to run VibeSeg on.")
    if out_path is None:
        raise ValueError("Cannot run VibeSeg without an output path. Pass `rib_seg_out` or a BIDS_FILE ct + dataset.")

    # Import lazily so importing this module never forces the segmentation stack.
    from TPTBox.segmentation import run_vibeseg

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(vibeseg_kwargs or {})
    logger.print(f"Running VibeSeg (dataset {_VIBESEG_RIB_DATASET_ID}) -> {out_path}")
    return run_vibeseg(ct, out_path, dataset_id=_VIBESEG_RIB_DATASET_ID, override=override, **kwargs)


def _save_if_path(nii: NII, target: Image_Reference | str | Path | None) -> None:
    if target is None:
        return
    if isinstance(target, BIDS_FILE):
        nii.save(target.file["nii.gz"])
        return
    if isinstance(target, (str, Path)):
        nii.save(Path(target))


def add_ribs_to_vert_spine(
    vert: Image_Reference,
    spine: Image_Reference,
    rib_seg: Image_Reference | None = None,
    ct: Image_Reference | None = None,
    dataset: str | Path | None = None,
    *,
    rib_seg_out: str | Path | None = None,
    derivatives_folder: str = "derivatives-rib",
    override: bool = False,
    save: bool = False,
    vibeseg_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
    vert_path_out: Path | None = None,
    spine_path_out: Path | None = None,
    **assign_kwargs: Any,
) -> tuple[NII, NII]:
    """Merge rib labels into a vertebra + spine segmentation.

    If ``rib_seg`` is not given, VibeSeg (dataset 12) is run on ``ct`` and the
    result is written to a path derived from a BIDS_FILE ct or given explicitly
    via ``rib_seg_out``. Otherwise a ``dataset`` root must resolve a BIDS path.

    Args:
        vert: Vertebra instance segmentation (path, NII, or BIDS_FILE).
        spine: Spine subregion segmentation (path, NII, or BIDS_FILE).
        rib_seg: Optional rib segmentation. If ``None``, VibeSeg is executed.
        ct: Source CT, required when ``rib_seg`` is missing.
        dataset: BIDS dataset root, used together with a ``BIDS_FILE`` ``ct`` to
            derive the VibeSeg output path.
        rib_seg_out: Explicit output path for the generated rib segmentation
            (overrides BIDS derivation).
        derivatives_folder: Sub-folder of ``dataset`` for the VibeSeg output.
        override: If True, re-run VibeSeg even when its output already exists.
        save: If True, write the merged results back to the ``vert`` and
            ``spine`` locations (only when they are paths or BIDS_FILEs).
        vibeseg_kwargs: Extra kwargs forwarded to :func:`run_vibeseg`.
        verbose: Verbose logging for the assignment step.
        **assign_kwargs: Extra kwargs forwarded to
            :func:`assign_ribs_to_vert_segmentation` (e.g. ``split_touching``,
            ``max_span_factor``, ``erosion_pixels``).

    Returns:
        ``(vert_with_ribs, spine_with_ribs)`` — instance and subregion masks.
    """
    vert_nii = to_nii(vert, seg=True)
    spine_nii = to_nii(spine, seg=True)

    if spine_nii.extract_label(Location.Rib_Left).sum() != 0 or spine_nii.extract_label(Location.Rib_Right).sum() != 0:
        logger.on_warning("ribs already present in spine segmentation — returning input unchanged")
        return vert_nii, spine_nii

    rib_nii = _ensure_rib_seg(
        rib_seg=rib_seg,
        ct=ct,
        rib_seg_out=rib_seg_out,
        dataset=dataset,
        derivatives_folder=derivatives_folder,
        override=override,
        vibeseg_kwargs=vibeseg_kwargs,
    )
    rib_nii = rib_nii.resample_from_to(vert_nii)

    vert_out, spine_out = assign_ribs_to_vert_segmentation(
        vert_nii,
        spine_nii,
        rib_nii,
        verbose=verbose,
        **assign_kwargs,
    )

    if save:
        _save_if_path(vert_out, vert_path_out if vert_path_out is not None else vert)
        _save_if_path(spine_out, spine_path_out if spine_path_out is not None else spine)

    return vert_out, spine_out
