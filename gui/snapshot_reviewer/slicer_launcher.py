"""BIDS-aware 3D Slicer launcher integrated with the review GUI.

Responsibilities
----------------
1.  ``build_fam_for_snapshot(snp_path, dataset_path, bgi)``
    Given a snapshot path (JPG) and the dataset root, finds the matching
    BIDS family (fam) via BIDS_Global_info and returns it as a
    ``dict[str, list[BIDS_FILE]]``.

2.  ``SlicerLaunchDialog``
    A PyQt6 dialog attached to the review GUI.  For the *currently viewed*
    snapshot it lets the reviewer select:
      • one or more image files   (any key that does NOT start with "msk_"
        and is not a .mrk.json file)
      • zero or more segmentation masks  (keys starting with "msk_")
      • zero or more POI markups         (.mrk.json files only)
    Clicking "Open in Slicer" launches Slicer exactly as ``run_slicer.py``
    does and records which files were viewed.

3.  ``SlicerFileWatcher``
    A QThread that watches the files that were opened in Slicer.  When any
    watched file is overwritten (mtime changes) the ``file_overwritten``
    signal fires with:
      (path_str, fam_snapshot, viewed_files)
    where ``fam_snapshot`` is the fam dict at launch time and
    ``viewed_files`` is the list of BIDS_FILE objects that were sent to
    Slicer.

Usage
-----
    from slicer_launcher import SlicerLaunchDialog, build_fam_for_snapshot

    # Inside ReviewWindow, e.g. connected to a toolbar button:
    def _open_in_slicer(self):
        snp_path = self._current_path()
        if snp_path is None:
            return
        dlg = SlicerLaunchDialog(snp_path, self.dataset_path, self.bgi, parent=self)
        dlg.exec()
"""  # noqa: INP001

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from TPTBox import BIDS_FILE, BIDS_Global_info

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _default_slicer_exe() -> str:
    """Best-effort default for the Slicer executable.

    Checked in order:
      1. ``TPTBOX_SLICER_EXE`` environment variable
      2. ``Slicer`` on ``$PATH``
      3. Empty string (user must configure via the GUI/CLI).
    """
    env = os.environ.get("TPTBOX_SLICER_EXE", "").strip()
    if env:
        return env
    which = shutil.which("Slicer")
    return which or ""


SLICER_EXE = _default_slicer_exe()

# Derivatives folders that contain the raw images + masks + markups.
# Extend this list as new derivative families are added to the project.
# This is a *default* — callers may pass their own list into
# ``build_fam_for_snapshot`` or ``SlicerLaunchDialog`` and should not mutate
# this global in place.
_DERIVATIVES_SEARCH = [
    "derivatives-final-points",
    "derivatives-final",
    "derivatives-treg",
    "derivatives",
    "derivatives-spineps",
    "rawdata",
]


# ---------------------------------------------------------------------------
# Helper: derive fam for a snapshot path
# ---------------------------------------------------------------------------


def build_fam_for_snapshot(
    snp_path: Path,
    dataset_path: Path,
    bgi: BIDS_Global_info | None = None,
    derivatives_search: list[str] | None = None,
) -> dict[str, list[BIDS_FILE]]:
    """Return the BIDS family (fam) that corresponds to *snp_path*.

    The snapshot is a JPG produced by ``snapshot_generator.py``.  Its BIDS
    entities (sub, ses, sequ, …) uniquely identify a subject × sequence
    combination.  We query ``BIDS_Global_info`` for every file that shares
    those entities and return them grouped by their BIDS "family key"
    (format + optional seg/desc tags), exactly as ``loop_dict()`` yields
    them in the generator.

    Parameters
    ----------
    snp_path:
        Absolute path to the ``*_snp.jpg`` file currently on screen.
    dataset_path:
        Root of the BIDS dataset (same value passed to ``ReviewWindow``).
    bgi:
        Optional pre-built ``BIDS_Global_info`` instance.  When *None* a
        new one is constructed (takes a few seconds the first time).

    Returns:
    -------
    dict mapping family-key strings to lists of ``BIDS_FILE`` objects.
    An empty dict is returned when the subject cannot be found.
    """
    if bgi is None:
        bgi = BIDS_Global_info(dataset_path, list(derivatives_search or _DERIVATIVES_SEARCH))

    # Parse entities from the snapshot filename.
    snp_bf = BIDS_FILE(snp_path, dataset_path)
    sub = snp_bf.get("sub")
    ses = snp_bf.get("ses")
    sequ = snp_bf.get("sequ")

    if not sub:
        return {}

    # Iterate subjects in bgi until we find the matching one.
    for subj_key, sub_obj in bgi.iter_subjects():
        if subj_key != sub:
            continue

        q = sub_obj.new_query()
        # Filter to the correct session and sequence if present.
        if ses:
            q.filter("ses", ses)
        if sequ:
            q.filter("sequ", sequ)
        q.flatten()
        q.unflatten()

        # Return the first matching family dict.
        for fam in q.loop_dict():
            return fam  # type: ignore[return-value]

        break  # subject found but no query result

    return {}


# ---------------------------------------------------------------------------
# Internal: classify files inside a fam
# ---------------------------------------------------------------------------


def _classify_fam(
    fam: dict[str, list[BIDS_FILE]],
) -> tuple[
    list[tuple[str, BIDS_FILE]],  # images
    list[tuple[str, BIDS_FILE]],  # segmentations (msk_*)
    list[tuple[str, BIDS_FILE]],  # markups (*.mrk.json)
]:
    """Split *fam* into three lists: images, masks, markups.

    Rules
    -----
    - Markup  : any BIDS_FILE where the file dict has a ``json`` entry
                whose path ends with ``.mrk.json``.
    - Mask    : family key starts with ``msk_``.
    - Image   : everything else.
    """
    images: list[tuple[str, BIDS_FILE]] = []
    masks: list[tuple[str, BIDS_FILE]] = []
    markups: list[tuple[str, BIDS_FILE]] = []
    from TPTBox import Print_Logger

    log = Print_Logger()
    log.on_ok(fam)
    for key, bids_files in fam.items():
        for bf in bids_files:
            # Detect .mrk.json markups
            json_path = bf.file.get("mrk.json", "")
            if json_path == "":
                json_path = bf.file.get("json", "")
                json_path = Path(str(json_path).replace(".json", ".mrk.json"))
                if not json_path.exists():
                    json_path = ""
                else:
                    bf.file["mrk.json"] = json_path
            log.on_debug(json_path, "|", bf)
            if json_path and str(json_path).endswith(".mrk.json"):
                log.on_warning(bf)
                markups.append((key, bf))
            elif key.startswith("msk_"):
                masks.append((key, bf))
            else:
                # Plain image (nii.gz, etc.)
                nii_path = bf.file.get("nii.gz") or bf.file.get("nii")
                if nii_path:
                    images.append((key, bf))

    return sorted(images), sorted(masks), sorted(markups)


# ---------------------------------------------------------------------------
# Slicer startup script builder
# ---------------------------------------------------------------------------


def _build_startup_script(
    images: list[BIDS_FILE],
    masks: list[BIDS_FILE],
    markups: list[BIDS_FILE],
) -> str:
    """Build the Python script that Slicer executes at startup.

    Mirrors the pattern from ``run_slicer.py``.
    """
    lines: list[str] = ["import slicer", ""]

    # Load scalar volumes (images)
    for bf in images:
        p = bf.file.get("nii.gz") or bf.file.get("nii", "")
        if p:
            lines.append(f'node = slicer.util.loadVolume(r"{p}")')
            lines.append(f'node.SetName(r"{p.name.split(".")[0]}")')

    lines.append("")

    # Load segmentations
    for bf in masks:
        p = bf.file.get("nii.gz") or bf.file.get("nii", "")
        if p:
            lines.append(f'node = slicer.util.loadSegmentation(r"{p}")')
            lines.append(f'node.SetName(r"{p.name.split(".")[0]}")')

    lines.append("")

    # Load markups (.mrk.json)
    for bf in markups:
        p = bf.file.get("mrk.json", "")
        if p:
            lines.append(f'node = slicer.util.loadMarkups(r"{p}")')
            lines.append(f'node.SetName(r"{p.name.split(".")[0]}")')

    lines += [
        "",
        "# Show every segmentation in 3D",
        "for segNode in slicer.util.getNodesByClass('vtkMRMLSegmentationNode'):",
        "    segNode.CreateClosedSurfaceRepresentation()",
        "    dn = segNode.GetDisplayNode()",
        "    if dn:",
        "        dn.SetVisibility3D(True)",
        "        dn.SetVisibility2D(True)",
        "",
        "slicer.util.resetThreeDViews()",
        "slicer.util.resetSliceViews()",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File watcher thread
# ---------------------------------------------------------------------------


class SlicerFileWatcher(QThread):
    """Watches a set of files for modification (mtime changes).

    Signals
    -------
    file_overwritten(path_str, fam, viewed_files)
        Emitted once per modified file.
        ``fam``          – the fam dict captured at Slicer launch
        ``viewed_files`` – all BIDS_FILE objects that were opened
    """

    file_overwritten = pyqtSignal(str, object, object)

    def __init__(
        self, watch_paths: list[Path], fam: dict[str, list[BIDS_FILE]], viewed_files: list[BIDS_FILE], poll_interval: float = 2.0, fun=None
    ):
        super().__init__()
        self._watch_paths = watch_paths
        self._fam = fam
        self._viewed_files = viewed_files
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        # Capture baseline mtimes
        self._mtimes: dict[str, float] = {}
        for p in watch_paths:
            try:
                self._mtimes[str(p)] = p.stat().st_mtime
            except OSError:
                self._mtimes[str(p)] = 0.0
        self.fun = fun

    def run(self) -> None:
        """Start looking if file was changed."""
        while not self._stop_event.is_set():
            for p in self._watch_paths:
                ps = str(p)
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if mtime != self._mtimes.get(ps, mtime):
                    self._mtimes[ps] = mtime
                    if self.fun is not None:
                        try:
                            self.fun(ps, self._fam, self._viewed_files)
                        except Exception as ex:
                            print(f"[slicer_launcher] watcher callback failed: {ex}")
                    self.file_overwritten.emit(ps, self._fam, self._viewed_files)
            time.sleep(self._poll_interval)

    def stop(self) -> None:
        """Stop looking if file was changed."""
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class SlicerLaunchDialog(QDialog):
    """Modal dialog for selecting files from a BIDS fam and launching Slicer.

    Parameters
    ----------
    snp_path:
        The snapshot currently displayed in the review GUI.
    dataset_path:
        BIDS dataset root.
    bgi:
        Optional pre-built ``BIDS_Global_info`` (avoids repeated disk scans).
    overwrite_callback:
        Optional callable ``(path_str, fam, viewed_files)`` invoked when a
        watched file is overwritten.  This is the "backhook" for downstream
        processing.  If *None*, overwrite events are only printed to stdout.
    parent:
        Qt parent widget (the ``ReviewWindow``).
    """

    def __init__(
        self,
        snp_path: Path,
        dataset_path: Path,
        bgi: BIDS_Global_info | None = None,
        overwrite_callback: Callable[[Path, str, dict, list[BIDS_FILE]], None] | None = None,
        parent: QWidget | None = None,
        slicer_exe: str | None = None,
        derivatives_search: list[str] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Open in Slicer")
        self.resize(900, 1200)

        self._snp_path = snp_path
        self._dataset_path = dataset_path
        self._overwrite_callback = overwrite_callback
        self._watchers: list[SlicerFileWatcher] = []
        self._slicer_exe = slicer_exe or SLICER_EXE
        self._derivatives_search = list(derivatives_search or _DERIVATIVES_SEARCH)

        # Build fam
        self._fam = build_fam_for_snapshot(snp_path, dataset_path, bgi, self._derivatives_search)
        self._images, self._masks, self._markups = _classify_fam(self._fam)

        self._img_checks: list[tuple[QCheckBox, BIDS_FILE]] = []
        self._msk_checks: list[tuple[QCheckBox, BIDS_FILE]] = []
        self._mrk_checks: list[tuple[QCheckBox, BIDS_FILE]] = []

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Subject info
        snp_bf = BIDS_FILE(self._snp_path, self._dataset_path)
        info_lbl = QLabel(f"<b>Subject:</b> sub-{snp_bf.get('sub')} &nbsp; ses-{snp_bf.get('ses')} &nbsp; sequ-{snp_bf.get('sequ')}")
        layout.addWidget(info_lbl)

        if not self._fam:
            layout.addWidget(QLabel("⚠  Could not resolve BIDS family for this snapshot."))
        else:
            # ── Images ────────────────────────────────────────────────────
            img_group = QGroupBox("Images (scalar volumes)")
            img_v = QVBoxLayout(img_group)
            if self._images:
                for key, bf in self._images:
                    p = bf.file.get("nii.gz") or bf.file.get("nii", "")
                    cb = QCheckBox(f"{key}\n  {Path(str(p)).name}")
                    cb.setChecked(True)
                    img_v.addWidget(cb)
                    self._img_checks.append((cb, bf))
            else:
                img_v.addWidget(QLabel("  (none found)"))
            layout.addWidget(img_group)

            # ── Segmentations ─────────────────────────────────────────────
            scroll_msk = QScrollArea()
            scroll_msk.setWidgetResizable(True)
            msk_group = QGroupBox("Segmentations (msk_*)")
            msk_v = QVBoxLayout(msk_group)
            if self._masks:
                for key, bf in self._masks:
                    p = bf.file.get("nii.gz") or bf.file.get("nii", "")
                    cb = QCheckBox(f"{key}\n  {Path(str(p)).name}")
                    cb.setChecked(False)
                    msk_v.addWidget(cb)
                    self._msk_checks.append((cb, bf))
            else:
                msk_v.addWidget(QLabel("  (none found)"))
            scroll_msk.setWidget(msk_group)
            layout.addWidget(scroll_msk)

            # ── Markups ───────────────────────────────────────────────────
            mrk_group = QGroupBox("POI Markups (*.mrk.json)")
            mrk_v = QVBoxLayout(mrk_group)
            if self._markups:
                for key, bf in self._markups:
                    p = bf.file.get("json", "")
                    cb = QCheckBox(f"{key}\n  {Path(str(p)).name}")
                    cb.setChecked(False)
                    mrk_v.addWidget(cb)
                    self._mrk_checks.append((cb, bf))
            else:
                mrk_v.addWidget(QLabel("  (none found)"))
            layout.addWidget(mrk_group)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_box = QDialogButtonBox()
        self._launch_btn = QPushButton("Open in Slicer")
        self._launch_btn.setEnabled(bool(self._fam))
        self._launch_btn.clicked.connect(self._launch)
        btn_box.addButton(self._launch_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_box.addButton(cancel_btn, QDialogButtonBox.ButtonRole.RejectRole)
        layout.addWidget(btn_box)

    # ── Launch ────────────────────────────────────────────────────────────

    def _launch(self):
        """Collect selections, write startup script, launch Slicer, start watcher."""
        sel_images = [bf for cb, bf in self._img_checks if cb.isChecked()]
        sel_masks = [bf for cb, bf in self._msk_checks if cb.isChecked()]
        sel_markups = [bf for cb, bf in self._mrk_checks if cb.isChecked()]

        viewed_files: list[BIDS_FILE] = sel_images + sel_masks + sel_markups

        script = _build_startup_script(sel_images, sel_masks, sel_markups)

        if not self._slicer_exe or not Path(self._slicer_exe).exists():
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Slicer not found",
                (
                    "Slicer executable is not configured or does not exist:\n"
                    f"  {self._slicer_exe or '<unset>'}\n\n"
                    "Set the path via the ⚙ Settings dialog, the --slicer-exe CLI flag, "
                    "or the TPTBOX_SLICER_EXE environment variable."
                ),
            )
            return

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name
        subprocess.Popen([self._slicer_exe, "--python-script", script_path])
        # Collect all file paths that might be overwritten by the user in Slicer.
        watch_paths: list[Path] = []
        print("WATCHING:")
        for bf in viewed_files:
            for p in bf.file.values():
                path_obj = Path(str(p))
                if path_obj.exists():
                    watch_paths.append(path_obj)
                    print(" ", path_obj)

        if watch_paths:
            watcher = SlicerFileWatcher(watch_paths, self._fam, viewed_files, fun=self._on_file_overwritten)
            watcher.file_overwritten.connect(self._on_file_overwritten)
            watcher.start()
            self._watchers.append(watcher)

        self.accept()
        # print("slicer closed")

    # ── Overwrite hook ────────────────────────────────────────────────────

    def _on_file_overwritten(
        self,
        path_str: str,
        fam: dict,
        viewed_files: list[BIDS_FILE],
    ):
        """Called when one of the files opened in Slicer is written to disk.

        Forwards to the caller-supplied callback if provided; otherwise
        prints a summary to stdout.

        Parameters
        ----------
        path_str:
            Absolute path of the file that was overwritten.
        fam:
            The full BIDS family dict at the time of launch.
        viewed_files:
            All BIDS_FILE objects that were opened in Slicer.
        """
        print(f"[slicer_launcher] File overwritten: {path_str}")
        print(f"  fam keys   : {list(fam.keys())}")
        print(f"  viewed     : {[str(bf) for bf in viewed_files]}")

        if self._overwrite_callback is not None:
            self._overwrite_callback(self._snp_path, path_str, fam, viewed_files)
        else:
            print("no callback set")

    def closeEvent(self, event) -> None:
        """Stop Event."""
        for w in self._watchers:
            w.stop()
            w.wait(2000)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Integration helper for ReviewWindow
# ---------------------------------------------------------------------------


def add_slicer_button_to_review_window(review_window) -> None:
    """Attach a "Open in Slicer" button and keyboard shortcut to a ReviewWindow.

    Call this *after* the window has been fully constructed:

        from slicer_launcher import add_slicer_button_to_review_window
        win = ReviewWindow(dataset_path, cpus, name, parent)
        add_slicer_button_to_review_window(win)
        win.show()

    The function:
    • Builds (and caches) a ``BIDS_Global_info`` on the review window.
    • Adds a "🔬 Slicer [L]" button to the center panel's button area.
    • Registers ``L`` as a keyboard shortcut.
    • Wires a default overwrite callback that prints a report and could be
      extended for downstream automation.
    """
    from PyQt6.QtGui import QKeySequence
    from PyQt6.QtWidgets import QHBoxLayout, QShortcut

    # Build / cache bgi on the window object so it is only constructed once.
    if not hasattr(review_window, "_slicer_bgi"):
        review_window._slicer_bgi = BIDS_Global_info(
            review_window.dataset_path,
            list(_DERIVATIVES_SEARCH),
        )

    def _open_slicer():
        snp_path = review_window._current_path()
        if snp_path is None:
            review_window.status_bar.showMessage("No snapshot selected.", 3000)
            return

        def _on_overwrite(jpg_path, path_str: str, fam: dict, viewed: list[BIDS_FILE]):  # noqa: ARG001
            """Default back-hook: log and update status bar."""
            fname = Path(path_str).name
            viewed_names = ", ".join(Path(str(bf)).name for bf in viewed)
            review_window.status_bar.showMessage(f"Slicer wrote: {fname}  |  session had: {viewed_names}", 8000)
            # TODO: trigger downstream re-processing here if needed.

        dlg = SlicerLaunchDialog(
            snp_path,
            review_window.dataset_path,
            bgi=review_window._slicer_bgi,
            overwrite_callback=_on_overwrite,
            parent=review_window,
        )
        dlg.exec()

    # Add button — find the center panel's layout by walking the central widget.
    slicer_btn = QPushButton("Slicer  [L]")
    slicer_btn.setObjectName("skip")  # reuse the amber skip style
    slicer_btn.clicked.connect(_open_slicer)

    # Insert into the navigation row (the first QHBoxLayout in the center panel).
    central = review_window.centralWidget()
    for child in central.findChildren(QHBoxLayout):
        # The nav row contains btn_prev, lbl_position, btn_next.
        if hasattr(review_window, "btn_prev") and review_window.btn_prev in (
            child.itemAt(i).widget() for i in range(child.count()) if child.itemAt(i).widget()
        ):
            child.addWidget(slicer_btn)
            break

    # Keyboard shortcut
    QShortcut(QKeySequence("L"), review_window, _open_slicer)
    review_window.status_bar.showMessage("Slicer integration ready  [L]", 3000)
