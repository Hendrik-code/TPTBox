#!/usr/bin/env python3
"""PyQt6 GUI for reviewing anatomy point-of-interest snapshots.

Features
--------
• Displays snapshots (JPG) one at a time; keyboard-friendly navigation
• Good / Bad (Fixable) / Allowed Missing / Remove Entirely / Remove Partial buttons
• Processed (set externally in JSON) + Final Confirmed (upgraded from Processed via Good)
• Hide-already-reviewed is now a multi-select per verdict type
• Region filter applies to both the queue and all verdict log panels
• Double-clicking a log entry navigates to that snapshot (temporarily injected if needed)
• Image cache: the next 10 snapshots are prefetched in the background, and the
  last 100 loaded images stay buffered in an LRU cache so re-visiting recently
  viewed snapshots is instant.

Run:
    python review_gui.py /path/to/dataset [--cpus N]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import KeysView

from TPTBox import BIDS_FILE, POI_Global, Print_Logger

sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from slicer_launcher import _DERIVATIVES_SEARCH, SLICER_EXE, BIDS_Global_info, SlicerLaunchDialog

logger = Print_Logger()
# ── Palette ───────────────────────────────────────────────────────────────
BG = "#0d1117"
SURFACE = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
TEXT_DIM = "#7d8590"
CYAN = "#39d0f0"
GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"
PINK = "#ff6ac1"
TEAL = "#2dd4bf"  # Processed
GOLD = "#fbbf24"  # Final Confirmed
LIME = "#84cc16"  # Image Issue (new)
MAGENTA = "#e879f9"  # Bad Segmentation (new)

# Verdict metadata:
#   key -> (display label, color, button_text or None, shortcut or None)
# When button_text is None the verdict has no action button (informational only).
# A D L S V are already in use!
VERDICT_META: dict[str, tuple[str, str, str | None, str | None]] = {
    "good": ("✓ Good", GREEN, "✓  Good", "G"),
    "final_confirmed": ("★ Final Confirmed", GOLD, None, None),
    "processed": ("⟳ Processed", TEAL, None, None),
    "image_issue": ("✗ Image Issue (implant etc)", PINK, "✗  Image Issue (implant etc)", "I"),
    "bad": ("✗ Bad (Points only)", RED, "✗  Bad (Points only)", "B"),
    "bad_seg": ("✗ Bad (Segmentation)", RED, "✗  Bad (Segmentation)", "R"),
    "allowed_missing": ("~ Some Points are outside FOV", ORANGE, "~  Some Points are outside FOV", "M"),
    "remove_partial": ("✂ Remove Autodetected issues", ORANGE, "✂  Remove Autodetected issues", "P"),
    "remove_entirely": ("✂ Remove All Points", ORANGE, "✂  Remove All Points (FOV)", "E"),
}

# Verdicts that count as "resolved" for the progress bar
RESOLVED_VERDICTS = {"good", "final_confirmed", "allowed_missing"}

# Number of action buttons per row in the review panel (change to relayout).
# Default; overridable via CLI (--buttons-per-row) or the ⚙ Settings dialog.
BUTTONS_PER_ROW = 4

# Image cache tuning. Defaults; overridable via CLI or the ⚙ Settings dialog.
PREFETCH_COUNT = 20  # how many upcoming snapshots to warm in the background
CACHE_MAX_SIZE = 250  # how many decoded pixmaps to keep buffered (LRU)

# Default derivatives folder scanned for snapshots (JPG/PNG). CLI: --parent-dir.
DEFAULT_SNAPSHOT_PARENT = "derivatives-VIBESeg-12-points-snp"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}}
QFrame#card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QPushButton {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton:hover {{ background: #21262d; border-color: {CYAN}; }}
QPushButton:pressed {{ background: #0d1117; }}
QPushButton#good {{
    background: #0f2d1f; border-color: {GREEN}; color: {GREEN};
    font-weight: 700; font-size: 14px; padding: 10px 20px;
}}
QPushButton#good:hover {{ background: #1a4a2e; }}
QPushButton#bad {{
    background: #2d0f0f; border-color: {RED}; color: {RED};
    font-weight: 700; font-size: 14px; padding: 10px 20px;
}}
QPushButton#bad:hover {{ background: #4a1a1a; }}
QPushButton#allowed {{
    background: #1a1a2d; border-color: {PURPLE}; color: {PURPLE};
    font-weight: 700; font-size: 14px; padding: 10px 20px;
}}
QPushButton#allowed:hover {{ background: #2a2a4a; }}
QPushButton#remove_entirely {{
    background: #2d1a0a; border-color: {ORANGE}; color: {ORANGE};
    font-weight: 700; font-size: 13px; padding: 8px 16px;
}}
QPushButton#remove_entirely:hover {{ background: #4a2a10; }}
QPushButton#remove_partial {{
    background: #2d0a1a; border-color: {PINK}; color: {PINK};
    font-weight: 700; font-size: 13px; padding: 8px 16px;
}}
QPushButton#remove_partial:hover {{ background: #4a1030; }}
QPushButton#skip {{
    border-color: {AMBER}; color: {AMBER}; padding: 10px 20px;
}}
QComboBox {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 4px; padding: 4px 8px; color: {TEXT};
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background: {SURFACE}; border: 1px solid {BORDER};
    color: {TEXT}; selection-background-color: #21262d;
}}
QTextEdit, QLineEdit {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 4px; color: {TEXT}; padding: 4px;
}}
QProgressBar {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 3px; height: 8px; text-align: center;
}}
QProgressBar::chunk {{ background: {CYAN}; border-radius: 3px; }}
QListWidget {{
    background: {SURFACE}; border: 1px solid {BORDER};
    color: {TEXT}; font-size: 12px;
}}
QListWidget::item:selected {{ background: #1f3a4a; }}
QListWidget::item {{ padding: 3px 2px; }}
QLabel#title {{
    color: {CYAN}; font-size: 18px; font-weight: 700; letter-spacing: 0.5px;
}}
QLabel#subtitle {{ color: {TEXT_DIM}; font-size: 12px; }}
QLabel#stat_good             {{ color: {GREEN};  font-weight: 600; }}
QLabel#stat_final_confirmed  {{ color: {GOLD};   font-weight: 600; }}
QLabel#stat_processed        {{ color: {TEAL};   font-weight: 600; }}
QLabel#stat_bad              {{ color: {RED};    font-weight: 600; }}
QLabel#stat_allowed          {{ color: {PURPLE}; font-weight: 600; }}
QLabel#stat_pend             {{ color: {AMBER};  font-weight: 600; }}
QLabel#stat_remove_entirely  {{ color: {ORANGE}; font-weight: 600; }}
QLabel#stat_remove_partial   {{ color: {PINK};   font-weight: 600; }}
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 5px;
    margin-top: 10px; padding-top: 6px;
    color: {TEXT_DIM}; font-size: 11px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QGroupBox::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {BORDER}; border-radius: 3px;
    background: {SURFACE};
}}
QGroupBox::indicator:checked {{ background: {CYAN}; border-color: {CYAN}; }}
QGroupBox::indicator:hover   {{ border-color: {CYAN}; }}
QStatusBar {{
    background: {SURFACE}; color: {TEXT_DIM}; border-top: 1px solid {BORDER};
}}
QSplitter::handle {{ background: {BORDER}; width: 1px; }}
QTabWidget::pane {{
    border: 1px solid {BORDER}; background: {SURFACE}; border-radius: 4px;
}}
QTabBar::tab {{
    background: {BG}; color: {TEXT_DIM};
    border: 1px solid {BORDER}; border-bottom: none;
    padding: 4px 7px; font-size: 11px;
}}
QTabBar::tab:selected {{ background: {SURFACE}; color: {TEXT}; }}
QTabBar::tab:hover    {{ color: {CYAN}; }}
QCheckBox {{ spacing: 5px; }}
QCheckBox::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {BORDER}; border-radius: 3px;
    background: {SURFACE};
}}
QCheckBox::indicator:checked {{ background: {CYAN}; border-color: {CYAN}; }}
"""

# ── Helpers ───────────────────────────────────────────────────────────────


# BIDS field names used for the category/region filters. Persisted per-dataset
# in the SQLite settings table and edited via the "⚙ Keys…" dialog.
KEY_CFG: dict[str, str] = {"category": "seg", "region": "desc"}


def _bids_get(p: Path, field: str) -> str | None:
    """Read a BIDS field from a filename.

    Prefer TPTBox, fall back to a simple '<field>-<value>' scan so we work
    even outside a full BIDS tree.
    """
    try:
        v = BIDS_FILE(p, "").get(field, None)
        if v not in (None, ""):
            return v
    except Exception:
        pass
    prefix = f"{field}-"
    for part in p.stem.split("_"):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def category_key_from_path(p: Path) -> str | None:
    """The category key (default BIDS field: 'seg')."""
    return _bids_get(p, KEY_CFG["category"])


def region_key_from_path(p: Path) -> str | None:
    """The region key (default BIDS field: 'desc'); falls back to category."""
    v = _bids_get(p, KEY_CFG["region"])
    return v if v is not None else category_key_from_path(p)


def get_subj(p: Path) -> str:
    """Return a ``sub-…_ses-…_sequ-…`` label parsed from a BIDS filename."""
    bf = {}
    for k in p.name.split("_"):
        if "-" in k:
            a, b = k.split("-", maxsplit=1)
            bf[a] = b
    return f"sub-{bf.get('sub')}_ses-{bf.get('ses')}_sequ-{bf.get('sequ')}"


def subject_label(p: Path) -> tuple[str, str]:
    """Return the ``(subject_label, region_label)`` pair displayed in list rows."""
    subj = get_subj(p)
    rk = region_key_from_path(p)
    region_lbl = rk or "?"
    return subj, region_lbl


def scan_snapshots(dataset_path: Path, parent: str = "derivatives-VIBESeg-12-points-snp") -> list[Path]:
    """Return a sorted list of ``*_snp.jpg``/``*_snp.png`` paths under ``parent``."""
    print("scan_snapshots")
    snp_root = dataset_path / parent
    if not snp_root.exists():
        return []

    snapshots: list[Path] = list(tqdm(snp_root.rglob("*_snp.jpg"), desc="Scanning JPGs"))
    snapshots.extend(tqdm(snp_root.rglob("*_snp.png"), desc="Scanning PNGs"))
    return sorted(snapshots)


class ImageCache:
    """Thread-safe LRU cache of decoded QPixmaps, keyed by path string.

    • get_or_load() is used by the UI thread for the snapshot currently being
      displayed: it loads synchronously (usually a cache hit) and marks the
      entry as most-recently-used.
    • prefetch() schedules a background QThread that decodes a list of
      upcoming paths and inserts them into the cache without blocking the UI.
    • The cache holds at most `max_size` entries; once full, the least
      recently used entry is evicted.
    """

    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        """Create an empty LRU cache with room for ``max_size`` pixmaps."""
        self.max_size = max_size
        self._cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_load(self, path: Path) -> QPixmap:
        """Return the cached QPixmap for ``path``, loading it if not present."""
        key = str(path)
        with self._lock:
            pix = self._cache.get(key)
            if pix is not None:
                self._cache.move_to_end(key)
                return pix
        pix = QPixmap(key)
        self._insert(key, pix)
        return pix

    def peek(self, path: Path) -> QPixmap | None:
        """Non-mutating lookup used by the prefetch worker to skip cache hits."""
        with self._lock:
            return self._cache.get(str(path))

    def _insert(self, key: str, pix: QPixmap):
        with self._lock:
            self._cache[key] = pix
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def _remove(self, key):
        skey = str(key)
        with self._lock:
            self._cache.pop(skey, None)

    def prefetch(self, paths: list[Path], max_new: int | None = None) -> None:
        """Fire-and-forget background decode of `paths` into the cache.

        ``max_new`` caps how many *newly-loaded* items a single call kicks off
        (defaults to the whole list).  This lets the caller throttle bursts.
        """
        missing = [p for p in paths if self.peek(p) is None]
        if not missing:
            return
        if max_new is not None and max_new > 0 and len(missing) > max_new:
            missing = missing[:max_new]
        worker = threading.Thread(target=self._prefetch_worker, args=(missing,), daemon=True)
        worker.start()

    def _prefetch_worker(self, paths: list[Path]):
        for p in paths:
            if self.peek(p) is not None:
                continue
            pix = QPixmap(str(p))
            self._insert(str(p), pix)

    def clear(self) -> None:
        """Drop every cached entry."""
        with self._lock:
            self._cache.clear()


# ── Review log ────────────────────────────────────────────────────────────


class ReviewLog:
    """Persistent log of review decisions.

    Primary store: SQLite (crash-safe, WAL mode, per-row updates in ms).
    Backup: the original JSON file is still loaded (for legacy datasets)
    and re-written periodically in the background as a human-readable
    snapshot. Any verdict key is accepted — the verdict column is free text,
    so adding new entries to VERDICT_META needs no schema migration.
    """

    def __init__(
        self, dataset_path: Path, name: str = "review_log", parent="derivatives-VIBESeg-12-points-snp", load_shared=False, migrate=False
    ):
        self.path = dataset_path / parent / f".{name}.snp-review.json"  # legacy / backup
        self.db_path = dataset_path / parent / f".{name}.snp-review.sqlite"
        self.root = dataset_path / parent
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = {}
        self._lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._save_running = False
        self._save_pending = False
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._init_db()
        self._load(load_shared, migrate)

    def _init_db(self):
        with self._db_lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS reviews ( key TEXT PRIMARY KEY, verdict TEXT NOT NULL, reason TEXT, ts REAL)")
            self._db.execute("CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT)")

    def get_setting(self, name: str, default: str | None = None) -> str | None:
        """Return the persisted setting ``name``, or ``default`` if unset."""
        with self._db_lock:
            row = self._db.execute("SELECT value FROM settings WHERE name = ?", (name,)).fetchone()
        return row[0] if row else default

    def set_setting(self, name: str, value: str) -> None:
        """Upsert a persisted setting."""
        with self._db_lock:
            self._db.execute("INSERT OR REPLACE INTO settings(name, value) VALUES (?, ?)", (name, value))

    # ── persistence ───────────────────────────────────────────────────────

    def _load(self, load_shared, migrate):
        self._data = {}

        # 1) Load whatever is already in SQLite (fast, per-row).
        with self._db_lock:
            rows = self._db.execute("SELECT key, verdict, reason, ts FROM reviews").fetchall()
        for key, verdict, reason, ts in rows:
            self._data[key] = {"verdict": verdict, "reason": reason or "", "ts": ts or 0.0}

        # 2) Merge legacy JSON as a fallback / backup source.
        legacy: dict = {}
        if load_shared:
            for s in self.path.parent.glob("*.snp-review.json"):
                try:
                    legacy.update(json.loads(s.read_text()))
                except Exception:
                    pass
        else:
            try:
                legacy.update(json.loads(self.path.read_text()))
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.on_fail(e)

        if migrate:
            migrated = {}
            for k, v in legacy.items():
                p = Path(k)
                try:
                    new_key = self._key(p) if p.is_absolute() else p.as_posix()
                except Exception:
                    new_key = p.name
                migrated[new_key] = v
            legacy = migrated

        # Prefer SQLite entries; import any JSON-only ones (and persist them).
        new_from_json = {k: v for k, v in legacy.items() if k not in self._data}
        if new_from_json:
            self._data.update(new_from_json)
            self._bulk_upsert(new_from_json)

    def _bulk_upsert(self, items: dict):
        rows = [(k, v.get("verdict", ""), v.get("reason", ""), v.get("ts", time.time())) for k, v in items.items()]
        with self._db_lock:
            self._db.execute("BEGIN")
            try:
                self._db.executemany(
                    "INSERT OR REPLACE INTO reviews(key, verdict, reason, ts) VALUES (?, ?, ?, ?)",
                    rows,
                )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def _db_upsert(self, key: str, verdict: str, reason: str, ts: float):
        with self._db_lock:
            self._db.execute(
                "INSERT OR REPLACE INTO reviews(key, verdict, reason, ts) VALUES (?, ?, ?, ?)",
                (key, verdict, reason, ts),
            )

    def _save(self):
        with self._lock:
            self._save_pending = True
            if self._save_running:
                return
            self._save_running = True
        s = threading.Thread(target=self._save_worker, daemon=True)
        s.start()
        return s

    def _write_atomic(self, data: dict):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        backup = self.path.with_suffix(self.path.suffix + ".back")
        try:
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            if self.path.exists():
                if backup.exists():
                    backup.unlink()
                self.path.rename(backup)

            tmp.rename(self.path)
            if backup.exists():
                backup.unlink()
        except Exception:
            try:
                if backup.exists() and not self.path.exists():
                    backup.rename(self.path)
            except Exception:
                pass
            raise

    def _save_worker(self):
        # JSON is now only a periodic backup (SQLite is the source of truth).
        # Coalesce bursts of edits into a single rewrite every ~60s so a
        # single click never pays the cost of dumping 10k entries to disk.
        BACKUP_INTERVAL = 60.0
        while True:
            time.sleep(BACKUP_INTERVAL)
            with self._lock:
                if not self._save_pending:
                    self._save_running = False
                    return
                self._save_pending = False
                data = dict(self._data)
            try:
                self._write_atomic(data)
            except Exception as e:
                logger.on_fail(e)

    # ── read helpers ──────────────────────────────────────────────────────

    def _key(self, p: Path):
        if p.is_absolute():
            return p.relative_to(self.root).as_posix()
        else:
            return str(p)

    def get_verdict(self, snp_path: Path) -> dict | None:
        """Return the review-log entry for ``snp_path`` (``None`` if unmarked)."""
        key = self._key(snp_path)

        # New format
        if key in self._data:
            return self._data[key]

        # Old format (absolute path)
        return self._data.get(str(snp_path))

    def get_effective_verdict(self, snp_path: Path) -> str | None:
        """Return the effective verdict string.

        External software marks items 'processed' directly in the JSON.
        If our log has no entry, fall back to checking for an externally-set
        'processed' field in any sibling JSON (simple convention: same key).
        """
        entry = self._data.get(self._key(snp_path))
        if entry:
            return entry.get("verdict")
        return None

    def all_with_verdict(self, verdict: str) -> list[str]:
        """Return every key currently tagged with ``verdict``."""
        return [k for k, v in self._data.items() if v.get("verdict") == verdict]

    # ── write helpers ─────────────────────────────────────────────────────

    def _set(self, snp_path: Path, verdict: str, reason: str = "", save=True):
        key = self._key(snp_path)
        ts = time.time()
        self._data[key] = {"verdict": verdict, "reason": reason, "ts": ts}
        # Primary durable write: one-row SQLite upsert (ms, crash-safe via WAL).
        # Works for any verdict string — schema doesn't constrain keys.
        try:
            self._db_upsert(key, verdict, reason, ts)
        except Exception as e:
            logger.on_fail(e)
        # JSON backup is now a slow, periodic snapshot (not on every change).
        if save:
            self._save()

    def mark_good(self, snp_path: Path, reason: str = "") -> None:
        """Mark ``snp_path`` as *good* (upgrades *processed* to *final_confirmed*)."""
        existing = self.get_verdict(snp_path)
        if existing and existing.get("verdict") in ["processed", "final_confirmed"]:
            self._set(snp_path, "final_confirmed", reason)
        else:
            self._set(snp_path, "good", reason)

    def mark_processed(self, snp_path: Path, reason: str = "", save=False) -> None:
        """Mark ``snp_path`` as externally processed."""
        self._set(snp_path, "processed", reason, save=save)

    def mark_bad(self, snp_path: Path, reason: str = "", save=True) -> None:
        """Mark ``snp_path`` as bad (POI-only issue)."""
        self._set(snp_path, "bad", reason, save=save)

    def mark_allowed_missing(self, snp_path: Path, reason: str = "") -> None:
        """Mark ``snp_path`` as having points outside the FOV (allowed)."""
        self._set(snp_path, "allowed_missing", reason)

    def mark_remove_entirely(self, snp_path: Path, reason: str = "") -> None:
        """Mark ``snp_path`` for full-point removal."""
        self._set(snp_path, "remove_entirely", reason)

    def mark_remove_partial(self, snp_path: Path, reason: str = "") -> None:
        """Mark ``snp_path`` for removal of auto-detected issues only."""
        self._set(snp_path, "remove_partial", reason)

    def stats(self) -> dict[str, int]:
        """Return per-verdict counts plus a ``total`` key."""
        counts: dict[str, int] = dict.fromkeys(VERDICT_META, 0)
        for v in self._data.values():
            vk = v.get("verdict", "")
            if vk in counts:
                counts[vk] += 1

        counts["total"] = len(self._data)
        return counts

    def get_keys(self) -> KeysView[str]:
        """Return the review-log's key iterator (relative paths as stored)."""
        return self._data.keys()


# ── Verdict log panel ─────────────────────────────────────────────────────


class VerdictLogPanel(QWidget):
    """Filterable list for one verdict type.

    Double-clicking emits navigate_to(path_str).
    Region/category filter is set externally via set_filter().
    """

    navigate_to = pyqtSignal(str)

    def __init__(self, verdict: str, color: str, parent=None):
        super().__init__(parent)
        self.verdict = verdict
        self.color = color
        self._entries: list[str] = []
        self._log: ReviewLog | None = None
        # Current external filter state
        self._cat_filter: str = "All categories"
        self._rkey_filter: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        self._list = QListWidget()
        self._list.setWordWrap(True)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list)

        self._count_label = QLabel("0 entries")
        self._count_label.setObjectName("subtitle")
        layout.addWidget(self._count_label)

    def set_filter(self, cat: str, rkey: str | None) -> None:
        """Restrict displayed entries to a given category/region combination."""
        self._cat_filter = cat
        self._rkey_filter = rkey
        self._refresh_display()

    def set_entries(self, path_strings: list[str], log: ReviewLog) -> None:
        """Replace the underlying entry list and repaint."""
        self._entries = path_strings
        self._log = log
        self._refresh_display()

    def _matches_filter(self, p: Path) -> bool:
        ck = category_key_from_path(p) or "?"
        if self._cat_filter not in ("All categories", ck):
            return False
        rk = region_key_from_path(p)
        return not (self._rkey_filter and self._rkey_filter not in ("all", rk))

    def _refresh_display(self):
        self._list.clear()
        shown = 0
        for path_str in self._entries:
            p = Path(path_str)
            if not self._matches_filter(p):
                continue
            subj_name, region_lbl = subject_label(p)
            verdict = self._log.get_verdict(p) if self._log else None
            reason = verdict.get("reason", "") if verdict else ""
            text = f"{subj_name}  ·  {region_lbl}"
            if reason:
                text += f"\n↳ {reason}"
            item = QListWidgetItem(text)
            item.setForeground(QColor(self.color))
            item.setData(Qt.ItemDataRole.UserRole, path_str)
            self._list.addItem(item)
            shown += 1
        total = len(self._entries)
        self._count_label.setText(f"{shown} of {total} entries")

    def _on_double_click(self, item: QListWidgetItem):
        path_str = item.data(Qt.ItemDataRole.UserRole)
        if path_str:
            self.navigate_to.emit(path_str)


def _make_collapsible(group: QGroupBox, expanded: bool = True):
    """Turn a QGroupBox into a collapsible section.

    Click the title checkbox to hide/show its children. Uses no extra widgets.
    """
    group.setCheckable(True)
    group.setChecked(expanded)
    layout = group.layout()

    def _apply(visible: bool):
        if layout is None:
            return
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget()
            if w is not None:
                w.setVisible(visible)

    _apply(expanded)
    group.toggled.connect(_apply)


# ── Filter-key settings dialog ────────────────────────────────────────────


class KeyConfigDialog(QDialog):
    """Edit the BIDS field names used by the category/region key helpers.

    Values are persisted in the SQLite settings table.
    """

    def __init__(self, category: str, region: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Keys")
        self.setModal(True)
        self.setMinimumSize(560, 260)
        layout = QVBoxLayout(self)

        info = QLabel("BIDS field names read from each snapshot filename.\nCategory groups snapshots; region is the finer split.")
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        self.category_edit = QLineEdit(category)
        self.region_edit = QLineEdit(region)
        form.addRow("Category key (e.g. 'seg'):", self.category_edit)
        form.addRow("Region key (e.g. 'desc'):", self.region_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, str]:
        """Return the edited ``(category_key, region_key)`` pair."""
        return self.category_edit.text().strip() or "seg", self.region_edit.text().strip() or "desc"


# ── General settings dialog ───────────────────────────────────────────────


class SettingsDialog(QDialog):
    """Runtime-editable settings.

    Every field maps to a value that is otherwise a module-level default:
      • Slicer executable path
      • Derivatives folders to scan (one per line)
      • Image prefetch count
      • LRU image-cache max size
      • Number of action buttons per row (relayout needs a restart)
    """

    def __init__(
        self,
        slicer_exe: str,
        derivatives_search: list[str],
        prefetch_count: int,
        cache_max_size: int,
        buttons_per_row: int,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Snapshot Reviewer — Settings")
        self.setModal(True)
        self.setMinimumSize(640, 460)
        layout = QVBoxLayout(self)

        info = QLabel(
            "Values are persisted per-dataset in the SQLite settings table.\n"
            "Leave the Slicer path empty to fall back to $PATH / TPTBOX_SLICER_EXE."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        self.slicer_edit = QLineEdit(slicer_exe or "")
        self.slicer_edit.setPlaceholderText("/path/to/Slicer")
        form.addRow("Slicer executable:", self.slicer_edit)

        from PyQt6.QtWidgets import QPlainTextEdit, QSpinBox

        self.deriv_edit = QPlainTextEdit("\n".join(derivatives_search))
        self.deriv_edit.setPlaceholderText("derivatives\nderivatives-spineps\nrawdata\n…")
        self.deriv_edit.setFixedHeight(140)
        form.addRow("Derivatives search folders\n(one per line):", self.deriv_edit)

        self.prefetch_spin = QSpinBox()
        self.prefetch_spin.setRange(0, 500)
        self.prefetch_spin.setValue(int(prefetch_count))
        form.addRow("Prefetch count:", self.prefetch_spin)

        self.cache_spin = QSpinBox()
        self.cache_spin.setRange(1, 10_000)
        self.cache_spin.setValue(int(cache_max_size))
        form.addRow("Image cache max size:", self.cache_spin)

        self.btn_row_spin = QSpinBox()
        self.btn_row_spin.setRange(1, 12)
        self.btn_row_spin.setValue(int(buttons_per_row))
        form.addRow("Buttons per row (restart):", self.btn_row_spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict:
        """Return the edited settings as a keyword-mapping dict."""
        deriv_lines = [ln.strip() for ln in self.deriv_edit.toPlainText().splitlines()]
        deriv = [ln for ln in deriv_lines if ln]
        return {
            "slicer_exe": self.slicer_edit.text().strip(),
            "derivatives_search": deriv,
            "prefetch_count": int(self.prefetch_spin.value()),
            "cache_max_size": int(self.cache_spin.value()),
            "buttons_per_row": int(self.btn_row_spin.value()),
        }


# ── Main window ───────────────────────────────────────────────────────────


class ReviewWindow(QMainWindow):
    """Top-level snapshot-reviewer window and controller."""

    def __init__(
        self,
        dataset_path: Path,
        name: str,
        parent: str = DEFAULT_SNAPSHOT_PARENT,
        slicer_exe: str | None = None,
        derivatives_search: list[str] | None = None,
        prefetch_count: int | None = None,
        cache_max_size: int | None = None,
        buttons_per_row: int | None = None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self._parent = parent
        self.log = ReviewLog(dataset_path, name=name, parent=parent)
        # Hydrate the module-level filter-key config from stored settings.
        KEY_CFG["category"] = self.log.get_setting("key.category", KEY_CFG["category"]) or "seg"
        KEY_CFG["region"] = self.log.get_setting("key.region", KEY_CFG["region"]) or "desc"

        # Resolve tunables: CLI argument (if supplied) overrides the stored
        # setting, which overrides the module default. Every value is
        # afterwards persisted back to the settings table.
        self.slicer_exe: str = slicer_exe if slicer_exe is not None else (self.log.get_setting("slicer.exe", SLICER_EXE) or SLICER_EXE)
        stored_deriv = self.log.get_setting("derivatives.search", None)
        if derivatives_search is not None:
            self.derivatives_search: list[str] = list(derivatives_search)
        elif stored_deriv:
            try:
                self.derivatives_search = [s for s in json.loads(stored_deriv) if isinstance(s, str)]
            except Exception:
                self.derivatives_search = list(_DERIVATIVES_SEARCH)
        else:
            self.derivatives_search = list(_DERIVATIVES_SEARCH)
        self.prefetch_count: int = int(
            prefetch_count
            if prefetch_count is not None
            else (self.log.get_setting("image.prefetch_count", str(PREFETCH_COUNT)) or PREFETCH_COUNT)
        )
        self.cache_max_size: int = int(
            cache_max_size
            if cache_max_size is not None
            else (self.log.get_setting("image.cache_max_size", str(CACHE_MAX_SIZE)) or CACHE_MAX_SIZE)
        )
        self.buttons_per_row: int = max(
            1,
            int(
                buttons_per_row
                if buttons_per_row is not None
                else (self.log.get_setting("ui.buttons_per_row", str(BUTTONS_PER_ROW)) or BUTTONS_PER_ROW)
            ),
        )
        # Persist the resolved values so they are stable across restarts.
        self.log.set_setting("slicer.exe", self.slicer_exe)
        self.log.set_setting("derivatives.search", json.dumps(self.derivatives_search))
        self.log.set_setting("image.prefetch_count", str(self.prefetch_count))
        self.log.set_setting("image.cache_max_size", str(self.cache_max_size))
        self.log.set_setting("ui.buttons_per_row", str(self.buttons_per_row))

        self.all_snapshots: list[Path] = []
        self.queue: list[Path] = []
        self.current_idx: int = 0
        self._injected_path: Path | None = None  # temporarily injected entry
        self.image_cache = ImageCache(max_size=self.cache_max_size)

        self.setWindowTitle("Snapshot Reviewer")
        self.resize(1560, 920)
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._setup_shortcuts()
        self._reload_snapshots()

        QTimer.singleShot(0, self._apply_filter)
        self._known_paths: set[str] = {str(p) for p in self.all_snapshots}

    # ─────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left panel ────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(470)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(10)

        lbl_title = QLabel("SNP Review")
        lbl_title.setObjectName("title")
        lv.addWidget(lbl_title)
        lbl_sub = QLabel("snapshot reviewer")
        lbl_sub.setObjectName("subtitle")
        lv.addWidget(lbl_sub)
        lv.addSpacing(4)

        # ── Filter-key + general settings buttons ─────────────────────────
        settings_row = QHBoxLayout()
        self.btn_key_cfg = QPushButton("⚙  Filter Keys…")
        self.btn_key_cfg.clicked.connect(self._open_key_config)
        settings_row.addWidget(self.btn_key_cfg)
        self.btn_settings = QPushButton("⚙  Settings…")
        self.btn_settings.clicked.connect(self._open_settings)
        settings_row.addWidget(self.btn_settings)
        lv.addLayout(settings_row)

        # ── Region filter ─────────────────────────────────────────────────
        cat_group = QGroupBox("Region filter")
        cat_layout = QVBoxLayout(cat_group)
        cat_layout.setSpacing(4)

        self.cat_combo = QComboBox()
        self.cat_combo.addItem("All categories")
        self.cat_combo.currentTextChanged.connect(self._on_category_changed)
        cat_layout.addWidget(self.cat_combo)

        self.region_combo = QComboBox()
        self.region_combo.addItem("All regions", None)
        self.region_combo.currentIndexChanged.connect(self._on_filter_changed)
        cat_layout.addWidget(self.region_combo)

        lv.addWidget(cat_group)

        # populated once snapshots are scanned, from the seg-*/desc-* keys
        # actually present on disk (see _populate_filter_keys)
        self._regions_by_cat: dict[str, set[str]] = {}

        # ── Hide-reviewed selector ────────────────────────────────────────
        hide_group = QGroupBox("Hide from review queue")
        hide_layout = QVBoxLayout(hide_group)
        hide_layout.setSpacing(3)

        # One checkbox per verdict — derived from VERDICT_META.
        # Resolved verdicts default to hidden; others default to visible.
        self._hide_checks: dict[str, QCheckBox] = {}
        for verdict_key, (label, color, _btn, _sc) in VERDICT_META.items():
            cb = QCheckBox(label)
            cb.setChecked(verdict_key in RESOLVED_VERDICTS)
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; }}")
            cb.stateChanged.connect(self._on_filter_changed)
            hide_layout.addWidget(cb)
            self._hide_checks[verdict_key] = cb

        _make_collapsible(hide_group, expanded=False)
        lv.addWidget(hide_group)

        # ── Stats ─────────────────────────────────────────────────────────
        stats_group = QGroupBox("Progress")
        sv = QVBoxLayout(stats_group)
        sv.setSpacing(3)
        self._stat_labels: dict[str, QLabel] = {}
        for key, (label, color, _btn, _sc) in VERDICT_META.items():
            lbl = QLabel(f"{label}: 0")
            lbl.setStyleSheet(f"color: {color}; font-weight: 600;")
            sv.addWidget(lbl)
            self._stat_labels[key] = lbl
        self.lbl_pend = QLabel("· Pending: 0")
        self.lbl_pend.setObjectName("stat_pend")
        self.lbl_total = QLabel("· Total: 0")
        sv.addWidget(self.lbl_pend)
        sv.addWidget(self.lbl_total)
        self.prog_bar = QProgressBar()
        self.prog_bar.setTextVisible(False)
        sv.addWidget(self.prog_bar)
        _make_collapsible(stats_group, expanded=True)
        lv.addWidget(stats_group)

        # ── Queue list ────────────────────────────────────────────────────
        lv.addWidget(QLabel("Review queue"))
        self.queue_list = QListWidget()
        self.queue_list.itemDoubleClicked.connect(self._on_queue_item_clicked)
        self.queue_list.currentRowChanged.connect(self._on_queue_row_changed)
        lv.addWidget(self.queue_list)

        splitter.addWidget(left)

        # ── Center panel ──────────────────────────────────────────────────
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(16, 16, 16, 16)
        cv.setSpacing(10)

        img_frame = QFrame()
        img_frame.setObjectName("card")
        img_layout = QVBoxLayout(img_frame)
        img_layout.setContentsMargins(0, 0, 0, 0)
        self.img_label = QLabel("No snapshots available")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.img_label.setMinimumHeight(420)
        img_layout.addWidget(self.img_label)
        cv.addWidget(img_frame, stretch=1)

        self.lbl_info = QLabel("")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_info.setObjectName("subtitle")
        cv.addWidget(self.lbl_info)

        # Navigation row
        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton("◀  Prev  [A]")
        self.btn_prev.clicked.connect(self._prev)
        nav_row.addWidget(self.btn_prev)
        self.lbl_position = QLabel("0 / 0")
        self.lbl_position.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_position.setMinimumWidth(80)
        nav_row.addWidget(self.lbl_position)
        self.btn_next = QPushButton("Next  [D]  ▶")
        self.btn_next.clicked.connect(self._next)
        nav_row.addWidget(self.btn_next)
        self.btn_slicer = QPushButton("Slicer  [L]")
        self.btn_slicer.setObjectName("skip")  # amber outline, same as Skip
        self.btn_slicer.clicked.connect(self._open_in_slicer)
        nav_row.addWidget(self.btn_slicer)
        self.btn_toggle_log = QPushButton("Show Log  [V]")
        self.btn_toggle_log.setCheckable(True)
        self.btn_toggle_log.clicked.connect(self._toggle_verdict_log)
        nav_row.addWidget(self.btn_toggle_log)
        cv.addLayout(nav_row)

        # ── Dynamic verdict-button grid (BUTTONS_PER_ROW per row) ─────────
        self._verdict_buttons: dict[str, QPushButton] = {}
        grid = QGridLayout()
        grid.setSpacing(12)
        actionable = [(k, m) for k, m in VERDICT_META.items() if m[2] is not None]
        n = max(1, self.buttons_per_row)
        for i, (vk, (_label, color, btn_text, shortcut)) in enumerate(actionable):
            text = f"{btn_text}  [{shortcut}]" if shortcut else btn_text
            btn = QPushButton(text)
            btn.setStyleSheet(
                f"QPushButton {{ border: 1px solid {color}; color: {color};"
                f" font-weight: 700; font-size: 13px; padding: 8px 16px; }}"
                f"QPushButton:hover {{ background: {color}22; }}"
            )
            btn.clicked.connect(lambda _=False, key=vk: self._mark(key))
            grid.addWidget(btn, i // n, i % n)
            self._verdict_buttons[vk] = btn
        # Skip slots into the next free grid cell so it stays with the group.
        self.btn_skip = QPushButton("Skip  [S]")
        self.btn_skip.setObjectName("skip")
        self.btn_skip.clicked.connect(self._next)
        skip_idx = len(actionable)
        grid.addWidget(self.btn_skip, skip_idx // n, skip_idx % n)
        cv.addLayout(grid)
        self.btn_good = self._verdict_buttons.get("good")

        # Reason field
        reason_row = QHBoxLayout()
        reason_lbl = QLabel("Reason (optional):")
        reason_lbl.setObjectName("subtitle")
        reason_row.addWidget(reason_lbl)
        self.reason_edit = QLineEdit()
        self.reason_edit.setPlaceholderText("e.g. 'point on wrong bone', 'shifted FOV', 'no data for region' …")
        reason_row.addWidget(self.reason_edit)
        cv.addLayout(reason_row)

        splitter.addWidget(center)

        # ── Right panel: tabbed verdict logs (dynamic from VERDICT_META) ──
        self._right_panel = QWidget()
        self._right_panel.setFixedWidth(360)
        rv = QVBoxLayout(self._right_panel)
        rv.setContentsMargins(8, 12, 12, 12)
        rv.setSpacing(6)
        rv.addWidget(QLabel("Verdict Log"))

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)

        self._log_panels: dict[str, VerdictLogPanel] = {}
        for verdict_key, (label, color, _btn, _sc) in VERDICT_META.items():
            panel = VerdictLogPanel(verdict_key, color)
            panel.navigate_to.connect(self._navigate_to_path)
            self._tabs.addTab(panel, label)
            self._log_panels[verdict_key] = panel

        rv.addWidget(self._tabs)
        splitter.addWidget(self._right_panel)
        self._splitter = splitter
        self._right_panel.setVisible(False)  # default hidden
        splitter.setSizes([270, 1280, 0])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Loading…")

    # ─────────────────────────────────────────────────────────────────────
    # Shortcuts
    # ─────────────────────────────────────────────────────────────────────

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self._delete)
        for vk, (_lbl, _c, btn_text, shortcut) in VERDICT_META.items():
            if btn_text is None or not shortcut:
                continue
            QShortcut(QKeySequence(shortcut), self, lambda key=vk: self._mark(key))
        # Legacy alias: F also triggers Allowed Missing
        if "allowed_missing" in VERDICT_META:
            QShortcut(QKeySequence("F"), self, lambda: self._mark("allowed_missing"))
        QShortcut(QKeySequence("A"), self, self._prev)
        QShortcut(QKeySequence("D"), self, self._next)
        QShortcut(QKeySequence("S"), self, self._next)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._prev)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._next)
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, self._prev)
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, self._next)
        QShortcut(QKeySequence("L"), self, self._open_in_slicer)
        QShortcut(QKeySequence("V"), self, self._toggle_verdict_log)

    # ─────────────────────────────────────────────────────────────────────
    # Filter helpers
    # ─────────────────────────────────────────────────────────────────────

    def _current_filter(self) -> tuple[str, str | None]:
        """Return (category_string, region_key_or_None)."""
        return self.cat_combo.currentText(), self.region_combo.currentData()

    def _hidden_verdicts(self) -> set[str]:
        return {k for k, cb in self._hide_checks.items() if cb.isChecked()}

    # ─────────────────────────────────────────────────────────────────────
    # Data management
    # ─────────────────────────────────────────────────────────────────────

    def _reload_snapshots(self):
        self.all_snapshots = scan_snapshots(self.dataset_path, self._parent)
        self._populate_filter_keys()
        self._apply_filter()
        self._update_stats()
        self._update_verdict_log()

    def _apply_filter(self):
        cat, rkey = self._current_filter()
        hide_set = self._hidden_verdicts()

        filtered: list[Path] = []
        resolved_count = 0

        for p in self.all_snapshots:
            pcat = category_key_from_path(p) or "?"
            if cat not in ("All categories", pcat):
                continue
            rk = region_key_from_path(p) or pcat
            if rkey and rkey not in ("all", rk):
                continue
            verdict = self.log.get_verdict(p)
            vk = verdict["verdict"] if verdict else None
            if vk in hide_set:
                continue
            filtered.append(p)
            if vk in RESOLVED_VERDICTS:
                resolved_count += 1
        # Keep any temporarily injected path even if it would be filtered out
        if self._injected_path and self._injected_path not in filtered:
            filtered.insert(0, self._injected_path)
        self.queue = filtered
        if self.current_idx >= len(self.queue):
            self.current_idx = max(0, len(self.queue) - 1)
        self._refresh_queue_list()
        self._show_current()
        self._update_stats()
        # Propagate filter to all log panels
        for panel in self._log_panels.values():
            panel.set_filter(cat, rkey)

    def _populate_filter_keys(self):
        """Rebuild the category/region filter combos from disk.

        Uses the seg-*/desc-* keys actually present in self.all_snapshots,
        instead of the static REGIONS list. Category = seg-* value; region =
        desc-* value if present, otherwise falls back to the category itself.
        """
        regions_by_cat: dict[str, set[str]] = {}
        for p in self.all_snapshots:
            cat = category_key_from_path(p) or "?"
            rk = region_key_from_path(p) or cat
            regions_by_cat.setdefault(cat, set()).add(rk)
        self._regions_by_cat = regions_by_cat

        prev_cat = self.cat_combo.currentText() if self.cat_combo.count() else "All categories"
        prev_region = self.region_combo.currentData() if self.region_combo.count() else None

        self.cat_combo.blockSignals(True)
        self.cat_combo.clear()
        self.cat_combo.addItem("All categories")
        for cat in sorted(regions_by_cat):
            self.cat_combo.addItem(cat)
        idx = self.cat_combo.findText(prev_cat)
        self.cat_combo.setCurrentIndex(max(idx, 0))
        self.cat_combo.blockSignals(False)

        self._rebuild_region_combo(keep_region=prev_region)

    def _rebuild_region_combo(self, keep_region: str | None = None):
        """Repopulate region_combo for the currently selected category."""
        cat = self.cat_combo.currentText()
        if cat == "All categories":
            keys: set[str] = set()
            for s in self._regions_by_cat.values():
                keys |= s
        else:
            keys = self._regions_by_cat.get(cat, set())

        self.region_combo.blockSignals(True)
        self.region_combo.clear()
        self.region_combo.addItem("All regions", None)
        for rk in sorted(keys):
            label = rk
            self.region_combo.addItem(label, rk)
        if keep_region:
            idx = self.region_combo.findData(keep_region)
            if idx >= 0:
                self.region_combo.setCurrentIndex(idx)
        self.region_combo.blockSignals(False)

    def _on_category_changed(self, *_):
        self._rebuild_region_combo()
        self._apply_filter()

    def _on_filter_changed(self, *_):
        self._apply_filter()

    def _open_key_config(self):
        dlg = KeyConfigDialog(KEY_CFG["category"], KEY_CFG["region"], self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cat, reg = dlg.values()
        if cat == KEY_CFG["category"] and reg == KEY_CFG["region"]:
            return
        KEY_CFG["category"] = cat
        KEY_CFG["region"] = reg
        self.log.set_setting("key.category", cat)
        self.log.set_setting("key.region", reg)
        # Rebuild the region combos from the new keys, then re-filter/redraw.
        self._populate_filter_keys()
        self._apply_filter()
        self.status_bar.showMessage(f"Filter keys updated: category='{cat}', region='{reg}'", 4000)

    def _open_settings(self):
        dlg = SettingsDialog(
            self.slicer_exe,
            self.derivatives_search,
            self.prefetch_count,
            self.cache_max_size,
            self.buttons_per_row,
            self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        v = dlg.values()

        self.slicer_exe = v["slicer_exe"]
        self.derivatives_search = list(v["derivatives_search"]) or list(_DERIVATIVES_SEARCH)
        old_prefetch, old_cache, old_bpr = self.prefetch_count, self.cache_max_size, self.buttons_per_row
        self.prefetch_count = v["prefetch_count"]
        self.cache_max_size = v["cache_max_size"]
        self.buttons_per_row = v["buttons_per_row"]

        self.log.set_setting("slicer.exe", self.slicer_exe)
        self.log.set_setting("derivatives.search", json.dumps(self.derivatives_search))
        self.log.set_setting("image.prefetch_count", str(self.prefetch_count))
        self.log.set_setting("image.cache_max_size", str(self.cache_max_size))
        self.log.set_setting("ui.buttons_per_row", str(self.buttons_per_row))

        # Reset the cached BIDS_Global_info so the new derivatives list applies.
        if hasattr(self, "_slicer_bgi"):
            del self._slicer_bgi

        # Cache resize: swap in a fresh cache; the old one gets GC'd.
        if self.cache_max_size != old_cache:
            self.image_cache = ImageCache(max_size=self.cache_max_size)

        note = "Settings updated."
        if self.buttons_per_row != old_bpr:
            note += " Buttons-per-row change takes effect after a restart."
        if self.prefetch_count != old_prefetch:
            note += " Prefetch tuned."
        self.status_bar.showMessage(note, 5000)

    def _refresh_queue_list(self):
        self.queue_list.clear()
        for p in self.queue:
            rk = region_key_from_path(p)
            region_lbl = rk or p.name
            subj_name = get_subj(p)
            item = QListWidgetItem(f"{subj_name} · {region_lbl}")
            verdict = self.log.get_verdict(p)
            vk = verdict["verdict"] if verdict else None
            meta = VERDICT_META.get(vk) if vk else None
            if meta is not None:
                item.setForeground(QColor(meta[1]))
            self.queue_list.addItem(item)
        if self.current_idx < self.queue_list.count():
            self.queue_list.setCurrentRow(self.current_idx)

    def _on_queue_item_clicked(self, item: QListWidgetItem):
        self.current_idx = self.queue_list.row(item)
        self._show_current()

    def _on_queue_row_changed(self, row):
        if row < 0:
            return

        self.current_idx = row
        self._show_current()

    # ─────────────────────────────────────────────────────────────────────
    # Log double-click → navigate
    # ─────────────────────────────────────────────────────────────────────

    def _navigate_to_path(self, path_str: str):
        """Jump to the snapshot for `path_str`.

        The verdict log stores DB keys (paths relative to `self.log.root`);
        the queue holds absolute paths. Match by key so relative/absolute
        never mismatches, then inject the absolute path when not present.
        """
        raw = Path(path_str)
        target_abs = raw if raw.is_absolute() else (self.log.root / raw)
        target_key = self.log._key(target_abs)

        # Try existing queue first — compare by log-key, not by Path equality.
        for i, p in enumerate(self.queue):
            if self.log._key(p) == target_key:
                self.current_idx = i
                self._show_current()
                return

        # Not present — inject the absolute path temporarily.
        self._injected_path = target_abs
        self._apply_filter()  # rebuilds queue with the injected entry at [0]

        for i, p in enumerate(self.queue):
            if self.log._key(p) == target_key:
                self.current_idx = i
                self._show_current()
                self.status_bar.showMessage(
                    f"Temporarily showing filtered entry: {target_abs.name}  (navigate away to dismiss)",
                    6000,
                )
                return

        self.status_bar.showMessage(f"Could not locate snapshot: {target_abs.name}", 4000)

    def _clear_injection(self):
        """Remove any temporary injection and re-filter."""
        if self._injected_path is not None:
            self._injected_path = None
            self._apply_filter()

    # ─────────────────────────────────────────────────────────────────────
    # Image cache / prefetch
    # ─────────────────────────────────────────────────────────────────────

    def _prefetch_upcoming(self):
        """Kick off a background prefetch of the next `prefetch_count` snapshots."""
        if not self.queue:
            return
        start = self.current_idx + 1
        upcoming = self.queue[start : start + self.prefetch_count]
        if upcoming:
            # Cap the per-tick disk burst so a big jump doesn't stall the UI.
            self.image_cache.prefetch(upcoming, max_new=max(2, self.prefetch_count // 4))

    # ─────────────────────────────────────────────────────────────────────
    # Display
    # ─────────────────────────────────────────────────────────────────────

    def _show_current(self):
        if not self.queue:
            self.img_label.setText("No snapshots to review\n\nGenerator is running in the background…")
            self.lbl_info.setText("")
            self.lbl_position.setText("0 / 0")
            return

        if self.current_idx >= len(self.queue):
            self.current_idx = len(self.queue) - 1

        p = self.queue[self.current_idx]

        # If we navigated away from the injected entry, clear it
        if self._injected_path is not None and p != self._injected_path:
            # Defer to avoid recursion inside _apply_filter
            QTimer.singleShot(0, self._clear_injection)

        # Use the LRU cache (the last 100 viewed images stay buffered; this
        # call also promotes `p` to most-recently-used so it survives evictions).
        pix = self.image_cache.get_or_load(p)
        if pix.isNull():
            self.img_label.setText(f"Cannot load image:\n{p.name}")
        else:
            scaled = pix.scaled(
                self.img_label.width() - 4,
                self.img_label.height() - 4,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.img_label.setPixmap(scaled)

        rk = region_key_from_path(p)
        region_lbl = rk or "?"
        verdict = self.log.get_verdict(p)
        vk = verdict["verdict"] if verdict else None

        v_suffix = {
            "good": "  ✓ marked good",
            "final_confirmed": "  ★ final confirmed",
            "processed": "  ⟳ processed (external)",
            "bad": f"  ✗ bad (fixable) — {verdict.get('reason', '') if verdict else ''}",
            "allowed_missing": f"  ~ allowed missing — {verdict.get('reason', '') if verdict else ''}",
            "remove_entirely": f"  ✂ remove entirely — {verdict.get('reason', '') if verdict else ''}",
            "remove_partial": f"  ✂ remove partial — {verdict.get('reason', '') if verdict else ''}",
        }.get(vk, "")  # type: ignore

        # Good button label changes if item is 'processed'
        if vk == "processed":
            self.btn_good.setText("★  Final Confirm  [G]")
        else:
            self.btn_good.setText("✓  Good  [G]")

        subj_name = get_subj(p)
        self.lbl_info.setText(f"{subj_name}  ·  {region_lbl}{v_suffix}")
        self.lbl_position.setText(f"{self.current_idx + 1} / {len(self.queue)}")

        if self.current_idx < self.queue_list.count():
            self.queue_list.setCurrentRow(self.current_idx)

        if verdict:
            self.reason_edit.setText(verdict.get("reason", ""))
        else:
            self.reason_edit.clear()

    def resizeEvent(self, event) -> None:
        """ResizeEvent."""
        super().resizeEvent(event)
        self._show_current()

    # ─────────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────────

    def _current_path(self) -> Path | None:
        if not self.queue or self.current_idx >= len(self.queue):
            return None
        return self.queue[self.current_idx]

    def _delete(self):
        p = self._current_path()
        print("_delete", p)
        if p is not None:
            p.unlink(missing_ok=True)
            self.image_cache._remove(p)
        self._next()

    def _mark(self, verdict_key: str):
        p = self._current_path()
        if p is None or verdict_key not in VERDICT_META:
            return
        reason = self.reason_edit.text().strip()
        self._next()
        QApplication.processEvents()
        if verdict_key == "good":
            self.log.mark_good(p, reason)  # may upgrade to 'final_confirmed'
        else:
            self.log._set(p, verdict_key, reason)

        actual = self.log.get_verdict(p)
        actual_vk = actual["verdict"] if actual else verdict_key
        label = VERDICT_META.get(actual_vk, (verdict_key,))[0]
        self.status_bar.showMessage(f"{label}: {p.name}", 3000)
        self._post_mark()

    def _post_mark(self):
        # Warm the next 10 snapshots in the background so stepping forward
        # (or jumping back into recently-seen territory) doesn't stall on I/O.
        self._prefetch_upcoming()
        self._update_stats()
        self._update_verdict_log()
        self._refresh_queue_list()
        self._show_current()

    def _open_in_slicer(self):
        """Open the current snapshot's BIDS family in 3D Slicer."""
        p = self._current_path()
        if p is None:
            self.status_bar.showMessage("No snapshot selected.", 3000)
            return

        # Build BIDS_Global_info once and cache it on the window. Copy the
        # module default so we never mutate a shared list; also make sure our
        # snapshot parent folder is included.
        if not hasattr(self, "_slicer_bgi"):
            search = list(self.derivatives_search)
            if self._parent not in search:
                search.append(self._parent)
            self._slicer_bgi = BIDS_Global_info(self.dataset_path, search)

        def _on_overwrite(jpg_path: Path, path_str: str, _fam: dict[str, list[BIDS_FILE]], viewed: list[BIDS_FILE]):
            """Default back-hook: log and update status bar."""
            path = Path(path_str)
            fname = path.name
            viewed_names = ", ".join(Path(str(bf)).name for bf in viewed)
            print(f"Slicer wrote: {fname}  |  session had: {viewed_names}")
            # TODO: trigger downstream re-processing here if needed.
            if fname.endswith("mrk.json"):
                logger.on_debug(fname, "mrk.json")
                jpg_path.unlink(missing_ok=True)
                f = path.parent / fname.replace("mrk.json", "json")

                logger.on_debug(f.exists(), f)
                if f.exists():
                    p = POI_Global.load(f)
                    p2 = POI_Global.load(path)
                    assert p.itk_coords == p2.itk_coords
                    for k1, k2, coord in p2.items():
                        p[k1, k2] = coord
                    p.save(f)

        dlg = SlicerLaunchDialog(
            p,
            self.dataset_path,
            bgi=self._slicer_bgi,
            overwrite_callback=_on_overwrite,
            parent=self,
            slicer_exe=self.slicer_exe,
            derivatives_search=self.derivatives_search,
        )
        dlg.exec()

    def _prev(self):
        if not self.queue:
            return
        self.current_idx = max(0, self.current_idx - 1)
        self._show_current()

    def _next(self, *_):
        if not self.queue:
            return
        self.current_idx = min(len(self.queue) - 1, self.current_idx + 1)
        self._show_current()

    # ─────────────────────────────────────────────────────────────────────
    # Stats + log panels
    # ─────────────────────────────────────────────────────────────────────

    def _update_stats(self):
        stats = self.log.stats()
        total_snp = len(self.all_snapshots)

        label_fmt = {
            "good": "✓ Good: {}",
            "final_confirmed": "★ Final Confirmed: {}",
            "processed": "⟳ Processed: {}",
            "allowed_missing": "~ Allowed Missing: {}",
            "bad": "✗ Bad – Fixable: {}",  # noqa: RUF001
            "remove_entirely": "✂ Remove Entirely: {}",
            "remove_partial": "✂ Remove Partial: {}",
        }
        for key, fmt in label_fmt.items():
            self._stat_labels[key].setText(fmt.format(stats.get(key, 0)))
        pending = sum(1 for a in self.queue if self.log._key(a) not in self.log._data)
        self.lbl_pend.setText(f"· Pending: {max(0, pending)}")
        self.lbl_total.setText(f"· Snapshots on disk: {total_snp}")
        self.prog_bar.setMaximum(max(1, total_snp))
        resolved = stats.get("good", 0) + stats.get("final_confirmed", 0) + stats.get("allowed_missing", 0)
        self.prog_bar.setValue(resolved)

    def _update_verdict_log(self):
        if not self._right_panel.isVisible():
            return
        for i, (vk, (label, _c, _b, _s)) in enumerate(VERDICT_META.items()):
            entries = self.log.all_with_verdict(vk)
            panel = self._log_panels.get(vk)
            if panel is not None:
                panel.set_entries(entries, self.log)
            self._tabs.setTabText(i, f"{label} ({len(entries)})")

    def _toggle_verdict_log(self):
        show = not self._right_panel.isVisible()
        self._right_panel.setVisible(show)
        sizes = self._splitter.sizes()
        if show:
            right_w = 360
            sizes = [sizes[0], max(200, sizes[1] - right_w), right_w]
        else:
            sizes = [sizes[0], sizes[1] + sizes[2], 0]
        self._splitter.setSizes(sizes)
        self.btn_toggle_log.setChecked(show)
        self.btn_toggle_log.setText("Hide Log  [V]" if show else "Show Log  [V]")
        if show:
            self._update_verdict_log()


# ── Entry point ───────────────────────────────────────────────────────────


def main() -> None:
    """Main loop."""
    parser = argparse.ArgumentParser(description="Review anatomy point snapshots")
    parser.add_argument("dataset", help="Path to BIDS dataset root")
    parser.add_argument("--name", default="review_log", help="Log file name stem")
    parser.add_argument(
        "-p",
        "--parent-dir",
        dest="parent_dir",
        type=str,
        default=DEFAULT_SNAPSHOT_PARENT,
        help=f"Top-level folder under the dataset that holds the *_snp.jpg/png snapshots (default: {DEFAULT_SNAPSHOT_PARENT})",
    )
    parser.add_argument("--slicer-exe", default=None, help="Path to the 3D Slicer executable (overrides settings and $TPTBOX_SLICER_EXE)")
    parser.add_argument(
        "--derivatives",
        default=None,
        help="Comma-separated list of derivatives folders to scan for BIDS files (overrides stored setting)",
    )
    parser.add_argument("--prefetch", type=int, default=None, help="How many upcoming snapshots to prefetch")
    parser.add_argument("--cache-size", type=int, default=None, help="LRU image cache size (decoded pixmaps)")
    parser.add_argument("--buttons-per-row", type=int, default=None, help="Number of action buttons per row in the review panel")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset path does not exist: {dataset_path}")
        sys.exit(1)

    deriv = None
    if args.derivatives:
        deriv = [s.strip() for s in args.derivatives.split(",") if s.strip()]

    app = QApplication(sys.argv)
    app.setApplicationName("SNP Review")
    win = ReviewWindow(
        dataset_path,
        args.name,
        args.parent_dir,
        slicer_exe=args.slicer_exe,
        derivatives_search=deriv,
        prefetch_count=args.prefetch,
        cache_max_size=args.cache_size,
        buttons_per_row=args.buttons_per_row,
    )
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
