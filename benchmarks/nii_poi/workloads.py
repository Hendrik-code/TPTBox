"""Workload construction for the NII/POI benchmarks.

Six cases spanning a small/large x 2D/3D grid, so a change that only hurts (say)
large 3D volumes shows up as such instead of being averaged away:

    ct_3d     sample CT   (73, 47, 73)        small 3D
    ct_2d     centre slice of ct_3d           small 2D
    mri_3d    sample MRI  (68, 52, 67)        small 3D
    mri_2d    centre slice of mri_3d          small 2D
    synth_3d  generated   (400, 400, 400)     large 3D
    synth_2d  generated   (400, 400,   1)     large 2D

"2D" is a singleton third axis, not a 2-element shape: ``NII`` has no 2D code
path (orientation/reorient/rescale all assume three axes), so a genuine 2-D
``Nifti1Image`` would crash. A ``(X, Y, 1)`` volume is geometrically 2D and goes
through the normal machinery unharmed.

The sample data is read by path rather than through ``TPTBox.tests.test_utils``.
The benchmark workflow swaps the whole ``TPTBox/`` tree to the PR base commit to
produce the baseline numbers, so anything imported from inside that tree would
silently become the *old* version; this module lives outside it and stays fixed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nibabel as nib  # noqa: E402

from TPTBox.core.nii_wrapper import NII  # noqa: E402

SAMPLE_CT = REPO_ROOT / "TPTBox" / "tests" / "sample_ct"
SAMPLE_MRI = REPO_ROOT / "TPTBox" / "tests" / "sample_mri"

CASE_NAMES = ("ct_3d", "ct_2d", "mri_3d", "mri_2d", "synth_3d", "synth_2d")
DEFAULT_LARGE_SIZE = 400
#: Cases above this voxel count skip measurements marked as quadratic/per-label.
HEAVY_VOXEL_LIMIT = 4_000_000


@dataclass
class Case:
    """One benchmark workload: an intensity image plus its two segmentations."""

    name: str
    img: NII
    vert: NII
    subreg: NII
    is_2d: bool
    #: True for the real spine samples, where the full POI pipeline is meaningful.
    anatomical: bool
    img_path: Path | None = None
    vert_path: Path | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.vert.shape)

    @property
    def voxels(self) -> int:
        return int(np.prod(self.shape))

    @property
    def is_heavy(self) -> bool:
        return self.voxels > HEAVY_VOXEL_LIMIT


def _center_slice(nii: NII) -> NII:
    """Take the centre slice along the raw third array axis, keeping it singleton.

    Uses ``NII.apply_crop`` so the affine is updated with the slice, which a plain
    numpy index would not do. Indexing the raw axis (rather than an anatomical
    one) keeps the result reproducible regardless of the file's orientation.
    """
    c = nii.shape[2] // 2
    return nii.apply_crop((slice(None), slice(None), slice(c, c + 1)))


def _load_sample(folder: Path, stem_img: str, stem_subreg: str, stem_vert: str):
    img = NII.load(folder / stem_img, seg=False)
    subreg = NII.load(folder / stem_subreg, seg=True)
    vert = NII.load(folder / stem_vert, seg=True)
    return img, subreg, vert


def _sample_ct():
    return _load_sample(
        SAMPLE_CT,
        "sub-ct_label-22_ct.nii.gz",
        "sub-ct_seg-subreg_label-22_msk.nii.gz",
        "sub-ct_seg-vert_label-22_msk.nii.gz",
    )


def _sample_mri():
    return _load_sample(
        SAMPLE_MRI,
        "sub-mri_label-6_T2w.nii.gz",
        "sub-mri_seg-subreg_label-6_msk.nii.gz",
        "sub-mri_seg-vert_label-6_msk.nii.gz",
    )


def _synthetic_arrays(size: int, flat: bool):
    """Build a deterministic labelled volume plus a matching intensity volume.

    Labels are hollow cuboids laid out on a regular grid: hollow so ``fill_holes``
    has real work to do, on a grid so they never touch (one connected component
    per label) and the result is identical on every machine. Roughly 20% of the
    volume is foreground, which is enough that ``use_crop=True`` cannot trivialise
    the morphology measurements.

    The intensity volume is a smooth ramp plus the labels rather than noise, so
    that gzip-compressed save/load stays a sane thing to measure.
    """
    shape = (size, size, 1) if flat else (size, size, size)
    seg = np.zeros(shape, dtype=np.uint16)

    grid = 5 if flat else 3
    axes = (0, 1) if flat else (0, 1, 2)
    cell = [shape[a] // grid for a in axes]
    label = 1
    steps = [range(grid) for _ in axes]
    for i in steps[0]:
        for j in steps[1]:
            for k in steps[2] if len(steps) > 2 else [0]:
                idx = (i, j, k)[: len(axes)]
                lo = [idx[a] * cell[a] + cell[a] // 8 for a in range(len(axes))]
                hi = [lo[a] + (cell[a] * 5) // 8 for a in range(len(axes))]
                sl = [slice(lo[a], hi[a]) for a in range(len(axes))]
                if flat:
                    sl.append(slice(None))
                seg[tuple(sl)] = label
                # carve a cavity so fill_holes is not a no-op
                inner = []
                for a in range(len(axes)):
                    pad = max((hi[a] - lo[a]) // 4, 1)
                    inner.append(slice(lo[a] + pad, hi[a] - pad))
                if flat:
                    inner.append(slice(None))
                seg[tuple(inner)] = 0
                label += 1

    ramp = np.linspace(-500, 500, shape[0], dtype=np.int16)[:, None, None] + np.linspace(-200, 200, shape[1], dtype=np.int16)[None, :, None]
    img = (seg.astype(np.int16) * 40 + ramp).astype(np.int16)
    return seg, img


def _synthetic_case(name: str, size: int, flat: bool) -> Case:
    seg, img = _synthetic_arrays(size, flat)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.5
    affine[:3, 3] = (-100.0, -50.0, -25.0)
    vert = NII(nib.nifti1.Nifti1Image(seg, affine), seg=True)
    # A second label space so calc_poi_from_two_segs-style calls have two masks;
    # the synthetic subreg is the vertebra mask remapped into the 50-range.
    subreg = NII(nib.nifti1.Nifti1Image((seg > 0).astype(np.uint16) * 50, affine), seg=True)
    image = NII(nib.nifti1.Nifti1Image(img, affine), seg=False)
    return Case(name=name, img=image, vert=vert, subreg=subreg, is_2d=flat, anatomical=False)


def build_case(name: str, large_size: int = DEFAULT_LARGE_SIZE) -> Case:
    """Construct a single case by name. Built one at a time so the large volumes are not all resident at once."""
    if name in ("ct_3d", "ct_2d"):
        img, subreg, vert = _sample_ct()
        paths = (SAMPLE_CT / "sub-ct_label-22_ct.nii.gz", SAMPLE_CT / "sub-ct_seg-vert_label-22_msk.nii.gz")
    elif name in ("mri_3d", "mri_2d"):
        img, subreg, vert = _sample_mri()
        paths = (SAMPLE_MRI / "sub-mri_label-6_T2w.nii.gz", SAMPLE_MRI / "sub-mri_seg-vert_label-6_msk.nii.gz")
    elif name == "synth_3d":
        return _synthetic_case(name, large_size, flat=False)
    elif name == "synth_2d":
        return _synthetic_case(name, large_size, flat=True)
    else:
        raise ValueError(f"unknown case {name!r}; known: {', '.join(CASE_NAMES)}")

    if name.endswith("_2d"):
        return Case(
            name=name,
            img=_center_slice(img),
            vert=_center_slice(vert),
            subreg=_center_slice(subreg),
            is_2d=True,
            anatomical=False,  # a single slice is not a spine
        )
    return Case(name=name, img=img, vert=vert, subreg=subreg, is_2d=False, anatomical=True, img_path=paths[0], vert_path=paths[1])


def parse_case_names(raw: str | None) -> list[str]:
    if not raw:
        return list(CASE_NAMES)
    out = [c.strip() for c in raw.split(",") if c.strip()]
    unknown = [c for c in out if c not in CASE_NAMES]
    if unknown:
        raise ValueError(f"unknown case(s) {unknown}; known: {', '.join(CASE_NAMES)}")
    return out
