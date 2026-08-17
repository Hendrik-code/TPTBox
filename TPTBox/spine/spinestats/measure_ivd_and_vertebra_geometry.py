"""Geometric and signal measurements for intervertebral discs (IVDs) and vertebrae.

Only :func:`measure_ivd_and_vertebra_geometry` is part of the public API.
Everything else is an implementation detail (prefixed with ``_``) and may
change without notice.

IVD vs. vertebra
----------------
The pipeline works for **both** structure types because orientation is
determined differently depending on the label:

- **Vertebra** (label < ``structure_label``, i.e. the usual case ``structure_label=100``
  and label < 100): anatomical directions (inferior/posterior/right) are
  read directly from precomputed points of interest (POIs), since a
  vertebra's shape does not reliably reveal its own orientation via PCA.
  The exception is the dens axis (label 2 / C2), whose unusual shape makes
  the POI-based "up" direction unreliable, so it is replaced by a
  PCA-estimated axis.

- **IVD** (label >= ``structure_label``): discs have no POIs of their own, so
  the "up" direction is estimated directly from the disc's own voxel mask
  via PCA (:func:`_pca_principal_axes`). The right/posterior axes needed
  for x1-x6 are instead derived from the neighbouring vertebra's bony
  landmarks (spinous process / vertebral arch), see
  :func:`_estimate_right_posterior_axes`.

Which mode is used for a given call is controlled by the ``structure_label``
argument of :func:`measure_ivd_and_vertebra_geometry` (and threaded down to
:func:`_compute_basic_geometry`): pass the default ``100`` to evaluate IVD
labels (i.e. label > 100), or ``0`` to evaluate vertebra labels instead
(i.e. label > 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np
import trimesh
from numpy.linalg import norm
from skimage import measure
from sklearn.decomposition import PCA

from TPTBox import NII, POI, Location, Vertebra_Instance, calc_poi_from_subreg_vert
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import POI
from TPTBox.spine.spinestats.torso_vat_sat import peak_centered_mean

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def measure_ivd_and_vertebra_geometry(
    t2w: NII | None,
    vert: NII,
    spine: NII,
    buffer_poi: Path | None = None,
    step_size_mm: float = 0.5,
    instance_labels: list[int] | None = None,
    structure_label: int = 100,
    erode=1,
) -> dict[int, dict[str, float]]:
    """Extract geometric (and optionally T2 signal) measurements per structure.

    Works for intervertebral discs (IVDs) as well as vertebrae, see the
    module docstring for how the two modes differ internally. Which one is
    used is selected via ``structure_label``.

    Parameters
    ----------
    t2w : NII, optional
        T2-weighted image used to compute the normalized disc/vertebra
        signal. If ``None``, the ``"signal"`` entry is left as ``NaN``.

    vert : NII
        Instance segmentation (each disc or vertebra has a unique label).

    spine : NII
        Semantic subregion segmentation (e.g. spinous process, vertebral
        arch, vertebral body, spinal canal) used to estimate anatomical
        orientation and, for the signal ratio, the spinal canal reference
        region.

    step_size_mm : float, default=0.5
        Grid spacing (in mm) used when sampling height/diameter profiles
        across the structure's surface. Smaller values are more accurate
        but proportionally slower.

    instance_labels : list[int], optional
        Labels to evaluate. If ``None``, defaults to every label in
        ``vert`` greater than ``structure_label``.

    structure_label : int, default=100
        Selects IVD mode (``100``, the default: only labels > 100 are
        valid) or vertebra mode (``0``: only labels > 0 are valid). See the
        module docstring for details. Any value other than ``100`` is
        currently treated like vertebra mode for the minimum-label check.

    Returns:
    -------
    dict[int, dict[str, float]]
        Keyed by structure label. Each entry contains:

        - ``volume_voxel``: volume from the raw voxel mask (mm^3)
        - ``volume_mesh``: volume from the reconstructed surface mesh (mm^3)
        - ``height_center``: height measured through the structure's center
        - ``mean_height`` / ``max_height``: statistics over sampled heights
        - ``lower_10_percent_height``: 10th percentile of sampled heights
        - ``mean_diameter``: diameter of a circle with the same projected
          area as the structure
        - ``anterior_height_x1``, ``posterior_height_x2``,
          ``right_height_x3``, ``left_height_x4``: directional heights,
          see the figure/table below
        - ``width_lateral_x5``, ``width_sagittal_x6``: directional widths
        - ``signal``: peak-centered T2 signal in the structure divided
          by the peak-centered T2 signal in the spinal canal (unitless).
          Peak-centered means only voxels around the histogram mode of
          the mask are averaged; this suppresses darker contamination
          such as nerve roots inside the canal (only if ``t2w`` is given)
        - ``structure_signal``: peak-centered mean T2 signal inside the
          (eroded) structure mask, a.u. (only if ``t2w`` is given)
        - ``spinal_canal_signal``: peak-centered mean T2 signal inside
          the (eroded) spinal canal reference region, a.u. (only if
          ``t2w`` is given)
        - ``signal_old`` / ``structure_signal_old`` /
          ``spinal_canal_signal_old``: same three quantities computed as
          plain per-voxel means (no peak centering). Kept for backward
          comparison so a user can see the effect of the peak filter.

        If evaluation of a label fails, its entry instead contains
        ``"error"`` (the exception message) plus all the same keys set to
        ``NaN``.

    Notes:
    -----
    x1-x6 are the six anatomical dimensions commonly used to describe disc
    (or vertebral body) geometry:

    ====  =========================
    x1    anterior height
    x2    posterior height
    x3    right height
    x4    left height
    x5    lateral width
    x6    sagittal width
    ====  =========================
    """
    results = {}

    # Vertebral direction landmarks (inferior/posterior/right), needed for
    # vertebra mode and as an orientation anchor for the neighbouring
    # vertebra in IVD mode.
    poi = calc_poi_from_subreg_vert(
        vert,
        spine,
        subreg_id=[
            Location.Vertebra_Direction_Inferior,
            Location.Vertebra_Direction_Posterior,
            Location.Vertebra_Direction_Right,
            Location.Vertebra_Corpus,
            Location.Endplate,
        ],
        buffer_file=buffer_poi,
        save_buffer_file=True,
    )
    if instance_labels is None:
        instance_labels = [int(i) for i in vert.unique() if i > structure_label and i < structure_label + 100]
    for label in instance_labels:
        if label == 26:
            continue
        try:
            raw = {}
            # Isolate the current structure (disc or vertebra).
            structure_mask = vert.extract_label(label)
            # 1. volume, central height, mean diameter
            info, raw = _compute_basic_geometry(vert, poi, label, step_size_mm=2, raw=raw, structure_label=structure_label)
            # 2. x1-x6 directional heights/widths
            raw = _compute_directional_heights_widths(vert, structure_mask * spine, poi, label, step_size_mm=step_size_mm, raw=raw)
            # 3. normalized T2 signal
            if t2w is not None:
                raw = _compute_t2_signal_ratio(t2w, vert, spine, label, raw=raw, erode=erode)
            results[label] = _result_from_info(info)
        except Exception as e:
            results[label] = _nan_result(error=str(e))

    return results


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

_RESULT_FIELDS = (
    "volume_voxel",
    "volume_mesh",
    "height_center",
    "mean_height",
    "max_height",
    "lower_10_percent_height",
    "mean_diameter",
    "anterior_height_x1",
    "posterior_height_x2",
    "right_height_x3",
    "left_height_x4",
    "width_lateral_x5",
    "width_sagittal_x6",
    "signal",
    "structure_signal",
    "spinal_canal_signal",
    "signal_old",
    "structure_signal_old",
    "spinal_canal_signal_old",
)


def _result_from_info(info: _StructureMeasurements) -> dict[str, float]:
    """Build the public result dict (see Returns section of the public API) from a filled-in measurement object."""
    return {
        "volume_voxel": round(info.volume_voxel, 4),
        "volume_mesh": round(info.volume_mesh, 4),
        "height_center": round(info.height_center, 4),
        "mean_height": round(info.mean_height, 4),
        "max_height": round(info.max_height, 4),
        "lower_10_percent_height": round(info.get_quantile(10), 4),
        "mean_diameter": round(info.mean_diameter, 4),
        "anterior_height_x1": round(info.anterior_height_x1, 4),
        "posterior_height_x2": round(info.posterior_height_x2, 4),
        "right_height_x3": round(info.right_height_x3, 4),
        "left_height_x4": round(info.left_height_x4, 4),
        "width_lateral_x5": round(info.width_lateral_x5, 4),
        "width_sagittal_x6": round(info.width_sagittal_x6, 4),
        "signal": round(info.signal, 4),
        "structure_signal": round(info.structure_signal, 4),
        "spinal_canal_signal": round(info.spinal_canal_signal, 4),
        "signal_old": round(info.signal_old, 4),
        "structure_signal_old": round(info.structure_signal_old, 4),
        "spinal_canal_signal_old": round(info.spinal_canal_signal_old, 4),
    }  # type: ignore


def _nan_result(error: str | None = None) -> dict[str, float]:
    """Build a NaN-filled result dict, e.g. for a structure whose evaluation raised an exception."""
    result = dict.fromkeys(_RESULT_FIELDS, np.nan)
    if error is not None:
        result["error"] = error
    return result


# ---------------------------------------------------------------------------
# Measurement container
# ---------------------------------------------------------------------------


@dataclass
class _StructureMeasurements:
    """Accumulates the measurements for a single disc/vertebra as they are computed in stages.

    Populated incrementally by :func:`_compute_basic_geometry` (volume/height/area),
    :func:`_compute_directional_heights_widths` (x1-x6), and
    :func:`_compute_t2_signal_ratio` (signal). The ``*_values`` flags mark which
    stages have already run so repeated calls (e.g. across retries) don't
    redo expensive work.
    """

    volume_voxel: float
    volume_mesh: float
    height_center: float
    area: float
    quantiles: dict = None  # type: ignore
    anterior_height_x1: float = np.nan
    posterior_height_x2: float = np.nan
    right_height_x3: float = np.nan
    left_height_x4: float = np.nan
    width_lateral_x5: float = np.nan
    width_sagittal_x6: float = np.nan
    x_values: bool = False
    signal_values: bool = False
    signal: float = np.nan
    structure_signal: float = np.nan
    spinal_canal_signal: float = np.nan
    signal_old: float = np.nan
    structure_signal_old: float = np.nan
    spinal_canal_signal_old: float = np.nan

    @property
    def mean_diameter(self):
        """Diameter of the circle whose area equals the structure's projected area (A = pi*r^2)."""
        return np.sqrt(self.area / np.pi) * 2

    def _update_height_statistics(self, sampled_heights):
        """Compute mean/max/quantiles from a list of sampled point heights."""
        if hasattr(self, "list_heights"):
            delattr(self, "list_heights")
        sampled_heights = sorted(sampled_heights)
        if len(sampled_heights) != 0:
            self.mean_height = np.mean(sampled_heights)
            self.max_height = max(sampled_heights)
            for q in range(0, 100, 5):
                self.get_quantile(q, sampled_heights)
        else:
            self.mean_height = np.nan
            self.max_height = np.nan

    def get_quantile(self, q: int = 50, _sampled_heights: list | None = None):
        """Return the q-th percentile (0-100) of sampled heights, computing and caching it on first use."""
        if self.quantiles is None:
            self.quantiles = {}
        if q in self.quantiles:
            return self.quantiles[q]
        if _sampled_heights is None:
            return None
        self.quantiles[q] = _sampled_heights[int(len(_sampled_heights) * q / 100)]
        return self.quantiles[q]


# ---------------------------------------------------------------------------
# Mesh / geometry primitives
# ---------------------------------------------------------------------------


def _segmentation_to_surface_mesh(segmentation: np.ndarray, voxel_size) -> trimesh.Trimesh:
    """Turn a binary voxel mask into a surface mesh via marching cubes, scaled to real-world (mm) units."""
    verts, faces, _, _ = measure.marching_cubes(segmentation)
    verts *= voxel_size
    return trimesh.Trimesh(vertices=verts, faces=faces)


def _pca_principal_axes(segmentation: np.ndarray, voxel_size, verbose=False, up_axis=-1):
    """Estimate an orthogonal axis system from a voxel mask's principal component axes (PCA).

    Used when no anatomical landmarks are available to derive orientation
    directly from the shape of the structure itself (e.g. IVDs, or the C2
    vertebra whose dens makes the usual POI-based "up" direction unreliable).

    Parameters
    ----------
    up_axis : int, default=-1
        If -1, the "up" vector is simply the 3rd principal component (least
        variance / thinnest axis of the structure). Otherwise, "up" is
        chosen as whichever principal component has the largest projection
        onto this array axis (e.g. the superior/inferior image axis),
        which is more robust when the structure isn't disc-shaped.

    Returns:
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (up_vector, remaining_axis_1, remaining_axis_2)
    """
    points = np.argwhere(segmentation > 0) * voxel_size
    pca = PCA(n_components=3)
    pca.fit(points)
    if up_axis == -1:
        up_vector = pca.components_[2]
        if verbose:
            print(f"Main Axis (PC1): {pca.components_[0]}")
            print(f"Secondary Axis (PC2): {pca.components_[1]}")
            print(f"Up Vector (PC3): {up_vector}")
        return up_vector, pca.components_[0], pca.components_[1]

    # Pick whichever component best aligns with the requested image axis.
    abs_up_values = np.abs(pca.components_[:, up_axis])
    max_index = np.argmax(abs_up_values)
    up_vector = pca.components_[max_index]
    return (
        up_vector,
        pca.components_[(max_index + 1) % 3],
        pca.components_[(max_index + 2) % 3],
    )


def _center_of_mass_voxels(segmentation: np.ndarray, voxel_size=1) -> np.ndarray:
    """Voxel-space center of mass of a binary mask, scaled to real-world units."""
    points = np.argwhere(segmentation > 0)
    return np.mean(points, axis=0) * voxel_size


def _project_mesh_onto_plane(mesh: trimesh.Trimesh, up_vector: np.ndarray) -> trimesh.Trimesh:
    """Flatten a mesh onto the plane perpendicular to ``up_vector`` (orthographic projection along "up")."""
    up_vector = up_vector / np.linalg.norm(up_vector)
    flattened_vertices = mesh.vertices - np.dot(mesh.vertices, up_vector[:, np.newaxis]) * up_vector
    return trimesh.Trimesh(vertices=flattened_vertices, faces=mesh.faces)


def _projected_area(flattened_mesh: trimesh.Trimesh) -> float:
    """2D area of a mesh already flattened onto a plane (trimesh's area works for degenerate/2D meshes too)."""
    return flattened_mesh.area


# ---------------------------------------------------------------------------
# Ray casting helpers
# ---------------------------------------------------------------------------


def _intersect_ray(ray_vector, v1, v2, mesh: trimesh.Trimesh, x: float = 0.0, y: float = 0.0, center=None):
    """Cast a ray through the mesh along ``ray_vector``, offset from ``center`` by ``x``/``y`` along ``v1``/``v2``.

    Used both to measure heights (ray along the "up" direction) and to
    measure diameters (ray along an in-plane direction), depending on which
    vector is passed as ``ray_vector``.
    """
    start_point = mesh.center_mass if center is None else center
    # Start far outside the mesh (on the opposite side of ray_vector) so the ray reliably enters through the surface.
    start_point = start_point - ray_vector * 1000 + v1 * x + v2 * y
    intersections = mesh.ray.intersects_location(ray_origins=np.array([start_point]), ray_directions=np.array([ray_vector]))
    return intersections[0]


def _intersect_ray_from_dirs(dirs: tuple[np.ndarray, np.ndarray, np.ndarray], mesh: trimesh.Trimesh, x=0, y=0, center=None):
    """Convenience wrapper around :func:`_intersect_ray` that unpacks a ``(up, v1, v2)`` direction triple."""
    up_vector, v1, v2 = dirs
    return _intersect_ray(up_vector, v1, v2, mesh=mesh, x=x, y=y, center=center)


def _segment_length(intersection_points) -> float:
    """Euclidean distance between the first and last point where a ray crossed the mesh surface.

    For a ray shot straight through a (roughly convex) structure, this is
    the entry-to-exit distance, i.e. the structure's height/diameter along
    that ray.
    """
    if len(intersection_points) == 0:
        return 0
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(intersection_points[0], intersection_points[-1], strict=False)))


def _swap(a, b):
    """Return ``(b, a)``."""
    return b, a


# ---------------------------------------------------------------------------
# Mesh + orientation extraction (shared between IVD and vertebra mode)
# ---------------------------------------------------------------------------


def _get_mesh_and_directions(nii: NII, poi: POI | None, label: int, raw: dict, recompute_mesh: bool = False):
    """Build (or fetch from cache) the surface mesh and ``(up, posterior, right)`` direction vectors for one structure.

    This is the key place where IVD and vertebra handling diverge:

    - **Vertebra** (``label < 100``): directions come from the POI
      landmarks (``Vertebra_Direction_Inferior/Posterior/Right``), which
      are far more reliable than PCA for the often irregular vertebral
      shape. Exception: label 2 (the C2/dens vertebra) still gets its "up"
      vector from PCA, since its odd shape makes the POI-derived "up"
      direction unreliable.
    - **IVD** (``label >= 100``): there are no POIs for discs, so all three
      directions are estimated from the disc's own voxel mask via PCA
      (:func:`_pca_principal_axes`).

    Results are cached in ``raw`` so repeated calls for the same structure
    reuse the mesh/directions instead of recomputing them.

    Returns:
    -------
    tuple[trimesh.Trimesh, tuple[np.ndarray, np.ndarray, np.ndarray], NII]
        (surface mesh, (up, dir_1, dir_2) direction vectors, binary segmentation)
    """
    voxel_size = nii.zoom
    nii = nii.apply_crop(nii.compute_crop(dist=1))
    segmentation = nii.extract_label(label)
    arr = segmentation.get_array()

    if label < 100:
        assert poi is not None
        center = np.array(poi[label, Location.Vertebra_Corpus])
        up = -np.array(poi[label, Location.Vertebra_Direction_Inferior]) - center
        post = np.array(poi[label, Location.Vertebra_Direction_Posterior]) - center
        right = np.array(poi[label, Location.Vertebra_Direction_Right]) - center
        up /= norm(up)
        post /= norm(post)
        right /= norm(right)
        if label == 2:
            up = _pca_principal_axes(arr, voxel_size, up_axis=nii.get_axis("S"))[0]
        direction_vectors = (up, post, right)
    else:
        up_axis = nii.get_axis("S") if label < 100 else -1
        direction_vectors = (
            raw["direction_vectors"] if "direction_vectors" in raw else _pca_principal_axes(arr, voxel_size, up_axis=up_axis)
        )

    mesh = raw["mesh"] if "mesh" in raw and not recompute_mesh else _segmentation_to_surface_mesh(arr, voxel_size)
    raw["mesh"] = mesh
    raw["direction_vectors"] = direction_vectors
    return mesh, direction_vectors, segmentation


# ---------------------------------------------------------------------------
# Measurement stages (called in order from measure_ivd_and_vertebra_geometry)
# ---------------------------------------------------------------------------

# Maps a `structure_label` value to the minimum label a structure must exceed to be valid.
# `structure_label=100` -> IVD mode (labels must be > 100); any other value -> vertebra mode (labels must be > 0).
_MIN_LABEL_FOR_OFFSET = {100: 100}


def _compute_basic_geometry(
    nii: NII, poi: POI, label: int, step_size_mm: int = 2, raw: dict | None = None, structure_label: int = 100
) -> tuple[_StructureMeasurements, dict]:
    """Compute basic geometric measurements of a single disc or vertebra (stage 1).

    How it's computed
    ------------------
    - ``volume_voxel``: voxel count times voxel volume.
    - ``volume_mesh``: signed volume of the reconstructed surface mesh.
    - ``height_center``: length of the ray-mesh intersection segment through
      the mesh's center of mass, along the "up" direction.
    - ``area`` / ``mean_diameter``: projected area of the mesh flattened
      along "up", converted to an equivalent circle diameter.
    - ``mean_height`` / ``max_height`` / height quantiles: heights are
      sampled on a grid (spacing ``step_size_mm``) covering a disc of
      radius ``mean_diameter`` around the center, each height being the
      ray-mesh intersection length at that grid point.

    Expensive intermediate results (mesh, direction vectors, sampled
    heights) are cached in ``raw`` and reused by later stages/calls.

    Parameters
    ----------
    nii : NII
        Instance segmentation containing the structure.
    poi : POI
        Vertebral points of interest, used for orientation (vertebra mode only).
    label : int
        Label of the structure to evaluate.
    step_size_mm : float, default=2.0
        Sampling spacing (mm) for the height grid.
    raw : dict, optional
        Cache of previously computed mesh / direction vectors / measurements / sampled heights.
    structure_label : int, default=100
        Selects IVD mode (100, labels must be > 100) or vertebra mode
        (any other value, labels must be > 0). See module docstring.

    Returns:
    -------
    tuple[_StructureMeasurements, dict]
        The computed measurements together with the updated cache.
    """
    if raw is None:
        raw = {}

    min_label = _MIN_LABEL_FOR_OFFSET.get(structure_label, 0)
    assert label > min_label, label

    info = raw.get("info")

    # ------------------------------------------------------------------
    # Mesh-based quantities (volume, central height, projected area)
    # ------------------------------------------------------------------
    if not isinstance(info, _StructureMeasurements) or info.height_center == 0:
        mesh, direction_vectors, segmentation = _get_mesh_and_directions(nii, poi, label, raw, recompute_mesh=True)
        volume_voxel = (segmentation.sum() * np.prod(segmentation.zoom)).item()
        center_intersections = _intersect_ray_from_dirs(direction_vectors, mesh, 0, 0)
        height_center = _segment_length(center_intersections)
        area = _projected_area(_project_mesh_onto_plane(mesh.copy(), direction_vectors[0]))
        info = _StructureMeasurements(
            volume_voxel=volume_voxel, volume_mesh=abs(mesh.volume.item()), height_center=height_center, area=area
        )
        raw["info"] = info

    # ------------------------------------------------------------------
    # Sampled heights (mean/max/quantiles) on a grid around the center
    # ------------------------------------------------------------------
    if "list_heights" not in raw or len(raw["list_heights"]) == 0:
        mesh, direction_vectors, _ = _get_mesh_and_directions(nii, poi, label, raw)

        search_radius = int(info.mean_diameter)
        sampled_heights = []

        for x in range(-search_radius, search_radius, step_size_mm):
            for y in range(-search_radius, search_radius, step_size_mm):
                intersections = _intersect_ray_from_dirs(direction_vectors, mesh, x, y)
                height = _segment_length(intersections)
                if height > 0:
                    sampled_heights.append(height)

        raw["list_heights"] = sampled_heights

    info._update_height_statistics(raw["list_heights"])

    return info, raw


def _local_max_height(mesh, direction_vectors, around_point, search_diameter, step_size_mm) -> float:
    """Maximum sampled height in a small neighbourhood around a point (used to pin down x1-x4 at each edge point).

    A single ray through, e.g., the exact anterior-most point can graze the
    mesh edge and under-measure; sampling a small patch around the point
    and taking the max is more robust.
    """
    sampled_heights = []
    d = int(search_diameter / step_size_mm)
    for xi in range(-d, d):
        x = xi * step_size_mm
        for yi in range(-d, d):
            y = yi * step_size_mm
            intersections = _intersect_ray_from_dirs(direction_vectors, mesh, x, y, around_point)
            height = _segment_length(intersections)
            if height != 0:
                sampled_heights.append(height)
    return np.max(sampled_heights)


def _max_diameter_in_plane(ray_vector, v1, v2, mesh, diameter: float = 30, step_size_mm: float = 2.0):
    """Find the widest ray-mesh intersection on a grid in the plane spanned by ``v1``/``v2`` (stage 2 helper).

    Used for both the lateral (x5) and sagittal (x6) width measurements:
    the mesh is probed with rays parallel to ``ray_vector`` (e.g. "right"
    for the lateral width) on a grid spanned by ``v1``/``v2`` (e.g.
    "posterior" and "up"), and the widest intersection found is returned
    together with its two endpoints (used as the anterior/posterior or
    left/right reference points for the height measurements).

    Returns:
    -------
    tuple[float, np.ndarray, np.ndarray, float, float]
        (max_width, point_1, point_2, grid_x, grid_y) for the widest ray found.
    """
    d = ceil(diameter / step_size_mm)
    best_width = 0
    out = (0.0, None, None, 0.0, 0.0)
    for xi in range(-d, d):
        x = xi * step_size_mm
        for yi in range(-d, d):
            y = yi * step_size_mm
            intersections = _intersect_ray(ray_vector, v1, v2, mesh, x, y)
            width = _segment_length(intersections)
            if width == 0:
                continue
            p1 = intersections[0]
            p2 = intersections[-1]
            if best_width < width:
                best_width = width
                out = (width, p1.round(2), p2.round(2), x, y)
    return out


def _compute_directional_heights_widths(nii: NII, subreg: NII, poi, label: int = 123, step_size_mm: float = 0.5, raw: dict | None = None):  # noqa: ARG001
    """Compute the x1-x6 directional heights and widths for one structure (stage 2).

    How it's computed
    ------------------
    1. The right/posterior anatomical axes are estimated from the
       this is shared between IVD and vertebra mode.
    2. ``width_lateral_x5``: widest ray parallel to "right", scanned over a
       grid in the (posterior, up) plane -> also yields the right-most and
       left-most surface points.
    3. ``width_sagittal_x6``: widest ray parallel to "posterior", scanned
       over a grid in the (right, up) plane -> also yields the
       posterior-most and anterior-most surface points.
    4. ``anterior_height_x1`` / ``posterior_height_x2`` / ``right_height_x3``
       / ``left_height_x4``: maximum sampled height (:func:`_local_max_height`)
       in a small patch around each of the four extreme points found above.

    Requires the segmentation to be in an orientation with Right along the
    3rd axis and Posterior along the 1st axis (checked via ``nii.orientation``).
    """
    assert "R" in nii.orientation[2]  # No guarantee this works for other orientations.
    assert "P" in nii.orientation[0]
    if raw is None:
        raw = {}
    info: _StructureMeasurements = raw["info"]
    if info.x_values:
        return raw
    try:
        mesh, direction_vectors, _ = _get_mesh_and_directions(nii, poi, label, raw, recompute_mesh=True)
        up = direction_vectors[0]
        center = np.asarray(poi[label % 100, Location.Vertebra_Corpus], dtype=float)

        right = np.asarray(poi[label % 100, Location.Vertebra_Direction_Right], dtype=float) - center
        front = np.asarray(poi[label % 100, Location.Vertebra_Direction_Posterior], dtype=float) - center
        right /= norm(right)
        front /= norm(front)
        if label >= 100:
            next_vert = Vertebra_Instance(label % 100).get_next_poi(poi)
            right2 = np.asarray(poi[next_vert, Location.Vertebra_Direction_Right], dtype=float) - center
            front2 = np.asarray(poi[next_vert, Location.Vertebra_Direction_Posterior], dtype=float) - center
            right2 /= norm(right2)
            front2 /= norm(front2)
            right = (right + right2) / 2
            front = (front + front2) / 2

        (width_lateral_x5, p_r, p_l, *_) = _max_diameter_in_plane(right, front, up, mesh, 30, step_size_mm)
        axis = nii.get_axis("R")
        flip = 1 if "R" in nii.orientation else -1
        if p_r[axis] * flip < p_l[axis] * flip:
            p_r, p_l = _swap(p_r, p_l)

        (width_sagittal_x6, p_p, p_a, *_) = _max_diameter_in_plane(front, right, up, mesh, 2, step_size_mm)
        axis = nii.get_axis("P")
        flip = 1 if "P" in nii.orientation else -1
        if p_p[axis] * flip < p_a[axis] * flip:
            # NOTE: kept as in the original implementation, though this looks like it
            # should swap (p_p, p_a) rather than (p_r, p_a) -- flag before relying on x6 edge points.
            p_p, p_a = _swap(p_r, p_a)

        info.anterior_height_x1 = _local_max_height(mesh, direction_vectors, p_a, 3, step_size_mm)
        info.posterior_height_x2 = _local_max_height(mesh, direction_vectors, p_p, 3, step_size_mm)
        info.right_height_x3 = _local_max_height(mesh, direction_vectors, p_r, 3, step_size_mm)
        info.left_height_x4 = _local_max_height(mesh, direction_vectors, p_l, 3, step_size_mm)
        info.width_lateral_x5 = width_lateral_x5
        info.width_sagittal_x6 = width_sagittal_x6
        info.x_values = True
    except Exception:
        info.anterior_height_x1 = None  # type: ignore
        info.posterior_height_x2 = None  # type: ignore
        info.right_height_x3 = None  # type: ignore
        info.left_height_x4 = None  # type: ignore
        info.width_lateral_x5 = None  # type: ignore
        info.width_sagittal_x6 = None  # type: ignore

    return raw


def _compute_t2_signal_ratio(
    t2w_nii: NII,
    nii: NII,
    subregs: NII,
    label: int = 123,
    raw: dict | None = None,
    erode=1,
    spinal_bins: int = 64,
    spinal_peak_frac_height: float = 0.5,
):
    """Compute the normalized T2 signal for one structure (stage 3).

    How it's computed
    ------------------
    The mean T2 intensity inside the (slightly eroded, to avoid partial-volume
    edge voxels) structure mask is divided by the mean T2 intensity in the
    spinal canal (subregion label 61, also eroded). The spinal canal is used
    as an internal reference to normalize away scanner/sequence-dependent
    intensity scaling.

    The spinal canal segmentation may contain darker structures such as
    nerve roots. To reduce their influence, both the structure and canal
    signals are also estimated as the mean over a window around the
    histogram peak (see :func:`peak_centered_mean`). Both the plain-mean
    values (``*_old``) and the peak-centered values are stored so the
    caller can compare them.

    Parameters
    ----------
    spinal_bins : int, default=64
        Histogram bin count passed to :func:`peak_centered_mean`.
    spinal_peak_frac_height : float, default=0.5
        Fractional height cutoff passed to :func:`peak_centered_mean`.
    """
    assert "R" in nii.orientation[2]
    assert "P" in nii.orientation[0]
    if raw is None:
        raw = {}
    info: _StructureMeasurements = raw["info"]
    if info.signal_values:
        return raw
    if t2w_nii.shape != nii.shape:
        t2w_nii.resample_from_to_(nii, verbose=False)
    structure_mask = nii.extract_label(label)
    eroded_mask = structure_mask.erode_msk(erode, connectivity=1, verbose=False)
    structure_mask = eroded_mask if eroded_mask.sum() != 0 else structure_mask
    spinal_mask = subregs.extract_label(61).erode_msk(erode, connectivity=1, verbose=False)

    t2w_arr = t2w_nii.get_array()
    structure_vals = t2w_arr[structure_mask.get_array().astype(bool)]
    spinal_vals = t2w_arr[spinal_mask.get_array().astype(bool)]

    structure_signal_old = float(np.mean(structure_vals)) if structure_vals.size > 0 else np.nan
    spinal_canal_signal_old = float(np.mean(spinal_vals)) if spinal_vals.size > 0 else np.nan

    structure_signal = peak_centered_mean(structure_vals, bins=spinal_bins, peak_frac_height=spinal_peak_frac_height)
    spinal_canal_signal = peak_centered_mean(spinal_vals, bins=spinal_bins, peak_frac_height=spinal_peak_frac_height)

    info.signal = structure_signal / spinal_canal_signal
    info.structure_signal = structure_signal
    info.spinal_canal_signal = spinal_canal_signal
    info.signal_old = structure_signal_old / spinal_canal_signal_old
    info.structure_signal_old = structure_signal_old
    info.spinal_canal_signal_old = spinal_canal_signal_old
    info.signal_values = True
    return raw
