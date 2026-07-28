from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import trimesh
from stl import Mesh

from TPTBox import NII, POI, Location, Logger_Interface, Print_Logger
from TPTBox.core.vert_constants import Vertebra_Instance

_log = Print_Logger()

# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _ray_triangle_intersect(
    orig: np.ndarray, direction: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, epsilon: float = 1e-8
) -> float | None:
    """Moeller-Trumbore ray-triangle intersection test.

    Parameters
    ----------
    orig : np.ndarray
        Ray origin (3,).
    direction : np.ndarray
        Unit ray direction (3,).
    v0, v1, v2 : np.ndarray
        Triangle vertices (3,) each.

    Returns:
    -------
    float | None
        Distance ``t`` along ``direction`` to the intersection point
        (only ``t > 0`` is returned, i.e. intersections behind the ray
        origin are ignored), or ``None`` if the ray misses the triangle.
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)
    if -epsilon < a < epsilon:
        return None  # ray is parallel to the triangle
    f = 1.0 / a
    s = orig - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, edge1)
    v = f * np.dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * np.dot(edge2, q)
    return t if t > epsilon else None


def _ray_cast_to_mesh(mesh: Mesh | trimesh.Trimesh, origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    direction = direction / np.linalg.norm(direction)

    if isinstance(mesh, trimesh.Trimesh):
        locations, _, _ = mesh.ray.intersects_location(ray_origins=origin[None], ray_directions=direction[None])
        if len(locations) == 0:
            return None
        # closest hit
        # d = np.linalg.norm(locations - origin, axis=1)
        # return locations[np.argmin(d)]
        d = np.linalg.norm(locations - origin, axis=1)
        return origin + d.mean() * direction
    # numpy-stl fallback
    ts = []
    for v0, v1, v2 in zip(mesh.v0, mesh.v1, mesh.v2):
        t = _ray_triangle_intersect(origin, direction, v0, v1, v2)
        if t is not None:
            ts.append(t)
    if not ts:
        return None
    return origin + np.mean(ts) * direction


def _local_curvature(mesh: Mesh | trimesh.Trimesh, point: np.ndarray, radius: float = 8.0) -> float:
    """Rough curvature estimate (1/mm) of ``mesh`` near ``point``.

    Fits a best-fit plane (via SVD) to all mesh vertices within ``radius``
    of ``point`` and returns the RMS deviation of those vertices from the
    plane, normalized by ``radius**2``. This is a cheap proxy for how
    "bowl-shaped" the surface is locally -- 0 for a flat patch, larger for
    a more curved one. Swap out for principal-curvature-from-quadric-fit
    if you need a more rigorous measure.
    """
    verts = mesh.vertices if isinstance(mesh, trimesh.Trimesh) else np.vstack([mesh.v0, mesh.v1, mesh.v2])
    dists = np.linalg.norm(verts - point, axis=1)
    neighborhood = verts[dists <= radius]
    if len(neighborhood) < 3:
        return 0.0
    centroid = neighborhood.mean(axis=0)
    centered = neighborhood - centroid
    _, _, vt = np.linalg.svd(centered)
    normal = vt[-1]
    deviations = centered @ normal
    rms = float(np.sqrt(np.mean(deviations**2)))
    return rms / (radius**2)


_endplate_curvature_key = {
    Location.Vertebral_Body_Endplate_Superior: "curvature_superior_endplate",
    Location.Vertebral_Body_Endplate_Inferior: "curvature_inferior_endplate",
}
_endplate_angle_key = {
    Location.Vertebral_Body_Endplate_Superior: "angle_superior_endplate",
    Location.Vertebral_Body_Endplate_Inferior: "angle_inferior_endplate",
}


def _endplate(
    poi: POI,
    nii: NII,
    endplate: Location,
    vert_id,
    log: Logger_Interface,
    normals_by_vert,
    cms_local_override: Sequence[float] | None = None,
    flip_direction=False,
    compute_curvature=False,
):
    if nii.max() == 0:
        log.print(f"[calc_endplate_points] no {endplate.name} voxels for vertebra {vert_id}, skipping.")
        return
    bb = nii.compute_crop(0, 1)
    mesh = nii.apply_crop(bb).to_stl(1, to_world=True)
    verts_ = np.vstack([mesh.v0, mesh.v1, mesh.v2])
    cms_local = poi[vert_id, Location.Vertebra_Corpus] if cms_local_override is None else cms_local_override
    cms_global = np.asarray(poi.local_to_global(cms_local), dtype=float)
    # Remove duplicate vertices (optional but recommended)
    verts = np.unique(verts_, axis=0)
    # Center the point cloud
    centroid = verts.mean(axis=0)
    X = verts - centroid
    # PCA via SVD
    _, _, Vt = np.linalg.svd(X, full_matrices=False)

    # Smallest variance direction = plane normal
    direction = Vt[-1]
    to_endplate = centroid - cms_global
    if np.dot(direction, to_endplate) < 0:
        direction *= -1
    direction /= np.linalg.norm(direction)

    casted_point = _ray_cast_to_mesh(mesh, cms_global, direction)
    if casted_point is None:
        faces = np.arange(len(verts_)).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=verts_, faces=faces, process=False)
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        casted_point = _ray_cast_to_mesh(mesh, cms_global, direction)
    if casted_point is None:
        log.print(f"[calc_endplate_points] ray cast missed {endplate.name} mesh for vertebra {vert_id};")
        return
        # Ray missed (e.g. off-axis endplate) -- fall back to the
        # nearest mesh vertex to the centroid.
        verts_all = np.vstack([mesh.v0, mesh.v1, mesh.v2])
        idx = int(np.argmin(np.linalg.norm(verts_all - cms_global, axis=1)))
        casted_point = verts_all[idx]
        log.print(f"[calc_endplate_points] ray cast missed {endplate.name} mesh for vertebra {vert_id}; using nearest mesh vertex instead.")
    local_point = poi.global_to_local(tuple(casted_point))
    poi[vert_id, endplate] = tuple(local_point)

    normal_at_point = casted_point - np.array(cms_global)
    normal_at_point /= np.linalg.norm(normal_at_point)
    if flip_direction:
        normal_at_point *= -1
    normals_by_vert.setdefault((vert_id), {})[endplate] = normal_at_point
    if compute_curvature:
        curvature = _local_curvature(mesh, casted_point)
        poi.info[_endplate_curvature_key[endplate]][Vertebra_Instance(vert_id).name] = curvature

    poi.info[_endplate_angle_key[endplate]][Vertebra_Instance(vert_id).name] = tuple(direction)


# --------------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------------


def calc_endplate_points_(
    poi: POI,
    vert: NII,
    spine: NII,
    _vert_ids: Sequence[int] | None = None,
    compute_curvature=False,
    log: Logger_Interface = _log,
    inplace=True,
) -> tuple[POI, NII, NII]:
    """Estimate superior/inferior vertebral endplate landmark points.

    For every relevant vertebra id, this extracts the superior and
    inferior endplate surface mesh (from ``spine``, restricted to that
    vertebra via ``vert``), ray-casts from the vertebral body centroid
    (``Location.Vertebra_Corpus``) toward each endplate surface, and
    stores the intersection ("casted") point back into ``poi`` under
    ``Location.Vertebral_Body_Endplate_Superior`` /
    ``Location.Vertebral_Body_Endplate_Inferior``.

    It additionally computes and stores, in ``poi.info``:

    - ``"endplate_internal_angle"``: ``{vert_id: angle_degrees}`` -- the
      angle between the superior and inferior endplate surface normals at
      their respective casted points. This is a proxy for local vertebral
      body wedging (0 deg = perfectly parallel endplates).
    - ``"curvature_superior_endplate"``: ``{vert_id: float}`` -- curvature
      proxy (see ``_local_curvature``) of the superior endplate surface
      near its casted point.
    - ``"curvature_inferior_endplate"``: ``{vert_id: float}`` -- same, for
      the inferior endplate.

    Parameters
    ----------
    poi : POI
        Points of interest. Must already contain
        ``Location.Vertebra_Corpus`` for every vertebra to be processed.
        Copied internally; the input is not mutated.
    vert : NII
        Vertebra instance segmentation.
    spine : NII
        Spine sub-structure segmentation, expected to contain
        ``Location.Endplate`` voxels which get restricted to a single
        vertebra via ``vert * spine.extract_label(...) % 100``.
    _vert_ids : list[int], optional
        Restrict processing to these vertebra ids. Defaults to every id
        present in ``vert``. In both cases, ids are filtered down to
        cervical/thoracic/lumbar vertebrae only.
    log : Print_Logger, optional
        Logger used for warnings (e.g. missing endplate voxels, missed
        ray casts).

    Returns:
    -------
    tuple[POI, NII, NII]
        The updated ``poi`` (endplate points + ``poi.info`` entries
        above), and the original ``vert``/``spine`` passed through
        unchanged.
    """
    if _vert_ids is None:
        vert_ids = vert.unique()
    else:
        vert_ids: list[int] = list(_vert_ids)
    if not inplace:
        poi = poi.copy()
    _spine_ids = [a.value for a in Vertebra_Instance.cervical()[1:] + Vertebra_Instance.thoracic() + Vertebra_Instance.lumbar()]
    vert_ids_org = vert_ids
    vert_ids = [a for a in vert_ids if a in _spine_ids]
    sp_u = spine.unique()
    if Location.Endplate.value in sp_u:
        vert, spine = endplate_to_super_infer_endplate(vert, spine)
        sp_u = spine.unique()
    if not any(a in sp_u for a in [Location.Vertebral_Body_Endplate_Superior.value, Location.Vertebral_Body_Endplate_Inferior.value]):
        log.print(f"[calc_endplate_points] No endplates, {sp_u}")
        return (poi, vert, spine)
    poi.info.setdefault("endplate_internal_angle", {})

    poi.info.setdefault("angle_superior_endplate", {})
    poi.info.setdefault("angle_inferior_endplate", {})

    if compute_curvature:
        poi.info.setdefault("curvature_superior_endplate", {})
        poi.info.setdefault("curvature_inferior_endplate", {})

    # Collect normals per vertebra so we can compute the inter-endplate
    # angle once both superior and inferior have been processed.
    normals_by_vert: dict[int, dict[Location, np.ndarray]] = {}
    for endplate in (Location.Vertebral_Body_Endplate_Superior, Location.Vertebral_Body_Endplate_Inferior):
        endplate_nii = vert * spine.extract_label(endplate) % 100

        for vert_id in vert_ids:
            if vert_id == 2 and Location.Vertebral_Body_Endplate_Superior == endplate:
                continue
            nii = endplate_nii.extract_label(vert_id)
            _endplate(poi, nii, endplate, vert_id, log, normals_by_vert, compute_curvature=compute_curvature)

    # Location.Sacrum_Endplate,
    endplate_nii = spine.extract_label(Location.Sacrum_Endplate)  # vert *
    endplate_nii = endplate_nii.apply_crop(endplate_nii.compute_crop(0, 2))
    if endplate_nii.max() > 0:
        c = endplate_nii.dilate_msk(2).get_connected_components()
        if c.max() != 1:
            coms = c.reorient().center_of_masses()  # {int: (P,I,R)}
            superior_label = min(coms, key=lambda lbl: coms[lbl][1])
            endplate_nii = c.extract_label(superior_label)
        cms_local_override = None

        last_vert = max(vert_ids)

        if (last_vert, Location.Vertebral_Body_Endplate_Inferior.value) in poi:
            cms_local_override = poi[last_vert, Location.Vertebral_Body_Endplate_Inferior]
        elif (last_vert, Location.Vertebra_Disc.value) in poi:
            cms_local_override = poi[last_vert, Location.Vertebra_Disc.value]
        elif 100 + last_vert in vert_ids_org:
            cms_local_override = vert.extract_label(100 + last_vert).center_of_masses()[1]
        else:
            cms_local_override = poi[last_vert, Location.Vertebra_Corpus]
        _endplate(
            poi,
            endplate_nii,
            Location.Vertebral_Body_Endplate_Superior,
            Vertebra_Instance.S1.value,
            log,
            normals_by_vert,
            cms_local_override=cms_local_override,
            flip_direction=True,
            compute_curvature=compute_curvature,
        )
    # Angle between superior and inferior endplate normals, per vertebra.
    for vert_id, normals in normals_by_vert.items():
        n_sup = normals.get(Location.Vertebral_Body_Endplate_Superior)
        n_inf = normals.get(Location.Vertebral_Body_Endplate_Inferior)
        if n_sup is None or n_inf is None:
            continue  # one side missing (e.g. no voxels) -- skip this vertebra
        cos_angle = float(np.clip(np.dot(n_sup, n_inf), -1.0, 1.0))
        poi.info["endplate_internal_angle"][Vertebra_Instance(vert_id).name] = 180 - float(np.degrees(np.arccos(cos_angle)))

    return poi, vert, spine


def endplate_to_super_infer_endplate(vert: NII, spine: NII) -> tuple[NII, NII]:
    """Split a combined ``Location.Endplate`` label into superior/inferior sub-labels.

    Endplate segmentations (spineps T2w segmentation) are produced adjacent to the
    intervertebral disc without indicating which vertebra they belong to
    or whether they sit on that vertebra's superior or inferior surface.
    This function resolves both: it grows each vertebra's corpus label
    outward (via ``NII.infect``) into the surrounding endplate/disc/
    corpus-border region so every endplate voxel is claimed by its
    nearest vertebra, then reclassifies ``spine`` so those voxels carry
    ``Location.Vertebral_Body_Endplate_Superior`` or
    ``Location.Vertebral_Body_Endplate_Inferior`` instead of the generic
    ``Location.Endplate`` label, and offsets the corresponding ``vert``
    instance ids by ``+200`` to mark them as reassigned.

    Parameters
    ----------
    vert : NII
        Vertebra instance segmentation.
    spine : NII
        Spine sub-structure segmentation containing a combined
        ``Location.Endplate`` label.

    Returns:
    -------
    tuple[NII, NII]
        The same ``vert``/``spine`` objects, updated. If ``spine``
        contains no ``Location.Endplate`` voxels, both are returned
        unchanged (no-op).

    """
    endplate_nii = spine.extract_label(Location.Endplate)
    if endplate_nii.sum() == 0:
        return vert, spine
    spine = spine.copy()
    vert_org = vert.copy()
    vert[vert >= 40] = 0
    vert[spine.extract_label([Location.Vertebra_Corpus, Location.Vertebra_Corpus_border]) != 1] = 0
    vert %= 100
    v = vert.infect(
        spine.extract_label([Location.Vertebra_Corpus, Location.Vertebra_Corpus_border, Location.Endplate, Location.Vertebra_Disc]),
        verbose=False,
    )
    endplate_nii = v * endplate_nii
    spine[np.logical_and(endplate_nii == vert_org % 100, endplate_nii != 0)] = Location.Vertebral_Body_Endplate_Inferior.value
    spine[spine == Location.Endplate.value] = Location.Vertebral_Body_Endplate_Superior.value
    vert_org[endplate_nii != 0] = v[endplate_nii != 0] + 200
    return vert_org, spine


if __name__ == "__main__":
    from pathlib import Path

    from TPTBox import calc_poi_from_subreg_vert, to_nii

    p = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-myelom/derivatives-final/sub-CTFU00065/ses-00000")
    poi = calc_poi_from_subreg_vert(
        p / "sub-CTFU00065_ses-00000_sequ-2_mod-ct_seg-vert_msk.nii.gz",
        p / "sub-CTFU00065_ses-00000_sequ-2_mod-ct_seg-spine_msk.nii.gz",
        subreg_id=Location.Endplate,
    )
    poi.make_point_cloud_nii(s=3)[1].save(p / "out_point.nii.gz")
    poi.save(p / "out.json")
    print(poi.centroids)
    # p = Path("TPTBox/tests/sample_mri")
    #
    ## vert, spine = endplate_to_super_infer_endplate(
    ##    to_nii(p / "sub-mri_seg-vert_label-6_msk.nii.gz", True),
    ##    to_nii(p / "sub-mri_seg-subreg_label-6_msk.nii.gz", True),
    ## )
    ## vert.save(p / "out_v.nii.gz")
    ## spine.save(p / "out_s.nii.gz")
    # poi = calc_poi_from_subreg_vert(
    #    p / "sub-mri_seg-vert_label-6_msk.nii.gz",
    #    p / "sub-mri_seg-subreg_label-6_msk.nii.gz",
    #    subreg_id=Location.Endplate,
    # )
    ## poi.make_point_cloud_nii(s=3)[1].save(p / "out_point.nii.gz")
    ## poi.save(p / "out.json")
    # print(poi.centroids)
