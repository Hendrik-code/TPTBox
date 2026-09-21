"""Helpers for auto-transforming direction-vector fields stored in ``POI.info``.

Producers write direction vectors into ``poi.info["<field_name>"]`` as
``{key: (x, y, z)}`` dicts and register ``<field_name>`` in
``poi.info[POI_INFO_VECTOR_FIELDS_KEY]``. The helpers here are then called by
``POI.reorient`` / ``POI.resample_from_to`` / ``POI_Global.to_cord_system`` /
``POI.map_labels`` to keep the vectors and their keys aligned with the POI
points as the POI is transformed.

Vectors are assumed to live in mm-space aligned with the POI's current voxel
axes; ``rescale`` is therefore a no-op for them.
"""

from __future__ import annotations

import numpy as np

from TPTBox.core.vert_constants import Vertebra_Instance

# poi.info key naming the info dicts whose values are direction vectors
# in the POI's current axis frame (i.e. mm-space aligned with poi.orientation).
POI_INFO_VECTOR_FIELDS_KEY = "_vector_fields"

# poi.info key naming *additional* info dicts (typically scalar-valued, e.g.
# per-vertebra angles or curvature) whose top-level keys are region labels or
# ``Vertebra_Instance`` names. They participate in :func:`map_labels` key
# remapping but not in reorient / resample / to_cord_system vector transforms.
POI_INFO_LABEL_KEYED_FIELDS_KEY = "_label_keyed_fields"


def _transform_direction_vectors_inplace(info: dict, trans: np.ndarray) -> None:
    """Reorient every registered direction-vector field in ``info`` according to ``trans``.

    ``trans`` is the output of ``nibabel.orientations.ornt_transform`` applied to
    the POI's source and target axcodes. Each value stored under a registered
    field must be a 3-tuple / 3-list (or ``None`` / non-3-tuple, which is skipped)
    interpreted as a unit direction in the POI's current mm-space voxel-axis
    frame. Values are updated in place.
    """
    field_names = info.get(POI_INFO_VECTOR_FIELDS_KEY)
    if not field_names:
        return
    perm = np.asarray(trans[:, 0], dtype=int)
    flip = np.asarray(trans[:, 1], dtype=int)
    for name in field_names:
        vectors = info.get(name)
        if not isinstance(vectors, dict):
            continue
        for k, v in list(vectors.items()):
            if v is None:
                continue
            try:
                v_arr = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                continue
            if v_arr.shape != (3,):
                continue
            new_v = np.zeros(3, dtype=float)
            new_v[perm] = v_arr * flip
            vectors[k] = tuple(float(x) for x in new_v)


def _remap_vector_field_keys_inplace(info: dict, region_map: dict) -> None:
    """Remap the top-level keys of every registered label-keyed field via ``region_map``.

    Considers fields registered under both :data:`POI_INFO_VECTOR_FIELDS_KEY`
    (direction vectors) and :data:`POI_INFO_LABEL_KEYED_FIELDS_KEY` (scalar
    per-label fields like ``endplate_internal_angle`` or ``curvature_*``).
    Keys may be either integer region labels or ``Vertebra_Instance``-name
    strings ("L1", "T12", ...); both are matched against ``region_map``
    (int-keyed). Duplicates after remapping keep the last write. No-op if
    no fields are registered or ``region_map`` is empty.
    """
    if not region_map:
        return
    field_names: list[str] = []
    for key in (POI_INFO_VECTOR_FIELDS_KEY, POI_INFO_LABEL_KEYED_FIELDS_KEY):
        names = info.get(key)
        if names:
            field_names.extend(names)
    if not field_names:
        return
    for name in field_names:
        vectors = info.get(name)
        if not isinstance(vectors, dict):
            continue
        remapped = {}
        for k, v in vectors.items():
            new_k = k
            if isinstance(k, int) and k in region_map:
                new_k = region_map[k]
            elif isinstance(k, str):
                try:
                    label = Vertebra_Instance[k].value
                except KeyError:
                    label = None
                if label is not None and label in region_map:
                    try:
                        new_k = Vertebra_Instance(region_map[label]).name
                    except ValueError:
                        new_k = k
            remapped[new_k] = v
        vectors.clear()
        vectors.update(remapped)


def _remap_label_name_inplace(info: dict, region_map: dict | None, subregion_map: dict | None) -> None:
    """Remap the region + subregion keys of ``info["label_name"]`` in place.

    ``label_name`` uses the nested format
    ``{region:int -> {subregion:int -> name:str, "name": group_name:str}}``
    (see :func:`normalize_label_name`). This helper remaps top-level region keys
    via ``region_map`` and, for each inner dict, remaps subregion keys via
    ``subregion_map``. The special ``"name"`` group-name entry is preserved.
    No-op if the field is absent or both maps are empty.
    """
    from TPTBox.core.poi_fun.poi_abstract import LABEL_NAME, _GROUP_NAME_KEY, label_name_dict

    if not region_map and not subregion_map:
        return
    if LABEL_NAME not in info:
        return
    ln = label_name_dict(info)  # ensures nested form
    remapped: dict[int, dict] = {}
    for region, inner in ln.items():
        new_region = region_map[region] if region_map and region in region_map else region
        new_inner: dict = {}
        for k, v in inner.items():
            if k == _GROUP_NAME_KEY:
                new_inner[_GROUP_NAME_KEY] = v
            elif subregion_map and k in subregion_map:
                new_inner[subregion_map[k]] = v
            else:
                new_inner[k] = v
        # merge if two source regions collide onto one target (last-write-wins on inner keys).
        if new_region in remapped:
            remapped[new_region].update(new_inner)
        else:
            remapped[new_region] = new_inner
    info[LABEL_NAME] = remapped


def _rotate_direction_vectors_inplace(info: dict, src_rot, tgt_rot) -> None:
    """Rotate registered direction-vector fields from ``src_rot`` to ``tgt_rot``.

    Composes ``tgt_rot.T @ src_rot`` and applies it in place. No-op when either
    rotation is None or no vector fields are registered.
    """
    if src_rot is None or tgt_rot is None:
        return
    field_names = info.get(POI_INFO_VECTOR_FIELDS_KEY)
    if not field_names:
        return
    R = np.asarray(tgt_rot, dtype=float).T @ np.asarray(src_rot, dtype=float)
    for name in field_names:
        vectors = info.get(name)
        if not isinstance(vectors, dict):
            continue
        for k, v in list(vectors.items()):
            if v is None:
                continue
            try:
                v_arr = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                continue
            if v_arr.shape != (3,):
                continue
            new_v = R @ v_arr
            vectors[k] = tuple(float(x) for x in new_v)
