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


def _map_one_key(k, key_map: dict | None):
    """Map a single dict key via ``key_map``.

    ``int`` keys are looked up directly. ``str`` keys resolve through
    ``Vertebra_Instance`` (name -> value) and, on hit, the mapped value is
    turned back into the corresponding ``Vertebra_Instance`` name (or kept
    as ``str(id)`` when the target isn't a known instance).
    """
    if not key_map:
        return k
    if isinstance(k, int) and k in key_map:
        return key_map[k]
    if isinstance(k, str):
        try:
            label = Vertebra_Instance[k].value
        except KeyError:
            return k
        if label in key_map:
            try:
                return Vertebra_Instance(key_map[label]).name
            except ValueError:
                return str(key_map[label])
    return k


def _remap_vector_field_keys_inplace(info: dict, region_map: dict | None, subregion_map: dict | None = None) -> None:
    """Remap keys of every registered label-keyed field in ``info``.

    Considers fields registered under both :data:`POI_INFO_VECTOR_FIELDS_KEY`
    (direction vectors) and :data:`POI_INFO_LABEL_KEYED_FIELDS_KEY` (scalar
    per-label fields), plus ``label_name`` (always handled; migrated to the
    nested form on the fly).

    Behaviour is dispatched by *value type*:

    - **Flat fields** (values are tuples / scalars): only outer keys are
      remapped via ``region_map``.
    - **Nested fields** (values are dicts, e.g. ``label_name`` /
      ``{region: {subregion: name, "name": group}}``): outer keys are
      remapped via ``region_map``, inner keys via ``subregion_map``, and the
      special ``"name"`` group entry is preserved. When two source regions
      collide onto one target, their inner dicts merge (last-write-wins on
      overlapping keys).

    Keys may be integer labels or ``Vertebra_Instance``-name strings; both
    are matched against the int-keyed maps. No-op if there is nothing to do.
    """
    from TPTBox.core.poi_fun.poi_abstract import _GROUP_NAME_KEY, LABEL_NAME, label_name_dict

    if not region_map and not subregion_map:
        return
    field_names: list[str] = []
    for key in (POI_INFO_VECTOR_FIELDS_KEY, POI_INFO_LABEL_KEYED_FIELDS_KEY):
        names = info.get(key)
        if names:
            field_names.extend(names)
    # label_name is always handled -- ensure the nested-form migration runs, then include it.
    if LABEL_NAME in info and LABEL_NAME not in field_names:
        label_name_dict(info)
        field_names.append(LABEL_NAME)
    if not field_names:
        return
    for name in field_names:
        vectors = info.get(name)
        if not isinstance(vectors, dict):
            continue
        remapped: dict = {}
        for k, v in vectors.items():
            new_k = _map_one_key(k, region_map)
            if new_k is None:
                continue  # drop entries whose region is mapped to None
            new_v = v
            # Nested field: recurse into inner dict.
            if isinstance(v, dict):
                new_inner: dict = {}
                for ik, iv in v.items():
                    if ik == _GROUP_NAME_KEY:
                        new_inner[_GROUP_NAME_KEY] = iv
                        continue
                    new_ik = _map_one_key(ik, subregion_map)
                    if new_ik is None:
                        continue  # drop entries whose subregion is mapped to None
                    new_inner[new_ik] = iv
                new_v = new_inner
            # Merge on outer-key collision when both values are dicts (label_name-style).
            if new_k in remapped and isinstance(remapped[new_k], dict) and isinstance(new_v, dict):
                remapped[new_k].update(new_v)
            else:
                remapped[new_k] = new_v
        vectors.clear()
        vectors.update(remapped)


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
