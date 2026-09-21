# POI Strategies

Internal modules that implement the various strategies for computing Points of Interest
from segmentation volumes.

## Ray Casting

::: TPTBox.core.poi_fun.ray_casting
    options:
      show_source: true
      filters: ["!^_"]

## Vertebra Direction

::: TPTBox.core.poi_fun.vertebra_direction
    options:
      show_source: true
      filters: ["!^_"]

## Vertebra Non-Centroid POIs

::: TPTBox.core.poi_fun.vertebra_pois_non_centroids
    options:
      show_source: true
      filters: ["!^_"]

## Pixel-Based Point Finder

::: TPTBox.core.poi_fun.pixel_based_point_finder
    options:
      show_source: true
      filters: ["!^_"]

## Strategies

::: TPTBox.core.poi_fun.strategies
    options:
      show_source: true
      filters: ["!^_"]

## Save / Load

::: TPTBox.core.poi_fun.save_load
    options:
      show_source: true
      filters: ["!^_"]

## Direction-Vector Fields in `POI.info`

`POI.info` can hold *auxiliary* per-vertebra data alongside the main POI points —
direction vectors (e.g. endplate PCA normals) and scalar metadata (e.g. per-vertebra
wedge angles or curvature). Producers register field names under two well-known
keys in `poi.info`, and `POI.reorient` / `POI.resample_from_to` /
`POI_Global.to_cord_system` / `POI.map_labels` then keep those fields aligned
with the POI points automatically.

**Two registries:**

- `poi.info["_vector_fields"]` (`POI_INFO_VECTOR_FIELDS_KEY`) — a list of field
  names whose values are `{key: (x, y, z)}` dicts holding **unit direction
  vectors in mm-space aligned with `poi.orientation`**. These fields are
  auto-transformed by `reorient`, `resample_from_to`, and
  `POI_Global.to_cord_system`, and their keys are also remapped by `map_labels`.
  `rescale` is a no-op for them.
- `poi.info["_label_keyed_fields"]` (`POI_INFO_LABEL_KEYED_FIELDS_KEY`) — a
  list of field names whose values are `{key: <anything>}` dicts (typically
  scalars). Only their **keys** are remapped by `map_labels`; no
  orientation-based transform is applied.

Keys in either type of field may be integer region labels (`20`) or
`Vertebra_Instance`-name strings (`"L1"`); both forms are supported.

**Producer pattern:**

```python
from TPTBox.core.poi_fun.vector_fields import (
    POI_INFO_VECTOR_FIELDS_KEY,
    POI_INFO_LABEL_KEYED_FIELDS_KEY,
)

# Direction vectors (auto-rotated on reorient / resample):
vec_fields = poi.info.setdefault(POI_INFO_VECTOR_FIELDS_KEY, [])
if "my_vector_field" not in vec_fields:
    vec_fields.append("my_vector_field")
poi.info["my_vector_field"] = {"L1": (0.09, -0.99, -0.02), ...}

# Scalar per-vertebra metadata (key-remapped by map_labels only):
lbl_fields = poi.info.setdefault(POI_INFO_LABEL_KEYED_FIELDS_KEY, [])
if "my_scalar_field" not in lbl_fields:
    lbl_fields.append("my_scalar_field")
poi.info["my_scalar_field"] = {"L1": 3.14, ...}
```

**Caveats:**

- `resample_from_to` uses the real `R_ref.T @ R_self` rotation of the two
  affines (safe for skewed grids); axcode-only heuristics are deliberately
  avoided.
- `map_labels` handles duplicates with a *last-write-wins* policy after remap.
- `map_labels`' `label_map_full` (mapping `(region, subreg)` tuples) does NOT
  trigger the key-remap of these fields — only `label_map_region` does.

::: TPTBox.core.poi_fun.vector_fields
    options:
      show_source: true
      filters: ["!^_"]
      members:
        - POI_INFO_VECTOR_FIELDS_KEY
        - POI_INFO_LABEL_KEYED_FIELDS_KEY
