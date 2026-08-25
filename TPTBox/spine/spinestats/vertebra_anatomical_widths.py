from __future__ import annotations

from TPTBox import POI, Image_Reference, Location, calc_poi_from_subreg_vert


def _compute_distance(
    poi: POI,
    l1: Location,
    l2: Location,
    key: str,
    vert: Image_Reference | None = None,
    subreg: Image_Reference | None = None,
    all_pois_computed: bool = False,
    recompute: bool = False,
) -> POI:
    """Compute pairwise distances between two anatomical locations across vertebral regions.

    The distances are stored in ``poi.info[key]`` as a dictionary mapping
    vertebra region IDs to distances in mm.  If the key already exists and
    ``recompute`` is ``False`` the function returns ``poi`` unchanged.

    Args:
        poi: POI object containing (or to be extended with) anatomical
            landmark coordinates.
        l1: First anatomical location (e.g. superior disc face).
        l2: Second anatomical location (e.g. inferior disc face).
        key: Name under which the computed distances are stored in
            ``poi.info``.
        vert: Vertebra segmentation image used to compute missing POIs.
            Required when ``all_pois_computed=False`` and the locations are
            absent from ``poi``.
        subreg: Subregion segmentation image used together with ``vert`` for
            POI computation.
        all_pois_computed: Skip POI computation and use existing centroids
            directly.  Requires ``l1`` and ``l2`` to already be present in
            ``poi``.
        recompute: When ``True`` overwrite an existing entry in ``poi.info``.

    Returns:
        The updated :class:`~TPTBox.POI` object with distances stored under
        ``poi.info[key]``.
    """
    if key in poi.info and not recompute:
        return poi
    if not all_pois_computed:
        if vert is None or (subreg is None and l1.value not in poi.keys_region() and l2.value in poi.keys_region()):
            raise ValueError(f"{vert=} and {subreg=} must be set or precomputed all pois; {all_pois_computed=} -- {poi.keys_region()=}")
    else:
        all_pois_computed = True
    if not all_pois_computed:
        poi = calc_poi_from_subreg_vert(vert, subreg, extend_to=poi, subreg_id=[l1, l2])
    poi.info[key] = poi.calculate_distances_poi_across_regions(l1, l2, keep_zoom=False)
    return poi


distances_funs: dict[str, tuple[Location, Location]] = {
    "ivd_heights_center_mm": (
        Location.Vertebra_Disc_Inferior,
        Location.Vertebra_Disc_Superior,
    ),
    "vertebra_heights_center_mm": (
        Location.Additional_Vertebral_Body_Middle_Superior_Median,
        Location.Additional_Vertebral_Body_Middle_Inferior_Median,
    ),
    "vertebra_width_LR_center_mm": (
        Location.Muscle_Inserts_Vertebral_Body_Right,
        Location.Muscle_Inserts_Vertebral_Body_Left,
    ),
    "vertebra_width_AP_center_mm": (
        Location.Additional_Vertebral_Body_Posterior_Central_Median,
        Location.Additional_Vertebral_Body_Anterior_Central_Median,
    ),
}


def compute_all_distances(
    poi: POI,
    vert: Image_Reference | None = None,
    subreg: Image_Reference | None = None,
    all_pois_computed: bool = False,
    recompute: bool = False,
) -> POI:
    """Compute all registered anatomical distances and store them in ``poi.info``.

    Iterates over :data:`distances_funs` and calls :func:`_compute_distance`
    for each entry.  Results are accumulated in the returned POI object under
    the corresponding key in ``poi.info``.

    Args:
        poi: POI object to compute distances for and to store results in.
        vert: Vertebra segmentation image required when POIs have not been
            pre-computed.  May be ``None`` when ``all_pois_computed=True``.
        subreg: Subregion segmentation image required when POIs have not been
            pre-computed.  May be ``None`` when ``all_pois_computed=True``.
        all_pois_computed: When ``True`` the function skips POI computation
            and uses the existing centroids in ``poi`` directly.
        recompute: When ``True`` existing entries in ``poi.info`` are
            overwritten; otherwise already-computed keys are skipped.

    Returns:
        The updated :class:`~TPTBox.POI` object with distance dictionaries
        stored under the keys defined in :data:`distances_funs`.
    """
    for key, (l1, l2) in distances_funs.items():
        poi = _compute_distance(poi, l1, l2, key, vert, subreg, all_pois_computed, recompute)
    return poi
