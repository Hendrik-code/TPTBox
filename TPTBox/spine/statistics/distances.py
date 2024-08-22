from TPTBox import POI, Image_Reference, Location, calc_poi_from_subreg_vert


def _compute_distance(
    poi: POI,
    l1: Location,
    l2: Location,
    key: str,
    vert: Image_Reference | None = None,
    subreg: Image_Reference | None = None,
    all_pois_computed=False,
    recompute=False,
):
    """Compute the IVD height with a single point. Returns poi-object with computed points poi heights in
    poi.info["ivd_heights_center_mm"]
    all_pois_computed requires [Location.Vertebra_Direction_Inferior,  Location.Vertebra_Disc_Superior] to be computed
    recompute skips if poi.info["ivd_heights_center_mm"] exist
    Args:
        vert (Image_Reference): _description_
        subreg (Image_Reference): _description_
        poi (POI): _description_
    """
    if key in poi.info and not recompute:
        return poi
    if not all_pois_computed:
        if (
            vert is None
            or subreg is None
            and l1.value not in poi.keys_region()
            and l2.value in poi.keys_region()
        ):
            raise ValueError(
                f"{vert=} and {subreg=} must be set or precomputed all pois; {all_pois_computed=} -- {poi.keys_region()=}"
            )
    else:
        all_pois_computed = True
    if not all_pois_computed:
        poi = calc_poi_from_subreg_vert(vert, subreg, extend_to=poi, subreg_id=[l1, l2])
    poi.info[key] = poi.calculate_distances_poi_two_locations(l1, l2, keep_zoom=False)
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
    all_pois_computed=False,
    recompute=False,
):
    for key, (l1, l2) in distances_funs.items():
        poi = _compute_distance(
            poi, l1, l2, key, vert, subreg, all_pois_computed, recompute
        )
    return poi
