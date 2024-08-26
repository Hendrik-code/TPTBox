import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

from TPTBox import NII, POI, Logger_Interface, Print_Logger, Vertebra_Instance, calc_poi_from_subreg_vert
from TPTBox.core.poi_fun.strategies import (
    strategy_extreme_points,
    strategy_find_corner,
    strategy_ligament_attachment_point_flava,
    strategy_line_cast,
    strategy_shifted_line_cast,
)
from TPTBox.core.poi_fun.vertebra_direction import calc_center_spinal_cord, calc_orientation_of_vertebra_PIR
from TPTBox.core.vert_constants import Location, vert_directions
from TPTBox.spine.statistics import calculate_IVD_POI

_log = Print_Logger()
all_poi_functions: dict[int, "Strategy_Pattern"] = {}
pois_computed_by_side_effect: dict[int, Location] = {}
_sacrum = set(Vertebra_Instance.sacrum())

ivd_pois_list = (Location.Vertebra_Disc_Inferior, Location.Vertebra_Disc_Superior, Location.Vertebra_Disc)


def run_poi_pipeline(vert: NII, subreg: NII, poi_path: Path, logger: Logger_Interface = _log):
    poi = calc_poi_from_subreg_vert(vert, subreg, buffer_file=poi_path, save_buffer_file=True, subreg_id=list(Location), verbose=logger)
    poi.save(poi_path)


def _strategy_side_effect(*args, **qargs):  # noqa: ARG001
    pass


def add_prerequisites(locs: Sequence[Location]):
    addendum = set()
    locs2 = set(locs)
    loop_var = locs2
    i = 0
    while i != 1000:  # Prevent Deadlock
        for l in loop_var:
            if l.value in all_poi_functions:
                for prereq in all_poi_functions[l.value].prerequisite:
                    if prereq not in locs:
                        addendum.add(prereq)
        if len(addendum) == 0:
            break
        locs2 = addendum | locs2
        loop_var = addendum
        addendum = set()
        i += 1
    else:
        warnings.warn("Deadlock in add_prerequisites", stacklevel=10)
    return sorted(list(locs2), key=lambda x: x.value)  # type: ignore # noqa: C414


class Strategy_Pattern:
    """Implements the Strategy design pattern by encapsulating different strategies as callable objects.

    Args:
        target (Location): The target location for which this strategy is defined.
        strategy (Callable): The strategy function that implements the desired behavior.
        prerequisite (set[Location] | None, optional): A set of prerequisite locations that must be satisfied before applying this strategy. Defaults to None.
        **args: Additional keyword arguments to be passed to the strategy function.

    Attributes:
        target (Location): The target location for which this strategy is defined.
        args (dict): Additional keyword arguments to be passed to the strategy function.
        prerequisite (set[Location]): A set of prerequisite locations that must be satisfied before applying this strategy.
        strategy (Callable): The strategy function that implements the desired behavior.

    Note:
        The strategy function should accept the following arguments:
        - poi (POI): The point of interest.
        - current_subreg (NII): The current subregion.
        - vert_id (int): The vertex ID.
        - bb: The bounding box.
        - log (Logger_Interface, optional): The logger interface. Defaults to _log, which should be defined globally.

    Example:
        >>> def strategy_function(poi, current_subreg, location, log, vert_id, bb, **kwargs):
        ...     # Strategy implementation
        ...     pass
        >>> strategy = Strategy_Pattern(target_location, strategy_function, prerequisite={prerequisite_location}, additional_arg=value)
        >>> result = strategy(poi, current_subreg, vert_id, bb)
    """

    def __init__(
        self,
        target: Location,
        strategy: Callable,
        prerequisite: set[Location] | None = None,
        prio=0,
        sakrum=False,
        **args,
    ) -> None:
        self.target = target
        self.args = args
        self.sacrum = sakrum
        if prerequisite is None:
            prerequisite = set()
        if "direction" in args.keys():
            prerequisite.add(Location.Vertebra_Direction_Inferior)
        for i in args.values():
            if isinstance(i, Location):
                prerequisite.add(i)
            elif isinstance(i, Sequence):
                for j in i:
                    if isinstance(j, Location):
                        prerequisite.add(j)
        self.prerequisite = prerequisite
        self.strategy = strategy
        all_poi_functions[target.value] = self
        self._prio = prio

    def __call__(
        self,
        poi: POI,
        current_subreg: NII,
        vert_id: int,
        bb,
        log: Logger_Interface = _log,
    ):
        try:
            return self.strategy(
                poi=poi,
                current_subreg=current_subreg,
                location=self.target,
                log=log,
                vert_id=vert_id,
                bb=bb,
                **self.args,
            )
        except Exception:
            _log.print_error()
            return None

    def prority(self):
        return self.target.value + self._prio


class Strategy_Pattern_Side_Effect(Strategy_Pattern):
    def __init__(self, target: Location, prerequisite: Location, **args) -> None:
        super().__init__(target, _strategy_side_effect, {prerequisite}, **args)
        pois_computed_by_side_effect[target.value] = prerequisite


class Strategy_Computed_Before(Strategy_Pattern):
    def __init__(self, target: Location, *prerequisite: Location, **args) -> None:
        super().__init__(target, _strategy_side_effect, set(prerequisite), **args)


##### Add all Strategy to the strategy list #####
# fmt: off
L = Location
Strategy_Pattern_Side_Effect(L.Vertebra_Direction_Posterior,L.Vertebra_Direction_Inferior)
Strategy_Pattern_Side_Effect(L.Vertebra_Direction_Right,L.Vertebra_Direction_Inferior)
Strategy_Pattern_Side_Effect(L.Vertebra_Direction_Inferior,L.Vertebra_Corpus)
S = strategy_extreme_points

Strategy_Pattern(L.Muscle_Inserts_Spinosus_Process, strategy=S, subreg_id=L.Spinosus_Process, direction=(("P",.2),"I"))  # 81
Strategy_Pattern(L.Muscle_Inserts_Transverse_Process_Right, strategy=S, subreg_id=L.Costal_Process_Right, direction=("P","R"))  # 82
Strategy_Pattern(L.Muscle_Inserts_Transverse_Process_Left, strategy=S, subreg_id=L.Costal_Process_Left, direction=("P","L"))  # 83
Strategy_Pattern(L.Muscle_Inserts_Articulate_Process_Inferior_Left, strategy=S, subreg_id=L.Inferior_Articular_Left, direction=("I")) # 86
Strategy_Pattern(L.Muscle_Inserts_Articulate_Process_Inferior_Right, strategy=S, subreg_id=L.Inferior_Articular_Right, direction=("I")) # 87
Strategy_Pattern(L.Muscle_Inserts_Articulate_Process_Superior_Left, strategy=S, subreg_id=L.Superior_Articular_Left, direction=("S")) # 88
Strategy_Pattern(L.Muscle_Inserts_Articulate_Process_Superior_Right, strategy=S, subreg_id=L.Superior_Articular_Right, direction=("S")) # 89
#Strategy_Pattern(L.Vertebra_Disc_Post, strategy=S, subreg_id=L.Vertebra_Disc, direction=("P"))

S = strategy_line_cast
args = {"strategy": S, "regions_loc": [L.Vertebra_Corpus, L.Vertebra_Corpus_border],"start_point": L.Vertebra_Corpus, }
Strategy_Pattern(L.Muscle_Inserts_Vertebral_Body_Right, **args, normal_vector_points ="R" ) # 84
Strategy_Pattern(L.Muscle_Inserts_Vertebral_Body_Left,  **args, normal_vector_points ="L" ) # 85
Strategy_Pattern(L.Additional_Vertebral_Body_Middle_Superior_Median,   **args, normal_vector_points ="S" ) # 105
Strategy_Pattern(L.Additional_Vertebral_Body_Posterior_Central_Median, **args, normal_vector_points ="P" ) # 106
Strategy_Pattern(L.Additional_Vertebral_Body_Middle_Inferior_Median,   **args, normal_vector_points ="I" ) # 107
Strategy_Pattern(L.Additional_Vertebral_Body_Anterior_Central_Median,  **args, normal_vector_points ="A" ) # 108
S = strategy_shifted_line_cast
_pre = {L.Costal_Process_Left,L.Costal_Process_Right,L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left,L.Vertebra_Direction_Inferior}
args = {"strategy": S, "regions_loc": [L.Vertebra_Corpus, L.Vertebra_Corpus_border],"start_point": L.Vertebra_Corpus,"prerequisite":_pre}
Strategy_Pattern(L.Additional_Vertebral_Body_Middle_Superior_Right,  direction="R", normal_vector_points = "S", **args) # 121
Strategy_Pattern(L.Additional_Vertebral_Body_Posterior_Central_Right,direction="R", normal_vector_points = "P", **args) # 122
Strategy_Pattern(L.Additional_Vertebral_Body_Middle_Inferior_Right,  direction="R", normal_vector_points = "I", **args) # 123
Strategy_Pattern(L.Additional_Vertebral_Body_Anterior_Central_Right, direction="R", normal_vector_points = "A", **args) # 124

Strategy_Pattern(L.Additional_Vertebral_Body_Middle_Superior_Left,   direction="L", normal_vector_points = "S", **args) # 113
Strategy_Pattern(L.Additional_Vertebral_Body_Posterior_Central_Left, direction="L", normal_vector_points = "P", **args) # 114
Strategy_Pattern(L.Additional_Vertebral_Body_Middle_Inferior_Left,   direction="L", normal_vector_points = "I", **args) # 115
Strategy_Pattern(L.Additional_Vertebral_Body_Anterior_Central_Left,  direction="L", normal_vector_points = "A", **args) # 116
S = strategy_find_corner
Strategy_Pattern(L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Anterior_Central_Median,vec2 = L.Additional_Vertebral_Body_Middle_Superior_Median,prio=150) #101

Strategy_Pattern(L.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Median,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Posterior_Central_Median,vec2 = L.Additional_Vertebral_Body_Middle_Superior_Median,prio=150) #102

Strategy_Pattern(L.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Median,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Anterior_Central_Median,vec2 = L.Additional_Vertebral_Body_Middle_Inferior_Median,prio=150) #103

Strategy_Pattern(L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Median,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Posterior_Central_Median,vec2 = L.Additional_Vertebral_Body_Middle_Inferior_Median,prio=150) #104

Strategy_Pattern(L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Anterior_Central_Right,vec2 = L.Additional_Vertebral_Body_Middle_Superior_Right,prio=100,shift_direction="R") #117

Strategy_Pattern(L.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Right,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Posterior_Central_Right,vec2 = L.Additional_Vertebral_Body_Middle_Superior_Right,prio=100,shift_direction="R") #118

Strategy_Pattern(L.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Right,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Anterior_Central_Right,vec2 = L.Additional_Vertebral_Body_Middle_Inferior_Right,prio=100,shift_direction="R") #119

Strategy_Pattern(L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Right,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Posterior_Central_Right,vec2 = L.Additional_Vertebral_Body_Middle_Inferior_Right,prio=100,shift_direction="R") #120

Strategy_Pattern(L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Anterior_Central_Left,vec2 = L.Additional_Vertebral_Body_Middle_Superior_Left,prio=100,shift_direction="L") #109

Strategy_Pattern(L.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Left,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Posterior_Central_Left,vec2 = L.Additional_Vertebral_Body_Middle_Superior_Left,prio=100,shift_direction="L") #110

Strategy_Pattern(L.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Left,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Anterior_Central_Left,vec2 = L.Additional_Vertebral_Body_Middle_Inferior_Left,prio=100,shift_direction="L") #111

Strategy_Pattern(L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Left,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Posterior_Central_Left,vec2 = L.Additional_Vertebral_Body_Middle_Inferior_Left,prio=100,shift_direction="L") #112
S = strategy_ligament_attachment_point_flava
Strategy_Pattern(L.Ligament_Attachment_Point_Flava_Superior_Median,goal = L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median,strategy=S,prio=200,
                 prerequisite={L.Spinosus_Process,L.Arcus_Vertebrae}
                 ) #125
Strategy_Pattern(L.Ligament_Attachment_Point_Flava_Inferior_Median,goal = L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Median,strategy=S,prio=200,
                 prerequisite={L.Spinosus_Process,L.Arcus_Vertebrae}) #127
Strategy_Computed_Before(L.Dens_axis,L.Vertebra_Direction_Inferior)
Strategy_Computed_Before(L.Spinal_Canal_ivd_lvl,L.Vertebra_Disc,L.Vertebra_Corpus,L.Dens_axis)
Strategy_Computed_Before(L.Spinal_Cord,L.Vertebra_Disc,L.Vertebra_Corpus,L.Dens_axis)
Strategy_Computed_Before(L.Spinal_Canal,L.Vertebra_Corpus)
Strategy_Computed_Before(L.Vertebra_Disc_Inferior,L.Vertebra_Disc_Inferior)

# fmt: on
def compute_non_centroid_pois(  # noqa: C901
    poi: POI,
    locations: Sequence[Location] | Location,
    vert: NII,
    subreg: NII,
    _vert_ids: Sequence[int] | None = None,
    log: Logger_Interface = _log,
):
    if _vert_ids is None:
        _vert_ids = vert.unique()

    locations = list(locations) if isinstance(locations, Sequence) else [locations]
    ### STEP 1 Vert Direction###
    assert 52 not in poi.keys_region()

    if Location.Vertebra_Direction_Inferior in locations:
        log.on_text("Compute Vertebra DIRECTIONS")
        ### Calc vertebra direction; We always need them, so we just compute them. ###
        sub_regions = poi.keys_subregion()
        if any(a.value not in sub_regions for a in vert_directions):
            poi, _ = calc_orientation_of_vertebra_PIR(poi, vert, subreg, do_fill_back=False, save_normals_in_info=False)
            [locations.remove(i) for i in vert_directions if i in locations]

    locations = [pois_computed_by_side_effect.get(l.value, l) for l in locations]
    locations = sorted(
        set(locations),
        key=lambda x: all_poi_functions[x.value].prority() if x.value in all_poi_functions else x.value,
    )  # type: ignore
    log.on_text("Calc pois from subregion id", {l.name for l in locations})
    ### DENSE ###
    if Location.Dens_axis in locations and 2 in _vert_ids and (2, Location.Dens_axis.value) not in poi:
        a = subreg * vert.extract_label(2)
        bb = a.compute_crop()
        a = a.apply_crop(bb)
        s = [Location.Vertebra_Corpus, Location.Vertebra_Corpus_border]
        if a.sum() != 0:
            strategy_extreme_points(poi, a, location=Location.Dens_axis, direction=["S", "P"], vert_id=2, subreg_id=s, bb=bb)
    ### STEP 2 (Other global non centroid poi; Spinal heights ###

    if Location.Spinal_Canal in locations:
        locations.remove(Location.Spinal_Canal)
        subregs_ids = subreg.unique()
        _a = Location.Spinal_Canal.value in subregs_ids or Location.Spinal_Canal.value in subregs_ids
        if _a and Location.Spinal_Canal.value not in poi.keys_subregion():
            poi = calc_center_spinal_cord(poi, subreg, add_dense=True)

    if Location.Spinal_Cord in locations:
        locations.remove(Location.Spinal_Cord)
        subregs_ids = subreg.unique()
        v = Location.Spinal_Cord.value
        if (v in subregs_ids or Location.Spinal_Cord.value in subregs_ids) and v not in poi.keys_subregion():
            poi = calc_center_spinal_cord(
                poi,
                subreg,
                source_subreg_point_id=Location.Vertebra_Disc,
                subreg_id=Location.Spinal_Cord,
                add_dense=True,
                intersection_target=[Location.Spinal_Cord],
            )
    if Location.Spinal_Canal_ivd_lvl in locations:
        locations.remove(Location.Spinal_Canal_ivd_lvl)
        subregs_ids = subreg.unique()
        v = Location.Spinal_Canal_ivd_lvl.value
        if (v in subregs_ids or Location.Spinal_Cord.value in subregs_ids) and v not in poi.keys_subregion():
            poi = calc_center_spinal_cord(
                poi, subreg, source_subreg_point_id=Location.Vertebra_Disc, subreg_id=Location.Spinal_Canal_ivd_lvl, add_dense=True
            )
    # Step 3 Compute on individual Vertebras
    ivd_location = set()

    for vert_id in _vert_ids:
        if vert_id >= 39:
            continue
        current_vert = vert.extract_label(vert_id)
        bb = current_vert.compute_crop()
        current_vert.apply_crop_(bb)
        current_subreg = subreg.apply_crop(bb) * current_vert
        for location in locations:
            # TODO IVD STUFF
            if location.value <= 50:
                continue
            if (vert_id, location.value) in poi:
                continue
            if location in [
                Location.Implant_Entry_Left,
                Location.Implant_Entry_Right,
                Location.Implant_Target_Left,
                Location.Implant_Target_Right,
                Location.Spinal_Canal,
            ]:
                continue
            if location in ivd_pois_list:
                ivd_location.add(location)
                continue
            if location.value in all_poi_functions:
                fun = all_poi_functions[location.value]

                if not fun.sacrum and vert_id in _sacrum:
                    continue

                fun(poi, current_subreg, vert_id, bb=bb, log=log)
            else:
                warnings.warn(f"NotImplementedError: {location}", stacklevel=0)
    # Step 4 IVD points
    if len(ivd_location) != 0:
        poi = calculate_IVD_POI(vert, subreg, poi, ivd_location)


def _print_prerequisites():
    print("digraph G {")
    for source, strategy in all_poi_functions.items():
        for prereq in strategy.prerequisite:
            print(f"{source} -> {prereq.value}")
    print("}")
