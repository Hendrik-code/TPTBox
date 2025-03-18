from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, Union

import numpy as np
from typing_extensions import TYPE_CHECKING, TypeGuard  # noqa: UP035

# from TPTBox import POI, POI_Global
from TPTBox.core import bids_files
from TPTBox.core.nii_poi_abstract import Has_Grid
from TPTBox.core.poi_fun.poi_abstract import POI_Descriptor
from TPTBox.core.vert_constants import (
    AX_CODES,
    COORDINATE,
    LABEL_MAX,
    ROTATION,
    ZOOMS,
    Abstract_lvl,
    Location,
    Vertebra_Instance,
    _register_lvl,
    conversion_poi,
    conversion_poi2text,
    log,
    logging,
    v_idx2name,
    v_name2idx,
)
from TPTBox.logger import Log_Type

if TYPE_CHECKING:
    from TPTBox import POI, POI_Global, POI_Reference


### LEGACY DEFINITIONS ###
class _Point3D(TypedDict):
    X: float
    Y: float
    Z: float
    label: int


class _Orientation(TypedDict):
    direction: tuple[str, str, str]


_Centroid_DictList = Sequence[Union[_Orientation, _Point3D]]


######## Saving #######
def _is_Point3D(obj) -> TypeGuard[_Point3D]:
    return "label" in obj and "X" in obj and "Y" in obj and "Z" in obj


FORMAT_DOCKER = 0
FORMAT_GRUBER = 1
FORMAT_POI = 2
FORMAT_GLOBAL = 3
FORMAT_OLD_POI = 10

format_key = {FORMAT_DOCKER: "docker", FORMAT_GRUBER: "guber", FORMAT_POI: "POI", FORMAT_GLOBAL: "POI_GLOBAL"}
format_key2value = {value: key for key, value in format_key.items()}
global_spacing_name_key2value = {"itk": True, "nib": False}
global_spacing_name = {value: key for key, value in global_spacing_name_key2value.items()}
global_formats = (FORMAT_GLOBAL,)
ctd_info_blacklist = [
    "zoom",
    "shape",
    "direction",
    "format",
    "rotation",
    "origin",
    "level_one_info",
    "level_two_info",
]  # "location"


def save_poi(
    poi: POI | POI_Global,
    out_path: Path | str,
    make_parents=False,
    additional_info: dict | None = None,
    resample_reference: Has_Grid | None = None,
    verbose: logging = True,
    save_hint=2,
) -> None:
    """
    Saves the POIs to a JSON file.

    Args:
        out_path (Path | str): The path where the JSON file will be saved.
        make_parents (bool, optional): If True, create any necessary parent directories for the output file.
            Defaults to False.
        verbose (bool, optional): If True, print status messages to the console. Defaults to True.
        save_hint: 0 Default, 1 Gruber, 2 POI (readable), 10 ISO-POI (outdated)
    Returns:
        None

    Raises:
        TypeError: If any of the POIs have an invalid type.

    Example:
        >>> POIs = Centroids(...)
        >>> POIs.save("output/POIs.json")
    """
    _file_types = ["json"]
    file_ending = Path(out_path).name.split(".")[-1]
    if file_ending not in _file_types:
        raise ValueError(f"Not supported file ending for POI: {file_ending} not in {_file_types}")
    if make_parents:
        Path(out_path).parent.mkdir(exist_ok=True, parents=True)

    poi.sort()
    out_path = str(out_path)
    if len(poi.centroids) == 0:
        log.print("POIs empty, not saved:", out_path, ltype=Log_Type.FAIL, verbose=verbose)
        return
    json_object, print_add = _poi_to_dict_list(poi, additional_info, save_hint, resample_reference, verbose)

    # Problem with python 3 and int64 serialization.
    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(out_path, "w") as f:
        json.dump(json_object, f, default=convert, indent=4)
    log.print("POIs saved:", out_path, print_add, ltype=Log_Type.SAVE, verbose=verbose)


def _poi_to_dict_list(  # noqa: C901
    ctd: POI | POI_Global, additional_info: dict | None, save_hint=0, resample_reference: Has_Grid | None = None, verbose: logging = False
):
    from TPTBox import POI, POI_Global

    if isinstance(ctd, POI_Global) and resample_reference is None:
        save_hint = FORMAT_GLOBAL
    # noqa: C901
    if save_hint in global_formats:
        ctd = ctd.to_global()
        ori: dict[str, str | COORDINATE | AX_CODES] = {"coordinate_system": global_spacing_name[ctd.itk_coords]}
    else:
        if resample_reference is not None:
            ctd = ctd.resample_from_to(resample_reference)
        assert isinstance(ctd, POI), f"Try to save a {type(ctd)} in local coords without a resample_reverence"

        ori: dict[str, str | COORDINATE | AX_CODES] = {"direction": ctd.orientation}
        print_out = ""
        if ctd.zoom is not None:
            ori["zoom"] = ctd.zoom
        if ctd.origin is not None:
            ori["origin"] = ctd.origin  # type: ignore
        if ctd.rotation is not None:
            ori["rotation"] = ctd.rotation  # type: ignore
        if ctd.shape is not None:
            ori["shape"] = ctd.shape  # type: ignore

    if save_hint in format_key:
        ori["format"] = format_key[save_hint]  # type: ignore
        print_out = "in format " + format_key[save_hint]

    ori["level_one_info"] = str(ctd.level_one_info.__name__)
    ori["level_two_info"] = str(ctd.level_two_info.__name__)
    if additional_info is not None:
        for k, v in additional_info.items():
            if k not in ori:
                ori[k] = v

    for k, v in ctd.info.items():
        if k not in ori:
            ori[k] = v

    dict_list: list[_Orientation | (_Point3D | dict)] = [ori]

    if save_hint == FORMAT_OLD_POI:
        assert isinstance(ctd, POI)
        ctd = ctd.rescale((1, 1, 1), verbose=verbose).reorient_(("R", "P", "I"), verbose=verbose)
        dict_list = []

    temp_dict = {}
    ctd.sort()
    for vert_id, subreg_id, (x, y, z) in ctd.items():
        if save_hint == FORMAT_DOCKER:
            dict_list.append({"label": subreg_id * LABEL_MAX + vert_id, "X": x, "Y": y, "Z": z})
        elif save_hint == FORMAT_GRUBER:
            v = v_idx2name[vert_id].replace("T", "TH") + "_" + conversion_poi2text[subreg_id]
            dict_list.append({"label": v, "X": x, "Y": y, "Z": z})
        elif save_hint in (FORMAT_POI, FORMAT_GLOBAL):
            v_name = ctd.level_one_info._get_name(vert_id, no_raise=True)
            subreg_id = ctd.level_two_info._get_name(subreg_id, no_raise=True)  # noqa: PLW2901
            # sub_name = v_idx2name[subreg_id]
            if v_name not in temp_dict:
                temp_dict[v_name] = {}
            temp_dict[v_name][subreg_id] = (x, y, z)
        elif save_hint == FORMAT_OLD_POI:
            if vert_id not in temp_dict:
                temp_dict[vert_id] = {}
            temp_dict[vert_id][str(subreg_id)] = str((float(x), float(y), float(z)))
        else:
            raise NotImplementedError(save_hint)
    if len(temp_dict) != 0:
        if save_hint == FORMAT_OLD_POI:
            for k, v in temp_dict.items():
                out_dict = {"vert_label": str(k), **v}
                dict_list.append(out_dict)
        else:
            dict_list.append(temp_dict)
    return dict_list, print_out


######### Load  #############
def load_poi(ctd_path: POI_Reference, verbose=True) -> POI | POI_Global:  # noqa: ARG001
    """
    Load POIs from a file or a BIDS file object.

    Args:
        ctd_path (Centroid_Reference): Path to a file or BIDS file object from which to load POIs.
            Alternatively, it can be a tuple containing the following items:
            - vert: str, the name of the vertebra.
            - subreg: str, the name of the subregion.
            - ids: list[int | Location], a list of integers and/or Location objects used to filter the POIs.

    Returns:
        A Centroids object containing the loaded POIs.

    Raises:
        AssertionError: If `ctd_path` is not a recognized type.

    """
    from TPTBox import POI, calc_poi_from_subreg_vert

    ### Check Datatype ###
    if isinstance(ctd_path, POI):
        return ctd_path
    elif isinstance(ctd_path, bids_files.BIDS_FILE):
        dict_list: _Centroid_DictList = ctd_path.open_json()  # type: ignore
    elif isinstance(ctd_path, (Path, str)):
        with open(ctd_path) as json_data:
            dict_list: _Centroid_DictList = json.load(json_data)
            json_data.close()
    elif isinstance(ctd_path, tuple):
        vert = ctd_path[0]
        subreg = ctd_path[1]
        ids: list[int | Location] = ctd_path[2]  # type: ignore
        return calc_poi_from_subreg_vert(vert, subreg, subreg_id=ids)
    else:
        raise TypeError(f"{type(ctd_path)}\n{ctd_path}")
    ### New Spine_r has a dict instead of a dict list. ###
    if isinstance(dict_list, dict):
        return _load_form_POI_spine_r2(dict_list)
    ### format_POI_old has no META header ###
    if "direction" not in dict_list[0] and "vert_label" in dict_list[0]:
        return _load_format_POI_old(dict_list)  # This file if used in the old POI-pipeline and is deprecated
    ### Ours Global Space ##
    format_ = dict_list[0].get("format", None)
    format_ = format_key2value[format_] if format_ is not None else None
    level_one_info = _register_lvl[dict_list[0].get("level_one_info", Vertebra_Instance.__name__)]
    level_two_info = _register_lvl[dict_list[0].get("level_two_info", Location.__name__)]
    info = {k: v for k, v in dict_list[0].items() if k not in ctd_info_blacklist}
    if format_ == FORMAT_GLOBAL:
        from TPTBox import POI_Global

        centroids = POI_Descriptor()
        itk_coords = global_spacing_name_key2value[dict_list[0].get("coordinate_system", "nib")]
        _load_POI_centroids(dict_list, centroids, level_one_info, level_two_info)
        return POI_Global(centroids, itk_coords=itk_coords, level_one_info=level_one_info, level_two_info=level_two_info, info=info)

    ### Ours ###
    assert "direction" in dict_list[0], f'File format error: first index must be a "Direction" but got {dict_list[0]}'
    axcode: AX_CODES = tuple(dict_list[0]["direction"])  # type: ignore
    zoom: ZOOMS = dict_list[0].get("zoom", None)  # type: ignore
    shape = dict_list[0].get("shape", None)  # type: ignore
    shape = tuple(shape) if shape is not None else None
    origin = dict_list[0].get("origin", None)
    origin = tuple(origin) if origin is not None else None
    rotation: ROTATION = dict_list[0].get("rotation", None)

    centroids = POI_Descriptor()
    if format_ in (FORMAT_DOCKER, FORMAT_GRUBER) or format_ is None:
        _load_docker_centroids(dict_list, centroids, format_)
    elif format_ == FORMAT_POI:
        _load_POI_centroids(dict_list, centroids, level_one_info, level_two_info)
    else:
        raise NotImplementedError(format_)
    return POI(
        centroids=centroids,
        orientation=axcode,
        zoom=zoom,
        shape=shape,  # type: ignore
        format=format_,
        info=info,
        origin=origin,  # type: ignore
        rotation=rotation,  # type: ignore
        level_one_info=level_one_info,
        level_two_info=level_two_info,
    )  # type: ignore


def _load_docker_centroids(dict_list, centroids: POI_Descriptor, format_):  # noqa: ARG001
    for d in dict_list[1:]:
        assert "direction" not in d, f'File format error: only first index can be a "direction" but got {dict_list[0]}'
        if "nan" in str(d):  # skipping NaN POIs
            continue
        elif _is_Point3D(d):
            try:
                a = int(d["label"])
                subreg = a // LABEL_MAX
                if subreg == 0:
                    subreg = 50
                centroids[a % LABEL_MAX, subreg] = (d["X"], d["Y"], d["Z"])
            except Exception:
                try:
                    number, subreg = str(d["label"]).split("_", maxsplit=1)
                    number = number.replace("TH", "T").replace("SA", "S1")
                    vert_id = v_name2idx[number]
                    subreg_id = conversion_poi[subreg]
                    centroids[vert_id, subreg_id] = (d["X"], d["Y"], d["Z"])
                except Exception:
                    print(f"Label {d['label']} is not an integer and cannot be converted to an int")
                    centroids[0, d["label"]] = (d["X"], d["Y"], d["Z"])
        else:
            raise ValueError(d)


def _load_format_POI_old(dict_list):
    # [
    # {
    #    "vert_label": "8",
    #    "85": "(281, 185, 274)",
    #    ...
    # }{...}
    # ...
    # ]
    from TPTBox import POI

    centroids = POI_Descriptor()
    for d in dict_list:
        d: dict[str, str]
        vert_id = int(d["vert_label"])
        for k, v in d.items():
            if k == "vert_label":
                continue
            sub_id = int(k)
            t = v.replace("(", "").replace(")", "").replace(" ", "").split(",")
            t = tuple(float(x) for x in t)
            centroids[vert_id, sub_id] = t
    return POI(centroids, orientation=("R", "P", "I"), zoom=(1, 1, 1), shape=None, format=FORMAT_OLD_POI, rotation=None)  # type: ignore


def _load_form_POI_spine_r2(data: dict):
    from TPTBox import POI

    orientation = None
    centroids = POI_Descriptor()
    for d in data["centroids"]["centroids"]:
        if "direction" in d:
            orientation = d["direction"]
            continue
        centroids[d["label"], 50] = (d["X"], d["Y"], d["Z"])
    zoom = data.get("Spacing")
    shape = data.get("Shape")
    rotation = data.get("Rotation")  # Does not exist
    origin = data.get("Origin")  # Does not exist
    return POI(
        centroids,
        orientation=orientation,  # type: ignore
        zoom=zoom,  # type: ignore
        shape=shape,  # type: ignore
        info=data,
        rotation=rotation,  # type: ignore
        origin=origin,  # type: ignore
    )


def _load_POI_centroids(
    dict_list,
    centroids: POI_Descriptor,
    level_one_info: Abstract_lvl,
    level_two_info: Abstract_lvl,
):
    assert len(dict_list) == 2
    d: dict[int | str, dict[int | str, tuple[float, float, float]]] = dict_list[1]
    for vert_id, v in d.items():
        vert_id = level_one_info._get_id(vert_id, no_raise=True)  # noqa: PLW2901
        for sub_id, t in v.items():
            sub_id = level_two_info._get_id(sub_id, no_raise=True)  # noqa: PLW2901
            centroids[vert_id, sub_id] = tuple(t)
