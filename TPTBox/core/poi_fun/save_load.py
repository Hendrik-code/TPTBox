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
    Any,
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
FORMAT_PLST = 4
FORMAT_OLD_POI = 10

format_key = {FORMAT_DOCKER: "docker", FORMAT_GRUBER: "guber", FORMAT_POI: "POI", FORMAT_GLOBAL: "POI_GLOBAL", FORMAT_PLST: "POINT_LIST"}
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
        if isinstance(o, Path):
            return str(o.absolute())
        raise TypeError(type(o))

    with open(out_path, "w") as f:
        json.dump(json_object, f, default=convert, indent=4)
    log.print("POIs saved:", out_path, print_add, ltype=Log_Type.SAVE, verbose=verbose)


def _poi_to_dict_list(  # noqa: C901
    ctd: POI | POI_Global,
    additional_info: dict | None,
    save_hint=0,
    resample_reference: Has_Grid | None = None,
    verbose: logging = False,
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
            if "_ignore_level_one_info_range" in ctd.info:
                try:
                    if vert_id in ctd.info["_ignore_level_one_info_range"]:
                        v_name = vert_id
                except Exception:
                    pass
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


def _open_file(ctd_path: Union[Path, str, bids_files.BIDS_FILE]) -> dict | list:
    # BIDS JSON
    if isinstance(ctd_path, bids_files.BIDS_FILE):
        if "json" in ctd_path.file:
            try:
                return ctd_path.open_json()
            except json.JSONDecodeError:
                pass  # not JSON → continue
            except OSError as e:
                raise OSError(f"Could not open file: {ctd_path}") from e
        elif "txt" in ctd_path.file:
            return _load_landmark_txt(ctd_path.file["txt"])
        else:
            raise OSError(f"Could not open file: {ctd_path}, need a json or txt file")
    # filesystem path
    path = Path(ctd_path)  # type: ignore

    # --- 1) try JSON ---
    try:
        with path.open("r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        pass  # not JSON → continue
    except OSError as e:
        raise OSError(f"Could not open file: {path}") from e
    except UnicodeDecodeError as e:
        raise OSError(f"Could not open file: {path}") from e

    # --- 2) try landmark TXT ---
    try:
        return _load_landmark_txt(path)
    except Exception as e:
        raise ValueError(f"File {path} is neither valid JSON nor a supported landmark TXT format") from e


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

    # already loaded
    if isinstance(ctd_path, POI):
        return ctd_path
    if isinstance(ctd_path, tuple):
        vert = ctd_path[0]
        subreg = ctd_path[1]
        ids: list[int | Location] = ctd_path[2]  # type: ignore
        return calc_poi_from_subreg_vert(vert, subreg, subreg_id=ids)

    ### Check Datatype ###
    dict_list = _open_file(ctd_path)
    ### New Spine_r has a dict instead of a dict list. ###
    if isinstance(dict_list, dict):
        if "markups" in dict_list:
            return _load_mkr_POI(dict_list)
        else:
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
    if format_ in (FORMAT_GLOBAL, FORMAT_PLST):
        from TPTBox import POI_Global

        centroids = POI_Descriptor()
        itk_coords = global_spacing_name_key2value[dict_list[0].get("coordinate_system", "nib")]
        _load_POI_centroids(dict_list, centroids, level_one_info, level_two_info)

        return POI_Global(centroids, itk_coords=itk_coords, level_one_info=level_one_info, level_two_info=level_two_info, info=info)

    ### Ours ###
    assert "direction" in dict_list[0] or format_ in [FORMAT_PLST], (
        f'File format error: first index must be a "Direction" but got {dict_list[0]}'
    )
    axcode: AX_CODES = tuple(dict_list[0].get("direction", ("U", "U", "U")))  # type: ignore
    zoom: ZOOMS = dict_list[0].get("zoom", None)  # type: ignore
    shape = dict_list[0].get("shape", None)  # type: ignore
    shape = tuple(shape) if shape is not None else None
    origin = dict_list[0].get("origin", None)
    origin = tuple(origin) if origin is not None else None
    rotation: ROTATION = dict_list[0].get("rotation", None)

    centroids = POI_Descriptor()
    if format_ in (FORMAT_DOCKER, FORMAT_GRUBER) or format_ is None:
        _load_docker_centroids(dict_list, centroids, format_)
    elif format_ in (FORMAT_POI,):
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


def _get_poi_idx_from_text(idx: str, label: str, centroids):
    has_ids = False
    if "-" in label:
        try:
            a, b = label.split("-")
            region, subregion = int(a), int(b)
            has_ids = True
        except Exception:
            pass
        if not has_ids:
            try:
                a, b = label.split("-")
                region, subregion = Any._get_id(a), Any._get_id(b)
                has_ids = True
            except Exception:
                pass

    if "-" in idx:
        try:
            a, b = idx.split("-")
            region, subregion = int(a), int(b)
            has_ids = True
        except Exception:
            pass
        if not has_ids:
            try:
                a, b = idx.split("-")
                region, subregion = Any._get_id(a), Any._get_id(b)
                has_ids = True
            except Exception:
                pass
    if not has_ids:
        try:
            region, subregion = int(idx), 1
            has_ids = True
        except Exception:
            pass
    if not has_ids:
        try:
            subregion: int = 1
            region: int = Any._get_id(idx)
            has_ids = True
        except Exception:
            pass
    if not has_ids:
        region = 1
        subregion = 1
    while (region, subregion) in centroids:
        subregion += 1
    return region, subregion


def _load_mkr_POI(dict_mkr: dict):
    centroids = POI_Descriptor()

    if "@schema" not in dict_mkr or "markups-schema-v1.0.3" not in dict_mkr["@schema"]:
        log.on_warning(
            "this file is possible incompatible. Tested only with markups-schema-v1.0.3 and not",
            dict_mkr.get("@schmea", "No Schema"),
        )
    if "markups" not in dict_mkr:
        raise ValueError("markups is missing")
    itk_coords = None
    if dict_mkr.get("coordinateSystem") in ["LPS", "RAS"]:
        itk_coords = dict_mkr["coordinateSystem"] == "LPS"
    label_name = {}
    label_group_name = {}
    for markup in dict_mkr["markups"]:
        if markup["type"] != "Fiducial":
            log.on_warning("skip unknown markup type:", markup["type"])
            continue
        if markup["coordinateSystem"] not in ["LPS", "RAS"]:
            log.on_warning("unknown coordinate system:", markup["coordinateSystem"])
            continue
        if itk_coords is not None:
            assert (markup["coordinateSystem"] == "LPS") == itk_coords, "multiple rotations not supported"
        itk_coords = markup["coordinateSystem"] == "LPS"
        if markup.get("coordinateUnits", "mm") != "mm":
            log.on_warning("unknown coordinateUnits:", markup["coordinateUnits"])
            continue
        if "measurements" in markup and len(markup["measurements"]) != 0:
            log.on_warning("this loader ignores measurements key", markup["measurements"])
        if "controlPoints" not in markup:
            log.on_warning("no controlPoints")
            continue
        for e, control_points in enumerate(markup["controlPoints"]):
            if "position" not in control_points:
                log.on_warning("controlPoints without position", control_points)

            idx: str = control_points.get("id", str(e))

            label = control_points.get("label", str(e))
            position = control_points["position"]
            # orientation = controlPoints.get("orientation", None)
            region, subregion = _get_poi_idx_from_text(idx, label, centroids)
            centroids[region, subregion] = tuple(position)

            if region not in label_group_name:
                description = control_points.get("description", region)
                associatedNodeID = control_points.get("associatedNodeID", description)
                label_group_name[region] = associatedNodeID

            label_name[str((region, subregion))] = label
    assert itk_coords is not None, "itk_coords not set"
    from TPTBox import POI_Global

    poi = POI_Global(centroids, itk_coords=itk_coords)
    if "display" in dict_mkr and "color" in dict_mkr["display"]:
        # TODO keep all display, locked etc info in the info dict
        poi.info["color"] = dict_mkr["display"]["color"]
    poi.info["label_name"] = label_name
    poi.info["label_group_name"] = label_group_name
    return poi


def _parse_coords(coord_str: str) -> list[float]:
    """
    Parse '(x, y, z)' → [x, y, z]
    """
    coord_str = coord_str.strip()

    if not (coord_str.startswith("(") and coord_str.endswith(")")):
        raise ValueError(f"Invalid coordinate format: {coord_str}")

    values = coord_str[1:-1].split(",")

    if len(values) != 3:
        raise ValueError(f"Expected 3 coordinates, got {values}")

    return [float(v.strip()) for v in values]


def _parse_header_value(value: str):
    """
    Parse header values into numbers, lists, or nested lists if possible.

    Examples:
        "3.14" -> 3.14
        "256 931 27" -> [256, 931, 27]
        "[1, 2, 3]" -> [1, 2, 3]
        "[[1, 0, 0], [0, 1, 0]]" -> [[1, 0, 0], [0, 1, 0]]
        "-3.5e-12" -> -3.5e-12
    """

    value = value.strip()

    # --- single number ---
    try:
        num = float(value)
        return int(num) if num.is_integer() else num
    except ValueError:
        pass

    # --- bracketed list ---
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []

        items = []
        depth = 0
        token = ""

        for ch in inner:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1

            if ch == "," and depth == 0:
                items.append(_parse_header_value(token))
                token = ""
            else:
                token += ch

        if token:
            items.append(_parse_header_value(token))

        return items

    # --- space separated list ---
    if " " in value:
        parts = value.split()
        parsed = []
        for p in parts:
            try:
                num = float(p)
                parsed.append(int(num) if num.is_integer() else num)
            except ValueError:
                return value  # mixed types → keep string
        return parsed

    # --- fallback ---
    return value


def _load_landmark_txt(path: Path):
    header: dict = {
        "format": format_key[FORMAT_PLST],
        "coordinate_system": "nib",
        "level_one_info": "Any",
        "level_two_info": "Any",
    }
    points: dict = {}
    label_group_name = {}
    label_name = {}
    label_group_id = 1
    current_group: str | None = None
    with path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # split only once
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # --- group header (e.g. "Femur proximal:") ---
            if value == "":
                current_group = key
                points.setdefault(label_group_id, {})
                label_group_name[current_group] = label_group_id
                label_group_id += 1
                continue

            # --- file header ---
            if current_group is None or "(" not in value:
                header[key] = _parse_header_value(value)
                continue

            coords = _parse_coords(value)
            id_ = label_group_name[current_group]
            new_id = len(points[id_]) + 1
            points[id_][new_id] = coords
            label_name[str((id_, new_id))] = key
    if len(label_name) != 0:
        header["label_name"] = label_name
    if len(label_group_name) != 0:
        header["label_group_name"] = {v: k for k, v in label_group_name.items()}

    return [header, points]


if __name__ == "__main__":
    from TPTBox import NII, POI, Location, POI_Global, Vertebra_Instance, calc_poi_from_subreg_vert, to_nii

    # nii = "/DATA/NAS/datasets_processed/CT_spine/dataset-myelom/rawdata/CTFU01051/ses-20130430/sub-CTFU01051_ses-20130430_sequ-2_ct.nii.gz"
    # fam = "/DATA/NAS/datasets_processed/CT_spine/dataset-myelom/"
    # poi = POI_Global.load("/DATA/NAS/tools/TPTBox/sub-CTFU01051_ses-20130430.txt")  # reference=BIDS_FILE(nii, fam).get_grid_info()
    # poi.save_mrk(
    #    "/DATA/NAS/datasets_processed/CT_spine/dataset-myelom/rawdata/CTFU01051/ses-20130430/sub-CTFU01051_ses-20130430_sequ-2_poi.json"
    # )
    #
    #
    r = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-myelom/derivatives-spineps-Dataset612-v3/MM00100/ses-20180314")
    vert = r / "sub-MM00100_ses-20180314_sequ-202_seg-vert-post-test_msk.nii.gz"
    spine = r / "sub-MM00100_ses-20180314_sequ-202_mod-ct_seg-spine_msk.nii.gz"
    first_class_citizen = r / "sub-MM00100_ses-20180314_sequ-202_mod-ct_seg-spine_msk.nii.gz.seg.nrrd"
    second_class_citizen = r / "sub-MM00100_ses-20180314_sequ-202_mod-ct_seg-spine-2_msk.nii.gz.seg.nrrd"

    to_nii(spine, True).save_nrrd(second_class_citizen)
    # n = NII.load_nrrd(first_class_citizen, True)

    n = NII.load_nrrd(second_class_citizen, True)
    print(n.info)
    print(n.unique())
    # {
    #    "encoding": "gzip",
    #    "containedRepresentationNames": ["Binary labelmap", "Closed surface"],
    #    "masterRepresentation": "Binary labelmap",
    #    "referenceImageExtentOffset": [0, 0, 0],
    #    "segments": [
    #        {
    #            "id": "Segment_41",
    #            "color": [0.933333, 0.909804, 0.666667],
    #            "colorAutoGenerated": False,
    #            "labelValue": 41,
    #            "layer": 0,
    #            "name": "Segment_41",
    #            "nameAutoGenerated": True,
    #            "terminology": {
    #                "contextName": "Segmentation category and type - 3D Slicer General Anatomy list",
    #                "category": ["SCT", "85756007", "Tissue"],
    #                "type": ["SCT", "85756007", "Tissue"],
    #                "anatomicContextName": "Anatomic codes - DICOM master list",
    #            },
    #        }
    #    ],
    # }
    # exit()
    vert = to_nii(vert, True)
    spine = to_nii(spine, True)
    poi = calc_poi_from_subreg_vert(
        vert,
        spine,
        subreg_id=[
            *list(range(41, 51)),
            100,
            *list(range(81, 90)),
            *list(range(101, 126)),
            *list(range(127, 130)),
        ],
    ).to_global()
    for v in [Vertebra_Instance.C5, Vertebra_Instance.T8, Vertebra_Instance.L4]:
        subreg = spine * vert.extract_label(v)
        subreg.save_nrrd(r / f"subreg_{v.name}.nrrd")

        out = r / f"cms_{v.name}.json"
        poi.extract_subregion(*list(range(41, 51)), 100).extract_region(v).save_mrk(out, split_by_subregion=True, glyphScale=3)
        out = r / f"extrem_{v.name}.json"
        poi.extract_subregion(*list(range(81, 90))).extract_region(v).save_mrk(out, split_by_subregion=True, glyphScale=3)
        out = r / f"corpus_{v.name}.json"
        poi.extract_subregion(*list(range(101, 125))).extract_region(v).save_mrk(out, split_by_subregion=True, glyphScale=3)
        out = r / f"flavum_{v.name}.json"
        poi.extract_subregion(125, 127).extract_region(v).save_mrk(out, split_by_subregion=True, glyphScale=3)
