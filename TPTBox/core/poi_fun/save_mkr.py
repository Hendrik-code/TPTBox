from __future__ import annotations

import json
import random
from pathlib import Path

###### GLOBAL POI #####
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Union

import numpy as np
from typing_extensions import NotRequired

from TPTBox.core.internal.nii_help import save_json
from TPTBox.logger.log_file import log
from TPTBox.mesh3D.mesh_colors import RGB_Color, get_color_by_label

if TYPE_CHECKING:
    from TPTBox import POI_Global
CoordinateSystem = Literal["LPS", "RAS"]
ControlPointStatus = Literal["undefined", "preview", "defined"]
MarkupType = Literal["Fiducial", "Line", "Angle", "Curve", "ClosedCurve", "Plane", "ROI", "MeasurementVolume"]
VolumeUnit = Literal["mm3", "cm3"]


class MKR_Display(TypedDict, total=False):
    """TypedDict for 3D Slicer markup display properties."""

    visibility: NotRequired[bool]
    opacity: NotRequired[float]
    color: NotRequired[list[float]]
    selectedColor: NotRequired[list[float]]
    activeColor: NotRequired[list[float]]
    propertiesLabelVisibility: NotRequired[bool]
    pointLabelsVisibility: NotRequired[bool]
    textScale: NotRequired[float]
    glyphType: NotRequired[str]
    glyphScale: NotRequired[float]
    glyphSize: NotRequired[float]
    useGlyphScale: NotRequired[bool]
    sliceProjection: NotRequired[bool]
    sliceProjectionUseFiducialColor: NotRequired[bool]
    sliceProjectionOutlinedBehindSlicePlane: NotRequired[bool]
    sliceProjectionColor: NotRequired[list[float]]
    sliceProjectionOpacity: NotRequired[float]
    lineThickness: NotRequired[float]
    lineColorFadingStart: NotRequired[float]
    lineColorFadingEnd: NotRequired[float]
    lineColorFadingSaturation: NotRequired[float]
    lineColorFadingHueOffset: NotRequired[float]
    handlesInteractive: NotRequired[bool]
    translationHandleVisibility: NotRequired[bool]
    rotationHandleVisibility: NotRequired[bool]
    scaleHandleVisibility: NotRequired[bool]
    interactionHandleScale: NotRequired[float]
    snapMode: NotRequired[str]


class ControlPoint(TypedDict, total=False):
    """TypedDict representing a single Slicer markup control point."""

    id: str
    label: str
    description: str
    associatedNodeID: str
    position: list[float]  # length 3
    orientation: list[float]  # length 9
    selected: bool
    locked: bool
    visibility: bool
    positionStatus: Literal["undefined", "preview", "defined"]


class MKR_Lines(TypedDict):
    """TypedDict defining a Slicer line markup connecting a list of POI key points."""

    key_points: list[tuple[int, int]]
    color: NotRequired[list[float]]
    name: NotRequired[str]
    display: NotRequired[MKR_Display]
    controlPoint: NotRequired[ControlPoint]


class Markup(TypedDict, total=False):
    """TypedDict representing a single 3D Slicer markup node."""

    type: MarkupType
    name: str
    coordinateSystem: CoordinateSystem
    coordinateUnits: str | list[str]
    locked: bool
    fixedNumberOfControlPoints: bool
    labelFormat: str
    lastUsedControlPointNumber: int
    controlPoints: list[ControlPoint]

    # Optional (ROI/Plane)
    roiType: str | None
    insideOut: bool | None
    planeType: str | None
    sizeMode: str | None
    autoScalingSizeFactor: float | None
    center: list[float] | None
    normal: list[float] | None
    size: list[float] | None
    planeBounds: list[float] | None
    objectToBase: list[float] | None
    baseToNode: list[float] | None
    orientation: list[float] | None
    display: MKR_Display
    measurements: Any


class MeasurementVolumeMarkup(Markup, total=False):
    """Markup subtype carrying volume, surface area, and bounding-box measurements."""

    type: Literal["MeasurementVolume"]
    volume: float
    volumeUnit: VolumeUnit
    surfaceArea: float
    boundingBox: list[float]


MKR_DEFINITION = Union[MKR_Lines, dict]


def _get_display_dict(
    display: MKR_Display | dict,
    color: list[float] | None = None,
    selectedColor: list[float] | None = None,
    activeColor: list[float] | None = None,
    pointLabelsVisibility: bool = False,
    glyphScale: float = 1.0,
) -> dict:
    """Build a complete Slicer markup display dict with sensible defaults.

    Args:
        display: Partial display dict whose entries take precedence over the
            defaults.
        color: Default fiducial colour as an RGB list in ``[0, 1]`` range.
            Defaults to cyan ``[0.4, 1.0, 1.0]``.
        selectedColor: Colour when the point is selected.  Defaults to
            ``[1.0, 0.5, 0.5]``.
        activeColor: Colour when the point is active.  Defaults to
            ``[0.4, 1.0, 0.0]``.
        pointLabelsVisibility: Whether to show point labels in the 3D view.
            Defaults to ``False``.
        glyphScale: Glyph scale factor.  Defaults to ``1.0``.

    Returns:
        Complete ``MKR_Display``-compatible dict with all Slicer-required keys.
    """
    if activeColor is None:
        activeColor = [0.4, 1.0, 0.0]
    if selectedColor is None:
        selectedColor = [1.0, 0.5, 0.5]
    if color is None:
        color = [0.4, 1.0, 1.0]
    # hard cast to float, or all of "display" will be ignored...
    return {
        "visibility": display.get("visibility", True),
        "opacity": float(display.get("opacity", 1.0)),
        "color": display.get("color", color),
        "selectedColor": display.get("selectedColor", selectedColor),
        "activeColor": display.get("activeColor", activeColor),
        # Add other properties as needed, using similar patterns:
        "propertiesLabelVisibility": display.get("propertiesLabelVisibility", False),
        "pointLabelsVisibility": display.get("pointLabelsVisibility", pointLabelsVisibility),
        "textScale": float(display.get("textScale", 3.0)),
        "glyphType": display.get("glyphType", "Sphere3D"),
        "glyphScale": display.get("glyphScale", glyphScale),
        "glyphSize": float(display.get("glyphSize", 5.0)),
        "useGlyphScale": display.get("useGlyphScale", True),
        "sliceProjection": display.get("sliceProjection", False),
        "sliceProjectionUseFiducialColor": display.get("sliceProjectionUseFiducialColor", True),
        "sliceProjectionOutlinedBehindSlicePlane": display.get("sliceProjectionOutlinedBehindSlicePlane", False),
        "sliceProjectionColor": display.get("sliceProjectionColor", [1.0, 1.0, 1.0]),
        "sliceProjectionOpacity": float(display.get("sliceProjectionOpacity", 0.6)),
        "lineThickness": float(display.get("lineThickness", 0.2)),
        "lineColorFadingStart": float(display.get("lineColorFadingStart", 1.0)),
        "lineColorFadingEnd": float(display.get("lineColorFadingEnd", 10.0)),
        "lineColorFadingSaturation": float(display.get("lineColorFadingSaturation", 1.0)),
        "lineColorFadingHueOffset": float(display.get("lineColorFadingHueOffset", 0.0)),
        "handlesInteractive": display.get("handlesInteractive", False),
        "translationHandleVisibility": display.get("translationHandleVisibility", False),
        "rotationHandleVisibility": display.get("rotationHandleVisibility", False),
        "scaleHandleVisibility": display.get("scaleHandleVisibility", False),
        "interactionHandleScale": float(display.get("interactionHandleScale", 3.0)),
        "snapMode": display.get("snapMode", "toVisibleSurface"),
    }


def _get_markup_color(
    definition: MKR_DEFINITION,
    region: int,
    subregion: int,
    split_by_region: bool = False,
    split_by_subregion: bool = False,
) -> list[float]:
    """Resolve the display colour for a POI markup group.

    Priority: explicit colour in ``definition`` > colour derived from region or
    subregion ID > random colour.

    Args:
        definition: Markup definition dict that may contain a ``"color"`` key.
        region: Region (vertebra) integer label used when
            ``split_by_region`` is ``True``.
        subregion: Subregion integer label used when ``split_by_subregion`` is
            ``True``.
        split_by_region: Assign distinct colours per region.
            Defaults to ``False``.
        split_by_subregion: Assign distinct colours per subregion.
            Defaults to ``False``.

    Returns:
        RGB colour as a list of three floats in the ``[0.0, 1.0]`` range.
    """
    color = definition.get("color", None)
    if color is None:
        if split_by_region:
            color = get_color_by_label(region)
        if split_by_subregion:
            color = get_color_by_label(subregion + 10) if subregion == 83 else get_color_by_label(subregion).rgb
    if color is None:
        color = RGB_Color.init_list([random.randint(20, 245), random.randint(20, 245), random.randint(20, 245)])
    if isinstance(color, RGB_Color):  # or str(type(color)) == "RGB_Color":
        color = (color.rgb / 255.0).tolist()
    if isinstance(color, np.ndarray):
        color = color.tolist()
    assert isinstance(color, list), (color, type(color))
    color = [float(c) / 255.0 for c in color] if max(color) > 2 else [float(c) for c in color]
    return color


def _get_control_point(
    cp: ControlPoint,
    position: list[float],
    id_name: str = "",
    label: str = "",
    name: str = "",
    name2: str = "",
) -> ControlPoint:
    """Build a Slicer markup control-point dict, falling back to provided defaults.

    Args:
        cp: Partial ``ControlPoint`` dict whose entries take precedence over the
            defaults below.
        position: 3-element ``[x, y, z]`` coordinate list in the markup's
            coordinate system.
        id_name: Default control-point ID string.
        label: Default short label shown in the viewer.
        name: Default description string (used as ``description`` field).
        name2: Default ``associatedNodeID`` string.

    Returns:
        Complete ``ControlPoint`` dict ready for inclusion in a markup's
        ``controlPoints`` list.
    """
    return {
        "id": cp.get("id", id_name),
        "label": cp.get("label", label),
        "description": cp.get("description", name),
        "associatedNodeID": cp.get("associatedNodeID", name2),
        "position": cp.get("position", position),
        "orientation": cp.get("orientation", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        "selected": cp.get("selected", True),
        "locked": cp.get("locked", False),
        "visibility": cp.get("visibility", True),
        "positionStatus": cp.get("positionStatus", "defined"),
    }


def _make_default_markup(
    markup_type: MarkupType,
    name: str | None,
    coordinateSystem: CoordinateSystem,
    controlPoints: list[ControlPoint] | None = None,
    display: MKR_Display | dict | None = None,
) -> Markup | MeasurementVolumeMarkup:
    """Construct a default Slicer markup dict for the given type.

    Args:
        markup_type: Slicer markup type string (e.g. ``"Fiducial"``,
            ``"Line"``, ``"ROI"``, ``"Plane"``, ``"MeasurementVolume"``).
        name: Optional markup name shown in the Slicer hierarchy.
        coordinateSystem: Coordinate system of the control points
            (``"LPS"`` or ``"RAS"``).
        controlPoints: Initial list of control points.  Defaults to an empty
            list.
        display: Optional display property dict merged into the markup.

    Returns:
        A ``Markup`` or ``MeasurementVolumeMarkup`` dict ready for serialisation
        into a ``.mrk.json`` file.
    """
    if controlPoints is None:
        controlPoints = []
    base: Markup = {
        "type": markup_type,
        "coordinateSystem": coordinateSystem,
        "coordinateUnits": "mm",
        "locked": True,
        "fixedNumberOfControlPoints": False,
        "labelFormat": "%N-%d",
        "lastUsedControlPointNumber": 0,
        "controlPoints": controlPoints,
    }
    if name is not None:
        base["name"] = name
    if display is not None:
        base["display"] = display
    if markup_type == "ROI":
        base.update({"roiType": "Box", "insideOut": False})
    elif markup_type == "Plane":
        base.update(
            {
                "planeType": "PointNormal",
                "sizeMode": "auto",
                "autoScalingSizeFactor": 1.0,
            }
        )
    elif markup_type == "MeasurementVolume":
        mv: MeasurementVolumeMarkup = {
            **base,
            "type": "MeasurementVolume",
            "volume": 0.0,
            "volumeUnit": "mm3",
            "surfaceArea": 0.0,
            "boundingBox": [0.0] * 6,
        }
        return mv

    return base


def _get_markup_lines(
    definition: MKR_Lines,
    poi: POI_Global,
    coordinate_system: Literal["LPS", "RAS"],
    split_by_region: bool = False,
    split_by_subregion: bool = False,
    display: MKR_Display | dict | None = None,
) -> Markup:
    """Build a Slicer ``"Line"`` markup dict from a ``MKR_Lines`` definition.

    Args:
        definition: Line definition containing ``key_points``, and optionally
            ``color``, ``name``, and ``display`` overrides.
        poi: ``POI_Global`` object providing coordinates and label metadata.
        coordinate_system: Coordinate system of the POI data
            (``"LPS"`` or ``"RAS"``).
        split_by_region: Colour lines by region ID.  Defaults to ``False``.
        split_by_subregion: Colour lines by subregion ID.  Defaults to
            ``False``.
        display: Base display properties to merge with ``definition``'s display
            overrides.  Defaults to ``{}``.

    Returns:
        A ``Markup`` dict of type ``"Line"`` ready for serialisation.
    """
    if display is None:
        display = {}
    key_points = definition.get("key_points")
    region, subregion = key_points[0]
    color = _get_markup_color(definition, region, subregion, split_by_region, split_by_subregion)
    display = _get_display_dict(display | definition.get("display", {}), selectedColor=color, color=color, activeColor=color)

    controlPoints = []
    for region, subregion in key_points:
        name, name2, label = get_desc(poi, region, subregion)
        controlPoints.append(
            _get_control_point(
                definition.get("controlPoint", {}),
                poi[region, subregion],
                f"{region}-{subregion}",
                f"{region}-{subregion}",
                name,
                name2,
            )
        )
    return _make_default_markup(
        "Line",
        definition.get("name"),
        coordinate_system,
        controlPoints=controlPoints,
        display=display,
    )


def get_desc(self: POI_Global, region: int, subregion: int) -> tuple[str, str, str]:
    """Resolve the human-readable description strings for a POI.

    Looks up the label name from ``self.info["label_name"]`` if available,
    then falls back to the level-two-info enum name or the raw integer.
    The group name (region) is resolved similarly from ``self.info["label_group_name"]``
    or the level-one-info enum name.

    Args:
        self: The ``POI_Global`` instance providing ``info``,
            ``level_one_info``, and ``level_two_info``.
        region: Region (vertebra) integer label.
        subregion: Subregion integer label.

    Returns:
        Tuple ``(name, name2, label)`` where ``name`` and ``name2`` are the
        group/region name and ``label`` is the subregion label string.
    """
    label = self.info.get("label_name", {}).get(str((region, subregion)))
    if label is None:
        label = str(subregion)
        try:
            label = self.level_two_info(subregion).name
        except Exception:
            label = str(subregion)
    try:
        if region in self.info.get("label_group_name", {}):
            name2 = self.info["label_group_name"][region]
        else:
            name2 = self.level_one_info(region).name
            if "_ignore_level_one_info_range" in self.info:
                try:
                    if region in self.info["_ignore_level_one_info_range"]:
                        name2 = str(region)
                except Exception:
                    pass

    except Exception:
        name2 = str(region)
    name = name2
    return name, name2, label


def _get_key(
    region: int,
    subregion: int,
    split_by_region: bool,
    split_by_subregion: bool,
    main_key: str = "POI",
) -> str:
    """Build the markup group key string for a (region, subregion) pair.

    Args:
        region: Region (vertebra) integer label.
        subregion: Subregion integer label.
        split_by_region: Append the region ID to the key.
        split_by_subregion: Append the subregion ID to the key.
        main_key: Base key string.  Defaults to ``"POI"``.

    Returns:
        Group key string used as the ``name`` field for the Slicer markup group.
    """
    key = main_key
    if split_by_region:
        key += str(region) + "_"
    if split_by_subregion:
        key += str(subregion)

    return key


def _save_mrk(
    poi: POI_Global,
    filepath: str | Path,
    color: list[float] | None = None,
    split_by_region: bool = True,
    split_by_subregion: bool = False,
    add_points: bool = True,
    add_lines: list[MKR_Lines] | None = None,
    display: MKR_Display | dict = None,  # type: ignore
    pointLabelsVisibility: bool = False,
    glyphScale: float = 1.0,
    main_key: str = "P",
    **args,
) -> None:
    """Save a ``POI_Global`` to a 3D Slicer ``.mrk.json`` markup file.

    The coordinate system is derived from ``poi.itk_coords`` (``"LPS"`` when
    ``True``, ``"RAS"`` otherwise).  ``level_one_info`` and ``level_two_info``
    names are embedded in the control-point description fields.  Metadata from
    ``poi.info`` is not written to the file but the display colour and label
    dictionaries are preserved.

    Args:
        poi: ``POI_Global`` object to serialise.
        filepath: Output path.  The extension is forced to ``.mrk.json`` if not
            already present.
        color: Default colour for all markup groups (RGB in ``[0, 1]`` range).
            When ``None`` the colour is derived from ``split_by_*`` flags or
            set randomly per group.
        split_by_region: Create a separate markup group per region (vertebra).
            Defaults to ``True``.
        split_by_subregion: Create a separate markup group per subregion.
            Defaults to ``False``.
        add_points: Include Fiducial control-point markups.
            Defaults to ``True``.
        add_lines: Optional list of ``MKR_Lines`` definitions to add as
            ``"Line"`` markups.  Defaults to ``None`` (no lines).
        display: Base display property overrides applied to every markup group.
            Defaults to ``{}``.
        pointLabelsVisibility: Show point label text in the 3D view.
            Defaults to ``False``.
        glyphScale: Glyph size scale factor.  Defaults to ``1.0``.
        main_key: Base group key string.  Defaults to ``"P"``.
        **args: Additional key-value pairs merged into the display dict.
    """
    if display is None:
        display = {}
    if add_lines is None:
        add_lines = []
    filepath = Path(filepath)
    if not filepath.name.endswith(".mrk.json"):
        filepath = filepath.parent / (filepath.stem + ".mrk.json")
    coordinate_system: CoordinateSystem = "LPS" if poi.itk_coords else "RAS"
    list_markups = {}
    # ADD POINTS
    addendum = {
        "pointLabelsVisibility": pointLabelsVisibility,
        "glyphScale": float(glyphScale),
        **args,
    }
    display = addendum | display
    if add_points:
        # Create list of control points
        for region, subregion, coords in poi.centroids.items():
            key = _get_key(region, subregion, split_by_region, split_by_subregion, main_key=main_key)
            name, name2, label = get_desc(poi, region, subregion)
            if key not in list_markups:
                list_markups[key] = _make_default_markup(
                    "Fiducial",
                    key,
                    coordinate_system,
                    controlPoints=[],
                    display=_get_display_dict(
                        display,
                        selectedColor=_get_markup_color(
                            {"color": color}, region, subregion, split_by_region=split_by_subregion, split_by_subregion=split_by_subregion
                        ),
                        **addendum,
                    ),
                )
            list_markups[key]["controlPoints"].append(_get_control_point({}, coords, f"{region}-{subregion}", label, name, name2))
    markups = list(list_markups.values())

    [markups.append(_get_markup_lines(line, poi, coordinate_system, split_by_region, split_by_subregion, display)) for line in add_lines]
    mrk_data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": markups,
    }
    save_json(filepath, mrk_data, indent=2)
    log.on_save(f"Saved .mrk.json to {filepath}")
