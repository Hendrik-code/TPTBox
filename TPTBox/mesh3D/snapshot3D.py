# Source: https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/preview.py
from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np
import vtk
from fury import window
from PIL import Image
from vtk.util import numpy_support  # type: ignore
from xvfbwrapper import Xvfb

from TPTBox import NII, Image_Reference, to_nii_seg
from TPTBox.core.compat import zip_strict
from TPTBox.logger import Reflection_Logger
from TPTBox.mesh3D.mesh_colors import get_color_by_label

logger = Reflection_Logger()

# This is not the same as an orientation; we could add diagonal views here.
VIEW = Literal["R", "A", "L", "P", "S", "I"]

_red = [1, 0, 0]


def make_snapshot3D(
    img: Image_Reference,
    output_path: str | Path | None,
    view: VIEW | list[VIEW] = "A",
    ids_list: list[Sequence[int]] | None = None,
    smoothing: int = 20,
    resolution: float | None = None,
    width_factor: float = 1.0,
    scale_factor: int = 1,
    verbose: bool = True,
    crop: bool = True,
    png_magnify: int = 1,
    opacity: dict[int, float] | None = None,
) -> Image.Image:
    """Generate a 3D surface-rendered snapshot from a segmentation image.

    Renders each label in the segmentation with its ITK color, arranges the
    specified views side-by-side, and saves the composite image as a PNG.

    Args:
        img: Source segmentation image reference (NIfTI path, NII, etc.).
        output_path: Destination PNG path. If None, a temporary file is used and
            the resulting image is returned without being saved permanently.
        view: Camera direction(s) for the render. Accepts a single direction
            string or a list. Valid values: ``"R"``, ``"A"``, ``"L"``, ``"P"``,
            ``"S"``, ``"I"``.
        ids_list: Per-view lists of label IDs to render. If None, all unique
            non-zero labels are used for every view.
        smoothing: Number of VTK smoothing iterations applied to each surface.
        resolution: Isotropic voxel size (mm) to resample to before rendering.
            Defaults to the minimum zoom of the image.
        width_factor: Multiplier applied to the per-view pixel width.
        scale_factor: PNG magnification factor passed to fury's record function.
        verbose: If True, logs the output path after saving.
        crop: If True, crops the image to its bounding box before rendering.
        png_magnify: Window pixel density multiplier for the fury renderer.
        opacity: mapping idx to opacity (1 means full, 0 invisible)

    Returns:
        The rendered snapshot as a PIL Image object.
    """
    if opacity is None:
        opacity = {}
    is_tmp = output_path is None
    t = None
    if output_path is None:
        t = NamedTemporaryFile(suffix="_snap3D.png")  # noqa: SIM115
        output_path = str(t.name)
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    nii = to_nii_seg(img)
    if crop:
        try:
            nii.apply_crop_(nii.compute_crop(dist=2))
        except ValueError:
            pass
    if resolution is None:
        resolution = min(nii.zoom)
    if isinstance(view, str):
        view = [view]
    if ids_list is None:
        u = nii.unique()
        ids_list = [u for _ in view]
    if len(ids_list) < len(view):
        ids_list2 = []
        for i in ids_list:
            for _ in view:
                ids_list2.append(i)  # noqa: PERF401
        ids_list = ids_list2

    # TOP : ("A", "I", "R")
    nii = nii.reorient(("A", "S", "L")).rescale_((resolution, resolution, resolution), mode="constant")
    width = int(max(nii.shape[0], nii.shape[2]) * width_factor)
    window_size = (width * len(ids_list) * png_magnify, nii.shape[1] * png_magnify)
    with Xvfb():
        scene = window.Scene()
        show_m = window.ShowManager(scene=scene, size=window_size, reset_camera=False, png_magnify=png_magnify)
        show_m.initialize()
        for i, ids in enumerate(ids_list):
            x = width * i
            _plot_sub_seg(scene, nii.extract_label(ids, keep_label=True), x, 0, smoothing, view[i % len(view)], opacity=opacity)
        scene.projection(proj_type="parallel")
        scene.reset_camera_tight(margin_factor=1.02)
        window.record(
            scene=scene,
            size=window_size,
            out_path=output_path,
            reset_camera=False,
            magnification=scale_factor,
        )
        scene.clear()
    if not is_tmp:
        logger.on_save("Save Snapshot3D:", output_path, verbose=verbose)
    out_img = Image.open(output_path)
    if t is not None:
        t.close()
    return out_img


def make_snapshot3D_parallel(
    imgs: list[Image_Reference],
    output_paths: list[Path | str],
    view: VIEW | list[VIEW] = "A",
    ids_list: list[Sequence[int]] | None = None,
    smoothing: int = 20,
    resolution: float = 1,
    cpus: int = 10,
    width_factor: float = 1.0,
    png_magnify: int = 1,
    scale_factor: int = 1,
    override: bool = True,
    crop: bool = True,
    opacity: dict[int, float] | None = None,
) -> None:
    """Run :func:`make_snapshot3D` in parallel across multiple images.

    Args:
        imgs: List of segmentation image references to render.
        output_paths: Destination PNG paths, one per image in ``imgs``.
        view: Camera direction(s) forwarded to :func:`make_snapshot3D`.
        ids_list: Per-view label ID lists forwarded to :func:`make_snapshot3D`.
        smoothing: VTK smoothing iterations forwarded to :func:`make_snapshot3D`.
        resolution: Isotropic voxel size (mm) for resampling.
        cpus: Number of worker processes in the multiprocessing pool.
        width_factor: Per-view width multiplier.
        png_magnify: Window pixel density multiplier.
        scale_factor: PNG magnification factor.
        override: If False, skips images whose output file already exists.
        crop: If True, crops each image to its bounding box before rendering.
    """
    ress = []
    with Pool(cpus) as p:  # type: ignore
        for out_path, img in zip_strict(output_paths, imgs):
            if not override and Path(out_path).exists():
                continue
            res = p.apply_async(
                make_snapshot3D,
                kwds={
                    "output_path": out_path,
                    "img": img,
                    "view": view,
                    "ids_list": ids_list,
                    "smoothing": smoothing,
                    "resolution": resolution,
                    "width_factor": width_factor,
                    "png_magnify": png_magnify,
                    "crop": crop,
                    "scale_factor": scale_factor,
                    "opacity": opacity,
                },
            )
            ress.append(res)
        for res in ress:
            res.get()
        p.close()
        p.join()


make_sub_snapshot_parallel = make_snapshot3D_parallel


def _plot_sub_seg(
    scene: window.Scene, nii: NII, x: int, y: int, smoothing: int, orientation: VIEW, opacity: dict[int, float] | None = None
) -> None:
    """Render all labels from a segmentation NII into the fury scene at the given viewport offset."""
    if opacity is None:
        opacity = {}
    if orientation == "A":
        #               [  axis1(w)   ]  [  axis2(h)   ]  [  view in ]
        affine = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        # nii = nii.reorient(("A", "S", "L"))
    elif orientation == "P":
        affine = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        nii = nii.reorient(("A", "S", "R"))
    elif orientation == "L":
        affine = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        nii = nii.reorient(("P", "S", "R"))
    elif orientation == "R":
        affine = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # nii = nii.reorient(("A", "S", "L"))
    elif orientation == "S":
        affine = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    elif orientation == "I":
        affine = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    else:
        raise NotImplementedError()
    for idx in nii.unique():
        o = opacity.get(idx, 1)
        if o == 0:
            continue
        color = get_color_by_label(idx)
        cont_actor = _plot_mask(
            nii.extract_label(idx),
            affine,
            x,
            y,
            smoothing=smoothing,
            color=color,
            opacity=o,
        )
        scene.add(cont_actor)


def _plot_mask(
    nii: NII,
    affine: np.ndarray,
    x_current: int,
    y_current: int,
    smoothing: int = 10,
    color: list | np.ndarray = _red,
    opacity: float = 1,
) -> vtk.vtkActor:
    """Create a smoothed surface actor for a binary mask and position it in the scene."""
    mask = nii.get_seg_array()
    cont_actor = _contour_from_roi_smooth(mask, affine=affine, color=color, opacity=opacity, smoothing=smoothing)
    cont_actor.SetPosition(x_current, y_current, 0)
    return cont_actor


def _set_input(
    vtk_object: vtk.vtkImageReslice | vtk.vtkPolyData | vtk.vtkImageData | vtk.vtkAlgorithmOutput,
    inp: vtk.vtkImageData | vtk.vtkAlgorithmOutput,
) -> vtk.vtkImageReslice | vtk.vtkPolyData | vtk.vtkImageData | vtk.vtkAlgorithmOutput:
    """Set input data on a VTK object, compatible with VTK 5 and 6+ APIs.

    Copied from dipy.viz.utils.
    """
    if isinstance(inp, (vtk.vtkPolyData, vtk.vtkImageData)):
        vtk_object.SetInputData(inp)  # type: ignore
    elif isinstance(inp, vtk.vtkAlgorithmOutput):
        vtk_object.SetInputConnection(inp)  # type: ignore
    vtk_object.Update()  # type: ignore
    return vtk_object


def _contour_from_roi_smooth(
    data: np.ndarray,
    affine: np.ndarray | None = None,
    color: np.ndarray | list = _red,
    opacity: float = 1,
    smoothing: int = 0,
) -> vtk.vtkActor:
    """Generate a smoothed surface VTK actor from a binary 3-D ROI array.

    Adapted from dipy.viz.utils with added VTK poly-data smoothing.

    Args:
        data: Binary array of shape (X, Y, Z) representing the region of interest.
        affine: 4x4 grid-to-space transformation matrix (RAS 1 mm convention).
            If None the identity matrix is used.
        color: RGB values in [0, 1] with shape (3,).
        opacity: Surface opacity between 0 (transparent) and 1 (opaque).
        smoothing: Number of VTK smoothing iterations. 0 disables smoothing.

    Returns:
        A vtkActor ready to be added to a VTK scene, positioned in world space
        as defined by ``affine``.

    Raises:
        ValueError: If ``data`` is not a 3-D array.
    """
    major_version = vtk.vtkVersion.GetVTKMajorVersion()

    if data.ndim != 3:
        raise ValueError("Only 3D arrays are currently supported.")
    else:
        nb_components = 1

    vol = data.astype("uint8") * 255
    assert data.max() <= 1, np.unique(data)
    im = vtk.vtkImageData()
    if major_version <= 5:
        im.SetScalarTypeToUnsignedChar()  # type: ignore
    di, dj, dk = vol.shape[:3]
    im.SetDimensions(di, dj, dk)
    voxsz = (1.0, 1.0, 1.0)
    # im.SetOrigin(0,0,0)
    im.SetSpacing(voxsz[2], voxsz[0], voxsz[1])
    if major_version <= 5:
        im.AllocateScalars()  # type: ignore
        im.SetNumberOfScalarComponents(nb_components)  # type: ignore
    else:
        im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, nb_components)
    vol = np.swapaxes(vol, 0, 2)
    # vol = np.ascontiguousarray(vol) # already is

    vol = vol.reshape(-1) if nb_components == 1 else np.reshape(vol, [np.prod(vol.shape[:3]), vol.shape[3]])

    uchar_array = numpy_support.numpy_to_vtk(vol, deep=0)
    im.GetPointData().SetScalars(uchar_array)

    if affine is None:
        affine = np.eye(4)

    # Set the transform (identity if none given)
    transform = vtk.vtkTransform()
    transform_matrix = vtk.vtkMatrix4x4()
    transform_matrix.DeepCopy(
        (
            affine[0][0],
            affine[0][1],
            affine[0][2],
            affine[0][3],
            affine[1][0],
            affine[1][1],
            affine[1][2],
            affine[1][3],
            affine[2][0],
            affine[2][1],
            affine[2][2],
            affine[2][3],
            affine[3][0],
            affine[3][1],
            affine[3][2],
            affine[3][3],
        )
    )
    transform.SetMatrix(transform_matrix)  # type: ignore
    transform.Inverse()

    # Set the reslicing
    image_resliced = vtk.vtkImageReslice()
    _set_input(image_resliced, im)
    image_resliced.SetResliceTransform(transform)
    image_resliced.AutoCropOutputOn()

    # Adding this will allow to support anisotropic voxels
    # and also gives the opportunity to slice per voxel coordinates

    rzs = affine[:3, :3]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    image_resliced.SetOutputSpacing(*zooms)

    image_resliced.SetInterpolationModeToLinear()
    image_resliced.Update()

    # skin_extractor = vtk.vtkContourFilter()
    skin_extractor = vtk.vtkMarchingCubes()
    if major_version <= 5:
        skin_extractor.SetInput(image_resliced.GetOutput())  # type: ignore
    else:
        skin_extractor.SetInputData(image_resliced.GetOutput())
    skin_extractor.SetValue(0, 100)

    if smoothing > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(skin_extractor.GetOutputPort())
        smoother.SetNumberOfIterations(smoothing)
        smoother.SetRelaxationFactor(0.1)
        smoother.SetFeatureAngle(60)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOff()
        smoother.SetConvergence(0)
        smoother.Update()

    skin_normals = vtk.vtkPolyDataNormals()
    if smoothing > 0:
        skin_normals.SetInputConnection(smoother.GetOutputPort())
    else:
        skin_normals.SetInputConnection(skin_extractor.GetOutputPort())
    skin_normals.SetFeatureAngle(60.0)

    skin_mapper = vtk.vtkPolyDataMapper()
    skin_mapper.SetInputConnection(skin_normals.GetOutputPort())
    skin_mapper.ScalarVisibilityOff()

    skin_actor = vtk.vtkActor()
    skin_actor.SetMapper(skin_mapper)
    skin_actor.GetProperty().SetOpacity(opacity) if opacity != 1 else None
    skin_actor.GetProperty().SetColor(color[0], color[1], color[2])

    return skin_actor
