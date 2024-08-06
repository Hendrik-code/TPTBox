# Source: https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/preview.py


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
    smoothing=20,
    resolution: float | None = None,
    width_factor=1.0,
    verbose=True,
    crop=True,
) -> Image.Image:
    """
    Generate a 3D snapshot from a medical image and save it to the specified output path.

    Parameters:
    ----------
    img : Image_Reference
        The medical image reference from which to generate the snapshot.

    output_path : str | Path | None
        The file path where the snapshot will be saved.

    view : VIEW | list[VIEW], optional
        The orientation(s) for the snapshot. Default is "A" (Axial).
        Can be a single orientation or a list of orientations.

    ids_list : list[Sequence[int]] | None, optional
        A list of lists containing the IDs of the structures to be included in the snapshot.
        If None, all unique IDs in the image will be used. Default is None.

    smoothing : int, optional
        The smoothing factor to apply to the structures. Default is 20.

    resolution : float | None, optional
        The resolution to which the image should be rescaled. If None, the minimum zoom level of the image is used. Default is None.

    width_factor : float, optional
        A factor to adjust the width of the final snapshot. Default is 1.0.

    Returns:
    -------
    None
        The function saves the generated snapshot to the specified output path.
    """
    is_tmp = output_path is None
    if output_path is None:
        t = NamedTemporaryFile(suffix="_snap3D.png")
        output_path = str(t.name)
    nii = to_nii_seg(img)
    if crop:
        nii.apply_crop_(nii.compute_crop())
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
    nii = nii.reorient(("A", "S", "L")).rescale_((resolution, resolution, resolution))
    width = int(max(nii.shape[0], nii.shape[2]) * width_factor)
    window_size = (width * len(ids_list), nii.shape[1])
    with Xvfb():
        scene = window.Scene()
        show_m = window.ShowManager(scene, size=window_size, reset_camera=False)
        show_m.initialize()
        for i, ids in enumerate(ids_list):
            x = width * i
            _plot_sub_seg(scene, nii.extract_label(ids, keep_label=True), x, 0, smoothing, view[i % len(view)])
        scene.projection(proj_type="parallel")
        scene.reset_camera_tight(margin_factor=1.02)
        window.record(scene, size=window_size, out_path=output_path, reset_camera=False)
        scene.clear()
    if not is_tmp:
        logger.on_save("Save Snapshot3D:", output_path, verbose=verbose)
    return Image.open(output_path)


def make_sub_snapshot_parallel(
    output_paths: list[Path],
    imgs: list[Path],
    orientation: VIEW | list[VIEW] = "A",
    ids_list: list[Sequence[int]] | None = None,
    smoothing=20,
    resolution=2,
    cpus=10,
):
    with Pool(cpus) as p:  # type: ignore
        for out_path, img in zip(output_paths, imgs, strict=True):
            p.apply_async(
                make_snapshot3D,
                kwds={
                    "output_path": out_path,
                    "img": img,
                    "orientation": orientation,
                    "ids_list": ids_list,
                    "smoothing": smoothing,
                    "resolution": resolution,
                },
            )
    p.close()
    p.join()


def _plot_sub_seg(scene: window.Scene, nii: NII, x, y, smoothing, orientation: VIEW):
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
        color = get_color_by_label(idx)
        cont_actor = _plot_mask(nii.extract_label(idx), affine, x, y, smoothing=smoothing, color=color, opacity=1)
        scene.add(cont_actor)


def _plot_mask(nii: NII, affine, x_current, y_current, smoothing=10, color: list | np.ndarray = _red, opacity=1):
    mask = nii.get_seg_array()
    cont_actor = _contour_from_roi_smooth(mask, affine=affine, color=color, opacity=opacity, smoothing=smoothing)
    cont_actor.SetPosition(x_current, y_current, 0)
    return cont_actor


def _set_input(
    vtk_object: vtk.vtkImageReslice | vtk.vtkPolyData | vtk.vtkImageData | vtk.vtkAlgorithmOutput,
    inp: vtk.vtkImageData | vtk.vtkAlgorithmOutput,
):
    """Set Generic input function which takes into account VTK 5 or 6.
    This function is copied from dipy.viz.utils
    """
    if isinstance(inp, (vtk.vtkPolyData, vtk.vtkImageData)):
        vtk_object.SetInputData(inp)  # type: ignore
    elif isinstance(inp, vtk.vtkAlgorithmOutput):
        vtk_object.SetInputConnection(inp)  # type: ignore
    vtk_object.Update()  # type: ignore
    return vtk_object


def _contour_from_roi_smooth(data, affine=None, color: np.ndarray | list = _red, opacity=1, smoothing=0):
    """Generates surface actor from a binary ROI.
    Code from dipy, but added awesome smoothing!

    Parameters
    ----------
    data : array, shape (X, Y, Z)
        An ROI file that will be binarized and displayed.
    affine : array, shape (4, 4)
        Grid to space (usually RAS 1mm) transformation matrix. Default is None.
        If None then the identity matrix is used.
    color : (1, 3) ndarray
        RGB values in [0,1].
    opacity : float
        Opacity of surface between 0 and 1.
    smoothing: int
        Smoothing factor e.g. 10.
    Returns
    -------
    contour_assembly : vtkAssembly
        ROI surface object displayed in space
        coordinates as calculated by the affine parameter.

    """
    major_version = vtk.vtkVersion.GetVTKMajorVersion()

    if data.ndim != 3:
        raise ValueError("Only 3D arrays are currently supported.")
    else:
        nb_components = 1

    data = (data > 0) * 1
    vol = np.interp(data, xp=[data.min(), data.max()], fp=[0, 255])
    vol = vol.astype("uint8")

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

    # copy data
    vol = np.swapaxes(vol, 0, 2)
    vol = np.ascontiguousarray(vol)

    vol = vol.ravel() if nb_components == 1 else np.reshape(vol, [np.prod(vol.shape[:3]), vol.shape[3]])

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
    skin_actor.GetProperty().SetOpacity(opacity)
    skin_actor.GetProperty().SetColor(color[0], color[1], color[2])

    return skin_actor
