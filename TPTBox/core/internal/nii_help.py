from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel.processing as nip
import numpy as np

from TPTBox.core import bids_files

if TYPE_CHECKING:
    from TPTBox.core.nii_poi_abstract import Has_Grid
    from TPTBox.core.nii_wrapper import NII

from TPTBox.core.vert_constants import AFFINE, MODES, SHAPE, ZOOMS, Sentinel, _supported_img_files

# NIfTI-1 headers can encode most numpy dtypes, but a couple of common ones
# don't have a NIfTI datatype code (notably float16 and bool). The NII wrapper
# still wants to *carry* an array in one of those dtypes without crashing -
# e.g. an nnU-Net pre-processing step producing a float16 volume, or a
# boolean mask. We only upcast when the array actually has to enter a
# Nifti1Image (i.e. writing to disk or handing the underlying nibabel object
# out), so the caller-visible ``_arr.dtype`` stays whatever they set.
_NIFTI_UNSUPPORTED_DTYPE_UPCAST: dict[np.dtype, np.dtype] = {
    np.dtype(np.float16): np.dtype(np.float32),
    np.dtype(np.bool_): np.dtype(np.uint8),
}


def _nifti_safe_dtype(dtype: np.dtype | type) -> np.dtype:
    """Return the closest nibabel-supported dtype for storing in a Nifti1 header.

    For a dtype that NIfTI-1 already accepts, this is the identity. For the
    unsupported cases we upcast conservatively (``float16 → float32``,
    ``bool → uint8``). The array itself is *not* touched – see
    :func:`_arr_for_nifti1`.
    """
    d = np.dtype(dtype)
    return _NIFTI_UNSUPPORTED_DTYPE_UPCAST.get(d, d)


def _arr_for_nifti1(arr: np.ndarray) -> np.ndarray:
    """Return *arr* (or a copy in a safe dtype) fit to be passed to Nifti1Image."""
    safe = _nifti_safe_dtype(arr.dtype)
    if safe == arr.dtype:
        return arr
    return arr.astype(safe, copy=False)


def secure_save(func, *, file_types=tuple(_supported_img_files)) -> Callable:
    """Decorator that writes to a `.backup` file first and restores it if saving fails.

    Steps: (1) back up existing file, (2) call the wrapped save function, (3) delete backup on
    success, or (4) restore backup and clean up on error.

    Args:
        func (callable): The function to be wrapped. It should take a file path (`str`, `Path`, or `bids_files.BIDS_FILE`)
                         as one of its arguments.
        file_types (tuple[str, ...], keyword-only): File-extension keys tried in order when a ``bids_files.BIDS_FILE`` is
            passed in place of a path. Defaults to ``tuple(_supported_img_files)``.

    Returns:
        callable: The wrapped function with added safety mechanisms.

    Example Usage:
        @secure_save
        def save_to_file(self, file: Path, data: Any):
            # Logic to write data to the file
            ...

    Notes:
        - The decorator supports file paths as strings, `Path` objects, or `bids_files.BIDS_FILE` objects.
        - If a `bids_files.BIDS_FILE` object is provided, the decorator will extract the appropriate file path
          based on supported image file types.

    Raises:
        Exception: Propagates any exception raised by the wrapped function after handling backups appropriately.
    """

    @wraps(func)
    def wrapper(self, file: str | Path | bids_files.BIDS_FILE, *args, **kwargs):
        if isinstance(file, bids_files.BIDS_FILE):
            for file_type in file_types:
                if file_type in file.file:
                    file = file.file[file_type]
                    break
                else:
                    raise ValueError(f"No supported file type found in BIDS_FILE. Expected one of: {file_types}")

        file = Path(file) if isinstance(file, str) else file  # Ensure the file is a Path object
        backup_file = file.with_suffix(file.suffix + ".backup")
        file_existed = file.exists()

        try:
            # Step 1: Check if the file exists
            if file_existed:
                # logging.info(f"Backup created for existing file: {file}")
                shutil.move(file, backup_file)
            # Call the original function
            func(self, file, *args, **kwargs)
            # Step 3a: Delete the backup if there was no error
            if backup_file.exists():
                backup_file.unlink()

        except Exception:
            # logging.exception(f"Error during saving file: {e}")

            # Step 3b: Handle errors
            if file.exists():
                file.unlink()  # Delete the partially written file
            if file_existed and backup_file.exists():
                shutil.move(backup_file, file)  # Restore the backup

            raise

    return wrapper


def _convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj.absolute())
    raise TypeError(type(obj))


def _save_json(data, filepath: str | Path | bids_files.BIDS_FILE, indent=4, convert=_convert):

    if isinstance(filepath, bids_files.BIDS_FILE):
        if "json" in filepath.file:
            filepath = filepath.file["json"]
        else:
            nf = filepath.get_nii_file()
            if nf is not None:
                filepath = (nf.parent) / (nf.name.split(".")[0] + ".json")
            else:
                nf = next(iter(filepath.file.values()))
                filepath = (nf.parent) / (nf.name.rsplit(".", maxsplit=1)[0] + ".json")
    # print(markups[-1].get("display"))
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent, default=convert)


def save_json(filepath: str | Path | bids_files.BIDS_FILE, data, indent=4, convert=_convert) -> None:
    """Safely save a Python object as a JSON file with automatic backup protection.

    This function writes JSON data to disk using a safe save mechanism:
    if the target file already exists, it is first moved to a `.backup`
    file. If writing succeeds, the backup is removed. If writing fails,
    the original file is restored.

    The function supports flexible input types for the target path:
        - str or Path: written directly to disk
        - bids_files.BIDS_FILE: resolved to an appropriate `.json` path

    Non-JSON-serializable types are handled via a custom converter
    that supports:
        - numpy integers → int
        - numpy floats → float
        - numpy arrays → list
        - pathlib.Path → absolute string path

    Args:
        filepath (str | Path | bids_files.BIDS_FILE):
            Target file path or BIDS file container.
        data (Any):
            Python object to serialize into JSON.
        indent (int, optional):
            Pretty-print indentation level. Default is 4.
        convert (callable, optional):
            Custom serialization function for unsupported types.
            Defaults to `_convert`.

    Returns:
        None

    Notes:
        - Uses `secure_save` to ensure atomic write semantics with backup/restore.
        - If `filepath` is a `BIDS_FILE`, the `.json` path is inferred from:
            1. explicit "json" entry in the file map
            2. associated NIfTI file path
            3. fallback to any available file in the container
        - This function is intended for structured metadata and annotation storage.

    Raises:
        Exception:
            Propagates any error raised during serialization or file writing,
            after attempting automatic recovery of the original file.
    """
    return secure_save(_save_json, file_types=["json"])(data, filepath, indent=indent, convert=convert)


def _resample_from_to(
    from_img: NII,
    to_img: tuple[SHAPE, AFFINE, ZOOMS] | Has_Grid,
    order: int = 3,
    mode: MODES = "nearest",
    align_corners: bool | Sentinel = Sentinel(),  # noqa: B008
    out_dtype: np.dtype | type | str | None = None,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Resample *from_img* into the voxel space defined by *to_img*.

    Implements an optional ``align_corners`` mode (analogous to PyTorch) for
    order-0 (nearest-neighbour) interpolation by adjusting both affines so that
    voxel corners rather than voxel centres are aligned.

    Args:
        from_img: Source NII image to be resampled.
        to_img: Target space, given either as a ``(shape, affine, zooms)`` tuple
            or as any object that implements the ``Has_Grid`` interface (providing
            ``shape_int``, ``affine``, and ``zoom`` attributes).
        order: Spline interpolation order (0 = nearest, 1 = linear, 3 = cubic).
        mode: Border handling mode passed to ``scipy.ndimage.affine_transform``
            (e.g. ``"nearest"``, ``"constant"``).
        align_corners: When ``True`` (or when left as a ``Sentinel`` and
            ``order == 0``), voxel corners are aligned between source and target
            grids.  When ``False``, voxel centres are aligned (standard
            nibabel/scipy behaviour).
        out_dtype: Optional NumPy dtype (or dtype-like) requested for the
            resampled array. When set, this is forwarded to ``scipy.ndimage``
            as its ``output=`` argument, so the cast happens in-place during
            interpolation without an extra full-volume copy afterwards.
            ``None`` (default) keeps the source dtype.
            With ``order > 0`` and an integer target dtype the float→int cast
            is a plain truncation (matches NumPy's cast rules); pass ``order=0``
            for label maps. ``scipy.ndimage.affine_transform`` only accepts
            ``uint8/uint16/int16/int32/float32/float64`` for its ``output=``
            argument. Requesting an unsupported dtype (notably ``float16``)
            transparently falls back to resampling into ``float32`` and casting
            to the requested dtype in a single extra pass - still much cheaper
            than staying in the source dtype (e.g. float64) throughout.

    Returns:
        A 3-tuple ``(data, affine, header)`` where *data* is the resampled
        NumPy array, *affine* is the target affine matrix, and *header* is the
        NIfTI header taken from *from_img*.

    Raises:
        AffineError: If *from_img* has fewer than 3 spatial dimensions.
        ValueError: If *from_img* does not have spatial axes first.
    """
    import numpy.linalg as npl
    import scipy.ndimage as scipy_img
    from nibabel.affines import AffineError, to_matvec
    from nibabel.imageclasses import spatial_axes_first

    # This check requires `shape` attribute of image
    if not spatial_axes_first(from_img.nii):
        raise ValueError(f"Cannot predict position of spatial axes for Image type {type(from_img)}")
    if isinstance(to_img, tuple):
        to_shape, to_affine, zoom_to = to_img
    else:
        assert to_img.affine is not None
        assert to_img.zoom is not None
        to_shape: SHAPE = to_img.shape_int
        to_affine: AFFINE = to_img.affine
        zoom_to = np.array(to_img.zoom)
    from_n_dim = len(from_img.shape)
    if from_n_dim < 3:
        raise AffineError("from_img must be at least 3D")
    if (isinstance(align_corners, Sentinel) and order == 0) or align_corners:
        # https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/6
        # https://discuss.pytorch.org/uploads/default/original/2X/6/6a242715685b8192f07c93a57a1d053b8add97bf.png
        # Simulate align_corner=True, by manipulating the affine
        # Updated to matrix:
        # make the output by one voxel larger
        # z_new = z * num_pixel/(num_pixel+1)
        to_affine_new = to_affine.copy()
        num_pixel = np.array(to_shape)
        zoom_new = zoom_to * num_pixel / (1 + num_pixel)
        rotation_zoom = to_affine[:3, :3]
        to_affine_new[:3, :3] = rotation_zoom / np.array(zoom_to) * zoom_new
        ## Shift origin to corner
        corner = np.array([-0.5, -0.5, -0.5, 0])
        to_affine_new[:, 3] -= to_affine_new @ corner
        # Update from matrix
        # z_new = z * num_pixel/(num_pixel+1)
        zoom_from = np.array(from_img.zoom)
        from_affine_new = from_img.affine.copy()
        num_pixel = np.array(from_img.shape)
        zoom_new = zoom_from * num_pixel / (1 + num_pixel)
        rotation_zoom = from_img.affine[:3, :3]
        from_affine_new[:3, :3] = rotation_zoom / np.array(zoom_from) * zoom_new
        ## Shift origin to corner
        from_affine_new[:, 3] -= from_affine_new @ corner

        a_to_affine = nip.adapt_affine(to_affine_new, len(to_shape))
        a_from_affine = nip.adapt_affine(from_affine_new, from_n_dim)
    else:
        a_to_affine = nip.adapt_affine(to_affine, len(to_shape))
        a_from_affine = nip.adapt_affine(from_img.affine, from_n_dim)
    to_vox2from_vox = npl.inv(a_from_affine).dot(a_to_affine)
    rzs, trans = to_matvec(to_vox2from_vox)

    # scipy.ndimage.affine_transform can only write into a small set of dtypes
    # (u8/u16/i16/i32/f32/f64). For anything else - notably float16, which the
    # nnU-Net inference path wants for memory - we resample into float32 first
    # and cast in one pass at the end.
    _scipy_supported = (np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64)
    if out_dtype is None:
        scipy_out = None
        post_cast: np.dtype | None = None
    else:
        req = np.dtype(out_dtype)
        if req.type in _scipy_supported:
            scipy_out = req
            post_cast = None
        else:
            scipy_out = np.dtype(np.float32)
            post_cast = req
    data = scipy_img.affine_transform(  # type: ignore
        from_img.get_array(),
        rzs,
        trans,
        to_shape,
        order=order,
        mode=mode,
        cval=from_img.get_c_val(),
        output=scipy_out,
    )
    if post_cast is not None:
        data = data.astype(post_cast, copy=False)
    return data, to_affine, from_img.header


def _add_grid_info_to_json(nii_path: Path | str, simp_json: Path | str, force_update: bool = False, add: bool = True) -> dict:
    """Append grid metadata (shape, spacing, orientation, affine) to a sidecar JSON file.

    This lives here rather than next to the DICOM converters because it needs no
    DICOM library at all - only ``NII`` and the JSON helpers. It used to sit in
    ``TPTBox.core.dicom.dicom_extract``, which made the fully public
    ``BIDS_FILE.get_grid_info()`` drag in ``pydicom`` and ``dicom2nifti`` for
    users who never touched DICOM data.

    Args:
        nii_path: Path to the NIfTI file from which grid info is read.
        simp_json: Path to the JSON sidecar file to update.
        force_update: Re-compute and overwrite existing grid info when ``True``.
        add: Write the updated dictionary back to disk when ``True``.

    Returns:
        The updated JSON dictionary including the ``"grid"`` key.
    """
    from datetime import datetime

    from TPTBox.core.nii_wrapper import NII

    nii_path = Path(nii_path)
    simp_json = Path(simp_json)

    # Always preserve the existing JSON contents (DICOM metadata written by save_json).
    # The mtime comparison is only used to short-circuit re-computing the grid when the
    # sidecar is already up to date; it must NOT decide whether to keep the DICOM keys.
    json_dict: dict = {}
    if simp_json.exists():
        with open(simp_json, encoding="utf-8") as f:
            json_dict = json.load(f)
    json_up_to_date = (
        simp_json.exists()
        and nii_path.exists()
        and datetime.fromtimestamp(simp_json.stat().st_mtime) > datetime.fromtimestamp(nii_path.stat().st_mtime)
    )
    if "grid" in json_dict and not force_update and json_up_to_date:
        return json_dict
    nii = NII.load(nii_path, False)
    json_dict["grid"] = {
        "shape": nii.shape,
        "spacing": nii.spacing,
        "orientation": nii.orientation,
        "rotation": nii.rotation.reshape(-1).tolist(),
        "origin": nii.origin,
        "dims": nii.get_num_dims(),
    }
    # Matches the previous `save_json(..., override=add)` semantics: write when
    # `add` is set, or whenever the sidecar does not exist yet.
    if add or not simp_json.exists():
        from TPTBox.logger import Print_Logger

        Print_Logger().on_save("save json with grid info", simp_json)
        save_json(simp_json, json_dict, indent=4)
    return json_dict
