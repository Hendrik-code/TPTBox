from __future__ import annotations

import shutil
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


def secure_save(func):
    """
    A decorator that ensures a safe file-saving process by creating a backup of the target file before
    overwriting it and restoring the backup if an error occurs during the save operation.

    The decorator wraps a function that takes a file path as one of its arguments and adds the following safety steps:
    1. If the target file exists, create a backup with a `.backup` suffix.
    2. Call the wrapped function to perform the file-saving operation.
    3. If the save is successful, delete the backup.
    4. If an error occurs, restore the backup and clean up any partially written files.

    Args:
        func (callable): The function to be wrapped. It should take a file path (`str`, `Path`, or `bids_files.BIDS_FILE`)
                         as one of its arguments.

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
            for file_type in _supported_img_files:
                if file_type in file.file:
                    file = file.file[file_type]
                    break
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


def _resample_from_to(
    from_img: NII,
    to_img: tuple[SHAPE, AFFINE, ZOOMS] | Has_Grid,
    order=3,
    mode: MODES = "nearest",
    align_corners: bool | Sentinel = Sentinel(),  # noqa: B008
):
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

    data = scipy_img.affine_transform(from_img.get_array(), rzs, trans, to_shape, order=order, mode=mode, cval=from_img.get_c_val())  # type: ignore
    return data, to_affine, from_img.header
