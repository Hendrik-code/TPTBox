from __future__ import annotations

import math
import traceback
import warnings
import zlib
from collections.abc import Sequence
from enum import Enum
from math import ceil, floor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
from nibabel import Nifti1Header, Nifti1Image  # type: ignore
from typing_extensions import Self

from TPTBox.core import bids_files
from TPTBox.core.compat import zip_strict
from TPTBox.core.internal.nii_help import _resample_from_to, secure_save
from TPTBox.core.nii_poi_abstract import Has_Grid
from TPTBox.core.nii_wrapper_math import NII_Math
from TPTBox.core.np_utils import (
    _pad_to_parameters,
    np_bbox_binary,
    np_calc_boundary_mask,
    np_calc_convex_hull,
    np_calc_overlapping_labels,
    np_center_of_mass,
    np_compute_surface,
    np_connected_components,
    np_connected_components_per_label,
    np_dilate_msk,
    np_erode_msk,
    np_extract_label,
    np_fill_holes,
    np_fill_holes_global_with_majority_voting,
    np_filter_connected_components,
    np_get_connected_components_center_of_mass,
    np_is_empty,
    np_map_labels,
    np_map_labels_based_on_majority_label_mask_overlap,
    np_point_coordinates,
    np_smooth_gaussian_labelwise,
    np_translate_arr,
    np_unique,
    np_unique_withoutzero,
    np_volume,
)
from TPTBox.core.vert_constants import (
    AFFINE,
    AX_CODES,
    COORDINATE,
    DIRECTIONS,
    LABEL_MAP,
    LABEL_REFERENCE,
    MODES,
    SHAPE,
    ZOOMS,
    Location,
    _same_direction,
    log,
    logging,
    v_name2idx,
)
from TPTBox.logger.log_file import Log_Type

if TYPE_CHECKING:
    from torch import device
MODES = Literal["constant", "nearest", "reflect", "wrap"]
_unpacked_nii = tuple[np.ndarray, AFFINE, nib.nifti1.Nifti1Header]
_formatwarning = warnings.formatwarning


def formatwarning_tb(*args, **kwargs):
    s = "####################################\n"
    s += _formatwarning(*args, **kwargs)
    tb = traceback.format_stack()[:-3]
    s += "".join(tb[:-1])
    s += "####################################\n"
    return s


_dtype_max = {
    "int8": 128,
    "uint8": 256,
    "int16": 32768,
    "uint16": 65536,
    "int32": 2147483647,
    "uint32": 4294967294,
}  # uint32 not supported by nifty
_dtype_u = {"uint8", "uint16"}
_dtype_non_u = {"int8", "int16"}


def _check_if_nifty_is_lying_about_its_dtype(self: NII):
    change_dtype = False
    arr = self.nii.dataobj
    dtype = self._nii.dataobj.dtype  # type: ignore
    dtype_s = str(self._nii.dataobj.dtype)
    try:
        mi = np.min(arr)
    except Exception:
        return np.float32
    ma = np.max(arr)
    has_neg = mi < 0
    max_v = _dtype_max.get(str(dtype), 0)
    positive = str(dtype) in _dtype_u

    if has_neg and positive:
        warnings.warn(
            f"Loaded NIfTY: incorrect dtype detected: {dtype}, but has negative values; min={mi}",
            stacklevel=3,
        )
        change_dtype = True
        positive = False
    if not has_neg and self.seg:
        positive = True
        if str(dtype) in _dtype_non_u:
            change_dtype = True
    if ma > max_v and "float" not in dtype_s:
        change_dtype = True
        warnings.warn(
            f"Loaded NIfTY: incorrect dtype detected: {dtype}, but has larger max values; max={max_v}",
            stacklevel=3,
        )

    out_dtype = dtype
    if dtype == np.float16:
        warnings.warn(
            f"Loaded NIfTY: incorrect dtype detected: {dtype} is not supported",
            stacklevel=3,
        )
        out_dtype = np.float32
    if "float" in dtype_s and not change_dtype:
        pass
    elif positive and change_dtype:
        if ma < 256:
            out_dtype = np.uint8
        elif ma < 65536:
            out_dtype = np.uint16
        else:
            out_dtype = np.int32
    elif change_dtype:
        ma_abs = np.max(np.abs(arr))
        if ma_abs < 128:
            out_dtype = np.int8
        elif ma_abs < 32768:
            out_dtype = np.int16
        else:
            out_dtype = np.int32
    # print("check", out_dtype, change_dtype, positive, has_neg)
    return out_dtype


warnings.formatwarning = formatwarning_tb

N = TypeVar("N", bound="NII")
Image_Reference = Union[bids_files.BIDS_FILE, Nifti1Image, Path, str, N]
Interpolateable_Image_Reference = Union[
    bids_files.BIDS_FILE,
    tuple[Nifti1Image, bool],
    tuple[Path, bool],
    tuple[str, bool],
    N,
]

Proxy = tuple[tuple[int, int, int], np.ndarray]
suppress_dtype_change_printout_in_set_array = False
# fmt: off

class NII(NII_Math):
    """
    The `NII` class represents a NIfTI image and provides various methods for manipulating and analyzing NIfTI images. It supports loading and saving NIfTI images, rescaling and reorienting images, applying operations on segmentation masks, and more.

    Example Usage:
    ```python
    # Create an instance of NII class
    nii = NII(nib.load('image.nii.gz'),seg=False)

    # Get the shape of the image
    shape = nii.shape

    # Rescale the image to a new voxel spacing
    rescaled = nii.rescale(voxel_spacing=(1, 1, 1))

    # Reorient the image to a new orientation
    reoriented = nii.reorient(axcodes_to=("P", "I", "R"))

    # Apply a segmentation mask to the image
    masked = nii.apply_mask(mask)

    # Save the image to a new file
    nii.save('output.nii.gz')
    ```

    Main functionalities:
    - Loading and saving NIfTI images
    - Rescaling and reorienting images
    - Applying operations on segmentation masks

    Methods:
    - `load`: Loads a NIfTI image from a file
    - `load_bids`: Loads a NIfTI image from a BIDS file
    - `shape`: Returns the shape of the image
    - `dtype`: Returns the data type of the image
    - `header`: Returns the header of the image
    - `affine`: Returns the affine transformation matrix of the image
    - `orientation`: Returns the orientation of the image
    - `zoom`: Returns the voxel sizes of the image
    - `origin`: Returns the origin of the image
    - `rotation`: Returns the rotation matrix of the image
    - `reorient`: Reorients the image to a new orientation
    - `rescale`: Rescales the image to a new voxel spacing
    - `apply_mask`: Applies a segmentation mask to the image
    - `save`: Saves the image to a file

    Fields:
    - `nii`: The NIfTI image object
    - `seg`: A flag indicating whether the image is a segmentation mask
    - `c_val`: The default value for the segmentation mask

    Note: This class assumes that the input NIfTI images are in the NIfTI-1 format.
    """
    def __init__(self, nii: Nifti1Image|_unpacked_nii, seg=False,c_val=None, desc:str|None=None, info=None) -> None:
        assert nii is not None
        self.__divergent = False
        self._checked_dtype = False
        self.nii = nii
        self.seg:bool = seg
        self.c_val:float|None=c_val # default c_vale if seg is None
        self.__min = None
        self.info = info if info is not None else {}
        self.set_description(desc)
        if seg:
            self._unpack()
            if isinstance(self.dtype,np.floating):
                self.set_dtype_("smallest_uint")


    @classmethod
    def load(cls, path: Image_Reference, seg, c_val=None)-> Self:
        nii= to_nii(path,seg)
        if seg:
            nii = nii.set_dtype("smallest_uint")
        nii.c_val = c_val
        return nii

    @classmethod
    def load_nrrd(cls, path: str | Path, seg: bool,verbos=False):
        """
        Load an NRRD file and convert it into a Nifti1Image object.

        Args:
            path (str | Path): The file path to the NRRD file to be loaded.
            seg (bool): A flag indicating if the data represents segmentation data.

        Returns:
            NII: An NII object containing the loaded Nifti1Image and the segmentation flag.

        Raises:
            ImportError: If the `pynrrd` package is not installed.
            FileNotFoundError: If the specified NRRD file cannot be found.

        Example:
            >>> nii = cls.load_nrrd("example.nrrd", seg=True)
            >>> print(nii)
        """

        try:
            import nrrd  # pip install pynrrd, if pynrrd is not already installed
        except ModuleNotFoundError:
            raise ImportError("The `pynrrd` package is required but not installed. Install it with `pip install pynrrd`.") from None
        from TPTBox.core.internal.slicer_nrrd import load_slicer_nrrd
        return load_slicer_nrrd(path,seg,verbos=verbos)
    @classmethod
    def load_bids(cls, nii_bids: bids_files.BIDS_FILE):
        nifty = None
        if "nii" in nii_bids.file:
            path = nii_bids.file['nii']
            nifty = nib.load(path)
        elif "nii.gz" in nii_bids.file:
            path = nii_bids.file['nii.gz']
            nifty = nib.load(path)
        elif "nrrd" in nii_bids.file:
            path = nii_bids.file['nrrd']
            nifty = NII.load_nrrd(path,seg=False)
        else:
            import SimpleITK as sitk  # noqa: N813

            from TPTBox.core.sitk_utils import sitk_to_nib
            for f in nii_bids.file:
                try:
                    img = sitk.ReadImage(nii_bids.file[f])
                    nifty =  sitk_to_nib(img)
                except Exception:
                    pass
                break
        if nii_bids.get_interpolation_order() == 0:
            seg = True
            c_val=0
        else:
            seg = False
            c_val = -1024 if "ct" in nii_bids.format.lower() else 0
        assert nifty is not None, f"could not find {nii_bids}"
        return NII(nifty,seg,c_val) # type: ignore
    def _unpack(self):
        try:
            if self.__unpacked:
                return
            if not self._checked_dtype or self.seg:
                dtype = _check_if_nifty_is_lying_about_its_dtype(self)
                #print("unpack-nii",f"{self.seg=}",dtype)
                self._checked_dtype = True
                self._arr = np.asanyarray(self.nii.dataobj, dtype=dtype).copy()
            else:
                self._arr = np.asanyarray(self.nii.dataobj, dtype=self.nii.dataobj.dtype).copy() #type: ignore

            self._aff = self.nii.affine
            self._header:Nifti1Header = self.nii.header # type: ignore
            self.__unpacked = True
        except EOFError as e:
            raise EOFError(f"{self.nii.get_filename()}: {e!s}\nThe file is probably brocken beyond repair, due killing a software during nifty saving.") from None
        except zlib.error as e:
            raise EOFError(f"{self.nii.get_filename()}: {e!s}\nThe file is probably brocken beyond repair, due killing a software during nifty saving.") from None
        except OSError as e:
            raise EOFError(f"{self.nii.get_filename()}: {e!s}\nThe file is probably brocken beyond repair, due killing a software during nifty saving.") from None
    @property
    def nii_abstract(self) -> Nifti1Image|_unpacked_nii:
        if self.__unpacked:
            return self._arr,self.affine,self.header
        return self._nii

    @property
    def nii(self) -> Nifti1Image:
        if self.__divergent:
            self._nii = Nifti1Image(self._arr,self.affine,self.header)
            if self.dtype == self._arr.dtype: #type: ignore
                nii = Nifti1Image(self._arr,self.affine,self.header)
            else:
                if not suppress_dtype_change_printout_in_set_array:
                    log.print(f"'set_array' with different dtype: from {self.dtype} to {self._arr.dtype}",verbose=True) #type: ignore
                nii2 = Nifti1Image(self._arr,self.affine,self.header)
                nii2.set_data_dtype(self._arr.dtype)
                nii = Nifti1Image(self._arr,nii2.affine,nii2.header) # type: ignore
            if all(a is None for a in self.header.get_slope_inter()):
                nii.header.set_slope_inter(1,self.get_c_val()) # type: ignore
            #if self.header is not None:
            #    self.header.set_sform(self.affine, code=1)

            self._nii = nii
            self.__divergent = False
        return self._nii
    @nii.setter
    def nii(self,nii:Nifti1Image|_unpacked_nii):
        if isinstance(nii,tuple):
            assert len(nii) == 3, nii
            self.__divergent = True
            self.__unpacked = True
            arr, aff, header = nii
            n = aff.shape[0]-1
            if header is None or n != header['dim'][0]:
                header = None
            if len(arr.shape) != n:
                # is there a dimesion with size 1?
                arr = arr.squeeze()
                # TODO try to get back to a saveabel state, if this did not work
            if arr.dtype == np.uint64:#throws error
                arr = arr.astype(np.uint32)
            if arr.dtype == np.int64:#throws error
                arr = arr.astype(np.int32)
            self._arr = arr
            self._aff = aff
            self._checked_dtype = True
            if header is not None:
                header = header.copy()
                header.set_sform(aff, code='aligned')
                header.set_qform(aff, code='unknown')
                header.set_data_dtype(arr.dtype)
                rotation_zoom = aff[:n, :n]
                zoom = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
                #print(aff.shape,arr.shape,zoom)
                header.set_zooms(zoom)
                self._header = header
                return
            else:
                nii = Nifti1Image(arr,aff)
        self.__unpacked = False
        self.__divergent = False
        self._nii = nii

    @property
    def shape(self) -> tuple[int, int, int]:
        if self.__unpacked:
            return tuple(self._arr.shape) # type: ignore
        return self.nii.shape # type: ignore
    @property
    def dtype(self)->type:
        if self.__unpacked:
            return self._arr.dtype # type: ignore
        return self.nii.dataobj.dtype #type: ignore
    @property
    def header(self) -> Nifti1Header:
        if self.__unpacked:
            return self._header
        return self.nii.header  # type: ignore
    @property
    def affine(self) -> np.ndarray:
        if self.__unpacked:
            return self._aff # type: ignore
        return self.nii.affine # type: ignore

    @affine.setter
    def affine(self,affine:np.ndarray):
        self._unpack()
        self.__divergent = True
        self._aff = affine
    @property
    def orientation(self) -> AX_CODES:
        ort = nio.io_orientation(self.affine)
        return nio.ornt2axcodes(ort) # type: ignore
    @property
    def dims(self)->int:
        self._unpack()
        return self.affine.shape[0]-1
    @property
    def zoom(self) -> ZOOMS:
        n = self.dims
        rotation_zoom = self.affine[:n, :n]
        zoom = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0)) if self.__divergent else self.header.get_zooms()

        z = tuple(np.round(zoom,7))
        if len(z) >= n:
            z = z[:n]
        #assert len(z) == 3,z
        return z # type: ignore

    @property
    def origin(self) -> tuple[float, float, float]:
        n = self.dims
        z = tuple(np.round(self.affine[:n,n],7))
        assert len(z) == 3
        return z # type: ignore
    @origin.setter
    def origin(self,x:tuple[float, float, float]):
        n = self.dims
        self._unpack()
        self.__divergent = True
        affine = self._aff
        affine[:n,n] = np.array(x) # type: ignore
        self._aff = affine
    @property
    def rotation(self)->np.ndarray:
        n = self.dims
        rotation_zoom = self.affine[:n, :n]
        zoom = np.array(self.zoom)
        rotation = rotation_zoom / zoom
        return rotation

    @property
    def direction(self) -> list:
        direction_matrix = self.rotation
        direction_flat = direction_matrix.flatten().tolist()
        return direction_flat
    @property
    def direction_itk(self) -> list:
        a = np.array(self.direction)
        a[:len(a)//3*2]*=-1
        return a.tolist()

    @orientation.setter
    def orientation(self, value: AX_CODES):
        self.reorient_(value, verbose=False)


    def split_4D_image_to_3D(self):
        assert self.get_num_dims() == 4,self.get_num_dims()
        arr_4d = self.get_array()
        out:list[NII] = []
        for i in range(self.shape[-1]):
            arr = arr_4d[...,i]
            out.append(NII(Nifti1Image(arr, self.affine, self.header.copy()),self.seg,self.c_val,self.header['descrip']))
        return out


    @property
    def orientation_ornt(self):
        return nio.io_orientation(self.affine)

    def set_description(self,v:str|None):
        if v is None:
            self.header['descrip'] = "seg" if self.seg else "img"
        else:
            self.header['descrip'] = v

    def get_c_val(self,default=None):
        if self.seg:
            return 0
        if self.c_val is not None:
            return self.c_val
        if default is not None:
            return default
        if self.__min is None:
            self.__min = self.min()
        return self.__min

    def get_seg_array(self) -> np.ndarray:
        if not self.seg:
            warnings.warn(
                "requested a segmentation array, but NII is not set as a segmentation", UserWarning, stacklevel=5
            )
        self._unpack()
        return self._arr.copy() #type: ignore
    def get_array(self) -> np.ndarray:
        if self.seg:
            return self.get_seg_array()
        self._unpack()
        return self._arr.copy()
    def numpy(self, *_args):
        return self.get_array()
    def set_array(self,arr:np.ndarray|Self, inplace=False,verbose:logging=False,seg=None)-> Self:  # noqa: ARG002
        """Creates a NII where the array is replaces with the input array.

        Note: This function works "Out-of-place" by default, like all other methods.

        Args:
            arr (np.ndarray): _description_
            inplace (bool, optional): _description_. Defaults to False.

        Returns:
            self
        """

        if hasattr(arr,"get_array"):
            arr = arr.get_array() # type: ignore
        if arr.dtype == bool:
            arr = arr.astype(np.uint8)
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        if self.seg and isinstance(arr, (np.floating, float)):
            arr = arr.astype(np.int32)
        #if self.dtype == arr.dtype: #type: ignore
        nii:_unpacked_nii = (arr,self.affine,self.header.copy())
        self.header.set_data_dtype(arr.dtype)
        #else:
        #    if not suppress_dtype_change_printout_in_set_array:
        #        log.print(f"'set_array' with different dtype: from {self.nii.dataobj.dtype} to {arr.dtype}",verbose=verbose) #type: ignore
        #    nii2 = Nifti1Image(self.get_array(),self.affine,self.header)
        #    nii2.set_data_dtype(arr.dtype)
        #    nii = (arr,nii2.affine,nii2.header) # type: ignore
        #if all(a is None for a in self.header.get_slope_inter()):
        #    nii.header.set_slope_inter(1,self.get_c_val()) # type: ignore
        if inplace:
            if seg is not None:
                self.seg = seg
            self.nii = nii
            return self
        else:
            return self.copy(nii,seg=seg) # type: ignore

    def set_array_(self,arr:np.ndarray,verbose:logging=True):
        return self.set_array(arr,inplace=True,verbose=verbose)
    def set_dtype(self,dtype:type|Literal['smallest_int','smallest_uint'] = np.float32,order:Literal["C","F","A","K"] ='K',casting:Literal["no","equiv","safe","same_kind","unsafe"] = "unsafe", inplace=False):
        sel = self if inplace else self.copy()
        if dtype == "smallest_uint":
            arr = self.get_array()
            if arr.max()<256:
                dtype = np.uint8
            elif arr.max()<65536:
                dtype = np.uint16
            else:
                dtype = np.int32
        elif dtype == "smallest_int":
            arr = self.get_array()
            if arr.max()<128:
                dtype = np.int8
            elif arr.max()<32768:
                dtype = np.int16
            else:
                dtype = np.int32
        if self.__unpacked:
            self._unpack()
            sel._arr = sel._arr.astype(dtype)
            sel.header.set_data_dtype(dtype)
        else:
            sel.nii.set_data_dtype(dtype)
            if sel.nii.get_data_dtype() != self.dtype: #type: ignore
                sel.nii = Nifti1Image(self.get_array().astype(dtype,casting=casting,order=order),self.affine,self.header)

        return sel
    def set_dtype_(self,dtype:type|Literal['smallest_uint','smallest_int'] = np.float32,order:Literal["C","F","A","K"] ='K',casting:Literal["no","equiv","safe","same_kind","unsafe"] = "unsafe"):
        return self.set_dtype(dtype=dtype,order=order,casting=casting, inplace=True)

    def astype(self,dtype,order:Literal["C","F","A","K"] ='K', casting:Literal["no","equiv","safe","same_kind","unsafe"] = "unsafe",subok=True, copy=True)->Self:
        ''' numpy wrapper '''
        if subok:
            c = self.set_dtype(dtype,order=order,casting=casting, inplace=copy)
            return c
        else:
            return self.get_array().astype(dtype,order=order,casting=casting, subok=subok,copy=copy)
    def reorient(self:Self, axcodes_to: AX_CODES|str|None = ("P", "I", "R"), verbose:logging=False, inplace=False)-> Self:
        """
        Reorients the input Nifti image to the desired orientation, specified by the axis codes.

        Args:
            axcodes_to (tuple): A tuple of three strings representing the desired axis codes. Default value is ("P", "I", "R").
            verbose (bool): If True, prints a message indicating the orientation change. Default value is False.
            inplace (bool): If True, modifies the input image in place. Default value is False.

        Returns:
            If inplace is True, returns None. Otherwise, returns a new instance of the NII class representing the reoriented image.

        Note:
        The nibabel axes codes describe the direction, not the origin, of axes. The direction "PIR+" corresponds to the origin "ASL".
        """
        # Note: nibabel axes codes describe the direction not origin of axes
        # direction PIR+ = origin ASL
        if isinstance(axcodes_to,str):
            axcodes_to = tuple(axcodes_to)
        if axcodes_to is not None:
            aff = self.affine
            ornt_fr = self.orientation_ornt
            arr = self.get_array()
            ornt_to = nio.axcodes2ornt(axcodes_to)
            ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
            if (ornt_fr == ornt_to).all():
                log.print("Image is already rotated to", axcodes_to,verbose=verbose)
                if inplace:
                    return self
                return self.copy() # type: ignore
            arr = nio.apply_orientation(arr, ornt_trans)
            aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
            new_aff = np.matmul(aff, aff_trans)
            ### Reset origin ###
            flip = ornt_trans[:, 1]
            change = ((-flip) + 1) / 2  # 1 if flip else 0
            change = tuple(a * (s-1) for a, s in zip(change, self.shape))
            new_aff[:3, 3] = nib.affines.apply_affine(aff,change) # type: ignore
            ######
            #if self.header is not None:
            #    self.header.set_sform(new_aff, code=1)
            new_img = arr, new_aff,self.header
            log.print("Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to,verbose=verbose)
        else:
            return self if not inplace else self.copy()
        if inplace:
            self.nii = new_img
            return self

        return self.copy(new_img) # type: ignore
    def reorient_(self:Self, axcodes_to: AX_CODES|None = ("P", "I", "R"), verbose:logging=False) -> Self:
        if axcodes_to is None:
            return self
        return self.reorient(axcodes_to=axcodes_to, verbose=verbose,inplace=True)


    def compute_crop(self,minimum: float=0, dist: float = 0, use_mm=False, other_crop:tuple[slice,...]|None=None, maximum_size:tuple[slice,...]|int|tuple[int,...]|None=None, raise_error=True)->tuple[slice,slice,slice]:
        """
        Computes the minimum slice that removes unused space from the image and returns the corresponding slice tuple along with the origin shift required for centroids.

        Args:
            minimum (int): The minimum value of the array (0 for MRI, -1024 for CT). Default value is 0.
            dist (int): The amount of padding to be added to the cropped image. Default value is 0.
            use_mm: dist will be mm instead of number of voxels
            other_crop (tuple[slice,...], optional): A tuple of slice objects representing the slice of an other image to be combined with the current slice. Default value is None.
            raise_error: if crop is empty a "ValueError: bbox_nd: img is empty, cannot calculate a bbox" is produced. When False return None instead.

        Returns:
            ex_slice: A tuple of slice objects that need to be applied to crop the image.
            origin_shift: A tuple of integers representing the shift required to obtain the centroids of the cropped image.

        Note:
            - The computed slice removes the unused space from the image based on the minimum value.
            - The padding is added to the computed slice.
            - If the computed slice reduces the array size to zero, a ValueError is raised.
            - If other_crop is not None, the computed slice is combined with the slice of another image to obtain a common region of interest.
            - Only None slice is supported for combining slices.
        """
        d = np.around(dist / np.asarray(self.zoom)).astype(int) if use_mm else (int(dist),int(dist),int(dist))
        array = self.get_array() #+ minimum

        ex_slice = list(np_bbox_binary(array > minimum, px_dist=d,raise_error=raise_error))

        if other_crop is not None:
            assert all((a.step is None) for a in other_crop), 'Only None slice is supported for combining x'
            ex_slice = [slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(ex_slice, other_crop)]

        if maximum_size is not None:
            if isinstance(maximum_size,int):
                maximum_size = (maximum_size,maximum_size,maximum_size)
            for i, min_w in enumerate(maximum_size):
                if isinstance(min_w,slice):
                    min_w = min_w.stop - min_w.start  # noqa: PLW2901
                curr_w =  ex_slice[i].stop - ex_slice[i].start
                dif = min_w - curr_w
                if min_w > 0 and dif > 0:
                    new_start = ex_slice[i].start - floor(dif/2)
                    new_goal = ex_slice[i].stop + ceil(dif/2)
                    if new_goal > self.shape[i]:
                        new_start -= new_goal - self.shape[i]
                        new_goal = self.shape[i]
                    if new_start < 0:
                        new_goal -= new_start
                        new_start = 0
                    ex_slice[i] = slice(new_start,new_goal)


        #origin_shift = tuple([int(ex_slice[i].start) for i in range(len(ex_slice))])
        return tuple(ex_slice)# type: ignore

    def apply_center_crop(self, center_shape: tuple[int,int,int], inplace=False, verbose: bool = False):
        shp_x, shp_y, shp_z = self.shape
        crop_x, crop_y, crop_z = center_shape
        arr = self.get_array()

        if crop_x > shp_x or crop_y > shp_y or crop_z > shp_z:
            padding_ltrb = [
                ((crop_x - shp_x +1) // 2 if crop_x > shp_x else 0,(crop_x - shp_x) // 2 if crop_x > shp_x else 0),
                ((crop_y - shp_y +1) // 2 if crop_y > shp_y else 0,(crop_y - shp_y) // 2 if crop_y > shp_y else 0),
                ((crop_z - shp_z +1) // 2 if crop_z > shp_z else 0,(crop_z - shp_z) // 2 if crop_z > shp_z else 0),
            ]
            arr_padded = np.pad(arr, padding_ltrb, "constant", constant_values=0)  # PIL uses fill value 0
            log.print(f"Pad from {self.shape} to {arr_padded.shape}", verbose=verbose)
            shp_x, shp_y, shp_z = arr_padded.shape
            if crop_x == shp_x and crop_y == shp_y and crop_z == shp_z:
                return self.set_array(arr_padded)
        else:
            arr_padded = arr

        crop_rel_x = round((shp_x - crop_x) / 2.0)
        crop_rel_y = round((shp_y - crop_y) / 2.0)
        crop_rel_z = round((shp_z - crop_z) / 2.0)

        crop_slices = (slice(crop_rel_x, crop_rel_x + crop_x),slice(crop_rel_y, crop_rel_y + crop_y),slice(crop_rel_z, crop_rel_z + crop_z))
        arr_cropped = arr_padded[crop_slices]
        log.print(f"Center cropped from {arr_padded.shape} to {arr_cropped.shape}", verbose=verbose)
        shp_x, shp_y, shp_z = arr_cropped.shape
        assert crop_x == shp_x and crop_y == shp_y and crop_z == shp_z
        return self.set_array(arr_cropped, inplace=inplace)
        #return self.apply_crop(crop_slices, inplace=inplace)

    def apply_crop_slice(self,*args,**qargs):
        import warnings
        warnings.warn("apply_crop_slice id deprecated use apply_crop instead",stacklevel=5) #TODO remove in version 1.0
        return self.apply_crop(*args,**qargs)

    def apply_crop_slice_(self,*args,**qargs):
        import warnings
        warnings.warn("apply_crop_slice_ id deprecated use apply_crop_ instead",stacklevel=5) #TODO remove in version 1.0
        return self.apply_crop_(*args,**qargs)

    def apply_crop(self,ex_slice:tuple[slice,slice,slice]|Sequence[slice]|None , inplace=False):
        """
        The apply_crop_slice function applies a given slice to reduce the Nifti image volume. If a list of slices is provided, it computes the minimum volume of all slices and applies it.

        Args:
            ex_slice (tuple[slice,slice,slice] | list[tuple[slice,slice,slice]]): A tuple or a list of tuples, where each tuple represents a slice for each axis (x, y, z).
            inplace (bool, optional): If True, it applies the slice to the original image and returns it. If False, it returns a new NII object with the sliced image.
        Returns:
            NII: A new NII object containing the sliced image if inplace=False. Otherwise, it returns the original NII object after applying the slice.
        """
        nii = self.nii.slicer[tuple(ex_slice)] if ex_slice is not None else self.nii_abstract
        if inplace:
            self.nii = nii
            return self
        x= self.copy(nii)
        return x

    def apply_crop_(self,ex_slice:tuple[slice,slice,slice]|Sequence[slice]):
        return self.apply_crop(ex_slice=ex_slice,inplace=True)

    def pad_to(self,target_shape:list[int]|tuple[int,int,int] | Self, mode:MODES="constant",crop=False,inplace = False):
        if isinstance(target_shape, NII):
            target_shape = target_shape.shape
        padding, crop, requires_crop = _pad_to_parameters(self.shape, target_shape)
        s = self
        if crop and requires_crop:
            s = s.apply_crop(tuple(crop),inplace=inplace)
        return s.apply_pad(padding,inplace=inplace,mode=mode)

    def apply_pad(self,padd:Sequence[tuple[int|None,int]],mode:MODES="constant",inplace = False,verbose:logging=True):
        #TODO add other modes
        #TODO add testcases and options for modes
        transform = np.eye(self.dims+1, dtype=int)
        assert len(padd) == self.dims
        for i, (before,_) in enumerate(padd):
            #transform[i, i] = pad_slice.step if pad_slice.step is not None else 1
            transform[i, 3] = -before  if before is not None else 0

        while len(padd) < len(self.shape):
            padd = (*tuple(padd), (0, 0))
        affine = self.affine.dot(transform)
        args = {}
        if mode == "constant":
            args["constant_values"]=self.get_c_val()
        log.print(f"Padd {padd}; {mode=}, {args}",verbose=verbose)
        arr = np.pad(self.get_array(),padd,mode=mode,**args) # type: ignore

        nii:_unpacked_nii = (arr,affine,self.header)
        if inplace:
            self.nii = nii
            return self
        return self.copy(nii)

    def rescale_and_reorient(self, axcodes_to=None, voxel_spacing=(-1, -1, -1), verbose:logging=True, inplace=False,c_val:float|None=None,mode:MODES='nearest'):

        ## Resample and rotate and Save Tempfiles
        if axcodes_to is None:
            curr = self
            ornt_img = self.orientation
            axcodes_to = nio.ornt2axcodes(ornt_img)
        else:
            curr = self.reorient(axcodes_to=axcodes_to, verbose=verbose, inplace=inplace)
        return curr.rescale(voxel_spacing=voxel_spacing, verbose=verbose, inplace=inplace,c_val=c_val,mode=mode)

    def rescale_and_reorient_(self,axcodes_to=None, voxel_spacing=(-1, -1, -1),c_val:float|None=None,mode:MODES='nearest', verbose:logging=True):
        return self.rescale_and_reorient(axcodes_to=axcodes_to,voxel_spacing=voxel_spacing,c_val=c_val,mode=mode,verbose=verbose,inplace=True)

    def reorient_same_as(self, img_as: Nifti1Image | Self, verbose:logging=False, inplace=False) -> Self:
        axcodes_to: AX_CODES = nio.ornt2axcodes(nio.io_orientation(img_as.affine)) # type: ignore
        return self.reorient(axcodes_to=axcodes_to, verbose=verbose, inplace=inplace)
    def reorient_same_as_(self, img_as: Nifti1Image | Self, verbose:logging=False) -> Self:
        return self.reorient_same_as(img_as=img_as,verbose=verbose,inplace=True)
    def rescale(self, voxel_spacing:float|tuple[float,...]=(1, 1, 1), c_val:float|None=None, verbose:logging=False, inplace=False,mode:MODES='nearest',order: int |None = None,align_corners:bool=False,atol=0.001):
        """
        Rescales the NIfTI image to a new voxel spacing.

        Args:
            voxel_spacing (tuple[float, float, float] | float): The desired voxel spacing in millimeters (x, y, z). -1 is keep the voxel spacing.
                Defaults to (1, 1, 1).
            c_val (float | None, optional): The padding value. Defaults to None, meaning that the padding value will be
                inferred from the image data.
            verbose (bool, optional): Whether to print a message indicating that the image has been resampled. Defaults to
                False.
            inplace (bool, optional): Whether to modify the current object or return a new one. Defaults to False.
            mode (str, optional): One of the supported modes by scipy.ndimage.interpolation (e.g., "constant", "nearest",
                "reflect", "wrap"). See the documentation for more details. Defaults to "constant".
            align_corners (bool|default): If True or not set and seg==True. Aline corners for scaling. This prevents segmentation mask to shift in a direction.
            atol: absolute tolerance for skipping if already close in voxel_spacing
        Returns:
            NII: A new NII object with the resampled image data.
        """
        if isinstance(voxel_spacing, (int,float)):
            voxel_spacing =(voxel_spacing for _ in range(min(3,self.affine.shape[0]-1)))
        n = self.dims
        while  n> len(voxel_spacing):
            voxel_spacing = (*voxel_spacing, -1)
        if all(a in (-1, b) for a,b in zip(voxel_spacing, self.zoom)):
            log.print(f"Image already resampled to voxel size {self.zoom}",verbose=verbose)
            return self.copy() if inplace else self

        c_val = self.get_c_val(c_val)
        # resample to new voxel spacing based on the current x-y-z-orientation
        aff = self.affine
        shp = self.shape
        zms = self.zoom
        if order is None:
            order = 0 if self.seg else 3
        #print(aff.shape,shp,zms,voxel_spacing)
        voxel_spacing = tuple([v if v != -1 else z for v,z in zip_strict(voxel_spacing,zms)])
        if np.isclose(voxel_spacing, self.zoom,atol=atol).all():
            log.print(f"Image already resampled to voxel size {self.zoom}",verbose=verbose)
            return self.copy() if inplace else self

        # Calculate new shape
        new_shp = tuple(np.rint([shp[i] * zms[i] / voxel_spacing[i] for i in range(len(voxel_spacing))]).astype(int))
        if len(new_shp) < len(shp):
            new_shp = new_shp + shp[len(new_shp):]
        new_aff = _rescale_affine(aff, shp, voxel_spacing, new_shp)  # type: ignore
        new_aff[:n, n] = nib.affines.apply_affine(aff, [0 for _ in range(n)])# type: ignore
        new_img = _resample_from_to(self, (new_shp, new_aff,voxel_spacing), order=order, mode=mode,align_corners=align_corners)
        log.print(f"Image resampled from {zms} to voxel size {voxel_spacing}",verbose=verbose)
        if inplace:
            self.nii = new_img
            return self
        return self.copy(new_img)

    def rescale_(self, voxel_spacing=(1, 1, 1), c_val:float|None=None, verbose:logging=False,mode:MODES='nearest'):
        return self.rescale( voxel_spacing=voxel_spacing, c_val=c_val, verbose=verbose,mode=mode, inplace=True)

    def resample_from_to(self, to_vox_map:Image_Reference|Has_Grid|tuple[SHAPE,AFFINE,ZOOMS], mode:MODES='nearest', order: int |None=None, c_val=None, inplace = False,verbose:logging=True,align_corners:bool=False):
        """self will be resampled in coordinate of given other image. Adheres to global space not to local pixel space
        Args:
            to_vox_map (Image_Reference|Proxy): If object, has attributes shape giving input voxel shape, and affine giving mapping of input voxels to output space. If length 2 sequence, elements are (shape, affine) with same meaning as above. The affine is a (4, 4) array-like.\n
            mode (str, optional): Points outside the boundaries of the input are filled according to the given mode ('constant', 'nearest', 'reflect' or 'wrap').Defaults to 'constant'.\n
            cval (float, optional): Value used for points outside the boundaries of the input if mode='nearest'. Defaults to 0.0.\n
            aline_corners (bool|default): If True or not set and seg==True. Aline corners for scaling. This prevents segmentation mask to shift in a direction.
            inplace (bool, optional): Defaults to False.

        Returns:
            NII:
        """        ''''''
        c_val = self.get_c_val(c_val)
        if isinstance(to_vox_map,Has_Grid):
            mapping = to_vox_map.to_gird()
        else:
            mapping = to_vox_map if isinstance(to_vox_map, tuple) else to_nii_optional(to_vox_map, seg=self.seg, default=to_vox_map)
        if isinstance(mapping,Has_Grid) and mapping.assert_affine(self,raise_error=False,origin_tolerance=0.000001,error_tolerance=0.000001,shape_tolerance=0):
            log.print(f"resample_from_to skipped; already in space: {self}",verbose=verbose)
            return self if inplace else self.copy()
        assert mapping is not None
        log.print(f"resample_from_to: {self} to {mapping}",verbose=verbose)
        if order is None:
            order = 0 if self.seg else 3
        nii = _resample_from_to(self, mapping,order=order, mode=mode,align_corners=align_corners)
        if inplace:
            self.nii = nii
            return self
        else:
            return self.copy(nii)
    def resample_from_to_(self, to_vox_map:Image_Reference|Has_Grid|tuple[SHAPE,AFFINE,ZOOMS], mode:MODES='nearest', c_val:float|None=None,verbose:logging=True,aline_corners=False):
        return self.resample_from_to(to_vox_map,mode=mode,c_val=c_val,inplace=True,verbose=verbose,align_corners=aline_corners)

    @property
    def is_empty(self) -> bool:
        """Checks if the array in the nifti is empty

        Returns:
            bool: Whether the nifti is empty or not
        """
        return np_is_empty(self.get_array())

    def n4_bias_field_correction(
        self,
        threshold = 60,
        mask=None, # type: ignore
        shrink_factor=4,
        convergence=None,
        spline_param=200,
        verbose=False,
        weight_mask=None,
        crop=False,
        inplace=False
    ):
        """Runs a n4 bias field correction over the nifty

        Args:
            threshold (int, optional): If != 0, will mask the input based on the threshold. Defaults to 60.
            mask (_type_, optional): If threshold==0, this can be set to input a individual mask. If none, lets the algorithm automatically determine the mask. Defaults to None.
            shrink_factor (int, optional): _description_. Defaults to 4.
            convergence (dict, optional): _description_. Defaults to {"iters": [50, 50, 50, 50], "tol": 1e-07}.
            spline_param (int, optional): _description_. Defaults to 200.
            verbose (bool, optional): _description_. Defaults to False.
            weight_mask (_type_, optional): _description_. Defaults to None.
            crop (bool, optional): _description_. Defaults to False.
            inplace (bool, optional): _description_. Defaults to False.

        Returns:
            NII: The NII with bias field corrected image
        """
        if convergence is None:
            convergence = {"iters": [50, 50, 50, 50], "tol": 1e-07}
        assert self.seg is False, "n4 bias field correction on a segmentation does not make any sense"
        # install antspyx not ants!
        import ants
        import ants.utils.bias_correction as bc  # install antspyx not ants!
        from ants.utils.convert_nibabel import from_nibabel
        from scipy.ndimage import binary_dilation, generate_binary_structure
        dtype = self.dtype
        input_ants:ants.ANTsImage = from_nibabel(nib.nifti1.Nifti1Image(self.get_array(),self.affine))
        if threshold != 0:
            mask_arr = self.get_array()
            mask_arr[mask_arr < threshold] = 0
            mask_arr[mask_arr != 0] = 1
            mask_arr = mask_arr.astype(np.uint8)
            struct = generate_binary_structure(3, 3)
            mask_arr = binary_dilation(mask_arr.copy(), structure=struct, iterations=3)
            mask_arr = mask_arr.astype(np.uint8)
            mask:ants.ANTsImage = from_nibabel(nib.nifti1.Nifti1Image(mask_arr,self.affine))#self.set_array(mask,verbose=False).nii
            mask = mask.set_spacing(input_ants.spacing) # type: ignore
        out = bc.n4_bias_field_correction(
            input_ants,
            mask=mask,
            shrink_factor=shrink_factor,
            convergence=convergence,
            spline_param=spline_param,
            verbose=verbose,
            weight_mask=weight_mask,

        )


        out_nib:Nifti1Image = out.to_nibabel()
        if crop:
            # Crop to regions that had a normalization applied. Removes a lot of dead space
            dif = NII((input_ants - out).to_nibabel())
            dif_arr = dif.get_array()
            dif_arr[dif_arr != 0] = 1
            dif.set_array_(dif_arr,verbose=verbose)
            ex_slice = dif.compute_crop()
            out_nib:Nifti1Image = out_nib.slicer[ex_slice]

        if inplace:
            self.nii = out_nib
            self.set_dtype_(dtype)
            return self
        return self.copy(out_nib).set_dtype_(dtype)

    def n4_bias_field_correction_(self,threshold = 60,mask=None,shrink_factor=4,convergence=None,spline_param=200,verbose=False,weight_mask=None,crop=False):
        if convergence is None:
            convergence = {"iters": [50, 50, 50, 50], "tol": 1e-07}
        return self.n4_bias_field_correction(mask=mask,shrink_factor=shrink_factor,convergence=convergence,spline_param=spline_param,verbose=verbose,weight_mask=weight_mask,crop=crop,inplace=True,threshold = threshold)

    def normalize_to_range_(self, min_value: int = 0, max_value: int = 1500, verbose:logging=True):
        assert not self.seg
        mi, ma = self.min(), self.max()
        self += -mi + min_value  # min = 0
        self_dtype = self.dtype
        max_value2 = self.max() # this is a new value if min got shifted
        if max_value2 > max_value:
            self *= max_value / max_value2
            self.set_dtype_(self_dtype)
        log.print(f"Shifted from range {mi, ma} to range {self.min(), self.max()}", verbose=verbose)

    def get_histogram(self, bins=256, hrange=None, density=False, c_val:float|None=None):
        """Returns the histogram of the image array.

        Args:
            bins (int, optional): Number of bins for the histogram. Defaults to 256.
            range (tuple, optional): Range of values to consider for the histogram. Defaults to None.
            density (bool, optional): If True, the result is the probability density function at the bin, normalized such that the integral over the range is 1. Defaults to False.
            c_val (float|None, optional): The value below which all values are set to c_val. Defaults to None.

        Returns:
            tuple: A tuple containing the histogram values and the bin edges.
        """
        arr = self.get_array()
        if c_val is not None:
            arr[arr <= c_val] = c_val
        return np.histogram(arr, bins=bins, range=hrange, density=density)

    def match_histograms(self, reference:Image_Reference,c_val = 0,inplace=False):
        assert not self.seg
        ref_nii = to_nii(reference)
        assert ref_nii.seg is False
        assert self.seg is False
        c_val = self.get_c_val(c_val)
        if c_val <= -999:
            raise ValueError('match_histograms only functions on MRI, which have a minimum 0.')

        from skimage.exposure import match_histograms as ski_match_histograms
        img_arr = self.get_array()
        matched = ski_match_histograms(img_arr, ref_nii.get_array())
        matched[matched <= c_val] = c_val
        return self.set_array(matched, inplace=inplace,verbose=False)

    def match_histograms_(self, reference:Image_Reference,c_val = 0):
        return self.match_histograms(reference,c_val = c_val,inplace=True)

    def smooth_gaussian(self, sigma:float|list[float]|tuple[float],truncate:float=4.0,nth_derivative=0,inplace=False):
        assert self.seg is False, "You really want to smooth a segmentation? If yes, use smooth_gaussian_channelwise() instead"
        from scipy.ndimage import gaussian_filter
        arr = gaussian_filter(self.get_array(), sigma, order=nth_derivative,cval=self.get_c_val(), truncate=truncate)# radius=None, axes=None
        return self.set_array(arr,inplace,verbose=False)

    def smooth_gaussian_(self, sigma:float|list[float]|tuple[float],truncate=4.0,nth_derivative=0):
        return self.smooth_gaussian(sigma=sigma,truncate=truncate,nth_derivative=nth_derivative,inplace=True)


    def smooth_gaussian_labelwise(
        self,
        label_to_smooth: list[int] | int,
        sigma: float = 3.0,
        radius: int = 6,
        truncate: int = 4,
        boundary_mode: str = "nearest",
        dilate_prior: int = 0,
        dilate_connectivity: int = 1,
        dilate_channelwise: bool = False,
        smooth_background: bool = True,
        background_threshold: float | None = None,
        inplace: bool = False,
        verbose:logging=False,
    ):
        """Smoothes the segmentation mask by applying a gaussian filter label-wise and then using argmax to derive the smoothed segmentation labels again.

        Args:
            label_to_smooth (list[int] | int): Which labels to smooth in the mask. Every other label will be untouched
            sigma (float, optional): Sigma of the gaussian blur. Defaults to 3.0.
            radius (int, optional): Radius of the gaussian blur. Defaults to 6.
            truncate (int, optional): Truncate of the gaussian blur. Defaults to 4.
            boundary_mode (str, optional): Boundary Mode of the gaussian blur. Defaults to "nearest".
            dilate_prior (int, optional): Dilate this many voxels before starting the gaussian blur algorithm. Defaults to 0.
            dilate_connectivity (int, optional): Connectivity of the dilation process, if applied. Defaults to 3.
            smooth_background (bool, optional): If true, will also smooth the background. If False, the background voxels stay the same and the segmentation cannot add voxels. Defaults to True.
            inplace (bool, optional): If true, will overwrite the input NII instead of making a copy. Defaults to False.

        Returns:
            NII: The smoothed NII object.
        """
        assert self.seg, "You cannot use this on a non-segmentation NII"
        log.print("smooth_gaussian_labelwise",verbose=verbose)
        smoothed = np_smooth_gaussian_labelwise(
            self.get_seg_array(),
            label_to_smooth=label_to_smooth,
            sigma=sigma,
            radius=radius,
            truncate=truncate,
            boundary_mode=boundary_mode,
            dilate_prior=dilate_prior,
            dilate_connectivity=dilate_connectivity,
            smooth_background=smooth_background,
            background_threshold=background_threshold,
            dilate_channelwise=dilate_channelwise,
        )
        return self.set_array(smoothed, inplace, verbose=False)

    def smooth_gaussian_labelwise_(
        self,
        label_to_smooth: list[int] | int,
        sigma: float = 3.0,
        radius: int = 6,
        truncate: int = 4,
        boundary_mode: str = "nearest",
        dilate_prior: int = 1,
        dilate_connectivity: int = 1,
        dilate_channelwise: bool = False,
        smooth_background: bool = True,
        background_threshold: float | None = None,
    ):
        return self.smooth_gaussian_labelwise(
            label_to_smooth=label_to_smooth,
            sigma=sigma,
            radius=radius,
            truncate=truncate,
            boundary_mode=boundary_mode,
            dilate_prior=dilate_prior,
            dilate_connectivity=dilate_connectivity,
            smooth_background=smooth_background,
            inplace=True,
            background_threshold=background_threshold,
            dilate_channelwise=dilate_channelwise,
        )

    def to_ants(self):
        try:
            import ants
        except Exception:
            log.print_error()
            log.on_fail("run 'pip install antspyx' to install hf-deepali")
            raise
        return ants.from_nibabel(self.nii)

    def to_simpleITK(self):
        from TPTBox.core.sitk_utils import nii_to_sitk
        return nii_to_sitk(self)

    def to_deepali(self,align_corners: bool = True,dtype=None,device:device|str = "cpu"):
        import torch
        try:
            from deepali.data import Image as deepaliImage  # type: ignore
        except Exception:
            log.print_error()
            log.on_fail("run 'pip install hf-deepali' to install deepali")
            raise
        dim = np.asarray(self.header["dim"])
        ndim = int(dim[0])
        # Image data array
        data = self.get_array()
        # Squeeze unused dimensions
        # https://github.com/InsightSoftwareConsortium/ITK/blob/3454d857dc46e4333ad1178be8c186547fba87ef/Modules/IO/NIFTI/src/itkNiftiImageIO.cxx#L1112-L1156
        intent_code = int(self.header["intent_code"]) # type: ignore
        if intent_code in (1005, 1006, 1007):
            # Vector or matrix valued image
            for realdim in range(4, 1, -1):
                if dim[realdim] > 1:
                    break
            else:
                realdim = 1
        elif intent_code == 1004:
            raise NotImplementedError("NII has an intent code of NIFTI_INTENT_GENMATRIX which is not yet implemented")
        else:
            # Scalar image
            realdim = ndim
            while realdim > 3 and dim[realdim] == 1:
                realdim -= 1
        data = np.reshape(data, data.shape[:realdim] + data.shape[5:])
        # Reverse order of axes
        data = np.transpose(data, axes=tuple(reversed(range(data.ndim))))
        grid = self.to_deepali_grid(align_corners=align_corners)
        # Add leading channel dimension
        if data.ndim == grid.ndim:
            data = np.expand_dims(data, 0)
        if data.dtype == np.uint16:
            data = data.astype(np.int32)
        elif data.dtype == np.uint32:
            data = data.astype(np.int64)
        grid = grid.align_corners_(align_corners)
        data = torch.Tensor(data)
        #if len(torch.Tensor(data)) == 3:
        #   data = data.unsqueeze(0)
        return deepaliImage(data, grid, dtype=dtype, device=device)  # type: ignore

    def erode_msk(self, n_pixel: int = 5, labels: LABEL_REFERENCE = None, connectivity: int = 3, inplace=False,verbose:logging=True,border_value=0, use_crop=True,ignore_direction:DIRECTIONS|int|None=None):
        """
        Erodes the binary segmentation mask by the specified number of voxels.

        Args:
            mm (int, optional): The number of voxels to erode the mask by. Defaults to 5.
            labels (LABEL_REFERENCE, optional): Labels that should be dilated. If None, will erode all labels (not including zero!)
            connectivity (int, optional): Elements up to a squared distance of connectivity from the center are considered neighbors. connectivity may range from 1 (no diagonal elements are neighbors) to rank (all elements are neighbors).
            inplace (bool, optional): Whether to modify the mask in place or return a new object. Defaults to False.
            verbose (bool, optional): Whether to print a message indicating that the mask was eroded. Defaults to True.
            use_crop: speed up computation by cropping and un-cropping the segmentation. Minor overhead if the segmentation fills most of the image
        Returns:
            NII: The eroded mask.

        Notes:
            The method uses binary erosion with a 3D structuring element to erode the mask by the specified number of voxels.

        """
        assert self.seg
        log.print("erode mask",end='\r',verbose=verbose)
        msk_i_data = self.get_seg_array()
        labels = self.unique() if labels is None else labels
        if isinstance(ignore_direction,str):
            ignore_direction = self.get_axis(ignore_direction)
        out = np_erode_msk(msk_i_data, label_ref=labels, n_pixel=n_pixel, connectivity=connectivity,border_value=border_value,ignore_axis=ignore_direction, use_crop=use_crop)
        out = out.astype(self.dtype)
        log.print("Mask eroded by", n_pixel, "voxels",verbose=verbose)
        return self.set_array(out,inplace=inplace)

    def erode_msk_(self, n_pixel:int = 5, labels: LABEL_REFERENCE = None, connectivity: int=3, verbose:logging=True,border_value=0,use_crop=True,ignore_direction:DIRECTIONS|int|None=None):
        return self.erode_msk(n_pixel=n_pixel, labels=labels, connectivity=connectivity, inplace=True, verbose=verbose,border_value=border_value,use_crop=use_crop,ignore_direction=ignore_direction)

    def dilate_msk(self, n_pixel: int = 5, labels: LABEL_REFERENCE = None, connectivity: int = 3, mask: Self | None = None, inplace=False, verbose:logging=True,use_crop=True, ignore_direction:DIRECTIONS|int|None=None):
        """
        Dilates the binary segmentation mask by the specified number of voxels.

        Args:
            n_pixel (int, optional): The number of voxels to dilate the mask by. Defaults to 5.
            labels (list[int], optional): Labels that should be dilated. If None, will dilate all labels (not including zero!)
            connectivity (int, optional): Elements up to a squared distance of connectivity from the center are considered neighbors. connectivity may range from 1 (no diagonal elements are neighbors) to rank (all elements are neighbors).
            mask (NII, optional): If set, after each iteration, will zero out everything based on this mask
            inplace (bool, optional): Whether to modify the mask in place or return a new object. Defaults to False.
            verbose (bool, optional): Whether to print a message indicating that the mask was dilated. Defaults to True.
            use_crop: speed up computation by cropping and un-cropping the segmentation. Minor overhead if the segmentation fills most of the image
        Returns:
            NII: The dilated mask.

        Notes:
            The method uses binary dilation with a 3D structuring element to dilate the mask by the specified number of voxels.

        """
        assert self.seg
        log.print("dilate mask",end='\r',verbose=verbose)
        if labels is None:
            labels = self.unique()
        msk_i_data = self.get_seg_array()
        mask_ = mask.get_seg_array() if mask is not None else None
        if isinstance(ignore_direction,str):
            ignore_direction = self.get_axis(ignore_direction)
        out = np_dilate_msk(arr=msk_i_data, label_ref=labels, n_pixel=n_pixel, mask=mask_, connectivity=connectivity,ignore_axis=ignore_direction, use_crop=use_crop)
        out = out.astype(self.dtype)
        log.print("Mask dilated by", n_pixel, "voxels",verbose=verbose)
        return self.set_array(out,inplace=inplace)

    def dilate_msk_(self, n_pixel:int = 5, labels: LABEL_REFERENCE = None, connectivity: int=3, mask: Self | None = None, verbose:logging=True,use_crop=True,ignore_direction:DIRECTIONS|int|None=None):
        return self.dilate_msk(n_pixel=n_pixel, labels=labels, connectivity=connectivity, mask=mask, inplace=True, verbose=verbose,use_crop=use_crop,ignore_direction=ignore_direction)


    def fill_holes(self, labels: LABEL_REFERENCE = None, slice_wise_dim: int|str | None = None, verbose:logging=False, inplace=False,use_crop=True):  # noqa: ARG002
        """Fills holes in segmentation

        Args:
            labels (LABEL_REFERENCE, optional): Labels that the hole-filling should be applied to. If none, applies on all labels found in arr. Defaults to None.
            verbose: whether to print which labels have been filled
            inplace (bool): Whether to modify the current NIfTI image object in place or create a new object with the mapped labels.
                Default is False.
            slice_wise_dim (int | None, optional): If the input is 3D, the specified dimension here cna be used for 2D slice-wise filling. Defaults to None.
            use_crop: speed up computation by cropping and un-cropping the segmentation. Minor overhead if the segmentation fills most of the image
        Returns:
            NII: If inplace is True, returns the current NIfTI image object with filled holes. Otherwise, returns a new NIfTI image object with filled holes.
        """
        assert self.seg
        if labels is None:
            labels = list(self.unique())

        if isinstance(labels, int):
            labels = [labels]

        seg_arr = self.get_seg_array()
        if isinstance(slice_wise_dim,str):
            slice_wise_dim = self.get_axis(slice_wise_dim)
        #seg_arr = self.get_seg_array()
        filled = np_fill_holes(seg_arr, label_ref=labels, slice_wise_dim=slice_wise_dim, use_crop=use_crop)
        return self.set_array(filled,inplace=inplace)

    def fill_holes_(self, labels: LABEL_REFERENCE = None, slice_wise_dim: int |str| None = None, verbose:logging=True,use_crop=True):
        return self.fill_holes(labels, slice_wise_dim, verbose, inplace=True,use_crop=use_crop)

    def calc_convex_hull(
        self,
        axis: DIRECTIONS|None = "S",
        inplace: bool = False,
        verbose: bool = False
    ):
        """Calculates the convex hull of this segmentation nifty

        Args:
            axis (int | None, optional): If given axis, will calculate convex hull along that axis (remaining dimension must be at least 2). Defaults to None.
        """
        assert self.seg, "To calculate the convex hull, this must be a segmentation"
        axis_int = self.get_axis(axis) if axis is not None else None
        convex_hull_arr = np_calc_convex_hull(self.get_seg_array(), axis=axis_int, verbose=verbose)
        if inplace:
            return self.set_array_(convex_hull_arr)
        return self.set_array(convex_hull_arr)

    def calc_convex_hull_(self, axis: None|DIRECTIONS="S", verbose: bool = False,):
        return self.calc_convex_hull(axis=axis, inplace=True, verbose=verbose)


    def boundary_mask(self, threshold: float,inplace = False):
        """
        Calculate a boundary mask based on the input image.

        Parameters:
        - img (NII): The image used to create the boundary mask.
        - threshold(float): threshold

        Returns:
        NII: A segmentation of the boundary.


        This function takes a NII and generates a boundary mask by marking specific regions.
        The intensity of the image can be adjusted for CT scans by adding 1000. The boundary mask is created by initializing
        corner points and using an "infect" process to mark neighboring points. The boundary mask is initiated with
        zeros, and specific boundary points are set to 1. The "infect" function iteratively marks neighboring points in the mask.
        The process starts from the initial points and corner points of the image. The infection process continues until the
        infect_list is empty. The resulting boundary mask is modified by subtracting 1 from all non-zero values and setting
        the remaining zeros to 2. The sum of the boundary mask values is printed before returning the modified NII object.

        """
        return self.set_array(np_calc_boundary_mask(self.get_array(),threshold),inplace=inplace,verbose=False)

    def get_connected_components(self, labels: int |list[int]|None=None, connectivity: int = 3, include_zero: bool=False,inplace=False) -> Self:  # noqa: ARG002
        assert self.seg, "This only works on segmentations"
        out, _ = np_connected_components(self.get_seg_array(), label_ref=labels, connectivity=connectivity, include_zero=include_zero)
        return self.set_array(out,inplace=inplace)

    def get_connected_components_per_label(self, labels: int |list[int], connectivity: int = 3, include_zero: bool=False) -> dict[int, Self]:  # noqa: ARG002
        assert self.seg, "This only works on segmentations"
        out = np_connected_components_per_label(self.get_seg_array(), label_ref=labels, connectivity=connectivity, include_zero=include_zero)
        cc = {i: self.set_array(k) for i,k in out.items()}
        return cc

    def filter_connected_components(self, labels: int |list[int]|None=None,min_volume:int=0,max_volume:int|None=None, max_count_component = None, connectivity: int = 3,removed_to_label=0,keep_label=False, inplace=False,):
        """
        Filter connected components in a segmentation array based on specified volume constraints.

        Parameters:
        labels (int | list[int]): The labels of the components to filter.
        min_volume (int | None): Minimum volume for a component to be retained. Components smaller than this will be removed.
        max_volume (int | None): Maximum volume for a component to be retained. Components larger than this will be removed.
        max_count_component (int | None): Maximum number of components to retain. Once this limit is reached, remaining components will be removed.
        connectivity (int): Connectivity criterion for defining connected components (default is 3).
        removed_to_label (int): Label to assign to removed components (default is 0).
        TODO : max_count_component currently filters over all labels instead of per label. will be changed.
        TODO : removed_to_label does not work when keep_label=False
        Returns:
        None
        """
        assert self.seg, "This only works on segmentations"
        arr = np_filter_connected_components(self.get_seg_array(), largest_k_components=max_count_component,label_ref=labels,connectivity=connectivity,return_original_labels=keep_label,min_volume=min_volume,max_volume=max_volume,removed_to_label=removed_to_label,)
        assert arr.shape == self.shape, f"Shape mismatch: {arr.shape} != {self.shape}"
        if keep_label and labels is not None:
            if isinstance(labels,int):
                labels = [labels]
            old_labels = [i for i in self.unique() if i not in labels]
            if len(old_labels) != 0:
                s = self.extract_label(old_labels,keep_label=True).get_array()
                arr[s != 0] = s[s!=0]
        #print("filter",nii.unique())
        #assert max_count_component is None or nii.max() <= max_count_component, nii.unique()
        return self.set_array(arr, inplace=inplace)
    def filter_connected_components_(self, labels: int |list[int]|None=None,min_volume:int=0,max_volume:int|None=None, max_count_component = None, connectivity: int = 3,keep_label=False,removed_to_label=0):
        return self.filter_connected_components(labels,min_volume=min_volume,max_volume=max_volume, max_count_component = max_count_component, connectivity = connectivity,removed_to_label=removed_to_label,keep_label=keep_label,inplace=True)

    def get_segmentation_connected_components_center_of_mass(self, label: int, connectivity: int = 3, sort_by_axis: int | None = None) -> list[COORDINATE]:
        """Calculates the center of mass of the different connected components of a given label in an array

        Args:
            label (int): the label of the connected components
            connectivity (int, optional): Connectivity for the connected components. Defaults to 3.
            sort_by_axis (int | None, optional): If not none, will sort the center of mass list by this axis values. Defaults to None.

        Returns:
            _type_: _description_
        """
        assert self.seg, "This only works on segmentations"
        arr = self.get_seg_array()
        return np_get_connected_components_center_of_mass(arr, label=label, connectivity=connectivity, sort_by_axis=sort_by_axis)


    def get_largest_k_segmentation_connected_components(self, k: int | None, labels: int | list[int] | None = None, connectivity: int = 1, return_original_labels: bool = True,inplace=False,min_volume:int=0,max_volume:int|None=None,removed_to_label=0) -> Self:
        """Finds the largest k connected components in a given array (does NOT work with zero as label!)

        Args:
            arr (np.ndarray): input array
            k (int | None): finds the k-largest components. If k is None, will find all connected components and still sort them by size
            labels (int | list[int] | None, optional): Labels that the algorithm should be applied to. If none, applies on all labels found in this NII. Defaults to None.
            return_original_labels (bool): If set to False, will label the components from 1 to k. Defaults to True
        """
        raise DeprecationWarning("Use filter_connected_components instead")
        msk_i_data = self.get_seg_array()
        out = np_filter_connected_components(msk_i_data, largest_k_components=k, label_ref=labels, connectivity=connectivity, return_original_labels=return_original_labels,min_volume=min_volume,max_volume=max_volume,removed_to_label=removed_to_label)
        return self.set_array(out,inplace=inplace)

    def compute_surface_mask(self, connectivity: int = 3, dilated_surface: bool = False) -> Self:
        """ Removes everything but surface voxels

        Args:
            connectivity (int): Connectivity for surface calculation
            dilated_surface (bool): If False, will return msk - eroded mask. If true, will return dilated msk - msk
        """
        assert self.seg, "This only works on segmentations"
        return self.set_array(np_compute_surface(self.get_seg_array(), connectivity=connectivity, dilated_surface=dilated_surface))


    def compute_surface_points(self, connectivity: int = 3, dilated_surface: bool = False) -> list[tuple[int,int,int]]:
        assert self.seg, "This only works on segmentations"
        surface = self.compute_surface_mask(connectivity, dilated_surface)
        return np_point_coordinates(surface.get_seg_array()) # type: ignore


    def fill_holes_global_with_majority_voting(self, connectivity: int = 3, inplace: bool = False, verbose: bool = False) -> Self:
        """Fills 3D holes globally, and resolves inter-label conflicts with majority voting by neighbors

        Args:
            connectivity (int, optional): Connectivity of fill holes. Defaults to 3.
            inplace (bool, optional): Defaults to False.
            verbose (bool, optional): Defaults to False.

        Returns:
            NII:
        """
        assert self.seg, "This only works on segmentations"
        arr = np_fill_holes_global_with_majority_voting(self.get_seg_array(), connectivity=connectivity, verbose=verbose, inplace=inplace)
        return self.set_array(arr,inplace=inplace)


    def map_labels_based_on_majority_label_mask_overlap(self, label_mask: Self, labels: int | list[int] | None = None, dilate_pixel: int = 1, inplace: bool = False,no_match_label=0) -> Self:
        """Relabels all individual labels from input array to the majority labels of a given label_mask

        Args:
            label_mask (np.ndarray): the mask from which to pull the target labels.
            labels (int | list[int] | None, optional): Which labels in the input to process. Defaults to None.
            dilate_pixel (int, optional): If true, will dilate the input to calculate the overlap. Defaults to 1.
            inplace (bool, optional): Defaults to False.

        Returns:
            NII: Relabeled nifti
        """
        assert self.seg and label_mask.seg, "This only works on segmentations"
        return self.set_array(np_map_labels_based_on_majority_label_mask_overlap(self.get_seg_array(), label_mask.get_seg_array(), label_ref=labels, dilate_pixel=dilate_pixel, inplace=inplace,no_match_label=no_match_label), inplace=inplace,)


    def get_segmentation_difference_to(self, mask_gt: Self, ignore_background_tp: bool = False) -> Self:
        """Calculates an NII that represents the segmentation difference between self and given groundtruth mask

        Args:
            mask_groundtruth (Self): The ground truth mask. Must match in orientation, zoom, and shape

        Returns:
            NII: Difference NII (1: FN, 2: TP, 3: FP, 4: Wrong label)
        """
        assert self.seg and mask_gt.seg, "This only works on segmentations"
        if self.orientation != mask_gt.orientation:
            mask_gt = mask_gt.reorient_same_as(self)

        self.assert_affine(zoom=mask_gt.zoom, shape=mask_gt.shape)
        arr = self.get_seg_array()
        gt = mask_gt.get_seg_array()
        diff_arr = arr.copy() * 0
        # TP
        diff_arr[gt == arr] = 2
        # FN
        diff_arr[(gt != 0) & (arr == 0)] = 1
        # FP
        diff_arr[(gt == 0) & (arr != 0)] = 3
        # Wrong label
        diff_arr[(diff_arr == 0) & (gt != arr)] = 4

        if ignore_background_tp:
            diff_arr[(gt == 0) & (arr == 0)] = 0

        return self.set_array(diff_arr)

    def get_overlapping_labels_to(
        self,
        mask_other: Self
    ) -> list[tuple[int, int]]:
        """Calculates the pairs of labels that are overlapping in at least one voxel (fast)

        Args:
            mask_other (NII): The array to be compared with.

        Returns:
            list[tuple[int, int]]: List of tuples of labels that overlap in at least one voxel. First label in the tuple is Self NII, the second is of the mask_other
        """
        assert self.seg and mask_other.seg
        return np_calc_overlapping_labels(self.get_seg_array(), mask_other.get_seg_array())

    def is_segmentation_in_border(self,minimum=0, voxel_tolerance: int = 2,use_mm=False) -> bool:
        """
        Checks if the segmentation is touching the border of the image volume.

        Parameters:
        - minimum (int, optional): Minimum intensity threshold for segmentation. Defaults to 0.
        - voxel_tolerance (int, optional): Number of voxels allowed as tolerance from the border. Defaults to 2.
        - use_mm (bool, optional): Whether to use millimeter units instead of voxels. Defaults to False.

        Returns:
        - bool: True if the segmentation is within the defined voxel tolerance of the border, False otherwise.
        """
        slices = self.compute_crop(minimum,dist=0,use_mm=use_mm)
        shp = self.shape
        seg_at_border = False
        for d in range(3):
            if slices[d].start <= voxel_tolerance or slices[d].stop - 1 >= shp[d] - voxel_tolerance:
                seg_at_border = True
                break
        return seg_at_border

    def truncate_labels_beyond_reference_(
        self, idx: int | list[int] = 1, not_beyond: int | list[int] = 1, fill: int = 0,  axis: DIRECTIONS = "S", inclusion: bool = False, inplace: bool = True
    ):
        """
        Modifies the NIfTI object to remove all voxels with the label `idx` beyond a reference label `not_beyond`
        along a specified axis, replacing them with `fill (default = 0)`.

        Parameters:
            nii (NII): The NIfTI-like object with 3D imaging data.
            idx (int or list[int]): The index/label(s) to process in the array. Default is 1.
            not_beyond (int or list[int]): The label/index used to determine the reference position. Default is 1.
            fill (int): The value to set for voxels beyond the reference point. Default is 0.
            axis (str): The anatomical axis along which truncation is applied. Default is "S" (superior).
                Options:
                - "S" (Superior)
                - "I" (Inferior)
                - "R" (Right)
                - "L" (Left)
                - "A" (Anterior)
                - "P" (Posterior)
            inclusion (bool): Controls whether the reference label `not_beyond` itself is considered a boundary.
                - `False` (default): The truncation occurs strictly beyond the reference label.
                - `True`: The truncation includes the reference label as well.
            inplace (bool): If `True`, modifies the NIfTI object in place. If `False`, returns a modified copy.

        Returns:
            NII: The modified NIfTI object.
        """
        # Identify the axis to work on
        axis_ = self.get_axis(axis)
        flip = self.orientation[axis_] != axis  # Check orientation for flipping
        # Get the array data
        np_array = self.get_array()
        np_array_cond = self.extract_label(idx).get_seg_array()

        # Find the lowest point (smallest index) along the axis where `not_above` exists
        threshold = np.where(self.extract_label(not_beyond).get_seg_array() == 1)
        if len(threshold[axis_]) == 0:
            return self if inplace else self.copy()
        flip_up = flip
        if inclusion:
            flip_up = not flip_up
        # Determine the lowest index along the axis
        limit = threshold[axis_].min() if flip_up else threshold[axis_].max()

        # Create an array of indices along the specified axis
        index_grid = np.arange(self.shape[axis_])

        # Create a mask to identify the region above or below the threshold
        mask = index_grid < limit if flip else index_grid >= limit

        # Apply the mask along the specified axis
        mask = np.expand_dims(mask, axis=tuple(i for i in range(np_array.ndim) if i != axis_))
        mask = np.broadcast_to(mask, self.shape)

        # Replace values of `idx` with `fill` in the masked region
        np_array = np.where((np_array_cond == 1) & mask, fill, np_array)

        # Update the NIfTI object with the modified array
        return self.set_array(np_array, inplace=inplace)
    def truncate_labels_beyond_reference(
        self,
        idx: int | list[int] = 1,
        not_beyond: int | list[int] = 1,
        fill: int = 0,
        axis: DIRECTIONS = "S",
        inclusion: bool = False
    ):
        return self.truncate_labels_beyond_reference_(idx,not_beyond,fill,axis,inclusion)

    def infect(self: NII, reference_mask: NII, inplace=False,verbose=True,axis:int|str|None=None):
        """
        Expands labels from self_mask into regions of reference_mask == 1 via breadth-first diffusion.

        Args:
            self_mask (ndarray): (H, W) or (D, H, W) integer-labeled array.
            reference_mask (ndarray): Binary array of same shape as self_mask.
            max_iters (int): Maximum number of propagation steps.

        Returns:
            ndarray: Updated label mask.
        """
        self.assert_affine(reference_mask)
        self_mask = self.compute_surface_mask().get_seg_array().copy()
        self_mask_org = self.get_seg_array().copy()
        ref_mask = np.clip(reference_mask.get_seg_array(), 0, 1)
        ref_mask[self_mask_org != 0] = 0
        searched = np.clip(self_mask,0,1).astype(np.uint8)

        # Define neighborhood kernel
        if axis is None:
            kernel = [(1,0,0),(0,1,0),(0,0,1),(-1,0,0),(0,-1,0),(0,0,-1)]
        else:
            if isinstance(axis,str):
                axis = self.get_axis(axis)
            if axis == 0:
                kernel = [(0,1,0),(0,0,1),(0,-1,0),(0,0,-1)]
            elif axis == 1:
                kernel = [(1,0,0),(0,0,1),(-1,0,0),(0,0,-1)]
            elif axis == 2:
                kernel = [(1,0,0),(0,1,0),(-1,0,0),(0,-1,0)]
            else:
                raise NotImplementedError(axis)

        search = []
        coords = np.where(self_mask != 0)
        def _add_idx(x,y,z,v):
            for x1,y1,z1 in kernel:
                a = x+x1
                b = y+y1
                c = z+z1
                if a < 0 or b < 0 or c < 0:
                    continue
                if a >= self_mask.shape[0] or b >= self_mask.shape[1] or c >= self_mask.shape[2]:
                    continue
                #try:
                if searched[a,b,c] == 0 and ref_mask[a,b,c] == 1:
                    search.append((a,b,c,v))
                #except Exception:
                #    pass
        def _infect(a,b,c,v):
            if searched[a,b,c] != 0:
                return
            if ref_mask[a,b,c] == 0:
                return
            #print(a,b,c)
            searched[a,b,c] = 1
            self_mask[a,b,c] = v
            _add_idx(x,y,z,v)

        from tqdm import tqdm
        for x,y,z in tqdm(zip(coords[0],coords[1],coords[2]),total=len(coords[0]),disable=not verbose,desc="Collecting Surface"):
            _add_idx(x,y,z,self_mask[x,y,z])
        while len(search) != 0:
            search2 = search
            search = []
            for x,y,z,v in tqdm(search2,disable=not verbose,desc="infect"):
                _infect(x,y,z,v)
        self_mask[self_mask == 0] = self_mask_org[self_mask == 0]
        return self.set_array(self_mask,inplace=inplace)

    def infect_(self: NII, reference_mask: NII,verbose=True,axis:int|str|None=None):
        return self.infect(reference_mask, inplace=True,verbose=verbose,axis=axis)

    def map_labels(self, label_map:LABEL_MAP , verbose:logging=True, inplace=False):
        """
        Maps labels in the given NIfTI image according to the label_map dictionary.
        Args:
            label_map (dict): A dictionary that maps the original label values (str or int) to the new label values (int).
                For example, `{"T1": 1, 2: 3, 4: 5}` will map the original labels "T1", 2, and 4 to the new labels 1, 3, and 5, respectively.
            verbose (bool): Whether to print the label mapping and the number of labels reassigned. Default is True.
            inplace (bool): Whether to modify the current NIfTI image object in place or create a new object with the mapped labels.
                Default is False.

        Returns:
            If inplace is True, returns the current NIfTI image object with mapped labels. Otherwise, returns a new NIfTI image object with mapped labels.
        """
        data_orig = self.get_seg_array()
        labels_before = [v for v in np_unique(data_orig) if v > 0]
        # enforce keys to be str to support both str and int
        label_map_ = {
            (v_name2idx[k] if k in v_name2idx else int(k)): (
                v_name2idx[v] if v in v_name2idx else (0 if v is None else int(v))
            )
            for k, v in label_map.items()
        }
        log.print("label_map_ =", label_map_, verbose=verbose)
        data = np_map_labels(data_orig, label_map_)
        labels_after = [v for v in np_unique(data) if v > 0]
        log.print(
                "N =",
                len(label_map_),
                "labels reassigned, before labels: ",
                labels_before,
                " after: ",
                labels_after,verbose=verbose
            )
        nii = data.astype(np.uint16), self.affine, self.header
        if inplace:
            self.nii = nii
            return self
        return self.copy(nii)

    def map_labels_(self, label_map: LABEL_MAP, verbose:logging=True):
        return self.map_labels(label_map,verbose=verbose,inplace=True)

    def copy(self, nib:Nifti1Image|_unpacked_nii|None = None,seg=None):
        if nib is None:
            nib = (self.get_array().copy(), self.affine.copy(), self.header.copy())
        return NII(nib,seg=self.seg if seg is None else seg,c_val = self.c_val,info = self.info)


    def flip(self, axis:int|str,keep_global_coords=True,inplace=False):
        axis = self.get_axis(axis) if not isinstance(axis,int) else axis
        if keep_global_coords:
            orient = list(self.orientation)
            orient[axis] = _same_direction[orient[axis]]
            return self.reorient(tuple(orient),inplace=inplace)
        else:
            return self.set_array(np.flip(self.get_array(),axis),inplace=inplace)

    def clone(self):
        return self.copy()
    @secure_save
    def save(self,file:str|Path,make_parents=True,verbose:logging=True, dtype = None):
        if make_parents:
            Path(file).parent.mkdir(0o777,exist_ok=True,parents=True)
        arr = self.get_array() if not self.seg else self.get_seg_array()
        if isinstance(arr,np.floating) and self.seg:
            self.set_dtype_("smallest_uint")
            arr = self.get_array() if not self.seg else self.get_seg_array()

        self.header.set_data_dtype(arr.dtype)
        out = Nifti1Image(arr, self.affine,self.header)#,dtype=arr.dtype)
        if dtype is not None:
            out.set_data_dtype(dtype)
        if out.header["qform_code"] == 0: #NIFTI_XFORM_UNKNOWN Will cause an error for some rounding of the affine in ITKSnap ...
            # 1 means Scanner coordinate system
            # 2 means align (to something) coordinate system
            out.header["qform_code"] = 2 if self.seg else 1

        nib.save(out, file) #type: ignore
        log.print(f"Save {file} as {out.get_data_dtype()}",verbose=verbose,ltype=Log_Type.SAVE)

    @secure_save
    def save_nrrd(self:Self, file: str | Path|bids_files.BIDS_FILE,make_parents=True,verbose:logging=True,**args):
        """
        Save an NII object to an NRRD file.

        Args:
            nii_obj (NII): The NII object to be saved.
            path (str | Path): The file path where the NRRD file will be saved.

        Raises:
            ImportError: If the `pynrrd` package is not installed.
            ValueError: If the affine matrix is invalid or incompatible.
        """
        try:
            import nrrd
        except ModuleNotFoundError:
            raise ImportError("The `pynrrd` package is required but not installed. Install it with `pip install pynrrd`." ) from None
        if isinstance(file, bids_files.BIDS_FILE):
            file = file.file['nrrd']

        from TPTBox.core.internal.slicer_nrrd import save_slicer_nrrd
        save_slicer_nrrd(self,file,make_parents=make_parents,verbose=verbose,**args)


    def __str__(self) -> str:
        return f"{super().__str__()}, seg={self.seg}" # type: ignore
    def __repr__(self)-> str:
        return self.__str__()
    def __array__(self,dtype=None):
            self._unpack()
            if dtype is None:
                return self._arr
            else:
                return self._arr.astype(dtype, copy=False)
    def __array_wrap__(self, array,context=None, return_scalar=False):
        assert not return_scalar,context
        if array.shape != self.shape:
            raise SyntaxError(f"Function call induce a shape change of nii image. Before {self.shape} after {array.shape}. {context}")
        return self.set_array(array)
    def __getitem__(self, key)-> Any:
        if isinstance(key,Sequence):
            ellipsis_type = type(Ellipsis)

            if all(isinstance(k, (slice, ellipsis_type)) for k in key):
                #if all(k.step is not None and k.step == 1 for k in key):
                #    raise NotImplementedError(f"Slicing is not implemented. Attempted {key}")
                if len(key)!= len(self.shape) or Ellipsis in key:
                    raise ValueError(f"Number slices must have exact number of slices like in dimension. Attempted: {key} - Shape {self.shape}")
                return self.apply_crop(key) # type: ignore
            elif  all(isinstance(k, int) for k in key):
                if len(key)!= len(self.shape):
                    raise ValueError(f"Number ints must have exact number of slices like in dimension. Attempted: {key} - Shape {self.shape}")
                self._unpack()
                return self._arr.__getitem__(key)
            else:
                self._unpack()
                return self._arr.__getitem__(key)
                #raise TypeError("Invalid argument type:", (key))
        elif isinstance(key,self.__class__):
            return self.get_array()[key.get_array()==1]
        elif isinstance(key,np.ndarray):
            return self.get_array()[key]
        elif isinstance(key,slice):
            self.__getitem__((key,Ellipsis,Ellipsis))
        else:
            raise TypeError("Invalid argument type:", type(key))
    def __setitem__(self, key,value):
        if isinstance(key,self.__class__):
            key = key.get_array()==1
        self._unpack()
        self.__divergent = True
        self._arr[key] = value


    @classmethod
    def suppress_dtype_change_printout_in_set_array(cls, value=True):
        global suppress_dtype_change_printout_in_set_array  # noqa: PLW0603
        suppress_dtype_change_printout_in_set_array = value
    def is_intersecting_vertical(self, b: Self, min_overlap=40) -> bool:
        '''
        Test if the image intersect in global space.
        assumes same Rotation
        TODO: Testing
        '''

        #warnings.warn('is_intersecting is untested use get_intersecting_volume instead')
        x1 = self.affine.dot([0, 0, 0, 1])[:3] # type: ignore
        x2 = self.affine.dot((*self.shape, 1))[:3]# type: ignore
        y1 = b.affine.dot([0, 0, 0, 1])[:3]# type: ignore
        y2 = b.affine.dot((*b.shape, 1))[:3]# type: ignore
        max_v = max(x1[2],x2[2])- min_overlap
        min_v = min(x1[2],x2[2])+ min_overlap
        if min_v < y1[2] < max_v:
            return True
        if min_v < y2[2] < max_v:
            return True

        max_v = max(y1[2],y2[2])- min_overlap
        min_v = min(y1[2],y2[2])+ min_overlap
        if min_v < x1[2] < max_v:
            return True
        return min_v < x2[2] < max_v

    def get_intersecting_volume(self, b: Self) -> float:
        '''
        computes intersecting volume
        '''
        b = to_nii(b).copy() # type: ignore
        b.nii = Nifti1Image(b.get_array()*0+1,affine=b.affine)
        b.seg = True
        b.set_dtype_(np.uint8)
        b.c_val = 0
        b = b.resample_from_to(self,c_val=0,verbose=False,mode="constant") # type: ignore
        return b.get_array().sum()

    def extract_background(self,inplace=False):
        assert self.seg, "extracting the background only makes sense for a segmentation mask"
        arr_bg = self.get_seg_array()
        arr_bg = np_extract_label(arr_bg, label=0, to_label=1)
        return self.set_array(arr_bg, inplace, False)

    def extract_label(self,label:int|Enum|Sequence[int]|Sequence[Enum]|None, keep_label=False,inplace=False):
        '''If this NII is a segmentation you can single out one label with [0,1].'''
        assert self.seg, "extracting a label only makes sense for a segmentation mask"
        if label is None:
            if keep_label:
                return self.copy() if inplace else self
            else:
                return self.clamp(0,1,inplace=inplace)
        seg_arr = self.get_seg_array()

        if isinstance(label, Sequence):
            label_int:list[int] = [idx.value if isinstance(idx,Enum) else idx for idx in label]
            assert 0 not in label_int, 'Zero label does not make sense. This is the background'
            seg_arr = np_extract_label(seg_arr, label_int, to_label=1, inplace=True)
        else:
            if isinstance(label,Enum):
                label = label.value
            if isinstance(label,str):
                label = int(label)

            assert label != 0, 'Zero label does not make sense. This is the background'
            seg_arr = np_extract_label(seg_arr, label, to_label=1, inplace=True)
        if keep_label:
            seg_arr = seg_arr * self.get_seg_array()
        return self.set_array(seg_arr,inplace=inplace)
    def extract_label_(self,label:int|Location|Sequence[int]|Sequence[Location], keep_label=False):
        return self.extract_label(label,keep_label,inplace=True)
    def remove_labels(self,label:int|Location|Sequence[int]|Sequence[Location], inplace=False, verbose:logging=True, removed_to_label=0):
        '''If this NII is a segmentation you can single out one label.'''
        assert label != 0, 'Zero label does not make sens.  This is the background'
        seg_arr = self.get_seg_array()
        if not isinstance(label,Sequence):
            label = [label] # type: ignore
        for l in label:
            if isinstance(l, list):
                for g in l:
                    seg_arr[seg_arr == g] = removed_to_label
            else:
                seg_arr[seg_arr == l] = removed_to_label
        return self.set_array(seg_arr,inplace=inplace, verbose=verbose)
    def remove_labels_(self,label:int|Location|Sequence[int]|Sequence[Location], verbose:logging=True):
        return self.remove_labels(label,inplace=True,verbose=verbose)
    def apply_mask(self,mask:Self, inplace=False):
        assert mask.shape == self.shape, f"[def apply_mask] Mask and Shape are not equal: \nMask - {mask},\nSelf - {self})"
        seg_arr = mask.get_seg_array()
        seg_arr[seg_arr != 0] = 1
        arr = self.get_array()
        return self.set_array(arr*seg_arr,inplace=inplace)

    def unique(self,verbose:logging=False):
        '''Returns all integer labels WITHOUT 0. Must be performed only on a segmentation nii'''
        out = np_unique_withoutzero(self.get_seg_array())
        log.print(out,verbose=verbose)
        return out
    def voxel_volume(self):

        product = math.prod(self.spacing)
        return product

    def volumes(self, include_zero: bool = False, in_mm3=False,sort=False) -> dict[int, float]|dict[int, int]:
        '''Returns a dict stating how many pixels are present for each label'''
        dic =  np_volume(self.get_seg_array(), include_zero=include_zero)
        if sort:
            dic = dict(sorted(dic.items()))
        if in_mm3:
            voxel_size = self.voxel_volume()
            dic = {k:v*voxel_size for k,v in dic.items()}
        return dic
    def translate_arr(
        self,
        translation_vector: tuple[int, int, int] | dict[str,int]| dict[DIRECTIONS,int],
        inplace: bool = False,
        verbose: bool = True
    ):
        """
        Translates the NIfTI image array by a given vector. Translation can be specified as a tuple
        or as a dict using anatomical directions ('S', 'I', 'R', 'L', 'P', 'A').

        Args:
            translation_vector (tuple or dict): Vector to translate with. Can be a tuple of ints
                                                or a dict with keys from 'SIRLPA' and int values.
            inplace (bool): Whether to modify the array in place.
            verbose (bool): Whether to print log messages.

        Returns:
            NII: Translated image.
        """
        log.print("Translating array", end="\r", verbose=verbose)
        arr = self.get_array()

        if isinstance(translation_vector, dict):
            # Start with zero translation for all axes
            vector = [0] * arr.ndim
            for direction, amount in translation_vector.items():
                axis = self.get_axis(direction=direction) # type: ignore
                sign = +1 if direction in self.orientation else -1
                vector[axis] += sign * amount
            v: tuple[int, int, int] = tuple(vector) # type: ignore
        else:
            v = translation_vector
        arr_translated = np_translate_arr(arr, v)
        log.print(f"Translated by {v}; ", verbose=verbose)
        return self.set_array(arr_translated, inplace=inplace)

    def center_of_masses(self) -> dict[int, COORDINATE]:
        '''Returns a dict stating the center of mass for each present label (not including zero!)'''
        return np_center_of_mass(self.get_seg_array())




def to_nii_optional(img_bids: Image_Reference|None, seg=False, default=None) -> NII | None:
    if img_bids is None:
        return default
    try:
        return to_nii(img_bids,seg=seg)
    except ValueError:
        return default
    except KeyError:
        return default


def to_nii(img_bids: Image_Reference, seg=False) -> NII:
    if isinstance(img_bids, Path):
        img_bids = str(img_bids)
    # ugly workaround due to module import issues with nii files
    if isinstance(img_bids, NII) or img_bids.__class__.__name__ == "NII":
        return img_bids.copy()
    elif isinstance(img_bids, bids_files.BIDS_FILE):
        return img_bids.open_nii()

    elif isinstance(img_bids, str):
        ending = img_bids.split(".")[-1]
        if ending in ("mha",):
            import SimpleITK as sitk  # noqa: N813
            img = sitk.ReadImage(img_bids)
            from TPTBox.core.sitk_utils import sitk_to_nii
            return sitk_to_nii(img,seg)
        if ending in ("nrrd",):
            return NII.load_nrrd(img_bids,seg=seg)
        return NII(nib.load(img_bids), seg) #type: ignore
    elif isinstance(img_bids, Nifti1Image):
        return NII(img_bids, seg)
    else:
        raise TypeError(type(img_bids))

def to_nii_seg(img: Image_Reference) -> NII:
    return to_nii(img,seg=True)

def to_nii_interpolateable(i_img:Interpolateable_Image_Reference) -> NII:

    if isinstance(i_img,tuple):
        img, seg = i_img
        return to_nii(img,seg=seg)
    elif isinstance(i_img, NII):
        return i_img
    elif isinstance(i_img, bids_files.BIDS_FILE):
        return i_img.open_nii()
    else:
        raise TypeError("to_nii_interpolateable",i_img)

def _rescale_affine(affine, shape, zooms, new_shape=None):
    shape = np.asarray(shape)
    new_shape = np.array(new_shape if new_shape is not None else shape)
    s = nib.affines.voxel_sizes(affine)
    n = len(zooms)
    if len(shape)>= n:
        shape = shape[:n]
        new_shape = new_shape[:n]
    rzs_out = affine[:n, :n] * zooms / s

    # Using xyz = A @ ijk, determine translation
    centroid = nib.affines.apply_affine(affine, (shape - 1) // 2)
    t_out = centroid - rzs_out @ ((new_shape - 1) // 2)
    return nib.affines.from_matvec(rzs_out, t_out)
