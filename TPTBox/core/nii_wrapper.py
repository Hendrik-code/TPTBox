import traceback
import warnings
import zlib
from collections.abc import Sequence
from enum import Enum
from math import ceil, floor
from pathlib import Path
from typing import Any, Literal, TypeVar

import nibabel as nib
import nibabel.orientations as nio
import nibabel.processing as nip
import numpy as np
from nibabel import Nifti1Header, Nifti1Image  # type: ignore
from typing_extensions import Self

from TPTBox.core.nii_wrapper_math import NII_Math
from TPTBox.core.np_utils import (
    np_calc_boundary_mask,
    np_calc_convex_hull,
    np_calc_overlapping_labels,
    np_center_of_mass,
    np_connected_components,
    np_dilate_msk,
    np_erode_msk,
    np_fill_holes,
    np_get_connected_components_center_of_mass,
    np_get_largest_k_connected_components,
    np_map_labels,
    np_unique,
    np_unique_withoutzero,
    np_volume,
)
from TPTBox.core.vert_constants import COORDINATE
from TPTBox.logger.log_file import Log_Type

from . import bids_files
from .vert_constants import (
    AFFINE,
    AX_CODES,
    DIRECTIONS,
    LABEL_MAP,
    LABEL_REFERENCE,
    SHAPE,
    ZOOMS,
    Location,
    Sentinel,
    log,
    logging,
    v_name2idx,
)

_unpacked_nii = tuple[np.ndarray, AFFINE, nib.nifti1.Nifti1Header]
_formatwarning = warnings.formatwarning


def formatwarning_tb(*args, **kwargs):
    s = "####################################\n"
    s += _formatwarning(*args, **kwargs)
    tb = traceback.format_stack()[:-3]
    s += "".join(tb[:-1])
    s += "####################################\n"
    return s


_dtyp_max = {"int8": 128, "uint8": 256, "int16": 32768, "uint16": 65536}

warnings.formatwarning = formatwarning_tb

N = TypeVar("N", bound="NII")
Image_Reference = bids_files.BIDS_FILE | Nifti1Image | Path | str | N
Interpolateable_Image_Reference = bids_files.BIDS_FILE | tuple[Nifti1Image, bool] | tuple[Path, bool] | tuple[str, bool] | N

Proxy = tuple[tuple[int, int, int], np.ndarray]
suppress_dtype_change_printout_in_set_array = False
# fmt: off

class NII(NII_Math):
    """
    The `NII` class represents a NIfTI image and provides various methods for manipulating and analyzing NIfTI images. It supports loading and saving NIfTI images, rescaling and reorienting images, applying operations on segmentation masks, and more.

    Example Usage:
    ```python
    # Create an instance of NII class
    nii = NII(nib.load('image.nii.gz'))

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
    def __init__(self, nii: Nifti1Image|_unpacked_nii, seg=False,c_val=None, desc:str|None=None) -> None:
        assert nii is not None
        self.__divergent = False
        self._checked_dtype = False
        self.nii = nii
        self.seg:bool = seg
        self.c_val:float|None=c_val # default c_vale if seg is None
        self.__min = None
        self.set_description(desc)


    @classmethod
    def load(cls, path: Image_Reference, seg, c_val=None):
        nii= to_nii(path,seg)
        nii.c_val = c_val
        return nii
        #return NII(nib.load(path), seg, c_val) #type: ignore
    @classmethod
    def load_bids(cls, nii_bids: bids_files.BIDS_FILE):
        nifty = None
        if "nii" in nii_bids.file:
            path = nii_bids.file['nii']
            nifty = nib.load(path)
        elif "nii.gz" in nii_bids.file:
            path = nii_bids.file['nii.gz']
            nifty = nib.load(path)
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
            if self.seg:
                m = np.max(self.nii.dataobj)
                if m<256:
                    dtype = np.uint8
                elif m<65536:
                    dtype = np.uint16
                else:
                    dtype = np.int32
                self._arr = np.asanyarray(self.nii.dataobj, dtype=self.nii.dataobj.dtype).astype(dtype).copy()
                self._checked_dtype = True
            elif not self._checked_dtype:
                # if the maximum is lager than the dtype, we use float.
                self._checked_dtype = True
                dtype = str(self.dtype)
                if dtype not in _dtyp_max:
                    self._arr = np.asanyarray(self.nii.dataobj, dtype=self.nii.dataobj.dtype).copy() #type: ignore
                else:
                    m = np.max(self.nii.dataobj)
                    if m > _dtyp_max[dtype]:
                        self._arr = self.nii.get_fdata()
                    else:
                        self._arr = np.asanyarray(self.nii.dataobj, dtype=self.nii.dataobj.dtype).copy() #type: ignore
            else:
                self._arr = np.asanyarray(self.nii.dataobj, dtype=self.nii.dataobj.dtype).copy() #type: ignore

            self._aff = self.nii.affine
            self._header:Nifti1Header = self.nii.header # type: ignore
            self.__unpacked = True
        except EOFError as e:
            raise EOFError(f"{self.nii.get_filename()}: {e!s}\nThe file is probably brocken beyond repair, due killing a software during nifty saving.") from None
        except zlib.error as e:
            raise zlib.error(f"{self.nii.get_filename()}: {e!s}\nThe file is probably brocken beyond repair, due killing a software during nifty saving.") from None
        except OSError as e:
            raise zlib.error(f"{self.nii.get_filename()}: {e!s}\nThe file is probably brocken beyond repair, due killing a software during nifty saving.") from None
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
            self._nii = nii
            self.__divergent = False
        return self._nii
    @nii.setter
    def nii(self,nii:Nifti1Image|_unpacked_nii):
        if isinstance(nii,tuple):
            assert len(nii) == 3
            self.__divergent = True
            self.__unpacked = True
            arr, aff, header = nii
            self._arr = arr
            self._aff = aff
            header = header.copy()
            header.set_sform(aff, code='aligned')
            header.set_qform(aff, code='unknown')
            header.set_data_dtype(arr.dtype)
            rotation_zoom = aff[:3, :3]
            zoom = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
            header.set_zooms(zoom)
            self._header = header
            self._checked_dtype = True
        else:
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
            return self._arr.dtype
        return self.nii.dataobj.dtype #type: ignore
    @property
    def header(self) -> Nifti1Header:
        if self.__unpacked:
            return self._header
        return self.nii.header  # type: ignore
    @property
    def affine(self) -> np.ndarray:
        if self.__unpacked:
            return self._aff
        return self.nii.affine

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
    def zoom(self) -> ZOOMS:
        rotation_zoom = self.affine[:3, :3]
        zoom = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0)) if self.__divergent else self.header.get_zooms()

        z = tuple(np.round(zoom,7))
        assert len(z) == 3
        return z # type: ignore
    @property
    def origin(self) -> tuple[float, float, float]:
        z = tuple(np.round(self.affine[:3,3],7))
        assert len(z) == 3
        return z # type: ignore
    @origin.setter
    def origin(self,x:tuple[float, float, float]):
        self._unpack()
        affine = self._aff
        affine[:3,3] = np.array(x)
        self._aff = affine
    @property
    def rotation(self)->np.ndarray:
        rotation_zoom = self.affine[:3, :3]
        zoom = np.array(self.zoom)
        rotation = rotation_zoom / zoom
        return rotation


    @orientation.setter
    def orientation(self, value: AX_CODES):
        self.reorient_(value, verbose=False)

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
    def set_array(self,arr:np.ndarray, inplace=False,verbose:logging=False)-> Self:  # noqa: ARG002
        """Creates a NII where the array is replaces with the input array.

        Note: This function works "Out-of-place" by default, like all other methods.

        Args:
            arr (np.ndarray): _description_
            inplace (bool, optional): _description_. Defaults to False.

        Returns:
            self
        """
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
            self.nii = nii
            return self
        else:
            return NII(nii,self.seg) # type: ignore

    def set_array_(self,arr:np.ndarray,verbose:logging=True):
        return self.set_array(arr,inplace=True,verbose=verbose)
    def set_dtype_(self,dtype:type|Literal['smallest_int'] = np.float32):
        if dtype == "smallest_int":
            arr = self.get_array()
            if arr.max()<128:
                dtype = np.int8
            elif arr.max()<32768:
                dtype = np.int16
            else:
                dtype = np.int32

        self.nii.set_data_dtype(dtype)
        if self.nii.get_data_dtype() != self.dtype: #type: ignore
            self.nii = Nifti1Image(self.get_array().astype(dtype),self.affine,self.header)

        return self

    def global_to_local(self, x: COORDINATE):
        a = self.rotation.T @ (np.array(x) - self.origin) / np.array(self.zoom)
        return tuple(round(float(v), 7) for v in a)

    def local_to_global(self, x:  COORDINATE):
        a = self.rotation @ (np.array(x) * np.array(self.zoom)) + self.origin
        return tuple(round(float(v), 7) for v in a)

    def reorient(self:Self, axcodes_to: AX_CODES = ("P", "I", "R"), verbose:logging=False, inplace=False)-> Self:
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

        aff = self.affine
        ornt_fr = self.orientation_ornt
        arr = self.get_array()
        ornt_to = nio.axcodes2ornt(axcodes_to)
        ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
        if (ornt_fr == ornt_to).all():
            log.print("Image is already rotated to", axcodes_to,verbose=verbose)
            if inplace:
                return self
            return self.copy()
        arr = nio.apply_orientation(arr, ornt_trans)
        aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
        new_aff = np.matmul(aff, aff_trans)
        ### Reset origin ###
        flip = ornt_trans[:, 1]
        change = ((-flip) + 1) / 2  # 1 if flip else 0
        change = tuple(a * (s-1) for a, s in zip(change, self.shape, strict=False))
        new_aff[:3, 3] = nib.affines.apply_affine(aff,change) # type: ignore
        ######
        new_img = arr, new_aff,self.header
        log.print("Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to,verbose=verbose)
        if inplace:
            self.nii = new_img
            return self

        return NII(new_img, self.seg) # type: ignore
    def reorient_(self:Self, axcodes_to: AX_CODES|None = ("P", "I", "R"), verbose:logging=False) -> Self:
        if axcodes_to is None:
            return self
        return self.reorient(axcodes_to=axcodes_to, verbose=verbose,inplace=True)

    def compute_crop_slice(self,**qargs):
        import warnings
        warnings.warn("compute_crop_slice id deprecated use compute_crop instead",stacklevel=5) #TODO remove in version 1.0
        return self.compute_crop(**qargs)

    def compute_crop(self,minimum: float=0, dist: float = 0, other_crop:tuple[slice,...]|None=None, maximum_size:tuple[slice,...]|int|tuple[int,...]|None=None)->tuple[slice,slice,slice]:
        """
        Computes the minimum slice that removes unused space from the image and returns the corresponding slice tuple along with the origin shift required for centroids.

        Args:
            minimum (int): The minimum value of the array (0 for MRI, -1024 for CT). Default value is 0.
            dist (int): The amount of padding to be added to the cropped image. Default value is 0.
            other_crop (tuple[slice,...], optional): A tuple of slice objects representing the slice of an other image to be combined with the current slice. Default value is None.

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
        shp = self.shape
        zms = self.zoom
        d = np.around(dist / np.asarray(zms)).astype(int)
        array = self.get_array() #+ minimum
        msk_bin = np.zeros(array.shape,dtype=bool)
        #bool_arr[array<minimum] = 0
        msk_bin[array>minimum] = 1
        #msk_bin = np.asanyarray(bool_arr, dtype=bool)
        msk_bin[np.isnan(msk_bin)] = 0
        cor_msk = np.where(msk_bin > 0)
        if cor_msk[0].shape[0] == 0:
            raise ValueError(f'Array would be reduced to zero size; Before {self}; {self.unique()=}')
        c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
        c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
        x0 = max(0, c_min[0] - d[0])
        y0 = max(0, c_min[1] - d[1])
        z0 = max(0, c_min[2] - d[2])
        x1 = min(shp[0], c_max[0] + d[0])
        y1 = min(shp[1], c_max[1] + d[1])
        z1 = min(shp[2], c_max[2] + d[2])
        ex_slice = [slice(x0, x1+1), slice(y0, y1+1), slice(z0, z1+1)]

        if other_crop is not None:
            assert all((a.step is None) for a in other_crop), 'Only None slice is supported for combining x'
            ex_slice = [slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(ex_slice, other_crop, strict=False)]

        if maximum_size is not None:
            if isinstance(maximum_size,int):
                maximum_size = (maximum_size,maximum_size,maximum_size)
            for i, min_w in enumerate(maximum_size):
                if isinstance(min_w,slice):
                    min_w = min_w.stop - min_w.start  # noqa: PLW2901
                curr_w =  ex_slice[i].stop - ex_slice[i].start
                dif = min_w - curr_w
                if min_w > 0:
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

    def apply_center_crop(self, center_shape: tuple[int,int,int], verbose: bool = False):
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

        crop_rel_x = int(round((shp_x - crop_x) / 2.0))
        crop_rel_y = int(round((shp_y - crop_y) / 2.0))
        crop_rel_z = int(round((shp_z - crop_z) / 2.0))

        crop_slices = (slice(crop_rel_x, crop_rel_x + crop_x),slice(crop_rel_y, crop_rel_y + crop_y),slice(crop_rel_z, crop_rel_z + crop_z))
        arr_cropped = arr_padded[crop_slices]
        log.print(f"Centercropped from {arr_padded.shape} to {arr_cropped.shape}", verbose=verbose)
        shp_x, shp_y, shp_z = arr_cropped.shape
        assert crop_x == shp_x and crop_y == shp_y and crop_z == shp_z
        return self.set_array(arr_cropped)

    def apply_crop_slice(self,*args,**qargs):
        import warnings
        warnings.warn("apply_crop_slice id deprecated use apply_crop instead",stacklevel=5) #TODO remove in version 1.0
        return self.apply_crop(*args,**qargs)

    def apply_crop_slice_(self,*args,**qargs):
        import warnings
        warnings.warn("apply_crop_slice_ id deprecated use apply_crop_ instead",stacklevel=5) #TODO remove in version 1.0
        return self.apply_crop_(*args,**qargs)

    def apply_crop(self,ex_slice:tuple[slice,slice,slice]|Sequence[slice] , inplace=False):
        """
        The apply_crop_slice function applies a given slice to reduce the Nifti image volume. If a list of slices is provided, it computes the minimum volume of all slices and applies it.

        Args:
            ex_slice (tuple[slice,slice,slice] | list[tuple[slice,slice,slice]]): A tuple or a list of tuples, where each tuple represents a slice for each axis (x, y, z).
            inplace (bool, optional): If True, it applies the slice to the original image and returns it. If False, it returns a new NII object with the sliced image.
        Returns:
            NII: A new NII object containing the sliced image if inplace=False. Otherwise, it returns the original NII object after applying the slice.
        """        ''''''
        nii = self.nii.slicer[ex_slice] if ex_slice is not None else self.nii_abstract
        if inplace:
            self.nii = nii
            return self
        return self.copy(nii)

    def apply_crop_(self,ex_slice:tuple[slice,slice,slice]|Sequence[slice]):
        return self.apply_crop(ex_slice=ex_slice,inplace=True)

    def pad_to(self,target_shape:list[int]|tuple[int,int,int] | Self, mode="constant",crop=False,inplace = False):
        if isinstance(target_shape, NII):
            target_shape = target_shape.shape
        padding = []
        crop = []
        requires_crop = False
        for in_size, out_size in zip(self.shape[-3:], target_shape[-3:],strict=True):
            to_pad_size = max(0, out_size - in_size) / 2.0
            to_crop_size = -min(0, out_size - in_size) / 2.0
            padding.extend([(ceil(to_pad_size), floor(to_pad_size))])
            if to_crop_size == 0:
                crop.append(slice(None))
            else:
                end = -floor(to_crop_size)
                if end == 0:
                    end = None
                crop.append(slice(ceil(to_crop_size), end))
                requires_crop = True

        s = self
        if crop and requires_crop:
            s = s.apply_crop(tuple(crop),inplace=inplace)
        return s.apply_pad(padding,inplace=inplace,mode=mode)

    def apply_pad(self,padd:Sequence[tuple[int|None,int]],mode="constant",inplace = False):
        transform = np.eye(4, dtype=int)
        for i, (before,_) in enumerate(padd):
            #transform[i, i] = pad_slice.step if pad_slice.step is not None else 1
            transform[i, 3] = -before  if before is not None else 0
        affine = self.affine.dot(transform)
        arr = np.pad(self.get_array(),padd,mode=mode,constant_values=self.get_c_val()) # type: ignore
        nii:_unpacked_nii = (arr,affine,self.header)
        if inplace:
            self.nii = nii
            return self
        return self.copy(nii)

    def rescale_and_reorient(self, axcodes_to=None, voxel_spacing=(-1, -1, -1), verbose:logging=True, inplace=False,c_val:float|None=None,mode='constant'):

        ## Resample and rotate and Save Tempfiles
        if axcodes_to is None:
            curr = self
            ornt_img = self.orientation
            axcodes_to = nio.ornt2axcodes(ornt_img)
        else:
            curr = self.reorient(axcodes_to=axcodes_to, verbose=verbose, inplace=inplace)
        return curr.rescale(voxel_spacing=voxel_spacing, verbose=verbose, inplace=inplace,c_val=c_val,mode=mode)

    def rescale_and_reorient_(self,axcodes_to=None, voxel_spacing=(-1, -1, -1),c_val:float|None=None,mode='constant', verbose:logging=True):
        return self.rescale_and_reorient(axcodes_to=axcodes_to,voxel_spacing=voxel_spacing,c_val=c_val,mode=mode,verbose=verbose,inplace=True)

    def reorient_same_as(self, img_as: Nifti1Image | Self, verbose:logging=False, inplace=False) -> Self:
        axcodes_to: AX_CODES = nio.ornt2axcodes(nio.io_orientation(img_as.affine)) # type: ignore
        return self.reorient(axcodes_to=axcodes_to, verbose=verbose, inplace=inplace)
    def reorient_same_as_(self, img_as: Nifti1Image | Self, verbose:logging=False) -> Self:
        return self.reorient_same_as(img_as=img_as,verbose=verbose,inplace=True)
    def rescale(self, voxel_spacing=(1, 1, 1), c_val:float|None=None, verbose:logging=False, inplace=False,mode='constant',align_corners:bool=False):
        """
        Rescales the NIfTI image to a new voxel spacing.

        Args:
            voxel_spacing (tuple[float, float, float]): The desired voxel spacing in millimeters (x, y, z). -1 is keep the voxel spacing.
                Defaults to (1, 1, 1).
            c_val (float | None, optional): The padding value. Defaults to None, meaning that the padding value will be
                inferred from the image data.
            verbose (bool, optional): Whether to print a message indicating that the image has been resampled. Defaults to
                False.
            inplace (bool, optional): Whether to modify the current object or return a new one. Defaults to False.
            mode (str, optional): One of the supported modes by scipy.ndimage.interpolation (e.g., "constant", "nearest",
                "reflect", "wrap"). See the documentation for more details. Defaults to "constant".
            align_corners (bool|default): If True or not set and seg==True. Aline corners for scaling. This prevents segmentation mask to shift in a direction.
        Returns:
            NII: A new NII object with the resampled image data.
        """
        if voxel_spacing in ((-1, -1, -1), self.zoom):
            log.print(f"Image already resampled to voxel size {self.zoom}",verbose=verbose)
            return self.copy() if inplace else self

        c_val = self.get_c_val(c_val)
        # resample to new voxel spacing based on the current x-y-z-orientation
        aff = self.affine
        shp = self.shape
        zms = self.zoom
        order = 0 if self.seg else 3
        voxel_spacing = tuple([v if v != -1 else z for v,z in zip(voxel_spacing,zms,strict=True)])
        if voxel_spacing == self.zoom:
            return self.copy() if inplace else self

        # Calculate new shape
        new_shp = tuple(np.rint([shp[i] * zms[i] / voxel_spacing[i] for i in range(len(voxel_spacing))]).astype(int))
        new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)  # type: ignore
        new_aff[:3, 3] = nib.affines.apply_affine(aff, [0, 0, 0])# type: ignore
        new_img = _resample_from_to(self, (new_shp, new_aff,voxel_spacing), order=order, mode=mode,align_corners=align_corners)
        log.print(f"Image resampled from {zms} to voxel size {voxel_spacing}",verbose=verbose)
        if inplace:
            self.nii = new_img
            return self
        return NII(new_img, self.seg,self.c_val)

    def rescale_(self, voxel_spacing=(1, 1, 1), c_val:float|None=None, verbose:logging=False,mode='constant'):
        return self.rescale( voxel_spacing=voxel_spacing, c_val=c_val, verbose=verbose,mode=mode, inplace=True)

    def resample_from_to(self, to_vox_map:Image_Reference|Proxy, mode='constant', c_val=None, inplace = False,verbose:logging=True,align_corners:bool=False):
        """self will be resampled in coordinate of given other image. Adheres to global space not to local pixel space
        Args:
            to_vox_map (Image_Reference|Proxy): If object, has attributes shape giving input voxel shape, and affine giving mapping of input voxels to output space. If length 2 sequence, elements are (shape, affine) with same meaning as above. The affine is a (4, 4) array-like.\n
            mode (str, optional): Points outside the boundaries of the input are filled according to the given mode ('constant', 'nearest', 'reflect' or 'wrap').Defaults to 'constant'.\n
            cval (float, optional): Value used for points outside the boundaries of the input if mode='constant'. Defaults to 0.0.\n
            aline_corners (bool|default): If True or not set and seg==True. Aline corners for scaling. This prevents segmentation mask to shift in a direction.
            inplace (bool, optional): Defaults to False.

        Returns:
            NII:
        """        ''''''
        c_val = self.get_c_val(c_val)

        mapping = to_nii_optional(to_vox_map,seg=self.seg,default=to_vox_map)
        assert mapping is not None
        log.print(f"resample_from_to: {self} to {mapping}",verbose=verbose)
        nii = _resample_from_to(self, mapping,order=0 if self.seg else 3, mode=mode,align_corners=align_corners)
        if inplace:
            self.nii = nii
            return self
        else:
            return NII(nii,self.seg,self.c_val)
    def resample_from_to_(self, to_vox_map:Image_Reference|Proxy, mode='constant', c_val:float|None=None,verbose:logging=True,aline_corners=False):
        return self.resample_from_to(to_vox_map,mode=mode,c_val=c_val,inplace=True,verbose=verbose,align_corners=aline_corners)

    def n4_bias_field_correction(
        self,
        threshold = 60,
        mask=None,
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
        assert self.seg is False, "n4 bias field correction on a segmentation doesnt make any sense"
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
            mask = mask.set_spacing(input_ants.spacing)
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
        return self.n4_bias_field_correction(mask_arr=mask,shrink_factor=shrink_factor,convergence=convergence,spline_param=spline_param,verbose=verbose,weight_mask=weight_mask,crop=crop,inplace=True,threshold = threshold)

    def normalize_to_range_(self, min_value: int = 0, max_value: int = 1500, verbose:logging=True):
        assert not self.seg
        mi, ma = self.min(), self.max()
        self += -mi + min_value  # min = 0  # noqa: PLW0642
        self_dtype = self.dtype
        max_value2 = ma
        if max_value2 > max_value:
            self *= max_value / max_value2  # noqa: PLW0642
            self.set_dtype_(self_dtype)
        log.print(f"Shifted from range {mi, ma} to range {self.min(), self.max()}", verbose=verbose)

    def match_histograms(self, reference:Image_Reference,c_val = 0,inplace=False):
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
        assert self.seg is False, "You really want to smooth a segmentation?"
        from scipy.ndimage import gaussian_filter
        arr = gaussian_filter(self.get_array(), sigma, order=nth_derivative,cval=self.get_c_val(), truncate=truncate)# radius=None, axes=None
        return self.set_array(arr,inplace,verbose=False)

    def smooth_gaussian_(self, sigma:float|list[float]|tuple[float],truncate=4.0,nth_derivative=0):
        return self.smooth_gaussian(sigma=sigma,truncate=truncate,nth_derivative=nth_derivative,inplace=True)

    def to_ants(self):
        import ants
        return ants.from_nibabel(self.nii)


    def erode_msk(self, mm: int = 5, labels: LABEL_REFERENCE = None, connectivity: int = 3, inplace=False,verbose:logging=True,border_value=0):
        """
        Erodes the binary segmentation mask by the specified number of voxels.

        Args:
            mm (int, optional): The number of voxels to erode the mask by. Defaults to 5.
            labels (LABEL_REFERENCE, optional): Labels that should be dilated. If None, will erode all labels (not including zero!)
            connectivity (int, optional): Elements up to a squared distance of connectivity from the center are considered neighbors. connectivity may range from 1 (no diagonal elements are neighbors) to rank (all elements are neighbors).
            inplace (bool, optional): Whether to modify the mask in place or return a new object. Defaults to False.
            verbose (bool, optional): Whether to print a message indicating that the mask was eroded. Defaults to True.

        Returns:
            NII: The eroded mask.

        Notes:
            The method uses binary erosion with a 3D structuring element to erode the mask by the specified number of voxels.

        """
        log.print("erode mask",end='\r',verbose=verbose)
        labels = self.unique() if labels is None else labels
        msk_i_data = self.get_seg_array()
        out = np_erode_msk(msk_i_data, label_ref=labels, mm=mm, connectivity=connectivity,border_value=border_value)
        msk_e = out.astype(np.uint16), self.affine, self.header
        log.print("Mask eroded by", mm, "voxels",verbose=verbose)
        if inplace:
            self.nii = msk_e
            return self
        return NII(msk_e,seg=True,c_val=0)

    def erode_msk_(self, mm:int = 5, labels: LABEL_REFERENCE = None, connectivity: int=3, verbose:logging=True,border_value=0):
        return self.erode_msk(mm=mm, labels=labels, connectivity=connectivity, inplace=True, verbose=verbose,border_value=border_value)

    def dilate_msk(self, mm: int = 5, labels: LABEL_REFERENCE = None, connectivity: int = 3, mask: Self | None = None, inplace=False, verbose:logging=True):
        """
        Dilates the binary segmentation mask by the specified number of voxels.

        Args:
            mm (int, optional): The number of voxels to dilate the mask by. Defaults to 5.
            labels (list[int], optional): Labels that should be dilated. If None, will dilate all labels (not including zero!)
            connectivity (int, optional): Elements up to a squared distance of connectivity from the center are considered neighbors. connectivity may range from 1 (no diagonal elements are neighbors) to rank (all elements are neighbors).
            mask (NII, optional): If set, after each iteration, will zero out everything based on this mask
            inplace (bool, optional): Whether to modify the mask in place or return a new object. Defaults to False.
            verbose (bool, optional): Whether to print a message indicating that the mask was dilated. Defaults to True.

        Returns:
            NII: The dilated mask.

        Notes:
            The method uses binary dilation with a 3D structuring element to dilate the mask by the specified number of voxels.

        """
        log.print("dilate mask",end='\r',verbose=verbose)
        if labels is None:
            labels = self.unique()
        msk_i_data = self.get_seg_array()
        mask_ = mask.get_seg_array() if mask is not None else None
        out = np_dilate_msk(arr=msk_i_data, label_ref=labels, mm=mm, mask=mask_, connectivity=connectivity)
        msk_e = out.astype(np.uint16), self.affine,self.header
        log.print("Mask dilated by", mm, "voxels",verbose=verbose)
        if inplace:
            self.nii = msk_e
            return self
        return NII(msk_e,seg=True,c_val=0)

    def dilate_msk_(self, mm:int = 5, labels: LABEL_REFERENCE = None, connectivity: int=3, mask: Self | None = None, verbose:logging=True):
        return self.dilate_msk(mm=mm, labels=labels, connectivity=connectivity, mask=mask, inplace=True, verbose=verbose)


    def fill_holes(self, labels: LABEL_REFERENCE = None, slice_wise_dim: int | None = None, verbose:logging=True, inplace=False):
        """Fills holes in segmentation

        Args:
            labels (LABEL_REFERENCE, optional): Labels that the hole-filling should be applied to. If none, applies on all labels found in arr. Defaults to None.
            verbose: whether to print which labels have been filled
            inplace (bool): Whether to modify the current NIfTI image object in place or create a new object with the mapped labels.
                Default is False.
            slice_wise_dim (int | None, optional): If the input is 3D, the specified dimension here cna be used for 2D slice-wise filling. Defaults to None.

        Returns:
            NII: If inplace is True, returns the current NIfTI image object with filled holes. Otherwise, returns a new NIfTI image object with filled holes.
        """
        if labels is None:
            labels = list(self.unique())
        if isinstance(labels, int):
            labels = [labels]

        seg_arr = self.get_seg_array()
        #volumes = np_volume(seg_arr, label_ref=labels)
        filled = np_fill_holes(seg_arr, label_ref=labels, slice_wise_dim=slice_wise_dim)
        #volumes_filled = np_volume(filled, label_ref=labels)
        #changes_in_labels = [i for i in labels if i in volumes_filled and i in volumes and volumes_filled[i] != volumes[i]]
        #if len(changes_in_labels) > 0:
        #    log.print(f"Filled holes in {changes_in_labels}", verbose=verbose)
        #else:
        #    log.print("Fill holes: No holes have been filled", verbose=verbose)
        log.print("Fill holes called", verbose=verbose)
        return self.set_array(filled, inplace=inplace)

    def fill_holes_(self, labels: LABEL_REFERENCE = None, slice_wise_dim: int | None = None, verbose:logging=True):
        return self.fill_holes(labels, slice_wise_dim, verbose, inplace=True)

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

    def calc_convex_hull_(self, axis: DIRECTIONS="S", verbose: bool = False,):
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

    def get_segmentation_connected_components(self, labels: int |list[int], connectivity: int = 3, transform_back_to_nii: bool = False, verbose: bool=False):
        """Calculates and returns the connected components of this segmentation NII

        Args:
            label (int): the label(s) of the connected components
            connectivity (int, optional): Connectivity for the connected components. Defaults to 3.
            transform_back_to_nii (bool): If True, will map the labels to niftys, not numpy arrays. Defaults to False.

        Returns:
            cc: dict[label, cc_idx, arr], cc_n: dict[label, int]
        """
        arr = self.get_seg_array()
        cc, cc_n = np_connected_components(arr, connectivity=connectivity, label_ref=labels, verbose=verbose)
        if transform_back_to_nii:
            cc = {i: self.set_array(k) for i,k in cc.items()}
        return cc, cc_n

    def filter_connected_components(self, labels: int |list[int]|None,min_volume:int|None=None,max_volume:int|None=None, max_count_component = None, connectivity: int = 3,removed_to_label=0):
        """
        Filter connected components in a segmentation array based on specified volume constraints.

        Parameters:
        labels (int | list[int]): The labels of the components to filter.
        min_volume (int | None): Minimum volume for a component to be retained. Components smaller than this will be removed.
        max_volume (int | None): Maximum volume for a component to be retained. Components larger than this will be removed.
        max_count_component (int | None): Maximum number of components to retain. Once this limit is reached, remaining components will be removed.
        connectivity (int): Connectivity criterion for defining connected components (default is 3).
        removed_to_label (int): Label to assign to removed components (default is 0).

        Returns:
        None
        """
        arr = self.get_seg_array()
        nii = self.get_largest_k_segmentation_connected_components(None,labels,connectivity=connectivity,return_original_labels=False)
        for k, idx in enumerate(nii.unique(),start=1):
            msk = nii.extract_label(idx)
            nii *=(-msk+1)
            s = msk.sum()
            #print(idx,k,s)
            if min_volume is not None and s < min_volume:
                arr[msk.get_array()!=0] = removed_to_label
                arr[nii.get_array()!=0] = removed_to_label # set all future to 0
                break
            if max_volume is not None and s>max_volume:
                arr[msk.get_array()==1] = removed_to_label
            if max_count_component is not None and k == max_count_component:
                arr[nii.get_array()!=0] = removed_to_label # set all future to 0
                break
        #print("Finish")
        return self.set_array(arr)
    def get_segmentation_connected_components_center_of_mass(self, label: int, connectivity: int = 3, sort_by_axis: int | None = None):
        """Calculates the center of mass of the different connected components of a given label in an array

        Args:
            label (int): the label of the connected components
            connectivity (int, optional): Connectivity for the connected components. Defaults to 3.
            sort_by_axis (int | None, optional): If not none, will sort the center of mass list by this axis values. Defaults to None.

        Returns:
            _type_: _description_
        """
        arr = self.get_seg_array()
        return np_get_connected_components_center_of_mass(arr, label=label, connectivity=connectivity, sort_by_axis=sort_by_axis)


    def get_largest_k_segmentation_connected_components(self, k: int | None, labels: int | list[int] | None = None, connectivity: int = 1, return_original_labels: bool = True):
        """Finds the largest k connected components in a given array (does NOT work with zero as label!)

        Args:
            arr (np.ndarray): input array
            k (int | None): finds the k-largest components. If k is None, will find all connected components and still sort them by size
            labels (int | list[int] | None, optional): Labels that the algorithm should be applied to. If none, applies on all labels found in this NII. Defaults to None.
            return_original_labels (bool): If set to False, will label the components from 1 to k. Defaults to True
        """
        return self.set_array(np_get_largest_k_connected_components(self.get_seg_array(), k=k, label_ref=labels, connectivity=connectivity, return_original_labels=return_original_labels))


    def get_segmentation_difference_to(self, mask_gt: Self, ignore_background_tp: bool = False) -> Self:
        """Calculates an NII that represents the segmentation difference between self and given groundtruth mask

        Args:
            mask_groundtruth (Self): The ground truth mask. Must match in orientation, zoom, and shape

        Returns:
            NII: Difference NII (1: FN, 2: TP, 3: FP, 4: Wrong label)
        """
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
        mask_other: Self,
    ) -> list[tuple[int, int]]:
        """Calculates the pairs of labels that are overlapping in at least one voxel (fast)

        Args:
            mask_other (NII): The array to be compared with.

        Returns:
            list[tuple[int, int]]: List of tuples of labels that overlap in at least one voxel. First label in the tuple is Self NII, the second is of the mask_other
        """
        assert self.seg and mask_other.seg
        return np_calc_overlapping_labels(self.get_seg_array(), mask_other.get_seg_array())


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
        return NII(nii, True)

    def map_labels_(self, label_map: LABEL_MAP, verbose:logging=True):
        return self.map_labels(label_map,verbose=verbose,inplace=True)
    def copy(self, nib:Nifti1Image|_unpacked_nii|None = None):
        if nib is None:
            return NII((self.get_array(), self.affine.copy(), self.header.copy()),seg=self.seg,c_val = self.c_val)
        else:
            return NII(nib,seg=self.seg,c_val = self.c_val)
    def clone(self):
        return self.copy()
    def save(self,file:str|Path|bids_files.BIDS_FILE,make_parents=True,verbose:logging=True, dtype = None):
        if isinstance(file, bids_files.BIDS_FILE):
            file = file.file['nii.gz']
        if make_parents:
            Path(file).parent.mkdir(exist_ok=True,parents=True)
        arr = self.get_array()
        out = Nifti1Image(arr, self.affine,self.header)#,dtype=arr.dtype)
        if dtype is not None:
            out.set_data_dtype(dtype)
        elif self.seg:
            if arr.max()<256:
                out.set_data_dtype(np.uint8)
            elif arr.max()<65536:
                out.set_data_dtype(np.uint16)
            else:
                out.set_data_dtype(np.int32)
        log.print(f"Save {file} as {out.get_data_dtype()}",verbose=verbose,ltype=Log_Type.SAVE)
        nib.save(out, file) #type: ignore
    def __str__(self) -> str:
        return f"shp={self.shape}; ori={self.orientation}, zoom={tuple(np.around(self.zoom, decimals=2))}, seg={self.seg}" # type: ignore
    def __repr__(self)-> str:
        return self.__str__()
    def __array__(self,dtype=None):
            self._unpack()
            if dtype is None:
                return self._arr
            else:
                return self._arr.astype(dtype, copy=False)
    def __array_wrap__(self, array):
        if array.shape != self.shape:
            raise SyntaxError(f"Function call induce a shape change of nii image. Before {self.shape} after {array.shape}.")
        return self.set_array(array)
    def __getitem__(self, key)-> Any:
        if isinstance(key,Sequence):
            from types import EllipsisType

            if all(isinstance(k, (slice,EllipsisType)) for k in key):
                #if all(k.step is not None and k.step == 1 for k in key):
                #    raise NotImplementedError(f"Slicing is not implemented. Attemted {key}")
                if len(key)!= len(self.shape) or Ellipsis in key:
                    raise ValueError(f"Number slices must have exact number of slices like in dimension. Attemted: {key} - Shape {self.shape}")
                return self.apply_crop(key) # type: ignore
            elif  all(isinstance(k, int) for k in key):
                if len(key)!= len(self.shape):
                    raise ValueError(f"Number ints must have exact number of slices like in dimension. Attemted: {key} - Shape {self.shape}")
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
        else:
            raise TypeError("Invalid argument type:", type(key))
    def __setitem__(self, key,value):
        if isinstance(key,self.__class__):
            key = key.get_array()==1
        self._unpack()
        self._arr[key] = value
        #if isinstance(key,Sequence):
        #    if all(isinstance(k, slice) for k in key):
        #        #if all(k.step is not None and k.step == 1 for k in key):
        #        #    raise NotImplementedError(f"Slicing is not implemented. Attemted {key}")
        #        if len(key)!= len(self.shape):
        #            raise ValueError(f"Number slices must have exact number of slices like in dimension. Attemted: {key} - Shape {self.shape}")
        #        return self.apply_crop(key)
        #    elif  all(isinstance(k, int) for k in key):
        #        if len(key)!= len(self.shape):
        #            raise ValueError(f"Number ints must have exact number of slices like in dimension. Attemted: {key} - Shape {self.shape}")
        #        self._unpack()
        #        self._arr[key] = value
        #else:
        #    raise TypeError("Invalid argument type.")

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

    def get_intersecting_volume(self, b: Self) -> bool:
        '''
        computes intersecting volume
        '''
        b = b.copy()
        b.nii = Nifti1Image(b.get_array()*0+1,affine=b.affine)
        b.seg = True
        b.set_dtype_(np.uint8)
        b = b.resample_from_to(self,c_val=0,verbose=False)
        return b.get_array().sum()

    def extract_label(self,label:int|Location|Sequence[int]|Sequence[Location], keep_label=False,inplace=False):
        '''If this NII is a segmentation you can single out one label with [0,1].'''
        seg_arr = self.get_seg_array()

        if isinstance(label, Sequence):
            label = [l.value if isinstance(l,Enum) else l for l in label]
            if 1 not in label:
                seg_arr[seg_arr == 1] = 0
            for l in label:
                seg_arr[seg_arr == l] = 1
            seg_arr[seg_arr != 1] = 0
        else:
            if isinstance(label,Location):
                label = label.value
            if isinstance(label,str):
                label = int(label)

            assert label != 0, 'Zero label does not make sens. This is the background'
            seg_arr[seg_arr != label] = 0
            seg_arr[seg_arr == label] = 1
        if keep_label:
            seg_arr = seg_arr * self.get_seg_array()
        return self.set_array(seg_arr,inplace=inplace)
    def extract_label_(self,label:int|Location|Sequence[int]|Sequence[Location], keep_label=False):
        return self.extract_label(label,keep_label,inplace=True)
    def remove_labels(self,*label:int|Location|Sequence[int]|Sequence[Location], inplace=False, verbose:logging=True):
        '''If this NII is a segmentation you can single out one label.'''
        assert label != 0, 'Zero label does not make sens.  This is the background'
        seg_arr = self.get_seg_array()
        for l in label:
            if isinstance(l, list):
                for g in l:
                    seg_arr[seg_arr == g] = 0
            else:
                seg_arr[seg_arr == l] = 0
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

    def volumes(self, include_zero: bool = False) -> dict[int, int]:
        '''Returns a dict stating how many pixels are present for each label'''
        return np_volume(self.get_seg_array(), include_zero=include_zero)

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
    if isinstance(img_bids, NII):
        return img_bids.copy()
    elif isinstance(img_bids, bids_files.BIDS_FILE):
        return img_bids.open_nii()

    elif isinstance(img_bids, str):
        if img_bids.split(".")[-1] in ("mha",):
            import SimpleITK as sitk  # noqa: N813
            img = sitk.ReadImage(img_bids)
            from TPTBox.core.sitk_utils import sitk_to_nii
            return sitk_to_nii(img,seg)
        return NII(nib.load(img_bids), seg) #type: ignore
    elif isinstance(img_bids, Nifti1Image):
        return NII(img_bids, seg)
    else:
        raise TypeError(img_bids)

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


def _resample_from_to(
    from_img:NII,
    to_img:NII|tuple[SHAPE,AFFINE,ZOOMS],
    order=3,
    mode="constant",
    align_corners:bool|Sentinel=Sentinel()  # noqa: B008
):
    import numpy.linalg as npl
    import scipy.ndimage as scipy_img
    from nibabel.affines import AffineError, to_matvec
    from nibabel.imageclasses import spatial_axes_first

    # This check requires `shape` attribute of image
    if not spatial_axes_first(from_img.nii):
        raise ValueError(f"Cannot predict position of spatial axes for Image type {type(from_img)}")
    if isinstance(to_img,tuple):
        to_shape, to_affine, zoom_to = to_img
    else:
        assert to_img.affine is not None
        assert to_img.zoom is not None
        to_shape:SHAPE = to_img.shape
        to_affine:AFFINE = to_img.affine
        zoom_to = np.array(to_img.zoom)
    from_n_dim = len(from_img.shape)
    if from_n_dim < 3:
        raise AffineError("from_img must be at least 3D")
    if (isinstance(align_corners,Sentinel) and order == 0) or align_corners:
        # https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/6
        # https://discuss.pytorch.org/uploads/default/original/2X/6/6a242715685b8192f07c93a57a1d053b8add97bf.png
        # Simulate align_corner=True, by manipulating the affine
        # Updated to matrix:
        # make the output by one voxel larger
        # z_new = z * num_pixel/(num_pixel+1)
        to_affine_new = to_affine.copy()
        num_pixel = np.array(to_shape)
        zoom_new = zoom_to*num_pixel/(1+num_pixel)
        rotation_zoom = to_affine[:3, :3]
        to_affine_new[:3, :3] = rotation_zoom / np.array(zoom_to)*zoom_new
        ## Shift origin to corner
        corner = np.array([-0.5,-0.5,-0.5,0])
        to_affine_new[:,3] -=to_affine_new@corner
        # Update from matrix
        # z_new = z * num_pixel/(num_pixel+1)
        zoom_from = np.array(from_img.zoom)
        from_affine_new = from_img.affine.copy()
        num_pixel = np.array(from_img.shape)
        zoom_new = zoom_from*num_pixel/(1+num_pixel)
        rotation_zoom = from_img.affine[:3, :3]
        from_affine_new[:3, :3] = rotation_zoom / np.array(zoom_from)*zoom_new
        ## Shift origin to corner
        from_affine_new[:,3] -=from_affine_new@corner

        a_to_affine = nip.adapt_affine(to_affine_new, len(to_shape))
        a_from_affine = nip.adapt_affine(from_affine_new, from_n_dim)
    else:
        a_to_affine = nip.adapt_affine(to_affine, len(to_shape))
        a_from_affine = nip.adapt_affine(from_img.affine, from_n_dim)
    to_vox2from_vox = npl.inv(a_from_affine).dot(a_to_affine)
    rzs, trans = to_matvec(to_vox2from_vox)

    data = scipy_img.affine_transform(from_img.get_array(), rzs, trans, to_shape, order=order, mode=mode, cval=from_img.get_c_val()) # type: ignore
    return data, to_affine, from_img.header
