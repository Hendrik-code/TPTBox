import traceback
import warnings
from collections.abc import Sequence
from enum import Enum
from math import ceil, floor
from pathlib import Path
from typing import Literal, TypeVar, Union

import nibabel as nib
import nibabel.orientations as nio
import nibabel.processing as nip
import numpy as np
from nibabel import Nifti1Header, Nifti1Image  # type: ignore
from typing_extensions import Self

from TPTBox.core.nii_wrapper_math import NII_Math
from TPTBox.core.np_utils import (
    np_calc_boundary_mask,
    np_connected_components,
    np_dilate_msk,
    np_erode_msk,
    np_fill_holes,
    np_get_connected_components_center_of_mass,
    np_get_largest_k_connected_components,
    np_map_labels,
    np_volume,
)

from . import bids_files
from . import vert_constants as vc

AFFINE = np.ndarray
_unpacked_nii = tuple[np.ndarray, AFFINE, nib.nifti1.Nifti1Header]
log = vc.log
Ax_Codes = vc.Ax_Codes
logging = vc.logging
v_name2idx = vc.v_name2idx
Label_Map = vc.Label_Map
Directions = vc.Directions
plane_dict = vc.plane_dict
_formatwarning = warnings.formatwarning


def formatwarning_tb(*args, **kwargs):
    s = "####################################\n"
    s += _formatwarning(*args, **kwargs)
    tb = traceback.format_stack()[:-3]
    s += "".join(tb[:-1])
    s += "####################################\n"
    return s


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
    def load_bids(cls, nii: bids_files.BIDS_FILE):
        if "nii" in nii.file:
            path = nii.file['nii']
        else:
            assert 'nii.gz' in nii.file, nii.file
            path = nii.file['nii.gz']
        if nii.get_interpolation_order() == 0:
            seg = True
            c_val=0
        else:
            seg = False
            c_val = -1024 if "ct" in nii.format.lower() else 0
        return NII(nib.load(path),seg,c_val=c_val) #type: ignore
    def _unpack(self):
        if self.__unpacked:
            return
        if self.seg:
            self._arr = np.asanyarray(self.nii.dataobj, dtype=self.nii.dataobj.dtype).astype(np.uint16).copy()

        else:
            self._arr = np.asanyarray(self.nii.dataobj, dtype=self.nii.dataobj.dtype).copy() #type: ignore

        self._aff = self.nii.affine
        self._header:Nifti1Header = self.nii.header # type: ignore
        self.__unpacked = True
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
        else:
            self.__unpacked = False
            self._nii = nii

    @property
    def shape(self) -> tuple[int, int, int]:
        if self.__unpacked:
            return tuple(self._arr.shape) # type: ignore
        return self.nii.shape
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
        self._aff = affine
    @property
    def orientation(self) -> vc.Ax_Codes:
        ort = nio.io_orientation(self.affine)
        return nio.ornt2axcodes(ort) # type: ignore

    @property
    def zoom(self) -> tuple[float, float, float]:
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
    def orientation(self, value: vc.Ax_Codes):
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
    def _extract_affine(self, rm_key=()):
        out =  {"zoom":self.zoom,"origin": self.origin, "shape": self.shape, "rotation": self.rotation, "orientation":self.orientation}
        for k in rm_key:
            out.pop(k)
        return out
    def get_array(self) -> np.ndarray:
        if self.seg:
            return self.get_seg_array()
        self._unpack()
        return self._arr.copy()
    def set_array(self,arr:np.ndarray, inplace=False,verbose:vc.logging=False)-> Self:
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
        if self.seg and isinstance(arr, (np.floating, float)):
            arr = arr.astype(np.int32)
        #if self.dtype == arr.dtype: #type: ignore
        nii:_unpacked_nii = (arr,self.affine,self.header)
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
            return NII(nii,self.seg)

    def set_array_(self,arr:np.ndarray,verbose:vc.logging=True):
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
    def global_to_local(self, x: vc.Coordinate):
        a = self.rotation.T @ (np.array(x) - self.origin) / np.array(self.zoom)
        return tuple(round(float(v), 7) for v in a)

    def local_to_global(self, x:  vc.Coordinate):
        a = self.rotation @ (np.array(x) * np.array(self.zoom)) + self.origin
        return tuple(round(float(v), 7) for v in a)


    def reorient(self:Self, axcodes_to: vc.Ax_Codes = ("P", "I", "R"), verbose:vc.logging=False, inplace=False)-> Self:
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
    def reorient_(self:Self, axcodes_to: vc.Ax_Codes|None = ("P", "I", "R"), verbose:vc.logging=False) -> Self:
        if axcodes_to is None:
            return self
        return self.reorient(axcodes_to=axcodes_to, verbose=verbose,inplace=True)

    def compute_crop_slice(self,**qargs):
        import warnings
        warnings.warn("compute_crop_slice id deprecated use compute_crop instead") #TODO remove in version 1.0
        return self.compute_crop(**qargs)

    def compute_crop(self,minimum: float=0, dist=0, other_crop:tuple[slice,...]|None=None, minimum_size:tuple[slice,...]|int|tuple[int,...]|None=None)->tuple[slice,slice,slice]:
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
            raise ValueError('Array would be reduced to zero size')
        c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
        c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
        x0 = c_min[0] - d[0] if (c_min[0] - d[0]) > 0 else 0
        y0 = c_min[1] - d[1] if (c_min[1] - d[1]) > 0 else 0
        z0 = c_min[2] - d[2] if (c_min[2] - d[2]) > 0 else 0
        x1 = c_max[0] + d[0] if (c_max[0] + d[0]) < shp[0] else shp[0]
        y1 = c_max[1] + d[1] if (c_max[1] + d[1]) < shp[1] else shp[1]
        z1 = c_max[2] + d[2] if (c_max[2] + d[2]) < shp[2] else shp[2]
        ex_slice = [slice(x0, x1+1), slice(y0, y1+1), slice(z0, z1+1)]

        if other_crop is not None:
            assert all((a.step == None) for a in other_crop), 'Only None slice is supported for combining x'
            ex_slice = [slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(ex_slice, other_crop, strict=False)]

        if minimum_size is not None:
            if isinstance(minimum_size,int):
                minimum_size = (minimum_size,minimum_size,minimum_size)
            for i, min_w in enumerate(minimum_size):
                if isinstance(min_w,slice):
                    min_w = min_w.stop - min_w.start
                curr_w =  ex_slice[i].stop - ex_slice[i].start
                dif = min_w - curr_w
                if min_w > 0:
                    new_start = ex_slice[i].start - floor(dif/2)
                    new_goal = ex_slice[i].stop + ceil(dif/2)
                    if new_goal > self.shape[i]:
                        new_start -= new_goal - self.shape[i]
                        new_goal = self.shape[i]
                    if new_start < 0:#
                        new_goal -= new_start
                        new_start = 0
                    ex_slice[i] = slice(new_start,new_goal)


        #origin_shift = tuple([int(ex_slice[i].start) for i in range(len(ex_slice))])
        return tuple(ex_slice)# type: ignore

    def apply_center_crop(self, center_shape: tuple[int,int,int], verbose: bool = False):
        shpX, shpY, shpZ = self.shape
        cropX, cropY, cropZ = center_shape

        if cropX > shpX or cropY > shpY or cropZ > shpZ:
            padding_ltrb = [
                ((cropX - shpX +1) // 2 if cropX > shpX else 0,(cropX - shpX) // 2 if cropX > shpX else 0),
                ((cropY - shpY +1) // 2 if cropY > shpY else 0,(cropY - shpY) // 2 if cropY > shpY else 0),
                ((cropZ - shpZ +1) // 2 if cropZ > shpZ else 0,(cropZ - shpZ) // 2 if cropZ > shpZ else 0),
            ]
            arr = self.get_array()
            arr_padded = np.pad(arr, padding_ltrb, "constant", constant_values=0)  # PIL uses fill value 0
            log.print(f"Pad from {self.shape} to {arr_padded.shape}", verbose=verbose)
            shpX, shpY, shpZ = arr_padded.shape
            if cropX == shpX and cropY == shpY and cropZ == shpZ:
                return self.set_array(arr_padded)

        crop_relX = int(round((shpX - cropX) / 2.0))
        crop_relY = int(round((shpY - cropY) / 2.0))
        crop_relZ = int(round((shpZ - cropZ) / 2.0))

        crop_slices = (slice(crop_relX, crop_relX + cropX),slice(crop_relY, crop_relY + cropY),slice(crop_relZ, crop_relZ + cropZ))
        arr_cropped = arr_padded[crop_slices]
        log.print(f"Centercropped from {arr_padded.shape} to {arr_cropped.shape}", verbose=verbose)
        shpX, shpY, shpZ = arr_cropped.shape
        assert cropX == shpX and cropY == shpY and cropZ == shpZ
        return self.set_array(arr_cropped)

    def apply_crop_slice(self,*args,**qargs):
        import warnings
        warnings.warn("apply_crop_slice id deprecated use apply_crop instead") #TODO remove in version 1.0
        return self.apply_crop(*args,**qargs)

    def apply_crop_slice_(self,*args,**qargs):
        import warnings
        warnings.warn("apply_crop_slice_ id deprecated use apply_crop_ instead") #TODO remove in version 1.0
        return self.apply_crop_(*args,**qargs)

    def apply_crop(self,ex_slice:tuple[slice,slice,slice]|tuple[slice,...] , inplace=False):
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
        return NII(nii,self.seg)

    def apply_crop_(self,ex_slice:tuple[slice,slice,slice]|tuple[slice,...]):
        return self.apply_crop(ex_slice=ex_slice,inplace=True)

    def rescale_and_reorient(self, axcodes_to=None, voxel_spacing=(-1, -1, -1), verbose:vc.logging=True, inplace=False,c_val:float|None=None,mode='constant'):

        ## Resample and rotate and Save Tempfiles
        if axcodes_to is None:
            curr = self
            ornt_img = self.orientation
            axcodes_to = nio.ornt2axcodes(ornt_img)
        else:
            curr = self.reorient(axcodes_to=axcodes_to, verbose=verbose, inplace=inplace)
        return curr.rescale(voxel_spacing=voxel_spacing, verbose=verbose, inplace=inplace,c_val=c_val,mode=mode)

    def rescale_and_reorient_(self,axcodes_to=None, voxel_spacing=(-1, -1, -1),c_val:float|None=None,mode='constant', verbose:vc.logging=True):
        return self.rescale_and_reorient(axcodes_to=axcodes_to,voxel_spacing=voxel_spacing,c_val=c_val,mode=mode,verbose=verbose,inplace=True)

    def reorient_same_as(self, img_as: Nifti1Image | Self, verbose:vc.logging=False, inplace=False) -> Self:
        axcodes_to: Ax_Codes = nio.ornt2axcodes(nio.io_orientation(img_as.affine)) # type: ignore
        return self.reorient(axcodes_to=axcodes_to, verbose=verbose, inplace=inplace)
    def reorient_same_as_(self, img_as: Nifti1Image | Self, verbose:vc.logging=False) -> Self:
        return self.reorient_same_as(img_as=img_as,verbose=verbose,inplace=True)
    def rescale(self, voxel_spacing=(1, 1, 1), c_val:float|None=None, verbose:vc.logging=False, inplace=False,mode='constant'):
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

        Returns:
            NII: A new NII object with the resampled image data.
        """
        if voxel_spacing in ((-1, -1, -1), self.zoom):
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
        new_img = nip.resample_from_to(self.nii, (new_shp, new_aff), order=order, cval=c_val,mode=mode)
        log.print(f"Image resampled from {zms} to voxel size {voxel_spacing}",verbose=verbose)
        if inplace:
            self.nii = new_img
            return self
        return NII(new_img, self.seg,self.c_val)

    def rescale_(self, voxel_spacing=(1, 1, 1), c_val:float|None=None, verbose:vc.logging=False,mode='constant'):
        return self.rescale( voxel_spacing=voxel_spacing, c_val=c_val, verbose=verbose,mode=mode, inplace=True)

    def resample_from_to(self, to_vox_map:Image_Reference|Proxy, mode='constant', c_val=None, inplace = False,verbose:vc.logging=True):
        """self will be resampled in coordinate of given other image. Adheres to global space not to local pixel space

        Args:
            to_vox_map (Image_Reference|Proxy): If object, has attributes shape giving input voxel shape, and affine giving mapping of input voxels to output space. If length 2 sequence, elements are (shape, affine) with same meaning as above. The affine is a (4, 4) array-like.\n
            mode (str, optional): Points outside the boundaries of the input are filled according to the given mode ('constant', 'nearest', 'reflect' or 'wrap').Defaults to 'constant'.\n
            cval (float, optional): Value used for points outside the boundaries of the input if mode='constant'. Defaults to 0.0.\n
            inplace (bool, optional): Defaults to False.

        Returns:
            NII:
        """        ''''''
        c_val = self.get_c_val(c_val)

        map = to_nii_optional(to_vox_map,seg=self.seg,default=to_vox_map)
        log.print(f"resample_from_to: {self} to {map}",verbose=verbose)

        nii = nip.resample_from_to(self.nii, map, order=0 if self.seg else 3, mode=mode, cval=c_val)
        if inplace:
            self.nii = nii
            return self
        else:
            return NII(nii,self.seg,self.c_val)
    def resample_from_to_(self, to_vox_map, mode='constant', c_val:float|None=None,verbose:logging=True):
        return self.resample_from_to(to_vox_map,mode=mode,c_val=c_val,inplace=True,verbose=verbose)

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
        assert self.seg == False, "n4 bias field correction on a segmentation doesnt make any sense"
        # install antspyx not ants!
        import ants
        import ants.utils.bias_correction as bc  # install antspyx not ants!
        from ants.utils.convert_nibabel import from_nibabel
        from scipy.ndimage import binary_dilation, generate_binary_structure
        dtype = self.dtype
        input_ants:ants.ANTsImage = from_nibabel(nib.nifti1.Nifti1Image(self.get_array(),self.affine))
        if threshold != 0:
            mask = self.get_array()
            mask[mask < threshold] = 0
            mask[mask != 0] = 1
            mask = mask.astype(np.uint8)
            struct = generate_binary_structure(3, 3)
            mask = binary_dilation(mask.copy(), structure=struct, iterations=3)
            mask = mask.astype(np.uint8)
            mask:ants.ANTsImage = from_nibabel(nib.nifti1.Nifti1Image(mask,self.affine))#self.set_array(mask,verbose=False).nii
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
        return NII(out_nib).set_dtype_(dtype)

    def n4_bias_field_correction_(self,threshold = 60,mask=None,shrink_factor=4,convergence=None,spline_param=200,verbose=False,weight_mask=None,crop=False):
        if convergence is None:
            convergence = {"iters": [50, 50, 50, 50], "tol": 1e-07}
        return self.n4_bias_field_correction(mask=mask,shrink_factor=shrink_factor,convergence=convergence,spline_param=spline_param,verbose=verbose,weight_mask=weight_mask,crop=crop,inplace=True,threshold = threshold)

    def match_histograms(self, reference:Image_Reference,c_val = 0,inplace=False):
        ref_nii = to_nii(reference)
        assert ref_nii.seg == False
        assert self.seg == False
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
        assert self.seg == False, "You really want to smooth a segmentation?"
        from scipy.ndimage import gaussian_filter
        arr = gaussian_filter(self.get_array(), sigma, order=nth_derivative,cval=self.get_c_val(), truncate=truncate)# radius=None, axes=None
        return self.set_array(arr,inplace,verbose=False)

    def smooth_gaussian_(self, sigma:float|list[float]|tuple[float],truncate=4.0,nth_derivative=0):
        return self.smooth_gaussian(sigma=sigma,truncate=truncate,nth_derivative=nth_derivative,inplace=True)

    def to_ants(self):
        import ants
        return ants.from_nibabel(self.nii)
    def get_plane(self) -> str:
        """Determines the orientation plane of the NIfTI image along the x, y, or z-axis.

        Returns:
            str: The orientation plane of the image, which can be one of the following:
                - 'ax': Axial plane (along the z-axis).
                - 'cor': Coronal plane (along the y-axis).
                - 'sag': Sagittal plane (along the x-axis).
                - 'iso': Isotropic plane (if the image has equal zoom values along all axes).
        Examples:
            >>> nii = NII(nib.load('my_image.nii.gz'))
            >>> nii.get_plane()
            'ax'
        """
        #plane_dict = {"S": "ax", "I": "ax", "L": "sag", "R": "sag", "A": "cor", "P": "cor"}
        img = to_nii(self)
        axc = np.array(nio.aff2axcodes(img.affine))
        zms = np.around(img.zoom, 1)
        ix_max = np.array(zms == np.amax(zms))
        num_max = np.count_nonzero(ix_max)
        if num_max == 2:
            plane = plane_dict[axc[~ix_max][0]]
        elif num_max == 1:
            plane = plane_dict[axc[ix_max][0]]
        else:
            plane = "iso"
        return plane

    def get_axis(self,direction:Directions = "S"):
        if direction not in self.orientation:
            direction = vc.same_direction[direction]
        return self.orientation.index(direction)


    def erode_msk(self, mm: int = 5, labels: Sequence[int] | None = None, connectivity: int = 3, inplace=False,verbose:vc.logging=True):
        """
        Erodes the binary segmentation mask by the specified number of voxels.

        Args:
            mm (int, optional): The number of voxels to erode the mask by. Defaults to 5.
            labels (list[int], optional): Labels that should be dilated. If None, will erode all labels (not including zero!)
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
        out = np_erode_msk(msk_i_data, label_ref=labels, mm=mm, connectivity=connectivity)
        msk_e = out.astype(np.uint16), self.affine, self.header
        log.print("Mask eroded by", mm, "voxels",verbose=verbose)
        if inplace:
            self.nii = msk_e
            return self
        return NII(msk_e,seg=True,c_val=0)

    def erode_msk_(self, mm:int = 5, labels: Sequence[int] | None = None, connectivity: int=3, verbose:logging=True):
        return self.erode_msk(mm=mm, labels=labels, connectivity=connectivity, inplace=True, verbose=verbose)

    def dilate_msk(self, mm: int = 5, labels: Sequence[int] | None = None, connectivity: int = 3, mask: Self | None = None, inplace=False, verbose:logging=True):
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
        if mask is not None:
            mask_ = mask.get_seg_array()
        else:
            mask_ = None
        out = np_dilate_msk(arr=msk_i_data, label_ref=labels, mm=mm, mask=mask_, connectivity=connectivity)
        msk_e = out.astype(np.uint16), self.affine,self.header
        log.print("Mask dilated by", mm, "voxels",verbose=verbose)
        if inplace:
            self.nii = msk_e
            return self
        return NII(msk_e,seg=True,c_val=0)

    def dilate_msk_(self, mm:int = 5, labels: list[int] | None = None, connectivity: int=3, mask: Self | None = None, verbose:logging=True):
        return self.dilate_msk(mm=mm, labels=labels, connectivity=connectivity, mask=mask, inplace=True, verbose=verbose)


    def fill_holes(self, labels: int | list[int] | None = None, verbose:logging=True, inplace=False):
        """Fills holes in segmentations

        Args:
            labels (int | list[int] | None, optional): Labels that the hole-filling should be applied to. If none, applies on all labels found in arr. Defaults to None.
            verbose: whether to print which labels have been filled
            inplace (bool): Whether to modify the current NIfTI image object in place or create a new object with the mapped labels.
                Default is False.

        Returns:
            NII: If inplace is True, returns the current NIfTI image object with filled holes. Otherwise, returns a new NIfTI image object with filled holes.
        """
        if labels is None:
            labels = list(self.unique())
        if isinstance(labels, int):
            labels = [labels]

        seg_arr = self.get_seg_array()
        #volumes = np_volume(seg_arr, label_ref=labels)
        filled = np_fill_holes(seg_arr, label_ref=labels)
        #volumes_filled = np_volume(filled, label_ref=labels)
        #changes_in_labels = [i for i in labels if i in volumes_filled and i in volumes and volumes_filled[i] != volumes[i]]
        #if len(changes_in_labels) > 0:
        #    log.print(f"Filled holes in {changes_in_labels}", verbose=verbose)
        #else:
        #    log.print("Fill holes: No holes have been filled", verbose=verbose)
        log.print("Fill holes called", verbose=verbose)
        return self.set_array(filled, inplace=inplace)

    def fill_holes_(self, labels: int | list[int] | None = None, verbose:logging=True):
        return self.fill_holes(labels, verbose, inplace=True)

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

    def get_segmentation_connected_components(self, labels: int |list[int], connectivity: int = 3, verbose: bool=False):
        """Calculates and returns the connected components of this segmentation NII

        Args:
            label (int): the label(s) of the connected components
            connectivity (int, optional): Connectivity for the connected components. Defaults to 3.

        Returns:
            cc: dict[label, cc_idx, arr], cc_stats: dict[label, key, values]
            keys:
                "voxel_counts","bounding_boxes","centroids","N"
        """
        arr = self.get_seg_array()
        return np_connected_components(arr, connectivity=connectivity, label_ref=labels, verbose=verbose)

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


    def get_largest_k_segmentation_connected_components(self, k: int, labels: int | list[int] | None = None, connectivity: int = 1, return_original_labels: bool = True):
        """Finds the largest k connected components in a given array (does NOT work with zero as label!)

        Args:
            arr (np.ndarray): input array
            k (int): finds the k-largest components
            labels (int | list[int] | None, optional): Labels that the algorithm should be applied to. If none, applies on all labels found in this NII. Defaults to None.
            return_original_labels (bool): If set to False, will label the components from 1 to k. Defaults to True
        """
        return self.set_array(np_get_largest_k_connected_components(self.get_seg_array(), k=k, label_ref=labels, connectivity=connectivity, return_original_labels=return_original_labels))


    def get_segmentation_difference_to(self, mask_gt: Self, ignore_background_TP: bool = False) -> Self:
        """Calculates an NII that represents the segmentation difference between self and given groundtruth mask

        Args:
            mask_groundtruth (Self): The ground truth mask. Must match in orientation, zoom, and shape

        Returns:
            NII: Difference NII (1: FN, 2: TP, 3: FP, 4: Wrong label)
        """
        if self.orientation != mask_gt.orientation:
            mask_gt = mask_gt.reorient_same_as(self)
        assert self.zoom == mask_gt.zoom, f"zoom mismatch, got {self.zoom} and gt zoom {mask_gt.zoom}"
        assert self.shape == mask_gt.shape, f"shape mismatch, got {self.shape} and gt shape {mask_gt.shape}"
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

        if ignore_background_TP:
            diff_arr[(gt == 0) & (arr == 0)] = 0

        return self.set_array(diff_arr)


    def map_labels(self, label_map:Label_Map , verbose:logging=True, inplace=False):
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
        labels_before = [v for v in np.unique(data_orig) if v > 0]
        # enforce keys to be str to support both str and int
        label_map_ = {
            (v_name2idx[k] if k in v_name2idx else int(k)): (
                v_name2idx[v] if v in v_name2idx else (0 if v is None else int(v))
            )
            for k, v in label_map.items()
        }
        log.print("label_map_ =", label_map_, verbose=verbose)
        data = np_map_labels(data_orig, label_map_)
        labels_after = [v for v in np.unique(data) if v > 0]
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

    def map_labels_(self, label_map: Label_Map, verbose:logging=True):
        return self.map_labels(label_map,verbose=verbose,inplace=True)
    def copy(self):
        return NII(Nifti1Image(self.get_array(), self.affine, self.header),seg=self.seg,c_val = self.c_val)
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
        log.print(f"Save {file} as {out.get_data_dtype()}",verbose=verbose,ltype=vc.log_file.Log_Type.SAVE)
        nib.save(out, file) #type: ignore
    def __str__(self) -> str:
        return f"shp={self.shape}; ori={self.orientation}, zoom={tuple(np.around(self.zoom, decimals=2))}, seg={self.seg}" # type: ignore
    def __repr__(self)-> str:
        return self.__str__()
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
        if min_v < x2[2] < max_v:
            return True
        return False

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

    def extract_label(self,label:int|vc.Location|list[int]|list[vc.Location], inplace=False):
        '''If this NII is a segmentation you can single out one label with [0,1].'''
        seg_arr = self.get_seg_array()

        if isinstance(label, list):
            label = [l.value if isinstance(l,Enum) else l for l in label]
            if 1 not in label:
                seg_arr[seg_arr == 1] = 0
            for l in label:
                seg_arr[seg_arr == l] = 1
            seg_arr[seg_arr != 1] = 0
        else:
            if isinstance(label,vc.Location):
                label = label.value
            assert label != 0, 'Zero label does not make sens. This is the background'
            seg_arr[seg_arr != label] = 0
            seg_arr[seg_arr == label] = 1
        return self.set_array(seg_arr,inplace=inplace)
    def remove_labels(self,*label:int | list[int], inplace=False, verbose:logging=True):
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
    def remove_labels_(self,*label:int, verbose:logging=True):
        return self.remove_labels(*label,inplace=True,verbose=verbose)
    def apply_mask(self,mask:Self, inplace=False):
        assert mask.shape == self.shape, f"[def apply_mask] Mask and Shape are not equal: \nMask - {mask},\nSelf - {self})"
        seg_arr = mask.get_seg_array()
        seg_arr[seg_arr != 0] = 1
        arr = self.get_array()
        return self.set_array(arr*seg_arr,inplace=inplace)

    def unique(self,verbose:logging=False):
        '''Returns all integer labels WITHOUT 0. Must be performed only on a segmentation nii'''
        out = list(np.unique(self.get_seg_array()))
        log.print(out,verbose=verbose)
        return tuple(int(o) for o in out if o != 0)

    def volumes(self, labels: vc.Label_Reference = None) -> dict[int, int]:
        '''Returns a dict stating how many pixels are present for each label (including zero!)'''
        return np_volume(self.get_seg_array(), label_ref=labels)


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
    if isinstance(img_bids, NII):
        return img_bids.copy()
    elif isinstance(img_bids, bids_files.BIDS_FILE):
        return img_bids.open_nii()
    elif isinstance(img_bids, Path):
        return NII(nib.load(str(img_bids)), seg) #type: ignore
    elif isinstance(img_bids, str):
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
