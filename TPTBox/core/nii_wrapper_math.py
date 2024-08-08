import operator
from math import ceil, floor
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from typing_extensions import Self

from .nii_poi_abstract import Has_Affine

# fmt: off
if TYPE_CHECKING:
    from nibabel.nifti1 import Nifti1Header

    from TPTBox import NII
    class NII_Proxy:
        seg = True
        c_val = 0
        def get_array(self) -> np.ndarray:
            ...
        def set_array(self,arr:np.ndarray,inplace=False,verbose=True)->Self:
            ...
        @property
        def shape(self) -> tuple[int, int, int]:
            ...
        @property
        def dtype(self)->type:
            ...
        @property
        def header(self) -> Nifti1Header:
            ...

        @property
        def affine(self) -> np.ndarray:
            ...
        def get_c_val(self)->int:
            ...
    C = NII|Number|np.ndarray
else:
    class NII_Proxy:
        pass
    C = Self|Number|np.ndarray
class NII_Math(NII_Proxy,Has_Affine):
    def _binary_opt(self, other:C, opt,inplace = False)-> Self:
        if isinstance(other,NII_Math):
            other = other.get_array()
        return self.set_array(opt(self.get_array(),other),inplace=inplace,verbose=False)
    def _uni_opt(self, opt,inplace = False)-> Self:
        return self.set_array(opt(self.get_array()),inplace=inplace,verbose=False)
    def __add__(self,p2):
        return self._binary_opt(p2,operator.add)
    def __sub__(self,p2):
        return self._binary_opt(p2,operator.sub)
    def __mul__(self,p2):
        return self._binary_opt(p2,operator.mul)
    def __pow__(self,p2):
        return self._binary_opt(p2,operator.pow)
    def __truediv__(self,p2):
        return self._binary_opt(p2,operator.truediv)
    def __floordiv__(self,p2):
        return self._binary_opt(p2,operator.floordiv)
    def __mod__(self,p2):
        return self._binary_opt(p2,operator.mod)
    def __lshift__(self,p2):
        return self._binary_opt(p2,operator.lshift)
    def __rshift__(self,p2):
        return self._binary_opt(p2,operator.rshift)
    def __and__(self,p2):
        return self._binary_opt(p2,operator.add)
    def __or__(self,p2):
        return self._binary_opt(p2,operator.or_)
    def __xor__(self,p2):
        return self._binary_opt(p2,operator.xor)
    def __invert__(self):
        return self._uni_opt(operator.invert)

    def __lt__(self,p2):
        return self._binary_opt(p2,operator.lt)
    def __le__(self,p2):
            return self._binary_opt(p2,operator.le)
    def __eq__(self,p2):
            return self._binary_opt(p2,operator.eq)
    def __ne__(self,p2):
            return self._binary_opt(p2,operator.ne)
    def __gt__(self,p2):
            return self._binary_opt(p2,operator.gt)
    def __ge__(self,p2):
            return self._binary_opt(p2,operator.ge)

    def __iadd__(self,p2):
        return self._binary_opt(p2,operator.add,inplace=True)
    def __isub__(self,p2:C):
        return self._binary_opt(p2,operator.sub,inplace=True)
    def __imul__(self,p2):
        return self._binary_opt(p2,operator.mul,inplace=True)
    def __ipow__(self,p2):
        return self._binary_opt(p2,operator.pow,inplace=True)
    def __itruediv__(self,p2):
        return self._binary_opt(p2,operator.truediv,inplace=True)
    def __ifloordiv__(self,p2):
        return self._binary_opt(p2,operator.floordiv,inplace=True)
    def __imod__(self,p2):
        return self._binary_opt(p2,operator.mod,inplace=True)

    def __neg__(self):
        return self._uni_opt(operator.neg)
    def __pos__(self):
        return self._uni_opt(operator.pos)
    def __abs__(self):
        return self._uni_opt(operator.abs)

    def __round__(self, decimals=0):
        return self._uni_opt(np.round)
    def round(self,decimals):
        return self.__round__(decimals=decimals)
    def __floor__(self):
        return self._uni_opt(np.floor)
    def __ceil__(self):
        return self._uni_opt(np.ceil)
    def max(self)->float:
        return self.get_array().max()
    def min(self)->float:
        return self.get_array().min()

    def clamp(self, min=None,max=None,inplace=False)->Self:  # noqa: A002
        arr = self.get_array()
        if min is not None:
            arr[arr<= min] = min
        if max is not None:
            arr[arr>= max] = max
        return self.set_array(arr,inplace=inplace,verbose=False)

    def clamp_(self, min=None,max=None):  # noqa: A002
        return self.clamp(min,max,inplace=True)

    def normalize(self,min_out = 0, max_out = 1, quantile = 1., clamp_lower:float|None=None,inplace=False):
        arr = self.get_array()
        max_v = np.quantile(arr[arr>0],q=quantile)
        arr = self.clamp(clamp_lower,max_v,inplace=inplace)
        arr -= arr.min() - min_out/max_out
        arr /= arr.max() *max_out
        assert arr.max() == max_out, f"{arr.max()} == {max_out}"
        assert arr.min() == min_out
        return self.set_array(arr.get_array(),inplace)
    def normalize_(self,min_out = 0, max_out = 1, quantile = 1., clamp_lower:float|None=None):
        return self.normalize(min_out = min_out, max_out = max_out, quantile = quantile, clamp_lower=clamp_lower,inplace=True)
    def normalize_mri(self,min_out = 0, max_out = 1, quantile = 0.99, inplace=False):
        a  = self.clamp(min=0,inplace=inplace)
        return a.normalize(min_out = min_out, max_out = max_out,quantile = quantile,clamp_lower=0,inplace=inplace)
    def normalize_ct(self,min_out = 0, max_out = 1,inplace=False):
        arr = self.clamp(min=-1024,max=1024,inplace=inplace)
        return arr.normalize(min_out = min_out, max_out = max_out, inplace=inplace)

    def sum(self,axis = None,keepdims=False,where = np._NoValue, **qargs)->float:  # type: ignore
        if hasattr(where,"get_array"):
            where=where.get_array().astype(bool)

        return np.sum(self.get_array(),axis=axis,keepdims=keepdims,where=where,**qargs)
    def threshold(self,threshold=0.5, inplace=False):
        arr = self.get_array()
        arr2 = arr.copy()
        arr[arr2>=threshold] = 1
        arr[arr2<=threshold] = 0
        nii = self.set_array(arr,inplace,verbose=False)
        nii.seg =True
        nii.c_val = 0
        return nii

    def nan_to_num(self, num=0,inplace=False):
        arr = self.get_array()
        return self.set_array(np.nan_to_num(arr,nan=num), inplace=inplace)
    def ssim(self, nii:NII_Proxy, min_v = 0):
        img_1 = nii.get_array() - min_v
        img_2 = self.get_array() - min_v
        img_1/= img_1.max()
        img_1[img_1<=0] = 0
        img_2= img_2/ img_2.max()
        img_2[img_2<=0] = 0
        ssim_value = ssim(img_1, img_2,data_range=img_1.max() - img_1.min())
        return ssim_value


    def psnr(self,nii: NII_Proxy,min_v=0):
        img_1 = nii.get_array() - min_v
        img_2 = self.get_array() - min_v
        img_1/= img_1.max()
        img_1[img_1<=0] = 0
        img_2= img_2/img_2.max()
        img_2[img_2<=0] = 0
        ssim_value = psnr(img_1, img_2,data_range=img_1.max() - img_1.min())
        return ssim_value


    def betti_numbers(self: "NII",verbose=False) -> dict[int, tuple[int, int, int]]: # type: ignore
        """
        Calculate Betti numbers for connected components in a 3D image.

        Parameters:
        - self (NII): An object representing a 3D image with labeled connected components.

        Returns:
        - dict: A dictionary containing Betti numbers for each unique connected component.

        The Betti numbers represent the topological features of the connected components in the input 3D image.
        The function iterates through each unique voxel label in the input image, extracts the connected component
        corresponding to that label, and calculates the Betti numbers (B0, B1, B2) using the np_betti_number function.

        Betti Numbers:
        - B0 (b0): Number of connected components.
        - B1 (b1): Number of holes.
        - B2 (b2): Number of fully engulfed empty spaces.

        The np_betti_number function uses Euler characteristic numbers to calculate these Betti numbers. The result is
        a dictionary where each voxel label is associated with its corresponding Betti numbers.

        Example Usage:
        ```python
        # Assuming 'nii' is a 3D image object
        betti_results = nii.betti_number()
        print(betti_results)
        ```

        Note: The input image 'nii' is expected to have connected components labeled, and the Betti numbers are calculated for each labeled region separately.
        """
        from TPTBox.core.np_utils import np_betti_numbers
        out = {}
        u = self.unique()
        if verbose:
            from tqdm import tqdm
            u = tqdm(u,total=len(u),desc="betti_numbers")
        for voxel_label in u:
            #print("\nvoxel_label = ", voxel_label)
            filtered_nii = self.extract_label(voxel_label)
            crop = filtered_nii.compute_crop()
            betti_numbers = np_betti_numbers(filtered_nii.apply_crop(crop).get_seg_array())
            out[voxel_label] = betti_numbers
        return out
