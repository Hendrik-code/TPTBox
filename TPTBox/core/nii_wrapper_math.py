from __future__ import annotations

import operator
from numbers import Number
from typing import TYPE_CHECKING, Union

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from typing_extensions import Self

from TPTBox.core.np_utils import np_dice

from .nii_poi_abstract import Has_Grid

# fmt: off
if TYPE_CHECKING:
    from nibabel.nifti1 import Nifti1Header

    from TPTBox import NII
    class NII_Proxy:
        """Minimal NII interface stub for TYPE_CHECKING — never instantiated at runtime."""

        seg = True
        c_val = 0

        def get_array(self) -> np.ndarray:
            """Return a copy of the underlying array."""
            ...

        def set_array(self, arr: np.ndarray, inplace=False, verbose=True) -> Self:
            """Replace the underlying array."""
            ...

        def get_seg_array(self) -> np.ndarray:
            """Return a copy of the underlying segmentation array."""
            ...

        @property
        def shape(self) -> tuple[int, int, int]:
            """Spatial shape of the image."""
            ...

        @property
        def dtype(self) -> type:
            """Data type of the array."""
            ...

        @property
        def header(self) -> Nifti1Header:
            """NIfTI header."""
            ...

        @property
        def affine(self) -> np.ndarray:
            """4×4 affine matrix."""
            ...

        def get_c_val(self) -> int:
            """Return the fill/background value."""
            ...

        def unique(self) -> list[int]:
            """Return sorted list of unique label values."""
            ...
    C = Union[NII, Number, np.ndarray]
else:
    class NII_Proxy:
        """Runtime no-op placeholder for the TYPE_CHECKING NII_Proxy stub."""

    C = Union[Self, Number, np.ndarray]


class NII_Math(NII_Proxy, Has_Grid):
    """Mixin that adds arithmetic, comparison, and statistical operations to ``NII``.

    All operators return a new ``NII`` by default (out-of-place).  In-place variants
    (ending in ``_``) modify ``self`` directly.
    """

    __hash__ = None  # type: ignore # explicitly mark as unhashable

    def _binary_opt(self, other: C, opt, inplace=False) -> Self:
        """Apply a binary operator element-wise between this array and ``other``."""
        if isinstance(other,NII_Math):
            other = other.get_array()
        return self.set_array(opt(self.get_array(),other),inplace=inplace,verbose=False)
    def _uni_opt(self, opt,inplace = False,**args)-> Self:
        """Apply a unary operator element-wise to this array."""
        return self.set_array(opt(self.get_array(),**args),inplace=inplace,verbose=False)
    def __add__(self,p2):
        """Element-wise addition."""
        return self._binary_opt(p2,operator.add)
    def __radd__(self,p2):
        """Right-hand element-wise addition."""
        return self._binary_opt(p2,operator.add)
    def __sub__(self,p2):
        """Element-wise subtraction."""
        return self._binary_opt(p2,operator.sub)
    def __rsub__(self,p2)->Self:
        """Right-hand element-wise subtraction."""
        return (-self)._binary_opt(p2,operator.add)
    def __mul__(self,p2):
        """Element-wise multiplication."""
        return self._binary_opt(p2,operator.mul)
    def __pow__(self,p2):
        """Element-wise exponentiation."""
        return self._binary_opt(p2,operator.pow)
    def __truediv__(self,p2):
        """Element-wise true division."""
        return self._binary_opt(p2,operator.truediv)
    def __floordiv__(self,p2):
        """Element-wise floor division."""
        return self._binary_opt(p2,operator.floordiv)
    def __mod__(self,p2):
        """Element-wise modulo."""
        return self._binary_opt(p2,operator.mod)
    def __lshift__(self,p2):
        """Element-wise left bit-shift."""
        return self._binary_opt(p2,operator.lshift)
    def __rshift__(self,p2):
        """Element-wise right bit-shift."""
        return self._binary_opt(p2,operator.rshift)
    def __and__(self,p2):
        """Element-wise bitwise AND (integer arrays only).

        Raises:
            TypeError: If the underlying array is not an integer dtype.
        """
        if not np.issubdtype(self.get_array().dtype, np.integer):
            raise TypeError("Bitwise operations require integer arrays")
        return self._binary_opt(p2,operator.and_)
    def __or__(self,p2):
        """Element-wise bitwise OR (integer arrays only).

        Raises:
            TypeError: If the underlying array is not an integer dtype.
        """
        if not np.issubdtype(self.get_array().dtype, np.integer):
            raise TypeError("Bitwise operations require integer arrays")
        return self._binary_opt(p2,operator.or_)
    def __xor__(self,p2):
        """Element-wise bitwise XOR (integer arrays only).

        Raises:
            TypeError: If the underlying array is not an integer dtype.
        """
        if not np.issubdtype(self.get_array().dtype, np.integer):
            raise TypeError("Bitwise operations require integer arrays")
        return self._binary_opt(p2,operator.xor)
    def __invert__(self):
        """Element-wise bitwise inversion (integer arrays only).

        Raises:
            TypeError: If the underlying array is not an integer dtype.
        """
        if not np.issubdtype(self.get_array().dtype, np.integer):
            raise TypeError("Bitwise operations require integer arrays")
        return self._uni_opt(operator.invert)

    def __lt__(self,p2):
        """Element-wise less-than comparison."""
        return self._binary_opt(p2,operator.lt)
    def __le__(self,p2):
            """Element-wise less-than-or-equal comparison."""
            return self._binary_opt(p2,operator.le)
    def __eq__(self,p2):
            return self._binary_opt(p2,operator.eq)
    def __ne__(self,p2):
            return self._binary_opt(p2,operator.ne)
    def __gt__(self,p2):
            """Element-wise greater-than comparison."""
            return self._binary_opt(p2,operator.gt)
    def __ge__(self,p2):
            """Element-wise greater-than-or-equal comparison."""
            return self._binary_opt(p2,operator.ge)

    def __iadd__(self,p2):
        """In-place element-wise addition."""
        return self._binary_opt(p2,operator.add,inplace=True)
    def __isub__(self,p2:C):
        """In-place element-wise subtraction."""
        return self._binary_opt(p2,operator.sub,inplace=True)
    def __imul__(self,p2):
        """In-place element-wise multiplication."""
        return self._binary_opt(p2,operator.mul,inplace=True)
    def __ipow__(self,p2):
        """In-place element-wise exponentiation."""
        return self._binary_opt(p2,operator.pow,inplace=True)
    def __itruediv__(self,p2):
        """In-place element-wise true division."""
        return self._binary_opt(p2,operator.truediv,inplace=True)
    def __ifloordiv__(self,p2):
        """In-place element-wise floor division."""
        return self._binary_opt(p2,operator.floordiv,inplace=True)
    def __imod__(self,p2):
        """In-place element-wise modulo."""
        return self._binary_opt(p2,operator.mod,inplace=True)

    def __neg__(self):
        """Element-wise arithmetic negation."""
        return self._uni_opt(operator.neg)
    def __pos__(self):
        """Element-wise unary plus (no-op for most dtypes)."""
        return self._uni_opt(operator.pos)
    def __abs__(self):
        """Element-wise absolute value."""
        return self._uni_opt(operator.abs)

    def __round__(self, decimals=0):
        """Round array values to the given number of decimal places."""
        return self._uni_opt(np.round,decimals=decimals)
    def round(self, decimals) -> Self:
        """Round array values to ``decimals`` decimal places.

        Args:
            decimals (int): Number of decimal places to round to.

        Returns:
            Self: New instance with rounded values.
        """
        return self.__round__(decimals=decimals)
    def __floor__(self):
        """Element-wise floor (round down to nearest integer)."""
        return self._uni_opt(np.floor)
    def __ceil__(self):
        """Element-wise ceiling (round up to nearest integer)."""
        return self._uni_opt(np.ceil)
    def max(self)->float:
        """Return the maximum value in the array.

        Returns:
            float: Maximum voxel value.
        """
        return self.get_array().max()
    def min(self)->float:
        """Return the minimum value in the array.

        Returns:
            float: Minimum voxel value.
        """
        return self.get_array().min()

    def clamp(self, min=None,max=None,inplace=False)->Self:  # noqa: A002
        """Clamp array values to the interval [min, max].

        Values at or below ``min`` are set to ``min``; values at or above
        ``max`` are set to ``max``.  Either bound may be omitted.

        Args:
            min (float | None, optional): Lower bound. Values <= min are clamped. Defaults to None.
            max (float | None, optional): Upper bound. Values >= max are clamped. Defaults to None.
            inplace (bool, optional): If True, modify the array in place. Defaults to False.

        Returns:
            Self: Instance with clamped values.
        """
        arr = self.get_array()
        if min is not None:
            arr[arr<= min] = min
        if max is not None:
            arr[arr>= max] = max
        return self.set_array(arr,inplace=inplace,verbose=False)

    def clamp_(self, min=None, max=None) -> Self:  # noqa: A002
        """In-place variant of `clamp`."""
        return self.clamp(min,max,inplace=True)

    def normalize(self,min_out = 0, max_out = 1, quantile = 1., clamp_lower:float|None=None,inplace=False)->Self:
        """Normalize array values to the range [min_out, max_out].

        Non-zero voxels are used to compute the upper percentile. Values are
        optionally clamped from below before scaling.

        Args:
            min_out (float, optional): Minimum value of the output range. Defaults to 0.
            max_out (float, optional): Maximum value of the output range. Defaults to 1.
            quantile (float, optional): Upper quantile (over non-zero voxels) used as the
                pre-clamp ceiling. Defaults to 1.0 (full maximum).
            clamp_lower (float | None, optional): If set, clamp values below this threshold
                before normalizing. Defaults to None.
            inplace (bool, optional): If True, modify the array in place. Defaults to False.

        Returns:
            Self: Instance with normalized values in [min_out, max_out].
        """
        arr = self.get_array()
        max_v = np.quantile(arr[arr>0],q=quantile)
        arr = self.clamp(clamp_lower,max_v,inplace=inplace)
        arr -= arr.min() - min_out/max_out
        arr /= arr.max() *max_out
        assert arr.max() == max_out, f"{arr.max()} == {max_out}"
        assert arr.min() == min_out
        return self.set_array(arr.get_array(),inplace)
    def normalize_(self,min_out = 0, max_out = 1, quantile = 1., clamp_lower:float|None=None)->Self:
        """In-place variant of `normalize`."""
        return self.normalize(min_out = min_out, max_out = max_out, quantile = quantile, clamp_lower=clamp_lower,inplace=True)
    def normalize_mri(self,min_out = 0, max_out = 1, quantile = 0.99, inplace=False)->Self:
        """Normalize an MRI volume by clamping negatives and applying quantile-based scaling.

        Negative voxels are clamped to zero before normalization.  The upper
        bound is set to the 99th percentile (by default) of positive voxels.

        Args:
            min_out (float, optional): Minimum value of the output range. Defaults to 0.
            max_out (float, optional): Maximum value of the output range. Defaults to 1.
            quantile (float, optional): Upper quantile used for clamping. Defaults to 0.99.
            inplace (bool, optional): If True, modify the array in place. Defaults to False.

        Returns:
            Self: Instance with normalized values.
        """
        a  = self.clamp(min=0,inplace=inplace)
        return a.normalize(min_out = min_out, max_out = max_out,quantile = quantile,clamp_lower=0,inplace=inplace)
    def normalize_ct(self,min_out = 0, max_out = 1,inplace=False)->Self:
        """Normalize a CT volume by clamping to [-1024, 1024] HU before scaling.

        Args:
            min_out (float, optional): Minimum value of the output range. Defaults to 0.
            max_out (float, optional): Maximum value of the output range. Defaults to 1.
            inplace (bool, optional): If True, modify the array in place. Defaults to False.

        Returns:
            Self: Instance with normalized values.
        """
        arr = self.clamp(min=-1024,max=1024,inplace=inplace)
        return arr.normalize(min_out = min_out, max_out = max_out, inplace=inplace)

    def sum(self,axis = None,keepdims=False,where = np._NoValue, **qargs)->float:  # type: ignore
        """Compute the sum of array values, optionally restricted to a mask.

        Args:
            axis (int | tuple[int, ...] | None, optional): Axis or axes along which to sum.
                If None, sum over all elements. Defaults to None.
            keepdims (bool, optional): If True, retain reduced axes as size-one dimensions.
                Defaults to False.
            where (NII | np.ndarray | np._NoValue, optional): Boolean mask selecting elements
                to include. Accepts an NII-like object or a NumPy array. Defaults to np._NoValue.
            **qargs: Additional keyword arguments forwarded to ``numpy.sum``.

        Returns:
            float: Sum of the selected elements.
        """
        if hasattr(where,"get_array"):
            where=where.get_array().astype(bool)

        return np.sum(self.get_array(),axis=axis,keepdims=keepdims,where=where,**qargs)
    def mean(self,axis = None,keepdims=False,where = np._NoValue, **qargs)->float:  # type: ignore
        """Compute the arithmetic mean of array values, optionally restricted to a mask.

        Args:
            axis (int | tuple[int, ...] | None, optional): Axis or axes along which to average.
                If None, average over all elements. Defaults to None.
            keepdims (bool, optional): If True, retain reduced axes as size-one dimensions.
                Defaults to False.
            where (NII | np.ndarray | np._NoValue, optional): Boolean mask selecting elements
                to include. Accepts an NII-like object or a NumPy array. Defaults to np._NoValue.
            **qargs: Additional keyword arguments forwarded to ``numpy.mean``.

        Returns:
            float: Mean of the selected elements.
        """
        if hasattr(where,"get_array"):
            where=where.get_array().astype(bool)

        return np.mean(self.get_array(),axis=axis,keepdims=keepdims,where=where,**qargs)
    def median(self, axis=None, keepdims=False,  **qargs)->float:  # type: ignore
        """Compute the median of array values.

        Args:
            axis (int | tuple[int, ...] | None, optional): Axis or axes along which to compute
                the median. If None, compute over all elements. Defaults to None.
            keepdims (bool, optional): If True, retain reduced axes as size-one dimensions.
                Defaults to False.
            **qargs: Additional keyword arguments forwarded to ``numpy.median``.

        Returns:
            float: Median of the array.
        """
        arr = self.get_array()
        return np.median(arr, axis=axis, keepdims=keepdims, **qargs)

    def std(self,axis = None,keepdims=False,where = np._NoValue, **qargs)->float:  # type: ignore
        """Compute the standard deviation of array values, optionally restricted to a mask.

        Args:
            axis (int | tuple[int, ...] | None, optional): Axis or axes along which to compute
                the standard deviation. If None, compute over all elements. Defaults to None.
            keepdims (bool, optional): If True, retain reduced axes as size-one dimensions.
                Defaults to False.
            where (NII | np.ndarray | np._NoValue, optional): Boolean mask selecting elements
                to include. Accepts an NII-like object or a NumPy array. Defaults to np._NoValue.
            **qargs: Additional keyword arguments forwarded to ``numpy.std``.

        Returns:
            float: Standard deviation of the selected elements.
        """
        if hasattr(where,"get_array"):
            where=where.get_array().astype(bool)

        return np.std(self.get_array(),axis=axis,keepdims=keepdims,where=where,**qargs)
    def threshold(self,threshold=0.5, inplace=False)->Self:
        """Binarise the array by applying a hard threshold.

        Voxels with values greater than or equal to ``threshold`` are set to 1;
        all others are set to 0.  The result is flagged as a segmentation
        (``seg=True``) and cast to the smallest integer dtype.

        Args:
            threshold (float, optional): Cut-off value. Defaults to 0.5.
            inplace (bool, optional): If True, modify the array in place. Defaults to False.

        Returns:
            Self: Binarised segmentation instance.
        """
        arr = self.get_array()
        arr2 = arr.copy()
        arr[arr2>=threshold] = 1
        arr[arr2<=threshold] = 0
        nii = self if inplace else self.copy()
        nii.seg = True
        nii:NII = nii.set_array(arr,inplace,verbose=False)
        nii.c_val = 0
        nii.set_dtype_('smallest_int')
        return nii

    def nan_to_num(self, num=0,inplace=False)->Self:
        """Replace NaN values in the array with a constant.

        Args:
            num (float, optional): Replacement value for NaN entries. Defaults to 0.
            inplace (bool, optional): If True, modify the array in place. Defaults to False.

        Returns:
            Self: Instance with NaN values replaced.
        """
        arr = self.get_array()
        return self.set_array(np.nan_to_num(arr,nan=num), inplace=inplace)
    def ssim(self, nii:NII_Proxy, min_v = 0)->float:
        """Compute the Structural Similarity Index (SSIM) between this image and another.

        Both images are shifted by ``min_v``, normalised to [0, 1] and
        negative values are zeroed before the SSIM is calculated.

        Args:
            nii (NII_Proxy): Reference image to compare against.
            min_v (float, optional): Value subtracted from both images before normalisation.
                Defaults to 0.

        Returns:
            float: SSIM score in the range [-1, 1] (1 = identical).
        """
        img_1 = nii.get_array() - min_v
        img_2 = self.get_array() - min_v
        img_1/= img_1.max()
        img_1[img_1<=0] = 0
        img_2= img_2/ img_2.max()
        img_2[img_2<=0] = 0
        ssim_value = ssim(img_1, img_2,data_range=img_1.max() - img_1.min())
        return ssim_value


    def psnr(self,nii: NII_Proxy,min_v=0)->float:
        """Compute the Peak Signal-to-Noise Ratio (PSNR) between this image and another.

        Both images are shifted by ``min_v``, normalised to [0, 1] and
        negative values are zeroed before the PSNR is calculated.

        Args:
            nii (NII_Proxy): Reference image to compare against.
            min_v (float, optional): Value subtracted from both images before normalisation.
                Defaults to 0.

        Returns:
            float: PSNR score in dB (higher is better; inf when images are identical).
        """
        img_1 = nii.get_array() - min_v
        img_2 = self.get_array() - min_v
        img_1/= img_1.max()
        img_1[img_1<=0] = 0
        img_2= img_2/img_2.max()
        img_2[img_2<=0] = 0
        ssim_value = psnr(img_1, img_2,data_range=img_1.max() - img_1.min())
        return ssim_value
    def dice(self,nii: NII_Proxy,bar=True)->dict[int,float]:
        """Compute per-label Dice similarity coefficients against another segmentation.

        Args:
            nii (NII_Proxy): Predicted segmentation to compare against (ground truth is
                ``self``).
            bar (bool, optional): If True, display a tqdm progress bar. Defaults to True.

        Returns:
            dict[int, float]: Mapping from integer label to Dice score in [0, 1].
        """
        out:dict[int,float] = {}
        gt = self.get_seg_array()
        pred = nii.get_seg_array()
        s = set(self.unique()+nii.unique())
        if bar:
            from tqdm import tqdm
            s = tqdm(s,desc="dice")
        for lbl in s:
            out[lbl] = np_dice(pred,gt,label=lbl)
            #print(out[lbl])
        return out

    def betti_numbers(self: NII,verbose=False) -> dict[int, tuple[int, int, int]]: # type: ignore
        """Calculate Betti numbers for connected components in a 3D image.

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
