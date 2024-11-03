import nibabel as nib
import numpy as np
import pytest

from TPTBox.stitching.n4_bias_field_correction import n4_bias_field_correction_nib, n4_bias_field_correction_nib2


# @pytest.fixture
def test_data():
    # Generate or load a small test image and mask for testing
    shape = (100, 100, 100)
    data = np.random.rand(*shape).astype(np.float32)
    mask_data = np.random.randint(0, 2, size=shape).astype(np.uint8)

    image = nib.Nifti1Image(data, affine=np.eye(4))
    mask = nib.Nifti1Image(mask_data, affine=np.eye(4))
    weight_mask = nib.Nifti1Image(mask_data, affine=np.eye(4))  # Example weight mask

    return image, mask, weight_mask


def test_n4_bias_field_correction_equivalence(test_data):
    image, mask, weight_mask = test_data

    # Define common parameters
    rescale_intensities = True
    shrink_factor = 4
    convergence = {"iters": [50, 50, 30, 20], "tol": 1e-7}
    return_bias_field = False
    verbose = False
    print("secont")
    corrected_image2 = n4_bias_field_correction_nib(
        image,
        mask=mask,
        rescale_intensities=rescale_intensities,
        shrink_factor=shrink_factor,
        convergence=convergence,
        return_bias_field=return_bias_field,
        verbose=verbose,
        weight_mask=weight_mask,
    )

    print("first")
    # Run both functions
    corrected_image1 = n4_bias_field_correction_nib2(
        image,
        mask=mask,
        rescale_intensities=rescale_intensities,
        shrink_factor=shrink_factor,
        convergence=convergence,
        return_bias_field=return_bias_field,
        verbose=verbose,
        weight_mask=weight_mask,
    )

    # Compare the corrected images and bias fields
    np.testing.assert_allclose(corrected_image1.get_fdata(), corrected_image2.get_fdata(), rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    test_n4_bias_field_correction_equivalence(test_data())
