from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk  # noqa: N813

if TYPE_CHECKING:
    from nibabel import Nifti1Image

    from TPTBox import NII, POI


def __s(a):
    return "(" + str.join(",", [f"{i:5.2f}" for i in a]) + ")"


def __s2(a):
    return "(" + str.join(",", [f"{i:<3}" for i in a]) + ")"


def to_str_sitk(img: sitk.Image):
    return (
        f"Size={__s2(img.GetSize())}, Spacing={__s(img.GetSpacing())}, Origin={__s(img.GetOrigin())}, Direction={__s(img.GetDirection())}"
    )


def resample_mask(mask: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    return sitk.Resample(mask, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())


def resample_img(img: sitk.Image, ref_img: sitk.Image, verbose=True) -> sitk.Image:
    #  Resample(image1,                 transform, interpolator, defaultPixelValue, outputPixelType ,useNearestNeighborExtrapolator,)
    # x Resample(image1, referenceImage, transform, interpolator, defaultPixelValue, outputPixelType, useNearestNeighborExtrapolator)
    #  Resample(image1, size,           transform, interpolator, outputOrigin, outputSpacing, outputDirection, defaultPixelValue , outputPixelType, useNearestNeighborExtrapolator)
    if (
        img.GetSize() == ref_img.GetSize()
        and img.GetSpacing() == ref_img.GetSpacing()
        and img.GetOrigin() == ref_img.GetOrigin()
        and img.GetDirection() == ref_img.GetDirection()
    ):
        print("[*] Image needs no resampling") if verbose else None
        return img
    print(f"[*] Resample Image to {to_str_sitk(ref_img)}") if verbose else None
    return sitk.Resample(img, ref_img, sitk.Transform(), sitk.sitkLinear, 0)


def get_3D_corners(img: sitk.Image):
    shape = img.GetSize()
    out = []
    for x in [0, 1]:
        for y in [0, 1]:
            for z in [0, 1]:
                point = img.TransformIndexToPhysicalPoint((shape[0] * x, shape[1] * y, shape[2] * z))
                out.append(point)
    return out


# def resample_shared_space(img: sitk.Image, ref_img: sitk.Image, transform=sitk.Transform(), verbose=True):
#    #  Resample(image1,                 transform, interpolator, defaultPixelValue, outputPixelType ,useNearestNeighborExtrapolator,)
#    #  Resample(image1, referenceImage, transform, interpolator, defaultPixelValue, outputPixelType, useNearestNeighborExtrapolator)
#    # x Resample(image1, size,           transform, interpolator, outputOrigin, outputSpacing, outputDirection, defaultPixelValue , outputPixelType, useNearestNeighborExtrapolator)
#    extreme_points = get_3D_corners(img) + get_3D_corners(ref_img)
#    # print(extreme_points)
#    # Use the original spacing (arbitrary decision).
#    output_spacing = ref_img.GetSpacing()
#    # Identity cosine matrix (arbitrary decision).
#    output_direction = ref_img.GetDirection()
#    # Minimal x,y coordinates are the new origin.
#    output_origin = [min(extreme_points, key=lambda p: p[i])[i] for i in range(3)]
#    output_max = [max(extreme_points, key=lambda p: p[i])[i] for i in range(3)]
#    # Compute grid size based on the physical size and spacing.
#    output_size = [abs(int((output_max[i] - output_origin[i]) / output_spacing[i])) for i in range(3)]
#    print(extreme_points)
#    print(to_str_sitk(img), to_str_sitk(ref_img))
#    print(output_origin, output_size)
#    print(f"[*] Resample and Join Image") if verbose else None
#    a = sitk.Resample(img, output_size, transform, sitk.sitkLinear, output_origin, output_spacing, output_direction)
#    b = sitk.Resample(ref_img, output_size, transform, sitk.sitkLinear, output_origin, output_spacing, output_direction)
#    # sitk.Compose()
#    return sitk.Add(a, b)


def resize_image_itk(ori_img: sitk.Image, target_img: sitk.Image, resample_method=sitk.sitkLinear):
    """
    https://programmer.ink/think/resample-method-and-code-notes-of-python-simpleitk-library.html
    use itk Method to convert the original image resample To be consistent with the target image
    :param ori_img: Original alignment required itk image
    :param target_img: Target to align itk image
    :param resample_method: itk interpolation method : sitk.sitkLinear-linear  sitk.sitkNearestNeighbor-Nearest neighbor
    :return:img_res_itk: Resampling okay itk image
    """
    target_size = target_img.GetSize()  # Target image size [x,y,z]
    target_spacing = target_img.GetSpacing()  # Voxel block size of the target [x,y,z]
    target_origin = target_img.GetOrigin()  # Starting point of target [x,y,z]
    target_direction = target_img.GetDirection()  # Target direction [crown, sagittal, transverse] = [z,y,x]

    # The method of itk is resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # Target image to resample
    # Set the information of the target image
    resampler.SetSize(target_size)  # Target image size
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_spacing)
    # Set different type according to the need to resample the image
    if resample_method == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt16)  # Nearest neighbor interpolation is used for mask, and uint16 is saved
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # Linear interpolation is used for PET/CT/MRI and the like, and float32 is saved
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_method)
    itk_img_resampled: sitk.Image = resampler.Execute(ori_img)  # Get the resampled image
    return itk_img_resampled


def pad_same(image: sitk.Image, ref_img: sitk.Image, default_value=-1) -> sitk.Image:
    upper = []
    lower = []
    for index in range(3):
        image0_min_extent = image.GetOrigin()[index]
        image0_max_extent = image.GetOrigin()[index] + image.GetSize()[index] * image.GetSpacing()[index]
        min_extent = min(image0_min_extent, ref_img.GetOrigin()[index])
        max_extent = max(image0_max_extent, ref_img.GetOrigin()[index] + ref_img.GetSize()[index] * ref_img.GetSpacing()[index])
        lower.append(int((image0_min_extent - min_extent) / image.GetSpacing()[index] + 1))
        upper.append(int((max_extent - image0_max_extent) / image.GetSpacing()[index] + 1))

    sitk_filter = sitk.ConstantPadImageFilter()
    #  filter->SetInput(input);
    print(lower, upper)
    sitk_filter.SetPadLowerBound(lower)
    sitk_filter.SetPadUpperBound(upper)
    sitk_filter.SetConstant(default_value)
    return sitk_filter.Execute(image)


def padZ(image: sitk.Image, pad_min_z, pad_max_z, unique_value) -> sitk.Image:
    sitk_filter = sitk.ConstantPadImageFilter()
    #  filter->SetInput(input);
    sitk_filter.SetPadLowerBound([0, 0, pad_min_z])
    sitk_filter.SetPadUpperBound([0, 0, pad_max_z])
    sitk_filter.SetConstant(unique_value)
    return sitk_filter.Execute(image)


def cropZ(image: sitk.Image, pad_min_z, pad_max_z, verbose=True, z_index=2) -> sitk.Image:
    if verbose:
        print("[*] crop ", pad_min_z, abs(pad_max_z), "pixels")
    sitk_filter = sitk.CropImageFilter()
    sitk_filter.SetLowerBoundaryCropSize([abs(pad_min_z) if i == z_index else 0 for i in range(3)])
    sitk_filter.SetUpperBoundaryCropSize([abs(pad_max_z) if i == z_index else 0 for i in range(3)])
    return sitk_filter.Execute(image)


def divide_by_max(img: sitk.Image) -> sitk.Image:
    sitk_filter = sitk.MinimumMaximumImageFilter()
    sitk_filter.Execute(img)
    maximum = sitk_filter.GetMaximum()
    if maximum == 0:
        print("[!] Warning the max of this image is 0. It is probably empty ")
        return img
    return sitk.Divide(img, maximum)


def affine_registration_transform(moving_image, fixed_image: sitk.Image, transform=None, verbose=True) -> sitk.Transform:
    if isinstance(moving_image, np.ndarray):
        moving_image = sitk.GetImageFromArray(moving_image)
    if isinstance(fixed_image, np.ndarray):
        fixed_image = sitk.GetImageFromArray(fixed_image)

    if transform is None:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    else:
        initial_transform = sitk.Transform(transform)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.00001,
        numberOfIterations=1000,
        convergenceMinimumValue=1e-32,
        convergenceWindowSize=0,  # Min n steps
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    tran = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))
    # Always check the reason optimization terminated.
    if verbose:
        print(f"Final metric value: {registration_method.GetMetricValue()}")
        print(f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}")
    return tran


def affine_registration_exhaustive_transform(fixed_image: sitk.Image, moving_image, transform) -> sitk.Transform:
    # initial_transform: sitk.Euler3DTransform = sitk.CenteredTransformInitializer(
    #    fixed_image, moving_image, sitk.Euler3DTransform(center), sitk.CenteredTransformInitializerFilter.MOMENTS
    # )
    initial_transform = sitk.Transform(transform)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    n = 20
    registration_method.SetOptimizerAsExhaustive(numberOfSteps=[n, n, n, 0, 0, 0], stepLength=np.pi / n)
    registration_method.SetOptimizerScales([1, 1, 1, 1, 1, 1])

    # Perform the registration in-place so that the initial_transform is modified.
    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    return registration_method.Execute(fixed_image, moving_image)


def apply_transform(moving_image, fixed_image, transform, is_segmentation=False) -> sitk.Image:
    if isinstance(moving_image, np.ndarray):
        moving_image = sitk.GetImageFromArray(moving_image)
    if isinstance(fixed_image, np.ndarray):
        fixed_image = sitk.GetImageFromArray(fixed_image)
    return sitk.Resample(
        moving_image,
        fixed_image,
        transform,
        sitk.sitkNearestNeighbor if is_segmentation else sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )


def register_on_sub_image(fixed_sub_image, moving_sub_image, fixed_image, moving_image):
    transform = affine_registration_transform(moving_sub_image, fixed_sub_image)
    sitk.CompositeTransform()
    return apply_transform(moving_image, fixed_image, transform)


def nii_to_sitk(nii: "NII") -> sitk.Image:
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L289
    """Create a SimpleITK image from a Nifti."""
    return nib_to_sitk(nii.nii)


def nib_to_sitk(nii: "Nifti1Image") -> sitk.Image:
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L289
    """Create a SimpleITK image from a Nifti."""
    array = np.asarray(nii.dataobj)
    affine = np.asarray(nii.affine).astype(np.float64)
    assert array.ndim == 3
    array = array.transpose()
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(float)

    image = sitk.GetImageFromArray(array, isVector=False)  # isVector = True, First dimension is used in parallel
    origin, spacing, direction = get_sitk_metadata_from_ras_affine(affine)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    assert len(image.GetSize()) == 3
    return image


###########################################################################################
# Functions are simplified from https://github.com/fepegar/torchio
def sitk_to_nib(image: sitk.Image) -> "Nifti1Image":
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L332
    data = sitk.GetArrayFromImage(image).transpose()
    assert image.GetNumberOfComponentsPerPixel() == 1
    assert image.GetDimension() == 3
    affine = get_ras_affine_from_sitk(image)
    import nibabel

    return nibabel.Nifti1Image(data, affine)


def sitk_to_nii(image: sitk.Image, seg: bool) -> "NII":
    import TPTBox

    return TPTBox.NII(sitk_to_nib(image), seg)


def transform_centroid(ctd: "POI", transform: sitk.Transform, img_fixed: sitk.Image, img_moving: sitk.Image, reg_type):
    import TPTBox
    from TPTBox.core.poi import POI, POI_DICT, POI_Descriptor

    out = TPTBox.core.poi.POI_Descriptor()

    if reg_type == "deformable":
        for key, key2, (x, y, z) in ctd.items():
            ctr_b = transform.TransformPoint((x, y, z))
            out[key, key2] = ctr_b
            out[key, key2] = ctr_b
    else:
        for key, key2, (x, y, z) in ctd.items():
            ctr_b = img_moving.TransformContinuousIndexToPhysicalPoint((x, y, z))
            ctr_b = transform.GetInverse().TransformPoint(ctr_b)
            ctr_b = img_fixed.TransformPhysicalPointToContinuousIndex(ctr_b)
            out[key, key2] = ctr_b
    nii = sitk_to_nii(img_fixed, True)

    return POI(
        orientation=nii.orientation,
        centroids=out,
        origin=ctd.origin,
        rotation=ctd.rotation,
        # location = ctd.location,
        zoom=nii.zoom,
        shape=nii.shape,
        **nii._extract_affine(),
    )


def get_sitk_metadata_from_ras_affine(affine: np.ndarray):
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L385
    direction_ras, spacing_array = get_rotation_and_spacing_from_affine(affine)
    origin_ras = affine[:3, 3]
    origin_array = np.dot(np.diag([-1, -1, 1]), origin_ras)
    direction_array = np.dot(np.diag([-1, -1, 1]), direction_ras).flatten()
    direction = tuple(direction_array)
    origin = tuple(origin_array)
    spacing = tuple(spacing_array)
    return origin, spacing, direction


def get_rotation_and_spacing_from_affine(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # From https://github.com/nipy/nibabel/blob/master/nibabel/orientations.py
    rotation_zoom = affine[:3, :3]
    spacing = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
    rotation = rotation_zoom / spacing
    return rotation, spacing


def get_ras_affine_from_sitk(sitk_object: sitk.Image | sitk.ImageFileReader) -> np.ndarray:
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L357
    spacing = np.array(sitk_object.GetSpacing())
    direction_lps = np.array(sitk_object.GetDirection())
    origin_lps = np.array(sitk_object.GetOrigin())
    direction_length = len(direction_lps)
    if direction_length == 9:
        rotation_lps = direction_lps.reshape(3, 3)
    elif direction_length == 4:  # ignore last dimension if 2D (1, W, H, 1)
        rotation_lps_2d = direction_lps.reshape(2, 2)
        rotation_lps = np.eye(3)
        rotation_lps[:2, :2] = rotation_lps_2d
        spacing = np.append(spacing, 1)
        origin_lps = np.append(origin_lps, 0)
    elif direction_length == 16:  # probably a bad NIfTI. Let's try to fix it
        rotation_lps = direction_lps.reshape(4, 4)[:3, :3]
        spacing = spacing[:-1]
        origin_lps = origin_lps[:-1]
    else:
        raise NotImplementedError()
    rotation_ras = np.dot(np.diag([-1, -1, 1]), rotation_lps)
    rotation_ras_zoom = rotation_ras * spacing
    translation_ras = np.dot(np.diag([-1, -1, 1]), origin_lps)
    affine = np.eye(4)
    affine[:3, :3] = rotation_ras_zoom
    affine[:3, 3] = translation_ras
    return affine


def get_ras_affine_from_sitk_meta(
    spacing: np.ndarray | tuple,
    direction_lps: np.ndarray | tuple,
    origin_lps: np.ndarray | tuple,
) -> np.ndarray:
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L357
    spacing = np.array(spacing)
    direction_lps = np.array(direction_lps)
    origin_lps = np.array(origin_lps)
    direction_length = len(direction_lps)
    if direction_length == 9:
        rotation_lps = direction_lps.reshape(3, 3)
    elif direction_length == 4:  # ignore last dimension if 2D (1, W, H, 1)
        rotation_lps_2d = direction_lps.reshape(2, 2)
        rotation_lps = np.eye(3)
        rotation_lps[:2, :2] = rotation_lps_2d
        spacing = np.append(spacing, 1)
        origin_lps = np.append(origin_lps, 0)
    elif direction_length == 16:  # probably a bad NIfTI. Let's try to fix it
        rotation_lps = direction_lps.reshape(4, 4)[:3, :3]
        spacing = spacing[:-1]
        origin_lps = origin_lps[:-1]
    else:
        raise NotImplementedError()
    rotation_ras = np.dot(np.diag([-1, -1, 1]), rotation_lps)
    rotation_ras_zoom = rotation_ras * spacing
    translation_ras = np.dot(np.diag([-1, -1, 1]), origin_lps)
    affine = np.eye(4)
    affine[:3, :3] = rotation_ras_zoom
    affine[:3, 3] = translation_ras
    return affine
