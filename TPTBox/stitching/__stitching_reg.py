# stitching - command line tool for whole-body image stitching.#
# Copyright 2016 Ben Glocker <b.glocker@imperial.ac.uk> and Robert Graf (Python translation)#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at#
#    h  ttp://www.apache.org/licenses/LICENSE-2.0#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import SimpleITK as sitk
import numpy as np
from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from sitk_utils import (
    padZ,
    resample_img,
    to_str_sitk,
    divide_by_max,
    cropZ,
    register_on_sub_image,
    affine_registration_transform,
    apply_transform,
)
import time

z_index = 2


def pad_to_space(image, min_extent, max_extent, default_value=0):
    image0_min_extent = image.GetOrigin()[z_index]
    image0_max_extent = image.GetOrigin()[z_index] + image.GetSize()[z_index] * image.GetSpacing()[z_index]

    pad_min_z = int((image0_min_extent - min_extent) / image.GetSpacing()[z_index] + 1)
    pad_max_z = int((max_extent - image0_max_extent) / image.GetSpacing()[z_index] + 1)
    return padZ(image, pad_min_z, pad_max_z, default_value)


def min_max_Z(img: sitk.Image) -> tuple[float, float]:
    min1 = img.GetOrigin()[z_index]
    max1 = img.GetOrigin()[z_index] + img.GetSize()[z_index] * img.GetSpacing()[z_index]
    return min1, max1


def compute_overlap(image1, image2, verbose=False):
    if image1 == None:
        return None
    min1, max1 = min_max_Z(image1)
    min2, max2 = min_max_Z(image2)
    # Case1 No Intersection
    if min1 > max2 and max1 > max2:
        print("[*] No Intersection; ", min1, max1, "<", max2, "|", min2) if verbose else None
        return None
    if min1 < min2 and max1 < min2:
        print("[*] No Intersection; ", min1, max1, ">", min2, "|", max2) if verbose else None
        return None
    # Case2 One image is inside of the other
    if min2 > min1 and max2 < max1:
        print("[*] Is 2 is contained in 1; ", min1, max1, "in", min2, max2) if verbose else None
        return None
    if min2 < min1 and max2 > max1:
        print("[*] Is 2 is contained in 1; ", min1, max1, "in", min2, max2) if verbose else None
        return None
    a = sorted([min1, min2, max1, max2])
    print("[*] Intersection found; ", a[1:3], "in", a[0], a[-1]) if verbose else None
    return a[1:3], [a[0], a[-1]]


def getOriginZ(img: sitk.Image) -> float:
    return img.GetOrigin()[z_index]


def main(
    files: list[str],
    filename_out: str = "",
    normalize=True,
    margin: int = 0,
    average_overlap: bool = False,
    mean_shift=True,
    reverse=True,
    verbose=True,
):

    stime = time.time()
    if len(files) > 1:
        print("stitching image...")
        images: list[sitk.Image] = []
        min_extent = 10e10
        max_extent = -10e10
        print("[*] Load images") if verbose else None

        for f in files:
            img1 = sitk.Cast(sitk.ReadImage(f), sitk.sitkFloat32)
            if margin != 0:
                image: sitk.Image = img1[:, :, margin:-margin]
            else:
                image: sitk.Image = img1
            if normalize:
                image = divide_by_max(image)

            images.append(image)
            min1, max1 = min_max_Z(image)
            # print(min1, max1)
            min_extent = min(min_extent, min1)
            max_extent = max(max_extent, max1)
        print("[*] Sort images from top to bottom") if verbose else None
        images = sorted(images, key=getOriginZ, reverse=reverse)
        print([getOriginZ(i) for i in images])
        print(f"[*] Final image range {min_extent} and {max_extent}") if verbose else None

        # generate stitched volume and fill in first image
        output_skit = pad_to_space(images[0], min_extent, max_extent)
        itk_composite = sitk.CompositeTransform(3)
        # Affine transform to current image. (Going from top to bottom, see sorted)
        for img1, img2 in zip(images[:-1], images[1:]):
            out = compute_overlap(img1, img2, verbose=False)
            if out is not None:
                print(f"[*] Intersection between {out}") if verbose else None

                # get Intersecting volumes
                z1 = round((out[0][1] - out[0][0]) / img1.GetSpacing()[z_index])
                z2 = round((out[0][1] - out[0][0]) / img2.GetSpacing()[z_index])
                # Crop to intersection
                # inter1 = cropZ(output_skit, pad_min_z + img1.GetSize()[-1] - z1, pad_max_z)
                out = img2
                if reverse:
                    inter1 = cropZ(img1, 0, img1.GetSize()[z_index] - z1, z_index=z_index)
                    inter2 = cropZ(img2, img2.GetSize()[z_index] - z2, 0, z_index=z_index)
                else:
                    inter1 = cropZ(img1, img1.GetSize()[z_index] - z1, 0, z_index=z_index)
                    inter2 = cropZ(img2, 0, img2.GetSize()[z_index] - z2, z_index=z_index)

                # register images
                cur_transform = affine_registration_transform(inter2, inter1)
                itk_composite.AddTransform(cur_transform)
                # out = pad_to_space(img2, min_extent, max_extent)
                out = apply_transform(out, output_skit, itk_composite)
                # Compute where the images overlap
                mask_old = sitk.Equal(output_skit, 0)
                mask_new = sitk.Equal(out, 0)
                mask_overlap = sitk.Cast(sitk.Or(mask_new, mask_old), sitk.sitkFloat32)
                if mean_shift:
                    # compute mean shift
                    mean1 = np.mean(sitk.GetArrayFromImage(inter1))
                    mean2 = np.mean(sitk.GetArrayFromImage(inter2))
                    out = sitk.Multiply(out, float(mean1 / mean2))
                    # out = sitk.Add(out, mask_overlap)
                    print("[*] mean shift from", mean2, "to", mean1) if verbose else None

                if average_overlap:
                    weighting = sitk.Divide(sitk.Add(mask_overlap, 1), 2)
                    output_skit = sitk.Add(out, output_skit)
                    output_skit = sitk.Multiply(output_skit, weighting)
                else:
                    mask_old = sitk.Cast(sitk.Equal(output_skit, 0), sitk.sitkFloat32)
                    out = sitk.Multiply(out, mask_old)
                    output_skit = sitk.Add(out, output_skit)
            else:
                print(f"[*] No Intersection, just adding the image without registration") if verbose else None
                img2 = resample_img(pad_to_space(img2, min_extent, max_extent), output_skit, verbose)
                output_skit = sitk.Add(output_skit, img2)

        output_arr = sitk.GetArrayFromImage(output_skit)

        off_z_max = -1
        off_z_min = 0

        while output_arr[:, :, off_z_min].sum() == 0:
            off_z_min += 1
        while output_arr[:, :, off_z_max].sum() == 0:
            off_z_max -= 1

        output_skit: sitk.Image = output_skit[:, :, off_z_min:off_z_max]  #
        output_skit = cropZ(output_skit, off_z_min, -off_z_max + 1, verbose=verbose)
        if verbose:
            print(to_str_sitk(output_skit))
            print("[#] Write Image ", filename_out)
        sitk.WriteImage(output_skit, filename_out)
        print("done. took ", time.time() - stime)


if __name__ == "__main__":
    folder = "3D_GRE_TRA_F"
    folder = "3D_GRE_TRA_W"
    folder = "3D_GRE_TRA_in"

    patient = ("100000_30", "20161014")
    patient = ("100023_30", "20161118")

    # main(
    #    [
    #        f"/media/data/NAKO/MRT/{folder}/{patient[0]}/ses-{patient[1]}/sub-{patient[0]}_ses-{patient[1]}_sequ-4_mr.nii.gz",
    #        f"/media/data/NAKO/MRT/{folder}/{patient[0]}/ses-{patient[1]}/sub-{patient[0]}_ses-{patient[1]}_sequ-3_mr.nii.gz",
    #        f"/media/data/NAKO/MRT/{folder}/{patient[0]}/ses-{patient[1]}/sub-{patient[0]}_ses-{patient[1]}_sequ-2_mr.nii.gz",
    #        f"/media/data/NAKO/MRT/{folder}/{patient[0]}/ses-{patient[1]}/sub-{patient[0]}_ses-{patient[1]}_sequ-1_mr.nii.gz",
    #    ],
    #    "test.nii.gz",
    #    normalize=True,
    #    margin=0,
    #    average_overlap=False,
    #    mean_shift=False,
    #    reverse=False,
    #    verbose=True,
    # )
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-i", "--images", nargs="+", default=[], help="filenames of images")
    parser.add_argument("-o", "--output", type=str, default="out.nii.gz", help="filename of output image")
    parser.add_argument("-m", "--margin", type=int, default=0, help="image margin that is ignored when stitching")
    parser.add_argument(
        "-a",
        "--averaging",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="enable averaging in overlap areas",
    )
    parser.add_argument("-v", "--verbose", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "-r", "--reverse", default=False, action=argparse.BooleanOptionalAction, help="flip registration direction"
    )
    parser.add_argument(
        "-me",
        "--mean_shift",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="shift means around to match them",
    )
    parser.add_argument("-n", "--normalize", default=False, action=argparse.BooleanOptionalAction, help="map to [0,1]")

    args = parser.parse_args()
    if args.verbose:
        print(args)
    main(
        args.images,
        args.output,
        normalize=args.normalize,
        margin=args.margin,
        average_overlap=args.averaging,
        mean_shift=args.mean,
        reverse=args.reverse,
        verbose=args.verbose,
    )

    # main(
    #    [
    #        f"/media/data/NAKO/MRT/{folder}/{patient[0]}/ses-{patient[1]}/sub-{patient[0]}_ses-{patient[1]}_sequ-4_mr.nii.gz",
    #        f"/media/data/NAKO/MRT/{folder}/{patient[0]}/ses-{patient[1]}/sub-{patient[0]}_ses-{patient[1]}_sequ-3_mr.nii.gz",
    #        f"/media/data/NAKO/MRT/{folder}/{patient[0]}/ses-{patient[1]}/sub-{patient[0]}_ses-{patient[1]}_sequ-2_mr.nii.gz",
    #        f"/media/data/NAKO/MRT/{folder}/{patient[0]}/ses-{patient[1]}/sub-{patient[0]}_ses-{patient[1]}_sequ-1_mr.nii.gz",
    #    ],
    #    "test.nii.gz",
    #    normalize=True,
    #    margin=0,
    #    average_overlap=False,
    #    mean_shift=False,
    #    reverse=False,
    #    verbose=True,
    # )
    # python3 stitching.py -i /media/data/NAKO/MRT/3D_GRE_TRA_in/100023_30/ses-20161118/sub-100023_30_ses-20161118_sequ-4_mr.nii.gz /media/data/NAKO/MRT/3D_GRE_TRA_in/100023_30/ses-20161118/sub-100023_30_ses-20161118_sequ-3_mr.nii.gz /media/data/NAKO/MRT/3D_GRE_TRA_in/100023_30/ses-20161118/sub-100023_30_ses-20161118_sequ-2_mr.nii.gz /media/data/NAKO/MRT/3D_GRE_TRA_in/100023_30/ses-20161118/sub-100023_30_ses-20161118_sequ-1_mr.nii.gz -v -me -o test.nii.gz
