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

import itertools
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage.morphology import distance_transform_edt

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

import time

from sitk_utils import padZ, resample_img, to_str_sitk


def main(files: list[str], filename_out: str = "", margin: int = 0, average_overlap: bool = False, verbose=True):
    stime = time.time()
    if len(files) > 1:
        print("stitching image...")
        unique_value: float = -1234567
        loaded = sitk.Cast(sitk.ReadImage(files[0]), sitk.sitkFloat32)  # copy

        if margin != 0:
            image0: sitk.Image = loaded[:, :, margin:-margin]
        else:
            image0: sitk.Image = loaded
        # determine physical extent of stitched volume
        image0_min_extent = image0.GetOrigin()[2]
        image0_max_extent = image0.GetOrigin()[2] + image0.GetSize()[2] * image0.GetSpacing()[2]
        min_extent = image0_min_extent
        max_extent = image0_max_extent
        images: list[sitk.Image] = []
        for i, f in enumerate(files):
            if i == 0:
                continue
            temp = sitk.Cast(sitk.ReadImage(f), sitk.sitkFloat32)
            if margin != 0:
                image: sitk.Image = temp[:, :, margin:-margin]
            else:
                image: sitk.Image = temp
            images.append(image)
            min_extent = min(min_extent, image.GetOrigin()[2])
            max_extent = max(max_extent, image.GetOrigin()[2] + image.GetSize()[2] * image.GetSpacing()[2])

        # this part match histogram for different chunks
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOn()
        image0 = matcher.Execute(image0, images[-1])
        for i, image in enumerate(images[:-1]):
            images[i] = matcher.Execute(image, images[-1])

        # generate stitched volume and fill in first image
        pad_min_z = int((image0_min_extent - min_extent) / image0.GetSpacing()[2] + 1)
        pad_max_z = int((max_extent - image0_max_extent) / image0.GetSpacing()[2] + 1)

        target = padZ(image0, pad_min_z, pad_max_z, unique_value)

        def get_threshold_as_np(input):
            # find valid image values
            arr = sitk.GetArrayFromImage(input)
            arr[arr < -100] = -128
            arr[arr > -100] = 1
            arr[arr < -100] = 0
            return arr

        def np_to_skit(arr, ref):
            sitk_img: sitk.Image = sitk.GetImageFromArray(arr)
            sitk_img.SetOrigin(ref.GetOrigin())
            sitk_img.SetSpacing(ref.GetSpacing())
            return sitk_img

        target_arr = sitk.GetArrayFromImage(target)
        counts_arr = get_threshold_as_np(target)
        target_arr *= counts_arr
        target_arr = target_arr / target_arr.max()

        # list for ramp stitching
        target_list = [target_arr]
        count_list = [counts_arr]

        # iterate over remaining images and add to stitched volume

        for cur_img in images:
            empty_arr = 1 - counts_arr

            cur_img_min_extent = cur_img.GetOrigin()[2]
            cur_img_max_extent = cur_img.GetOrigin()[2] + cur_img.GetSize()[2] * cur_img.GetSpacing()[2]
            pad_min_z = int((cur_img_min_extent - min_extent) / cur_img.GetSpacing()[2] + 1)
            pad_max_z = int((max_extent - cur_img_max_extent) / cur_img.GetSpacing()[2] + 1)
            cur_img = padZ(cur_img, pad_min_z, pad_max_z, unique_value)
            # Resample to same space
            cur_img = resample_img(cur_img, target, verbose)
            # print(min)
            cur_arr = sitk.GetArrayFromImage(cur_img).copy()

            binary_arr = get_threshold_as_np(cur_img)
            # take only value for empty voxels, otherwise average values in overlap areas
            if not average_overlap:
                binary_arr = empty_arr * binary_arr

            cur_arr = cur_arr * binary_arr
            cur_arr = cur_arr / cur_arr.max()
            target_list.append(cur_arr)
            target_arr = cur_arr + target_arr  #
            count_list.append(binary_arr)
            counts_arr = binary_arr + counts_arr

        # ramp stitching
        for item in list(itertools.combinations(range(len(target_list)), 2)):
            arr_1 = count_list[item[0]]
            arr_2 = count_list[item[1]]
            overlap = (arr_1 * arr_2) > 0.0
            if overlap.sum() > 0:
                arr_1_ = (arr_1 > 0.0).astype(float) - overlap
                arr_2_ = (arr_2 > 0.0).astype(float) - overlap
                arr_1[overlap] = distance_transform_edt(1.0 - arr_2_)[overlap]
                arr_2[overlap] = distance_transform_edt(1.0 - arr_1_)[overlap]
                arr_1_[overlap] = arr_1[overlap]
                arr_2_[overlap] = arr_2[overlap]
                sum_ = arr_1_ + arr_2_
                sum_[sum_ == 0] = 1.0
                count_list[item[0]] = arr_1 / sum_
                count_list[item[1]] = arr_2 / sum_
            else:
                continue

        counts_arr = np.stack(count_list)
        target_arr = np.stack(target_list) * counts_arr
        target_arr = target_arr.sum(0)

        # counts_arr[counts_arr == 0] = 1
        # target_arr /= counts_arr

        out_skit = np_to_skit(target_arr, target)

        off_z_min = 0

        while target_arr[:, :, off_z_min].sum() == 0:
            off_z_min += 1

        off_z_max = -1
        while target_arr[:, :, off_z_max].sum() == 0:
            off_z_max -= 1
        out_skit: sitk.Image = out_skit[:, :, off_z_min:off_z_max]  #
        if verbose:
            print(to_str_sitk(out_skit))
            print("[#] Write Image ", filename_out)
        sitk.WriteImage(out_skit, filename_out)
        print("done. took ", time.time() - stime)


if __name__ == "__main__":
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

    args = parser.parse_args()
    if args.verbose:
        print(args)
    main(args.images, args.output, args.margin, args.averaging, args.verbose)
# python stitching.py -i /media/data/NAKO/MRT/3D_GRE_TRA_in/102423_20/ses-20140811/sub-102423_20_ses-20140811_sequ-1_mr.nii.gz \
#    /media/data/NAKO/MRT/3D_GRE_TRA_in/102423_20/ses-20140811/sub-102423_20_ses-20140811_sequ-2_mr.nii.gz\
#    /media/data/NAKO/MRT/3D_GRE_TRA_in/102423_20/ses-20140811/sub-102423_20_ses-20140811_sequ-3_mr.nii.gz\
#    /media/data/NAKO/MRT/3D_GRE_TRA_in/102423_20/ses-20140811/sub-102423_20_ses-20140811_sequ-4_mr.nii.gz\
#    /media/data/NAKO/MRT/3D_GRE_TRA_in/102423_20/ses-20140811/sub-102423_20_ses-20140811_sequ-5_mr.nii.gz\
#    /media/data/NAKO/MRT/3D_GRE_TRA_in/102423_20/ses-20140811/sub-102423_20_ses-20140811_sequ-6_mr.nii.gz\
#    -o 6test.nii.gz -a -v


# python stitching.py -i /media/data/new_NAKO/NAKO/MRT/rawdata_super_resolution/100/sub-100391/t1dixon/sub-100391_acq-ax_rec-in_chunk-1_sequ-HWS_t1dixon.nii.gz \
#  /media/data/new_NAKO/NAKO/MRT/rawdata_super_resolution/100/sub-100391/t1dixon/sub-100391_acq-ax_rec-in_chunk-3_sequ-LWS_t1dixon.nii.gz \
#  /media/data/new_NAKO/NAKO/MRT/rawdata_super_resolution/100/sub-100391/t1dixon/sub-100391_acq-ax_rec-in_chunk-2_sequ-LWS_t1dixon.nii.gz \
#  /media/data/new_NAKO/NAKO/MRT/rawdata_super_resolution/100/sub-100391/t1dixon/sub-100391_acq-ax_rec-in_chunk-2_sequ-BWS_t1dixon.nii.gz \
#    -o /media/data/new_NAKO/NAKO/MRT/rawdata_super_resolution/100/sub-100391/t1dixon/sub-100391_acq-ax_rec-in_sequ-full_t1dixon.nii.gz  -v

# python stitching.py -i /media/data/new_NAKO/NAKO/MRT/rawdata/100/sub-100391/t1dixon/sub-100391_acq-ax_rec-in_chunk-1_t1dixon.nii.gz \
#  /media/data/new_NAKO/NAKO/MRT/rawdata/100/sub-100391/t1dixon/sub-100391_acq-ax_rec-in_chunk-2_t1dixon.nii.gz \
#  /media/data/new_NAKO/NAKO/MRT/rawdata/100/sub-100391/t1dixon/sub-100391_acq-ax_rec-in_chunk-3_t1dixon.nii.gz \
#    -o /media/data/new_NAKO/NAKO/MRT/rawdata_super_resolution/100/sub-100391/t1dixon/sub-100391_acq-ax_rec-in_sequ-fullorg_t1dixon.nii.gz  -v


# python stitching.py -i /media/data/new_NAKO/NAKO/MRT/rawdata_super_resolution/100/sub-100206/t1dixon/sub-100206_acq-ax_rec-in_chunk-*_sequ-*_t1dixon.nii.gz \
#    -o /media/data/new_NAKO/NAKO/MRT/rawdata_super_resolution/100/sub-100206/t1dixon/sub-100206_acq-ax_rec-in_sequ-full_t1dixon.nii.gz  -v -a
# python stitching.py -i /media/data/new_NAKO/NAKO/MRT/rawdata/100/sub-100206/t1dixon/sub-100206_acq-ax_rec-in_chunk-*_t1dixon.nii.gz \
#   -o /media/data/new_NAKO/NAKO/MRT/rawdata_super_resolution/100/sub-100206/t1dixon/sub-100206_acq-ax_rec-in_sequ-fullorg_t1dixon.nii.gz  -v
