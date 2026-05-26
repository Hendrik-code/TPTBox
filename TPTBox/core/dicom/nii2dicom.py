# Original Source: https://github.com/amine0110/nifti2dicom/blob/main/nifti2dicom.py
# Added that you can add json, we extract back into the dicom header
from __future__ import annotations

import json
import os
import time
from glob import glob
from pathlib import Path

import SimpleITK as sitk  # noqa: N813
from pydicom.tag import Tag

from TPTBox import Print_Logger


def writeSlices(series_tag_values: dict, new_img: sitk.Image, i: int, out_dir: str | Path, name: str = "slice") -> None:
    """Write a single axial slice of a SimpleITK image as a DICOM file.

    Applies shared series-level DICOM tags and slice-specific tags (instance
    creation date/time, image position, instance number) before writing.

    Args:
        series_tag_values: Dict of DICOM tag keys (e.g. ``"0008|0060"``) to
            values that are shared by all slices in the series.
        new_img: 3-D SimpleITK image from which the slice is extracted.
        i: Zero-based slice index along the Z axis.
        out_dir: Directory where the DICOM file is written.
        name: Filename prefix; the output file is named
            ``<name><i:04d>.dcm``.
    """
    image_slice: sitk.Image = new_img[:, :, i]
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    # Tags shared by the series.
    for k, v in series_tag_values.items():
        image_slice.SetMetaData(str(k), str(v))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time

    # Check if modality is specified in the metadata, otherwise set a default value
    modality = series_tag_values.get("0008|0060", "MR")  # Defaulting to MR (Magnetic Resonance)
    image_slice.SetMetaData("0008|0060", modality)  # Modality

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    image_slice.SetMetaData("0020|0032", "\\".join(map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
    image_slice.SetMetaData("0020|0013", str(i))  # Instance Number

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(str(Path(out_dir, f"{name}{str(i).zfill(4)}.dcm")))
    writer.Execute(image_slice)


def nifti2dicom_1file(
    in_nii: str | Path,
    out_dir: str | Path,
    no_json_ok: bool = False,
    secondary: bool = False,
    json_path: None | str | Path = None,
    out_name: str = "slice",
) -> None:
    r"""Convert a single NIfTI file to a DICOM series.

    Reads DICOM metadata from a sidecar JSON file (BIDSy format) when
    available, generates unique Series/Study Instance UIDs, and writes one
    DICOM file per axial slice.

    Args:
        in_nii: Path to the input NIfTI file.
        out_dir: Output directory; created automatically if it does not exist.
        no_json_ok: If ``True``, proceed even when no JSON sidecar is found.
            If ``False``, raises :class:`FileNotFoundError` when the sidecar
            is missing.
        secondary: If ``True``, marks the DICOM series as
            ``DERIVED\\SECONDARY``.
        json_path: Explicit path to the JSON metadata file. Defaults to the
            NIfTI stem with ``.json`` extension.
        out_name: Filename prefix for the individual DICOM slice files.

    Raises:
        FileNotFoundError: If the JSON sidecar is not found and
            ``no_json_ok`` is ``False``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Get JSON metadata
    json_path = Path(str(in_nii).split(".")[0] + ".json") if json_path is None else Path(json_path)
    if json_path.exists():
        with open(json_path) as j:
            meta: dict = json.load(j)
        series_tag_values = {}
        for k, v in meta.items():
            try:
                tag = Tag(k)
                a = tag.json_key
                a = a[:4] + "|" + a[4:]
                series_tag_values[a] = v
            except ValueError:
                Print_Logger().on_fail(k, "cannot be converted to DICOM tag")
    else:
        if not no_json_ok:
            raise FileNotFoundError(json_path)
        series_tag_values = {}

    new_img = sitk.ReadImage(in_nii)
    series_tag_values["0008|0031"] = time.strftime("%H%M%S")  # Modification Time
    series_tag_values["0008|0031"] = time.strftime("%Y%m%d")  # Modification Date
    direction = new_img.GetDirection()
    series_tag_values["0020|0037"] = "\\".join(
        map(str, (direction[0], direction[3], direction[6], direction[1], direction[4], direction[7]))
    )  # Image Orientation (Patient)

    if secondary:
        series_tag_values["0008|0008"] = "DERIVED\\SECONDARY"

    # Generate unique Series and Study Instance UIDs if not already present
    if "0020|000e" not in series_tag_values:
        series_tag_values["0020|000e"] = (
            f"1.2.826.0.1.3680043.2.1125.{series_tag_values['0008|0031']}.1{series_tag_values['0008|0031']}"  # Series Instance UID
        )
        series_tag_values["0020|000d"] = (
            f"1.2.826.0.1.3680043.2.1125.1{series_tag_values['0008|0031']}.1{series_tag_values['0008|0031']}"  # Study Instance UID
        )
        # series_tag_values["0008|103e"] = "Created-Pycad"
    else:
        series_tag_values["0020|000e"] = (
            f"{series_tag_values['0020|000e'][:27]}{series_tag_values['0008|0031']}.1{series_tag_values['0008|0031']}"  # Series Instance UID
        )
        series_tag_values["0020|000d"] = (
            f"{series_tag_values['0020|000e'][:28]}{series_tag_values['0008|0031']}.1{series_tag_values['0008|0031']}"  # Study Instance UID
        )
        # series_tag_values["0008|103e"] = "Created-Pycad"

    for i in range(new_img.GetDepth()):
        writeSlices(series_tag_values, new_img, i, out_dir, name=out_name)


def nifti2dicom_mfiles(nifti_dir: str | Path, out_dir: str | Path = "") -> None:
    """Convert all NIfTI files in a directory to individual DICOM series.

    Each ``.nii.gz`` file in ``nifti_dir`` is converted by calling
    :func:`nifti2dicom_1file`. Output sub-directories are created automatically.

    Args:
        nifti_dir: Directory containing ``.nii.gz`` files to convert.
        out_dir: Root output directory. Each NIfTI file's series is placed in
            a sub-directory named after the file stem.
    """
    images = glob(nifti_dir + "/*.nii.gz")  # noqa: PTH207

    for image in images:
        o_path = Path(out_dir, os.path.basename(image)[:-7])  # noqa: PTH119
        os.makedirs(o_path, exist_ok=True)  # noqa: PTH103

        nifti2dicom_1file(image, o_path)


if __name__ == "__main__":
    nifti2dicom_1file("sub-spinegan0004_ses-20210617_sequ-6_part-fat_dixon.nii.gz", "out_test")
