from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import dicom2nifti.exceptions
import nibabel.orientations as nio
import numpy as np
import pydicom
from dicom2nifti import common

from TPTBox.core.bids_constants import formats, modalities
from TPTBox.core.nii_wrapper import NII, to_nii

dixon_mapping = {
    "f": "fat",
    "w": "water",
    "in": "inphase",
    "ip": "inphase",
    "opp": "outphase",
    "opp1": "eco0-opp1",
    "pip1": "eco1-pip1",
    "opp2": "eco2-opp2",
    "in1": "eco3-in1",
    "pop1": "eco4-pop1",
    "arb1": "eco5-arb1",
    "fp": "fat-fraction",
    "eff": "r2s",
    "wp": "water-fraction",
    "in-phase": "inphase",
    "in_phase": "inphase",
    "out-phase": "outphase",
    "out_phase": "outphase",
    # "phase": "inphase",
    "wa": "water",
    "imaginary": "imag",
    "real": "real",
    "phase": "phase",
    "mag": "mag",
    "sub": "subtraction",
}
dixon_mapping = {**dixon_mapping, **{v: v for v in dixon_mapping.values()}}
map_series_description_to_file_format_default = {
    ".*t2w?_tse.*": "T2w",
    "t2w?_fse.*": "T2w",
    ".*t1w?_tse.*": "T1w",
    ".*t1w?_vibe_tra.*": "vibe",
    ".*Durchleuchtung.*": "fluroscopy",
    ".*fluroscopy.*": "fluroscopy",
    ".*scout": "localizer",
    "localizer": "localizer",
    ".*pilot.*": "localizer",
    "posdisp.*": "localizer",
    **{f".* {re.escape(k.lower())} .*": k for k in formats},
    ".*flair.*": "flair",
    ".*stir.*": "STIR",
    ".*dti.*": "DTI",
    ".*dwi.*": "DWI",
    ".*dir.*": "DIR",
    "se": "SE",  # Spine echo
    ".* fir .*": "IR",  # fast inversion recovery
    ".*irfse.*": "IR",  # fast inversion recovery
    "ir_.*": "IR",  # inversion recovery
    ".*mp?ra?ge?.*": "MPR",
    ".*mip.*": "MIP",
    "b0map": "b0map",
    ".*t2.*": "T2w",
    ".*t1.*": "T1w",
    ".*dixon.*": "dixon",
    ".*tof.*": "TOF",
    ".*adc.*": "DWI",
    ".*diff.*": "difference",
    ".*fl2d.*": "FLASH",
    ".*fl3d.*": "FLASH",
    ".*nerveview.*": ".*NerveVIEW.*",
    ".*drive.*": "3DDrive",
    ".*fa.*": "DTI",
    ".*sub.*": "subtraction",
    ".*dynamik.*": "DCE",
    ".*mdix.*": "dixon",
    ".*s3d.*": "s3D",
    ".*flip37.*": "s3D",
    ".*trak.*": "PWI",
    ".*trance.*": "PWI",
    # others
    ".*beschriftung.*": "localizer",
    ".*plan.*": "localizer",
    ".*localizer.*": "localizer",
    "3-plane loc": "localizer",
    "screen save": "localizer",
    "MobiView .*": "localizer",
    ".*vs.*": "compare",
    ".*com.*": "compare",
    ".*reformat.*": "reformat",
    ".*recon.*": "recon",
    ".*source.*ri.*": "RI",
    **{f".*{re.escape(k.lower())}.*": k for k in formats},
    re.escape("?") + "*": "mr",
    ".*": "mr",
}


def get_plane_dicom(dicoms: list[pydicom.FileDataset] | NII, hires_threshold: float = 0.8) -> str | None:
    """Determine the acquisition plane from a DICOM series or NIfTI image.

    Args:
        dicoms: Either a list of pydicom datasets (one per slice) representing
            a single DICOM series, or an already-loaded :class:`~TPTBox.NII`
            object.
        hires_threshold: Zoom threshold used to distinguish the slice axis from
            in-plane axes when all zooms are similar (iso detection).

    Returns:
        One of ``'ax'`` (axial), ``'cor'`` (coronal), ``'sag'`` (sagittal),
        ``'iso'`` (isotropic), or ``None`` on failure.

    Examples:
        >>> nii = NII(nib.load("my_image.nii.gz"))
        >>> nii.get_plane()
        'ax'
    """
    if isinstance(dicoms, NII):
        return dicoms.get_plane(res_threshold=hires_threshold)
    try:
        sorted_dicoms = common.sort_dicoms(dicoms)
        affine, _ = common.create_affine(sorted_dicoms)
        plane_dict = {"S": "ax", "I": "ax", "L": "sag", "R": "sag", "A": "cor", "P": "cor"}
        axc = np.array(nio.aff2axcodes(affine))
        affine = np.asarray(affine)
        q, p = affine.shape[0] - 1, affine.shape[1] - 1
        # extract the underlying rotation, zoom, shear matrix
        RZS = affine[:q, :p]  # noqa: N806
        zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
        # Zooms can be zero, in which case all elements in the column are zero, and
        # we can leave them as they are
        zooms[zooms == 0] = 1
        zooms = zooms if hires_threshold is None else tuple(max(i, hires_threshold) for i in zooms)
        zms = np.around(zooms, 1)
        ix_max = np.array(zms == np.amax(zms))
        num_max = np.count_nonzero(ix_max)
        if num_max == 2:
            plane = plane_dict[axc[~ix_max][0]]
        elif num_max == 1:
            plane = plane_dict[axc[ix_max][0]]
        else:
            plane = "iso"
        return plane  # noqa: TRY300
    except Exception:
        return None


def extract_keys_from_json(  # noqa: C901
    simp_json: dict,
    dcm_data_l: list[pydicom.FileDataset] | NII | Path,
    session: bool = False,
    parts: list[str] | None = None,
    map_series_description_to_file_format: dict | None = None,
    override_subject_name: Callable[[dict, Path], str] | None = None,
    chunk: int | str | None = None,
    keys: dict[str, str | None] | None = None,
) -> tuple[str, dict]:
    """Extract BIDS-style key-value pairs from a DICOM JSON metadata dictionary.

    Parses study and series descriptions together with DICOM tag values to
    infer the image format (e.g. ``"T2w"``, ``"vibe"``) and BIDS entities
    (``sub``, ``ses``, ``acq``, ``part``, ``chunk``, ``ce``, ``sequ``).
    Special handling is included for NAKO study data.

    Args:
        simp_json: Flattened DICOM metadata dict, typically produced by
            ``pydicom``'s JSON export or a BIDS sidecar.
        dcm_data_l: List of pydicom datasets for the series, an NIfTI object,
            or a Path to a NIfTI file. Used only to compute the acquisition
            plane when not already in ``simp_json``.
        session: If ``True``, populate the ``ses`` key from the study date.
        parts: Explicit Dixon part labels that override automatic detection.
        map_series_description_to_file_format: Custom regex-to-format mapping
            applied before the built-in defaults.
        override_subject_name: Optional callable that receives ``(simp_json,
            path)`` and returns the subject ID string.
        chunk: Explicit chunk identifier; overrides automatic detection.
        keys: Pre-populated BIDS key dict; updated in-place and returned.

    Returns:
        A tuple ``(mri_format, keys)`` where ``mri_format`` is the inferred
        image format string (e.g. ``"T2w"``) and ``keys`` is the updated BIDS
        entity dict.

    Raises:
        NotImplementedError: For unsupported modalities or unrecognised NAKO
            series descriptions.
    """
    if keys is None:
        keys = {}
    if map_series_description_to_file_format is None:
        map_series_description_to_file_format = {}
    if parts is None:
        parts = []

    def _get(key, default=None):
        if key not in simp_json:
            return keys.get(key, default)
        value = str(simp_json[key]).replace("_", "-").replace(" ", "-").replace(".", "-")
        # remove invalid filename characters
        value = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", value)
        # collapse repeated dashes
        value = re.sub(r"-+", "-", value)
        # strip leading/trailing dots and dashes
        value = value.strip(".-")

        return value

    #### NAKO FIXED ####
    if "StudyDescription" in simp_json and "nako" in _get("StudyDescription", "").lower():
        keys["sub"] = _get("PatientID", "unnamed").split("_")[0]
        series_description = _get("SeriesDescription", "unnamed")
        """Determine the MRI format based on the series description."""
        if "T2_TSE" in series_description:
            return "T2w", {"acq": "sag", "chunk": series_description.split("_")[-1], "sequ": simp_json["SeriesNumber"], **keys}, ".nii.gz"
        elif "3D_GRE_TRA" in series_description:
            return (
                "vibe",
                {
                    "acq": "ax",
                    "part": dixon_mapping[series_description.split("_")[-1].lower()],
                    "chunk": _get("ProtocolName", "unnamed").split("_")[-1],
                    **keys,
                },
                ".nii.gz",
            )
        elif "ME_vibe" in series_description:
            return (
                "mevibe",
                {"acq": "ax", "part": dixon_mapping[series_description.split("_")[-1].lower()], "sequ": simp_json["SeriesNumber"], **keys},
                ".nii.gz",
            )
        elif "PD" in series_description:
            return "pd", {"acq": "iso", **keys}, ".nii.gz"
        elif "T2_HASTE" in series_description:
            return "T2haste", {"acq": "ax", **keys}, ".nii.gz"
        else:
            raise NotImplementedError(series_description)
    # GENERAL
    else:
        if override_subject_name is not None:
            keys["sub"] = override_subject_name(
                simp_json,
                Path(str(dcm_data_l[0].filename)) if not isinstance(dcm_data_l, (str, Path, NII)) else dcm_data_l,  # type: ignore
            )
        else:
            keys["sub"] = _get("PatientID")
            if keys["sub"] is None:
                keys["sub"] = _get("StudyInstanceUID")
            if keys["sub"] is None:
                keys["sub"] = (
                    _get("PatientSex", "X")  # type: ignore
                    + "-"
                    + _get("PatientAge", "")
                    + "-"
                    + _get("PatientSize", "")
                    + "-"
                    + _get("PatientSex", "")
                    + "-"
                    + _get("PatientWeight", "")
                )
        if session:
            keys["ses"] = _get("StudyDate", keys.get("ses"))
        if isinstance(dcm_data_l, (str, Path, NII)):
            keys["acq"] = to_nii(dcm_data_l).get_plane(1)
        else:
            keys["acq"] = get_plane_dicom(dcm_data_l, 1)
        keys["part"] = dixon_mapping.get(_get("ProtocolName", "NO-PART").split("_")[-1])

        sequ = _get("SeriesNumber", None)
        if sequ is None:
            sequ = str(re.sub(r"[^0-9a-zA-Z]", "", str(simp_json.get("SeriesDescription", "")))).lower()
        if sequ != "":
            keys["sequ"] = sequ
        if len(parts) != 0:
            keys["part"] = "-".join(parts).replace("_", "-")
        if chunk is not None:
            keys["chunk"] = str(chunk)
        image_type = simp_json.get("ImageType", [])
        dx = [dixon_mapping[k.lower()] for k in image_type if k.lower() in dixon_mapping]
        if len(dx) != 0:
            keys["part"] = dx[0]
        # contrast agent
        # n Tag “ContrastAgent” oder “ContrastBolusTotalDose”, wenn
        ce = _get("ContrastAgent", _get("ContrastBolusIngredient"))
        if ce is not None:
            keys["ce"] = ce
        elif _get("ContrastBolusTotalDose") is not None or _get("ContrastBolusVolume") is not None:
            keys["ce"] = "ContrastAgent"
        # GET MRI FORMAT
        series_description = _get("SeriesDescription", "mr").lower()
        modality = _get("Modality", "mr").lower()

        mri_format = None
        ##################### Understand sequence by given times ####################
        # try:
        #    a, b = None, None
        #    if series_description.startswith("fse ") and "/" in series_description:
        #        # FSE [TR]/[TE] *
        #        a, b = series_description[4:].split(" ")[0].split("/")
        #        tr = float(a)
        #        te = float(b)
        #        if tr >= 2000 and (te < 150 and te > 80):
        #            mri_format = "T2w"
        #        print(series_description, "Tr", tr, "te", te, "format", mri_format, tr >= 2000)
        # except Exception:
        #    pass
        #################### Understand sequence by series_description ####################
        found = False
        if modality == "ct":
            mri_format = "ct"
        elif modality.lower() == "pt":
            mri_format = "pet"
        elif modality == "xa":  # Angiography
            biplane = False
            if "BIPLANE A" in image_type or "SINGLE A" in image_type:
                keys["acq"] = "A"
                biplane = True
            elif "BIPLANE B" in image_type or "SINGLE B" in image_type:
                keys["acq"] = "B"
                biplane = True
            derived = "DERIVED" in image_type
            series_description = _get("SeriesDescription", " ").lower()  # "SeriesDescription": "Durchleuchtung - gespeichert",
            monitor = _get("PositionerMotion", " ").lower()
            # ftv = _get("FrameTimeVector", None).lower()
            monitor = _get("PositionerMotion", " ").lower()
            tag = _get("DerivationDescription", " ").lower()
            # "ImagerPixelSpacing"
            # FrameTimeVector = _get("DerivationDescription", [])
            # ftv is not None
            if "durchleuchtung" in series_description or "fluroscopy" in series_description:
                mri_format = "fluroscopy"
            elif tag == "subtraction":
                mri_format = "DSA" if monitor == "static" and "VOLUME" not in image_type and "RECON" not in image_type else "subtraction"
            elif "3DRA_PROP" in image_type:
                mri_format = "3DRA"
            elif monitor == "dynamic" or "VOLUME" in image_type or "RECON" in image_type or "3DRA_PROP" in image_type:
                mri_format = "DSA3D"
            elif biplane and derived and "VOLUME" not in image_type and "RECON" not in image_type:
                ##len(FrameTimeVector) >= 1 and (monitor == "static" and "VOLUME" not in image_type and "RECON" not in image_type)
                mri_format = "DSA"
            else:
                mri_format = "XA"
        elif modality == "mr":
            for key, mri_format_new in map_series_description_to_file_format.items():
                regex = re.compile(key)
                if re.match(regex, series_description):
                    mri_format = mri_format_new
                    break
            if not found:
                for key, mri_format_new in map_series_description_to_file_format_default.items():
                    regex = re.compile(key)
                    if re.match(regex, series_description):
                        mri_format = mri_format_new
                        break
            if mri_format is None:
                mri_format = "mr"
            if mri_format == "T1w":
                if "sub" in series_description.lower() and keys.get("part") is None:
                    keys["part"] = "subtraction"
                if (
                    " km " in series_description.lower() or series_description.startswith("km") or series_description.endswith("km")
                ) and keys.get("ce") is None:
                    keys["ce"] = "ContrastAgent"
        elif modality.lower() == "pdf":
            return "report", keys, ".pdf"
        elif modality.lower() == "sr":
            keys["desc"] = _get("SeriesDescription", None)
            return "report", keys, ".txt"
        else:
            raise NotImplementedError(f"modality='{modality}', ({modalities.get(modality.upper(), 'Non Standard Modality key')})")

            # ".*sub.*t1.*": "subtraktion",
        # "subtraktion.*t1.*": "subtraktion",
        return mri_format, keys, ".nii.gz"
