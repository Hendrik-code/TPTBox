import re

import dicom2nifti.exceptions
import nibabel.orientations as nio
import numpy as np
import pydicom
from dicom2nifti import common

from TPTBox.core.nii_wrapper import NII

dixon_mapping = {
    "f": "fat",
    "w": "water",
    "in": "inphase",
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
    "out-phase": "outphase",
    "phase": "inphase",
    "wa": "water",
}
dixon_mapping = {**dixon_mapping, **{v: v for v in dixon_mapping.values()}}
map_series_description_to_file_format = {
    ".*t2w?_tse.*": "T2w",
    "t2w?_fse.*": "T2w",
    ".*t1w?_tse.*": "T1w",
    ".*t1w?_vibe_tra.*": "vibe",
    ".*flair.*": "flair",
    ".*stir.*": "STIR",
    ".*dti.*": "DTI",
    ".*dwi.*": "DWI",
    ".*dir.*": "DIR",
    "se": "SE",  # Spine echo
    ".* fir .*": "IR",  # fast inversion recovery
    ".*irfse.*": "IR",  # fast inversion recovery
    "ir_.*": "IR",  # inversion recovery
    ".*mp?ra?ge?.*": "MPRAGE",
    ".*mip.*": "MIP",
    "b0map": "b0map",
    ".*t2.*": "T2w",
    ".*t1.*": "T1w",
    # others
    "shim2d": "mr",
    "3-plane loc": "mr",
    "fgr": "mr",  # Fetal growth restriction (FGR) ????
    "screen save": "mr",
    ".*¶.*": "mr",
    ".*scout": "mr",
    "localizer": "mr",
    ".*pilot.*": "mr",
    ".*2d.*": "mr",
    ".*scno.*": "mr",
    ".*scano.*": "mr",
    "sys2dcard": "mr",
    "3-pl loc gr": "mr",
    re.escape("?") + "*": "mr",
    ".*": "mr",
}


def get_plane_dicom(dicoms: list[pydicom.FileDataset] | NII) -> str | None:
    """Determines the orientation plane of the NIfTI image along the x, y, or z-axis.

    Returns:
        str: The orientation plane of the image, which can be one of the following:
            - 'ax': Axial plane (along the z-axis).
            - 'cor': Coronal plane (along the y-axis).
            - 'sag': Sagittal plane (along the x-axis).
            - 'iso': Isotropic plane (if the image has equal zoom values along all axes).
    Examples:
        >>> nii = NII(nib.load("my_image.nii.gz"))
        >>> nii.get_plane()
        'ax'
    """
    if isinstance(dicoms, NII):
        return dicoms.get_plane()
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


def extract_keys_from_json(simp_json: dict, dcm_data_l: list[pydicom.FileDataset] | NII, session=False):
    def _get(key, default=None):
        if key not in simp_json:
            return default
        return str(simp_json[key])

    keys: dict[str, str | None] = {}

    """Extract keys from JSON based on study and series descriptions."""
    #### NAKO FIXED ####
    if "StudyDescription" in simp_json and "nako" in _get("StudyDescription", "").lower():
        keys["sub"] = _get("PatientID", "unnamed").split("_")[0]
        series_description = _get("SeriesDescription", "unnamed")
        """Determine the MRI format based on the series description."""
        if "T2_TSE" in series_description:
            return "T2w", {"acq": "sag", "chunk": series_description.split("_")[-1], "sequ": simp_json["SeriesNumber"], **keys}
        elif "3D_GRE_TRA" in series_description:
            return "vibe", {
                "acq": "ax",
                "part": dixon_mapping[series_description.split("_")[-1].lower()],
                "chunk": _get("ProtocolName", "unnamed").split("_")[-1],
                **keys,
            }
        elif "ME_vibe" in series_description:
            return "mevibe", {
                "acq": "ax",
                "part": dixon_mapping[series_description.split("_")[-1].lower()],
                "sequ": simp_json["SeriesNumber"],
                **keys,
            }
        elif "PD" in series_description:
            return "pd", {"acq": "iso", **keys}
        elif "T2_HASTE" in series_description:
            return "T2haste", {"acq": "ax", **keys}
        else:
            raise NotImplementedError(series_description)
    # GENERAL
    else:
        keys["sub"] = _get("PatientID")
        if session:
            keys["ses"] = _get("StudyDate")
        keys["acq"] = get_plane_dicom(dcm_data_l)
        keys["part"] = dixon_mapping.get(_get("ProtocolName", "NO-PART").split("_")[-1], None)
        # GET MRI FORMAT
        series_description = _get("SeriesDescription", "no_series_description").lower()
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
        for key, mri_format_new in map_series_description_to_file_format.items():
            regex = re.compile(key)
            if re.match(regex, series_description):
                mri_format = mri_format_new
                break
        if mri_format is None:
            mri_format = "mr"
        return mri_format, keys
