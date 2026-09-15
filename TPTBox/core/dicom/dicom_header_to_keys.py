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
    "m_ffe": "magnitude",
    "p_ffe": "phase",
    "magnitude": "magnitude",
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
    # Specific quantitative / specialised patterns MUST come before the greedy
    # ``.*t2.*`` / ``.*t1.*`` catch-alls below — otherwise "T2 STAR" / "T1 MAP"
    # get misclassified as plain T2w / T1w on the first-match win.
    ".*mp2rage.*": "MP2RAG",
    ".*t2\\s*star.*": "T2star",
    r".*t2\*.*": "T2star",
    ".*r2\\s*star.*": "R2star",
    r".*r2\*.*": "R2star",
    ".*swi.*": "SWI",
    ".*t1\\s*map.*": "T1map",
    ".*t2star\\s*map.*": "T2starmap",
    ".*t2\\s*map.*": "T2map",
    # Multi-echo VIBE / DIXON — NAKO Siemens ``ME_vibe_fatquant_*`` and the
    # generic ``mevibe``/``fatquant``/``fatfrac``/``pdff`` labels. Placed before
    # ``.*mdix.*`` so the multi-echo classification wins where both apply.
    ".*me[_\\s]?vibe.*": "mevibe",
    ".*mevibe.*": "mevibe",
    ".*fatquant.*": "mevibe",
    ".*fatfrac.*": "dixon",
    ".*pdff.*": "dixon",
    ".*ideal.*": "dixon",  # GE's Dixon variant
    # Philips-specific localizers / reference scans that the existing "pilot"
    # / "scout" entries above don't catch.
    ".*survey.*": "localizer",
    ".*ref\\s*scan.*": "localizer",
    ".*smartexam.*": "localizer",
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
    ".*mdixon.*": "dixon",
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


def _single_echo_for_plane(dicoms: list[pydicom.FileDataset]) -> list[pydicom.FileDataset]:
    """Return a DICOM subset with one echo per slice position for plane detection.

    Multi-echo Philips DIXON (e.g. "mDIX quant") exports N slice positions × M
    echos into a single DICOM sub-group. The M copies at each `ImagePositionPatient`
    collapse the slice axis to ≈0 in `dicom2nifti.common.create_affine`, so after
    clamping by `hires_threshold` every zoom is ~1 and the series is misdetected
    as isotropic. Keep only the smallest `EchoNumbers` value so each spatial
    position is represented once. No-op when the tag is absent or constant.
    """
    en_values = set()
    for d in dicoms:
        try:
            en = int(getattr(d, "EchoNumbers", 0) or 0)
        except (TypeError, ValueError):
            continue
        if en > 0:
            en_values.add(en)
    if len(en_values) <= 1:
        return dicoms
    keep = min(en_values)
    return [d for d in dicoms if int(getattr(d, "EchoNumbers", 0) or 0) == keep]


def _apply_view_keys(keys: dict, get: Callable) -> None:
    """Populate `acq` / `part` from DICOM ViewPosition + Laterality tags.

    Used by the 2D-modality fallback in :func:`extract_keys_from_json`. Without
    this, MG series with four views (R-CC, L-CC, R-MLO, L-MLO), radiographs
    with AP / PA / LAT projections, and ophthalmic photos of both eyes would
    all collapse onto the same BIDS filename and clobber each other.

    Convention chosen (matching this codebase's flexible ``acq`` usage):

    * ``ViewPosition`` (e.g. ``CC``, ``MLO``, ``AP``, ``PA``, ``LAT``) →
      lowercased into ``acq``. Only overwrites the existing ``acq`` when the
      plane-detector returned ``None`` or ``"iso"``, both of which are
      meaningless for single-slice imagery.
    * ``ImageLaterality`` / ``Laterality`` (``L`` / ``R``, or ophthalmic
      ``OS`` / ``OD`` → mapped to ``L`` / ``R``) → ``part``, only when
      ``part`` is not already set by the DIXON / ImageType branches above.
    """
    view = get("ViewPosition")
    if view:
        view_clean = str(view).lower().strip("-.")
        if view_clean and keys.get("acq") in (None, "iso"):
            keys["acq"] = view_clean
    laterality = get("ImageLaterality") or get("Laterality")
    if laterality:
        lat = str(laterality).upper()
        # Ophthalmic (OS = oculus sinister = left, OD = oculus dexter = right)
        # normalises to the same L/R vocabulary that radiography uses.
        lat = {"OS": "L", "OD": "R"}.get(lat, lat)
        if lat in {"L", "R", "B"} and keys.get("part") is None:
            keys["part"] = lat.lower()


def _apply_bodypart_key(keys: dict, get: Callable) -> None:
    """Populate ``desc`` from DICOM ``BodyPartExamined`` when nothing else has set it.

    ``BodyPartExamined`` (0018,0015) is a semi-standardised free-text tag with
    common values like ``ABDOMEN``, ``PELVIS``, ``ABDOMENPELVIS``, ``CHEST``,
    ``HEAD``, ``NECK``, ``SPINE``, ``KNEE``, ``HIP``, ``BREAST``. It is the
    main discriminator when a single session contains scans of several body
    regions and the ``SeriesDescription`` is not informative — typical for
    plain radiography, ultrasound, nuclear medicine, and RT objects. Only
    written when ``keys['desc']`` is empty, so an earlier branch that already
    assigned ``desc`` (e.g. the SR / report path) wins.
    """
    if keys.get("desc") is not None:
        return
    body = get("BodyPartExamined")
    if not body:
        return
    val = str(body).lower().strip("-.")
    if val:
        keys["desc"] = val


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
        sorted_dicoms = common.sort_dicoms(_single_echo_for_plane(dicoms))
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
    except (AttributeError, IndexError, KeyError, TypeError):
        # Not usable image geometry: non-imaging DICOMs legally lack
        # `ImagePositionPatient` / `ImageOrientationPatient` (AttributeError),
        # empty lists trip `create_affine` on `dicoms[0]` (IndexError), and
        # callers that hand in dicts or other pydicom-shaped-but-not-really
        # objects raise KeyError / TypeError. All of these mean "no plane to
        # compute" — return None silently instead of surfacing the noise.
        return None
    except Exception as e:  # noqa: BLE001
        # Log so a downstream `acq-None` filename can be traced back to its cause,
        # instead of the plane-detection silently swallowing every failure.
        try:
            from TPTBox import Print_Logger

            Print_Logger().on_warning(f"get_plane_dicom: plane detection failed ({type(e).__name__}: {e}); returning None.")
        except Exception:  # noqa: BLE001
            pass
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
        # Read PatientID directly from simp_json — `_get` rewrites `_` to `-`,
        # which would destroy the `<sub>_<sescode>` split we need below.
        pid_raw = str(simp_json.get("PatientID", "unnamed")).strip()
        sub_part, _sep, ses_part = pid_raw.partition("_")
        keys["sub"] = re.sub(r'[<>:"/\\|?*\x00-\x1F\s]', "", sub_part) or "unnamed"
        # NAKO encodes the exam wave as a suffix on PatientID:
        # `<sub>_30` = U1 Baseline, `<sub>_60` = U2 Follow-up. The main NAKO
        # baseline export has no suffix; leave `ses` untouched there so the
        # existing `use_session` (StudyDate) fallback in _get_paths still wins.
        if session and ses_part:
            _nako_ses_map = {"30": "baseline", "60": "followup"}
            ses_clean = re.sub(r'[<>:"/\\|?*\x00-\x1F\s]', "", ses_part)
            keys["ses"] = _nako_ses_map.get(ses_clean, ses_clean)
        # Raw values for pattern matching — `_get` rewrites `_`→`-`, which
        # would break every `T2_TSE` / `3D_GRE_TRA` / `T1_3D_SAG` check below
        # and the `ProtocolName.split("_")` chunk derivation.
        series_description = str(simp_json.get("SeriesDescription", "unnamed"))
        protocol_name = str(simp_json.get("ProtocolName", "unnamed"))
        sequ = simp_json.get("SeriesNumber")
        """Determine the MRI format based on the series description."""
        if "T2_TSE" in series_description:
            return "T2w", {"acq": "sag", "chunk": series_description.rsplit("_", maxsplit=1)[-1], "sequ": sequ, **keys}, ".nii.gz"
        elif "3D_GRE_TRA" in series_description:
            return (
                "vibe",
                {
                    "acq": "ax",
                    "part": dixon_mapping[series_description.rsplit("_", maxsplit=1)[-1].lower()],
                    "chunk": protocol_name.rsplit("_", maxsplit=1)[-1],
                    **keys,
                },
                ".nii.gz",
            )
        elif "ME_vibe" in series_description:
            return (
                "mevibe",
                {"acq": "ax", "part": dixon_mapping[series_description.rsplit("_", maxsplit=1)[-1].lower()], "sequ": sequ, **keys},
                ".nii.gz",
            )
        elif "T1_3D_SAG" in series_description:
            # NAKO-1157 head T1 — plain sagittal ND and the MPR-Tra reformat.
            acq = "tra" if "MPR_Tra" in series_description else "sag"
            return "T1w", {"acq": acq, "sequ": sequ, **keys}, ".nii.gz"
        elif "FLAIR" in series_description:
            # NAKO-1157 head FLAIR (2D transverse).
            acq = "tra" if "TRA" in series_description else "sag"
            return "FLAIR", {"acq": acq, "sequ": sequ, **keys}, ".nii.gz"
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
            keys["acq"] = to_nii(dcm_data_l).get_plane(0.8)
        else:
            keys["acq"] = get_plane_dicom(dcm_data_l, 0.8)
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
            _apply_bodypart_key(keys, _get)
        elif modality.lower() == "pt":
            mri_format = "pet"
            _apply_bodypart_key(keys, _get)
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
        # Non-imaging metadata DICOMs: Presentation State, Key Object Selection,
        # Registration, Fiducials, Real World Value Map, Plan, Slide Stainer.
        # Also physiological waveforms (RESP, HD, ECG, EPS) and ophthalmic
        # measurements (AR, KER, LEN, VA, OPV, OPM) — none of these carry a
        # NIfTI-shaped pixel volume. Route them through the same `.txt` report
        # path as SR so the caller neither writes an empty NIfTI nor crashes
        # in `_add_grid_info_to_json` on a file that was never produced.
        elif modality.lower() in {
            "pr",
            "ko",
            "reg",
            "fid",
            "rwv",
            "plan",
            "stain",
            "resp",
            "hd",
            "ecg",
            "eps",
            "ar",
            "ker",
            "len",
            "va",
            "opv",
            "opm",
        }:
            keys["desc"] = _get("SeriesDescription", None)
            return modality.lower(), keys, ".txt"
        # Sensible defaults for the remaining common imaging modalities so we can
        # keep converting instead of raising on every non-CT/PET/MR/XA series.
        # Format names mirror BIDS conventions where they exist and fall back to
        # the lowercased DICOM modality tag otherwise (e.g. `us`, `nm`, `sc`).
        # For 2D modalities we also lift the DICOM ViewPosition / Laterality tags
        # into `acq`, otherwise files that only differ by view (R-CC vs L-CC vs
        # R-MLO vs L-MLO for MG, AP vs PA vs LAT for DX/CR) would all collapse
        # to the same BIDS name.
        elif modality.lower() in {"cr", "dx", "rg", "px", "io", "mg"}:
            # 2D X-ray family: computed / digital radiography, general radiographic,
            # panoramic, intra-oral, mammography. Kept under one `xray` bucket.
            mri_format = "xray"
            _apply_view_keys(keys, _get)
            _apply_bodypart_key(keys, _get)
        elif modality.lower() == "us":
            mri_format = "us"  # ultrasound
            _apply_view_keys(keys, _get)
            _apply_bodypart_key(keys, _get)
        elif modality.lower() == "nm":
            mri_format = "nm"  # nuclear medicine (planar/SPECT)
            # Radiopharmaceutical (tracer) is the useful discriminator for NM —
            # e.g. FDG, PSMA, DOTATATE. When present, surface it as `ce`.
            tracer = _get("Radiopharmaceutical")
            if tracer and keys.get("ce") is None:
                keys["ce"] = tracer
            _apply_bodypart_key(keys, _get)
        elif modality.lower() == "sc":
            mri_format = "sc"  # secondary capture (screenshots, derived stills)
            _apply_bodypart_key(keys, _get)
        elif modality.lower() in {"op", "xc"}:
            mri_format = "photo"  # ophthalmic / external photography
            # Ophthalmic photos: OS = left eye, OD = right eye → same L/R signal
            # as radiography Laterality; reuse the same helper.
            _apply_view_keys(keys, _get)
            _apply_bodypart_key(keys, _get)
        elif modality.lower() == "es":
            mri_format = "endoscopy"
            _apply_bodypart_key(keys, _get)
        elif modality.lower() in {"rtimage", "rtstruct", "rtdose", "rtplan"}:
            mri_format = modality.lower()  # radiotherapy objects
            _apply_bodypart_key(keys, _get)
        elif modality.lower() == "ot":
            mri_format = "ot"  # explicit "Other" modality
            _apply_bodypart_key(keys, _get)
        else:
            # Unknown modality — warn once and fall back to a mri_format derived
            # from the modality tag so extraction can still complete. Callers
            # that really need to reject unknown modalities can inspect the
            # returned mri_format.
            from TPTBox import Print_Logger

            Print_Logger().on_warning(
                f"extract_keys_from_json: unhandled modality={modality!r} "
                f"({modalities.get(modality.upper(), 'Non Standard Modality key')}); "
                "falling back to modality tag as mri_format."
            )
            mri_format = str(modality).lower() or "mr"
            _apply_bodypart_key(keys, _get)

            # ".*sub.*t1.*": "subtraktion",
        # "subtraktion.*t1.*": "subtraktion",
        return mri_format, keys, ".nii.gz"
