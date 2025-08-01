### Legal formats ###
# https://flexikon.doccheck.com/de/MRT-Sequenz
from __future__ import annotations

formats = [
    "ct",
    "dixon",
    "T2c",
    "T1c",
    "pd",
    "mevibe",  # Multi Echo Volume Interpolated Breathhold Examination
    "vibe",  # Volume Interpolated Breathhold Examination
    "T1w",
    "T2w",  # TSE/FSE
    "T2star",
    "TIRM",  # (Turbo-Inversion Recovery-Magnitude)
    "STIR",  # (Short-Tau Inversion Recovery)
    "T2haste",
    "flair",
    "DWI",  # Diffusion weighted imaging
    "IR",  # Inversion recovery
    "SE",  # Spine ech
    "SE-SAT",  # Spin-Echo fat saturation, Synonyme: FAT SAT, FRF-SE, SPIR, CHESS, SPECIAL)
    "GRE",  # Gradient-Echo
    "PWI",  # Perfusion weighted imaging
    "DTI",  # Diffusion tensor imaging
    "R2star",
    "radiomics",
    "MPR",  # Rapid gradient-echo (MP RAGE) sampling
    "IRT",
    "MES",
    "MEGR",
    "MP2RAG",
    "MPM",
    "MT",
    "MT",
    "T1map",
    "T2map",
    "T2starmap",
    "R1map",
    "R2map",
    "R2starmap",
    "PDmap",
    "MTRmap",
    "MTsat",
    "UNIT1",
    "T1rho",
    "MWFmap",
    "MTVmap",
    "Chimap",
    "S0map",
    "M0ma",
    "T1w",
    "T2w",
    "PDw",
    "T2starw",
    "FLAIR",
    "inplaneT1",
    "inplaneT2",
    "PDT2",
    "angio",
    "T2star",
    "FLASH",
    "VF",
    "defacemas",
    "dw",
    "TB1TFL",
    "TB1RFM",
    "RB1CO",
    "TB1AF",
    "TB1DA",
    "TB1EP",
    "TB1SRG",
    "TB1ma",
    "RB1ma",
    "ep",
    "phase1",
    "phase2",
    "magnitude1",
    "magnitude2",
    "magnitude",
    "fieldma",
    "phasedif",
    "sbre",
    "cb",
    "bol",
    "as",
    "m0sca",
    "aslcontex",
    "asllabelin",
    "phas",
    "noR",
    "ee",
    "iee",
    "sti",
    "ME",
    "channel",
    "coordsyste",
    "headshap",
    "electrode",
    "marker",
    "me",
    "sti",
    "PE",
    "bloo",
    "pe",
    "event",
    "physio",
    "sti",
    "TEM",
    "SEM",
    "uCT",
    "BF",
    "DF",
    "PC",
    "DIC",
    "FLUO",
    "CONF",
    "PLI",
    "CARS",
    "2PE",
    "MPE",
    "SR",
    "NLO",
    "OCT",
    "SPIM",
    "XPC",
    "phot",
    "TOF",  # Time-of-flight
    "NerveVIEW",  # https://www.philips.de/healthcare/product/HCNMRB971/3D-NerveVIEW-Klinische-MR-Anwendung
    "3DDrive",  # https://www.philips.de/healthcare/product/HCNMRB178/3D-DRIVE-MR-Software
    "DCE",  # dynamic contrast-enhanced () "
    "s3D",
    "FFE",
    "SWI",
    "CISS",
    "compare",
    "recon",
    "reformat",
    "subtraction",
    "RI",  # Raw input
    "stat",
    "snp",
    "log",
    "msk",
    "ctd",
    "model",
    "poi",
    "uncertainty",
    "angles",
    "subvar",
    "logit",
    "localizer",
    "difference",
    "labels",
]
# https://bids-specification.readthedocs.io/en/stable/appendices/entity-table.html
formats_relaxed = [*formats, "t2", "t1", "t2c", "t1c", "cta", "mr", "snapshot", "t1dixon", "dwi"]
# Recommended writing style: T1c, T2c; This list is not official and can be extended.


# Actual official final folder
# func (task based and resting state functional MRI)
# dwi (diffusion weighted imaging)
# fmap (field inhomogeneity mapping data such as field maps)
# anat (structural imaging such as T1, T2, PD, and so on)
# perf (perfusion)
# meg (magnetoencephalography)
# eeg (electroencephalography)
# ieeg (intracranial electroencephalography)
# beh (behavioral)
# pet (positron emission tomography)
# micr (microscopy)


file_types = [
    "nii.gz",
    "json",
    "mrk.json",
    "png",
    "jpg",
    "tsv",
    "backup",
    "ply",
    "npz",
    "log",
    "txt",
    "stl",
    "csv",
    "subvar",
    "pkl",
    "xlsx",
    "bvec",
    "bval",
    "html",
]
# Description see: https://bids-specification.readthedocs.io/en/stable/99-appendices/09-entities.html

# Order of Entities defines order of file naming
entities = {
    # Person
    "Subject": "sub",
    "Session": "ses",
    # Recording
    ## OURS
    "Sequence": "sequ",  # Not BIDS conform :( Try to replace and remove this key
    # Should be acq+chunk+ce*+trc*+rec*+mod*+run instead. where * is used only im applicable.
    # Run is used when the file collides with an other. Run gets a increasing number
    ## Official
    "Acquisition": "acq",
    # sag, ax, iso usw. and In case different sequences are used to record the same modality (for example, RARE and FLASH for T1w)
    # Examples: sag, sag-RARE, RARE
    "Task": "task",
    "Chunk/Location": "chunk",  # Location or index
    "Hemisphere": "hemi",  # [L,R]
    "Sample": "sample",  # such as tissue, primary cell or cell-free sample.
    # Sub recordings - Use when necessary.
    ## Contrast
    "Contrast enhancement phase": "ce",
    "Tracer": "trc",  # use ce before this one.
    "stain": "stain",  # like trc,ce but for stains/antibodies for contrast
    ## Reconstruction and modalities
    "Reconstruction": "rec",
    "Processed (on device)": "proc",  # never used
    "Corresponding Modality": "mod",  # Use only sublementiral material. (aka this belongs to ..._[mod].nii.gz)
    "Recording": "recording",
    "Resolution": "res",
    ### MRI parameters (Never used)
    "Reconstruction Direction": "dir",
    "Echo": "echo",  # Index!
    "Flip Angle": "flip",  # Index!
    "Inversion Time": "inv",  # Index!
    "Magnetization Transfer": "mt",  # [on, off]
    "Part": "part",  # [mag, phase, real, imag]
    "Space": "space",
    ## OURS
    "Segmentation": "seg",
    "Source": "source",
    # "snapshot": "snapshot",
    "ovl": "ovl",
    # General parameters
    # Others (often used)
    "Run/ID": "run",
    # Single class segmentation
    "Label": "label",
    # Others (never used)
    "Split": "split",
    "Density": "den",
    "Description": "desc",
    "nameconflict": "nameconflict",
}
entities_keys = {v: k for k, v in entities.items()}

entity_alphanumeric = [entities[i] for i in ["Task", "Reconstruction Direction"]]
entity_decimal = [entities[i] for i in ["Run/ID", "Echo", "Flip Angle", "Inversion Time", "nameconflict"]]
entity_format = [entities[i] for i in ["Corresponding Modality"]]
entity_on_off = [entities[i] for i in ["Magnetization Transfer"]]
entity_left_right = [entities[i] for i in ["Hemisphere"]]
entity_parts = [entities[i] for i in ["Part"]]


parents = ["sourcedata", "rawdata", "derivatives"]

sequence_splitting_keys = [
    "ses",
    "sequ",
    "acq",
    "hemi",
    "sample",
    "ce",
    "trc",
    "stain",
    # "rec",
    # "proc",
    "res",
    "dir",
    "run",
    # "desc",
    "split",
    "chunk",
]
sequence_naming_keys = ["seg", "label"]
# sequence_mix_format_and_naming_formats = ["ctd", "msk"]
