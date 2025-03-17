from __future__ import annotations

# POI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
# packages
from TPTBox import core  # noqa: I001
from TPTBox.core import bids_files, np_utils

# BIDS
from TPTBox.core.bids_files import BIDS_FILE, BIDS_Family, BIDS_Global_info, Searchquery, Subject_Container

# NII
from TPTBox.core.nii_wrapper import (
    NII,
    Image_Reference,
    Interpolateable_Image_Reference,
    to_nii,
    to_nii_interpolateable,
    to_nii_optional,
    to_nii_seg,
    Has_Grid,
)
from TPTBox.core.poi import AX_CODES, POI, POI_Reference, calc_centroids, calc_poi_from_subreg_vert
from TPTBox.core.poi import calc_poi_from_two_segs
from TPTBox.core.poi import calc_poi_from_two_segs as calc_poi_labeled_buffered
from TPTBox.core.poi_fun.poi_global import POI_Global
from TPTBox.core.vert_constants import ZOOMS, Location, Vertebra_Instance, v_idx2name, v_idx_order, v_name2idx

# Logger
from TPTBox.logger import Log_Type, Logger, Logger_Interface, Print_Logger, String_Logger
from TPTBox.logger.log_file import No_Logger

Centroids = POI

__all__ = [
    "AX_CODES",
    "BIDS_FILE",
    "NII",
    "POI",
    "ZOOMS",
    "BIDS_Family",
    "BIDS_Global_info",
    "Image_Reference",
    "Interpolateable_Image_Reference",
    "Location",
    "Log_Type",
    "POI_Global",
    "POI_Reference",
    "Print_Logger",
    "Searchquery",
    "Subject_Container",
    "Vertebra_Instance",
    "bids_files",
    "calc_centroids",
    "calc_poi_from_subreg_vert",
    "calc_poi_from_two_segs",
    "core",
    "load_poi",
    "np_utils",
    "to_nii",
    "v_idx2name",
    "v_idx_order",
    "v_name2idx",
]
