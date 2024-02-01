from collections.abc import Sequence
from enum import Enum
from typing import Literal

import numpy as np

from TPTBox.logger import log_file

log = log_file.Reflection_Logger()
logging = bool | log_file.Logger_Interface
# R: Right, L: Left; S: Superior (up), I: Inferior (down); A: Anterior (front), P: Posterior (back)
Directions = Literal["R", "L", "S", "I", "A", "P"]
Ax_Codes = tuple[Directions, Directions, Directions]
LABEL_MAX = 256
Zooms = tuple[float, float, float] | Sequence[float]

Centroid_Dict = dict[int, tuple[float, float, float]]
Triple = tuple[float, float, float] | Sequence[float]
Coordinate = Triple
POI_Dict = dict[int, dict[int, Coordinate]]


Rotation = np.ndarray
Label_Map = dict[int | str, int | str] | dict[str, str] | dict[int, int]

Label_Reference = int | Sequence[int] | None


plane_dict: dict[Directions, str] = {"S": "ax", "I": "ax", "L": "sag", "R": "sag", "A": "cor", "P": "cor"}
same_direction: dict[Directions, Directions] = {"S": "I", "I": "S", "L": "R", "R": "L", "A": "P", "P": "A"}


class Location(Enum):
    Unknown = 0
    # Vertebral subregions
    Vertebra_Full = 40
    Arcus_Vertebrae = 41
    Spinosus_Process = 42
    Costal_Process_Left = 43
    Costal_Process_Right = 44
    Superior_Articular_Left = 45
    Superior_Articular_Right = 46
    Inferior_Articular_Left = 47
    Inferior_Articular_Right = 48
    Vertebra_Corpus_border = 49
    Vertebra_Corpus = 50
    Dens_axis = 51  # TODO Unused. Should be in C2
    Vertebral_Body_Endplate_Superior = 52
    Vertebral_Body_Endplate_Inferior = 53
    Superior_Articulate_Process_Facet_Joint_Surface_Left = 54
    Superior_Articulate_Process_Facet_Joint_Surface_Right = 55
    Inferior_Articulate_Process_Facet_Joint_Surface_Left = 56
    Inferior_Articulate_Process_Facet_Joint_Surface_Right = 57
    Vertebra_Disc_Superior = 58
    Vertebra_Disc_Inferior = 59
    Vertebra_Disc = 100
    Spinal_Cord = 60
    Spinal_Canal = 61  # Center of vertebra
    Spinal_Canal_ivd_lvl = 126
    Endplate = 62
    # 63-80 Free
    # Muscle inserts
    # https://www.frontiersin.org/articles/10.3389/fbioe.2022.862804/full
    # 81-91
    Muscle_Inserts_Spinosus_Process = 81
    Muscle_Inserts_Transverse_Process_left = 83
    Muscle_Inserts_Transverse_Process_right = 82
    Muscle_Inserts_Vertebral_Body_left = 84
    Muscle_Inserts_Vertebral_Body_right = 85
    Muscle_Inserts_Articulate_Process_Inferior_left = 86
    Muscle_Inserts_Articulate_Process_Inferior_right = 87
    Muscle_Inserts_Articulate_Process_Superior_left = 88
    Muscle_Inserts_Articulate_Process_Superior_right = 89
    #
    Implant_Entry_Left = 90
    Implant_Entry_Right = 91
    Implant_Target_Left = 92
    Implant_Target_Right = 93

    # Muscle_Inserts_Rib_left = 90
    # Muscle_Inserts_Rib_right = 91
    # Ligament attachment points
    # 101-151
    # 101-108;
    # 101-104 are the 4 corners of a center sagittal cut starting top, than front
    # 105-108; Centers between the 4 corners, same order. Starting between up-front/up-back
    Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median = 101
    Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Median = 102
    Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Median = 103
    Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Median = 104
    Additional_Vertebral_Body_Middle_Superior_median = 105
    Additional_Vertebral_Body_Posterior_Central_median = 106
    Additional_Vertebral_Body_Middle_Inferior_median = 107
    Additional_Vertebral_Body_Anterior_Central_median = 108
    # 109-116 Same but shifted to the left
    Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left = 109
    Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Left = 110
    Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Left = 111
    Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Left = 112
    Additional_Vertebral_Body_Middle_Superior_left = 113
    Additional_Vertebral_Body_Posterior_Central_left = 114
    Additional_Vertebral_Body_Middle_Inferior_left = 115
    Additional_Vertebral_Body_Anterior_Central_left = 116
    # 117-124 Same but shifted to the right
    Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right = 117
    Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Right = 118
    Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Right = 119
    Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Right = 120
    Additional_Vertebral_Body_Middle_Superior_right = 121
    Additional_Vertebral_Body_Posterior_Central_right = 122
    Additional_Vertebral_Body_Middle_Inferior_right = 123
    Additional_Vertebral_Body_Anterior_Central_right = 124

    Ligament_Attachment_Point_Flava_Superior_Median = 125
    Ligament_Attachment_Point_Flava_Inferior_Median = 127

    # Ligament_Attachment_Point_Flava_Superior_Right = 141
    # Ligament_Attachment_Point_Flava_Inferior_Right = 143
    # Ligament_Attachment_Point_Flava_Superior_Left = 149
    # Ligament_Attachment_Point_Flava_Inferior_Left = 151

    # Ligament_Attachment_Point_Interspinosa_Superior_Left = 133
    # Ligament_Attachment_Point_Interspinosa_Superior_Right = 134
    # Ligament_Attachment_Point_Interspinosa_Inferior_Left = 135
    # Ligament_Attachment_Point_Interspinosa_Inferior_Right = 136

    Multi = 256

    def __repr__(self):
        return self.name


def vert_subreg_labels(with_border: bool = True) -> list[Location]:
    labels = [
        Location.Arcus_Vertebrae,
        Location.Spinosus_Process,
        Location.Costal_Process_Left,
        Location.Costal_Process_Right,
        Location.Superior_Articular_Left,
        Location.Superior_Articular_Right,
        Location.Inferior_Articular_Left,
        Location.Inferior_Articular_Right,
        Location.Vertebra_Corpus,
    ]
    if with_border:
        labels.append(Location.Vertebra_Corpus_border)
    return labels


# fmt: off

subreg_idx2name = {}
for k in range(255):
    try:
        subreg_idx2name[k] = Location(k).name
    except Exception:
        pass
v_idx2name = {
     1: "C1",     2: "C2",     3: "C3",     4: "C4",     5: "C5",     6: "C6",     7: "C7",
     8: "T1",     9: "T2",    10: "T3",    11: "T4",    12: "T5",    13: "T6",    14: "T7",    15: "T8",    16: "T9",    17: "T10",   18: "T11",   19: "T12", 28: "T13",
    20: "L1",    21: "L2",    22: "L3",    23: "L4",    24: "L5",    25: "L6",
    26: "S1",    29: "S2",    30: "S3",    31: "S4",    32: "S5",    33: "S6",
    27: "Cocc", **subreg_idx2name
}
v_name2idx:dict[str, int] = {value: key for key,value in v_idx2name.items()}
sub_name2idx:dict[str, int] = {value: key for key,value in subreg_idx2name.items()}
v_idx_order = list(v_idx2name.keys())

# fmt: on

conversion_poi = {
    "SSL": 81,  # this POI is not included in our POI list
    "ALL_CR_S": 109,
    "ALL_CR": 101,
    "ALL_CR_D": 117,
    "ALL_CA_S": 111,
    "ALL_CA": 103,
    "ALL_CA_D": 119,
    "PLL_CR_S": 110,
    "PLL_CR": 102,
    "PLL_CR_D": 118,
    "PLL_CA_S": 112,
    "PLL_CA": 104,
    "PLL_CA_D": 120,
    "FL_CR_S": 149,
    "FL_CR": 125,
    "FL_CR_D": 141,
    "FL_CA_S": 151,
    "FL_CA": 127,
    "FL_CA_D": 143,
    "ISL_CR": 134,
    "ISL_CA": 136,
    "ITL_S": 142,
    "ITL_D": 144,
}
conversion_poi2text = {k: v for v, k in conversion_poi.items()}
