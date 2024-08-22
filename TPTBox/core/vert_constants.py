import typing
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Literal, NoReturn

import numpy as np

from TPTBox.logger import log_file

#####################
log = log_file.Reflection_Logger()
logging = bool | log_file.Logger_Interface
# R: Right, L: Left; S: Superior (up), I: Inferior (down); A: Anterior (front), P: Posterior (back)
DIRECTIONS = Literal["R", "L", "S", "I", "A", "P"]
AX_CODES = tuple[DIRECTIONS, DIRECTIONS, DIRECTIONS]
ROTATION = np.ndarray
ZOOMS = tuple[float, float, float] | Sequence[float]
#####################
LABEL_MAX = 256

CENTROID_DICT = dict[int, tuple[float, float, float]]
TRIPLE = tuple[float, float, float] | Sequence[float]
COORDINATE = TRIPLE
POI_DICT = dict[int, dict[int, COORDINATE]]

AFFINE = np.ndarray
SHAPE = TRIPLE | tuple[int, int, int]
ORIGIN = TRIPLE


LABEL_MAP = dict[int | str, int | str] | dict[str, str] | dict[int, int]

LABEL_REFERENCE = int | Sequence[int] | None

if TYPE_CHECKING:
    from TPTBox import NII, POI
_plane_dict: dict[DIRECTIONS, str] = {
    "S": "ax",
    "I": "ax",
    "L": "sag",
    "R": "sag",
    "A": "cor",
    "P": "cor",
}
_same_direction: dict[DIRECTIONS, DIRECTIONS] = {
    "S": "I",
    "I": "S",
    "L": "R",
    "R": "L",
    "A": "P",
    "P": "A",
}


def never_called(args: NoReturn) -> NoReturn:  # noqa: ARG001
    raise NotImplementedError()


class Sentinel:
    pass


# get range of all ribs

VERTEBRA_INSTANCE_RIB_LABEL_OFFSET = 40 - 8  # 40 - 55
VERTEBRA_INSTANCE_IVD_LABEL_OFFSET = 100  # 101 - 125
VERTEBRA_INSTANCE_ENDPLATE_LABEL_OFFSET = 200  # 201 - 225

_vidx2name = None
_vname2idx = None

_register_lvl = {}


class Abstract_lvl(Enum):
    def __init_subclass__(cls, **kwargs):
        _register_lvl[str(cls.__name__)] = cls

    @classmethod
    def order_dict(cls) -> dict[int, int]:
        return {}  # Default integer order

    @classmethod
    def _get_name(cls, i: int, no_raise=True) -> str:
        try:
            return cls(i).name
        except ValueError:
            if not no_raise:
                raise
            return str(i)

    @classmethod
    def _get_id(cls, s: str | int, no_raise=True) -> int:
        if isinstance(s, int):
            return s
        try:
            return cls[s].value
        except KeyError:
            if not no_raise:
                raise
            return int(s)


class Vertebra_Instance(Abstract_lvl):
    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(
        self,
        vertebra_label: int,
        has_rib: bool = False,
        has_ivd: bool = True,
    ):
        self._rib = None
        self._ivd = None
        self._endplate = None
        if has_rib:
            self._rib = (
                vertebra_label + VERTEBRA_INSTANCE_RIB_LABEL_OFFSET if vertebra_label != 28 else 21 + VERTEBRA_INSTANCE_RIB_LABEL_OFFSET
            )
        if has_ivd:
            self._ivd = vertebra_label + VERTEBRA_INSTANCE_IVD_LABEL_OFFSET
            self._endplate = vertebra_label + VERTEBRA_INSTANCE_ENDPLATE_LABEL_OFFSET

    @classmethod
    def vertebra_label_without_sacrum(cls) -> list[int]:
        # TODO add sacrum label and similar flags into init (also C, T, L)
        l = list(range(1, 26))
        l.append(28)
        return l

    @classmethod
    def rib_label(cls) -> list[int]:
        return [i._rib for i in Vertebra_Instance if i._rib is not None]

    @classmethod
    def endplate_label(cls) -> list[int]:
        return [i._endplate for i in Vertebra_Instance if i._endplate is not None]

    # TODO maybe easier to have a nested enum where the inner enum stands for vertebra, rib, ivd, ... makes naming independent of hard-coded attribute names
    @classmethod
    def name2idx(cls) -> dict[str, int]:
        global _vname2idx  # noqa: PLW0603
        if _vname2idx is not None:
            return _vname2idx
        name2idx = {}
        for instance in Vertebra_Instance:
            name2idx[instance.name] = instance.value
            for structure in ["rib", "ivd", "endplate"]:
                attrname = f"_{structure}"
                assert hasattr(instance, attrname)
                attr = getattr(instance, attrname)
                if attr is not None:
                    name2idx[instance.name + attrname] = attr
        _vname2idx = name2idx
        return name2idx

    @classmethod
    def idx2name(cls) -> dict[int, str]:
        global _vidx2name  # noqa: PLW0603
        if _vidx2name is not None:
            return _vidx2name
        _vidx2name = {value: key for key, value in Vertebra_Instance.name2idx().items()}
        return _vidx2name

    @classmethod
    def is_sacrum(cls, i: int):
        try:
            return cls(i) in cls.sacrum()
        except KeyError:
            return False

    @classmethod
    def cervical(cls):
        return (cls.C1, cls.C2, cls.C3, cls.C4, cls.C5, cls.C6, cls.C7)

    @classmethod
    def thoracic(cls):
        return (cls.T1, cls.T2, cls.T3, cls.T4, cls.T5, cls.T6, cls.T7, cls.T8, cls.T9, cls.T10, cls.T11, cls.T12, cls.T13)

    @classmethod
    def lumbar(cls):
        return (cls.L1, cls.L2, cls.L3, cls.L4, cls.L5, cls.L6)

    @classmethod
    def sacrum(cls):
        return (cls.S1, cls.S2, cls.S3, cls.S4, cls.S5, cls.S6, cls.COCC)

    @classmethod
    def order(cls):
        return cls.cervical() + cls.thoracic() + cls.lumbar() + cls.sacrum()

    @classmethod
    def order_dict(cls) -> dict[int, int]:
        return {a.value: e for e, a in enumerate(cls.order())}

    def get_next_poi(self, poi: typing.Union["POI", "NII", list[int]]):
        r = poi if isinstance(poi, list) else poi.keys_region() if hasattr(poi, "keys_region") else poi.unique()  # type: ignore
        o = self.order()
        idx = o.index(self)
        for vert in o[idx + 1 :]:
            if vert.value in r:
                return vert
        return None

    def get_previous_poi(self, poi: typing.Union["POI", "NII", list[int]]):
        r = poi if isinstance(poi, list) else poi.keys_region() if hasattr(poi, "keys_region") else poi.unique()  # type: ignore
        o = self.order()
        idx = o.index(self)
        for vert in reversed(o[:idx]):
            if vert.value in r:
                return vert
        return None

    C1 = 1
    C2 = 2
    C3 = 3
    C4 = 4
    C5 = 5
    C6 = 6
    C7 = 7, True, True
    T1 = 8, True, True
    T2 = 9, True, True
    T3 = 10, True, True
    T4 = 11, True, True
    T5 = 12, True, True
    T6 = 13, True, True
    T7 = 14, True, True
    T8 = 15, True, True
    T9 = 16, True, True
    T10 = 17, True, True
    T11 = 18, True, True
    T12 = 19, True, True
    T13 = 28, True, True
    L1 = 20, True, True
    L2 = 21
    L3 = 22
    L4 = 23
    L5 = 24
    L6 = 25
    S1 = 26
    COCC = 27
    S2 = 29
    S3 = 30
    S4 = 31
    S5 = 32
    S6 = 33

    @property
    def VERTEBRA(self) -> int:
        return self.value

    @property
    def RIB(self) -> int:
        assert self._rib is not None, (self.name, self.value)
        return self._rib

    @classmethod
    def rib2vert(cls, riblabel: int) -> int:
        assert riblabel in Vertebra_Instance.rib_label(), riblabel
        return riblabel - VERTEBRA_INSTANCE_RIB_LABEL_OFFSET if riblabel != 21 + VERTEBRA_INSTANCE_RIB_LABEL_OFFSET else 28

    @property
    def IVD(self) -> int:
        assert self._ivd is not None, (self.name, self.value)
        return self._ivd

    @property
    def ENDPLATE(self) -> int:
        assert self._endplate is not None, (self.name, self.value)
        return self._endplate

    def __str__(self):
        return str(self.name)


class Location(Abstract_lvl):
    Unknown = 0
    # S1 = 26  # SACRUM
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
    # Articulate_Process_Facet_Joint (Used anywhere?)
    # Superior_Articulate_Process_Facet_Joint_Surface_Left = 54
    # Superior_Articulate_Process_Facet_Joint_Surface_Right = 55
    # Inferior_Articulate_Process_Facet_Joint_Surface_Left = 56
    # Inferior_Articulate_Process_Facet_Joint_Surface_Right = 57
    Vertebra_Disc_Superior = 58
    Vertebra_Disc_Inferior = 59
    Vertebra_Disc = 100
    Spinal_Cord = 60
    Spinal_Canal = 61  # Center of vertebra
    Spinal_Canal_ivd_lvl = 126
    Endplate = 62
    Rib_Left = 63
    Rib_Right = 64
    # 66-80 Free
    # Muscle inserts
    # https://www.frontiersin.org/articles/10.3389/fbioe.2022.862804/full
    # 81-91
    Muscle_Inserts_Spinosus_Process = 81
    Muscle_Inserts_Transverse_Process_Left = 83
    Muscle_Inserts_Transverse_Process_Right = 82
    Muscle_Inserts_Vertebral_Body_Left = 84
    Muscle_Inserts_Vertebral_Body_Right = 85
    Muscle_Inserts_Articulate_Process_Inferior_Left = 86
    Muscle_Inserts_Articulate_Process_Inferior_Right = 87
    Muscle_Inserts_Articulate_Process_Superior_Left = 88
    Muscle_Inserts_Articulate_Process_Superior_Right = 89
    # Implants (No automatic generation)
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
    Additional_Vertebral_Body_Middle_Superior_Median = 105
    Additional_Vertebral_Body_Posterior_Central_Median = 106
    Additional_Vertebral_Body_Middle_Inferior_Median = 107
    Additional_Vertebral_Body_Anterior_Central_Median = 108
    # 109-116 Same but shifted to the left
    Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left = 109
    Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Left = 110
    Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Left = 111
    Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Left = 112
    Additional_Vertebral_Body_Middle_Superior_Left = 113
    Additional_Vertebral_Body_Posterior_Central_Left = 114
    Additional_Vertebral_Body_Middle_Inferior_Left = 115
    Additional_Vertebral_Body_Anterior_Central_Left = 116
    # 117-124 Same but shifted to the right
    Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right = 117
    Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Right = 118
    Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Right = 119
    Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Right = 120
    Additional_Vertebral_Body_Middle_Superior_Right = 121
    Additional_Vertebral_Body_Posterior_Central_Right = 122
    Additional_Vertebral_Body_Middle_Inferior_Right = 123
    Additional_Vertebral_Body_Anterior_Central_Right = 124

    Ligament_Attachment_Point_Flava_Superior_Median = 125
    Ligament_Attachment_Point_Flava_Inferior_Median = 127
    # Vertebra orientation (compute Vertebra_Direction - Vertebra_Corpus)
    Vertebra_Direction_Posterior = 128
    Vertebra_Direction_Inferior = 129
    Vertebra_Direction_Right = 130

    # Ligament_Attachment_Point_Flava_Superior_Right = 141
    # Ligament_Attachment_Point_Flava_Inferior_Right = 143
    # Ligament_Attachment_Point_Flava_Superior_Left = 149
    # Ligament_Attachment_Point_Flava_Inferior_Left = 151

    # Ligament_Attachment_Point_Interspinosa_Superior_Left = 133
    # Ligament_Attachment_Point_Interspinosa_Superior_Right = 134
    # Ligament_Attachment_Point_Interspinosa_Inferior_Left = 135
    # Ligament_Attachment_Point_Interspinosa_Inferior_Right = 136

    # Multi = 256

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
vert_directions = [
    Location.Vertebra_Direction_Inferior,
    Location.Vertebra_Direction_Right,
    Location.Vertebra_Direction_Posterior,
]

conversion_poi2text = {k: v for v, k in conversion_poi.items()}
