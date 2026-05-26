"""Vertebra constants, enumerations, and label-mapping utilities for TPTBox.

Vertebra numbering convention
------------------------------
Integer label IDs follow the TPTBox spine segmentation scheme:

- **Cervical** (C1–C7):   labels 1–7
- **Thoracic** (T1–T12):  labels 8–19;  T13 is label 28
- **Lumbar**   (L1–L6):   labels 20–25
- **Sacrum**   (S1):      label 26;  S2–S6: labels 29–33;  Coccyx: label 27

Associated structures use fixed offsets from the vertebra label:

- **Rib**:      label + 32  (``VERTEBRA_INSTANCE_RIB_LABEL_OFFSET = 40 - 8``)
- **IVD**:      label + 100 (``VERTEBRA_INSTANCE_IVD_LABEL_OFFSET``)
- **Endplate**: label + 200 (``VERTEBRA_INSTANCE_ENDPLATE_LABEL_OFFSET``)

Key exports
-----------
- :class:`Vertebra_Instance` — IntEnum of all vertebra instances with rib/IVD helpers.
- :class:`Location`          — IntEnum of vertebra subregion labels (POI types).
- :data:`v_name2idx`         — ``dict[str, int]`` mapping name (e.g. ``"L1"``) → label.
- :data:`v_idx2name`         — ``dict[int, str]`` mapping label → name.
- :data:`v_idx_order`        — List of label IDs in anatomical cranio-caudal order.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Literal, NoReturn, Union

import numpy as np

from TPTBox.logger import log_file

ROUNDING_LVL = 7
#####################
log = log_file.Reflection_Logger()
logging = Union[bool, log_file.Logger_Interface]
# R: Right, L: Left; S: Superior (up), I: Inferior (down); A: Anterior (front), P: Posterior (back)
DIRECTIONS = Literal["R", "L", "S", "I", "A", "P"]
AX_CODES = tuple[DIRECTIONS, DIRECTIONS, DIRECTIONS]
ROTATION = np.ndarray
ZOOMS = Union[tuple[float, float, float], Sequence[float]]
#####################
LABEL_MAX = 256

CENTROID_DICT = dict[int, tuple[float, float, float]]
TRIPLE = Union[tuple[float, float, float], Sequence[float]]
COORDINATE = TRIPLE
POI_DICT = dict[int, dict[int, COORDINATE]]

AFFINE = np.ndarray
SHAPE = Union[TRIPLE, tuple[int, int, int]]
ORIGIN = TRIPLE


LABEL_MAP = Union[dict[Union[int, str], Union[int, str]], dict[str, str], dict[int, int]]

LABEL_REFERENCE = Union[int, Sequence[int], None]

if TYPE_CHECKING:
    from TPTBox import NII, POI
_supported_img_files = ["nii.gz", "nii", "nrrd", "mha"]
MODES = Literal["constant", "nearest", "reflect", "wrap"]
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
    """Raise ``NotImplementedError``; used to mark branches that should be unreachable."""
    raise NotImplementedError()


class Sentinel:
    """Unique sentinel object used as a default argument to distinguish *not provided* from ``None``."""


# get range of all ribs

VERTEBRA_INSTANCE_RIB_LABEL_OFFSET = 40 - 8  # 40 - 55
VERTEBRA_INSTANCE_IVD_LABEL_OFFSET = 100  # 101 - 125
VERTEBRA_INSTANCE_ENDPLATE_LABEL_OFFSET = 200  # 201 - 225

_vidx2name = None
_vname2idx = None

_register_lvl = {}


class Abstract_lvl(Enum):
    """Base class for all TPTBox label-level enumerations.

    Subclasses are automatically registered in the global ``_register_lvl`` dict
    so that cross-enum name resolution works.  Override :meth:`save_as_name` to
    control whether values are serialised as human-readable names or raw integers.
    """

    def __init_subclass__(cls, **kwargs):
        _register_lvl[str(cls.__name__)] = cls

    @classmethod
    def save_as_name(cls) -> bool:
        """Return True if enum members should be saved using their name rather than their integer value."""
        return True

    @classmethod
    def order_dict(cls) -> dict[int, int]:
        """Return a mapping from enum value to sort position; empty dict means natural integer order."""
        return {}  # Default integer order

    @classmethod
    def _get_name(cls, i: int, no_raise=True) -> str:
        """Resolve an integer label to its enum name (or raw string if not found)."""
        if cls.save_as_name():
            try:
                return cls(i).name
            except ValueError:
                if not no_raise:
                    raise
        return str(i)

    @classmethod
    def _get_id(cls, s: str | int, no_raise=True) -> int:
        """Resolve a name or integer to its integer enum value."""
        if isinstance(s, int):
            return s
        try:
            return cls[s].value
        except KeyError:
            for c in cls:
                if c.name.lower() == s.lower():
                    return c.value
            if not no_raise:
                raise
            return int(s)


class Any(Abstract_lvl):
    """Wildcard label-level that resolves names by searching all registered enums.

    Unlike :class:`Abstract_lvl`, ``Any`` does not save member names — values are
    always serialised as plain integers.  Name resolution for :meth:`_get_id` tries
    every registered enum class until a match is found.
    """

    def __init_subclass__(cls, **kwargs):
        _register_lvl[str(cls.__name__)] = cls

    @classmethod
    def save_as_name(cls) -> bool:
        """Always returns False; values are serialised as integers."""
        return False

    @classmethod
    def _get_name(cls, i: int, no_raise=True) -> str:  # noqa: ARG003
        """Return the string representation of integer ``i`` without enum lookup."""
        return str(i)

    @classmethod
    def _get_id(cls, s: str | int, no_raise=True) -> int:  # noqa: ARG003
        """Resolve ``s`` by searching all registered enum classes; falls back to ``int(s)``."""
        self_name = str(cls.__name__)
        for n, cl in _register_lvl.items():
            if n == self_name:
                continue
            try:
                s = cl._get_id(cls, s, no_raise=False)
                return s  # type: ignore # noqa: TRY300
            except Exception:
                pass
        return int(s)


class Full_Body_Instance_Vibe(Abstract_lvl):
    """Enum of whole-body anatomy labels in the VIBESeg segmentation scheme.

    Similar to :class:`Full_Body_Instance` but uses a different numbering that
    matches the VIBESeg model output.  Provides a class method
    :meth:`get_Full_Body_Instance_mapping` to convert from
    :class:`Full_Body_Instance` labels.
    """

    spleen = 1
    kidney_right = 2
    kidney_left = 3
    gallbladder = 4
    liver = 5
    stomach = 6
    pancreas = 7
    adrenal_gland_right = 8
    adrenal_gland_left = 9
    lung_upper_lobe_left = 10
    lung_lower_lobe_left = 11
    lung_upper_lobe_right = 12
    lung_middle_lobe_right = 13
    lung_lower_lobe_right = 14
    esophagus = 15
    trachea = 16
    thyroid_gland = 17
    intestine = 18
    duodenum = 19
    unused = 20
    urinary_bladder = 21
    prostate = 22
    sacrum = 23
    heart = 24
    aorta = 25
    pulmonary_vein = 26
    brachiocephalic_trunk = 27
    subclavian_artery_right = 28
    subclavian_artery_left = 29
    common_carotid_artery_right = 30
    common_carotid_artery_left = 31
    brachiocephalic_vein_left = 32
    brachiocephalic_vein_right = 33
    atrial_appendage_left = 34
    superior_vena_cava = 35
    inferior_vena_cava = 36
    portal_vein_and_splenic_vein = 37
    iliac_artery_left = 38
    iliac_artery_right = 39
    iliac_vena_left = 40
    iliac_vena_right = 41
    humerus_left = 42
    humerus_right = 43
    scapula_left = 44
    scapula_right = 45
    clavicula_left = 46
    clavicula_right = 47
    femur_left = 48
    femur_right = 49
    pelvis_left = 50
    pelvis_right = 51
    spinal_cord = 52
    gluteus_maximus_left = 53
    gluteus_maximus_right = 54
    gluteus_medius_left = 55
    gluteus_medius_right = 56
    gluteus_minimus_left = 57
    gluteus_minimus_right = 58
    autochthon_left = 59
    autochthon_right = 60
    iliopsoas_left = 61
    iliopsoas_right = 62
    sternum = 63
    costal_cartilages = 64
    subcutaneous_fat = 65
    muscle = 66
    inner_fat = 67
    IVD = 68
    vertebra_body = 69
    vertebra_posterior_elements = 70
    spinal_channel = 71
    bone_other = 72

    @classmethod
    def get_Full_Body_Instance_mapping(cls) -> dict[int, int]:
        """Return a mapping from ``Full_Body_Instance`` label values to ``Full_Body_Instance_Vibe`` label values."""
        return {
            Full_Body_Instance.spleen.value: cls.spleen.value,  # spleen
            Full_Body_Instance.kidney_right.value: cls.kidney_right.value,  # kidney_right
            Full_Body_Instance.kidney_left.value: cls.kidney_left.value,  # kidney_left
            Full_Body_Instance.gallbladder.value: cls.gallbladder.value,  # gallbladder
            Full_Body_Instance.liver.value: cls.liver.value,  # liver
            Full_Body_Instance.stomach.value: cls.stomach.value,  # stomach
            Full_Body_Instance.pancreas.value: cls.pancreas.value,  # pancreas
            Full_Body_Instance.adrenal_gland_right.value: cls.adrenal_gland_right.value,  # adrenal_gland_right
            Full_Body_Instance.adrenal_gland_left.value: cls.adrenal_gland_left.value,  # adrenal_gland_left
            Full_Body_Instance.lung_left.value: cls.lung_upper_lobe_left.value,  # lung_upper_lobe_left
            Full_Body_Instance.lung_left.value: cls.lung_lower_lobe_left.value,  # lung_lower_lobe_left
            Full_Body_Instance.lung_right.value: cls.lung_upper_lobe_right.value,  # lung_upper_lobe_right
            Full_Body_Instance.lung_right.value: cls.lung_middle_lobe_right.value,  # lung_middle_lobe_right
            Full_Body_Instance.lung_right.value: cls.lung_lower_lobe_right.value,  # lung_lower_lobe_right
            Full_Body_Instance.esophagus.value: cls.esophagus.value,  # esophagus
            Full_Body_Instance.trachea.value: cls.trachea.value,  # trachea
            Full_Body_Instance.thyroid_gland_right.value: cls.thyroid_gland.value,  # thyroid_gland
            Full_Body_Instance.intestine.value: cls.intestine.value,  # intestine
            Full_Body_Instance.doudenum.value: cls.duodenum.value,  # duodenum
            Full_Body_Instance.rib_right.value: cls.unused.value,  # unused
            Full_Body_Instance.urinary_bladder.value: cls.urinary_bladder.value,  # urinary_bladder
            Full_Body_Instance.prostate.value: cls.prostate.value,  # prostate
            Full_Body_Instance.sacrum.value: cls.sacrum.value,  # sacrum
            Full_Body_Instance.heart.value: cls.heart.value,  # heart
            Full_Body_Instance.aorta.value: cls.aorta.value,  # aorta
            Full_Body_Instance.pulmonary_vein.value: cls.pulmonary_vein.value,  # pulmonary_vein
            Full_Body_Instance.brachiocephalic_trunk.value: cls.brachiocephalic_trunk.value,  # brachiocephalic_trunk
            Full_Body_Instance.subclavian_artery_right.value: cls.subclavian_artery_right.value,  # subclavian_artery_right
            Full_Body_Instance.subclavian_artery_left.value: cls.subclavian_artery_left.value,  # subclavian_artery_left
            Full_Body_Instance.common_carotid_artery_right.value: cls.common_carotid_artery_right.value,  # common_carotid_artery_right
            Full_Body_Instance.common_carotid_artery_left.value: cls.common_carotid_artery_left.value,  # common_carotid_artery_left
            Full_Body_Instance.brachiocephalic_vein_left.value: cls.brachiocephalic_vein_left.value,  # brachiocephalic_vein_left
            Full_Body_Instance.brachiocephalic_vein_right.value: cls.brachiocephalic_vein_right.value,  # brachiocephalic_vein_right
            Full_Body_Instance.atrial_appendage_left.value: cls.atrial_appendage_left.value,  # atrial_appendage_left
            Full_Body_Instance.superior_vena_cava.value: cls.superior_vena_cava.value,  # superior_vena_cava
            Full_Body_Instance.inferior_vena_cava.value: cls.inferior_vena_cava.value,  # inferior_vena_cava
            Full_Body_Instance.portal_vein_and_splenic_vein.value: cls.portal_vein_and_splenic_vein.value,  # portal_vein_and_splenic_vein
            Full_Body_Instance.iliac_artery_left.value: cls.iliac_artery_left.value,  # iliac_artery_left
            Full_Body_Instance.iliac_artery_right.value: cls.iliac_artery_right.value,  # iliac_artery_right
            Full_Body_Instance.iliac_vena_left.value: cls.iliac_vena_left.value,  # iliac_vena_left
            Full_Body_Instance.iliac_vena_right.value: cls.iliac_vena_right.value,  # iliac_vena_right
            Full_Body_Instance.humerus_left.value: cls.humerus_left.value,  # humerus_left
            Full_Body_Instance.humerus_right.value: cls.humerus_right.value,  # humerus_right
            Full_Body_Instance.scapula_left.value: cls.scapula_left.value,  # scapula_left
            Full_Body_Instance.scapula_right.value: cls.scapula_right.value,  # scapula_right
            Full_Body_Instance.clavicula_left.value: cls.clavicula_left.value,  # clavicula_left
            Full_Body_Instance.clavicula_right.value: cls.clavicula_right.value,  # clavicula_right
            Full_Body_Instance.femur_left.value: cls.femur_left.value,  # femur_left
            Full_Body_Instance.femur_right.value: cls.femur_right.value,  # femur_right
            Full_Body_Instance.pelvis_left.value: cls.hip_left.value,  # hip_left
            Full_Body_Instance.pelvis_right.value: cls.hip_right.value,  # hip_right
            Full_Body_Instance.channel.value: cls.spinal_cord.value,  # spinal_cord
            Full_Body_Instance.gluteus_maximus_left.value: cls.gluteus_maximus_left.value,  # gluteus_maximus_left
            Full_Body_Instance.gluteus_maximus_right.value: cls.gluteus_maximus_right.value,  # gluteus_maximus_right
            Full_Body_Instance.gluteus_medius_left.value: cls.gluteus_medius_left.value,  #  gluteus_medius_left
            Full_Body_Instance.gluteus_medius_right.value: cls.gluteus_medius_right.value,  #  gluteus_medius_right
            Full_Body_Instance.gluteus_minimus_left.value: cls.gluteus_minimus_left.value,  # gluteus_minimus_left
            Full_Body_Instance.gluteus_minimus_right.value: cls.gluteus_minimus_right.value,  # gluteus_minimus_right
            Full_Body_Instance.autochthon_left.value: cls.autochthon_left.value,  # autochthon_left
            Full_Body_Instance.autochthon_right.value: cls.autochthon_right.value,  # autochthon_right
            Full_Body_Instance.iliopsoas_left.value: cls.iliopsoas_left.value,  # iliopsoas_left
            Full_Body_Instance.iliopsoas_right.value: cls.iliopsoas_right.value,  # iliopsoas_right
            Full_Body_Instance.sternum.value: cls.sternum.value,  # sternum
            Full_Body_Instance.costal_cartilage.value: cls.costal_cartilages.value,  # costal_cartilages
            Full_Body_Instance.subcutaneous_fat.value: cls.subcutaneous_fat.value,  # subcutaneous_fat
            Full_Body_Instance.muscle_other.value: cls.muscle.value,  # muscle
            Full_Body_Instance.inner_fat.value: cls.inner_fat.value,  # inner_fat
            Full_Body_Instance.ivd.value: cls.IVD.value,  # IVD
            Full_Body_Instance.vert_body.value: cls.vertebra_body.value,  # vertebra_body
            Full_Body_Instance.vert_post.value: cls.vertebra_posterior_elements.value,  # vertebra_posterior_elements
            Full_Body_Instance.channel.value: cls.spinal_channel.value,  # spinal_channel
            Full_Body_Instance.ignore.value: cls.bone_other.value,  # bone_other
            100: Full_Body_Instance.ignore.value,
        }


class Full_Body_Instance(Abstract_lvl):
    """Enum of full-body anatomy instance labels used in whole-body segmentation.

    Each member represents a distinct anatomical structure identified in a
    whole-body scan.  Left/right pairs use an offset of 100 (right has the lower
    value, left adds 100 where applicable).
    """

    skull = 1
    clavicula_right = 2
    clavicula_left = 102
    scapula_right = 3
    scapula_left = 103
    humerus_right = 4
    humerus_left = 104
    hand_right = 5
    hand_left = 105
    radius_right = 60
    radius_left = 160
    ulna_right = 61
    ulna_left = 161
    sternum = 6
    costal_cartilage = 7
    rib_right = 8
    rib_left = 108
    vert_body = 9
    vert_post = 10
    sacrum = 11
    pelvis_right = 12
    pelvis_left = 112
    femur_right = 13
    femur_left = 113
    patella_right = 14
    patella_left = 114
    tibia_right = 15
    tibia_left = 115
    fibula_right = 16
    fibula_left = 116
    talus_right = 17
    talus_left = 117
    calcaneus_right = 18
    calcaneus_left = 118
    tarsals_right = 19
    tarsals_left = 119
    metatarsals_right = 20
    metatarsals_left = 120
    phalanges_right = 21
    phalanges_left = 121
    trachea = 22
    lung_right = 23
    lung_left = 123
    heart = 24
    spleen = 25
    kidney_right = 26
    kidney_left = 126
    liver = 27
    gallbladder = 28
    ivd = 29
    stomach = 30
    pancreas = 31
    adrenal_gland_right = 32
    adrenal_gland_left = 132
    esophagus = 33
    thyroid_gland_right = 34
    thyroid_gland_left = 134
    doudenum = 35
    intestine = 36
    urinary_bladder = 37
    prostate = 38
    channel = 39
    aorta = 40
    pulmonary_vein = 41
    brachiocephalic_trunk = 42
    subclavian_artery_right = 43
    subclavian_artery_left = 143
    common_carotid_artery_right = 44
    common_carotid_artery_left = 144
    brachiocephalic_vein_right = 45
    brachiocephalic_vein_left = 145
    atrial_appendage_left = 46
    superior_vena_cava = 47
    inferior_vena_cava = 48
    iliac_artery_right = 49
    iliac_artery_left = 149
    portal_vein_and_splenic_vein = 50
    iliac_vena_right = 51
    iliac_vena_left = 151
    gluteus_maximus_right = 52
    gluteus_maximus_left = 152
    gluteus_medius_right = 53
    gluteus_medius_left = 153
    gluteus_minimus_right = 54
    gluteus_minimus_left = 154
    autochthon_right = 55
    autochthon_left = 155
    iliopsoas_right = 56
    iliopsoas_left = 156
    subcutaneous_fat = 57
    muscle_other = 58
    inner_fat = 59
    ignore = 63

    @classmethod
    def bone(cls) -> list[Full_Body_Instance]:
        """Return all skeletal bone instance labels."""
        return [
            Full_Body_Instance.skull,
            Full_Body_Instance.clavicula_right,
            Full_Body_Instance.clavicula_left,
            Full_Body_Instance.scapula_right,
            Full_Body_Instance.scapula_left,
            Full_Body_Instance.humerus_right,
            Full_Body_Instance.humerus_left,
            Full_Body_Instance.hand_right,
            Full_Body_Instance.hand_left,
            Full_Body_Instance.radius_right,
            Full_Body_Instance.radius_left,
            Full_Body_Instance.ulna_right,
            Full_Body_Instance.ulna_left,
            Full_Body_Instance.sternum,
            Full_Body_Instance.costal_cartilage,
            Full_Body_Instance.rib_right,
            Full_Body_Instance.rib_left,
            Full_Body_Instance.vert_body,
            Full_Body_Instance.vert_post,
            Full_Body_Instance.sacrum,
            Full_Body_Instance.pelvis_right,
            Full_Body_Instance.pelvis_left,
            Full_Body_Instance.femur_right,
            Full_Body_Instance.femur_left,
            Full_Body_Instance.patella_right,
            Full_Body_Instance.patella_left,
            Full_Body_Instance.tibia_right,
            Full_Body_Instance.tibia_left,
            Full_Body_Instance.fibula_right,
            Full_Body_Instance.fibula_left,
            Full_Body_Instance.talus_right,
            Full_Body_Instance.talus_left,
            Full_Body_Instance.calcaneus_right,
            Full_Body_Instance.calcaneus_left,
            Full_Body_Instance.tarsals_right,
            Full_Body_Instance.tarsals_left,
            Full_Body_Instance.metatarsals_right,
            Full_Body_Instance.metatarsals_left,
            Full_Body_Instance.phalanges_right,
            Full_Body_Instance.phalanges_left,
        ]

    @classmethod
    def lung_system(cls) -> list[Full_Body_Instance]:
        """Return trachea and lung instance labels."""
        return [
            Full_Body_Instance.trachea,
            Full_Body_Instance.lung_right,
            Full_Body_Instance.lung_left,
        ]

    @classmethod
    def organs(cls) -> list[Full_Body_Instance]:
        """Return solid abdominal and thoracic organ instance labels."""
        return [
            Full_Body_Instance.heart,
            Full_Body_Instance.kidney_right,
            Full_Body_Instance.kidney_left,
            Full_Body_Instance.liver,
            Full_Body_Instance.gallbladder,
            Full_Body_Instance.adrenal_gland_right,
            Full_Body_Instance.adrenal_gland_left,
            Full_Body_Instance.thyroid_gland_right,
            Full_Body_Instance.thyroid_gland_left,
            Full_Body_Instance.urinary_bladder,
            Full_Body_Instance.prostate,
        ]

    @classmethod
    def digestion(cls) -> list[Full_Body_Instance]:
        """Return gastrointestinal organ instance labels."""
        return [
            Full_Body_Instance.stomach,
            Full_Body_Instance.pancreas,
            Full_Body_Instance.esophagus,
            Full_Body_Instance.doudenum,
            Full_Body_Instance.intestine,
        ]

    @classmethod
    def vessels(cls) -> list[Full_Body_Instance]:
        """Return vascular structure instance labels."""
        return [
            Full_Body_Instance.aorta,
            Full_Body_Instance.pulmonary_vein,
            Full_Body_Instance.brachiocephalic_trunk,
            Full_Body_Instance.subclavian_artery_right,
            Full_Body_Instance.subclavian_artery_left,
            Full_Body_Instance.common_carotid_artery_right,
            Full_Body_Instance.common_carotid_artery_left,
            Full_Body_Instance.brachiocephalic_vein_right,
            Full_Body_Instance.brachiocephalic_vein_left,
            Full_Body_Instance.atrial_appendage_left,
            Full_Body_Instance.superior_vena_cava,
            Full_Body_Instance.inferior_vena_cava,
            Full_Body_Instance.iliac_artery_right,
            Full_Body_Instance.iliac_artery_left,
            Full_Body_Instance.portal_vein_and_splenic_vein,
            Full_Body_Instance.iliac_vena_right,
            Full_Body_Instance.iliac_vena_left,
        ]

    @classmethod
    def full_spine(cls) -> list[Full_Body_Instance]:
        """Return spinal structure instance labels (channel, IVD, vertebra, sacrum)."""
        return [
            Full_Body_Instance.channel,
            Full_Body_Instance.ivd,
            Full_Body_Instance.vert_body,
            Full_Body_Instance.vert_post,
            Full_Body_Instance.sacrum,
        ]

    @classmethod
    def muscle(cls) -> list[Full_Body_Instance]:
        """Return individually segmented muscle group instance labels."""
        return [
            Full_Body_Instance.gluteus_maximus_right,
            Full_Body_Instance.gluteus_maximus_left,
            Full_Body_Instance.gluteus_medius_right,
            Full_Body_Instance.gluteus_medius_left,
            Full_Body_Instance.gluteus_minimus_right,
            Full_Body_Instance.gluteus_minimus_left,
            Full_Body_Instance.autochthon_right,
            Full_Body_Instance.autochthon_left,
            Full_Body_Instance.iliopsoas_right,
            Full_Body_Instance.iliopsoas_left,
        ]

    @classmethod
    def body_comp(cls) -> list[Full_Body_Instance]:
        """Return body composition instance labels (fat, generic muscle, and named muscles)."""
        return [
            Full_Body_Instance.subcutaneous_fat,
            Full_Body_Instance.muscle_other,
            Full_Body_Instance.inner_fat,
            *Full_Body_Instance.muscle(),
        ]

    @classmethod
    def get_VIBESeg_mapping(cls) -> dict[int, int]:
        """Return a mapping from VIBESeg label indices to ``Full_Body_Instance`` label values."""
        return {
            1: Full_Body_Instance.spleen.value,  # spleen
            2: Full_Body_Instance.kidney_right.value,  # kidney_right
            3: Full_Body_Instance.kidney_left.value,  # kidney_left
            4: Full_Body_Instance.gallbladder.value,  # gallbladder
            5: Full_Body_Instance.liver.value,  # liver
            6: Full_Body_Instance.stomach.value,  # stomach
            7: Full_Body_Instance.pancreas.value,  # pancreas
            8: Full_Body_Instance.adrenal_gland_right.value,  # adrenal_gland_right
            9: Full_Body_Instance.adrenal_gland_left.value,  # adrenal_gland_left
            10: Full_Body_Instance.lung_left.value,  # lung_upper_lobe_left
            11: Full_Body_Instance.lung_left.value,  # lung_lower_lobe_left
            12: Full_Body_Instance.lung_right.value,  # lung_upper_lobe_right
            13: Full_Body_Instance.lung_right.value,  # lung_middle_lobe_right
            14: Full_Body_Instance.lung_right.value,  # lung_lower_lobe_right
            15: Full_Body_Instance.esophagus.value,  # esophagus
            16: Full_Body_Instance.trachea.value,  # trachea
            17: Full_Body_Instance.thyroid_gland_right.value,  # thyroid_gland
            18: Full_Body_Instance.intestine.value,  # intestine
            19: Full_Body_Instance.doudenum.value,  # duodenum
            20: Full_Body_Instance.rib_right.value,  # unused
            21: Full_Body_Instance.urinary_bladder.value,  # urinary_bladder
            22: Full_Body_Instance.prostate.value,  # prostate
            23: Full_Body_Instance.sacrum.value,  # sacrum
            24: Full_Body_Instance.heart.value,  # heart
            25: Full_Body_Instance.aorta.value,  # aorta
            26: Full_Body_Instance.pulmonary_vein.value,  # pulmonary_vein
            27: Full_Body_Instance.brachiocephalic_trunk.value,  # brachiocephalic_trunk
            28: Full_Body_Instance.subclavian_artery_right.value,  # subclavian_artery_right
            29: Full_Body_Instance.subclavian_artery_left.value,  # subclavian_artery_left
            30: Full_Body_Instance.common_carotid_artery_right.value,  # common_carotid_artery_right
            31: Full_Body_Instance.common_carotid_artery_left.value,  # common_carotid_artery_left
            32: Full_Body_Instance.brachiocephalic_vein_left.value,  # brachiocephalic_vein_left
            33: Full_Body_Instance.brachiocephalic_vein_right.value,  # brachiocephalic_vein_right
            34: Full_Body_Instance.atrial_appendage_left.value,  # atrial_appendage_left
            35: Full_Body_Instance.superior_vena_cava.value,  # superior_vena_cava
            36: Full_Body_Instance.inferior_vena_cava.value,  # inferior_vena_cava
            37: Full_Body_Instance.portal_vein_and_splenic_vein.value,  # portal_vein_and_splenic_vein
            38: Full_Body_Instance.iliac_artery_left.value,  # iliac_artery_left
            39: Full_Body_Instance.iliac_artery_right.value,  # iliac_artery_right
            40: Full_Body_Instance.iliac_vena_left.value,  # iliac_vena_left
            41: Full_Body_Instance.iliac_vena_right.value,  # iliac_vena_right
            42: Full_Body_Instance.humerus_left.value,  # humerus_left
            43: Full_Body_Instance.humerus_right.value,  # humerus_right
            44: Full_Body_Instance.scapula_left.value,  # scapula_left
            45: Full_Body_Instance.scapula_right.value,  # scapula_right
            46: Full_Body_Instance.clavicula_left.value,  # clavicula_left
            47: Full_Body_Instance.clavicula_right.value,  # clavicula_right
            48: Full_Body_Instance.femur_left.value,  # femur_left
            49: Full_Body_Instance.femur_right.value,  # femur_right
            50: Full_Body_Instance.pelvis_left.value,  # hip_left
            51: Full_Body_Instance.pelvis_right.value,  # hip_right
            52: Full_Body_Instance.channel.value,  # spinal_cord
            53: Full_Body_Instance.gluteus_maximus_left.value,  # gluteus_maximus_left
            54: Full_Body_Instance.gluteus_maximus_right.value,  # gluteus_maximus_right
            55: Full_Body_Instance.gluteus_medius_left.value,  #  gluteus_medius_left
            56: Full_Body_Instance.gluteus_medius_right.value,  #  gluteus_medius_right
            57: Full_Body_Instance.gluteus_minimus_left.value,  # gluteus_minimus_left
            58: Full_Body_Instance.gluteus_minimus_right.value,  # gluteus_minimus_right
            59: Full_Body_Instance.autochthon_left.value,  # autochthon_left
            60: Full_Body_Instance.autochthon_right.value,  # autochthon_right
            61: Full_Body_Instance.iliopsoas_left.value,  # iliopsoas_left
            62: Full_Body_Instance.iliopsoas_right.value,  # iliopsoas_right
            63: Full_Body_Instance.sternum.value,  # sternum
            64: Full_Body_Instance.costal_cartilage.value,  # costal_cartilages
            65: Full_Body_Instance.subcutaneous_fat.value,  # subcutaneous_fat
            66: Full_Body_Instance.muscle_other.value,  # muscle
            67: Full_Body_Instance.inner_fat.value,  # inner_fat
            68: Full_Body_Instance.ivd.value,  # IVD
            69: Full_Body_Instance.vert_body.value,  # vertebra_body
            70: Full_Body_Instance.vert_post.value,  # vertebra_posterior_elements
            71: Full_Body_Instance.channel.value,  # spinal_channel
            72: Full_Body_Instance.ignore.value,  # bone_other
            73: 0,
            77: 0,  # Negative
            100: Full_Body_Instance.ignore.value,
        }

    @classmethod
    def get_to_VIBESeg(cls) -> dict[int, int]:
        """Return a mapping from ``Full_Body_Instance`` label values to VIBESeg label indices."""
        return {
            Full_Body_Instance.skull.value: 0,
            Full_Body_Instance.clavicula_right.value: 47,
            Full_Body_Instance.clavicula_left.value: 46,
            Full_Body_Instance.scapula_right.value: 45,
            Full_Body_Instance.scapula_left.value: 44,
            Full_Body_Instance.humerus_right.value: 43,
            Full_Body_Instance.humerus_left.value: 42,
            Full_Body_Instance.hand_right.value: 72,
            Full_Body_Instance.hand_left.value: 72,
            Full_Body_Instance.radius_right.value: 72,
            Full_Body_Instance.radius_left.value: 72,
            Full_Body_Instance.ulna_right.value: 72,
            Full_Body_Instance.ulna_left.value: 72,
            Full_Body_Instance.sternum.value: 63,
            Full_Body_Instance.costal_cartilage.value: 64,
            Full_Body_Instance.rib_right.value: 20,
            Full_Body_Instance.rib_left.value: 20,
            Full_Body_Instance.vert_body.value: 69,
            Full_Body_Instance.vert_post.value: 70,
            Full_Body_Instance.sacrum.value: 23,
            Full_Body_Instance.pelvis_right.value: 51,
            Full_Body_Instance.pelvis_left.value: 50,
            Full_Body_Instance.femur_right.value: 49,
            Full_Body_Instance.femur_left.value: 48,
            Full_Body_Instance.patella_right.value: 72,
            Full_Body_Instance.patella_left.value: 72,
            Full_Body_Instance.tibia_right.value: 72,
            Full_Body_Instance.tibia_left.value: 72,
            Full_Body_Instance.fibula_right.value: 72,
            Full_Body_Instance.fibula_left.value: 72,
            Full_Body_Instance.talus_right.value: 72,
            Full_Body_Instance.talus_left.value: 72,
            Full_Body_Instance.calcaneus_right.value: 72,
            Full_Body_Instance.calcaneus_left.value: 72,
            Full_Body_Instance.tarsals_right.value: 72,
            Full_Body_Instance.tarsals_left.value: 72,
            Full_Body_Instance.metatarsals_right.value: 72,
            Full_Body_Instance.metatarsals_left.value: 72,
            Full_Body_Instance.phalanges_right.value: 72,
            Full_Body_Instance.phalanges_left.value: 72,
            Full_Body_Instance.trachea.value: 16,
            Full_Body_Instance.lung_right.value: 910,
            Full_Body_Instance.lung_left.value: 910,
            Full_Body_Instance.heart.value: 24,
            Full_Body_Instance.spleen.value: 1,
            Full_Body_Instance.kidney_right.value: 2,
            Full_Body_Instance.kidney_left.value: 3,
            Full_Body_Instance.liver.value: 5,
            Full_Body_Instance.gallbladder.value: 4,
            Full_Body_Instance.ivd.value: 68,
            Full_Body_Instance.stomach.value: 6,
            Full_Body_Instance.pancreas.value: 7,
            Full_Body_Instance.adrenal_gland_right.value: 8,
            Full_Body_Instance.adrenal_gland_left.value: 9,
            Full_Body_Instance.esophagus.value: 15,
            Full_Body_Instance.thyroid_gland_right.value: 17,
            Full_Body_Instance.thyroid_gland_left.value: 17,
            Full_Body_Instance.doudenum.value: 19,
            Full_Body_Instance.intestine.value: 18,
            Full_Body_Instance.urinary_bladder.value: 21,
            Full_Body_Instance.prostate.value: 22,
            Full_Body_Instance.channel.value: 71,  # 52
            Full_Body_Instance.aorta.value: 25,
            Full_Body_Instance.pulmonary_vein.value: 26,
            Full_Body_Instance.brachiocephalic_trunk.value: 27,
            Full_Body_Instance.subclavian_artery_right.value: 28,
            Full_Body_Instance.subclavian_artery_left.value: 29,
            Full_Body_Instance.common_carotid_artery_right.value: 30,
            Full_Body_Instance.common_carotid_artery_left.value: 31,
            Full_Body_Instance.brachiocephalic_vein_right.value: 33,
            Full_Body_Instance.brachiocephalic_vein_left.value: 32,
            Full_Body_Instance.atrial_appendage_left.value: 34,
            Full_Body_Instance.superior_vena_cava.value: 35,
            Full_Body_Instance.inferior_vena_cava.value: 36,
            Full_Body_Instance.iliac_artery_right.value: 39,
            Full_Body_Instance.iliac_artery_left.value: 38,
            Full_Body_Instance.portal_vein_and_splenic_vein.value: 37,
            Full_Body_Instance.iliac_vena_right.value: 41,
            Full_Body_Instance.iliac_vena_left.value: 40,
            Full_Body_Instance.gluteus_maximus_right.value: 54,
            Full_Body_Instance.gluteus_maximus_left.value: 53,
            Full_Body_Instance.gluteus_medius_right.value: 56,
            Full_Body_Instance.gluteus_medius_left.value: 55,
            Full_Body_Instance.gluteus_minimus_right.value: 58,
            Full_Body_Instance.gluteus_minimus_left.value: 57,
            Full_Body_Instance.autochthon_right.value: 60,
            Full_Body_Instance.autochthon_left.value: 59,
            Full_Body_Instance.iliopsoas_right.value: 62,
            Full_Body_Instance.iliopsoas_left.value: 61,
            Full_Body_Instance.subcutaneous_fat.value: 65,
            Full_Body_Instance.muscle_other.value: 66,
            Full_Body_Instance.inner_fat.value: 67,
            Full_Body_Instance.ignore.value: 73,
        }


class Lower_Body(Abstract_lvl):
    """Points-of-interest labels for lower body structures (patella, femur, tibia, fibula).

    Members define anatomical landmark positions used in lower-extremity analysis.
    """

    # Patella
    PATELLA_PROXIMAL_POLE = 1
    PATELLA_DISTAL_POLE = 2
    PATELLA_MEDIAL_POLE = 3
    PATELLA_LATERAL_POLE = 4
    PATELLA_RIDGE_PROXIMAL_POLE = 5
    PATELLA_RIDGE_DISTAL_POLE = 6
    PATELLA_RIDGE_HIGH_POINT = 7

    # Trochlea ossis femoris
    TROCHLEAR_RIDGE_MEDIAL_POINT = 8
    TROCHLEAR_RIDGE_LATERAL_POINT = 9
    TROCHLEA_GROOVE_CENTRAL_POINT = 10

    # Femur
    PELVIS_CENTER = 11
    NECK_CENTER = 12
    TIP_OF_GREATER_TROCHANTER = 13
    LATERAL_CONDYLE_POSTERIOR = 14
    LATERAL_CONDYLE_POSTERIOR_CRANIAL = 15
    LATERAL_CONDYLE_DISTAL = 16
    MEDIAL_CONDYLE_DISTAL = 17
    NOTCH_POINT = 18
    # Femur, Tibia
    ANATOMICAL_AXIS_PROXIMAL = 19
    ANATOMICAL_AXIS_DISTAL = 20
    MEDIAL_CONDYLE_POSTERIOR = 21
    MEDIAL_CONDYLE_POSTERIOR_CRANIAL = 22

    # Tibia
    KNEE_CENTER = 23
    MEDIAL_INTERCONDYLAR_TUBERCLE = 24
    LATERAL_INTERCONDYLAR_TUBERCLE = 25
    MEDIAL_CONDYLE_ANTERIOR = 26
    LATERAL_CONDYLE_ANTERIOR = 27
    MEDIAL_CONDYLE_MEDIAL = 28
    LATERAL_CONDYLE_LATERAL = 29
    ANKLE_CENTER = 30
    MEDIAL_MALLEOLUS = 31
    TGPP = 99
    TTP = 98
    # Fibula
    LATERAL_MALLEOLUS = 32

    @classmethod
    def get_mapping(cls) -> dict[str, tuple]:
        """Return the abbreviation-to-enum mapping for lower body landmarks."""
        return _ABBREVIATION_TO_ENUM


_ABBREVIATION_TO_ENUM = {
    # Patella
    "PPP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_PROXIMAL_POLE),
    "PDP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_DISTAL_POLE),
    "PMP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_MEDIAL_POLE),
    "PLP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_LATERAL_POLE),
    "PRPP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_RIDGE_PROXIMAL_POLE),
    "PRDP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_RIDGE_DISTAL_POLE),
    "PRHP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_RIDGE_HIGH_POINT),
    # Femur
    "TRMP": (Full_Body_Instance.femur_right, Lower_Body.TROCHLEAR_RIDGE_MEDIAL_POINT),
    "TRLP": (Full_Body_Instance.femur_right, Lower_Body.TROCHLEAR_RIDGE_LATERAL_POINT),
    "TGCP": (Full_Body_Instance.femur_right, Lower_Body.TROCHLEA_GROOVE_CENTRAL_POINT),
    "FHC": (Full_Body_Instance.femur_right, Lower_Body.PELVIS_CENTER),
    "FNC": (Full_Body_Instance.femur_right, Lower_Body.NECK_CENTER),
    "TGT": (Full_Body_Instance.femur_right, Lower_Body.TIP_OF_GREATER_TROCHANTER),
    "FLCP": (Full_Body_Instance.femur_right, Lower_Body.LATERAL_CONDYLE_POSTERIOR),
    "FLCPC": (
        Full_Body_Instance.femur_right,
        Lower_Body.LATERAL_CONDYLE_POSTERIOR_CRANIAL,
    ),
    "FMCP": (Full_Body_Instance.femur_right, Lower_Body.MEDIAL_CONDYLE_POSTERIOR),
    "FMCPC": (
        Full_Body_Instance.femur_right,
        Lower_Body.MEDIAL_CONDYLE_POSTERIOR_CRANIAL,
    ),
    "FLCD": (Full_Body_Instance.femur_right, Lower_Body.LATERAL_CONDYLE_DISTAL),
    "FMCD": (Full_Body_Instance.femur_right, Lower_Body.MEDIAL_CONDYLE_DISTAL),
    "FNP": (Full_Body_Instance.femur_right, Lower_Body.NOTCH_POINT),
    "FAAP": (Full_Body_Instance.femur_right, Lower_Body.ANATOMICAL_AXIS_PROXIMAL),
    "FADP": (Full_Body_Instance.femur_right, Lower_Body.ANATOMICAL_AXIS_DISTAL),
    # Tibia
    "TKC": (Full_Body_Instance.tibia_right, Lower_Body.KNEE_CENTER),
    "TMIT": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_INTERCONDYLAR_TUBERCLE),
    "TLIT": (Full_Body_Instance.tibia_right, Lower_Body.LATERAL_INTERCONDYLAR_TUBERCLE),
    "TMCP": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_CONDYLE_POSTERIOR),
    "TLCP": (Full_Body_Instance.tibia_right, Lower_Body.LATERAL_CONDYLE_POSTERIOR),
    "TMCA": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_CONDYLE_ANTERIOR),
    "TLCA": (Full_Body_Instance.tibia_right, Lower_Body.LATERAL_CONDYLE_ANTERIOR),
    "TMCM": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_CONDYLE_MEDIAL),
    "TLCL": (Full_Body_Instance.tibia_right, Lower_Body.LATERAL_CONDYLE_LATERAL),
    "TAC": (Full_Body_Instance.tibia_right, Lower_Body.ANKLE_CENTER),
    "TMM": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_MALLEOLUS),
    "TAAP": (Full_Body_Instance.tibia_right, Lower_Body.ANATOMICAL_AXIS_PROXIMAL),
    "TADP": (Full_Body_Instance.tibia_right, Lower_Body.ANATOMICAL_AXIS_DISTAL),
    "TGPP": (Full_Body_Instance.tibia_right, Lower_Body.TGPP),
    "TTP": (Full_Body_Instance.tibia_right, Lower_Body.TTP),
    # Fibula
    "FLM": (Full_Body_Instance.fibula_right, Lower_Body.LATERAL_MALLEOLUS),
}


class Vertebra_Instance(Abstract_lvl):
    """Enum of individual vertebra instances with associated rib, IVD, and endplate labels.

    Each member represents one vertebra identified by its integer label (see module
    docstring for the full numbering scheme).  Associated structure labels are
    computed from fixed offsets and exposed as properties:

    - :attr:`VERTEBRA` — the raw vertebra label value.
    - :attr:`RIB`      — rib label (only for T1–T12 and C6–C7).
    - :attr:`IVD`      — intervertebral disc label (label + 100).
    - :attr:`ENDPLATE` — endplate label (label + 200).

    Class methods :meth:`cervical`, :meth:`thoracic`, :meth:`lumbar`, and
    :meth:`sacrum` return tuples of members for each spinal region.
    :meth:`order` provides the full cranio-caudal sequence.
    """

    def __new__(cls, *args):
        """Create a new Vertebra_Instance enum member, storing the vertebra label as its value."""
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
        self.has_rib = has_rib
        if has_rib:
            self._rib = (
                vertebra_label + VERTEBRA_INSTANCE_RIB_LABEL_OFFSET if vertebra_label != 28 else 21 + VERTEBRA_INSTANCE_RIB_LABEL_OFFSET
            )
            # 40 - 8 + 21 = 53 = rib for T13
            # 52 rib for L1
        if has_ivd:
            self._ivd = vertebra_label + VERTEBRA_INSTANCE_IVD_LABEL_OFFSET
            self._endplate = vertebra_label + VERTEBRA_INSTANCE_ENDPLATE_LABEL_OFFSET

    @classmethod
    def vertebra_label_without_sacrum(cls) -> list[int]:
        """Return the list of vertebra label integers excluding sacrum/coccyx (C1–L6 + T13)."""
        # TODO add sacrum label and similar flags into init (also C, T, L)
        l = list(range(1, 26))
        l.append(28)
        return l

    @classmethod
    def rib_label(cls) -> list[int]:
        """Return all rib label integers for vertebrae that have ribs."""
        return [i._rib for i in Vertebra_Instance if i._rib is not None]

    @classmethod
    def endplate_label(cls) -> list[int]:
        """Return all endplate label integers for vertebrae that have endplates."""
        return [i._endplate for i in Vertebra_Instance if i._endplate is not None]

    # TODO maybe easier to have a nested enum where the inner enum stands for vertebra, rib, ivd, ... makes naming independent of hard-coded attribute names
    @classmethod
    def name2idx(cls) -> dict[str, int]:
        """Return a cached mapping from vertebra/structure name to integer label.

        Names include plain vertebra names (e.g. ``"L1"``) as well as derived names
        for ribs, IVDs, and endplates (e.g. ``"L1_rib"``, ``"L1_ivd"``).
        """
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
        """Return a cached mapping from integer label to vertebra/structure name."""
        global _vidx2name  # noqa: PLW0603
        if _vidx2name is not None:
            return _vidx2name
        _vidx2name = {value: key for key, value in Vertebra_Instance.name2idx().items()}
        return _vidx2name

    @classmethod
    def is_sacrum(cls, i: int) -> bool:
        """Return True if integer label ``i`` corresponds to a sacral or coccyx vertebra."""
        try:
            return cls(i) in cls.sacrum()
        except KeyError:
            return False
        except ValueError:
            return False

    @classmethod
    def cervical(cls) -> tuple[Vertebra_Instance, ...]:
        """Return a tuple of all cervical vertebra members (C1–C7)."""
        return (cls.C1, cls.C2, cls.C3, cls.C4, cls.C5, cls.C6, cls.C7)

    @classmethod
    def thoracic(cls) -> tuple[Vertebra_Instance, ...]:
        """Return a tuple of all thoracic vertebra members (T1–T13)."""
        return (
            cls.T1,
            cls.T2,
            cls.T3,
            cls.T4,
            cls.T5,
            cls.T6,
            cls.T7,
            cls.T8,
            cls.T9,
            cls.T10,
            cls.T11,
            cls.T12,
            cls.T13,
        )

    @classmethod
    def lumbar(cls) -> tuple[Vertebra_Instance, ...]:
        """Return a tuple of all lumbar vertebra members (L1–L6)."""
        return (cls.L1, cls.L2, cls.L3, cls.L4, cls.L5, cls.L6)

    @classmethod
    def sacrum(cls) -> tuple[Vertebra_Instance, ...]:
        """Return a tuple of sacral and coccyx members (S1–S6, COCC)."""
        return (cls.S1, cls.S2, cls.S3, cls.S4, cls.S5, cls.S6, cls.COCC)

    @classmethod
    def order(cls) -> tuple[Vertebra_Instance, ...]:
        """Return all vertebra members in cranio-caudal anatomical order (C1 → COCC)."""
        return cls.cervical() + cls.thoracic() + cls.lumbar() + cls.sacrum()

    @classmethod
    def order_dict(cls) -> dict[int, int]:
        """Return a mapping from vertebra label value to its position in :meth:`order`."""
        return {a.value: e for e, a in enumerate(cls.order())}

    def get_next_poi(self, poi: POI | NII | list[int]) -> Vertebra_Instance | None:
        """Return the next vertebra (caudally) that is present in ``poi``.

        Args:
            poi: A POI container, NII segmentation, or list of label integers.
                The method checks which vertebra labels are present.

        Returns:
            The next :class:`Vertebra_Instance` below ``self`` that is present
            in ``poi``, or ``None`` if no such vertebra exists.
        """
        r = poi if isinstance(poi, list) else poi.keys_region() if hasattr(poi, "keys_region") else poi.unique()  # type: ignore
        o = self.order()
        idx = o.index(self)
        for vert in o[idx + 1 :]:
            if vert.value in r:
                return vert
        return None

    def get_previous_poi(self, poi: POI | NII | list[int]) -> Vertebra_Instance | None:
        """Return the previous vertebra (cranially) that is present in ``poi``.

        Args:
            poi: A POI container, NII segmentation, or list of label integers.
                The method checks which vertebra labels are present.

        Returns:
            The nearest :class:`Vertebra_Instance` above ``self`` that is present
            in ``poi``, or ``None`` if no such vertebra exists.
        """
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
    C6 = 6, True, True
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
        """Integer label of this vertebra (same as ``self.value``)."""
        return self.value

    @property
    def RIB(self) -> int:
        """Integer label of the rib associated with this vertebra.

        Raises:
            AssertionError: If this vertebra has no associated rib.
        """
        assert self._rib is not None, (self.name, self.value)
        return self._rib

    @classmethod
    def rib2vert(cls, riblabel: int) -> int:
        """Convert a rib label integer back to its parent vertebra label integer.

        Args:
            riblabel (int): A valid rib label from :meth:`rib_label`.

        Returns:
            int: The vertebra label corresponding to ``riblabel``.

        Raises:
            AssertionError: If ``riblabel`` is not a known rib label.
        """
        assert riblabel in Vertebra_Instance.rib_label(), riblabel
        return riblabel - VERTEBRA_INSTANCE_RIB_LABEL_OFFSET if riblabel != 21 + VERTEBRA_INSTANCE_RIB_LABEL_OFFSET else 28

    @property
    def IVD(self) -> int:
        """Integer label of the intervertebral disc at this level (vertebra label + 100).

        Raises:
            AssertionError: If this vertebra has no associated IVD.
        """
        assert self._ivd is not None, (self.name, self.value)
        return self._ivd

    @property
    def ENDPLATE(self) -> int:
        """Integer label of the endplate at this level (vertebra label + 200).

        Raises:
            AssertionError: If this vertebra has no associated endplate.
        """
        assert self._endplate is not None, (self.name, self.value)
        return self._endplate

    def __str__(self):
        return str(self.name)


class Location(Abstract_lvl):
    """IntEnum of vertebra subregion and anatomical landmark labels used as POI subregion IDs.

    Values below 100 correspond to anatomical subregions of the vertebra
    (e.g. corpus, spinosus process, articular processes).  Values ≥ 100 are
    used for computed locations such as intervertebral discs (100), endplates
    superior (200–), and endplates inferior (300–).

    Members are serialised as integers (not names) because :meth:`save_as_name`
    returns ``False``.
    """

    @classmethod
    def save_as_name(cls) -> bool:
        """Always returns False; Location values are saved as integers."""
        return False

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
    Vertebra_Corpus_border = 49  # actual corpus body
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
    """Return the standard list of vertebra subregion :class:`Location` labels.

    The returned labels cover the main anatomical subregions used in vertebra
    segmentation: arcus, spinosus process, costal processes, articular processes,
    and vertebra corpus.

    Args:
        with_border (bool, optional): If True, also includes
            :attr:`Location.Vertebra_Corpus_border`. Defaults to True.

    Returns:
        list[Location]: Ordered list of vertebra subregion ``Location`` members.
    """
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

"""
Abbreviations:
- SSL: Supraspinous Ligament
- ALL: Anterior Longitudinal Ligament
- PLL: Posterior Longitudinal Ligament
- FL: Flavum Ligament
- ISL: Interspinous Ligament
- ITL: Intertransverse Ligament

- CR: Cranial / Superior
- CA: Caudal / Inferior

- S: Sinistra / Left
- D: Dextra / Right
"""
# TODO clean this shit up (some values not defined in Location, some different values I think)
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
