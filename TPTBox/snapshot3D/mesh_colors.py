from __future__ import annotations

import numpy as np
from TPTBox import v_idx2name


class RGB_Color:
    def __init__(self, rgb: tuple[int, int, int]):
        assert isinstance(rgb, tuple) and [isinstance(i, int) for i in rgb], "did not receive a tuple of 3 ints"
        self.rgb = np.array(rgb)

    @classmethod
    def init_separate(cls, r: int, g: int, b: int):
        return cls((r, g, b))

    @classmethod
    def init_list(cls, rgb: list[int] | np.ndarray):
        assert len(rgb) == 3, "rgb requires exactly three integers"
        if isinstance(rgb, np.ndarray):
            assert rgb.dtype == int, "rgb numpy array not of type int!"
        return cls(tuple(rgb))

    def __call__(self, normed: bool = False):
        if normed:
            return self.rgb / 255.0
        return self.rgb

    def __getitem__(self, item):
        return self.rgb[item] / 255.0


class Mesh_Color_List:
    # General Colors
    BEIGE = RGB_Color.init_list([255, 250, 200])
    MAROON = RGB_Color.init_list([128, 0, 0])
    YELLOW = RGB_Color.init_list([255, 255, 25])
    ORANGE = RGB_Color.init_list([245, 130, 48])
    BLUE = RGB_Color.init_list([30, 144, 255])
    BLACK = RGB_Color.init_list([0, 0, 0])
    WHITE = RGB_Color.init_list([255, 255, 255])
    GREEN = RGB_Color.init_list([50, 250, 65])
    MAGENTA = RGB_Color.init_list([240, 50, 250])
    SPRINGGREEN = RGB_Color.init_list([0, 255, 128])
    CYAN = RGB_Color.init_list([70, 240, 240])
    PINK = RGB_Color.init_list([255, 105, 180])
    BROWN = RGB_Color.init_list([160, 100, 30])
    DARKGRAY = RGB_Color.init_list([95, 93, 68])
    GRAY = RGB_Color.init_list([143, 140, 110])
    NAVY = RGB_Color.init_list([0, 0, 128])
    LIME = RGB_Color.init_list([210, 245, 60])

    # ITK Snap Colors
    ## Vert Mask
    ITK_C1 = RGB_Color.init_list([255, 0, 0])
    ITK_C2 = RGB_Color.init_list([0, 255, 0])
    ITK_C3 = RGB_Color.init_list([0, 0, 255])
    ITK_C4 = RGB_Color.init_list([255, 255, 0])
    ITK_C5 = RGB_Color.init_list([0, 255, 255])
    ITK_C6 = RGB_Color.init_list([255, 0, 255])
    ITK_C7 = RGB_Color.init_list([255, 239, 213])
    ITK_T1 = RGB_Color.init_list([0, 0, 205])
    ITK_T2 = RGB_Color.init_list([205, 133, 63])
    ITK_T3 = RGB_Color.init_list([210, 180, 140])
    ITK_T4 = RGB_Color.init_list([102, 205, 170])
    ITK_T5 = RGB_Color.init_list([0, 0, 128])
    ITK_T6 = RGB_Color.init_list([0, 139, 139])
    ITK_T7 = RGB_Color.init_list([46, 139, 87])
    ITK_T8 = RGB_Color.init_list([255, 228, 225])
    ITK_T9 = RGB_Color.init_list([106, 90, 205])
    ITK_T10 = RGB_Color.init_list([221, 160, 221])
    ITK_T11 = RGB_Color.init_list([233, 150, 122])
    ITK_T12 = RGB_Color.init_list([165, 42, 42])
    ITK_T13 = RGB_Color.init_list([218, 165, 32])
    ITK_L1 = RGB_Color.init_list([255, 250, 250])
    ITK_L2 = RGB_Color.init_list([147, 112, 219])
    ITK_L3 = RGB_Color.init_list([218, 112, 214])
    ITK_L4 = RGB_Color.init_list([75, 0, 130])
    ITK_L5 = RGB_Color.init_list([255, 182, 193])
    ITK_L6 = RGB_Color.init_list([60, 179, 113])
    ITK_S1 = RGB_Color.init_list([255, 235, 205])
    ITK_S2 = RGB_Color.init_list([240, 255, 240])
    ITK_S3 = RGB_Color.init_list([32, 178, 170])
    ITK_S4 = RGB_Color.init_list([230, 20, 147])
    ITK_S5 = RGB_Color.init_list([25, 25, 112])
    ITK_S6 = RGB_Color.init_list([112, 128, 144])
    ITK_Cocc = RGB_Color.init_list([255, 160, 122])
    ITK_Vertebra_Disc = RGB_Color.init_list([176, 224, 230])
    ITK_Spinal_Cord = RGB_Color.init_list([244, 164, 96])
    ITK_Spinal_Canal = RGB_Color.init_list([255, 0, 0])
    ITK_Endplate = RGB_Color.init_list([0, 255, 0])
    ## Subregions
    ITK_Arcus_Vertebrae = RGB_Color.init_list([238, 232, 170])
    ITK_Spinosus_Process = RGB_Color.init_list([240, 255, 240])
    ITK_Costal_Process_Left = RGB_Color.init_list([245, 222, 179])
    ITK_Costal_Process_Right = RGB_Color.init_list([184, 134, 11])
    ITK_Superior_Articular_Left = RGB_Color.init_list([32, 178, 170])
    ITK_Superior_Articular_Right = RGB_Color.init_list([255, 20, 147])
    ITK_Inferior_Articular_Left = RGB_Color.init_list([25, 25, 112])
    ITK_Inferior_Articular_Right = RGB_Color.init_list([112, 128, 144])
    ITK_Vertebra_Corpus_border = RGB_Color.init_list([34, 139, 34])
    ITK_Vertebra_Corpus = RGB_Color.init_list([248, 248, 255])


color_dict = {v: getattr(Mesh_Color_List, v) for v in vars(Mesh_Color_List) if not callable(v) and not v.startswith("__")}


snap3d_color_list = (
    Mesh_Color_List.DARKGRAY,
    Mesh_Color_List.BEIGE,
    Mesh_Color_List.MAROON,
    Mesh_Color_List.YELLOW,
    Mesh_Color_List.ITK_Superior_Articular_Left,
    Mesh_Color_List.ITK_Superior_Articular_Right,
    Mesh_Color_List.ORANGE,
    Mesh_Color_List.BLUE,
    Mesh_Color_List.ITK_Vertebra_Corpus_border,
    Mesh_Color_List.ITK_Vertebra_Corpus_border,
)
subreg3d_color_dict = {i + 41: snap3d_color_list[i] for i in range(10)}

vert3d_color_dict = {
    i: color_dict[f"ITK_{v_idx2name[i]}"] if i in v_idx2name and f"ITK_{v_idx2name[i]}" in color_dict else Mesh_Color_List.BLACK
    for i in range(1, 150)
}
vert_color_map = np.array([v.rgb for v in vert3d_color_dict.values()])
