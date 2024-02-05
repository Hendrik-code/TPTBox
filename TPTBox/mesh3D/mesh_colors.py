from __future__ import annotations

import numpy as np


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

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "RGB_Color-" + str(self.rgb)

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
    ITK_1 = RGB_Color.init_list([255, 0, 0])
    ITK_2 = RGB_Color.init_list([0, 255, 0])
    ITK_3 = RGB_Color.init_list([0, 0, 255])
    ITK_4 = RGB_Color.init_list([255, 255, 0])
    ITK_5 = RGB_Color.init_list([0, 255, 255])
    ITK_6 = RGB_Color.init_list([255, 0, 255])
    ITK_7 = RGB_Color.init_list([255, 239, 213])
    ITK_8 = RGB_Color.init_list([0, 0, 205])
    ITK_9 = RGB_Color.init_list([205, 133, 63])
    ITK_10 = RGB_Color.init_list([210, 180, 140])
    ITK_11 = RGB_Color.init_list([102, 205, 170])
    ITK_12 = RGB_Color.init_list([0, 0, 128])
    ITK_13 = RGB_Color.init_list([0, 139, 139])
    ITK_14 = RGB_Color.init_list([46, 139, 87])
    ITK_15 = RGB_Color.init_list([255, 228, 225])
    ITK_16 = RGB_Color.init_list([106, 90, 205])
    ITK_17 = RGB_Color.init_list([221, 160, 221])
    ITK_18 = RGB_Color.init_list([233, 150, 122])
    ITK_19 = RGB_Color.init_list([165, 42, 42])
    ITK_28 = RGB_Color.init_list([218, 165, 32])
    ITK_20 = RGB_Color.init_list([255, 250, 250])
    ITK_21 = RGB_Color.init_list([147, 112, 219])
    ITK_22 = RGB_Color.init_list([218, 112, 214])
    ITK_23 = RGB_Color.init_list([75, 0, 130])
    ITK_24 = RGB_Color.init_list([255, 182, 193])
    ITK_25 = RGB_Color.init_list([60, 179, 113])
    ITK_26 = RGB_Color.init_list([255, 235, 205])
    ITK_29 = RGB_Color.init_list([240, 255, 240])
    ITK_30 = RGB_Color.init_list([32, 178, 170])
    ITK_31 = RGB_Color.init_list([230, 20, 147])
    ITK_32 = RGB_Color.init_list([25, 25, 112])
    ITK_33 = RGB_Color.init_list([112, 128, 144])
    ITK_27 = RGB_Color.init_list([255, 160, 122])
    ITK_100 = RGB_Color.init_list([176, 224, 230])
    ITK_60 = RGB_Color.init_list([244, 164, 96])
    ITK_61 = RGB_Color.init_list([255, 0, 0])
    ITK_62 = RGB_Color.init_list([0, 255, 0])
    ## Subregions
    ITK_41 = RGB_Color.init_list([238, 232, 170])
    ITK_42 = RGB_Color.init_list([240, 255, 240])
    ITK_43 = RGB_Color.init_list([245, 222, 179])
    ITK_44 = RGB_Color.init_list([184, 134, 11])
    ITK_45 = RGB_Color.init_list([32, 178, 170])
    ITK_46 = RGB_Color.init_list([255, 20, 147])
    ITK_47 = RGB_Color.init_list([25, 25, 112])
    ITK_48 = RGB_Color.init_list([112, 128, 144])
    ITK_49 = RGB_Color.init_list([34, 139, 34])
    ITK_50 = RGB_Color.init_list([248, 248, 255])


_color_dict = {v: getattr(Mesh_Color_List, v) for v in vars(Mesh_Color_List) if not callable(v) and not v.startswith("__")}

_color_mapping_by_label: dict[int, RGB_Color] = {
    i: _color_dict[f"ITK_{i}"] if f"ITK_{i}" in _color_dict else Mesh_Color_List.BLACK for i in range(1, 150)
}

_color_map_in_row = np.array([v.rgb for v in _color_mapping_by_label.values()])


def get_color_by_label(label: int):
    return _color_mapping_by_label[label]
