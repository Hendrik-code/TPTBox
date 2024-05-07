import numpy as np  # noqa: INP001

from TPTBox import v_idx2name
from TPTBox.mesh3D.mesh_colors import Mesh_Color_List, _color_dict

snap3d_vert_subregion_color_list = (
    Mesh_Color_List.DARKGRAY,
    Mesh_Color_List.BEIGE,
    Mesh_Color_List.MAROON,
    Mesh_Color_List.YELLOW,
    Mesh_Color_List.ITK_45,
    Mesh_Color_List.ITK_46,
    Mesh_Color_List.ORANGE,
    Mesh_Color_List.BLUE,
    Mesh_Color_List.ITK_49,
    Mesh_Color_List.ITK_49,
)
subreg3d_color_dict = {i + 41: snap3d_vert_subregion_color_list[i] for i in range(10)}

vert3d_color_dict = {
    i: _color_dict[f"ITK_{v_idx2name[i]}"] if i in v_idx2name and f"ITK_{v_idx2name[i]}" in _color_dict else Mesh_Color_List.BLACK
    for i in range(1, 150)
}
vert_color_map = np.array([v.rgb for v in vert3d_color_dict.values()])


if __name__ == "__main__":
    print(vert3d_color_dict)
