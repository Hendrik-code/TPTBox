from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyvista as pv

# from utils.filepaths import filepath_data, filepath_dataset
# from utils.poi_plotter import *
# from utils.poi_surface import project_pois_onto_segmentation_surface
from TPTBox import BIDS_FILE, POI, BIDS_Global_info, Log_Type, No_Logger, Print_Logger, logger
from TPTBox.core.nii_wrapper import NII, to_nii
from TPTBox.core.poi import calc_poi_average
from TPTBox.core.vert_constants import conversion_poi
from TPTBox.mesh3D.mesh import Mesh3D, POIMesh, SegmentationMesh
from TPTBox.mesh3D.mesh_colors import RGB_Color, get_color_by_label


def _add_mesh(pl, mesh: pv.PolyData | SegmentationMesh, color: str | RGB_Color, opacity: float = 1.0):
    if isinstance(mesh, Mesh3D):
        mesh = mesh.mesh
    if isinstance(color, RGB_Color):
        color = color.rgb.tolist()

    pl.add_mesh(mesh, opacity=opacity, color=color)


@dataclass
class Preview_Settings:
    obj: NII | POI
    offset: tuple[float, float, float] | None = None  # PIR
    opacity: float = 1.0
    color: Literal["auto"] | str | None = "auto"  # noqa: PYI051
    binary = False

    def _get_mesh(
        self,
        rescale_to_iso,
        poi_size,
        default_color_nii="bisque",
        default_poi_nii="red",
    ):
        img = self.obj
        if (self.color is None or self.color == "auto") and not isinstance(img, NII):
            self.color = default_poi_nii
        if self.binary or (self.color is not None and self.color != "auto"):
            if isinstance(img, NII):
                mesh = SegmentationMesh.from_segmentation_nii(img, rescale_to_iso=rescale_to_iso)
                is_poi = False
            else:
                mesh = POIMesh(img, rescale_to_iso=False, regions=None, subregions=None, size_factor=poi_size)
                is_poi = True
            if self.offset is not None:
                mesh = mesh.get_mesh_with_offset(self.offset)
            color = self.color
            if color is None or color == "auto":
                color = default_poi_nii if is_poi else default_color_nii
            yield mesh, color
        elif isinstance(img, NII):
            for u in img.unique():
                color = get_color_by_label(u)
                mesh = SegmentationMesh.from_segmentation_nii(img.extract_label(u), rescale_to_iso=rescale_to_iso)
                if self.offset is not None:
                    mesh = mesh.get_mesh_with_offset(self.offset)
                yield mesh, color
        else:
            raise NotImplementedError("auto poi color")


l = Print_Logger()

offset = tuple[float, float, float]


def make_html_preview(
    images: list[NII | POI | Preview_Settings],
    html_out: str | Path | None,
    # spaceing=None,
    background="black",
    rescale_to_iso=False,
    poi_size=1.7,
    logger=l,
    show=False,
    default_color_nii="bisque",
    default_poi_nii="red",
):
    assert (html_out is None) or str(html_out).endswith(".html"), f"not a valid file ending {html_out}; expected .html"
    assert html_out is not None or show, "show must be True or html_out must be set"
    pl: pv.Plotter = pv.Plotter()  # type: ignore
    pl.set_background(background, top=None)  # type: ignore
    pl.add_axes()  # type: ignore

    images_ = [Preview_Settings(obj) if not isinstance(obj, Preview_Settings) else obj for obj in images]

    for setting in images_:
        for m, color in setting._get_mesh(
            poi_size=poi_size,
            rescale_to_iso=rescale_to_iso,
            default_color_nii=default_color_nii,
            default_poi_nii=default_poi_nii,
        ):
            _add_mesh(pl, m, opacity=setting.opacity, color=color)

    if html_out is not None:
        pl.export_html(html_out)
        logger.print(f"Saved scene into {html_out}", Log_Type.SAVE)
    if show:
        pl.show()


if __name__ == "__main__":
    p = "/media/data/robert/dataset-myelom/dataset-myelom/derivatives-leg/MM00191/ses-20180502"
    nii = to_nii(Path(p, "sub-MM00191_ses-20180502_sequ-202_seg-leg-left_msk.nii.gz"), True)
    poi = POI.load(Path(p, "sub-MM00191_ses-20180502_sequ-202_seg-leg-subreg-left_poi.mrk.json"), reference=nii)
    nii_r = to_nii(Path(p, "sub-MM00191_ses-20180502_sequ-202_seg-leg-right_msk.nii.gz"), True)
    poi_r = POI.load(Path(p, "sub-MM00191_ses-20180502_sequ-202_seg-leg-subreg-right_poi.mrk.json"), reference=nii)
    make_html_preview(
        [
            nii,
            poi,
            nii_r,
            poi_r,
            Preview_Settings(nii, offset=(0, 0, -250), opacity=0.5),
            Preview_Settings(poi, offset=(0, 0, -250)),
        ],
        Path(p, "test.html"),
        show=True,
        poi_size=10,
    )
