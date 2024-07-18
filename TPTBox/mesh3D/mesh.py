import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from enum import Enum

import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes

from TPTBox import NII, POI, Image_Reference, Log_Type, to_nii_seg
from TPTBox.core import vert_constants as vc
from TPTBox.core.np_utils import np_bbox_binary
from TPTBox.core.poi import COORDINATE

log = vc.log
logging = vc.logging


class MeshOutputType(Enum):
    PLY = "ply"


class Mesh3D:
    def __init__(self, mesh: pv.PolyData) -> None:
        self.mesh = mesh

    def save(self, filepath: str | Path, mode: MeshOutputType = MeshOutputType.PLY, verbose: logging = True):
        filepath = str(filepath)
        if not filepath.endswith(mode.value):
            filepath += mode.value

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(filepath)

        if mode == MeshOutputType.PLY:
            self.mesh.export_obj(filepath)
        else:
            raise NotImplementedError(f"save with mode {mode}")
        log.print(f"Saved mesh: {filepath}", Log_Type.SAVE, verbose=verbose)

    @classmethod
    def load(cls, filepath: str | Path):
        assert Path(filepath).exists(), f"loading mesh from {filepath}, filepath does not exist"
        reader = pv.get_reader(str(filepath))
        mesh = reader.read()
        return Mesh3D(mesh)

    def show(self):
        print("1")
        pv.start_xvfb()
        pl = pv.Plotter()
        pl.set_background("black", top=None)
        pv.global_theme.axes.show = True
        pv.global_theme.edge_color = "white"
        pv.global_theme.interactive = True

        pl.add_mesh(self.mesh)
        print(2)
        pl.show()


class SegmentationMesh(Mesh3D):
    def __init__(self, int_arr: np.ndarray | Image_Reference) -> None:
        if not isinstance(int_arr, np.ndarray):
            seg_nii = to_nii_seg(int_arr)
            seg_nii.reorient_().rescale_()
            int_arr = seg_nii.get_array()
        assert np.min(int_arr) == 0, f"min value of image is not zero, got {np.min(int_arr)}"
        assert len(int_arr.shape) == 3, f"image does not have exactly 3 dimensions, got shape {int_arr.shape}"

        # Force dtype to uint
        if np.issubdtype(int_arr.dtype, np.floating):
            print("input is of type float, converting to int")
            int_arr.astype(np.uint16)
        print("calculate bounding box cutout")
        # calculate bounding box cutout
        bbox_crop = np_bbox_binary(int_arr, px_dist=2)
        x1, y1, z1 = bbox_crop[0].start, bbox_crop[1].start, bbox_crop[2].start
        arr_cropped = int_arr[bbox_crop]

        vertices, faces, normals, values = marching_cubes(arr_cropped, gradient_direction="ascent", step_size=1)
        vertices += (x1, y1, z1)  # so it has correct relative coordinates (not world coordinates!)
        print("column_stack")

        vfaces = np.column_stack((np.ones(len(faces)) * 3, faces)).astype(int)

        mesh = pv.PolyData(vertices, vfaces)
        mesh["Normals"] = normals
        mesh["values"] = values
        print("mesh")
        self.mesh = mesh

    @classmethod
    def from_segmentation_nii(cls, seg_nii: NII, rescale_to_iso: bool = True):
        assert seg_nii.seg, "NII is not a segmentation"
        seg_nii.reorient_()
        if rescale_to_iso:
            seg_nii.rescale_()

        return SegmentationMesh(seg_nii.get_seg_array())


class POIMesh(Mesh3D):
    def __init__(
        self,
        poi: POI,
        rescale_to_iso: bool = True,
        regions: list[int] | None = None,
        subregions: list[int] | None = None,
        size_factor: int = 5,
    ) -> None:
        poi.reorient_()
        if rescale_to_iso:
            poi.rescale_()

        if regions is None:
            regions = poi.keys_region()

        if subregions is None:
            subregions = poi.keys_subregion()

        poi_extracted: list[COORDINATE] = []

        for r_id, s_id, coord in poi.items():
            if r_id in regions and s_id in subregions:
                poi_extracted.append(coord)

        n = pv.PolyData(poi_extracted)
        n["radius"] = np.ones(shape=len(poi_extracted)) * size_factor
        geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
        glyphed = n.glyph(scale="radius", geom=geom, progress_bar=False, orient=False)
        self.mesh = glyphed


if __name__ == "__main__":
    p = "/media/hendrik/be5e95dd-27c8-4c31-adc5-7b75f8ebd5c5/data/hendrik/test_samples/bids_mesh/"
    seg_nii = NII.load(p + "sub-100001_sequ-31_acq-sag_chunk-LWS_mod-T2w_seg-vert_msk.nii.gz", seg=True)

    poi = POI.load(p + "sub-100001_sequ-31_acq-sag_chunk-LWS_mod-T2w_seg-spine_ctd.json")

    mesh = SegmentationMesh.from_segmentation_nii(seg_nii)

    poi_mesh = POIMesh(poi)

    pv.start_xvfb()
    pl = pv.Plotter()
    pl.set_background("black", top=None)
    pl.add_axes()
    pl.add_mesh(mesh.mesh, opacity=0.9, color="gray")
    pl.add_mesh(poi_mesh.mesh, color="red")

    pl.show()
