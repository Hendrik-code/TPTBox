from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from typing_extensions import Self

from TPTBox.core import poi
from TPTBox.core.nii_poi_abstract import Has_Grid
from TPTBox.core.poi_fun.poi_abstract import Abstract_POI, POI_Descriptor
from TPTBox.core.poi_fun.save_load import FORMAT_GLOBAL, load_poi, save_poi
from TPTBox.core.vert_constants import Abstract_lvl, logging

###### GLOBAL POI #####


class POI_Global(Abstract_POI):
    """
    Represents a global Point of Interest (POI) in a coordinate system.
    Inherits from the `Abstract_POI` class and contains methods for converting the POI to different coordinate systems.
    """

    def __init__(
        self,
        input_poi: poi.POI | POI_Descriptor | dict[str, dict[str, tuple[float, ...]]],
        itk_coords: bool = False,
        level_one_info: type[Abstract_lvl] | None = None,  # Must be Enum and must has order_dict
        level_two_info: type[Abstract_lvl] | None = None,
        info: dict | None = None,
    ):
        args = {}
        if level_one_info is not None:
            args["level_one_info"] = level_one_info
        if level_one_info is not None:
            args["level_two_info"] = level_two_info
        self.itk_coords = itk_coords
        _format = FORMAT_GLOBAL
        if isinstance(input_poi, dict):
            global_points = POI_Descriptor()
            for k1, d1 in input_poi.items():
                for k2, v in d1.items():
                    global_points[k1:k2] = v
        elif isinstance(input_poi, POI_Descriptor):
            global_points = input_poi

        elif isinstance(input_poi, poi.POI):
            local_poi = input_poi.copy()
            global_points = poi.POI_Descriptor(definition=local_poi.centroids.definition)
            for k1, k2, v in local_poi.items():
                global_points[k1:k2] = local_poi.local_to_global(v, itk_coords)
            info = input_poi.info.copy()
            _format = input_poi.format
        else:
            raise NotImplementedError(type(input_poi))
        if info is None:
            info = {}

        super().__init__(_centroids=global_points, format=_format, info=info, **args)

    def __str__(self) -> str:
        return str(self._centroids)

    @property
    def zoom(self):
        return (1, 1, 1)

    @property
    def origin(self):
        return (0, 0, 0)

    @property
    def is_global(self) -> bool:
        """
        Check if the POI is global.

        Returns:
            bool: True if the POI is global, False otherwise.
        """
        return True

    def to_other_nii(self, ref: poi.Image_Reference) -> poi.POI | poi.NII:
        """
        Convert the POI to another NII file.

        Args:
            ref (poi.Image_Reference): The reference to the NII file.

        Returns:
            Union[poi.POI, poi.NII]: The converted POI as either a POI or NII object.
        """
        return self.to_other(poi.to_nii(ref))

    def to_other_poi(self, ref: poi.POI | Self):
        """
        Convert the POI to another POI.

        Args:
            ref (poi.Centroid_Reference): The reference to the other POI.

        Returns:
            poi.POI: The converted POI.
        """
        p = poi.POI.load(ref)
        if isinstance(ref, poi.POI):
            return self.to_other(p)
        elif isinstance(ref, Self):
            return self.to_cord_system(ref.itk_coords)

    def to_global(self):
        return self

    def resample_from_to(self, msk: Has_Grid):
        return self.to_other(msk)

    def to_cord_system(self, itk_coords: bool, inplace=False):
        out = self if inplace else self.copy()
        if self.itk_coords == itk_coords:
            return out
        out.itk_coords = itk_coords
        for k1, k2, v in self.items():
            out[k1, k2] = (-v[0], -v[1], v[2])
        return out

    def to_other(self, msk: Has_Grid, verbose=False) -> poi.POI:
        """
        Convert the POI to another coordinate system.

        Args:
            msk (Union[poi.POI, poi.NII]): The reference to the other coordinate system.

        Returns:
            poi.POI: The converted POI.
        """
        out = poi.POI_Descriptor(definition=self._get_centroids().definition)
        for k1, k2, v in self.items():
            if self.itk_coords:
                assert len(v) == 3, "n-d vec not implemented for n != 3"
                v = (-v[0], -v[1], v[2])  # noqa: PLW2901
            v_out = msk.global_to_local(v)
            if verbose:
                print(v, "-->", v_out)
            out[k1, k2] = tuple(v_out)

        return poi.POI(centroids=out, **msk._extract_affine(), info=self.info, format=self.format)

    def copy(self, centroids: POI_Descriptor | None = None) -> Self:
        if centroids is None:
            centroids = self.centroids.copy()
        p = POI_Global(centroids)
        p.format = self.format
        p.info = deepcopy(self.info)
        p.itk_coords = self.itk_coords
        return p  # type: ignore

    @classmethod
    def load(cls, poi: poi.POI_Reference, itk_coords: bool | None = None) -> Self:
        poi_obj = load_poi(poi)
        if not isinstance(poi_obj, POI_Global):
            poi_obj = poi_obj.to_global(itk_coords if itk_coords is not None else False)
        if itk_coords is not None:
            assert itk_coords == poi_obj.itk_coords, "not implemented swichting to/from itk_coords to nii "
        return poi_obj  # type: ignore

    def save(
        self,
        out_path: str | Path,
        make_parents=False,
        additional_info: dict | None = None,
        save_hint=FORMAT_GLOBAL,
        resample_reference: Has_Grid | None = None,
        verbose: logging = True,
    ):
        return save_poi(
            self, out_path, make_parents, additional_info, save_hint=save_hint, resample_reference=resample_reference, verbose=verbose
        )
