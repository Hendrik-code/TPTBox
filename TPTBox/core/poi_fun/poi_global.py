from __future__ import annotations

from copy import deepcopy

from typing_extensions import Self

from TPTBox.core import poi
from TPTBox.core.nii_poi_abstract import Has_Grid
from TPTBox.core.poi_fun.poi_abstract import Abstract_POI, POI_Descriptor
from TPTBox.core.vert_constants import Abstract_lvl

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
        _format = poi.FORMAT_GLOBAL
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

    def to_other_poi(self, ref: poi.POI_Reference) -> poi.POI:
        """
        Convert the POI to another POI.

        Args:
            ref (poi.Centroid_Reference): The reference to the other POI.

        Returns:
            poi.POI: The converted POI.
        """
        p = poi.POI.load(ref)
        assert isinstance(p, poi.POI), "Not implemented"
        if isinstance(p, poi.POI):
            return self.to_other(p)
        else:
            assert self.itk_coords == p.itk_coords, "itk_coords not implemented"  # TODO
            return self  # type: ignore
            # TODO Generics ref -> output type

    def to_global(self):
        return self

    def resample_from_to(self, msk: Has_Grid):
        return self.to_other(msk)

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
