from copy import deepcopy

from typing_extensions import Self

from TPTBox.core import poi
from TPTBox.core.poi_fun.poi_abstract import Abstract_POI, POI_Descriptor

###### GLOBAL POI #####


class POI_Global(Abstract_POI):
    """
    Represents a global Point of Interest (POI) in a coordinate system.
    Inherits from the `Abstract_POI` class and contains methods for converting the POI to different coordinate systems.
    """

    def __init__(self, input_poi: poi.POI | POI_Descriptor):
        if isinstance(input_poi, POI_Descriptor):
            global_points = input_poi
            self.info = {}
            self.format = None
        elif isinstance(input_poi, poi.POI):
            local_poi = input_poi.copy()
            global_points = poi.POI_Descriptor(definition=local_poi.centroids.definition)
            for k1, k2, v in local_poi.items():
                global_points[k1:k2] = local_poi.local_to_global(v)
            self.info = input_poi.info.copy()
            self.format = input_poi.format
        else:
            raise NotImplementedError(input_poi)
        self._centroids = global_points

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
        return self.to_other(poi.POI.load(ref))

    def to_other(self, msk: poi.POI | poi.NII) -> poi.POI:
        """
        Convert the POI to another coordinate system.

        Args:
            msk (Union[poi.POI, poi.NII]): The reference to the other coordinate system.

        Returns:
            poi.POI: The converted POI.
        """
        out = poi.POI_Descriptor(definition=self._get_centroids().definition)
        for k1, k2, v in self.items():
            v_out = msk.global_to_local(v)
            out[k1, k2] = tuple(v_out)

        return poi.POI(centroids=out, **msk._extract_affine(), info=self.info, format=self.format)

    def copy(self, centroids: POI_Descriptor | None = None) -> Self:
        if centroids is None:
            centroids = self.centroids.copy()
        p = POI_Global(centroids)
        p.format = self.format
        p.info = deepcopy(self.info)
        return p  # type: ignore
