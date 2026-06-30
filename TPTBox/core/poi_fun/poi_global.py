from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

###### GLOBAL POI #####
from typing_extensions import Self

from TPTBox.core import poi
from TPTBox.core.nii_poi_abstract import Has_Grid
from TPTBox.core.poi_fun import save_mkr
from TPTBox.core.poi_fun.poi_abstract import Abstract_POI, POI_Descriptor
from TPTBox.core.poi_fun.save_load import FORMAT_GLOBAL, load_poi, save_poi
from TPTBox.core.vert_constants import Abstract_lvl, logging
from TPTBox.logger.log_file import log


class POI_Global(Abstract_POI):
    """POI container stored in world (mm) coordinates rather than voxel space.

    Extends :class:`~TPTBox.core.poi_fun.poi_abstract.Abstract_POI` with coordinate-system
    conversion methods.
    """

    def __init__(
        self,
        input_poi: poi.POI | POI_Descriptor | dict[str, dict[str, tuple[float, ...]]] = None,
        itk_coords: bool = False,
        level_one_info: type[Abstract_lvl] | None = None,  # Must be Enum and must has order_dict
        level_two_info: type[Abstract_lvl] | None = None,
        info: dict | None = None,
    ):
        if input_poi is None:
            input_poi = {}
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
            items = list(local_poi.items())
            if items:
                # one batched affine matmul instead of a per-point local_to_global loop
                arr = local_poi.local_to_global_arr(np.asarray([v for _, _, v in items]), itk_coords)
                for (k1, k2, _), row in zip(items, arr.tolist()):
                    global_points[k1:k2] = tuple(row)
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
    def zoom(self) -> tuple[int, int, int]:
        """Always returns ``(1, 1, 1)`` — global POIs are in mm so zoom is unity."""
        return (1, 1, 1)

    @property
    def origin(self) -> tuple[int, int, int]:
        """Always returns ``(0, 0, 0)`` — global POIs use a world origin."""
        return (0, 0, 0)

    @property
    def orientation(self) -> tuple[str, str, str]:
        """Return the axis-code orientation for the active coordinate system.

        Returns:
            ``("L", "A", "S")`` for ITK/LPS coordinates, ``("R", "P", "S")``
            for NIfTI/RAS coordinates.
        """
        if self.itk_coords:
            return ("L", "A", "S")
        return ("R", "P", "S")

    @property
    def is_global(self) -> bool:
        """Check if the POI is global.

        Returns:
            bool: True if the POI is global, False otherwise.
        """
        return True

    def to_other_nii(self, ref: poi.Image_Reference) -> poi.POI | poi.NII:
        """Convert the POI to another NII file.

        Args:
            ref (poi.Image_Reference): The reference to the NII file.

        Returns:
            Union[poi.POI, poi.NII]: The converted POI as either a POI or NII object.
        """
        return self.to_other(poi.to_nii(ref))

    def to_other_poi(self, ref: poi.POI | Self) -> poi.POI | Self | None:
        """Convert the POI to another POI.

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

    def to_global(self, itk_coords: bool | None = None) -> Self:
        """Return this object unchanged (already in global coordinates)."""
        return self.to_cord_system(itk_coords) if itk_coords is not None else self.copy()

    def to_local(self, msk: Has_Grid) -> poi.POI:
        """Convert this global POI to the voxel space of ``msk``.

        Args:
            msk: Reference grid (``NII`` or ``POI``) defining the target affine.

        Returns:
            ``POI`` in the local voxel coordinate system of ``msk``.
        """
        return self.resample_from_to(msk)

    def resample_from_to(self, msk: Has_Grid) -> poi.POI:
        """Alias for :meth:`to_local` / :meth:`to_other`.

        Args:
            msk: Reference grid defining the target affine.

        Returns:
            ``POI`` in the local coordinate system of ``msk``.
        """
        return self.to_other(msk)

    def to_cord_system(self, itk_coords: bool, inplace: bool = False) -> Self:
        """Convert between ITK (LPS) and NIfTI (RAS) coordinate systems.

        Flips the first two coordinate axes when switching between the two
        systems (LPS ↔ RAS only differs in the sign of x and y).

        Args:
            itk_coords: ``True`` for ITK/LPS output, ``False`` for NIfTI/RAS.
            inplace: Convert in place.  Defaults to ``False``.

        Returns:
            ``POI_Global`` in the requested coordinate system.
        """
        out = self if inplace else self.copy()
        if self.itk_coords == itk_coords:
            return out
        out.itk_coords = itk_coords
        for k1, k2, v in self.items():
            out[k1, k2] = (-v[0], -v[1], v[2])
        return out

    def to_other(self, msk: Has_Grid, verbose=False) -> poi.POI:
        """Convert the POI to another coordinate system.

        Args:
            msk (Union[poi.POI, poi.NII]): The reference to the other coordinate system.

        Returns:
            poi.POI: The converted POI.
        """
        out = poi.POI_Descriptor(definition=self._get_centroids().definition)
        items = list(self.items())
        if items and not verbose:
            # one batched inverse-affine matmul instead of a per-point global_to_local loop
            arr = np.asarray([v for _, _, v in items], dtype=float)
            if self.itk_coords:
                assert arr.shape[1] == 3, "n-d vec not implemented for n != 3"
                arr[:, 0] *= -1
                arr[:, 1] *= -1
            arr = msk.global_to_local_arr(arr)
            for (k1, k2, _), row in zip(items, arr.tolist()):
                out[k1, k2] = tuple(row)
        else:
            for k1, k2, v in items:
                if self.itk_coords:
                    assert len(v) == 3, "n-d vec not implemented for n != 3"
                    v = (-v[0], -v[1], v[2])  # noqa: PLW2901
                v_out = msk.global_to_local(v)
                if verbose:
                    log.print(v, "-->", v_out)
                out[k1, k2] = tuple(v_out)

        return poi.POI(centroids=out, **msk._extract_affine(), info=self.info, format=self.format)

    def copy(self, centroids: POI_Descriptor | None = None) -> Self:
        """Return a deep copy of this ``POI_Global``.

        Args:
            centroids: Optional replacement ``POI_Descriptor``.  When ``None``
                the current centroids are deep-copied.

        Returns:
            New ``POI_Global`` with the same metadata as this object.
        """
        if centroids is None:
            centroids = self.centroids.copy()
        p = POI_Global(centroids)
        p.level_one_info = self.level_one_info
        p.level_two_info = self.level_two_info
        p.format = self.format
        p.info = deepcopy(self.info)
        p.itk_coords = self.itk_coords
        return p  # type: ignore

    @classmethod
    def load(cls, poi: poi.POI_Reference, itk_coords: bool | None = None) -> Self:
        """Load a ``POI_Global`` from a file or POI reference.

        Args:
            poi: Path to a JSON or ``.mrk.json`` file, or any supported
                ``POI_Reference`` type.
            itk_coords: When ``None`` the coordinate system is inferred from
                the file.  When ``True`` or ``False``, the loaded POI is
                asserted to match.

        Returns:
            ``POI_Global`` in the requested coordinate system.

        Raises:
            AssertionError: If ``itk_coords`` is set and does not match the
                file's coordinate system.
        """
        poi_obj = load_poi(poi)

        if not poi_obj.is_global or itk_coords is not None:
            poi_obj = poi_obj.to_global(itk_coords if itk_coords is not None else False)  # type: ignore
        return poi_obj  # type: ignore

    def save(
        self,
        out_path: str | Path,
        make_parents: bool = False,
        additional_info: dict | None = None,
        save_hint: int = FORMAT_GLOBAL,
        resample_reference: Has_Grid | None = None,
        verbose: logging = True,
    ) -> None:
        """Save this ``POI_Global`` to a JSON file.

        Args:
            out_path: Output path (must end with ``.json``).
            make_parents: Create parent directories if missing.
                Defaults to ``False``.
            additional_info: Extra key-value pairs added to the file header.
            save_hint: Format identifier.  Defaults to ``FORMAT_GLOBAL``.
            resample_reference: When set, convert to local coordinates of this
                grid before saving.
            verbose: Emit a save log message.  Defaults to ``True``.
        """
        return save_poi(
            self, out_path, make_parents, additional_info, save_hint=save_hint, resample_reference=resample_reference, verbose=verbose
        )

    def save_mrk(
        self: Self,
        filepath: str | Path,
        color: list[float] | None = None,
        split_by_region: bool = False,
        split_by_subregion: bool = False,
        add_points: bool = True,
        add_lines: list[save_mkr.MKR_Lines] | None = None,
        display: save_mkr.MKR_Display | dict = None,  # type: ignore
        pointLabelsVisibility: bool = False,
        glyphScale: float = 5.0,
        main_key: str = "Point",
    ) -> None:
        """Save this ``POI_Global`` as a 3D Slicer ``.mrk.json`` markup file.

        Delegates to :func:`~TPTBox.core.poi_fun.save_mkr._save_mrk`.

        Args:
            filepath: Output path.  The extension is forced to ``.mrk.json``.
            color: Default group colour (RGB in ``[0, 1]`` range).
            split_by_region: Separate markup group per region.
                Defaults to ``False``.
            split_by_subregion: Separate markup group per subregion.
                Defaults to ``False``.
            add_points: Include Fiducial markups.  Defaults to ``True``.
            add_lines: Optional ``MKR_Lines`` definitions to add as line
                markups.
            display: Base display property overrides.
            pointLabelsVisibility: Show point labels in the 3D view.
                Defaults to ``False``.
            glyphScale: Glyph size factor.  Defaults to ``5.0``.
            main_key: Base markup group key.  Defaults to ``"Point"``.
        """
        save_mkr._save_mrk(
            poi=self,
            filepath=filepath,
            color=color,
            split_by_region=split_by_region,
            split_by_subregion=split_by_subregion,
            add_points=add_points,
            add_lines=add_lines,
            display=display,
            pointLabelsVisibility=pointLabelsVisibility,
            glyphScale=glyphScale,
            main_key=main_key,
        )
