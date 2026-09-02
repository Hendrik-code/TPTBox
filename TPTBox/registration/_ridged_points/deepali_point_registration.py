from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from deepali.core import Axes, Sampling
from deepali.core import Grid as Deepali_Grid
from deepali.modules import TransformImage
from deepali.spatial import HomogeneousTransform

from TPTBox import (
    NII,
    POI,
    Has_Grid,
    Image_Reference,
    Location,
    Log_Type,
    Logger_Interface,
    No_Logger,
    POI_Reference,
    calc_poi_from_subreg_vert,
    to_nii,
)
from TPTBox.core.internal.deep_learning_utils import DEVICES, get_device

NII_or_POI = TypeVar("NII_or_POI")


def _horn_rigid(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form rigid alignment (Kabsch/Horn) mapping ``p`` onto ``q``.

    Args:
        p: ``(N, 3)`` moving-side points.
        q: ``(N, 3)`` fixed-side points.

    Returns:
        ``(R, t)`` such that ``q ≈ R @ p + t``. With a single pair (N == 1)
        rotation is under-determined, so ``R`` is the identity and only the
        translation component ``t = q - p`` is fitted.
    """
    assert p.shape == q.shape, (p.shape, q.shape)
    assert p.shape[0] >= 1, f"need at least 1 pair, got {p.shape[0]}"
    if p.shape[0] == 1:
        return np.eye(3), (q[0] - p[0])
    c_p = p.mean(axis=0)
    c_q = q.mean(axis=0)
    pp = p - c_p
    qq = q - c_q
    h = pp.T @ qq
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    if not np.isfinite(d) or d == 0:
        d = 1.0
    s = np.diag([1.0, 1.0, d])
    r = vt.T @ s @ u.T
    t = c_q - r @ c_p
    return r, t


_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def _poi_world_ras(poi: POI, keys: list[tuple[int, int]]) -> np.ndarray:
    """Return NIfTI-RAS world coordinates for the given POI keys as an ``(N, 3)`` array."""
    coords = np.array([poi[k] for k in keys], dtype=float)
    return poi.local_to_global_arr(coords)


def _ras_to_lps_matrix(w_ras: np.ndarray) -> np.ndarray:
    """Convert a homogeneous 4x4 rigid matrix from RAS to LPS convention."""
    return _RAS_TO_LPS @ w_ras @ _RAS_TO_LPS


def _voxel_from_world_ras(grid: Has_Grid, world_pt: np.ndarray) -> tuple[float, float, float]:
    """Convert a single RAS world coordinate to voxel coords for the given grid."""
    return grid.global_to_local(tuple(world_pt))  # type: ignore[attr-defined]


class Deepali_Point_Registration:
    """Closed-form rigid point registration built on DeepALI.

    Mirrors the API of :class:`Point_Registration` (Kabsch/Horn SVD fit on shared
    landmark pairs) but the fitted transform is a DeepALI
    :class:`~deepali.spatial.HomogeneousTransform`. This makes the result usable
    as a pre-registration inside DeepALI training pipelines (see
    :class:`~TPTBox.registration._deformable.multilabel_segmentation.Template_Registration2`).

    Attributes:
        transform: DeepALI ``HomogeneousTransform`` living on the fixed grid.
            Its matrix is expressed in target-cube coordinates and represents the
            direction ``fixed → moving`` (as required by ``TransformImage``).
        target_grid: Fixed / reference grid (defines the output space).
        input_grid: Moving / source grid.
        error_reg: Mean residual (LPS world mm) of the fitted landmark pairs.
        error_natural: Mean absolute difference of consecutive point distances
            in fixed vs. moving space – a metric-free sanity check.
        world_matrix: 4x4 LPS-world rigid matrix such that
            ``moving_world ≈ world_matrix @ [fixed_world; 1]``.
    """

    def __init__(
        self,
        poi_fixed: POI,
        poi_moving: POI,
        exclusion: list | None = None,
        log: Logger_Interface = No_Logger(),  # noqa: B008
        verbose: bool = True,
        ax_code=None,
        zooms=None,
        leave_worst_percent_out: float = 0.0,
        device: torch.device | str | int | None = None,
        gpu: int = 0,
        ddevice: DEVICES = "cuda",
        align_corners: bool = True,
    ) -> None:
        """Fit a closed-form rigid registration between two POIs using DeepALI.

        Args:
            poi_fixed: Reference POI (target of the registration).
            poi_moving: Moving POI whose coordinates are aligned to ``poi_fixed``.
            exclusion: Vertebra-level keys (first tuple element) to skip during
                fitting.
            log: Logger used to emit diagnostics.
            verbose: If True, forwards verbose logging to ``log``.
            ax_code: Optional target orientation code; ``poi_fixed`` is reoriented
                to it before fitting.
            zooms: Optional voxel spacing to rescale ``poi_fixed`` to. Pass
                ``(-1, -1, -1)`` (or ``None``) to skip.
            leave_worst_percent_out: Fraction in ``[0, 1)`` of point pairs with
                the largest post-fit residual to discard before re-fitting.
            device: PyTorch device. When ``None``, resolved from ``ddevice``/``gpu``.
            gpu: GPU index used when ``device`` is ``None``.
            ddevice: Device type used when ``device`` is ``None``.
            align_corners: DeepALI grid corner convention.
        """
        assert 0.0 <= leave_worst_percent_out < 1.0
        if exclusion is None:
            exclusion = []
        if device is None:
            device = get_device(ddevice, gpu)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.align_corners = align_corners

        if ax_code is not None:
            poi_fixed = poi_fixed.reorient(ax_code)
        if zooms is not None and tuple(zooms) != (-1, -1, -1):
            poi_fixed = poi_fixed.rescale(zooms)

        f_keys = [k for k in poi_fixed.keys() if k[0] not in exclusion]
        m_keys = list(poi_moving.keys())
        inter = [k for k in f_keys if k in m_keys]
        log.print(f_keys, verbose=verbose)
        log.print(poi_fixed.orientation, verbose=verbose)
        if len(inter) < 1:
            log.print("[!] No shared points, skip registration", Log_Type.FAIL)
            raise ValueError(
                f"[!] No shared points, skip registration; {poi_fixed.keys()=}; {poi_moving.keys()=}",
            )
        if len(inter) == 1:
            log.print(
                "[!] Only one shared point pair - fitting a pure translation (rotation is under-determined)",
                Log_Type.WARNING,
                verbose=verbose,
            )

        if leave_worst_percent_out != 0.0:
            poi_fixed_pruned = poi_fixed.intersect(poi_moving)
            _r, _t, _err_reg, _err_nat, delta_after = self._fit_from_keys(inter, poi_fixed_pruned, poi_moving, verbose=False, log=log)
            delta_sorted = sorted(delta_after.items(), key=lambda x: -x[1])
            drop_out = f"Did not use the following keys for registaiton (worst {leave_worst_percent_out * 100} %) "
            for i, key in enumerate(delta_sorted):
                if i >= len(delta_sorted) * leave_worst_percent_out:
                    break
                poi_fixed_pruned.remove_centroid_(key[0])
                drop_out += f"{key}, "
            log.print(drop_out, verbose=verbose)
            log.print("Error with all points", _err_reg, Log_Type.STAGE, verbose=verbose)
            poi_fixed = poi_fixed_pruned
            f_keys = [k for k in poi_fixed.keys() if k[0] not in exclusion]
            inter = [k for k in f_keys if k in m_keys]

        r, t, err_reg, err_nat, _ = self._fit_from_keys(inter, poi_fixed, poi_moving, verbose=verbose, log=log)

        # world_matrix: fixed_world -> moving_world (direction used for resampling)
        self.world_matrix = np.eye(4)
        self.world_matrix[:3, :3] = r
        self.world_matrix[:3, 3] = t

        self.target_grid: Has_Grid = poi_fixed.to_gird()
        self.input_grid: Has_Grid = poi_moving.to_gird()
        self.error_reg: float = err_reg
        self.error_natural: float = err_nat
        # Backwards-compat aliases (mirroring Point_Registration)
        self.out_poi: Has_Grid = self.target_grid
        self.input_poi: Has_Grid = self.input_grid

        self.transform: HomogeneousTransform = self._build_deepali_transform(self.target_grid.to_deepali_grid(align_corners))
        self.transform.to(self.device)

    @staticmethod
    def _fit_from_keys(
        inter: list[tuple[int, int]],
        poi_fixed: POI,
        poi_moving: POI,
        verbose: bool,
        log: Logger_Interface,
    ) -> tuple[np.ndarray, np.ndarray, float, float, dict[tuple[int, int], float]]:
        """Kabsch fit and legacy-compatible diagnostics."""
        # Filter out NaNs
        clean = []
        for k in inter:
            fp = poi_fixed[k]
            mp = poi_moving[k]
            if any(math.isnan(v) for v in fp) or any(math.isnan(v) for v in mp):
                continue
            clean.append(k)
        # One pair is legal - it degenerates to a pure translation (see _horn_rigid).
        assert len(clean) >= 1, f"To few points after NaN filter: {clean}"
        fw = _poi_world_ras(poi_fixed, clean)
        mw = _poi_world_ras(poi_moving, clean)
        # Horn: q = R p + t  with p=moving, q=fixed  -> moving→fixed
        # For resampling we need fixed→moving, i.e. the inverse.
        r_mov_to_fix, t_mov_to_fix = _horn_rigid(mw, fw)
        # Inverse: fixed→moving
        r = r_mov_to_fix.T
        t = -r @ t_mov_to_fix

        # residuals in world mm
        pred_fixed = (r_mov_to_fix @ mw.T).T + t_mov_to_fix
        err_vecs = fw - pred_fixed
        per_key = {k: float(np.sum(err_vecs[i] ** 2)) for i, k in enumerate(clean)}
        err_reg = float(np.mean(np.linalg.norm(err_vecs, axis=1))) if err_vecs.size else 0.0

        # error_natural: mean |d_fixed_i - d_moving_i| for consecutive pairs whose
        # first key differs by <50 (matches _compute_versor).
        err_natural_terms = []
        for i in range(1, len(clean)):
            (k1, _), (k1p, _) = clean[i], clean[i - 1]
            if abs(k1 - k1p) < 50:
                d_f = float(np.linalg.norm(fw[i] - fw[i - 1]))
                d_m = float(np.linalg.norm(mw[i] - mw[i - 1]))
                err_natural_terms.append(abs(d_f - d_m))
        err_nat = float(np.mean(err_natural_terms)) if err_natural_terms else 0.0

        log.print(f"[Deepali_Point_Registration] used {len(clean)} points", verbose=verbose)
        log.print(
            f"[Deepali_Point_Registration] avg residual: {err_reg: 7.3f} mm",
            Log_Type.STAGE,
            verbose=verbose,
        )
        return r, t, err_reg, err_nat, per_key

    def _build_deepali_transform(self, target_dgrid: Deepali_Grid) -> HomogeneousTransform:
        """Convert the world-space rigid to target-cube coordinates.

        The transform tensor stored inside the ``HomogeneousTransform`` lives in
        target-cube coordinates. Given the world matrix ``W`` (fixed→moving in
        LPS convention), the equivalent target-cube matrix is ``A^-1 @ W @ A``
        where ``A`` is the target grid's ``CUBE_CORNERS→WORLD`` transform.
        ``self.world_matrix`` is stored in NIfTI RAS convention, so it is
        converted to LPS first. The moving-grid conversion is done automatically
        by :class:`SampleImage` at sampling time and is not baked in.
        """
        axes_cube = Axes.CUBE_CORNERS if target_dgrid.align_corners() else Axes.CUBE
        a34 = target_dgrid.transform(axes_cube, Axes.WORLD)  # (3, 4)
        a = torch.eye(4, dtype=a34.dtype)
        a[:3, :4] = a34
        w_lps = _ras_to_lps_matrix(self.world_matrix)
        w = torch.as_tensor(w_lps, dtype=a34.dtype)
        m_full = torch.linalg.inv(a) @ w @ a  # (4, 4)
        m34 = m_full[:3, :].unsqueeze(0).contiguous()  # (1, 3, 4)
        transform = HomogeneousTransform(target_dgrid, params=False)
        transform.matrix_(m34)
        return transform

    # --- Compatibility API with Point_Registration ---
    def get_affine(self) -> np.ndarray:
        """Return the 4x4 LPS world matrix (fixed→moving direction)."""
        return self.world_matrix.copy()

    def apply(self, x: NII_or_POI) -> NII_or_POI:
        """Dispatch helper: forwards to :meth:`transform_nii` / :meth:`transform_poi`.

        ``self.transform`` is the DeepALI ``HomogeneousTransform`` module and is
        intentionally left as an attribute so it can be plugged into DeepALI
        pipelines directly.
        """
        if isinstance(x, POI):
            return self.transform_poi(x)  # type: ignore[return-value]
        if isinstance(x, NII):
            return self.transform_nii(x)  # type: ignore[return-value]
        raise ValueError(type(x))

    @property
    def deepali_transform(self) -> HomogeneousTransform:
        """The fitted DeepALI ``HomogeneousTransform`` on the target grid."""
        return self.transform

    @torch.no_grad()
    def transform_nii(
        self,
        moving_img_nii: NII,
        allow_only_same_grid_as_moving: bool = True,
        output_space: Has_Grid | None = None,
        c_val: float | None = None,
        align_corners: bool | None = None,
        gpu: int | None = None,
        ddevice: DEVICES | None = None,
    ) -> NII:
        """Resample a moving NII into the fixed (or *output_space*) grid."""
        if allow_only_same_grid_as_moving:
            text = (
                "input image must be in the same space as moving. If you are sure that this input "
                "is in same space as the moving image you can turn of 'allow_only_same_grid_as_moving'"
            )
            moving_img_nii.assert_affine(self.input_grid, text=text, shape_tolerance=0.9)
        if c_val is None:
            c_val = moving_img_nii.get_c_val()
        align_corners = self.align_corners if align_corners is None else align_corners
        device = get_device(ddevice, 0 if gpu is None else gpu) if ddevice is not None else self.device

        out_grid_nii = output_space if output_space is not None else self.target_grid
        out_dgrid = out_grid_nii.to_deepali_grid(align_corners)
        source_dgrid = moving_img_nii.to_gird().to_deepali_grid(align_corners)

        # Rebuild transform on output grid if different from stored target
        transform = self._build_deepali_transform(out_dgrid).to(device) if output_space is not None else self.transform.to(device)

        warp = TransformImage(
            target=out_dgrid,
            source=source_dgrid,
            sampling=Sampling.NEAREST if moving_img_nii.seg else Sampling.LINEAR,
            padding=math.floor(c_val) if not moving_img_nii.seg else 0,
        ).to(device)
        src = moving_img_nii.to_deepali(align_corners=align_corners, device=device)
        data = warp(transform.tensor(), src)
        data = data.squeeze()
        data = data.permute(*torch.arange(data.ndim - 1, -1, -1))  # type: ignore
        out = out_grid_nii.make_nii(data.detach().cpu().numpy(), moving_img_nii.seg)
        if moving_img_nii.seg:
            out.set_dtype_("smallest_uint")
        return out

    def transform_poi(
        self,
        poi_moving: POI,
        allow_only_same_grid_as_moving: bool = True,
        output_space: Has_Grid | None = None,
    ) -> POI:
        """Transform landmarks from moving into fixed (or *output_space*) space."""
        if allow_only_same_grid_as_moving:
            text = (
                "input image must be in the same space as moving. If you are sure that this input "
                "is in same space as the moving image you can turn of 'allow_only_same_grid_as_moving'"
            )
            poi_moving.assert_affine(self.input_grid, text=text)
        out_grid = output_space if output_space is not None else self.target_grid
        # Direct: mov_world -> fix_world uses world_matrix^-1 (world_matrix is fixed→moving)
        w_inv = np.linalg.inv(self.world_matrix)
        out = {}
        for key, key2, cord in poi_moving.items():
            mov_world = np.array(poi_moving.local_to_global(cord))
            hp = np.append(mov_world, 1.0)
            fix_world = (w_inv @ hp)[:3]
            out[key, key2] = out_grid.global_to_local(tuple(fix_world))  # type: ignore[attr-defined]
        return out_grid.make_empty_POI(out)  # type: ignore[attr-defined]

    def transform_poi_inverse(
        self,
        poi_fixed: POI,
        allow_only_same_grid_as_moving: bool = True,
        output_space: Has_Grid | None = None,
    ) -> POI:
        """Inverse of :meth:`transform_poi` — from fixed into moving space."""
        if allow_only_same_grid_as_moving:
            text = (
                "input image must be in the same space as fixed. If you are sure that this input "
                "is in same space as the fixed image you can turn of 'allow_only_same_grid_as_moving'"
            )
            poi_fixed.assert_affine(self.target_grid, text=text)
        out_grid = output_space if output_space is not None else self.input_grid
        w = self.world_matrix
        out = {}
        for key, key2, cord in poi_fixed.items():
            fix_world = np.array(poi_fixed.local_to_global(cord))
            hp = np.append(fix_world, 1.0)
            mov_world = (w @ hp)[:3]
            out[key, key2] = out_grid.global_to_local(tuple(mov_world))  # type: ignore[attr-defined]
        return out_grid.make_empty_POI(out)  # type: ignore[attr-defined]

    def transform_cord(self, cord: tuple[float, ...]) -> np.ndarray:
        """Transform a single voxel coord from moving to fixed space."""
        mov_world = np.array(self.input_grid.local_to_global(cord))  # type: ignore[attr-defined]
        hp = np.append(mov_world, 1.0)
        fix_world = (np.linalg.inv(self.world_matrix) @ hp)[:3]
        return np.array(self.target_grid.global_to_local(tuple(fix_world)))  # type: ignore[attr-defined]

    def transform_cord_inverse(self, cord: tuple[float, ...]) -> np.ndarray:
        """Transform a single voxel coord from fixed to moving space."""
        fix_world = np.array(self.target_grid.local_to_global(cord))  # type: ignore[attr-defined]
        hp = np.append(fix_world, 1.0)
        mov_world = (self.world_matrix @ hp)[:3]
        return np.array(self.input_grid.global_to_local(tuple(mov_world)))  # type: ignore[attr-defined]

    # --- Serialisation ---
    def get_dump(self) -> tuple:
        return (
            1,
            self.target_grid,
            self.input_grid,
            self.world_matrix,
            self.error_reg,
            self.error_natural,
            self.align_corners,
        )

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as w:
            pickle.dump(self.get_dump(), w)

    @classmethod
    def load(cls, path: str | Path, gpu: int = 0, ddevice: DEVICES = "cuda") -> Deepali_Point_Registration:
        with open(path, "rb") as f:
            return cls.load_(pickle.load(f), gpu=gpu, ddevice=ddevice)

    @classmethod
    def load_(cls, w: tuple, gpu: int = 0, ddevice: DEVICES = "cuda") -> Deepali_Point_Registration:
        version, target_grid, input_grid, world_matrix, err_reg, err_nat, align_corners = w
        assert version == 1, f"Version mismatch {version=}"
        self = cls.__new__(cls)
        self.target_grid = target_grid
        self.input_grid = input_grid
        self.out_poi = target_grid
        self.input_poi = input_grid
        self.world_matrix = world_matrix
        self.error_reg = err_reg
        self.error_natural = err_nat
        self.align_corners = align_corners
        self.device = get_device(ddevice, gpu)
        self.transform = self._build_deepali_transform(target_grid.to_deepali_grid(align_corners))
        self.transform.to(self.device)
        return self


def ridged_points_from_poi_deepali(
    poi_fixed: POI,
    poi_moving: POI,
    exclusion: list | None = None,
    log: Logger_Interface = No_Logger(),  # noqa: B008
    verbose: bool = True,
    ax_code=None,
    zooms=None,
    leave_worst_percent_out: float = 0.0,
    gpu: int = 0,
    ddevice: DEVICES = "cuda",
) -> Deepali_Point_Registration:
    """DeepALI counterpart of :func:`ridged_points_from_poi`."""
    return Deepali_Point_Registration(
        poi_fixed,
        poi_moving,
        exclusion=exclusion,
        log=log,
        verbose=verbose,
        ax_code=ax_code,
        zooms=zooms,
        leave_worst_percent_out=leave_worst_percent_out,
        gpu=gpu,
        ddevice=ddevice,
    )


def ridged_points_from_subreg_vert_deepali(
    poi_moving: POI_Reference,
    vert: Image_Reference,
    subreg: POI_Reference,
    poi_target_buffer: Path | str | None = None,
    orientation=None,
    zoom: tuple[float, float, float] = (-1, -1, -1),
    subreg_id: int | Location | list[int | Location] | list[Location] | list[int] = 50,
    verbose: bool = True,
    save_buffer_file: bool = True,
    gpu: int = 0,
    ddevice: DEVICES = "cuda",
) -> Deepali_Point_Registration:
    """DeepALI counterpart of :func:`ridged_points_from_subreg_vert`."""
    if not isinstance(subreg_id, (list, tuple)):
        subreg_id = [subreg_id]
    instance_nii = to_nii(vert, True).copy()
    semantic_nii = to_nii(subreg, True).copy()
    target_poi = (
        calc_poi_from_subreg_vert(
            instance_nii,
            semantic_nii,
            subreg_id=subreg_id,
            buffer_file=poi_target_buffer,
            save_buffer_file=save_buffer_file,
        )
        .copy()
        .extract_subregion_(*subreg_id)
    )
    if orientation is not None:
        target_poi.reorient_(orientation)
    if zoom != (-1, -1, -1):
        target_poi.rescale_(zoom)
    moving_poi = POI.load(poi_moving)
    return ridged_points_from_poi_deepali(
        target_poi,
        moving_poi,
        verbose=verbose,
        gpu=gpu,
        ddevice=ddevice,
    )
