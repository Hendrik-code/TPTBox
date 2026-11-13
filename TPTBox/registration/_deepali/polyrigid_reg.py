"""Polyrigid registration: "Deformations Incorporating Rigid Structures" (Little, Hill, Hawkes; CVIU 66(2), 1997).

Every vertebra moves as a rigid body, the rest of the volume deforms smoothly, and landmarks outside the
vertebrae are interpolated exactly. In contrast to :class:`Rigid_Elements_Registration` nothing is optimised:

1. a closed-form rigid fit (Horn/Kabsch) per vertebra from corresponding POIs gives ``L_i`` (fixed -> moving),
2. the weights ``w_i`` (eq. 17) blend the *linear transforms* ``L(t) = sum_i w_i(t) L_i`` (eq. 19),
3. one linear system gives the coefficients ``B`` of the distance-weighted basis function (eq. 20),
4. ``f(t) = L(t) g(t) + sum_j B_j D0(t) D0(t_j) |t - t_j|`` (eq. 23) is evaluated in chunks with torch.

Conventions
-----------
* Direction is fixed -> moving. ``f`` maps a point of the fixed grid to its position in the moving image, so
  :meth:`Polyrigid_Registration.transform_nii` samples the moving image at ``f(t)``.
* All maths is done in world RAS millimetres (never voxel units), so anisotropic spacing is handled correctly.
* Warping uses Deepali (`Grid.transform_points` + `grid_sample`); the torch device is selectable (``ddevice="cuda"`` on a GPU node, default ``"cpu"``).
"""

from __future__ import annotations

import dataclasses
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from deepali.core import Axes, Sampling
from deepali.core.functional import grid_sample
from scipy.ndimage import distance_transform_edt, map_coordinates

from TPTBox import NII, POI, Image_Reference, Location, Vertebra_Instance, calc_poi_from_subreg_vert, to_nii
from TPTBox.core.internal.deep_learning_utils import DEVICES, get_device
from TPTBox.registration._deepali.polyrigid_optim import OptimLog, PolyrigidOptimConfig, centroid_world, refine_rigid
from TPTBox.registration._ridged_points.deepali_point_registration import _horn_rigid

RIGID_SUBREGS = (Location.Vertebra_Corpus, Location.Spinosus_Process, Location.Arcus_Vertebrae)
_EPS = 1e-6


def default_rigid_ids() -> list[int]:
    """Vertebrae C1-L6 and T13 plus the sacrum/coccyx labels (S1-S6, COCC): all are rigid bones. Only labels present in both images are used."""
    return [*Vertebra_Instance.vertebra_label_without_sacrum(), *(int(v.value) for v in Vertebra_Instance.sacrum())]


def _world(poi: POI, keys: Sequence[tuple[int, int]]) -> np.ndarray:
    """World-RAS coordinates ``(N, 3)`` of the given POI keys."""
    if len(keys) == 0:
        return np.zeros((0, 3))
    return poi.local_to_global_arr(np.array([poi[k] for k in keys], dtype=float))


def _to_voxel(affine: np.ndarray, world: np.ndarray) -> np.ndarray:
    """World RAS ``(N, 3)`` -> (fractional) voxel coordinates of a grid with the given affine."""
    inv = np.linalg.inv(affine)
    return world @ inv[:3, :3].T + inv[:3, 3]


def _inside_seg(seg: np.ndarray, affine: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Boolean ``(N,)``: is the point inside a non-zero voxel of ``seg``? Points outside the array count as outside."""
    vox = np.rint(_to_voxel(affine, world)).astype(int)
    ok = np.all((vox >= 0) & (vox < np.array(seg.shape)), axis=1)
    out = np.zeros(len(world), dtype=bool)
    out[ok] = seg[tuple(vox[ok].T)] != 0
    return out


class Polyrigid_Registration:
    """Polyrigid registration of a moving spine onto a fixed spine (rigid vertebrae, smooth blend in between).

    Per vertebra the rigid transform is initialised in closed form with whatever exists, then optionally refined:

    * ``>= 3`` shared, non-collinear POIs (given, or computed from the sub-region masks) -> Horn fit (rotation + translation),
    * 1-2 shared POIs -> translation only (mean offset), rotation = identity,
    * masks only -> translation only (offset of the mask centroids), rotation = identity.

    Args:
        fixed_vert / moving_vert: vertebra instance segmentations. ``fixed_vert`` defines the target grid.
        fixed_subreg / moving_subreg: optional sub-region segmentations. Only needed to compute POIs.
        fixed_poi / moving_poi: optional pre-computed POIs.
        init: ``"auto"`` (rule above), ``"poi"`` (>= 3 POIs required, other vertebrae are dropped) or ``"centroid"``
            (ignore POIs for the rigid init).
        refine_independent: stage A, per-object optimisation on the mask overlap. Accepts
            * ``True`` - refine every rigid object;
            * ``False`` - refine none;
            * ``None`` - refine every rigid object iff at least one has a non-Horn init (the
              historical behaviour, equivalent to ``True`` whenever any object fell back to a
              translation or centroid init);
            * ``"non_horn"`` - refine only the objects whose init kind is not ``"horn"``, i.e.
              those that had an under-determined POI init (0-2 shared non-collinear POIs). Use
              this when a well-determined POI fit should be preserved for most objects but a
              few under-determined ones should still be lifted by the mask overlap.
            * ``Sequence[int]`` - refine exactly these object ids.
        joint_refine: stage B, optimise all rigid bodies together through the blended field, with a folding penalty and
            an incompressibility penalty for non-rigid labelled structures (``optim.compressibility`` lowers the weight
            for structures that may compress).
        nonrigid_ids: vertebrae that may change their shape (compressed / fractured). ``None`` = all rigid; ``"all"`` = every
            vertebra (not the sacrum); ``"auto"`` = vertebrae that a first fit (all vertebrae may change shape) shows to be scaled by more than
            ``optim.shape_tol`` (default 5 %) after the fit in step 1; the others stay rigid (follow-ups of one subject); or a list of ids. These first
            run an independent stage with a bounded SI scale and isotropic scale (``optim.scale_range``, ``optim.w_scale``); the
            shape is then frozen and they are stiff again in the following stages. The scales are in ``self.shape_scales``.
        optim: :class:`PolyrigidOptimConfig` (weights, learning rates, steps, working-grid spacing ``levels_mm``).
        rigid_subregs: sub-regions whose centroids define the per-vertebra POI fit.
        landmarks: use landmarks outside the bone (``None`` = only if ``landmark_subregs`` is given). Default: none, ``B = 0``.
        landmark_subregs: extra sub-regions computed as landmark candidates (POIs inside a vertebra are filtered out).
        min_dist_mm: points closer than this to a mask surface are excluded, both from landmarks and from the
            penalty points of the joint stage.
        vert_ids: ids that may be rigid objects (only those present in both images are used). Default
            :func:`default_rigid_ids`: C1-L6, T13 (= 28) and the sacrum/coccyx labels S1-S6, COCC (26, 29-33, 27). The sacrum is a rigid
            bone: leaving it out makes it a non-rigid structure that is only carried by the blend of the neighbouring vertebrae.
        my: exponent of the inverse-distance weights (paper: 1.5).
        ridge: Tikhonov term added to ``S'`` (0 = exact interpolation of the landmarks).
        crop_margin_vox: crop the fixed grid to the vertebra bounding box plus this margin (``None``: no crop).
        reference: explicit output grid (any ``Has_Grid`` / image reference). Overrides ``crop_margin_vox``; use it to put several
            registrations on the same grid.
        ddevice / gpu: torch device used for optimisation and the dense field. Default CPU.
        auto_fit: run :meth:`fit` in the constructor.
    """

    def __init__(
        self,
        fixed_vert: Image_Reference,
        fixed_subreg: Image_Reference | None,
        moving_vert: Image_Reference,
        moving_subreg: Image_Reference | None,
        fixed_poi: POI | None = None,
        moving_poi: POI | None = None,
        init: str = "auto",
        refine_independent: bool  | Literal["non_horn"] | Sequence[int]|None = "non_horn",
        joint_refine: bool = False,
        nonrigid_ids: str | Sequence[int] | None = None,
        optim: PolyrigidOptimConfig | None = None,
        rigid_subregs: Sequence[Location] = RIGID_SUBREGS,
        landmarks: bool | None = None,
        landmark_subregs: Sequence[Location] = (),
        min_dist_mm: float = 2.0,
        my: float = 1.5,
        ridge: float = 0.0,
        crop_margin_vox: int | None = 20,
        reference: Image_Reference | None = None,
        ddevice: DEVICES = "cpu",
        gpu: int | None = None,
        vert_ids: Sequence[int] | None = None,
        auto_fit: bool = True,
        verbose: bool = True,
    ) -> None:
        assert init in ("auto", "poi", "centroid"), init
        self.fixed_vert = to_nii(fixed_vert, True)
        self.fixed_subreg = None if fixed_subreg is None else to_nii(fixed_subreg, True)
        self.moving_vert = to_nii(moving_vert, True)
        self.moving_subreg = None if moving_subreg is None else to_nii(moving_subreg, True)
        self.fixed_poi = fixed_poi
        self.moving_poi = moving_poi
        self.init = init
        self.refine_independent = refine_independent
        self.joint_refine = joint_refine
        self.nonrigid_ids = nonrigid_ids
        self.shape_scales: dict[int, tuple[float, float]] = {}  # vertebra id -> (SI scale, isotropic scale)
        self.shape_flagged: list[int] = []
        self.optim = optim or PolyrigidOptimConfig()
        self.rigid_subregs = [Location(int(s)) if not isinstance(s, Location) else s for s in rigid_subregs]
        self.landmark_subregs = list(landmark_subregs)
        self.use_landmarks = bool(self.landmark_subregs) if landmarks is None else landmarks
        self.my = my
        self.ridge = ridge
        self.min_dist_mm = min_dist_mm
        self.crop_margin_vox = crop_margin_vox
        self.reference = None if reference is None else to_nii(reference, True)
        self.device = get_device(ddevice, 0 if gpu is None else gpu)
        self.vert_ids = list(default_rigid_ids() if vert_ids is None else vert_ids)
        self.verbose = verbose
        self.fitted = False
        # filled by fit()
        self.ref: NII
        self.ids: list[int] = []
        self.init_kinds: dict[int, str] = {}  # vertebra id -> "horn" | "points" | "centroid"
        self.optim_log: OptimLog = OptimLog()
        self.L: np.ndarray = np.zeros((0, 3, 4))  # per vertebra [t | R], fixed world -> moving world
        self.landmarks_fixed = np.zeros((0, 3))  # t_j (world mm)
        self.landmarks_moving = np.zeros((0, 3))  # u_j (world mm)
        self.landmark_keys: list[tuple[int, int]] = []
        self.B = np.zeros((0, 3))
        self._d0_land = np.zeros(0)
        self._Lfield: np.ndarray | None = None  # (X, Y, Z, 12) blended [t | R]
        self._D0: np.ndarray | None = None  # (X, Y, Z) distance to the union of vertebrae in mm
        self._vert_arr: np.ndarray | None = None
        self._field: np.ndarray | None = None
        if auto_fit:
            self.fit()

    # ------------------------------------------------------------------ logging
    def _log(self, *a):
        if self.verbose:
            print("[Polyrigid]", *a)

    # ------------------------------------------------------------------ fitting
    def _compute_pois(self) -> tuple[POI | None, POI | None]:
        """Given POIs, or POIs computed from the sub-region masks (``None`` if neither is available)."""
        need = self.init != "centroid" or self.use_landmarks
        subregs = list(dict.fromkeys([*self.rigid_subregs, *(self.landmark_subregs if self.use_landmarks else [])]))
        fp, mp = self.fixed_poi, self.moving_poi
        if need and fp is None and self.fixed_subreg is not None:
            fp = calc_poi_from_subreg_vert(self.fixed_vert, self.fixed_subreg, subreg_id=subregs)
        if need and mp is None and self.moving_subreg is not None:
            mp = calc_poi_from_subreg_vert(self.moving_vert, self.moving_subreg, subreg_id=subregs)
        return (fp, mp) if (need and fp is not None and mp is not None) else (None, None)

    @staticmethod
    def _anatomical(ids: Sequence[int]) -> list[int]:
        """Cranial -> caudal (T13 = 28 sits between T12 and L1, so a numeric sort would be wrong)."""
        order = Vertebra_Instance.order_dict()
        return sorted(ids, key=lambda i: (order.get(i, 10_000), i))

    def _init_rigid(self, shared: Sequence[int], fp: POI | None, mp: POI | None):
        """Closed-form init per vertebra. Returns ``ids, R0 (n,3,3), t0 (n,3), c_fix (n,3), used_keys``."""
        rigid = sorted(int(s.value) for s in self.rigid_subregs)
        f_arr, m_arr = self.fixed_vert.get_seg_array(), self.moving_vert.get_seg_array()
        ids: list[int] = []
        R0, t0, cf, used_keys = [], [], [], set()
        for v in shared:
            kind = "centroid"
            keys = [(v, s) for s in rigid if fp is not None and mp is not None and (v, s) in fp and (v, s) in mp]
            p = q = None
            if self.init != "centroid" and len(keys) > 0:
                p, q = _world(fp, keys), _world(mp, keys)  # type: ignore[arg-type]
                collinear = len(keys) < 3 or np.linalg.matrix_rank(p - p.mean(0), tol=1e-3) < 2
                kind = "points" if collinear else "horn"
            if self.init == "poi" and kind != "horn":
                self._log(f"vertebra {v}: needs >= 3 non-collinear shared POIs for init='poi' -> not treated as rigid object")
                continue
            c_f = centroid_world(f_arr, self.fixed_vert.affine, v)
            if kind == "horn":
                r, t = _horn_rigid(p, q)
                used_keys.update(keys)
            elif kind == "points":
                r, t = np.eye(3), (q - p).mean(0)
                used_keys.update(keys)
            else:
                r, t = np.eye(3), centroid_world(m_arr, self.moving_vert.affine, v) - c_f
            ids.append(v)
            R0.append(r)
            t0.append(t)
            cf.append(c_f)
            self.init_kinds[v] = kind
        return ids, np.array(R0), np.array(t0), np.array(cf), used_keys

    @staticmethod
    def _si_axes(ids: Sequence[int], cen: np.ndarray) -> np.ndarray:
        """Unit cranio-caudal axis per vertebra from the neighbouring centroids (cranial minus caudal); world z as fallback."""
        out = np.tile([0.0, 0.0, 1.0], (len(ids), 1))
        for i in range(len(ids)):
            d = cen[max(i - 1, 0)] - cen[min(i + 1, len(ids) - 1)]
            if len(ids) > 1 and np.linalg.norm(d) > 1e-6:
                out[i] = d / np.linalg.norm(d)
        return out

    def _select_shape_ids(self, ids: Sequence[int]) -> list[int]:
        """Vertebrae that may change their shape, by ``nonrigid_ids`` (``auto``: all candidates, reduced after the fit)."""
        sel = self.nonrigid_ids
        sacral = {int(v.value) for v in Vertebra_Instance.sacrum()}
        if sel is None:
            return []
        if isinstance(sel, str):
            assert sel in ("auto", "all"), sel
            cand = [v for v in ids if v not in sacral]
            if sel == "all":
                return cand
            return cand  # "auto": all are fitted in step 1, those with a negligible scale are reset to rigid (optim.shape_tol)
        return [int(v) for v in sel if int(v) in ids]

    def fit(self) -> Polyrigid_Registration:
        self.init_kinds = {}
        fp, mp = self._compute_pois()
        allowed = set(self.vert_ids)
        shared = self._anatomical(
            {int(i) for i in self.fixed_vert.unique() if i in allowed} & {int(i) for i in self.moving_vert.unique() if i in allowed}
        )
        # 1) closed-form init per vertebra (fixed -> moving, world RAS)
        ids, R0, t0, c_fix, used_keys = self._init_rigid(shared, fp, mp)
        if not ids:
            raise ValueError("No shared vertebra to register (check vert_ids / the segmentations / init).")
        self._log("init:", {v: self.init_kinds[v] for v in ids})
        self.ids = ids

        # 2) target grid (only the vertebrae that are rigid objects), optional crop
        ref = self.fixed_vert.extract_label(ids, keep_label=True)
        crop = None
        if self.reference is not None:
            ref = ref.resample_from_to(self.reference, mode="nearest", verbose=False)
        elif self.crop_margin_vox is not None:
            crop = ref.compute_crop(dist=self.crop_margin_vox)
            ref = ref.apply_crop(crop)
        self.ref = ref
        self._vert_arr = ref.get_seg_array().astype(np.int32)

        # 3) optional refinement from the masks (stage A independent, stage B joint through the blended field)
        axes = self._si_axes(ids, c_fix)
        self.shape_flagged = self._select_shape_ids(ids)
        self.shape_scales = {}
        if self.shape_flagged:
            self._log("vertebrae that may change shape:", self.shape_flagged)
        refine_ids = self._resolve_refine_ids(ids)
        if self.nonrigid_ids == "auto" and self.refine_independent is None:
            # Reset vertebrae need to be re-fitted rigidly; same historical override.
            refine_ids = list(ids)
            if self.optim.shape_tol <= 0:
                self.optim = dataclasses.replace(self.optim, shape_tol=0.05)
        L = np.concatenate([t0[:, :, None], R0], axis=2)
        if refine_ids or self.joint_refine or self.shape_flagged:
            L, self.optim_log = self._refine(ids, R0, t0, c_fix, crop, refine_ids, axes)
            self.shape_scales = dict(self.optim_log.shape)
            self.shape_flagged = list(self.shape_scales)
            self._log("optim:", self.optim_log.summary())
        self.L = L
        self._precompute_fields()

        # 4) landmarks: shared, not used for the rigid fit, outside of the bone (default: none -> B = 0)
        if self.use_landmarks and fp is not None and mp is not None:
            self._select_landmarks(fp, mp, used_keys)
        else:
            self._set_landmarks([], np.zeros((0, 3)), np.zeros((0, 3)))
        self._solve_B()
        self.fitted = True
        self._field = None
        return self

    def _resolve_refine_ids(self, ids: Sequence[int]) -> list[int]:
        """Translate ``refine_independent`` into the explicit list of ids to refine."""
        ri = self.refine_independent
        if ri is True:
            return list(ids)
        if ri is False:
            return []
        if ri is None:
            return list(ids) if any(k != "horn" for k in self.init_kinds.values()) else []
        if isinstance(ri, str):
            assert ri == "non_horn", f"refine_independent string must be 'non_horn', got {ri!r}"
            return [v for v in ids if self.init_kinds.get(v) != "horn"]
        rset = {int(v) for v in ri}
        return [v for v in ids if v in rset]

    def _refine(self, ids, R0, t0, c_fix, crop, refine_ids: Sequence[int], axes=None):
        """Stage A / B of :mod:`polyrigid_optim` on the cropped fixed grid and a cropped moving label map.

        ``refine_ids`` names which rigid bodies take part in stage A. When it's a strict subset
        of ``ids`` the non-refined rows of ``L`` keep the closed-form POI init, and stage A runs
        only on the subset. The joint stage B still uses every vertebra, because its field is
        linear in all of them.
        """
        assert self.ref is not None
        refine_ids = list(refine_ids)
        subset = len(refine_ids) > 0 and len(refine_ids) < len(ids)
        independent = len(refine_ids) > 0  # anything to refine at all

        fixed_other = None
        if self.joint_refine and self.optim.w_incomp > 0:  # non-rigid labelled structures (ribs, discs, ...)
            arr = self.fixed_vert.get_seg_array().copy()
            arr[np.isin(arr, ids)] = 0
            other = self.fixed_vert.set_array(arr)
            if self.reference is not None:
                fixed_other = other.resample_from_to(self.reference, mode="nearest", verbose=False)
            else:
                fixed_other = other.apply_crop(crop) if crop is not None else other

        if not subset:
            mv = self.moving_vert.extract_label(ids, keep_label=True)
            margin = int(np.ceil(30.0 / float(min(mv.zoom))))
            mv = mv.apply_crop(mv.compute_crop(dist=margin))
            return refine_rigid(
                self.ref, mv, ids, R0, t0, c_fix, self.optim, self.device,
                independent=independent, joint=self.joint_refine, fixed_other=fixed_other,
                shape_ids=self.shape_flagged, axes=axes,
                my=self.my, min_dist_mm=self.min_dist_mm, verbose=self.verbose,
            )  # fmt: skip

        # Subset path: stage A on refine_ids, then (optional) stage B on all ids with the
        # subset's refined rows already spliced in.
        L_full = np.concatenate([t0[:, :, None], R0], axis=2)
        idx_map = {v: i for i, v in enumerate(ids)}
        sub_idx = [idx_map[v] for v in refine_ids]

        mv_sub = self.moving_vert.extract_label(refine_ids, keep_label=True)
        margin = int(np.ceil(30.0 / float(min(mv_sub.zoom))))
        mv_sub = mv_sub.apply_crop(mv_sub.compute_crop(dist=margin))
        fixed_sub = self.ref.extract_label(refine_ids, keep_label=True)
        shape_sub = [v for v in self.shape_flagged if v in refine_ids]
        axes_sub = None if axes is None else axes[sub_idx]
        L_sub, log = refine_rigid(
            fixed_sub, mv_sub, refine_ids,
            R0[sub_idx], t0[sub_idx], c_fix[sub_idx], self.optim, self.device,
            independent=True, joint=False, fixed_other=None,
            shape_ids=shape_sub, axes=axes_sub,
            my=self.my, min_dist_mm=self.min_dist_mm, verbose=self.verbose,
        )
        for k, i in enumerate(sub_idx):
            L_full[i] = L_sub[k]

        if not self.joint_refine:
            return L_full, log

        # Stage B with every vertebra, starting from the spliced L.
        R_all = L_full[:, :, 1:].copy()
        t_all = L_full[:, :, 0].copy()
        mv_all = self.moving_vert.extract_label(ids, keep_label=True)
        margin_all = int(np.ceil(30.0 / float(min(mv_all.zoom))))
        mv_all = mv_all.apply_crop(mv_all.compute_crop(dist=margin_all))
        L_all, log2 = refine_rigid(
            self.ref, mv_all, ids, R_all, t_all, c_fix, self.optim, self.device,
            independent=False, joint=True, fixed_other=fixed_other,
            shape_ids=self.shape_flagged, axes=axes,
            my=self.my, min_dist_mm=self.min_dist_mm, verbose=self.verbose,
        )
        # merge logs: keep subset-stage-A entries and joint-stage-B entries.
        log.stages = [*log.stages, *log2.stages]
        log.shape.update(log2.shape)
        return L_all, log

    def _precompute_fields(self) -> None:
        """Distances (mm), weights and the blended linear field L(t) on the fixed grid (eqs 13–19)."""
        assert self._vert_arr is not None
        arr = self._vert_arr
        zoom = tuple(float(z) for z in self.ref.zoom)
        # D0 is the distance to the union of all rigid objects (0 inside).
        self._D0 = distance_transform_edt(arr == 0, sampling=zoom).astype(np.float32)  # type: ignore[assignment]
        num = np.zeros((*arr.shape, 12), dtype=np.float32)
        q_sum = np.zeros(arr.shape, dtype=np.float32)
        for i, v in enumerate(self.ids):
            d = distance_transform_edt(arr != v, sampling=zoom)
            q = np.zeros_like(d, dtype=np.float32)
            nz = d > 0
            q[nz] = d[nz] ** (-self.my)
            num += q[..., None] * self.L[i].reshape(1, 1, 1, 12).astype(np.float32)
            q_sum += q
        num /= np.maximum(q_sum, _EPS)[..., None]  # no NaN where all q are 0
        for i, v in enumerate(self.ids):  # w_i = 1 inside object i, 0 inside the others
            num[arr == v] = self.L[i].reshape(12)
        self._Lfield = num

    def _interp(self, field: np.ndarray, world: np.ndarray) -> np.ndarray:
        """Trilinear lookup of a gridded (X,Y,Z[,C]) field at world positions (nodes are exact)."""
        vox = _to_voxel(self.ref.affine, world).T
        if field.ndim == 3:
            return map_coordinates(field, vox, order=1, mode="nearest")
        return np.stack([map_coordinates(field[..., c], vox, order=1, mode="nearest") for c in range(field.shape[-1])], axis=-1)

    def _L_at(self, world: np.ndarray) -> np.ndarray:
        assert self._Lfield is not None
        return self._interp(self._Lfield, world).reshape(-1, 3, 4)

    def _D0_at(self, world: np.ndarray) -> np.ndarray:
        assert self._D0 is not None
        return self._interp(self._D0, world)

    def _select_landmarks(self, fp: POI, mp: POI, used_keys: set[tuple[int, int]]) -> None:
        """Keep POIs present in both images that lie outside every vertebra (D0(t_i) != 0 is required by the paper)."""
        keys = [k for k in fp.keys() if k in mp and k not in used_keys]
        n0 = len(keys)
        if n0 == 0:
            self._set_landmarks([], np.zeros((0, 3)), np.zeros((0, 3)))
            self._log("no landmark candidates -> pure blend of the rigid motions (B = 0)")
            return
        t, u = _world(fp, keys), _world(mp, keys)
        assert self._vert_arr is not None
        keep = np.ones(n0, dtype=bool)
        keep &= ~_inside_seg(self._vert_arr, self.ref.affine, t)  # inside a fixed vertebra
        keep &= ~_inside_seg(self.moving_vert.get_seg_array(), self.moving_vert.affine, u)  # inside a moving vertebra
        vox = _to_voxel(self.ref.affine, t)
        keep &= np.all((vox >= 0) & (vox <= np.array(self.ref.shape) - 1), axis=1)  # outside the (cropped) grid
        d0 = np.where(keep, self._D0_at(t), 0.0)
        keep &= d0 >= self.min_dist_mm
        # duplicates (identical fixed positions) make S' singular
        _, first = np.unique(np.round(t, 3), axis=0, return_index=True)
        dup = np.ones(n0, dtype=bool)
        dup[first] = False
        keep &= ~dup
        self._log(
            f"landmarks: {int(keep.sum())}/{n0} kept ({n0 - int(keep.sum())} inside segmentation / too close / outside grid / duplicate)"
        )
        self._set_landmarks([k for k, m in zip(keys, keep) if m], t[keep], u[keep])

    def _set_landmarks(self, keys: list[tuple[int, int]], t: np.ndarray, u: np.ndarray) -> None:
        self.landmark_keys = keys
        self.landmarks_fixed = np.asarray(t, dtype=float).reshape(-1, 3)
        self.landmarks_moving = np.asarray(u, dtype=float).reshape(-1, 3)

    def _solve_B(self) -> None:
        n = len(self.landmarks_fixed)
        if n == 1:
            # S' = [0] for a single point (|h| kernel has a zero diagonal) -> cannot be interpolated exactly.
            self._log("only 1 landmark: the |h| basis needs >= 2 distinct points -> landmark ignored")
            self._set_landmarks([], np.zeros((0, 3)), np.zeros((0, 3)))
            n = 0
        if n == 0:
            self.B = np.zeros((0, 3))
            self._d0_land = np.zeros(0)
            return
        t = self.landmarks_fixed
        d0 = self._D0_at(t)
        L = self._L_at(t)
        T = np.einsum("nij,nj->ni", L[..., 1:], t) + L[..., 0]  # L(t_i) g(t_i)
        h = np.linalg.norm(t[:, None] - t[None], axis=-1)  # |t_i - t_j|, 3D => alpha = 1/2 (eq 16)
        S = d0[:, None] * d0[None] * h + self.ridge * np.eye(n)
        try:
            self.B = np.linalg.solve(S, self.landmarks_moving - T)
        except np.linalg.LinAlgError:
            self._log("S' is singular -> least-squares solution (consider ridge > 0)")
            self.B = np.linalg.lstsq(S, self.landmarks_moving - T, rcond=None)[0]
        self._d0_land = d0

    # ------------------------------------------------------------------ evaluation
    def _f_world(self, x: np.ndarray, Lx: np.ndarray, d0x: np.ndarray, device: torch.device | None = None) -> np.ndarray:
        """f(x) for world points ``x (N,3)`` given interpolated L(x) ``(N,3,4)`` and D0(x) ``(N,)`` (eq 23)."""
        out = np.einsum("nij,nj->ni", Lx[..., 1:], x) + Lx[..., 0]
        if len(self.landmarks_fixed) == 0:
            return out
        device = device or self.device
        t = torch.as_tensor(self.landmarks_fixed, dtype=torch.float32, device=device)
        wB = torch.as_tensor(self._d0_land[:, None] * self.B, dtype=torch.float32, device=device)  # D0(t_j) B_j
        chunk = 200_000
        for s in range(0, len(x), chunk):
            xs = torch.as_tensor(x[s : s + chunk], dtype=torch.float32, device=device)
            phi = torch.cdist(xs, t)  # |x - t_j|
            out[s : s + chunk] += (
                (torch.as_tensor(d0x[s : s + chunk], dtype=torch.float32, device=device)[:, None] * (phi @ wB)).cpu().numpy()
            )
        return out

    def _grid_world(self) -> np.ndarray:
        idx = np.stack(np.meshgrid(*[np.arange(s, dtype=np.float32) for s in self.ref.shape], indexing="ij"), -1).reshape(-1, 3)
        a = self.ref.affine
        return idx @ a[:3, :3].T.astype(np.float32) + a[:3, 3].astype(np.float32)

    def get_field(self) -> np.ndarray:
        """Dense fixed -> moving map ``f`` in world RAS mm, shape ``(X, Y, Z, 3)`` on the (cropped) fixed grid."""
        assert self.fitted, "call fit() first"
        if self._field is None:
            assert self._Lfield is not None and self._D0 is not None
            x = self._grid_world()
            f = self._f_world(x, self._Lfield.reshape(-1, 3, 4).astype(np.float64), self._D0.reshape(-1))
            self._field = f.reshape(*self.ref.shape, 3).astype(np.float32)
        return self._field

    @torch.no_grad()
    def transform_nii(self, img: NII, padding_value: float | None = None, ddevice: DEVICES | None = None, gpu: int | None = None) -> NII:
        """Warp a moving-space image onto the fixed grid (sampling at ``f(t)``).

        Segmentations use nearest-neighbour sampling, everything else trilinear. Voxels that map outside the
        moving image get ``padding_value`` (default: 0 for segmentations, the image minimum otherwise).
        """
        device = get_device(ddevice, 0 if gpu is None else gpu) if ddevice is not None else self.device
        f = self.get_field()
        arr = img.get_seg_array() if img.seg else img.get_array()
        fill = (0.0 if img.seg else float(arr.min())) if padding_value is None else float(padding_value)
        # Deepali conventions: world = LPS, image tensors are (N, C, Z, Y, X) and grid points are (x, y, z).
        src_grid = img.to_deepali_grid(align_corners=True)
        pts_lps = torch.as_tensor(f * np.array([-1.0, -1.0, 1.0], dtype=np.float32), device=device)
        pts_lps = pts_lps.permute(2, 1, 0, 3).contiguous()  # (X,Y,Z,3) -> (Z,Y,X,3)
        cube = src_grid.transform_points(pts_lps, axes=Axes.WORLD, to_axes=Axes.CUBE_CORNERS, decimals=None)
        data = torch.as_tensor(np.ascontiguousarray(arr.astype(np.float32).transpose(2, 1, 0)), device=device)[None, None]
        out = grid_sample(data, cube.unsqueeze(0), mode=Sampling.NEAREST if img.seg else Sampling.LINEAR, padding=fill, align_corners=True)
        res = out[0, 0].cpu().numpy().transpose(2, 1, 0)  # back to (X, Y, Z)
        if img.seg:
            res = np.rint(res).astype(np.int32)
        return self.ref.set_array(res, seg=img.seg)

    def transform_poi(self, poi: POI) -> POI:
        """Map points given in the *fixed* image into the moving image by evaluating ``f`` analytically.

        ``poi`` must be a POI on (or resampled to) the fixed grid. The result is a POI on ``moving_vert``'s grid.
        The inverse direction (moving -> fixed) is not provided in closed form.
        """
        assert self.fitted, "call fit() first"
        keys = list(poi.keys())
        out = self.moving_vert.make_empty_POI()
        if not keys:
            return out
        x = _world(poi, keys)
        f = self._f_world(x, self._L_at(x), self._D0_at(x))
        vox = _to_voxel(self.moving_vert.affine, f)
        for k, c in zip(keys, vox):
            out[k] = tuple(float(i) for i in c)
        return out

    def rigid_residual(self) -> dict[int, float]:
        """Max |f(t) - L_i g(t)| (mm) over the voxels of each vertebra. Must be ~0: the objects move rigidly."""
        f = self.get_field()
        assert self._vert_arr is not None
        x = self._grid_world().reshape(*self.ref.shape, 3)
        res: dict[int, float] = {}
        for i, v in enumerate(self.ids):
            m = self._vert_arr == v
            if not m.any():
                continue
            rigid = x[m] @ self.L[i, :, 1:].T + self.L[i, :, 0]
            res[v] = float(np.abs(f[m] - rigid).max())
        return res

    def landmark_residual(self) -> float:
        """Max |f(t_j) - u_j| (mm) over the landmarks (0 up to numerical error when ``ridge == 0``)."""
        if len(self.landmarks_fixed) == 0:
            return 0.0
        t = self.landmarks_fixed
        f = self._f_world(t, self._L_at(t), self._D0_at(t))
        return float(np.abs(f - self.landmarks_moving).max())

    # ------------------------------------------------------------------ persistence (small state only)
    def save(self, path: str | Path) -> None:
        assert self.fitted, "call fit() first"
        state = {
            "ids": self.ids, "L": self.L, "landmarks_fixed": self.landmarks_fixed, "landmarks_moving": self.landmarks_moving,
            "landmark_keys": self.landmark_keys, "B": self.B, "my": self.my, "ridge": self.ridge,
            "ref_shape": self.ref.shape, "ref_affine": np.asarray(self.ref.affine), "vert": self._vert_arr,
        }  # fmt: skip
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path, moving_vert: Image_Reference, fixed_vert: Image_Reference) -> Polyrigid_Registration:
        """Rebuild from :meth:`save`. ``fixed_vert`` only supplies the grid (it is cropped like at fit time)."""
        with open(path, "rb") as f:
            s = pickle.load(f)  # noqa: S301
        fv = to_nii(fixed_vert, True)
        self = cls.__new__(cls)
        self.moving_vert = to_nii(moving_vert, True)
        self.fixed_vert = fv
        self.my, self.ridge, self.ids, self.L = s["my"], s["ridge"], s["ids"], s["L"]
        self.verbose, self.device, self.landmark_keys = False, torch.device("cpu"), s["landmark_keys"]
        self._field = None
        # reproduce the crop: same affine/shape as stored
        ref = fv.make_empty_nii(True)
        ref = ref.set_array(np.zeros(fv.shape, dtype=np.int32), seg=True)
        origin_vox = np.rint(_to_voxel(np.asarray(fv.affine), np.asarray(s["ref_affine"])[:3, 3][None]))[0].astype(int)
        crop = tuple(slice(int(o), int(o) + int(n)) for o, n in zip(origin_vox, s["ref_shape"]))
        self.ref = ref.apply_crop(crop)
        self._vert_arr = s["vert"]
        self._set_landmarks(s["landmark_keys"], s["landmarks_fixed"], s["landmarks_moving"])
        self._precompute_fields()
        self._solve_B()
        self.fitted = True
        return self
