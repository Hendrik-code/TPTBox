"""Gradient-based estimation / refinement of the per-vertebra rigid transforms of :class:`Polyrigid_Registration`.

All maths is in world RAS millimetres on a coarse *working grid* (``PolyrigidOptimConfig.levels_mm``), so the cost
is independent of the full image resolution and the grid size is adjustable.

Parametrisation (per vertebra ``i``): ``f_i(x) = R_i (x - c_fix_i) + m0_i + delta_i`` with
``R_i = exp(omega_i) R0_i`` and ``m0_i = R0_i c_fix_i + t0_i``. ``(R0, t0)`` is the closed-form initialisation
(Horn fit from POIs, or only a translation with ``R0 = I``); ``omega`` and ``delta`` start at 0.

Stage A (``independent``): every vertebra is registered alone on its own mask overlap (soft Dice).
Stage B (``joint``): all rigid bodies are optimised together through the *blended* field
``f(x) = sum_i w_i(x) (R_i x + t_i)`` (eq. 19 of Little et al.; ``w_i`` depends only on the fixed masks, so the field
is linear in the ``L_i``). Loss = soft Dice of all vertebra labels + a folding penalty (``det J <= 0``) outside the
bone + an incompressibility penalty (``log det J``) inside non-rigid labelled structures; structures that may
compress get a lower weight through ``compressibility``. Points closer than ``min_dist_mm`` to a mask surface are not
used by the penalties.
"""

# ruff: noqa: B023  (the loss closures are created and used inside the same level iteration, so late binding is intended)
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from deepali.core import PaddingMode, Sampling
from deepali.core.functional import grid_sample
from deepali.losses.functional import bending_loss, dice_loss
from scipy.ndimage import distance_transform_edt, gaussian_filter

from TPTBox import NII

_EPS = 1e-6


# --------------------------------------------------------------------------------------------- configuration
@dataclass
class PolyrigidOptimConfig:
    """Weights and optimiser settings. Every field can be changed through ``Polyrigid_Registration(optim=...)``."""

    levels_mm: tuple[float, ...] = (6.0, 3.0)  # working-grid spacing per pyramid level (coarse -> fine)
    steps: tuple[int, ...] = (150, 100)  # Adam steps per level, stage A
    joint_steps: tuple[int, ...] = (100, 100)  # Adam steps per level, stage B
    blur_sigma_mm: tuple[float, ...] = (4.0, 0.0)  # label blur per level (widens the capture range)
    lr_rot: float = 0.02  # rad per step
    lr_trans_mm: float = 0.5  # mm per step
    joint_lr_scale: float = 0.5  # stage B learning rates = scale * stage A
    patience: int = 20
    bbox_margin_mm: float = 20.0  # stage A: region around each fixed vertebra that is sampled
    w_dice: float = 1.0
    w_fold: float = 10.0
    fold_eps: float = 0.1  # penalty relu(fold_eps - det J)^2
    w_incomp: float = 1.0  # penalty (log det J)^2 inside non-rigid labelled structures
    compressibility: Mapping[int, float] | None = None  # label -> multiplier of w_incomp (0 = may compress freely)
    w_bend: float = 0.0  # optional bending energy of the displacement field
    scale_range: tuple[float, float] = (0.6, 1.5)  # allowed SI scale / isotropic scale of a vertebra that may change its shape
    w_scale: float = 0.1  # penalty (log scale)^2 pulling a flexible vertebra towards its original size
    lr_scale: float = 0.01  # learning rate of the log scales
    shape_tol: float = 0.0  # after step 1, vertebrae whose scales are all within this fraction of 1 are reset to rigid (auto: 0.05)

    def per_level(self, values: Sequence, level: int, default=0.0):
        return values[level] if level < len(values) else (values[-1] if len(values) else default)


@dataclass
class OptimLog:
    """Timings and loss curves of the optimisation stages."""

    stages: list[dict] = field(default_factory=list)
    shape: dict[int, tuple[float, float]] = field(default_factory=dict)  # vertebra id -> (SI scale, isotropic scale)

    def add(self, **kw) -> None:
        self.stages.append(kw)

    def summary(self) -> str:
        return "; ".join(
            f"{s['name']}@{s['level_mm']}mm: {s['steps_done']} steps, loss {s['loss_first']:.3f}->{s['loss_best']:.3f}, {s['seconds']:.1f}s"
            for s in self.stages
        )


# --------------------------------------------------------------------------------------------- small helpers
def rotation_from_axis_angle(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues: ``(n, 3)`` axis-angle -> ``(n, 3, 3)`` rotation matrices (identity at 0, differentiable everywhere)."""
    th2 = (w * w).sum(-1, keepdim=True) + 1e-12
    th = th2.sqrt()
    a = (torch.sin(th) / th)[..., None]
    b = ((1 - torch.cos(th)) / th2)[..., None]
    z = torch.zeros_like(w[..., 0])
    k = torch.stack([z, -w[..., 2], w[..., 1], w[..., 2], z, -w[..., 0], -w[..., 1], w[..., 0], z], -1).reshape(*w.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(k)
    return eye + a * k + b * (k @ k)


def centroid_world(arr: np.ndarray, affine: np.ndarray, label: int) -> np.ndarray:
    """World-RAS centroid (mm) of the voxels equal to ``label``."""
    idx = np.argwhere(arr == label)
    if len(idx) == 0:
        raise ValueError(f"label {label} not present")
    return idx.mean(0) @ affine[:3, :3].T + affine[:3, 3]


def jacobian_det(f: torch.Tensor, zoom: Sequence[float]) -> torch.Tensor:
    """det(d f / d x) of a world-mm map ``f (X, Y, Z, 3)`` by central differences (1 = volume preserving).

    Derivatives are taken along the array axes with the voxel size in mm. For oblique grids the determinant is the same as in
    world axes, because the array->world rotation has determinant 1.
    """
    g = torch.gradient(f, spacing=tuple(float(z) for z in zoom), dim=(0, 1, 2))
    return torch.linalg.det(torch.stack(g, dim=-1))  # (..., i, j) = d f_i / d a_j


def fold_penalty(det: torch.Tensor, mask: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """Mean of ``relu(eps - det J)^2`` over the penalty points ``mask`` (0 for any ``det J >= eps``)."""
    return (torch.relu(eps - det) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def incompressibility_penalty(det: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean of ``(log det J)^2`` over the voxels where ``weight > 0`` (0 for volume-preserving maps)."""
    n = (weight > 0).sum().clamp(min=1)
    return (weight * torch.log(det.clamp(min=1e-3)) ** 2).sum() / n


def _world_coords(shape: Sequence[int], affine: np.ndarray, device: torch.device) -> torch.Tensor:
    idx = torch.stack(torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape], indexing="ij"), -1)
    a = torch.as_tensor(affine, dtype=torch.float32)
    return (idx @ a[:3, :3].T + a[:3, 3]).to(device)


def _sample(vol: torch.Tensor, world: torch.Tensor, inv_affine: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Sample ``vol (B, C, X, Y, Z)`` (array axes) at world positions ``world (B|1, x, y, z, 3)`` (trilinear, zero padding)."""
    vox = world @ inv_affine[:3, :3].T + inv_affine[:3, 3]
    n = torch.as_tensor([max(s - 1, 1) for s in shape], dtype=vox.dtype, device=vox.device)
    grid = (2.0 * vox / n - 1.0).flip(-1)  # grid last dim is (W, H, D) = (axis 2, axis 1, axis 0)
    return grid_sample(vol, grid, mode=Sampling.LINEAR, padding=PaddingMode.ZEROS, align_corners=True)


def _rescale_labels(nii: NII, spacing_mm: float) -> NII:
    return nii.rescale((spacing_mm,) * 3, verbose=False)


def _onehot(arr: np.ndarray, ids: Sequence[int], sigma_vox: float) -> np.ndarray:
    out = np.zeros((len(ids), *arr.shape), dtype=np.float32)
    for c, v in enumerate(ids):
        m = (arr == v).astype(np.float32)
        out[c] = gaussian_filter(m, sigma_vox) if sigma_vox > 0 else m
    return out


def blend_weights(arr: np.ndarray, ids: Sequence[int], zoom: Sequence[float], my: float) -> np.ndarray:
    """``w_i(x)`` of eq. 17 on a grid, shape ``(n, X, Y, Z)``: 1 inside object i, 0 inside the others, inverse distance outside."""
    q = np.zeros((len(ids), *arr.shape), dtype=np.float32)
    for c, v in enumerate(ids):
        d = distance_transform_edt(arr != v, sampling=tuple(float(z) for z in zoom))
        nz = d > 0
        q[c][nz] = d[nz] ** (-my)
    w = q / np.maximum(q.sum(0, keepdims=True), _EPS)
    for c, v in enumerate(ids):
        m = arr == v
        w[:, m] = 0.0
        w[c][m] = 1.0
    return w


# --------------------------------------------------------------------------------------------- optimisation
class _Params:
    """``(omega, delta)`` per vertebra on top of the closed-form init ``(R0, t0)``.

    Vertebrae listed in ``shape_idx`` additionally have ``(log k, log s)``: ``S = s (I + (k - 1) a a^T)`` with the SI axis ``a``
    (fixed frame) is applied before the rotation, ``f_i(x) = R S (x - c_fix) + m0 + delta``; the scales are clamped to ``cfg.scale_range``.
    """

    def __init__(self, R0, t0, c_fix, device, cfg=None, shape_idx: Sequence[int] = (), axes=None):
        f32 = {"dtype": torch.float32, "device": device}
        self.R0 = torch.as_tensor(R0, **f32)
        self.c_fix = torch.as_tensor(c_fix, **f32)
        self.m0 = torch.einsum("nij,nj->ni", self.R0, self.c_fix) + torch.as_tensor(t0, **f32)
        n = len(R0)
        self.omega = torch.zeros(n, 3, requires_grad=True, **f32)
        self.delta = torch.zeros(n, 3, requires_grad=True, **f32)
        self.logk = torch.zeros(n, requires_grad=True, **f32)
        self.logs = torch.zeros(n, requires_grad=True, **f32)
        self.smask = torch.zeros(n, **f32)
        self.smask[list(shape_idx)] = 1.0
        self.has_shape = len(shape_idx) > 0
        self.shape_train = False
        lo, hi = cfg.scale_range if cfg is not None else (0.6, 1.5)
        self.lo, self.hi = float(np.log(lo)), float(np.log(hi))
        a = np.tile([0.0, 0.0, 1.0], (n, 1)) if axes is None else np.asarray(axes, dtype=np.float64)
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), _EPS)
        self.axis = torch.as_tensor(a, **f32)

    def _logs(self) -> tuple[torch.Tensor, torch.Tensor]:
        k, s = self.logk * self.smask, self.logs * self.smask
        if not self.shape_train:
            k, s = k.detach(), s.detach()
        return k.clamp(self.lo, self.hi), s.clamp(self.lo, self.hi)

    def shape_penalty(self) -> torch.Tensor:
        if not (self.has_shape and self.shape_train):
            return torch.zeros((), device=self.omega.device)
        k, s = self._logs()
        return (k**2 + s**2).sum() / self.smask.sum()

    def S(self) -> torch.Tensor:
        k, s = self._logs()
        eye = torch.eye(3, dtype=k.dtype, device=k.device).expand(len(k), 3, 3)
        outer = self.axis[:, :, None] * self.axis[:, None, :]
        return torch.exp(s)[:, None, None] * (eye + (torch.exp(k) - 1)[:, None, None] * outer)

    def Rt(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Linear part ``A = R S`` (a rotation for rigid vertebrae) and translation of ``f_i(x) = A x + t``."""
        A = rotation_from_axis_angle(self.omega) @ self.R0
        if self.has_shape:
            A = A @ self.S()
        t = self.m0 + self.delta - torch.einsum("nij,nj->ni", A, self.c_fix)
        return A, t

    def drop_small_shapes(self, tol: float) -> None:
        """Vertebrae whose fitted scales deviate less than ``tol`` from 1 go back to rigid (scale 1, no shape)."""
        if tol <= 0:
            return
        k, s = self._logs()
        keep = (torch.maximum(k.abs(), s.abs()) >= float(np.log1p(tol))).to(self.smask.dtype)
        with torch.no_grad():
            self.smask.mul_(keep)

    def shape_report(self, ids: Sequence[int]) -> dict[int, tuple[float, float]]:
        k, s = self._logs()
        return {int(ids[i]): (float(torch.exp(k[i])), float(torch.exp(s[i]))) for i in range(len(ids)) if self.smask[i] > 0}

    def state(self):
        return tuple(x.detach().clone() for x in (self.omega, self.delta, self.logk, self.logs))

    def load(self, st) -> None:
        with torch.no_grad():
            for x, v in zip((self.omega, self.delta, self.logk, self.logs), st):
                x.copy_(v)

    def L(self) -> np.ndarray:
        R, t = self.Rt()
        return torch.cat([t[..., None], R], -1).detach().cpu().numpy().astype(np.float64)


def _adam(p: _Params, cfg: PolyrigidOptimConfig, scale: float) -> torch.optim.Optimizer:
    groups = [{"params": [p.omega], "lr": cfg.lr_rot * scale}, {"params": [p.delta], "lr": cfg.lr_trans_mm * scale}]
    if p.has_shape and p.shape_train:
        groups.append({"params": [p.logk, p.logs], "lr": cfg.lr_scale})
    return torch.optim.Adam(groups)


def _run(p: _Params, opt, loss_fn, steps: int, patience: int):
    """Adam loop with patience-based early stopping and best-state restore (pattern of ``Rigid_Registration_with_Tether``)."""
    best, best_state, bad, first, done = float("inf"), p.state(), 0, None, 0
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        v = float(loss.detach())
        first = v if first is None else first
        if v < best - 1e-5:
            best, best_state, bad = v, p.state(), 0
        else:
            bad += 1
            if bad > patience:
                break
        loss.backward()
        opt.step()
        done += 1
    p.load(best_state)
    return (first if first is not None else float("nan")), best, done


def refine_rigid(
    fixed_labels: NII,
    moving_labels: NII,
    ids: Sequence[int],
    R0: np.ndarray,
    t0: np.ndarray,
    c_fix: np.ndarray,
    cfg: PolyrigidOptimConfig,
    device: torch.device,
    *,
    independent: bool,
    joint: bool,
    fixed_other: NII | None = None,
    shape_ids: Sequence[int] = (),
    axes: np.ndarray | None = None,
    my: float = 1.5,
    min_dist_mm: float = 2.0,
    verbose: bool = False,
) -> tuple[np.ndarray, OptimLog]:
    """Estimate / refine ``L_i`` (fixed -> moving, world mm) from masks. Returns ``L (n, 3, 4) = [t | R]`` and a log.

    Args:
        fixed_labels: rigid vertebra labels of the fixed image (the registration grid, cropped).
        moving_labels: rigid vertebra labels of the moving image (any grid, ideally cropped to the spine).
        ids: vertebra ids, the order of ``R0``/``t0``/``c_fix``.
        R0, t0, c_fix: closed-form init per vertebra (``f0(x) = R0 x + t0``) and the fixed centroids in world mm.
        independent / joint: which stages to run.
        fixed_other: fixed labels of the *non-rigid* structures (ribs, discs, ...), same grid as ``fixed_labels``;
            used by the incompressibility penalty of stage B.
        shape_ids / axes: vertebrae that may change their shape (compressed / fractured) and, per vertebra of ``ids``, the unit
            superior-inferior axis ``(n, 3)`` in the fixed frame. They first run stage A with a bounded SI scale and isotropic
            scale (``cfg.scale_range``, ``cfg.w_scale``), then the shape is frozen and the normal stages follow. The fitted
            shape is in ``log.shape``; ``L`` is then ``[t | R S]`` for these vertebrae.
    """
    log = OptimLog()
    p = _Params(R0, t0, c_fix, device, cfg, [list(ids).index(v) for v in shape_ids if v in ids], axes)
    if not (independent or joint or p.has_shape):
        return p.L(), log
    comp = dict(cfg.compressibility or {})

    def _levels(independent: bool, joint: bool) -> None:
        for lvl, sp in enumerate(cfg.levels_mm):
            fx = _rescale_labels(fixed_labels, sp)
            mv = _rescale_labels(moving_labels, sp)
            fa, ma = fx.get_seg_array(), mv.get_seg_array()
            sigma = cfg.per_level(cfg.blur_sigma_mm, lvl) / sp
            zoom = [float(z) for z in fx.zoom]
            world = _world_coords(fa.shape, fx.affine, device)
            inv_m = torch.as_tensor(np.linalg.inv(mv.affine), dtype=torch.float32, device=device)
            mov_oh = torch.as_tensor(_onehot(ma, ids, sigma), device=device)[None]  # (1, C, X, Y, Z)
            fix_oh = torch.as_tensor(_onehot(fa, ids, sigma), device=device)[None]
            mshape = ma.shape

            if independent:
                tasks = []
                margin = int(np.ceil(cfg.bbox_margin_mm / sp))
                for c, v in enumerate(ids):
                    idx = np.argwhere(fa == v)
                    lo = np.maximum(idx.min(0) - margin, 0)
                    hi = np.minimum(idx.max(0) + margin + 1, fa.shape)
                    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
                    tasks.append((c, sl, world[sl][None], fix_oh[:, c : c + 1][(slice(None), slice(None), *sl)]))

                def loss_a():
                    R, t = p.Rt()
                    total = 0.0
                    for c, _sl, w_bb, tgt in tasks:
                        f = w_bb[0] @ R[c].T + t[c]  # fixed world -> moving world, rigid map of vertebra c
                        warped = _sample(mov_oh[:, c : c + 1], f[None], inv_m, mshape)
                        total = total + cfg.w_dice * dice_loss(warped, tgt)
                    return total / len(tasks) + cfg.w_scale * p.shape_penalty()

                t_start = time.perf_counter()
                first, best, done = _run(p, _adam(p, cfg, 1.0), loss_a, cfg.per_level(cfg.steps, lvl, 0), cfg.patience)
                log.add(
                    name="independent",
                    level_mm=sp,
                    steps_done=done,
                    loss_first=first,
                    loss_best=best,
                    seconds=time.perf_counter() - t_start,
                )
                if verbose:
                    print("[Polyrigid optim]", log.stages[-1])

            if joint:
                t_start = time.perf_counter()
                w = torch.as_tensor(blend_weights(fa, ids, zoom, my), device=device).reshape(len(ids), -1).T.contiguous()  # (N, C)
                x_flat = world.reshape(-1, 3)
                d0 = distance_transform_edt(~np.isin(fa, list(ids)), sampling=tuple(zoom))
                fold_mask = torch.as_tensor(((fa == 0) & (d0 >= min_dist_mm)).astype(np.float32), device=device)
                incomp = np.zeros(fa.shape, dtype=np.float32)
                if fixed_other is not None and cfg.w_incomp > 0:
                    oa = _rescale_labels(fixed_other, sp).get_seg_array()
                    for lab in np.unique(oa):
                        if lab == 0:
                            continue
                        inner = distance_transform_edt(oa == lab, sampling=tuple(zoom)) >= min_dist_mm
                        incomp[(oa == lab) & inner & (fa == 0)] = cfg.w_incomp * float(comp.get(int(lab), 1.0))
                incomp_t = torch.as_tensor(incomp, device=device)
                n_incomp = float((incomp > 0).sum())

                def field_of() -> torch.Tensor:
                    R, t = p.Rt()
                    Rb = (w @ R.reshape(len(ids), 9)).reshape(-1, 3, 3)
                    f = (Rb @ x_flat[..., None])[..., 0] + w @ t
                    return f.reshape(*fa.shape, 3)

                def loss_b():
                    f = field_of()
                    warped = _sample(mov_oh, f[None], inv_m, mshape)
                    loss = cfg.w_dice * dice_loss(warped, fix_oh, reduction="none").mean()
                    if cfg.w_fold > 0 or cfg.w_incomp > 0:
                        det = jacobian_det(f, zoom)
                        if cfg.w_fold > 0:
                            loss = loss + cfg.w_fold * fold_penalty(det, fold_mask, cfg.fold_eps)
                        if cfg.w_incomp > 0 and n_incomp > 0:
                            loss = loss + incompressibility_penalty(det, incomp_t)
                    loss = loss + cfg.w_scale * p.shape_penalty()
                    if cfg.w_bend > 0:
                        u = (f - world).permute(3, 0, 1, 2)[None]
                        loss = loss + cfg.w_bend * bending_loss(u, spacing=tuple(reversed(zoom)))
                    return loss

                first, best, done = _run(p, _adam(p, cfg, cfg.joint_lr_scale), loss_b, cfg.per_level(cfg.joint_steps, lvl, 0), cfg.patience)
                log.add(name="joint", level_mm=sp, steps_done=done, loss_first=first, loss_best=best, seconds=time.perf_counter() - t_start)
                if verbose:
                    print("[Polyrigid optim]", log.stages[-1])

    if p.has_shape:  # step 1: the flagged vertebrae may change their shape (bounded) while the pose is fitted ...
        p.shape_train = True
        _levels(True, False)
        p.shape_train = False
        p.drop_small_shapes(cfg.shape_tol)  # ... step 2: the fitted shape is frozen, the vertebrae are stiff again
        log.shape = p.shape_report(ids)
        if verbose:
            print("[Polyrigid optim] shape (SI scale, isotropic scale):", log.shape)
    _levels(independent, joint)
    return p.L(), log
