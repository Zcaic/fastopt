"""
High-performance C++17 traditional & gradient-enhanced surrogate modeling module (`fastopt.surrogate`).

Provides four core surrogate model families backed by the fused C++17 `SurrogateNode`
(`OpType.Surrogate`) with exact 0th, 1st (Jacobian/VJP/JVP), and 2nd-order (Hessian HVP) AD
for BOTH predictive mean and predictive variance:
  1. `RBF`        : Radial Basis Function (Gaussian, MQ, IMQ, Cubic, Thin-Plate Spline + polynomial tail)
  2. `Kriging`    : Ordinary / Universal Kriging with anisotropic ARD MLE and exact mean & variance AD
  3. `GEK` / `KPLS` / `GEKPLS` : Gradient-Enhanced Kriging & High-Dimensional PLS Kriging
  4. `CoKriging`  : Multi-Fidelity Auto-Regressive Kriging (supports optional LF/HF gradients)

Also includes differentiable acquisition functions (`lower_confidence_bound`, `expected_improvement`)
for direct gradient-based Bayesian optimization / EGO infill inside `fastopt.Opti`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from ._fastopt_impl import (
    CompiledSurrogate,
    SurrogateKernel,
    SurrogateOutputMode,
    op_surrogate,
)
from . import expr as fo_expr
from .expr import Expr, GraphBuilder


_KERNEL_MAP: Dict[str, SurrogateKernel] = {
    "gaussian": SurrogateKernel.Gaussian,
    "se": SurrogateKernel.Gaussian,
    "rbf": SurrogateKernel.Gaussian,
    "squared_exponential": SurrogateKernel.Gaussian,
    "multiquadric": SurrogateKernel.Multiquadric,
    "mq": SurrogateKernel.Multiquadric,
    "inv_multiquadric": SurrogateKernel.InvMultiquadric,
    "imq": SurrogateKernel.InvMultiquadric,
    "cubic": SurrogateKernel.Cubic,
    "thin_plate_spline": SurrogateKernel.ThinPlateSpline,
    "tps": SurrogateKernel.ThinPlateSpline,
    "matern32": SurrogateKernel.Matern32,
    "matern_32": SurrogateKernel.Matern32,
    "matern52": SurrogateKernel.Matern52,
    "matern_52": SurrogateKernel.Matern52,
}

_POLY_MAP: Dict[str, int] = {
    "none": -1,
    "constant": 0,
    "const": 0,
    "linear": 1,
    "quadratic": 2,
    "quad": 2,
}


def resolve_kernel(kernel: Union[str, SurrogateKernel]) -> SurrogateKernel:
    if isinstance(kernel, SurrogateKernel):
        return kernel
    key = str(kernel).strip().lower()
    if key not in _KERNEL_MAP:
        raise ValueError(f"Unsupported surrogate kernel '{kernel}'. Supported: {sorted(_KERNEL_MAP.keys())}")
    return _KERNEL_MAP[key]


def resolve_poly_degree(poly: Union[int, str, None]) -> int:
    if poly is None:
        return -1
    if isinstance(poly, (int, np.integer)):
        deg = int(poly)
        if deg < -1 or deg > 2:
            raise ValueError(f"poly_degree must be -1, 0, 1, or 2, got {deg}.")
        return deg
    key = str(poly).strip().lower()
    if key not in _POLY_MAP:
        raise ValueError(f"Unsupported poly trend '{poly}'. Supported: {sorted(_POLY_MAP.keys())}")
    return _POLY_MAP[key]


def _kernel_jet2_np(kern: SurrogateKernel, q: np.ndarray, r_eps: float = 1e-14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 0th, 1st, 2nd derivatives of phi(q) w.r.t. q for training matrix assembly."""
    if kern == SurrogateKernel.Gaussian:
        e = np.exp(-q)
        return e, -e, e
    if kern == SurrogateKernel.Multiquadric:
        qp1 = q + 1.0
        sq = np.sqrt(qp1)
        inv1 = 1.0 / sq
        inv3 = inv1 / qp1
        return sq, 0.5 * inv1, -0.25 * inv3
    if kern == SurrogateKernel.InvMultiquadric:
        qp1 = q + 1.0
        sq = np.sqrt(qp1)
        inv1 = 1.0 / sq
        inv3 = inv1 / qp1
        inv5 = inv3 / qp1
        return inv1, -0.5 * inv3, 0.75 * inv5
    if kern == SurrogateKernel.Cubic:
        qs = q + r_eps
        r = np.sqrt(qs)
        return qs * r, 1.5 * r, 0.75 / r
    if kern == SurrogateKernel.ThinPlateSpline:
        qs = q + r_eps
        lq = np.log(qs)
        return 0.5 * q * lq, 0.5 * (lq + q / qs), 0.5 * (q + 2.0 * r_eps) / (qs * qs)
    if kern == SurrogateKernel.Matern32:
        s2 = 3.0 * q + r_eps
        s = np.sqrt(s2)
        e = np.exp(-s)
        return (1.0 + s) * e, -1.5 * e, 2.25 * e / s
    if kern == SurrogateKernel.Matern52:
        s2 = 5.0 * q + r_eps
        s = np.sqrt(s2)
        e = np.exp(-s)
        return (1.0 + s + s2 / 3.0) * e, -(5.0 / 6.0) * (1.0 + s) * e, (25.0 / 12.0) * e
    raise ValueError(f"Unknown kernel: {kern}")


def _poly_design_matrix(Xt: np.ndarray, deg: int) -> np.ndarray:
    """Build polynomial trend matrix F of shape (N, n_poly) for normalized inputs Xt (N, d)."""
    N, d = Xt.shape
    if deg < 0:
        return np.empty((N, 0), dtype=np.float64)
    cols = [np.ones((N, 1), dtype=np.float64)]
    if deg >= 1:
        cols.append(Xt)
    if deg >= 2:
        quads = []
        for a in range(d):
            for b in range(a, d):
                quads.append((Xt[:, a] * Xt[:, b])[:, None])
        cols.extend(quads)
    return np.hstack(cols)


def _poly_grad_design_matrix(Xt: np.ndarray, deg: int) -> np.ndarray:
    """Build gradient polynomial trend matrix F_g of shape (N * d, n_poly)."""
    N, d = Xt.shape
    if deg < 0:
        return np.empty((N * d, 0), dtype=np.float64)
    np_dim = 1 if deg == 0 else (1 + d if deg == 1 else 1 + d + (d * (d + 1)) // 2)
    Fg = np.zeros((N * d, np_dim), dtype=np.float64)
    if deg >= 1:
        for i in range(N):
            for j in range(d):
                Fg[i * d + j, 1 + j] = 1.0
    if deg >= 2:
        for i in range(N):
            idx = 1 + d
            for a in range(d):
                for b in range(a, d):
                    if a == b:
                        Fg[i * d + a, idx] = 2.0 * Xt[i, a]
                    else:
                        Fg[i * d + a, idx] = Xt[i, b]
                        Fg[i * d + b, idx] = Xt[i, a]
                    idx += 1
    return Fg


def _build_kernel_matrix(
    Xt: np.ndarray,
    theta: np.ndarray,
    kern: SurrogateKernel,
    r_eps: float = 1e-14,
) -> np.ndarray:
    """Build (N, N) value correlation/kernel matrix."""
    diff = Xt[:, None, :] - Xt[None, :, :]  # (N, N, d)
    q = np.einsum("ijd,d->ij", diff * diff, theta)
    p0, _, _ = _kernel_jet2_np(kern, q, r_eps=r_eps)
    return p0


def _build_gek_kernel_matrix(
    Xt: np.ndarray,
    theta: np.ndarray,
    kern: SurrogateKernel,
    r_eps: float = 1e-14,
) -> np.ndarray:
    """Build symmetric (N*(1+d), N*(1+d)) value+gradient covariance matrix for GEK."""
    N, d = Xt.shape
    diff = Xt[:, None, :] - Xt[None, :, :]  # u[i, j, a] = x_i[a] - x_j[a]
    q = np.einsum("ijd,d->ij", diff * diff, theta)
    p0, p1, p2 = _kernel_jet2_np(kern, q, r_eps=r_eps)

    M = N * (1 + d)
    R = np.empty((M, M), dtype=np.float64)
    R[:N, :N] = p0

    # Value-Grad block: R_vg[i, j*d + b] = -2 * p1[i, j] * theta[b] * u[i, j, b]
    R_vg = (-2.0 * p1[:, :, None]) * (diff * theta[None, None, :])  # (N, N, d)
    R_vg_2d = R_vg.reshape((N, N * d))
    R[:N, N:] = R_vg_2d
    R[N:, :N] = R_vg_2d.T

    # Grad-Grad block: R_gg[i*d + a, j*d + b] = -2 * p1 * theta_a * delta_ab - 4 * p2 * theta_a * theta_b * u_a * u_b
    tu = diff * theta[None, None, :]  # (N, N, d)
    R_gg = (-4.0 * p2[:, :, None, None]) * (tu[:, :, :, None] * tu[:, :, None, :])  # (N, N, d, d)
    diag_add = (-2.0 * p1[:, :, None]) * theta[None, None, :]  # (N, N, d)
    idx_d = np.arange(d)
    R_gg[:, :, idx_d, idx_d] += diag_add
    R[N:, N:] = R_gg.transpose((0, 2, 1, 3)).reshape((N * d, N * d))
    return 0.5 * (R + R.T)


def _normalize_xy(
    X: np.ndarray,
    Y: np.ndarray,
    dY: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    N, d = X_arr.shape

    Y_arr = np.asarray(Y, dtype=np.float64)
    was_1d = Y_arr.ndim == 1
    if was_1d:
        Y_arr = Y_arr[:, None]
    if Y_arr.shape[0] != N:
        raise ValueError(f"X rows ({N}) and Y rows ({Y_arr.shape[0]}) mismatch.")
    m = Y_arr.shape[1]

    in_mean = X_arr.mean(axis=0)
    in_std = X_arr.std(axis=0)
    in_std = np.where(in_std < 1e-12, 1.0, in_std)
    in_inv_std = 1.0 / in_std
    Xt = (X_arr - in_mean) * in_inv_std

    out_mean = Y_arr.mean(axis=0)
    out_std = Y_arr.std(axis=0)
    out_std = np.where(out_std < 1e-12, 1.0, out_std)
    Yt = (Y_arr - out_mean) / out_std

    dYt = None
    if dY is not None:
        dY_arr = np.asarray(dY, dtype=np.float64)
        if m == 1 and dY_arr.ndim == 2:
            dY_arr = dY_arr[:, :, None]
        if dY_arr.shape != (N, d, m):
            raise ValueError(f"Expected dY shape {(N, d, m)}, got {dY_arr.shape}.")
        # d(y_tilde_c)/d(x_tilde_j) = (in_std[j] / out_std[c]) * d(y_c)/d(x_j)
        dYt = dY_arr * (in_std[None, :, None] / out_std[None, None, :])
    return Xt, Yt, dYt, in_mean, in_inv_std, out_mean, out_std, was_1d


def _fit_pls_weights(Xt: np.ndarray, Yt: np.ndarray, n_comp: int, dYt: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Pure-NumPy NIPALS Partial Least Squares returning unit-norm weight matrix W_pls of shape (d, h).
    When gradient samples `dYt` are available, augments PLS with gradient sensitivity directions
    so active high-dimensional subspaces are captured accurately even with small sample counts.
    """
    N, d = Xt.shape
    h = max(1, min(int(n_comp), d))
    if dYt is not None:
        # Combine gradient sensitivity matrix (N*m, d) with value-centered NIPALS
        G = dYt.transpose((0, 2, 1)).reshape((-1, d))
        # SVD of gradient matrix captures exact active subspace of f(x)
        _, _, Vt = np.linalg.svd(G, full_matrices=False)
        W_pls = Vt[:h, :].T.copy()  # (d, h)
        for k in range(W_pls.shape[1]):
            nrm = np.linalg.norm(W_pls[:, k])
            if nrm > 1e-12:
                W_pls[:, k] /= nrm
        return W_pls

    X_res = Xt - Xt.mean(axis=0, keepdims=True)
    Y_res = Yt - Yt.mean(axis=0, keepdims=True)
    W_list: List[np.ndarray] = []
    for _ in range(h):
        # Cross-covariance u-step
        if Y_res.shape[1] == 1:
            w = X_res.T @ Y_res[:, 0]
        else:
            C = X_res.T @ Y_res
            u_svd, _, _ = np.linalg.svd(C, full_matrices=False)
            w = u_svd[:, 0]
        nrm = float(np.linalg.norm(w))
        if nrm < 1e-12:
            # Fallback to remaining variance direction
            _, _, Vt = np.linalg.svd(X_res, full_matrices=False)
            w = Vt[0]
            nrm = max(float(np.linalg.norm(w)), 1e-12)
        w = w / nrm
        t = X_res @ w
        tt = float(t @ t)
        if tt < 1e-14:
            break
        p = (X_res.T @ t) / tt
        q = (Y_res.T @ t) / tt
        X_res = X_res - np.outer(t, p)
        Y_res = Y_res - np.outer(t, q)
        W_list.append(w)

    if not W_list:
        return np.eye(d, h, dtype=np.float64)
    return np.column_stack(W_list)


def _solve_kriging_gls(
    R: np.ndarray,
    F: np.ndarray,
    Y_aug: np.ndarray,
    nugget: float,
    theta: Optional[np.ndarray] = None,
    n_val: Optional[int] = None,
    return_matrices: bool = False,
) -> Dict[str, Any]:
    """Solve Kriging / GEK Generalized Least Squares via Cholesky and compute concentrated NLL."""
    M = R.shape[0]
    R_reg = R.copy()
    if n_val is not None and theta is not None and M > n_val:
        d = theta.size
        R_reg[:n_val, :n_val] += nugget * np.eye(n_val)
        grad_diag = np.tile(np.maximum(2.0 * theta, 1.0), n_val) * max(nugget, 1e-8)
        idx = np.arange(n_val, M)
        R_reg[idx, idx] += grad_diag
    else:
        R_reg[np.arange(M), np.arange(M)] += nugget

    jitter = 0.0
    c_and_lower = None
    for _ in range(6):
        try:
            if jitter > 0.0:
                R_try = R_reg + jitter * np.eye(M)
            else:
                R_try = R_reg
            c_and_lower = cho_factor(R_try, lower=True, check_finite=False)
            break
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0.0 else jitter * 100.0
    if c_and_lower is None:
        return {"ok": False, "nll": 1e30}

    L = c_and_lower[0]
    logdet_R = 2.0 * float(np.sum(np.log(np.maximum(np.diag(L), 1e-300))))

    Rinv_Y = cho_solve(c_and_lower, Y_aug, check_finite=False)
    n_poly = F.shape[1]
    if n_poly > 0:
        Rinv_F = cho_solve(c_and_lower, F, check_finite=False)
        A = F.T @ Rinv_F
        A = 0.5 * (A + A.T)
        try:
            A_fac = cho_factor(A + 1e-12 * np.eye(n_poly), lower=True, check_finite=False)
            beta_poly = cho_solve(A_fac, F.T @ Rinv_Y, check_finite=False)
            A_inv = cho_solve(A_fac, np.eye(n_poly), check_finite=False) if return_matrices else None
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
            beta_poly = A_inv @ (F.T @ Rinv_Y)
        res = Y_aug - F @ beta_poly
        weights = cho_solve(c_and_lower, res, check_finite=False)
    else:
        Rinv_F = np.empty((M, 0), dtype=np.float64)
        beta_poly = np.empty((0, Y_aug.shape[1]), dtype=np.float64)
        A_inv = np.empty((0, 0), dtype=np.float64)
        res = Y_aug
        weights = Rinv_Y

    sigma2 = np.sum(res * weights, axis=0) / float(M)
    sigma2 = np.maximum(sigma2, 1e-14)
    nll = float(np.sum(M * np.log(sigma2) + logdet_R))

    out: Dict[str, Any] = {
        "ok": True,
        "nll": nll,
        "weights": weights,
        "poly_coef": beta_poly,
        "sigma2": sigma2,
    }
    if return_matrices:
        from scipy.linalg import solve_triangular
        L_R = np.tril(L)
        if n_poly > 0:
            W_F = solve_triangular(L_R, F, lower=True, check_finite=False)
            A_mat = W_F.T @ W_F
            A_mat = 0.5 * (A_mat + A_mat.T)
            try:
                L_A = np.tril(cho_factor(A_mat + 1e-12 * np.eye(n_poly), lower=True, check_finite=False)[0])
            except np.linalg.LinAlgError:
                L_A = np.tril(cho_factor(A_mat + 1e-6 * np.eye(n_poly), lower=True, check_finite=False)[0])
        else:
            W_F = np.empty((M, 0), dtype=np.float64)
            L_A = np.empty((0, 0), dtype=np.float64)
        out["M_kk"] = L_R
        out["M_kp"] = W_F
        out["M_pp"] = L_A
    return out


class _BaseSurrogate:
    """Common symbolic/numeric interface for C++17 `CompiledSurrogate` models."""

    _compiled: CompiledSurrogate
    _was_1d: bool
    name: str

    @property
    def d_in(self) -> int:
        return int(self._compiled.d_in())

    @property
    def d_out(self) -> int:
        return int(self._compiled.d_out())

    @property
    def has_variance(self) -> bool:
        return bool(self._compiled.has_variance())

    def predict(self, X: np.ndarray, num_threads: int = 0) -> np.ndarray:
        """Evaluate predictive mean on row-major samples `X` of shape `(N, d)` or `(d,)`."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != self.d_in:
                raise ValueError(f"Expected 1D input of length {self.d_in}, got {arr.size}.")
            Y = np.asarray(self._compiled.eval(np.asfortranarray(arr.reshape(self.d_in, 1)), int(num_threads)), dtype=np.float64)
            return float(Y[0, 0]) if self._was_1d else Y[:, 0]
        if arr.ndim != 2 or arr.shape[1] != self.d_in:
            raise ValueError(f"Expected 2D input of shape (N, {self.d_in}), got {arr.shape}.")
        Y = np.asarray(self._compiled.eval(np.asfortranarray(arr.T), int(num_threads)), dtype=np.float64).T
        return Y[:, 0] if self._was_1d else Y

    def predict_variance(self, X: np.ndarray, num_threads: int = 0) -> np.ndarray:
        """Evaluate predictive variance `sigma^2(x)` on row-major samples `X` of shape `(N, d)` or `(d,)`."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != self.d_in:
                raise ValueError(f"Expected 1D input of length {self.d_in}, got {arr.size}.")
            V = np.asarray(self._compiled.eval_variance(np.asfortranarray(arr.reshape(self.d_in, 1)), int(num_threads)), dtype=np.float64)
            return float(V[0, 0]) if self._was_1d else V[:, 0]
        if arr.ndim != 2 or arr.shape[1] != self.d_in:
            raise ValueError(f"Expected 2D input of shape (N, {self.d_in}), got {arr.shape}.")
        V = np.asarray(self._compiled.eval_variance(np.asfortranarray(arr.T), int(num_threads)), dtype=np.float64).T
        return V[:, 0] if self._was_1d else V

    def predict_std(self, X: np.ndarray, num_threads: int = 0) -> np.ndarray:
        """Evaluate predictive standard deviation `sigma(x)` on numeric samples."""
        return np.sqrt(np.maximum(self.predict_variance(X, num_threads=num_threads), 0.0))

    def variance(self, x: Union[Expr, np.ndarray]) -> Union[Expr, np.ndarray]:
        """Differentiable predictive variance `sigma^2(x)` for symbolic `Expr` `(d, B)` or numeric input."""
        if isinstance(x, Expr):
            out_id = op_surrogate(x.graph, x.id, self._compiled, SurrogateOutputMode.Variance, f"{self.name}_var")
            return Expr(x.graph, out_id)
        return self.predict_variance(np.asarray(x, dtype=np.float64))

    def std(self, x: Union[Expr, np.ndarray], eps: float = 1e-12) -> Union[Expr, np.ndarray]:
        """Differentiable predictive standard deviation `sqrt(sigma^2(x) + eps)`."""
        if isinstance(x, Expr):
            return fo_expr.sqrt(self.variance(x) + float(eps))
        return np.sqrt(np.maximum(self.predict_variance(np.asarray(x, dtype=np.float64)), 0.0) + float(eps))

    def __call__(self, x: Union[Expr, np.ndarray]) -> Union[Expr, np.ndarray]:
        """Evaluate predictive mean on symbolic `Expr` `(d, B)` or numeric `np.ndarray`."""
        if isinstance(x, Expr):
            if x.shape[0] != self.d_in:
                raise ValueError(f"Surrogate '{self.name}' expects symbolic input rows == {self.d_in}, got {x.shape}.")
            out_id = op_surrogate(x.graph, x.id, self._compiled, SurrogateOutputMode.Mean, self.name)
            return Expr(x.graph, out_id)
        return self.predict(np.asarray(x, dtype=np.float64))


class RBF(_BaseSurrogate):
    """
    Native C++17 Radial Basis Function (RBF) surrogate with optional polynomial tail
    and exact 0th, 1st, and 2nd-order AD.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        *,
        kernel: Union[str, SurrogateKernel] = "multiquadric",
        length_scale: Union[float, Sequence[float]] = 1.0,
        poly: Union[int, str, None] = "linear",
        reg: float = 1e-10,
        r_eps: float = 1e-14,
        name: str = "rbf",
    ) -> None:
        self.name = str(name)
        kern = resolve_kernel(kernel)
        deg = resolve_poly_degree(poly)
        Xt, Yt, _, in_mean, in_inv_std, out_mean, out_std, was_1d = _normalize_xy(X_train, Y_train)
        N, d = Xt.shape
        m = Yt.shape[1]
        self._was_1d = was_1d

        ls = np.asarray(length_scale, dtype=np.float64).ravel()
        if ls.size == 1:
            theta = np.full(d, 1.0 / max(float(ls[0]) ** 2, 1e-12), dtype=np.float64)
        elif ls.size == d:
            theta = 1.0 / np.maximum(ls * ls, 1e-12)
        else:
            raise ValueError(f"length_scale must be scalar or length {d}.")
        self.theta = theta

        Phi = _build_kernel_matrix(Xt, theta, kern, r_eps=r_eps)
        Phi[np.arange(N), np.arange(N)] += float(reg)
        F = _poly_design_matrix(Xt, deg)
        np_dim = F.shape[1]

        if np_dim > 0:
            # Saddle-point augmented RBF system: [Phi, F; F^T, 0] [alpha; poly_coef] = [Yt; 0]
            A = np.zeros((N + np_dim, N + np_dim), dtype=np.float64)
            A[:N, :N] = Phi
            A[:N, N:] = F
            A[N:, :N] = F.T
            rhs = np.zeros((N + np_dim, m), dtype=np.float64)
            rhs[:N, :] = Yt
            try:
                sol = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
            alpha = sol[:N, :]
            poly_coef = sol[N:, :]
        else:
            try:
                sol = np.linalg.solve(Phi, Yt)
            except np.linalg.LinAlgError:
                sol, _, _, _ = np.linalg.lstsq(Phi, Yt, rcond=None)
            alpha = sol
            poly_coef = np.empty((0, m), dtype=np.float64)

        self._compiled = CompiledSurrogate(
            kern, deg,
            np.asfortranarray(Xt.T),
            np.ascontiguousarray(theta),
            np.asfortranarray(alpha),
            np.empty((0, 0), dtype=np.float64),
            np.asfortranarray(poly_coef),
            np.ascontiguousarray(in_mean),
            np.ascontiguousarray(in_inv_std),
            np.ascontiguousarray(out_mean),
            np.ascontiguousarray(out_std),
            np.empty(0, dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            float(r_eps),
            self.name,
        )


class Kriging(_BaseSurrogate):
    """
    Native C++17 Ordinary / Universal Kriging (Gaussian Process) surrogate with
    anisotropic ARD Maximum Likelihood Estimation (MLE) and exact 0/1/2 AD for
    both predictive mean and predictive variance.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        *,
        kernel: Union[str, SurrogateKernel] = "matern52",
        poly: Union[int, str, None] = "constant",
        theta: Optional[Union[float, Sequence[float]]] = None,
        theta_bounds: Tuple[float, float] = (1e-3, 1e2),
        nugget: float = 1e-8,
        optimize_theta: bool = True,
        n_restarts: int = 3,
        r_eps: float = 1e-14,
        name: str = "kriging",
    ) -> None:
        self.name = str(name)
        kern = resolve_kernel(kernel)
        deg = resolve_poly_degree(poly)
        Xt, Yt, _, in_mean, in_inv_std, out_mean, out_std, was_1d = _normalize_xy(X_train, Y_train)
        N, d = Xt.shape
        self._was_1d = was_1d
        self.nugget = float(nugget)
        F = _poly_design_matrix(Xt, deg)

        if theta is not None and not optimize_theta:
            th = np.asarray(theta, dtype=np.float64).ravel()
            best_theta = np.full(d, float(th[0])) if th.size == 1 else th.copy()
        else:
            log_lo, log_hi = np.log(theta_bounds[0]), np.log(theta_bounds[1])

            def _obj(log_th: np.ndarray) -> float:
                th_try = np.exp(np.clip(log_th, log_lo, log_hi))
                R = _build_kernel_matrix(Xt, th_try, kern, r_eps=r_eps)
                res = _solve_kriging_gls(R, F, Yt, self.nugget, return_matrices=False)
                return float(res["nll"])

            starts: List[np.ndarray] = []
            if theta is not None:
                th0 = np.asarray(theta, dtype=np.float64).ravel()
                starts.append(np.log(np.full(d, float(th0[0])) if th0.size == 1 else th0))
            else:
                starts.append(np.zeros(d, dtype=np.float64))
            rng = np.random.default_rng(1901)
            for _ in range(max(0, int(n_restarts) - 1)):
                starts.append(rng.uniform(max(log_lo, -3.0), min(log_hi, 2.0), size=(d,)))

            best_val = np.inf
            best_log_th = starts[0]
            bounds = [(log_lo, log_hi)] * d
            for s0 in starts:
                opt = minimize(_obj, s0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60})
                if opt.fun < best_val:
                    best_val = float(opt.fun)
                    best_log_th = opt.x.copy()
            best_theta = np.exp(best_log_th)

        self.theta = best_theta
        R = _build_kernel_matrix(Xt, self.theta, kern, r_eps=r_eps)
        fit = _solve_kriging_gls(R, F, Yt, self.nugget, return_matrices=True)
        if not fit["ok"]:
            raise RuntimeError("Kriging GLS Cholesky factorization failed.")
        self.poly_coef = fit["poly_coef"]
        self.sigma2 = fit["sigma2"]

        self._compiled = CompiledSurrogate(
            kern, deg,
            np.asfortranarray(Xt.T),
            np.ascontiguousarray(self.theta),
            np.asfortranarray(fit["weights"]),
            np.empty((0, 0), dtype=np.float64),
            np.asfortranarray(fit["poly_coef"]),
            np.ascontiguousarray(in_mean),
            np.ascontiguousarray(in_inv_std),
            np.ascontiguousarray(out_mean),
            np.ascontiguousarray(out_std),
            np.ascontiguousarray(fit["sigma2"]),
            np.asfortranarray(fit["M_kk"]),
            np.asfortranarray(fit["M_kp"]),
            np.asfortranarray(fit["M_pp"]),
            float(r_eps),
            self.name,
        )


class GEK(_BaseSurrogate):
    """
    Native C++17 Direct Gradient-Enhanced Kriging (GEK) surrogate.

    Incorporates both function values `Y_train` and analytical/adjoint gradients `dY_train`
    into the exact block covariance matrix, with exact 0th, 1st, and 2nd-order C++ AD for
    both predictive mean and predictive variance.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        dY_train: np.ndarray,
        *,
        kernel: Union[str, SurrogateKernel] = "gaussian",
        poly: Union[int, str, None] = "constant",
        theta: Optional[Union[float, Sequence[float]]] = None,
        theta_bounds: Tuple[float, float] = (1e-3, 5e1),
        nugget: float = 1e-8,
        optimize_theta: bool = True,
        n_restarts: int = 2,
        r_eps: float = 1e-14,
        name: str = "gek",
    ) -> None:
        self.name = str(name)
        kern = resolve_kernel(kernel)
        deg = resolve_poly_degree(poly)
        Xt, Yt, dYt, in_mean, in_inv_std, out_mean, out_std, was_1d = _normalize_xy(X_train, Y_train, dY_train)
        assert dYt is not None
        N, d = Xt.shape
        m = Yt.shape[1]
        self._was_1d = was_1d
        self.nugget = float(nugget)

        # Augmented target vector: [Y (N, m); dY (N*d, m)]
        Y_aug = np.vstack([Yt, dYt.reshape((N * d, m))])
        F_aug = np.vstack([_poly_design_matrix(Xt, deg), _poly_grad_design_matrix(Xt, deg)])

        if theta is not None and not optimize_theta:
            th = np.asarray(theta, dtype=np.float64).ravel()
            best_theta = np.full(d, float(th[0])) if th.size == 1 else th.copy()
        else:
            log_lo, log_hi = np.log(theta_bounds[0]), np.log(theta_bounds[1])

            def _obj(log_th: np.ndarray) -> float:
                th_try = np.exp(np.clip(log_th, log_lo, log_hi))
                R = _build_gek_kernel_matrix(Xt, th_try, kern, r_eps=r_eps)
                res = _solve_kriging_gls(R, F_aug, Y_aug, self.nugget, theta=th_try, n_val=N, return_matrices=False)
                return float(res["nll"])

            starts: List[np.ndarray] = []
            if theta is not None:
                th0 = np.asarray(theta, dtype=np.float64).ravel()
                starts.append(np.log(np.full(d, float(th0[0])) if th0.size == 1 else th0))
            else:
                starts.append(np.full(d, -0.7, dtype=np.float64))
            rng = np.random.default_rng(1902)
            for _ in range(max(0, int(n_restarts) - 1)):
                starts.append(rng.uniform(max(log_lo, -2.5), min(log_hi, 1.5), size=(d,)))

            best_val = np.inf
            best_log_th = starts[0]
            bounds = [(log_lo, log_hi)] * d
            for s0 in starts:
                opt = minimize(_obj, s0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 50})
                if opt.fun < best_val:
                    best_val = float(opt.fun)
                    best_log_th = opt.x.copy()
            best_theta = np.exp(best_log_th)

        self.theta = best_theta
        R = _build_gek_kernel_matrix(Xt, self.theta, kern, r_eps=r_eps)
        fit = _solve_kriging_gls(R, F_aug, Y_aug, self.nugget, theta=self.theta, n_val=N, return_matrices=True)
        if not fit["ok"]:
            raise RuntimeError("GEK GLS Cholesky factorization failed.")

        weights = fit["weights"]
        alpha = weights[:N, :]
        beta_grad = weights[N:, :]
        self.poly_coef = fit["poly_coef"]
        self.sigma2 = fit["sigma2"]

        self._compiled = CompiledSurrogate(
            kern, deg,
            np.asfortranarray(Xt.T),
            np.ascontiguousarray(self.theta),
            np.asfortranarray(alpha),
            np.asfortranarray(beta_grad),
            np.asfortranarray(fit["poly_coef"]),
            np.ascontiguousarray(in_mean),
            np.ascontiguousarray(in_inv_std),
            np.ascontiguousarray(out_mean),
            np.ascontiguousarray(out_std),
            np.ascontiguousarray(fit["sigma2"]),
            np.asfortranarray(fit["M_kk"]),
            np.asfortranarray(fit["M_kp"]),
            np.asfortranarray(fit["M_pp"]),
            float(r_eps),
            self.name,
        )


class KPLS(_BaseSurrogate):
    """
    High-Dimensional Partial Least Squares Kriging (KPLS, Bouhlel et al. 2016).

    Projects ARD hyperparameters onto `n_comp` PLS directions (`theta_j = sum_k th_pls_k * W_jk^2`),
    reducing MLE optimization from `d` dimensions to `n_comp` dimensions while compiling to the
    exact same native C++17 `SurrogateNode` with exact 0/1/2 AD for mean and variance.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        *,
        n_comp: int = 2,
        kernel: Union[str, SurrogateKernel] = "gaussian",
        poly: Union[int, str, None] = "constant",
        theta_bounds: Tuple[float, float] = (1e-3, 1e2),
        nugget: float = 1e-8,
        n_restarts: int = 2,
        r_eps: float = 1e-14,
        name: str = "kpls",
    ) -> None:
        self.name = str(name)
        kern = resolve_kernel(kernel)
        deg = resolve_poly_degree(poly)
        Xt, Yt, _, in_mean, in_inv_std, out_mean, out_std, was_1d = _normalize_xy(X_train, Y_train)
        N, d = Xt.shape
        self._was_1d = was_1d
        self.nugget = float(nugget)
        self.W_pls = _fit_pls_weights(Xt, Yt, n_comp=n_comp)
        W2 = self.W_pls * self.W_pls  # (d, h)
        h = W2.shape[1]
        F = _poly_design_matrix(Xt, deg)

        log_lo, log_hi = np.log(theta_bounds[0]), np.log(theta_bounds[1])

        def _map_theta(log_th_pls: np.ndarray) -> np.ndarray:
            th_pls = np.exp(np.clip(log_th_pls, log_lo, log_hi))
            return np.maximum(W2 @ th_pls, 1e-8)

        def _obj(log_th_pls: np.ndarray) -> float:
            R = _build_kernel_matrix(Xt, _map_theta(log_th_pls), kern, r_eps=r_eps)
            res = _solve_kriging_gls(R, F, Yt, self.nugget, return_matrices=False)
            return float(res["nll"])

        starts = [np.zeros(h, dtype=np.float64)]
        rng = np.random.default_rng(1903)
        for _ in range(max(0, int(n_restarts) - 1)):
            starts.append(rng.uniform(max(log_lo, -2.5), min(log_hi, 1.5), size=(h,)))

        best_val = np.inf
        best_log = starts[0]
        for s0 in starts:
            opt = minimize(_obj, s0, method="L-BFGS-B", bounds=[(log_lo, log_hi)] * h, options={"maxiter": 60})
            if opt.fun < best_val:
                best_val = float(opt.fun)
                best_log = opt.x.copy()

        self.theta_pls = np.exp(best_log)
        self.theta = _map_theta(best_log)
        R = _build_kernel_matrix(Xt, self.theta, kern, r_eps=r_eps)
        fit = _solve_kriging_gls(R, F, Yt, self.nugget, return_matrices=True)
        if not fit["ok"]:
            raise RuntimeError("KPLS GLS Cholesky factorization failed.")
        self.poly_coef = fit["poly_coef"]
        self.sigma2 = fit["sigma2"]

        self._compiled = CompiledSurrogate(
            kern, deg,
            np.asfortranarray(Xt.T),
            np.ascontiguousarray(self.theta),
            np.asfortranarray(fit["weights"]),
            np.empty((0, 0), dtype=np.float64),
            np.asfortranarray(fit["poly_coef"]),
            np.ascontiguousarray(in_mean),
            np.ascontiguousarray(in_inv_std),
            np.ascontiguousarray(out_mean),
            np.ascontiguousarray(out_std),
            np.ascontiguousarray(fit["sigma2"]),
            np.asfortranarray(fit["M_kk"]),
            np.asfortranarray(fit["M_kp"]),
            np.asfortranarray(fit["M_pp"]),
            float(r_eps),
            self.name,
        )


class GEKPLS(_BaseSurrogate):
    """
    High-Dimensional Gradient-Enhanced KPLS (GEKPLS).

    Uses sample gradients `dY_train` to identify the top `n_comp` active PLS directions
    (`W_pls`), optimizes only `n_comp` hyperparameters instead of `d`, and fits an exact
    Gradient-Enhanced Kriging system compiled to C++17 `SurrogateNode` with exact 0/1/2 AD.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        dY_train: np.ndarray,
        *,
        n_comp: int = 2,
        kernel: Union[str, SurrogateKernel] = "gaussian",
        poly: Union[int, str, None] = "constant",
        theta_bounds: Tuple[float, float] = (1e-3, 5e1),
        nugget: float = 1e-6,
        n_restarts: int = 2,
        r_eps: float = 1e-14,
        name: str = "gekpls",
    ) -> None:
        self.name = str(name)
        kern = resolve_kernel(kernel)
        deg = resolve_poly_degree(poly)
        Xt, Yt, dYt, in_mean, in_inv_std, out_mean, out_std, was_1d = _normalize_xy(X_train, Y_train, dY_train)
        assert dYt is not None
        N, d = Xt.shape
        m = Yt.shape[1]
        self._was_1d = was_1d
        self.nugget = float(nugget)

        self.W_pls = _fit_pls_weights(Xt, Yt, n_comp=n_comp, dYt=dYt)
        W2 = self.W_pls * self.W_pls
        h = W2.shape[1]

        Y_aug = np.vstack([Yt, dYt.reshape((N * d, m))])
        F_aug = np.vstack([_poly_design_matrix(Xt, deg), _poly_grad_design_matrix(Xt, deg)])
        log_lo, log_hi = np.log(theta_bounds[0]), np.log(theta_bounds[1])

        def _map_theta(log_th_pls: np.ndarray) -> np.ndarray:
            th_pls = np.exp(np.clip(log_th_pls, log_lo, log_hi))
            return np.maximum(W2 @ th_pls, 1e-4)

        def _obj(log_th_pls: np.ndarray) -> float:
            th_try = _map_theta(log_th_pls)
            R = _build_gek_kernel_matrix(Xt, th_try, kern, r_eps=r_eps)
            res = _solve_kriging_gls(R, F_aug, Y_aug, self.nugget, theta=th_try, n_val=N, return_matrices=False)
            return float(res["nll"])

        starts = [np.full(h, -0.7, dtype=np.float64)]
        rng = np.random.default_rng(1904)
        for _ in range(max(0, int(n_restarts) - 1)):
            starts.append(rng.uniform(max(log_lo, -2.5), min(log_hi, 1.5), size=(h,)))

        best_val = np.inf
        best_log = starts[0]
        for s0 in starts:
            opt = minimize(_obj, s0, method="L-BFGS-B", bounds=[(log_lo, log_hi)] * h, options={"maxiter": 50})
            if opt.fun < best_val:
                best_val = float(opt.fun)
                best_log = opt.x.copy()

        self.theta_pls = np.exp(best_log)
        self.theta = _map_theta(best_log)
        R = _build_gek_kernel_matrix(Xt, self.theta, kern, r_eps=r_eps)
        fit = _solve_kriging_gls(R, F_aug, Y_aug, self.nugget, theta=self.theta, n_val=N, return_matrices=True)
        if not fit["ok"]:
            raise RuntimeError("GEKPLS GLS Cholesky factorization failed.")

        weights = fit["weights"]
        self.poly_coef = fit["poly_coef"]
        self.sigma2 = fit["sigma2"]

        self._compiled = CompiledSurrogate(
            kern, deg,
            np.asfortranarray(Xt.T),
            np.ascontiguousarray(self.theta),
            np.asfortranarray(weights[:N, :]),
            np.asfortranarray(weights[N:, :]),
            np.asfortranarray(fit["poly_coef"]),
            np.ascontiguousarray(in_mean),
            np.ascontiguousarray(in_inv_std),
            np.ascontiguousarray(out_mean),
            np.ascontiguousarray(out_std),
            np.ascontiguousarray(fit["sigma2"]),
            np.asfortranarray(fit["M_kk"]),
            np.asfortranarray(fit["M_kp"]),
            np.asfortranarray(fit["M_pp"]),
            float(r_eps),
            self.name,
        )


def _eval_surrogate_grads_at(model: _BaseSurrogate, X_rows: np.ndarray) -> np.ndarray:
    """Compute exact analytical gradients dY/dX of shape (N, d, m) using C++ Evaluator."""
    from ._fastopt_impl import Evaluator

    N, d = X_rows.shape
    m = model.d_out
    gb = GraphBuilder()
    xv = gb.variable((d, 1), init=X_rows[0].reshape(d, 1), name="x_tmp")
    yv = model(xv)
    ev = Evaluator(gb._graph, [yv.id])
    grads = np.empty((N, d, m), dtype=np.float64)
    for i in range(N):
        J = np.asarray(ev.jacobian(X_rows[i], 0), dtype=np.float64)  # (m, d)
        grads[i] = J.T
    return grads


class CoKriging:
    """
    Multi-Fidelity Auto-Regressive Co-Kriging (Kennedy & O'Hagan 2000):
        y_HF(x) = rho * y_LF(x) + delta(x)
        var_HF(x) = rho^2 * var_LF(x) + var_delta(x)

    Fuses abundant low-fidelity samples (e.g. `fast_bemt`) with scarce high-fidelity samples
    (e.g. CFD) into a single differentiable surrogate with exact 0/1/2 AD for both mean and variance.
    Supports optional low-fidelity and/or high-fidelity gradients (Gradient-Enhanced Co-Kriging).
    """

    def __init__(
        self,
        X_lf: np.ndarray,
        Y_lf: np.ndarray,
        X_hf: np.ndarray,
        Y_hf: np.ndarray,
        *,
        dY_lf: Optional[np.ndarray] = None,
        dY_hf: Optional[np.ndarray] = None,
        kernel: Union[str, SurrogateKernel] = "matern52",
        poly: Union[int, str, None] = "constant",
        nugget: float = 1e-8,
        name: str = "cokriging",
    ) -> None:
        self.name = str(name)
        X_lf_arr = np.asarray(X_lf, dtype=np.float64)
        X_hf_arr = np.asarray(X_hf, dtype=np.float64)
        if X_lf_arr.ndim == 1:
            X_lf_arr = X_lf_arr[:, None]
        if X_hf_arr.ndim == 1:
            X_hf_arr = X_hf_arr[:, None]
        Y_lf_arr = np.asarray(Y_lf, dtype=np.float64)
        Y_hf_arr = np.asarray(Y_hf, dtype=np.float64)
        self._was_1d = Y_hf_arr.ndim == 1
        if Y_lf_arr.ndim == 1:
            Y_lf_arr = Y_lf_arr[:, None]
        if Y_hf_arr.ndim == 1:
            Y_hf_arr = Y_hf_arr[:, None]

        # 1. Fit low-fidelity Kriging or GEK model
        lf_kern = "gaussian" if (dY_lf is not None and kernel == "matern52") else kernel
        if dY_lf is not None:
            self.lf_model: _BaseSurrogate = GEK(
                X_lf_arr, Y_lf_arr, dY_lf, kernel=lf_kern, poly=poly, nugget=nugget, name=f"{name}_lf"
            )
        else:
            self.lf_model = Kriging(
                X_lf_arr, Y_lf_arr, kernel=kernel, poly=poly, nugget=nugget, name=f"{name}_lf"
            )

        # 2. Estimate scaling factor rho per output via least squares on Y_hf ~ rho * y_lf(X_hf) + c
        y_lf_at_hf = self.lf_model.predict(X_hf_arr)
        if y_lf_at_hf.ndim == 1:
            y_lf_at_hf = y_lf_at_hf[:, None]
        m = Y_hf_arr.shape[1]
        rho = np.ones(m, dtype=np.float64)
        for c in range(m):
            a = y_lf_at_hf[:, c]
            b = Y_hf_arr[:, c]
            ac = a - a.mean()
            bc = b - b.mean()
            denom = float(ac @ ac)
            if denom > 1e-12:
                rho[c] = float(ac @ bc) / denom
        self.rho = rho

        # 3. Fit discrepancy model delta(x) = Y_hf - rho * y_lf(X_hf)
        diff_Y = Y_hf_arr - y_lf_at_hf * rho[None, :]
        if dY_hf is not None:
            dY_hf_arr = np.asarray(dY_hf, dtype=np.float64)
            if dY_hf_arr.ndim == 2:
                dY_hf_arr = dY_hf_arr[:, :, None]
            dY_lf_at_hf = _eval_surrogate_grads_at(self.lf_model, X_hf_arr)
            diff_dY = dY_hf_arr - dY_lf_at_hf * rho[None, None, :]
            hf_kern = "gaussian" if kernel == "matern52" else kernel
            self.diff_model: _BaseSurrogate = GEK(
                X_hf_arr, diff_Y, diff_dY, kernel=hf_kern, poly=poly, nugget=nugget, name=f"{name}_diff"
            )
        else:
            self.diff_model = Kriging(
                X_hf_arr, diff_Y, kernel=kernel, poly=poly, nugget=nugget, name=f"{name}_diff"
            )

    @property
    def d_in(self) -> int:
        return self.lf_model.d_in

    @property
    def d_out(self) -> int:
        return self.lf_model.d_out

    def predict(self, X: np.ndarray, num_threads: int = 0) -> np.ndarray:
        yl = self.lf_model.predict(X, num_threads=num_threads)
        yd = self.diff_model.predict(X, num_threads=num_threads)
        if np.ndim(yl) == 1:
            out = self.rho * yl + yd
            return float(out[0]) if self._was_1d else out
        out = yl * self.rho[None, :] + yd
        return out[:, 0] if self._was_1d else out

    def predict_variance(self, X: np.ndarray, num_threads: int = 0) -> np.ndarray:
        vl = self.lf_model.predict_variance(X, num_threads=num_threads)
        vd = self.diff_model.predict_variance(X, num_threads=num_threads)
        rho2 = self.rho * self.rho
        if np.ndim(vl) == 1:
            out = rho2 * vl + vd
            return float(out[0]) if self._was_1d else out
        out = vl * rho2[None, :] + vd
        return out[:, 0] if self._was_1d else out

    def predict_std(self, X: np.ndarray, num_threads: int = 0) -> np.ndarray:
        return np.sqrt(np.maximum(self.predict_variance(X, num_threads=num_threads), 0.0))

    def variance(self, x: Union[Expr, np.ndarray]) -> Union[Expr, np.ndarray]:
        if isinstance(x, Expr):
            vl = self.lf_model.variance(x)
            vd = self.diff_model.variance(x)
            rho2_mat = np.broadcast_to((self.rho * self.rho)[:, None], (self.d_out, x.shape[1])).copy()
            return vl * rho2_mat + vd
        return self.predict_variance(np.asarray(x, dtype=np.float64))

    def std(self, x: Union[Expr, np.ndarray], eps: float = 1e-12) -> Union[Expr, np.ndarray]:
        if isinstance(x, Expr):
            return fo_expr.sqrt(self.variance(x) + float(eps))
        return np.sqrt(np.maximum(self.predict_variance(np.asarray(x, dtype=np.float64)), 0.0) + float(eps))

    def __call__(self, x: Union[Expr, np.ndarray]) -> Union[Expr, np.ndarray]:
        if isinstance(x, Expr):
            yl = self.lf_model(x)
            yd = self.diff_model(x)
            rho_mat = np.broadcast_to(self.rho[:, None], (self.d_out, x.shape[1])).copy()
            return yl * rho_mat + yd
        return self.predict(np.asarray(x, dtype=np.float64))


# ============================================================================
# Differentiable Acquisition Functions for Gradient-Based Infill (EGO / BO)
# ============================================================================
def lower_confidence_bound(
    model: Union[_BaseSurrogate, CoKriging],
    x: Union[Expr, np.ndarray],
    *,
    kappa: float = 2.0,
    eps: float = 1e-12,
) -> Union[Expr, np.ndarray]:
    """Differentiable Lower Confidence Bound: `LCB(x) = mu(x) - kappa * sqrt(var(x) + eps)`."""
    return model(x) - float(kappa) * model.std(x, eps=eps)


def expected_improvement(
    model: Union[_BaseSurrogate, CoKriging],
    x: Union[Expr, np.ndarray],
    y_best: float,
    *,
    xi: float = 0.0,
    eps: float = 1e-12,
) -> Union[Expr, np.ndarray]:
    """
    Smooth C-infinity differentiable Expected Improvement (for minimization):
        EI(x) = (y_best - mu(x) - xi) * Phi(z) + sigma(x) * phi(z)
    Uses a smooth tanh approximation to the standard normal CDF Phi(z) so exact
    1st and 2nd-order derivatives work natively inside `fastopt.Opti`.
    """
    mu = model(x)
    sig = model.std(x, eps=eps)
    imp = float(y_best) - mu - float(xi)
    z = imp / sig
    c_inv_sqrt2pi = 0.3989422804014327
    # High-accuracy C^inf approximation to Phi(z): 0.5 * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
    c_s2pi = 0.7978845608028654
    # Smoothly cap z to [-10, 10] before Phi(z)/phi(z) to prevent z^3 overflow & 0*inf=NaN in 2nd-order AD
    if isinstance(x, Expr):
        z_s = 10.0 * fo_expr.tanh(z / 10.0)
        phi_z = c_inv_sqrt2pi * fo_expr.exp(-0.5 * z_s * z_s)
        Phi_z = 0.5 * (1.0 + fo_expr.tanh(c_s2pi * (z_s + 0.044715 * z_s * z_s * z_s)))
        return imp * Phi_z + sig * phi_z
    z_s = 10.0 * np.tanh(z / 10.0)
    phi_z = c_inv_sqrt2pi * np.exp(-0.5 * z_s * z_s)
    Phi_z = 0.5 * (1.0 + np.tanh(c_s2pi * (z_s + 0.044715 * z_s * z_s * z_s)))
    return imp * Phi_z + sig * phi_z


__all__ = [
    "SurrogateKernel",
    "SurrogateOutputMode",
    "CompiledSurrogate",
    "RBF",
    "Kriging",
    "GEK",
    "KPLS",
    "GEKPLS",
    "CoKriging",
    "lower_confidence_bound",
    "expected_improvement",
    "resolve_kernel",
    "resolve_poly_degree",
]
