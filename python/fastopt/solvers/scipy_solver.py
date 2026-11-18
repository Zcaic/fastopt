"""
SciPy optimize backend adapter for fastopt (SLSQP, trust-constr, L-BFGS-B, BFGS, etc.).
Features:
  - Same-point evaluation caching
  - Automatic variable scaling (x = var_scale * x_tilde)
  - True sparse CSC Jacobian and Hessian integration for large-scale `trust-constr`
  - KKT Lagrange multiplier extraction (from `trust-constr` duals or active-set KKT recovery)
  - Per-iteration convergence history and user callback support
All comments are written in English.
"""

from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import optimize as sp_opt
from scipy import sparse as sp_sparse

from .. import _fastopt_impl as _core


class _SamePointCache:
    """Cache forward evaluations and gradients/Jacobians at the current physical iterate x."""

    def __init__(self, evaluator: _core.Evaluator) -> None:
        self.ev = evaluator
        self._x_eval: Optional[np.ndarray] = None
        self._outputs: Optional[List[np.ndarray]] = None
        self._x_jac: Dict[int, np.ndarray] = {}
        self._jac_cache: Dict[int, np.ndarray] = {}
        self._x_sjac: Dict[int, np.ndarray] = {}
        self._sjac_cache: Dict[int, np.ndarray] = {}

    def get_outputs(self, x: np.ndarray) -> List[np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        if self._x_eval is None or not np.array_equal(x_arr, self._x_eval):
            self._x_eval = x_arr.copy()
            self._outputs = self.ev.eval(self._x_eval)
        assert self._outputs is not None
        return self._outputs

    def get_grad_or_jac(self, x: np.ndarray, out_idx: int) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        prev_x = self._x_jac.get(out_idx)
        if prev_x is None or not np.array_equal(x_arr, prev_x):
            self._x_jac[out_idx] = x_arr.copy()
            if out_idx == 0:
                self._jac_cache[out_idx] = self.ev.gradient(x_arr, 0)
            else:
                self._jac_cache[out_idx] = self.ev.jacobian(x_arr, out_idx)
        return self._jac_cache[out_idx]

    def get_sparse_jac_vals(self, x: np.ndarray, out_idx: int) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        prev_x = self._x_sjac.get(out_idx)
        if prev_x is None or not np.array_equal(x_arr, prev_x):
            self._x_sjac[out_idx] = x_arr.copy()
            self._sjac_cache[out_idx] = self.ev.sparse_jacobian_values(x_arr, out_idx)
        return self._sjac_cache[out_idx]


def _compute_max_violation(
    outs: List[np.ndarray],
    x_phys: np.ndarray,
    lb_phys: np.ndarray,
    ub_phys: np.ndarray,
    eq_idx: Optional[int],
    ineq_idx: Optional[int],
) -> float:
    viol = 0.0
    if x_phys.size > 0:
        b_viol = float(np.max(np.maximum(0.0, np.maximum(lb_phys - x_phys, x_phys - ub_phys))))
        viol = max(viol, b_viol)
    if eq_idx is not None:
        eq_v = outs[eq_idx].ravel(order="F")
        if eq_v.size > 0:
            viol = max(viol, float(np.max(np.abs(eq_v))))
    if ineq_idx is not None:
        ineq_v = outs[ineq_idx].ravel(order="F")
        if ineq_v.size > 0:
            viol = max(viol, float(np.max(np.maximum(0.0, ineq_v))))
    return viol


def _recover_kkt_multipliers(
    cache: _SamePointCache,
    x_opt: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    eq_idx: Optional[int],
    ineq_idx: Optional[int],
    active_tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover KKT multipliers (mult_eq, mult_ineq, mult_x_L, mult_x_U) in the scaled Evaluator space
    from the first-order KKT stationarity equation:
        grad_f(x*) + J_eq^T * lam_eq + J_ineq_active^T * lam_ineq + lam_ub - lam_lb = 0
    """
    n = x_opt.size
    outs = cache.get_outputs(x_opt)
    grad_f = cache.get_grad_or_jac(x_opt, 0).ravel()

    m_eq = outs[eq_idx].size if eq_idx is not None else 0
    m_ineq = outs[ineq_idx].size if ineq_idx is not None else 0

    mult_eq = np.zeros(m_eq, dtype=np.float64)
    mult_ineq = np.zeros(m_ineq, dtype=np.float64)
    mult_x_L = np.zeros(n, dtype=np.float64)
    mult_x_U = np.zeros(n, dtype=np.float64)

    cols: List[np.ndarray] = []
    col_kinds: List[Tuple[str, int]] = []

    if eq_idx is not None and m_eq > 0:
        J_eq = cache.get_grad_or_jac(x_opt, eq_idx)
        for i in range(m_eq):
            cols.append(J_eq[i, :])
            col_kinds.append(("eq", i))

    if ineq_idx is not None and m_ineq > 0:
        g_val = outs[ineq_idx].ravel(order="F")
        J_ineq = cache.get_grad_or_jac(x_opt, ineq_idx)
        for i in range(m_ineq):
            if g_val[i] >= -active_tol:
                cols.append(J_ineq[i, :])
                col_kinds.append(("ineq", i))

    for j in range(n):
        if np.isfinite(lb[j]) and (x_opt[j] - lb[j]) <= active_tol:
            e_j = np.zeros(n, dtype=np.float64)
            e_j[j] = -1.0
            cols.append(e_j)
            col_kinds.append(("lb", j))
        elif np.isfinite(ub[j]) and (ub[j] - x_opt[j]) <= active_tol:
            e_j = np.zeros(n, dtype=np.float64)
            e_j[j] = 1.0
            cols.append(e_j)
            col_kinds.append(("ub", j))

    if cols:
        A = np.column_stack(cols)  # shape (n, n_active)
        lam_sol, _, _, _ = np.linalg.lstsq(A, -grad_f, rcond=1e-10)
        for val, (kind, idx) in zip(lam_sol, col_kinds):
            if kind == "eq":
                mult_eq[idx] = float(val)
            elif kind == "ineq":
                mult_ineq[idx] = max(0.0, float(val))
            elif kind == "lb":
                mult_x_L[idx] = max(0.0, float(val))
            elif kind == "ub":
                mult_x_U[idx] = max(0.0, float(val))

    return mult_eq, mult_ineq, mult_x_L, mult_x_U


def solve_scipy(
    evaluator: _core.Evaluator,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    has_eq: bool,
    has_ineq: bool,
    method: str = "SLSQP",
    options: Optional[Dict[str, Any]] = None,
    use_hessian: bool = True,
    var_scale: Optional[np.ndarray] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Solve the compiled NLP problem using scipy.optimize.minimize.
    Output convention in evaluator:
      - out_idx = 0: scalar objective f(x)
      - if has_eq: out_idx = 1 (equality constraints h(x) == 0)
      - if has_ineq: out_idx = (2 if has_eq else 1) (inequality constraints g(x) <= 0)
    """
    t_start = time.perf_counter()
    cache = _SamePointCache(evaluator)
    n_vars = evaluator.num_vars()
    opts = dict(options) if options is not None else {}

    if var_scale is not None:
        d_x = np.asarray(var_scale, dtype=np.float64).ravel()
        if d_x.size != n_vars:
            raise ValueError("var_scale size does not match num_vars.")
        has_var_scale = bool(np.any(d_x != 1.0))
    else:
        d_x = np.ones(n_vars, dtype=np.float64)
        has_var_scale = False

    def to_phys(x_tilde: np.ndarray) -> np.ndarray:
        arr = np.asarray(x_tilde, dtype=np.float64).ravel()
        return arr * d_x if has_var_scale else arr

    eq_idx = 1 if has_eq else None
    ineq_idx = (2 if has_eq else 1) if has_ineq else None

    # Sparsity structures for large-scale trust-constr
    h_rows, h_cols = evaluator.hessian_sparsity()
    use_sparse_hess = (
        n_vars >= 25 and (len(h_rows) < 0.35 * (n_vars * (n_vars + 1) // 2))
    )

    def _build_sparse_sym_hessian(vals_tril: np.ndarray) -> sp_sparse.csc_matrix:
        if vals_tril.size == 0:
            return sp_sparse.csc_matrix((n_vars, n_vars), dtype=np.float64)
        scaled_vals = vals_tril * (d_x[h_rows] * d_x[h_cols]) if has_var_scale else vals_tril
        off_mask = h_rows != h_cols
        all_r = np.concatenate([h_rows, h_cols[off_mask]])
        all_c = np.concatenate([h_cols, h_rows[off_mask]])
        all_v = np.concatenate([scaled_vals, scaled_vals[off_mask]])
        return sp_sparse.csc_matrix((all_v, (all_r, all_c)), shape=(n_vars, n_vars))

    def obj_fun(x_tilde: np.ndarray) -> float:
        x_p = to_phys(x_tilde)
        return float(cache.get_outputs(x_p)[0].item())

    def obj_grad(x_tilde: np.ndarray) -> np.ndarray:
        x_p = to_phys(x_tilde)
        g = cache.get_grad_or_jac(x_p, 0)
        return g * d_x if has_var_scale else g

    def obj_hess(x_tilde: np.ndarray) -> Any:
        x_p = to_phys(x_tilde)
        if use_sparse_hess:
            vals = evaluator.sparse_hessian_values(x_p, 1.0, [])
            return _build_sparse_sym_hessian(vals)
        H = evaluator.hessian(x_p, 0)
        return (H * d_x[np.newaxis, :]) * d_x[:, np.newaxis] if has_var_scale else H

    def obj_hessp(x_tilde: np.ndarray, v_tilde: np.ndarray) -> np.ndarray:
        x_p = to_phys(x_tilde)
        v_arr = np.asarray(v_tilde, dtype=np.float64).ravel()
        v_p = v_arr * d_x if has_var_scale else v_arr
        hv = evaluator.hessian_vec_prod(x_p, 1.0, [], v_p)
        return hv * d_x if has_var_scale else hv

    lb_phys = np.asarray(lb, dtype=np.float64).ravel()
    ub_phys = np.asarray(ub, dtype=np.float64).ravel()
    lb_tilde = lb_phys / d_x if has_var_scale else lb_phys
    ub_tilde = ub_phys / d_x if has_var_scale else ub_phys

    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
    if n_vars > 0 and (np.any(np.isfinite(lb_tilde)) or np.any(np.isfinite(ub_tilde))):
        bounds = [
            (None if not np.isfinite(lb_tilde[i]) else float(lb_tilde[i]),
             None if not np.isfinite(ub_tilde[i]) else float(ub_tilde[i]))
            for i in range(n_vars)
        ]

    method_upper = method.upper()
    constraints: Any = ()
    hess_arg: Any = None
    hessp_arg: Any = None
    jac_arg: Any = obj_grad

    if method_upper in ("NELDER-MEAD", "POWELL"):
        jac_arg = None

    if method_upper == "SLSQP":
        con_list = []
        if eq_idx is not None:
            con_list.append({
                "type": "eq",
                "fun": lambda xt, idx=eq_idx: cache.get_outputs(to_phys(xt))[idx].ravel(order="F"),
                "jac": lambda xt, idx=eq_idx: (
                    cache.get_grad_or_jac(to_phys(xt), idx) * d_x[np.newaxis, :]
                    if has_var_scale
                    else cache.get_grad_or_jac(to_phys(xt), idx)
                ),
            })
        if ineq_idx is not None:
            con_list.append({
                "type": "ineq",
                "fun": lambda xt, idx=ineq_idx: -cache.get_outputs(to_phys(xt))[idx].ravel(order="F"),
                "jac": lambda xt, idx=ineq_idx: -(
                    cache.get_grad_or_jac(to_phys(xt), idx) * d_x[np.newaxis, :]
                    if has_var_scale
                    else cache.get_grad_or_jac(to_phys(xt), idx)
                ),
            })
        constraints = con_list

    elif method_upper == "TRUST-CONSTR":
        con_list = []
        if eq_idx is not None:
            m_eq = evaluator.output_size(eq_idx)
            eq_r, eq_c = evaluator.jacobian_sparsity(eq_idx)
            use_sparse_eq_jac = (m_eq * n_vars >= 100) and (len(eq_r) < 0.35 * m_eq * n_vars)

            def eq_jac(xt, idx=eq_idx, sp_flag=use_sparse_eq_jac, r_idx=eq_r, c_idx=eq_c, m_rows=m_eq):
                x_p = to_phys(xt)
                if sp_flag:
                    vals = cache.get_sparse_jac_vals(x_p, idx)
                    if has_var_scale and vals.size > 0:
                        vals = vals * d_x[c_idx]
                    return sp_sparse.csc_matrix((vals, (r_idx, c_idx)), shape=(m_rows, n_vars))
                J = cache.get_grad_or_jac(x_p, idx)
                return J * d_x[np.newaxis, :] if has_var_scale else J

            eq_kwargs: Dict[str, Any] = {}
            if use_hessian:
                def eq_hess(xt, v, idx=eq_idx):
                    x_p = to_phys(xt)
                    mults = [np.asarray(v, dtype=np.float64).ravel()]
                    if use_sparse_hess:
                        vals = evaluator.sparse_hessian_values(x_p, 0.0, mults)
                        return _build_sparse_sym_hessian(vals)
                    H = evaluator.lagrangian_hessian(x_p, 0.0, mults)
                    return (H * d_x[np.newaxis, :]) * d_x[:, np.newaxis] if has_var_scale else H
                eq_kwargs["hess"] = eq_hess

            con_list.append(
                sp_opt.NonlinearConstraint(
                    fun=lambda xt, idx=eq_idx: cache.get_outputs(to_phys(xt))[idx].ravel(order="F"),
                    lb=np.zeros(m_eq),
                    ub=np.zeros(m_eq),
                    jac=eq_jac,
                    **eq_kwargs,
                )
            )
        if ineq_idx is not None:
            m_ineq = evaluator.output_size(ineq_idx)
            in_r, in_c = evaluator.jacobian_sparsity(ineq_idx)
            use_sparse_in_jac = (m_ineq * n_vars >= 100) and (len(in_r) < 0.35 * m_ineq * n_vars)

            def ineq_jac(xt, idx=ineq_idx, sp_flag=use_sparse_in_jac, r_idx=in_r, c_idx=in_c, m_rows=m_ineq):
                x_p = to_phys(xt)
                if sp_flag:
                    vals = cache.get_sparse_jac_vals(x_p, idx)
                    if has_var_scale and vals.size > 0:
                        vals = vals * d_x[c_idx]
                    return sp_sparse.csc_matrix((vals, (r_idx, c_idx)), shape=(m_rows, n_vars))
                J = cache.get_grad_or_jac(x_p, idx)
                return J * d_x[np.newaxis, :] if has_var_scale else J

            ineq_kwargs: Dict[str, Any] = {}
            if use_hessian:
                def ineq_hess(xt, v, idx=ineq_idx):
                    x_p = to_phys(xt)
                    mults: List[np.ndarray] = []
                    if eq_idx is not None:
                        mults.append(np.zeros(evaluator.output_size(eq_idx)))
                    mults.append(np.asarray(v, dtype=np.float64).ravel())
                    if use_sparse_hess:
                        vals = evaluator.sparse_hessian_values(x_p, 0.0, mults)
                        return _build_sparse_sym_hessian(vals)
                    H = evaluator.lagrangian_hessian(x_p, 0.0, mults)
                    return (H * d_x[np.newaxis, :]) * d_x[:, np.newaxis] if has_var_scale else H
                ineq_kwargs["hess"] = ineq_hess

            con_list.append(
                sp_opt.NonlinearConstraint(
                    fun=lambda xt, idx=ineq_idx: cache.get_outputs(to_phys(xt))[idx].ravel(order="F"),
                    lb=np.full(m_ineq, -np.inf),
                    ub=np.zeros(m_ineq),
                    jac=ineq_jac,
                    **ineq_kwargs,
                )
            )
        constraints = con_list
        if use_hessian:
            hess_arg = obj_hess

    elif method_upper in ("NEWTON-CG", "DOGLEG", "TRUST-NCG", "TRUST-KRYLOV", "TRUST-EXACT"):
        if use_hessian:
            if method_upper in ("NEWTON-CG", "TRUST-NCG", "TRUST-KRYLOV"):
                hessp_arg = obj_hessp
            else:
                hess_arg = obj_hess

    x0_phys = np.clip(np.asarray(x0, dtype=np.float64).ravel(), lb_phys, ub_phys)
    x0_tilde = x0_phys / d_x if has_var_scale else x0_phys

    history: List[Dict[str, Any]] = []

    class _ScipyUserStop(Exception):
        def __init__(self, xt_stop: np.ndarray, f_stop: float) -> None:
            self.x = xt_stop
            self.fun = f_stop
            self.success = True
            self.status = 5
            self.message = "User_Requested_Stop: Optimization stopped early by user callback."
            self.nit = len(history)

    def _record_step(xt: np.ndarray) -> None:
        xp = to_phys(xt)
        outs = cache.get_outputs(xp)
        f_val = float(outs[0].item())
        c_viol = _compute_max_violation(outs, xp, lb_phys, ub_phys, eq_idx, ineq_idx)
        entry = {
            "iter": len(history) + 1,
            "x": xp.copy(),
            "obj": f_val,
            "constr_viol": c_viol,
        }
        history.append(entry)
        if callback is not None:
            ret_cb = callback(entry)
            if ret_cb is False:
                raise _ScipyUserStop(np.asarray(xt, dtype=np.float64).copy(), f_val)

    if method_upper == "TRUST-CONSTR":
        def scipy_cb(xk, state=None):
            _record_step(xk)
            return False
    else:
        def scipy_cb(xk):
            _record_step(xk)

    minimize_kwargs: Dict[str, Any] = {
        "method": method,
        "jac": jac_arg,
        "bounds": bounds,
        "options": opts,
        "callback": scipy_cb,
    }
    if constraints:
        minimize_kwargs["constraints"] = constraints
    if hess_arg is not None:
        minimize_kwargs["hess"] = hess_arg
    if hessp_arg is not None:
        minimize_kwargs["hessp"] = hessp_arg

    try:
        res = sp_opt.minimize(obj_fun, x0_tilde, **minimize_kwargs)
    except _ScipyUserStop as stop_exc:
        res = stop_exc
    elapsed = time.perf_counter() - t_start

    x_opt_phys = to_phys(res.x)
    if not history:
        _record_step(res.x)

    # Extract or recover KKT multipliers in the scaled Evaluator space
    m_eq = evaluator.output_size(eq_idx) if eq_idx is not None else 0
    m_ineq = evaluator.output_size(ineq_idx) if ineq_idx is not None else 0

    if method_upper == "TRUST-CONSTR" and hasattr(res, "v") and res.v is not None and len(res.v) > 0:
        v_list = list(res.v)
        v_cursor = 0
        if has_eq:
            mult_eq = np.asarray(v_list[v_cursor], dtype=np.float64).ravel()
            v_cursor += 1
        else:
            mult_eq = np.zeros(0, dtype=np.float64)
        if has_ineq:
            mult_ineq = np.maximum(0.0, np.asarray(v_list[v_cursor], dtype=np.float64).ravel())
            v_cursor += 1
        else:
            mult_ineq = np.zeros(0, dtype=np.float64)
        if v_cursor < len(v_list):
            v_b = np.asarray(v_list[v_cursor], dtype=np.float64).ravel() / d_x
            mult_x_L = np.maximum(0.0, -v_b)
            mult_x_U = np.maximum(0.0, v_b)
        else:
            _, _, mult_x_L, mult_x_U = _recover_kkt_multipliers(
                cache, x_opt_phys, lb_phys, ub_phys, eq_idx, ineq_idx
            )
    else:
        mult_eq, mult_ineq, mult_x_L, mult_x_U = _recover_kkt_multipliers(
            cache, x_opt_phys, lb_phys, ub_phys, eq_idx, ineq_idx
        )

    mult_g = np.concatenate([mult_eq, mult_ineq]) if (m_eq + m_ineq) > 0 else np.zeros(0, dtype=np.float64)

    return {
        "x": x_opt_phys,
        "f": float(res.fun),
        "mult_g": mult_g,
        "mult_eq": mult_eq,
        "mult_ineq": mult_ineq,
        "mult_x_L": mult_x_L,
        "mult_x_U": mult_x_U,
        "history": history,
        "success": bool(res.success),
        "status": int(getattr(res, "status", 0)),
        "message": str(getattr(res, "message", "")),
        "nit": int(getattr(res, "nit", len(history))),
        "nfev": int(getattr(res, "nfev", 0)),
        "njev": int(getattr(res, "njev", 0)),
        "elapsed_time": elapsed,
        "raw": res,
    }
