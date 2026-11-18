"""
CasADi Opti-style declarative optimization modeling interface for fastopt.
Inherits from GraphBuilder so that both build_evaluator(...) and solve() are available on Opti.
Supports:
  - Gradient-based solvers (IPOPT, SLSQP, trust-constr, L-BFGS-B)
  - Evolutionary solvers via pymoo (DE, GA, PSO, CMAES, ISRES, SRES, NSGA2, NSGA3, MOEAD, SMSEMOA, AGEMOEA)
  - Automatic and manual variable/constraint/objective scaling (`scale=...`, `auto_scale=True`)
  - KKT Lagrange multiplier extraction (`sol.dual(con)`, `sol.bound_dual_lb(var)`, `sol.bound_dual_ub(var)`)
  - Per-iteration callbacks & convergence history (`opti.callback(fn)`, `sol.history`)
  - Infeasibility & bound violation diagnostics (`opti.debug_infeasibility(x)`, `sol.debug_infeasibility()`)
All comments are written in English.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

from . import _fastopt_impl as _core
from .expr import (
    ArrayLike,
    ComparisonExpr,
    Expr,
    GraphBuilder,
    ShapeLike,
    _coerce_with_graph,
    _normalize_shape,
    _to_fortran_flat,
    concat,
)
from .solvers.ipopt_solver import solve_ipopt
from .solvers.pymoo_solver import build_pymoo_problem, solve_pymoo
from .solvers.scipy_solver import solve_scipy

_PYMOO_METHODS = {
    "de",
    "ga",
    "pso",
    "cmaes",
    "isres",
    "sres",
    "patternsearch",
    "ps",
    "neldermead",
    "nm",
    "nsga2",
    "nsga3",
    "moead",
    "smsemoa",
    "sms-emoa",
    "agemoea",
    "agemoea2",
}


class OptiSol:
    """Solution container returned by Opti.solve()."""

    def __init__(
        self,
        opti: Opti,
        evaluator: _core.Evaluator,
        stats: Dict[str, Any],
        dual_map: Dict[int, Dict[str, Any]],
        var_dual_map: Dict[int, Tuple[int, int, Tuple[int, int]]],
        obj_scale: float = 1.0,
        eq_scale_vec: Optional[np.ndarray] = None,
        ineq_scale_vec: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        has_eq: bool = False,
        has_ineq: bool = False,
    ) -> None:
        """Initialize an `OptiSol` solution container from solver output statistics and scaling metadata."""
        self._opti = opti
        self._evaluator = evaluator
        self._stats = stats
        self._x = stats["x"]
        self._dual_map = dual_map
        self._var_dual_map = var_dual_map
        self._obj_scale = float(obj_scale)
        self._eq_scale_vec = eq_scale_vec if eq_scale_vec is not None else np.zeros(0, dtype=np.float64)
        self._ineq_scale_vec = ineq_scale_vec if ineq_scale_vec is not None else np.zeros(0, dtype=np.float64)
        self._lb = lb if lb is not None else np.full(self._x.size, -np.inf)
        self._ub = ub if ub is not None else np.full(self._x.size, np.inf)
        self._has_eq = bool(has_eq)
        self._has_ineq = bool(has_ineq)

    @property
    def x(self) -> np.ndarray:
        """Return a copy of the optimal 1D decision vector `x*` of shape `(num_vars,)`."""
        return self._x.copy()

    @property
    def f(self) -> Union[float, np.ndarray]:
        """Return the optimal objective value `f(x*)` in physical (unscaled) units."""
        f_val = self._stats["f"]
        if isinstance(f_val, np.ndarray):
            return f_val.copy()
        return float(f_val)

    @property
    def pareto_x(self) -> Optional[np.ndarray]:
        """Return Pareto set decision matrix (n_pareto, n_vars) for multi-objective pymoo runs."""
        px = self._stats.get("pareto_x", None)
        return px.copy() if isinstance(px, np.ndarray) else None

    @property
    def pareto_f(self) -> Optional[np.ndarray]:
        """Return Pareto front objective matrix (n_pareto, n_obj) for multi-objective pymoo runs."""
        pf = self._stats.get("pareto_f", None)
        return pf.copy() if isinstance(pf, np.ndarray) else None

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Return per-iteration convergence history `[{'iter': k, 'obj': ..., 'constr_viol': ...}, ...]`."""
        return list(self._stats.get("history", []))

    @property
    def success(self) -> bool:
        """Return `True` if the optimizer converged to a feasible optimal solution."""
        return bool(self._stats["success"])

    @property
    def message(self) -> str:
        """Return the solver termination status message."""
        return str(self._stats["message"])

    @property
    def stats(self) -> Dict[str, Any]:
        """Return raw solver statistics dictionary (`nit`, `elapsed_time`, `status`, dual vectors, etc.)."""
        return self._stats

    def value(self, expr: Union[Expr, float, int]) -> Union[float, np.ndarray]:
        """Evaluate any variable, parameter, or intermediate Expr at the optimal solution."""
        if not isinstance(expr, Expr):
            return float(expr) if np.ndim(expr) == 0 else np.asarray(expr, dtype=np.float64)
        if not expr.graph.is_compiled():
            tmp_ev = _core.Evaluator(expr.graph, [expr.id])
            mat = tmp_ev.eval(self._x)[0]
        else:
            mat = self._evaluator.eval_node(self._x, expr.id)
        if mat.size == 1:
            return float(mat.item())
        if mat.shape[1] == 1:
            return mat.ravel()
        return mat

    def dual(self, constraint: Union[ComparisonExpr, Expr]) -> Union[float, np.ndarray]:
        """
        Return the KKT Lagrange multiplier(s) associated with a registered constraint or simple bound
        in physical units (sensitivity d(f_opt) / d(constraint_violation)).
        """
        key = id(constraint)
        if key not in self._dual_map:
            raise KeyError(
                "Constraint handle not found in solution dual map. Pass the constraint expression "
                "returned by (or passed to) opti.subject_to(con)."
            )
        rec = self._dual_map[key]
        kind = rec["kind"]
        if kind == "eq":
            arr = self._stats.get("mult_eq", np.zeros(0, dtype=np.float64))[rec["start"] : rec["stop"]]
        elif kind == "ineq":
            arr = self._stats.get("mult_ineq", np.zeros(0, dtype=np.float64))[rec["start"] : rec["stop"]]
        elif kind == "bound_ub":
            arr = self._stats.get("mult_x_U", np.zeros(0, dtype=np.float64))[rec["start"] : rec["stop"]]
        elif kind == "bound_lb":
            arr = self._stats.get("mult_x_L", np.zeros(0, dtype=np.float64))[rec["start"] : rec["stop"]]
        elif kind == "inactive":
            arr = np.zeros(rec["size"], dtype=np.float64)
        else:
            raise RuntimeError(f"Unknown dual constraint kind: {kind}")

        if arr.size == 1:
            return float(arr.item())
        shape = rec["shape"]
        if shape[1] == 1:
            return arr.copy()
        return arr.reshape(shape, order="F")

    def bound_dual_lb(self, var: Expr) -> Union[float, np.ndarray]:
        """Return the physical KKT multiplier(s) for the lower bound `var >= lb`."""
        if var.id not in self._var_dual_map:
            raise KeyError(f"Variable {var} is not an active decision variable in this Opti instance.")
        start, stop, shape = self._var_dual_map[var.id]
        arr = self._stats.get("mult_x_L", np.zeros(0, dtype=np.float64))[start:stop]
        if arr.size == 1:
            return float(arr.item())
        return arr.copy() if shape[1] == 1 else arr.reshape(shape, order="F")

    def bound_dual_ub(self, var: Expr) -> Union[float, np.ndarray]:
        """Return the physical KKT multiplier(s) for the upper bound `var <= ub`."""
        if var.id not in self._var_dual_map:
            raise KeyError(f"Variable {var} is not an active decision variable in this Opti instance.")
        start, stop, shape = self._var_dual_map[var.id]
        arr = self._stats.get("mult_x_U", np.zeros(0, dtype=np.float64))[start:stop]
        if arr.size == 1:
            return float(arr.item())
        return arr.copy() if shape[1] == 1 else arr.reshape(shape, order="F")

    def debug_infeasibility(self, tol: float = 1e-5, print_report: bool = False) -> Dict[str, Any]:
        """Inspect bound and constraint violations at the returned solution x."""
        return self._opti.debug_infeasibility(x=self._x, tol=tol, print_report=print_report)

    def value_batch(self, expr: Expr, X_pop: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate any Expr across a population matrix X_pop (defaults to sol.pareto_x)
        in parallel via C++ Evaluator.eval_batch.
        """
        X_mat = self.pareto_x if X_pop is None else np.asarray(X_pop, dtype=np.float64)
        if X_mat is None:
            X_mat = self._x.reshape(1, -1)
        elif X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)
        ev = _core.Evaluator(expr.graph, [expr.id])
        return ev.eval_batch(np.ascontiguousarray(X_mat, dtype=np.float64))[0]

    def sensitivity(
        self,
        params: Union[Expr, Sequence[Expr]],
        target: Optional[Union[str, Expr]] = None,
        active_tol: float = 1e-4,
    ) -> Any:
        """
        Compute exact analytical post-optimality sensitivities at the converged KKT solution `(x*, lambda*)`
        with respect to one or more frozen `Parameter` expressions `params` without re-solving:
          - Objective sensitivity `df_dp` via the Envelope Theorem:
                df*(p)/dp = dL(x*, p, lambda*)/dp
          - Primal decision sensitivity `dx_dp` (shape `(n_vars, n_p)`) via the active KKT Implicit Function Theorem:
                [ H_FF   J_{A,F}^T ] [ dx_F*/dp ]   [ -d^2L/(dx_F dp) ]
                [ J_{A,F}    0     ] [ dlam_A/dp] = [    -J_{A,p}     ]
        """
        single_param = isinstance(params, Expr)
        param_list: List[Expr] = [params] if single_param else list(params)
        if not param_list:
            raise ValueError("sensitivity requires at least one frozen Parameter expression.")
        for p in param_list:
            if p.op_type != _core.OpType.Parameter:
                raise TypeError(f"sensitivity requires frozen Parameter expressions, got {p.op_type.name}.")
            if p.graph is not self._opti.cpp_graph:
                raise ValueError("sensitivity parameter belongs to a different Graph.")

        param_ids = [p.id for p in param_list]
        mult_eq_phys = self._stats.get("mult_eq", np.zeros(0, dtype=np.float64))
        mult_ineq_phys = self._stats.get("mult_ineq", np.zeros(0, dtype=np.float64))
        mult_x_L = self._stats.get("mult_x_L", np.zeros(0, dtype=np.float64))
        mult_x_U = self._stats.get("mult_x_U", np.zeros(0, dtype=np.float64))

        mults_ev: List[np.ndarray] = []
        if self._has_eq:
            mults_ev.append(mult_eq_phys * self._eq_scale_vec)
        if self._has_ineq:
            mults_ev.append(mult_ineq_phys * self._ineq_scale_vec)

        dL_dp, d2L_dxp, dout_dp = self._evaluator.parameter_sensitivities(
            self._x, param_ids, float(self._obj_scale), mults_ev
        )
        n_vars = self._x.size
        n_p = dL_dp.size

        if isinstance(target, str) and target in ("f", "obj"):
            return float(dL_dp.item()) if (single_param and n_p == 1) else dL_dp.copy()

        # Solve active KKT linear system for primal solution sensitivity dx*/dp
        dx_dp = np.zeros((n_vars, n_p), dtype=np.float64)
        if n_vars > 0 and n_p > 0:
            H_L = self._evaluator.lagrangian_hessian(self._x, float(self._obj_scale), mults_ev)
            free_idx: List[int] = []
            for j in range(n_vars):
                clamped_lb = (
                    np.isfinite(self._lb[j])
                    and (self._x[j] - self._lb[j]) <= active_tol
                    and (mult_x_L.size <= j or mult_x_L[j] > 1e-8)
                )
                clamped_ub = (
                    np.isfinite(self._ub[j])
                    and (self._ub[j] - self._x[j]) <= active_tol
                    and (mult_x_U.size <= j or mult_x_U[j] > 1e-8)
                )
                if not clamped_lb and not clamped_ub:
                    free_idx.append(j)

            n_free = len(free_idx)
            if n_free > 0:
                H_FF = H_L[np.ix_(free_idx, free_idx)]
                rhs_top = -d2L_dxp[free_idx, :]
                J_ax_list: List[np.ndarray] = []
                J_ap_list: List[np.ndarray] = []

                if self._has_eq:
                    J_ax_list.append(self._evaluator.jacobian(self._x, 1))
                    J_ap_list.append(dout_dp[1])

                if self._has_ineq:
                    ineq_idx = 2 if self._has_eq else 1
                    g_val = self._evaluator.eval(self._x)[ineq_idx].ravel()
                    act_rows = [
                        i
                        for i in range(g_val.size)
                        if g_val[i] >= -active_tol
                        and (mult_ineq_phys.size <= i or mult_ineq_phys[i] > 1e-8 or g_val[i] >= -1e-6)
                    ]
                    if act_rows:
                        J_ax_list.append(self._evaluator.jacobian(self._x, ineq_idx)[act_rows, :])
                        J_ap_list.append(dout_dp[ineq_idx][act_rows, :])

                if J_ax_list:
                    J_AF = np.vstack(J_ax_list)[:, free_idx]
                    J_Ap = np.vstack(J_ap_list)
                    m_act = J_AF.shape[0]
                    KKT_mat = np.block(
                        [
                            [H_FF, J_AF.T],
                            [J_AF, np.zeros((m_act, m_act), dtype=np.float64)],
                        ]
                    )
                    KKT_rhs = np.vstack([rhs_top, -J_Ap])
                    sol_kkt, _, _, _ = np.linalg.lstsq(KKT_mat, KKT_rhs, rcond=1e-10)
                    dx_dp[free_idx, :] = sol_kkt[:n_free, :]
                else:
                    sol_free, _, _, _ = np.linalg.lstsq(H_FF, rhs_top, rcond=1e-10)
                    dx_dp[free_idx, :] = sol_free

        if isinstance(target, str) and target == "x":
            return dx_dp.ravel() if (single_param and n_p == 1) else dx_dp

        if isinstance(target, Expr):
            if self._opti._objective is not None and target.id == self._opti._objective.id:
                return float(dL_dp.item()) if (single_param and n_p == 1) else dL_dp.copy()
            if target.id in self._var_dual_map:
                start, stop, _ = self._var_dual_map[target.id]
                sub_dx = dx_dp[start:stop, :]
                if sub_dx.size == 1:
                    return float(sub_dx.item())
                return sub_dx.ravel() if (single_param and n_p == 1) else sub_dx
            raise ValueError(
                "target Expr in sensitivity() must be an active decision variable or the objective expression."
            )

        return {
            "df_dp": float(dL_dp.item()) if (single_param and n_p == 1) else dL_dp.copy(),
            "dx_dp": dx_dp.ravel() if (single_param and n_p == 1) else dx_dp,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the optimal solution state (decision vector, named variable values, objective,
        convergence status, and KKT dual multipliers) to a serializable Python dictionary.
        Can be passed directly to `opti.load_state(state_dict)` to warm-start future runs.
        """
        var_dict: Dict[str, np.ndarray] = {}
        for rec in self._opti._var_records:
            vid = rec["id"]
            vname = rec["name"]
            off = self._opti._graph.node_offset(vid)
            r, c = rec["shape"]
            seg = self._x[off : off + r * c]
            var_dict[vname] = seg.reshape((r, c), order="F").copy()

        return {
            "x": self._x.copy(),
            "f": self.f,
            "success": self.success,
            "message": self.message,
            "variables": var_dict,
            "multipliers": {
                "mult_eq": self._stats.get("mult_eq", np.zeros(0, dtype=np.float64)).copy(),
                "mult_ineq": self._stats.get("mult_ineq", np.zeros(0, dtype=np.float64)).copy(),
                "mult_x_L": self._stats.get("mult_x_L", np.zeros(0, dtype=np.float64)).copy(),
                "mult_x_U": self._stats.get("mult_x_U", np.zeros(0, dtype=np.float64)).copy(),
            },
        }


class Opti(GraphBuilder):
    """
    CasADi Opti-style declarative optimization problem builder.
    Inherits from GraphBuilder (retaining variable, parameter, constant, and build_evaluator)
    while adding minimize, subject_to, solver, callback, debug_infeasibility, to_pymoo_problem, and solve.
    """

    def __init__(self) -> None:
        """Initialize a new declarative `Opti` optimization environment with an empty C++ computation graph."""
        super().__init__()
        self._objective: Optional[Expr] = None
        self._obj_scale_manual: Optional[float] = None
        self._constraint_entries: List[Dict[str, Any]] = []
        self._var_records: List[Dict[str, Any]] = []

        self._solver_backend: str = "scipy"
        self._solver_method: Any = "SLSQP"
        self._solver_options: Dict[str, Any] = {}
        self._use_hessian: bool = True
        self._auto_scale: bool = False
        self._iter_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._compiled_cache: Dict[bool, Tuple[Any, ...]] = {}

    def variable(
        self,
        shape: ShapeLike = (1, 1),
        init: ArrayLike = 0.0,
        lb: Optional[ArrayLike] = None,
        ub: Optional[ArrayLike] = None,
        freeze: bool = False,
        scale: Optional[ArrayLike] = None,
        name: str = "",
    ) -> Expr:
        """
        Declare an optimization variable (freeze=False) or fixed parameter (freeze=True).
        When freeze=True, lb, ub, and scale are silently ignored.
        Optional `scale` specifies the characteristic physical magnitude of the variable.
        """
        s = _normalize_shape(shape)
        init_flat = _to_fortran_flat(init, s, 0.0)
        if freeze:
            nid = self._graph.add_parameter(s, init_flat, name)
            return Expr(self._graph, nid)

        lb_flat = _to_fortran_flat(lb, s, -np.inf)
        ub_flat = _to_fortran_flat(ub, s, np.inf)
        scale_flat = (
            np.abs(_to_fortran_flat(scale, s, 1.0)) if scale is not None else None
        )
        if scale_flat is not None and np.any(scale_flat <= 0.0):
            raise ValueError("Variable scale must be strictly positive.")

        self._compiled_cache.clear()
        nid = self._graph.add_variable(s, init_flat, lb_flat, ub_flat, name)
        expr = Expr(self._graph, nid)
        self._var_records.append(
            {
                "id": nid,
                "name": name or f"var_{len(self._var_records)}",
                "shape": (int(s.rows), int(s.cols)),
                "size": int(s.size()),
                "lb": lb_flat,
                "ub": ub_flat,
                "scale": scale_flat,
            }
        )
        return expr

    def set_initial(self, var: Expr, val: ArrayLike) -> None:
        """Set initial guess for an active Variable."""
        var.set_value(val)

    def set_value(self, param: Expr, val: ArrayLike) -> None:
        """Set numerical value for a frozen Parameter (or initial value of a Variable)."""
        param.set_value(val)

    def callback(self, fn: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """Register a per-iteration callback `fn(info_dict)`."""
        self._iter_callback = fn

    def minimize(
        self,
        obj: Union[Expr, float, int, Sequence[Union[Expr, float, int]]],
        scale: Optional[float] = None,
    ) -> None:
        """
        Set the objective function(s) to minimize.
        - Single-objective: pass a scalar Expr (with optional characteristic `scale`).
        - Multi-objective (for pymoo NSGA2/NSGA3/etc.): pass a list/tuple of Exprs or a vector Expr.
        """
        if scale is not None and float(scale) <= 0.0:
            raise ValueError("Objective scale must be positive.")
        self._compiled_cache.clear()
        self._obj_scale_manual = float(scale) if scale is not None else None

        if isinstance(obj, (list, tuple)):
            if not obj:
                raise ValueError("Objective list cannot be empty.")
            coerced = [_coerce_with_graph(self._graph, item) for item in obj]
            reshaped = [e if e.shape[1] == 1 else e.reshape(e.size, 1) for e in coerced]
            self._objective = reshaped[0] if len(reshaped) == 1 else concat(reshaped, axis=0)
        else:
            obj_expr = _coerce_with_graph(self._graph, obj)
            if obj_expr.size > 1 and obj_expr.shape[1] != 1:
                obj_expr = obj_expr.reshape(obj_expr.size, 1)
            self._objective = obj_expr

    def subject_to(
        self,
        constraint: Union[ComparisonExpr, Expr, Iterable[Union[ComparisonExpr, Expr]]],
        scale: Optional[ArrayLike] = None,
        name: str = "",
    ) -> Any:
        """
        Register one or more constraints (e.g., g <= ub, g >= lb, h == target).
        Returns the constraint handle(s) so you can query `sol.dual(con)` after `solve()`.
        """
        if isinstance(constraint, (list, tuple)):
            return [self.subject_to(c, scale=scale, name=name) for c in constraint]
        if not isinstance(constraint, Expr):
            raise TypeError("Constraint must be a ComparisonExpr (e.g. g <= 0) or Expr.")

        self._compiled_cache.clear()
        self._constraint_entries.append(
            {
                "expr": constraint,
                "scale": scale,
                "name": name or f"con_{len(self._constraint_entries)}",
            }
        )
        return constraint

    def solver(
        self,
        name: str = "scipy",
        method: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
        use_hessian: bool = True,
        auto_scale: bool = False,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Configure the optimization solver backend and options.

        Args:
            name: Solver backend or algorithm name:
                - `'ipopt'`: Native C++17 COIN-OR IPOPT interior-point solver (with MUMPS sparse linear solver).
                - `'scipy'` or `'SLSQP'` / `'trust-constr'` / `'L-BFGS-B'`: `scipy.optimize.minimize` adapter.
                - `'pymoo'` or `'DE'` / `'GA'` / `'PSO'` / `'CMAES'` / `'ISRES'` / `'NSGA2'` / `'NSGA3'`: Evolutionary solver.
            method: Optional sub-algorithm name when `name` is `'scipy'` or `'pymoo'`.
            options: Solver-specific configuration dictionary:
                - IPOPT options: `{'tol': 1e-8, 'max_iter': 500, 'print_level': 0, 'linear_solver': 'mumps', 'mu_strategy': 'adaptive'}`.
                - SciPy options: `{'ftol': 1e-9, 'gtol': 1e-8, 'maxiter': 500, 'disp': False}`.
                - Pymoo options: `{'pop_size': 64, 'n_gen': 100, 'seed': 42, 'num_threads': 0, 'verbose': False}`.
            use_hessian: Whether to supply exact 2nd-order AD Lagrangian Hessians (`True`) or use quasi-Newton L-BFGS (`False`).
            auto_scale: Automatically normalize variables, objective, and constraints to `O(1)` magnitudes before solving.
            callback: Optional per-iteration callback `fn(info_dict)`.
        """
        lname = name.lower().strip()
        if lname == "ipopt":
            self._solver_backend = "ipopt"
            self._solver_method = "IPOPT"
        elif lname == "pymoo":
            self._solver_backend = "pymoo"
            self._solver_method = method if method is not None else "AUTO"
        elif lname in _PYMOO_METHODS:
            self._solver_backend = "pymoo"
            self._solver_method = name
        elif lname == "scipy":
            self._solver_backend = "scipy"
            self._solver_method = method if method is not None else "SLSQP"
        elif lname in ("slsqp", "trust-constr", "l-bfgs-b", "bfgs", "newton-cg", "tnc", "nelder-mead", "powell", "cg"):
            self._solver_backend = "scipy"
            self._solver_method = name
        else:
            raise ValueError(
                f"Unsupported solver '{name}'. Supported: 'ipopt', 'scipy' ('SLSQP', 'trust-constr', 'L-BFGS-B'), "
                f"or 'pymoo' ('DE', 'GA', 'PSO', 'CMAES', 'ISRES', 'SRES', 'NSGA2', 'NSGA3', 'MOEAD', 'SMSEMOA')."
            )
        self._solver_options = dict(options) if options is not None else {}
        self._use_hessian = bool(use_hessian)
        self._auto_scale = bool(auto_scale)
        if callback is not None:
            self._iter_callback = callback

    @staticmethod
    def _is_constant_zero(val: Union[Expr, ArrayLike]) -> bool:
        if isinstance(val, (int, float, np.number)):
            return float(val) == 0.0
        if isinstance(val, np.ndarray):
            return bool(np.all(val == 0.0))
        return False

    def _try_extract_simple_bound(
        self,
        lhs: Expr,
        op_str: str,
        rhs_raw: Union[Expr, ArrayLike],
        var_lb_map: Dict[int, np.ndarray],
        var_ub_map: Dict[int, np.ndarray],
    ) -> Optional[str]:
        if isinstance(rhs_raw, Expr):
            return None
        if lhs.op_type == _core.OpType.Parameter:
            return "inactive"
        if lhs.op_type != _core.OpType.Variable or op_str == "==":
            return None

        bound_flat = _to_fortran_flat(rhs_raw, _core.Shape(*lhs.shape))
        if op_str in ("<=", "<"):
            var_ub_map[lhs.id] = np.minimum(var_ub_map[lhs.id], bound_flat)
            return "bound_ub"
        if op_str in (">=", ">"):
            var_lb_map[lhs.id] = np.maximum(var_lb_map[lhs.id], bound_flat)
            return "bound_lb"
        return None

    def _compute_variable_scales(
        self,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        enable_auto: bool,
    ) -> np.ndarray:
        n = x0.size
        if n == 0:
            return np.ones(0, dtype=np.float64)

        d_x = np.ones(n, dtype=np.float64)
        offset = 0
        for rec in self._var_records:
            sz = rec["size"]
            s_manual = rec["scale"]
            if s_manual is not None:
                d_x[offset : offset + sz] = s_manual
            elif enable_auto:
                for i in range(offset, offset + sz):
                    span = ub[i] - lb[i]
                    if np.isfinite(span) and 1e-9 < span < 1e8:
                        d_x[i] = float(np.clip(0.5 * span, 1e-6, 1e6))
                    elif abs(x0[i]) > 1e-6:
                        d_x[i] = float(np.clip(abs(x0[i]), 1e-6, 1e6))
                    elif np.isfinite(lb[i]) and abs(lb[i]) > 1e-6:
                        d_x[i] = float(np.clip(abs(lb[i]), 1e-6, 1e6))
                    elif np.isfinite(ub[i]) and abs(ub[i]) > 1e-6:
                        d_x[i] = float(np.clip(abs(ub[i]), 1e-6, 1e6))
            offset += sz
        return d_x

    def _compile_problem(
        self,
        enable_auto_scale: bool = False,
    ) -> Tuple[
        _core.Evaluator,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        bool,
        bool,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Dict[int, Dict[str, Any]],
        Dict[int, Tuple[int, int, Tuple[int, int]]],
    ]:
        if self._objective is None:
            raise ValueError("No objective specified. Call opti.minimize(obj) before solve().")

        if self._objective.size != 1 and self._solver_backend != "pymoo":
            raise ValueError(
                f"Single-objective gradient solver '{self._solver_method}' requires a scalar objective (size=1), "
                f"got shape {self._objective.shape}. Use a multi-objective pymoo solver (e.g. 'NSGA2') for vector objectives."
            )

        cache_key = bool(enable_auto_scale)
        if cache_key in self._compiled_cache:
            (
                evaluator,
                lb,
                ub,
                has_eq,
                has_ineq,
                obj_scale,
                var_scale,
                eq_scale_vec,
                ineq_scale_vec,
                dual_map,
                var_dual_map,
            ) = self._compiled_cache[cache_key]
            x0 = self._graph.initial_decision_vector()
            return (
                evaluator,
                x0,
                lb,
                ub,
                has_eq,
                has_ineq,
                obj_scale,
                var_scale,
                eq_scale_vec,
                ineq_scale_vec,
                dual_map,
                var_dual_map,
            )

        var_lb_map: Dict[int, np.ndarray] = {}
        var_ub_map: Dict[int, np.ndarray] = {}
        var_dual_map: Dict[int, Tuple[int, int, Tuple[int, int]]] = {}
        v_cursor = 0
        for rec in self._var_records:
            vid = rec["id"]
            sz = rec["size"]
            var_lb_map[vid] = rec["lb"].copy()
            var_ub_map[vid] = rec["ub"].copy()
            var_dual_map[vid] = (v_cursor, v_cursor + sz, rec["shape"])
            v_cursor += sz

        eq_exprs: List[Expr] = []
        ineq_exprs: List[Expr] = []
        eq_scales: List[np.ndarray] = []
        ineq_scales: List[np.ndarray] = []
        dual_map: Dict[int, Dict[str, Any]] = {}
        eq_cursor = 0
        ineq_cursor = 0

        for entry in self._constraint_entries:
            c = entry["expr"]
            c_scale_raw = entry["scale"]

            if isinstance(c, ComparisonExpr):
                lhs = c.lhs
                op_str = c.op_str
                rhs_raw = c.rhs_raw

                bound_kind = self._try_extract_simple_bound(lhs, op_str, rhs_raw, var_lb_map, var_ub_map)
                if bound_kind is not None:
                    if bound_kind in ("bound_ub", "bound_lb"):
                        start, stop, shp = var_dual_map[lhs.id]
                        dual_map[id(c)] = {
                            "kind": bound_kind,
                            "start": start,
                            "stop": stop,
                            "size": stop - start,
                            "shape": shp,
                        }
                    else:
                        dual_map[id(c)] = {
                            "kind": "inactive",
                            "start": 0,
                            "stop": 0,
                            "size": lhs.size,
                            "shape": lhs.shape,
                        }
                    continue

                if self._is_constant_zero(rhs_raw):
                    if op_str in ("<=", "<"):
                        res_expr = lhs if lhs.shape[1] == 1 else lhs.reshape(lhs.size, 1)
                        is_eq = False
                    elif op_str in (">=", ">"):
                        neg_lhs = -lhs
                        res_expr = neg_lhs if neg_lhs.shape[1] == 1 else neg_lhs.reshape(neg_lhs.size, 1)
                        is_eq = False
                    elif op_str == "==":
                        res_expr = lhs if lhs.shape[1] == 1 else lhs.reshape(lhs.size, 1)
                        is_eq = True
                    else:
                        raise ValueError(f"Unsupported comparison operator: {op_str}")
                    rhs_mag = np.ones(res_expr.size, dtype=np.float64)
                else:
                    rhs = _coerce_with_graph(self._graph, rhs_raw)
                    if op_str in ("<=", "<"):
                        diff = lhs - rhs
                        res_expr = diff if diff.shape[1] == 1 else diff.reshape(diff.size, 1)
                        is_eq = False
                    elif op_str in (">=", ">"):
                        diff = rhs - lhs
                        res_expr = diff if diff.shape[1] == 1 else diff.reshape(diff.size, 1)
                        is_eq = False
                    elif op_str == "==":
                        diff = lhs - rhs
                        res_expr = diff if diff.shape[1] == 1 else diff.reshape(diff.size, 1)
                        is_eq = True
                    else:
                        raise ValueError(f"Unsupported comparison operator: {op_str}")
                    if not isinstance(rhs_raw, Expr):
                        rhs_flat = np.abs(_to_fortran_flat(rhs_raw, _core.Shape(res_expr.size, 1), 1.0))
                        rhs_mag = np.where(rhs_flat > 1.0, np.clip(rhs_flat, 1.0, 1e6), 1.0)
                    else:
                        rhs_mag = np.ones(res_expr.size, dtype=np.float64)
                orig_shape = lhs.shape
            else:
                res_expr = c if c.shape[1] == 1 else c.reshape(c.size, 1)
                is_eq = False
                rhs_mag = np.ones(res_expr.size, dtype=np.float64)
                orig_shape = c.shape

            sz = res_expr.size
            if c_scale_raw is not None:
                c_sc = np.abs(_to_fortran_flat(c_scale_raw, _core.Shape(sz, 1), 1.0))
                if np.any(c_sc <= 0.0):
                    raise ValueError("Constraint scale must be strictly positive.")
            elif enable_auto_scale:
                c_sc = rhs_mag
            else:
                c_sc = np.ones(sz, dtype=np.float64)

            if np.any(c_sc != 1.0):
                sc_const = self.constant(c_sc, shape=(sz, 1))
                scaled_expr = res_expr / sc_const
            else:
                scaled_expr = res_expr

            if is_eq:
                eq_exprs.append(scaled_expr)
                eq_scales.append(c_sc)
                dual_map[id(c)] = {
                    "kind": "eq",
                    "start": eq_cursor,
                    "stop": eq_cursor + sz,
                    "size": sz,
                    "shape": orig_shape,
                }
                eq_cursor += sz
            else:
                ineq_exprs.append(scaled_expr)
                ineq_scales.append(c_sc)
                dual_map[id(c)] = {
                    "kind": "ineq",
                    "start": ineq_cursor,
                    "stop": ineq_cursor + sz,
                    "size": sz,
                    "shape": orig_shape,
                }
                ineq_cursor += sz

        x0 = self._graph.initial_decision_vector()
        if self._var_records:
            lb = np.concatenate([var_lb_map[rec["id"]] for rec in self._var_records])
            ub = np.concatenate([var_ub_map[rec["id"]] for rec in self._var_records])
        else:
            lb = np.empty(0, dtype=np.float64)
            ub = np.empty(0, dtype=np.float64)

        var_scale = self._compute_variable_scales(x0, lb, ub, enable_auto_scale)

        obj_scale = 1.0
        if self._obj_scale_manual is not None:
            obj_scale = self._obj_scale_manual
        elif enable_auto_scale and self._objective.size == 1 and self._solver_backend != "pymoo":
            probe_ev = _core.Evaluator(self._graph, [self._objective.id])
            x_probe = np.clip(x0, lb, ub)
            f0 = abs(float(probe_ev.eval(x_probe)[0].item()))
            g0 = probe_ev.gradient(x_probe, 0) * var_scale
            g_norm = float(np.max(np.abs(g0))) if g0.size > 0 else 1.0
            scale_cand = max(f0, g_norm)
            if scale_cand > 10.0 or (0.0 < scale_cand < 0.1):
                obj_scale = float(np.clip(scale_cand, 1e-6, 1e6))

        obj_expr_compiled = (
            (self._objective / obj_scale) if (obj_scale != 1.0 and self._objective.size == 1) else self._objective
        )

        outputs: List[Expr] = [obj_expr_compiled]
        has_eq = len(eq_exprs) > 0
        has_ineq = len(ineq_exprs) > 0

        if has_eq:
            eq_vec = eq_exprs[0] if len(eq_exprs) == 1 else concat(eq_exprs, axis=0)
            outputs.append(eq_vec)
        if has_ineq:
            ineq_vec = ineq_exprs[0] if len(ineq_exprs) == 1 else concat(ineq_exprs, axis=0)
            outputs.append(ineq_vec)

        evaluator = _core.Evaluator(self._graph, [o.id for o in outputs])
        eq_scale_vec = np.concatenate(eq_scales) if eq_scales else np.zeros(0, dtype=np.float64)
        ineq_scale_vec = np.concatenate(ineq_scales) if ineq_scales else np.zeros(0, dtype=np.float64)

        self._compiled_cache[cache_key] = (
            evaluator,
            lb,
            ub,
            has_eq,
            has_ineq,
            obj_scale,
            var_scale,
            eq_scale_vec,
            ineq_scale_vec,
            dual_map,
            var_dual_map,
        )

        return (
            evaluator,
            x0,
            lb,
            ub,
            has_eq,
            has_ineq,
            obj_scale,
            var_scale,
            eq_scale_vec,
            ineq_scale_vec,
            dual_map,
            var_dual_map,
        )

    def debug_infeasibility(
        self,
        x: Optional[ArrayLike] = None,
        tol: float = 1e-5,
        print_report: bool = False,
    ) -> Dict[str, Any]:
        """
        Inspect all variable bounds and nonlinear constraints at point `x`
        (defaults to current initial/solved decision vector) and report any violations > `tol`.
        """
        evaluator, x0, lb, ub, _, _, _, _, _, _, _, _ = self._compile_problem(enable_auto_scale=False)
        x_vec = np.asarray(x0 if x is None else x, dtype=np.float64).ravel()
        if x_vec.size != x0.size:
            raise ValueError(f"debug_infeasibility: x size ({x_vec.size}) does not match num_vars ({x0.size}).")

        violated_bounds: List[Dict[str, Any]] = []
        max_bound_viol = 0.0
        offset = 0
        for rec in self._var_records:
            sz = rec["size"]
            vname = rec["name"]
            v_lb = lb[offset : offset + sz]
            v_ub = ub[offset : offset + sz]
            v_val = x_vec[offset : offset + sz]
            for i in range(sz):
                low_v = max(0.0, float(v_lb[i] - v_val[i]))
                high_v = max(0.0, float(v_val[i] - v_ub[i]))
                viol = max(low_v, high_v)
                max_bound_viol = max(max_bound_viol, viol)
                if viol > tol:
                    violated_bounds.append(
                        {
                            "variable": vname,
                            "index": i,
                            "value": float(v_val[i]),
                            "lb": float(v_lb[i]),
                            "ub": float(v_ub[i]),
                            "violation": viol,
                            "type": "lb" if low_v >= high_v else "ub",
                        }
                    )
            offset += sz

        violated_bounds.sort(key=lambda d: d["violation"], reverse=True)

        violated_constraints: List[Dict[str, Any]] = []
        max_con_viol = 0.0
        for idx, entry in enumerate(self._constraint_entries):
            c = entry["expr"]
            cname = entry["name"]
            if isinstance(c, ComparisonExpr):
                op_str = c.op_str
                lhs_val = evaluator.eval_node(x_vec, c.lhs.id).ravel(order="F")
                if isinstance(c.rhs_raw, Expr):
                    rhs_val = evaluator.eval_node(x_vec, c.rhs_raw.id).ravel(order="F")
                else:
                    rhs_val = _to_fortran_flat(c.rhs_raw, _core.Shape(lhs_val.size, 1), 0.0)
                diff = lhs_val - rhs_val
                if op_str in ("<=", "<"):
                    viols = np.maximum(0.0, diff)
                elif op_str in (">=", ">"):
                    viols = np.maximum(0.0, -diff)
                else:
                    viols = np.abs(diff)
            else:
                op_str = "<="
                lhs_val = evaluator.eval_node(x_vec, c.id).ravel(order="F")
                rhs_val = np.zeros_like(lhs_val)
                viols = np.maximum(0.0, lhs_val)

            for elem_i in range(viols.size):
                v = float(viols[elem_i])
                max_con_viol = max(max_con_viol, v)
                if v > tol:
                    violated_constraints.append(
                        {
                            "constraint_index": idx,
                            "name": cname,
                            "element": elem_i,
                            "op": op_str,
                            "lhs": float(lhs_val[elem_i]),
                            "rhs": float(rhs_val[elem_i]),
                            "violation": v,
                        }
                    )

        violated_constraints.sort(key=lambda d: d["violation"], reverse=True)
        max_viol = max(max_bound_viol, max_con_viol)
        is_feasible = max_viol <= tol

        lines = [
            f"=== Opti Feasibility Diagnostic Report (tol={tol:.1e}) ===",
            f"Status: {'FEASIBLE' if is_feasible else 'INFEASIBLE'} | Max Violation: {max_viol:.4e} "
            f"(Bounds: {max_bound_viol:.4e}, Constraints: {max_con_viol:.4e})",
        ]
        if violated_bounds:
            lines.append(f"Violated Variable Bounds ({len(violated_bounds)}):")
            for b in violated_bounds[:10]:
                lines.append(
                    f"  - {b['variable']}[{b['index']}]: val={b['value']:.6g} not in [{b['lb']:.6g}, {b['ub']:.6g}] "
                    f"(viol={b['violation']:.4e})"
                )
        if violated_constraints:
            lines.append(f"Violated Constraints ({len(violated_constraints)}):")
            for vc in violated_constraints[:10]:
                lines.append(
                    f"  - {vc['name']}[{vc['element']}]: {vc['lhs']:.6g} {vc['op']} {vc['rhs']:.6g} "
                    f"(viol={vc['violation']:.4e})"
                )
        report_str = "\n".join(lines)
        if print_report:
            print(report_str)

        return {
            "is_feasible": is_feasible,
            "max_violation": max_viol,
            "max_bound_violation": max_bound_viol,
            "max_constraint_violation": max_con_viol,
            "violated_bounds": violated_bounds,
            "violated_constraints": violated_constraints,
            "report": report_str,
        }

    def to_pymoo_problem(
        self,
        num_threads: int = 0,
        eq_tol: float = 1e-4,
        use_native_eq: bool = False,
    ) -> Any:
        """
        Export the current `Opti` model as a vectorized `pymoo.core.problem.Problem` instance
        backed by `Evaluator.eval_batch`.

        Args:
            num_threads: Number of OpenMP threads for batch evaluation (`0` = auto-detect all cores).
            eq_tol: Tolerance band `|h(x)| - eq_tol <= 0` when converting equality constraints to inequalities.
            use_native_eq: Pass equality constraints natively in `out["H"]` (for `ISRES`/`SRES`).

        Returns:
            A `FastOptPymooProblem` instance compatible with any custom `pymoo.optimize.minimize` workflow.
        """
        old_backend = self._solver_backend
        try:
            self._solver_backend = "pymoo"
            evaluator, _, lb, ub, has_eq, has_ineq, *_ = self._compile_problem(enable_auto_scale=False)
        finally:
            self._solver_backend = old_backend

        return build_pymoo_problem(
            evaluator=evaluator,
            lb=lb,
            ub=ub,
            n_obj=int(evaluator.output_size(0)),
            has_eq=has_eq,
            has_ineq=has_ineq,
            num_threads=num_threads,
            eq_tol=eq_tol,
            use_native_eq=use_native_eq,
        )

    def solve(
        self,
        auto_scale: Optional[bool] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> OptiSol:
        """Compile the problem, invoke the configured solver, and return an OptiSol."""
        use_auto_scale = self._auto_scale if auto_scale is None else bool(auto_scale)
        cb = self._iter_callback if callback is None else callback

        (
            evaluator,
            x0,
            lb,
            ub,
            has_eq,
            has_ineq,
            obj_scale,
            var_scale,
            eq_scale_vec,
            ineq_scale_vec,
            dual_map,
            var_dual_map,
        ) = self._compile_problem(enable_auto_scale=use_auto_scale)

        if self._solver_backend == "ipopt":
            stats = solve_ipopt(
                evaluator=evaluator,
                x0=x0,
                lb=lb,
                ub=ub,
                has_eq=has_eq,
                has_ineq=has_ineq,
                options=self._solver_options,
                use_hessian=self._use_hessian,
                callback=cb,
            )
        elif self._solver_backend == "pymoo":
            stats = solve_pymoo(
                evaluator=evaluator,
                x0=x0,
                lb=lb,
                ub=ub,
                has_eq=has_eq,
                has_ineq=has_ineq,
                method=self._solver_method,
                options=self._solver_options,
            )
        elif self._solver_backend == "scipy":
            stats = solve_scipy(
                evaluator=evaluator,
                x0=x0,
                lb=lb,
                ub=ub,
                has_eq=has_eq,
                has_ineq=has_ineq,
                method=self._solver_method,
                options=self._solver_options,
                use_hessian=self._use_hessian,
                var_scale=var_scale,
                callback=cb,
            )
        else:
            raise RuntimeError(f"Unknown solver backend: {self._solver_backend}")

        # Unscale objective value and KKT dual multipliers back into physical units:
        # lambda_phys = lambda_scaled * (obj_scale / constraint_scale)
        if obj_scale != 1.0 and isinstance(stats.get("f"), (int, float, np.floating)):
            stats["f"] = float(stats["f"]) * obj_scale
            for h_entry in stats.get("history", []):
                if "obj" in h_entry:
                    h_entry["obj"] = float(h_entry["obj"]) * obj_scale

        if "mult_eq" in stats and stats["mult_eq"].size > 0:
            stats["mult_eq"] = stats["mult_eq"] * (obj_scale / eq_scale_vec)
        if "mult_ineq" in stats and stats["mult_ineq"].size > 0:
            stats["mult_ineq"] = stats["mult_ineq"] * (obj_scale / ineq_scale_vec)
        if "mult_x_L" in stats and stats["mult_x_L"].size > 0:
            stats["mult_x_L"] = stats["mult_x_L"] * obj_scale
        if "mult_x_U" in stats and stats["mult_x_U"].size > 0:
            stats["mult_x_U"] = stats["mult_x_U"] * obj_scale

        # Write optimal solution back as new initial guess for warm-starting subsequent stages
        x_opt = stats["x"]
        for rec in self._var_records:
            vid = rec["id"]
            off = self._graph.node_offset(vid)
            sz = rec["size"]
            self._graph.set_variable_init(vid, x_opt[off : off + sz])

        return OptiSol(
            self,
            evaluator,
            stats,
            dual_map,
            var_dual_map,
            obj_scale=obj_scale,
            eq_scale_vec=eq_scale_vec,
            ineq_scale_vec=ineq_scale_vec,
            lb=lb,
            ub=ub,
            has_eq=has_eq,
            has_ineq=has_ineq,
        )

    def solve_sweep(
        self,
        params: Union[Expr, Sequence[Expr]],
        values: Sequence[Any],
        warm_start: bool = True,
        auto_scale: Optional[bool] = None,
        step_callback: Optional[Callable[[int, Any, OptiSol], None]] = None,
    ) -> List[OptiSol]:
        """
        Execute a parametric continuation / sweep across `values` for one or more frozen `Parameter`s
        without recompiling the computation graph or recomputing Graph-Coloring sparsity plans.
        When `warm_start=True` (default), each step automatically warm-starts from the previous step's
        optimal decision vector.
        """
        single_param = isinstance(params, Expr)
        param_list: List[Expr] = [params] if single_param else list(params)
        for p in param_list:
            if p.op_type != _core.OpType.Parameter:
                raise TypeError(f"solve_sweep requires frozen Parameter expressions, got {p.op_type.name}.")

        orig_inits = [
            self._graph.get_variable_init(rec["id"]).copy() for rec in self._var_records
        ]

        solutions: List[OptiSol] = []
        for idx, val_entry in enumerate(values):
            if not warm_start and idx > 0:
                for rec, init_v in zip(self._var_records, orig_inits):
                    self._graph.set_variable_init(rec["id"], init_v)

            if single_param:
                param_list[0].set_value(val_entry)
            else:
                if len(val_entry) != len(param_list):
                    raise ValueError(
                        f"solve_sweep step {idx}: expected {len(param_list)} parameter values, got {len(val_entry)}."
                    )
                for p, v in zip(param_list, val_entry):
                    p.set_value(v)

            sol = self.solve(auto_scale=auto_scale)
            solutions.append(sol)
            if step_callback is not None:
                step_callback(idx, val_entry, sol)

        return solutions

    def load_state(self, state: Union[OptiSol, Dict[str, Any]]) -> None:
        """
        Restore decision variable initial guesses from an `OptiSol` instance or a checkpoint dictionary
        returned by `sol.to_dict()`. Matches variables by name first, or falls back to flat vector `x`.

        Args:
            state: An `OptiSol` object or a dictionary created via `sol.to_dict()`.
        """
        if isinstance(state, OptiSol):
            state_dict = state.to_dict()
        elif isinstance(state, dict):
            state_dict = state
        else:
            raise TypeError("load_state expects an OptiSol or a dict returned by sol.to_dict().")

        var_map = state_dict.get("variables", {})
        matched_any = False
        if isinstance(var_map, dict) and var_map:
            for rec in self._var_records:
                vname = rec["name"]
                if vname in var_map:
                    arr = np.asarray(var_map[vname], dtype=np.float64)
                    r, c = rec["shape"]
                    if arr.size == r * c:
                        flat = arr.reshape((r, c)).ravel(order="F") if arr.ndim == 2 else arr.ravel()
                        self._graph.set_variable_init(rec["id"], flat)
                        matched_any = True

        if not matched_any and "x" in state_dict:
            x_vec = np.asarray(state_dict["x"], dtype=np.float64).ravel()
            if x_vec.size != self._graph.total_var_size():
                raise ValueError(
                    f"load_state: state['x'] size ({x_vec.size}) does not match total_var_size ({self._graph.total_var_size()})."
                )
            for rec in self._var_records:
                vid = rec["id"]
                off = self._graph.node_offset(vid)
                sz = rec["size"]
                self._graph.set_variable_init(vid, x_vec[off : off + sz])

    def solve_multistart(
        self,
        n_starts: int = 8,
        sampler: str = "lhs",
        seed: int = 42,
        include_current_init: bool = True,
        batch_screen_factor: int = 4,
        penalty_weight: float = 1e4,
        default_bound_span: float = 10.0,
        auto_scale: Optional[bool] = None,
        return_all: bool = False,
    ) -> Union[OptiSol, Tuple[OptiSol, List[OptiSol]]]:
        """
        Global-local hybrid multi-start optimization:
          1. Generates `n_starts * batch_screen_factor` candidate points inside variable bounds
             using stratified Latin Hypercube Sampling (`sampler='lhs'`) or uniform random (`sampler='random'`).
          2. Screens all candidates in parallel using C++ OpenMP `Evaluator.eval_batch`, ranking them by
             penalized merit `f(x) + penalty_weight * max_constraint_violation(x)`.
          3. Launches local NLP solves from the top `n_starts` candidates (including the user's initial guess
             when `include_current_init=True`), writes the global best solution back to `Opti`, and returns it.

        Args:
            n_starts: Number of local NLP solves to execute (default: `8`).
            sampler: Initial candidate sampling strategy: `'lhs'` (Latin Hypercube) or `'random'`.
            seed: Random number generator seed for reproducible sampling.
            include_current_init: Always include the current variable initial guess among the starts.
            batch_screen_factor: Ratio of batch-screened candidates to `n_starts` (default: `4`).
            penalty_weight: L-infinity constraint violation penalty weight used during batch screening.
            default_bound_span: Sampling half-width around `x0` for variables with infinite bounds.
            auto_scale: Override `auto_scale` setting for local solves.
            return_all: If `True`, return `(best_sol, all_sols)`; otherwise return `best_sol`.

        Returns:
            Global best `OptiSol` (or `(best_sol, all_sols)` when `return_all=True`).
        """
        if int(n_starts) < 1:
            raise ValueError("solve_multistart: n_starts must be >= 1.")
        use_auto_scale = self._auto_scale if auto_scale is None else bool(auto_scale)
        (
            evaluator,
            x0,
            lb,
            ub,
            has_eq,
            has_ineq,
            *_,
        ) = self._compile_problem(enable_auto_scale=use_auto_scale)

        n_vars = x0.size
        if n_vars == 0 or int(n_starts) == 1:
            sol = self.solve(auto_scale=auto_scale)
            return (sol, [sol]) if return_all else sol

        lb_box = np.where(np.isfinite(lb), lb, x0 - float(default_bound_span))
        ub_box = np.where(np.isfinite(ub), ub, x0 + float(default_bound_span))
        degenerate = ub_box <= lb_box
        if np.any(degenerate):
            ub_box = np.where(degenerate, lb_box + 1.0, ub_box)

        n_cand = max(int(n_starts), int(n_starts) * max(1, int(batch_screen_factor)))
        rng = np.random.default_rng(int(seed))
        smode = str(sampler).lower().strip()
        if smode == "lhs":
            U = np.empty((n_cand, n_vars), dtype=np.float64)
            cut = np.linspace(0.0, 1.0, n_cand + 1)
            for j in range(n_vars):
                u_col = rng.uniform(cut[:-1], cut[1:])
                rng.shuffle(u_col)
                U[:, j] = u_col
        elif smode in ("random", "uniform"):
            U = rng.uniform(0.0, 1.0, size=(n_cand, n_vars))
        else:
            raise ValueError(f"Unsupported sampler '{sampler}'. Use 'lhs' or 'random'.")

        X_cand = lb_box[None, :] + U * (ub_box - lb_box)[None, :]
        if include_current_init:
            X_cand[0, :] = np.clip(x0, lb, ub)

        batch_outs = evaluator.eval_batch(np.ascontiguousarray(X_cand, dtype=np.float64), 0)
        f_vals = batch_outs[0].ravel()
        viol_vals = np.zeros(n_cand, dtype=np.float64)
        out_cursor = 1
        if has_eq:
            viol_vals = np.maximum(viol_vals, np.max(np.abs(batch_outs[out_cursor]), axis=1))
            out_cursor += 1
        if has_ineq:
            viol_vals = np.maximum(viol_vals, np.max(np.maximum(0.0, batch_outs[out_cursor]), axis=1))

        merit = np.where(
            np.isfinite(f_vals) & np.isfinite(viol_vals),
            f_vals + float(penalty_weight) * viol_vals,
            np.inf,
        )
        ranked_indices = list(np.argsort(merit, kind="stable"))
        if include_current_init and 0 not in ranked_indices[: int(n_starts)]:
            selected_indices = [0] + [i for i in ranked_indices if i != 0][: int(n_starts) - 1]
        else:
            selected_indices = ranked_indices[: int(n_starts)]

        all_sols: List[OptiSol] = []
        for idx_c in selected_indices:
            x_start = X_cand[idx_c]
            for rec in self._var_records:
                vid = rec["id"]
                off = self._graph.node_offset(vid)
                sz = rec["size"]
                self._graph.set_variable_init(vid, x_start[off : off + sz])
            sol_i = self.solve(auto_scale=auto_scale)
            all_sols.append(sol_i)

        def _sol_sort_key(s: OptiSol) -> Tuple[int, float, float]:
            diag = self.debug_infeasibility(s.x, tol=1e-4)
            is_feas_and_ok = 0 if (s.success or diag["is_feasible"]) else 1
            f_scalar = float(np.sum(s.f)) if np.isfinite(np.sum(s.f)) else 1e30
            return (is_feas_and_ok, f_scalar if is_feas_and_ok == 0 else diag["max_violation"], f_scalar)

        all_sols_sorted = sorted(all_sols, key=_sol_sort_key)
        best_sol = all_sols_sorted[0]

        x_best = best_sol.x
        for rec in self._var_records:
            vid = rec["id"]
            off = self._graph.node_offset(vid)
            sz = rec["size"]
            self._graph.set_variable_init(vid, x_best[off : off + sz])

        return (best_sol, all_sols) if return_all else best_sol
