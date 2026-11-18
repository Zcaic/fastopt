"""
pymoo evolutionary optimization adapter for fastopt.
Vectorizes population evaluation directly through C++17 OpenMP `Evaluator.eval_batch(X_pop)`,
supporting both single-objective (GA, DE, PSO, CMAES, ISRES, SRES, PatternSearch, NelderMead)
and multi-objective (NSGA2, NSGA3, MOEAD, SMSEMOA, AGEMOEA) constrained problems.
All comments are written in English.
"""

from __future__ import annotations
import time
from typing import Any, Dict, Optional
import numpy as np

from .. import _fastopt_impl as _core


def build_pymoo_problem(
    evaluator: _core.Evaluator,
    lb: np.ndarray,
    ub: np.ndarray,
    n_obj: int = 1,
    has_eq: bool = False,
    has_ineq: bool = False,
    num_threads: int = 0,
    eq_tol: float = 1e-4,
    use_native_eq: bool = False,
) -> Any:
    """
    Construct a vectorized `pymoo.core.problem.Problem` instance backed by C++ `Evaluator.eval_batch`.
    By default (`use_native_eq=False`), equality constraints `h(x) == 0` are converted into
    `|h(x)| - eq_tol <= 0` in `out["G"]` so that all evolutionary algorithms in `pymoo`
    (including GA, DE, PSO, and NSGA2) can handle equality constraints out of the box.
    """
    try:
        from pymoo.core.problem import Problem
    except ImportError as exc:
        raise ImportError(
            "pymoo is required to use the 'pymoo' evolutionary solver backend. "
            "Please install it via 'pip install pymoo'."
        ) from exc

    n_vars = int(evaluator.num_vars())
    m_eq = int(evaluator.output_size(1)) if has_eq else 0
    m_ineq = int(evaluator.output_size(2 if has_eq else 1)) if has_ineq else 0

    if use_native_eq:
        n_ieq_constr = m_ineq
        n_eq_constr = m_eq
    else:
        n_ieq_constr = m_ineq + m_eq
        n_eq_constr = 0

    # Replace +/- inf bounds with finite bounds (+/- 1e6) if user left some variables unbounded,
    # since evolutionary algorithms require finite box bounds for population sampling.
    xl = np.where(np.isfinite(lb), lb, -1e6).astype(np.float64)
    xu = np.where(np.isfinite(ub), ub, 1e6).astype(np.float64)

    class FastOptPymooProblem(Problem):
        def __init__(self) -> None:
            super().__init__(
                n_var=n_vars,
                n_obj=int(n_obj),
                n_ieq_constr=n_ieq_constr,
                n_eq_constr=n_eq_constr,
                xl=xl,
                xu=xu,
            )

        def _evaluate(self, X: np.ndarray, out: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
            X_contig = np.ascontiguousarray(X, dtype=np.float64)
            if X_contig.ndim == 1:
                X_contig = X_contig.reshape(1, -1)

            batch_outs = evaluator.eval_batch(X_contig, int(num_threads))
            # batch_outs[0] has shape (n_pop, n_obj)
            out["F"] = batch_outs[0]

            if use_native_eq:
                if has_eq:
                    out["H"] = batch_outs[1]
                if has_ineq:
                    ineq_idx = 2 if has_eq else 1
                    out["G"] = batch_outs[ineq_idx]
            else:
                g_list = []
                if has_ineq:
                    ineq_idx = 2 if has_eq else 1
                    g_list.append(batch_outs[ineq_idx])
                if has_eq:
                    h_mat = batch_outs[1]
                    g_list.append(np.abs(h_mat) - float(eq_tol))
                if g_list:
                    out["G"] = g_list[0] if len(g_list) == 1 else np.hstack(g_list)

    return FastOptPymooProblem()


def _create_pymoo_algorithm(method: Any, n_obj: int, pop_size: int, alg_kwargs: Dict[str, Any]) -> Any:
    """Instantiate a pymoo Algorithm from a string name or return a user-provided Algorithm instance."""
    if not isinstance(method, str):
        return method

    m_upper = method.strip().upper().replace("-", "").replace("_", "")
    if m_upper in ("AUTO", "PYMOO", ""):
        m_upper = "DE" if n_obj == 1 else "NSGA2"

    if m_upper == "DE":
        from pymoo.algorithms.soo.nonconvex.de import DE
        return DE(pop_size=pop_size, **alg_kwargs)
    if m_upper == "GA":
        from pymoo.algorithms.soo.nonconvex.ga import GA
        return GA(pop_size=pop_size, **alg_kwargs)
    if m_upper == "PSO":
        from pymoo.algorithms.soo.nonconvex.pso import PSO
        return PSO(pop_size=pop_size, **alg_kwargs)
    if m_upper == "CMAES":
        from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
        return CMAES(**alg_kwargs)
    if m_upper == "ISRES":
        from pymoo.algorithms.soo.nonconvex.isres import ISRES
        return ISRES(**alg_kwargs)
    if m_upper == "SRES":
        from pymoo.algorithms.soo.nonconvex.sres import SRES
        return SRES(**alg_kwargs)
    if m_upper in ("PATTERNSEARCH", "PS"):
        from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
        return PatternSearch(**alg_kwargs)
    if m_upper in ("NELDERMEAD", "NM"):
        from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
        return NelderMead(**alg_kwargs)
    if m_upper == "NSGA2":
        from pymoo.algorithms.moo.nsga2 import NSGA2
        return NSGA2(pop_size=pop_size, **alg_kwargs)
    if m_upper == "NSGA3":
        from pymoo.algorithms.moo.nsga3 import NSGA3
        return NSGA3(pop_size=pop_size, **alg_kwargs)
    if m_upper == "MOEAD":
        from pymoo.algorithms.moo.moead import MOEAD
        return MOEAD(**alg_kwargs)
    if m_upper == "SMSEMOA":
        from pymoo.algorithms.moo.sms import SMSEMOA
        return SMSEMOA(pop_size=pop_size, **alg_kwargs)
    if m_upper == "AGEMOEA":
        from pymoo.algorithms.moo.age import AGEMOEA
        return AGEMOEA(pop_size=pop_size, **alg_kwargs)
    if m_upper == "AGEMOEA2":
        from pymoo.algorithms.moo.age2 import AGEMOEA2
        return AGEMOEA2(pop_size=pop_size, **alg_kwargs)

    raise ValueError(
        f"Unsupported pymoo algorithm '{method}'. Supported: "
        "'DE', 'GA', 'PSO', 'CMAES', 'ISRES', 'SRES', 'PatternSearch', 'NelderMead', "
        "'NSGA2', 'NSGA3', 'MOEAD', 'SMSEMOA', 'AGEMOEA', 'AGEMOEA2', or a pymoo Algorithm instance."
    )


def solve_pymoo(
    evaluator: _core.Evaluator,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    has_eq: bool,
    has_ineq: bool,
    method: Any = "DE",
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Solve a single-objective or multi-objective optimization problem using `pymoo`
    with C++17 OpenMP batch population evaluation (`Evaluator.eval_batch`).
    """
    try:
        from pymoo.optimize import minimize as pymoo_minimize
    except ImportError as exc:
        raise ImportError(
            "pymoo is required to use the 'pymoo' evolutionary solver backend. "
            "Please install it via 'pip install pymoo'."
        ) from exc

    t0 = time.perf_counter()
    opts = dict(options) if options is not None else {}

    pop_size = int(opts.pop("pop_size", 100))
    n_gen = int(opts.pop("n_gen", opts.pop("max_gen", opts.pop("maxiter", 100))))
    seed = opts.pop("seed", 42)
    verbose = bool(opts.pop("verbose", False))
    num_threads = int(opts.pop("num_threads", 0))
    eq_tol = float(opts.pop("eq_tol", 1e-4))
    use_native_eq = bool(opts.pop("use_native_eq", False))
    termination = opts.pop("termination", ("n_gen", n_gen))
    alg_override = opts.pop("algorithm", None)

    n_obj = int(evaluator.output_size(0))
    problem = build_pymoo_problem(
        evaluator=evaluator,
        lb=lb,
        ub=ub,
        n_obj=n_obj,
        has_eq=has_eq,
        has_ineq=has_ineq,
        num_threads=num_threads,
        eq_tol=eq_tol,
        use_native_eq=use_native_eq,
    )

    algorithm = _create_pymoo_algorithm(
        alg_override if alg_override is not None else method,
        n_obj=n_obj,
        pop_size=pop_size,
        alg_kwargs=opts,
    )

    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0

    # Extract solution(s): if res.X is None (no strictly feasible individual found),
    # fall back to least-infeasible individual in res.pop or x0
    X_res = getattr(res, "X", None)
    F_res = getattr(res, "F", None)
    success = X_res is not None

    if X_res is None:
        pop = getattr(res, "pop", None)
        if pop is not None and len(pop) > 0:
            X_pop = pop.get("X")
            F_pop = pop.get("F")
            CV_pop = pop.get("CV")
            best_idx = int(np.argmin(CV_pop.ravel())) if CV_pop is not None else 0
            x_best = np.asarray(X_pop[best_idx], dtype=np.float64).ravel()
            f_best_vec = np.asarray(F_pop[best_idx], dtype=np.float64).ravel()
            pareto_x = np.asarray(X_pop, dtype=np.float64)
            pareto_f = np.asarray(F_pop, dtype=np.float64)
            msg = "pymoo terminated without a strictly feasible solution; returning least-infeasible individual."
        else:
            x_best = np.asarray(x0, dtype=np.float64).ravel()
            f_eval = evaluator.eval(x_best)[0]
            f_best_vec = np.asarray(f_eval, dtype=np.float64).ravel()
            pareto_x = x_best.reshape(1, -1)
            pareto_f = f_best_vec.reshape(1, -1)
            msg = "pymoo terminated without solution; returning initial guess."
    else:
        X_arr = np.asarray(X_res, dtype=np.float64)
        F_arr = np.asarray(F_res, dtype=np.float64)
        if X_arr.ndim == 1:
            x_best = X_arr.ravel()
            f_best_vec = F_arr.ravel()
            pareto_x = x_best.reshape(1, -1)
            pareto_f = f_best_vec.reshape(1, -1)
        else:
            pareto_x = X_arr
            pareto_f = F_arr if F_arr.ndim == 2 else F_arr.reshape(-1, n_obj)
            # Pick compromise solution on the Pareto front (closest to normalized ideal point)
            f_min = np.min(pareto_f, axis=0)
            f_ptp = np.ptp(pareto_f, axis=0)
            f_ptp = np.where(f_ptp > 1e-14, f_ptp, 1.0)
            norm_f = (pareto_f - f_min) / f_ptp
            best_idx = int(np.argmin(np.sum(norm_f ** 2, axis=1)))
            x_best = pareto_x[best_idx].ravel()
            f_best_vec = pareto_f[best_idx].ravel()
        msg = "pymoo optimization terminated successfully."

    n_gen_done = int(getattr(getattr(res, "algorithm", None), "n_gen", n_gen) or n_gen)

    return {
        "x": x_best,
        "f": float(f_best_vec[0]) if n_obj == 1 else f_best_vec,
        "pareto_x": pareto_x,
        "pareto_f": pareto_f,
        "success": bool(success),
        "status": 0 if success else 1,
        "message": msg,
        "nit": n_gen_done,
        "elapsed_time": float(elapsed),
        "raw": res,
    }
