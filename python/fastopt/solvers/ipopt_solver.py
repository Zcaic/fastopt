"""
C++ native IPOPT backend adapter for fastopt (Path A: CasADi-backed IPOPT & MUMPS runtime).
Automatically locates the installed `casadi` Python package directory (`site-packages/casadi`)
on Windows, Linux, and macOS so that C++ `fastopt::solve_ipopt_cpp` can load `libipopt`
and `libcoinmumps` directly with zero Python callback overhead during optimization iterations.
All comments are written in English.
"""

from __future__ import annotations
import importlib.util
import os
from typing import Any, Dict, Optional
import numpy as np

from .. import _fastopt_impl as _core

# Known integer options in IPOPT to ensure int vs float option routing is exact
_IPOPT_INT_OPTIONS = {
    "max_iter",
    "print_level",
    "file_print_level",
    "watchdog_shortened_iter_trigger",
    "watchdog_trial_iter_max",
    "accept_every_trial_step",
    "max_soc",
    "max_hessian_perturbation",
    "limited_memory_max_history",
    "limited_memory_max_skipping",
    "mumps_mem_percent",
    "mumps_permuting_scaling",
    "mumps_pivot_order",
    "mumps_scaling",
}


def find_casadi_ipopt_dir(custom_dir: Optional[str] = None) -> str:
    """
    Locate the directory or shared library path providing COIN-OR IPOPT & MUMPS.
    Priority:
      1. Explicit `custom_dir` passed via solver options (`options={'lib_search_dir': ...}`).
      2. Environment variable `CASADIPATH` or `FASTOPT_IPOPT_PATH` if set.
      3. Installed Python `casadi` package directory (`site-packages/casadi`) via `importlib.util.find_spec('casadi')`.
      4. Fallback to the `fastopt` package directory.
    """
    if custom_dir and os.path.exists(custom_dir):
        return os.path.abspath(custom_dir)

    for env_var in ("FASTOPT_IPOPT_PATH", "CASADIPATH"):
        env_val = os.environ.get(env_var, "").strip()
        if env_val and os.path.exists(env_val):
            return os.path.abspath(env_val)

    spec = importlib.util.find_spec("casadi")
    if spec is not None:
        casadi_dir: Optional[str] = None
        if spec.submodule_search_locations:
            casadi_dir = list(spec.submodule_search_locations)[0]
        elif spec.origin:
            casadi_dir = os.path.dirname(spec.origin)
        if casadi_dir and os.path.isdir(casadi_dir):
            return os.path.abspath(casadi_dir)

    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pkg_dir


def solve_ipopt(
    evaluator: _core.Evaluator,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    has_eq: bool,
    has_ineq: bool,
    options: Optional[Dict[str, Any]] = None,
    use_hessian: bool = True,
    callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Solve the compiled NLP problem using the C++17 native IPOPT bridge (IpStdCInterface)
    backed by the IPOPT and MUMPS shared libraries bundled with `casadi`.
    Output convention in evaluator:
      - out_idx = 0: scalar objective f(x)
      - if has_eq: out_idx = 1 (equality constraints h(x) == 0)
      - if has_ineq: out_idx = (2 if has_eq else 1) (inequality constraints g(x) <= 0)
    """
    m_eq = evaluator.output_size(1) if has_eq else 0
    m_ineq = evaluator.output_size(2 if has_eq else 1) if has_ineq else 0
    m_total = m_eq + m_ineq

    g_L = np.empty(m_total, dtype=np.float64)
    g_U = np.empty(m_total, dtype=np.float64)
    if m_eq > 0:
        g_L[:m_eq] = 0.0
        g_U[:m_eq] = 0.0
    if m_ineq > 0:
        g_L[m_eq:] = -1e20
        g_U[m_eq:] = 0.0

    str_opts: Dict[str, str] = {}
    num_opts: Dict[str, float] = {}
    int_opts: Dict[str, int] = {}

    custom_lib_dir: Optional[str] = None
    if options is not None:
        for k, v in options.items():
            key = str(k)
            if key == "lib_search_dir":
                custom_lib_dir = str(v)
                continue
            if isinstance(v, str):
                str_opts[key] = v
            elif isinstance(v, (bool, np.bool_)):
                str_opts[key] = "yes" if bool(v) else "no"
            elif isinstance(v, (int, np.integer)) and (key in _IPOPT_INT_OPTIONS or not isinstance(v, float)):
                if key in _IPOPT_INT_OPTIONS:
                    int_opts[key] = int(v)
                else:
                    if key in (
                        "tol",
                        "acceptable_tol",
                        "constr_viol_tol",
                        "compl_inf_tol",
                        "dual_inf_tol",
                        "max_cpu_time",
                        "mu_init",
                    ):
                        num_opts[key] = float(v)
                    else:
                        int_opts[key] = int(v)
            elif isinstance(v, (float, np.floating)):
                if key in _IPOPT_INT_OPTIONS:
                    int_opts[key] = int(v)
                else:
                    num_opts[key] = float(v)
            else:
                raise TypeError(f"Unsupported IPOPT option type for '{key}': {type(v)}")

    lib_dir = find_casadi_ipopt_dir(custom_lib_dir)

    iter_cb_wrapper = None
    if callback is not None:
        def _ipopt_cb(it: int, obj: float, inf_pr: float, inf_du: float, mu: float, alpha_pr: float) -> bool:
            info = {
                "iter": int(it),
                "obj": float(obj),
                "constr_viol": float(inf_pr),
                "inf_du": float(inf_du),
                "mu": float(mu),
                "alpha_pr": float(alpha_pr),
            }
            res_cb = callback(info)
            return False if res_cb is False else True

        iter_cb_wrapper = _ipopt_cb

    cpp_res = _core.solve_ipopt_cpp(
        evaluator=evaluator,
        x0=np.asarray(x0, dtype=np.float64).ravel(),
        x_L=np.asarray(lb, dtype=np.float64).ravel(),
        x_U=np.asarray(ub, dtype=np.float64).ravel(),
        g_L=g_L,
        g_U=g_U,
        has_eq=bool(has_eq),
        has_ineq=bool(has_ineq),
        use_exact_hessian=bool(use_hessian),
        str_options=str_opts,
        num_options=num_opts,
        int_options=int_opts,
        lib_search_dir=lib_dir,
        iter_callback=iter_cb_wrapper,
    )

    mg = np.asarray(cpp_res.mult_g, dtype=np.float64)
    mult_eq = mg[:m_eq].copy() if m_eq > 0 else np.zeros(0, dtype=np.float64)
    mult_ineq = mg[m_eq:].copy() if m_ineq > 0 else np.zeros(0, dtype=np.float64)
    history = [
        {
            "iter": int(it),
            "obj": float(obj),
            "constr_viol": float(ipr),
            "inf_du": float(idu),
            "mu": float(mu_v),
            "alpha_pr": float(apr),
        }
        for it, obj, ipr, idu, mu_v, apr in zip(
            cpp_res.hist_iter,
            cpp_res.hist_obj,
            cpp_res.hist_inf_pr,
            cpp_res.hist_inf_du,
            cpp_res.hist_mu,
            cpp_res.hist_alpha_pr,
        )
    ]
    return {
        "x": np.asarray(cpp_res.x, dtype=np.float64),
        "f": float(cpp_res.obj_val),
        "mult_g": mg,
        "mult_eq": mult_eq,
        "mult_ineq": mult_ineq,
        "mult_x_L": np.asarray(cpp_res.mult_x_L, dtype=np.float64),
        "mult_x_U": np.asarray(cpp_res.mult_x_U, dtype=np.float64),
        "history": history,
        "success": bool(cpp_res.success),
        "status": int(cpp_res.status),
        "message": str(cpp_res.message),
        "nit": int(cpp_res.iterations),
        "elapsed_time": float(cpp_res.elapsed_time),
        "raw": cpp_res,
    }
