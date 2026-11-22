"""
Solver backend adapters and typed backend-scoped configurations for fastopt
(`fo.solvers.ipopt.*`, `fo.solvers.scipy.*`, `fo.solvers.pymoo.*`).
"""

from .configs import SolverConfig, ipopt, pymoo, scipy
from .ipopt_solver import find_casadi_ipopt_dir, solve_ipopt
from .pymoo_solver import build_pymoo_problem, solve_pymoo
from .scipy_solver import solve_scipy

__all__ = [
    "SolverConfig",
    "ipopt",
    "scipy",
    "pymoo",
    "solve_scipy",
    "solve_ipopt",
    "find_casadi_ipopt_dir",
    "solve_pymoo",
    "build_pymoo_problem",
]
