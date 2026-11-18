"""
Solver backend adapters for fastopt (scipy.optimize, C++ native IPOPT, and pymoo).
"""

from .ipopt_solver import find_casadi_ipopt_dir, solve_ipopt
from .pymoo_solver import build_pymoo_problem, solve_pymoo
from .scipy_solver import solve_scipy

__all__ = [
    "solve_scipy",
    "solve_ipopt",
    "find_casadi_ipopt_dir",
    "solve_pymoo",
    "build_pymoo_problem",
]
