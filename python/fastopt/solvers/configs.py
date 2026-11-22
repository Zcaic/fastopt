"""
Typed, backend-scoped solver configuration classes for `fastopt`.

Usage:
    import fastopt as fo

    opti = fo.Opti()
    opti.solver(fo.solvers.ipopt.IPOPT(max_iter=400, tol=1e-4, mu_strategy="adaptive"))
    opti.solver(fo.solvers.ipopt.IPOPT.lbfgs(max_iter=400))
    opti.solver(fo.solvers.scipy.SLSQP(ftol=1e-10, maxiter=100))
    opti.solver(fo.solvers.scipy.TrustConstr(gtol=1e-8, xtol=1e-8, verbose=0))
    opti.solver(fo.solvers.pymoo.DE(pop_size=80, n_gen=150, CR=0.9, F=0.5))

All solver classes are organized strictly under their backend namespace
(`ipopt`, `scipy`, `pymoo`) to avoid ambiguity for algorithms that exist
across multiple backends (e.g., `DE`).

All code comments are written in English.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable, ClassVar, Dict, Literal, Optional, Tuple, Union


# Fields handled directly by `Opti.solver(...)` rather than forwarded into `options` dict
_META_FIELDS = frozenset({"use_hessian", "auto_scale", "callback", "extra"})


@dataclass
class SolverConfig:
    """
    Base class for typed `fastopt` solver configurations.

    Subclasses define only the legal options for a specific backend algorithm so that
    IDEs provide exact autocompletion, type checking, and enum suggestions.
    Any backend-specific option not explicitly listed can be supplied via `extra`.
    """

    _backend: ClassVar[str] = ""
    _method: ClassVar[str] = ""

    use_hessian: bool = True
    auto_scale: bool = False
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def backend(self) -> str:
        """Return the solver backend identifier (`'ipopt'`, `'scipy'`, or `'pymoo'`)."""
        return self._backend

    @property
    def method(self) -> str:
        """Return the backend-specific algorithm identifier."""
        return self._method

    def to_options(self) -> Dict[str, Any]:
        """Serialize all non-None algorithm options plus `extra` into a backend options dict."""
        opts: Dict[str, Any] = {}
        for f in fields(self):
            if f.name in _META_FIELDS:
                continue
            val = getattr(self, f.name)
            if val is not None:
                opts[f.name] = val
        if self.extra:
            opts.update(self.extra)
        return opts


# ============================================================================
# 1. COIN-OR IPOPT Backend (`fo.solvers.ipopt`)
# ============================================================================

@dataclass
class IPOPT(SolverConfig):
    """
    Native C++17 COIN-OR IPOPT interior-point nonlinear programming solver.

    Supports both exact 2nd-order AD Hessians (`use_hessian=True`) and quasi-Newton
    L-BFGS approximation (`use_hessian=False`, e.g., via `IPOPT.lbfgs(...)`).
    """

    _backend: ClassVar[str] = "ipopt"
    _method: ClassVar[str] = "IPOPT"

    tol: float = 1e-8
    max_iter: int = 500
    acceptable_tol: Optional[float] = None
    acceptable_constr_viol_tol: Optional[float] = None
    acceptable_iter: Optional[int] = None
    constr_viol_tol: Optional[float] = None
    dual_inf_tol: Optional[float] = None
    compl_inf_tol: Optional[float] = None
    limited_memory_max_history: Optional[int] = None
    mu_strategy: Literal["adaptive", "monotone"] = "adaptive"
    mu_init: Optional[float] = None
    linear_solver: Literal["mumps", "ma27", "ma57", "ma77", "ma86", "ma97", "pardiso", "spral"] = "mumps"
    hessian_approximation: Optional[Literal["exact", "limited-memory"]] = None
    nlp_scaling_method: Optional[Literal["none", "gradient-based", "equilibration-based"]] = None
    max_cpu_time: Optional[float] = None
    print_level: int = 0
    use_hessian: bool = True

    @classmethod
    def lbfgs(
        cls,
        *,
        max_iter: int = 400,
        tol: float = 1e-4,
        acceptable_tol: float = 5e-4,
        acceptable_constr_viol_tol: float = 5e-4,
        acceptable_iter: int = 60,
        limited_memory_max_history: int = 50,
        mu_strategy: Literal["adaptive", "monotone"] = "adaptive",
        print_level: int = 5,
        auto_scale: bool = False,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "IPOPT":
        """
        Engineering preset for quasi-Newton L-BFGS IPOPT (`use_hessian=False`),
        well suited for coupled blackbox / BEMT / NeuralFoil aerodynamic optimizations.
        """
        return cls(
            tol=tol,
            max_iter=max_iter,
            acceptable_tol=acceptable_tol,
            acceptable_constr_viol_tol=acceptable_constr_viol_tol,
            acceptable_iter=acceptable_iter,
            limited_memory_max_history=limited_memory_max_history,
            mu_strategy=mu_strategy,
            print_level=print_level,
            use_hessian=False,
            auto_scale=auto_scale,
            callback=callback,
            extra=dict(extra) if extra is not None else {},
        )

    @classmethod
    def exact(
        cls,
        *,
        max_iter: int = 500,
        tol: float = 1e-8,
        mu_strategy: Literal["adaptive", "monotone"] = "adaptive",
        linear_solver: Literal["mumps", "ma27", "ma57", "ma77", "ma86", "ma97", "pardiso", "spral"] = "mumps",
        print_level: int = 0,
        auto_scale: bool = False,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "IPOPT":
        """Exact 2nd-order AD Hessian preset for smooth analytical NLPs (`use_hessian=True`)."""
        return cls(
            tol=tol,
            max_iter=max_iter,
            mu_strategy=mu_strategy,
            linear_solver=linear_solver,
            print_level=print_level,
            use_hessian=True,
            auto_scale=auto_scale,
            callback=callback,
            extra=dict(extra) if extra is not None else {},
        )


class ipopt:
    """Backend namespace for COIN-OR IPOPT solver configurations (`fo.solvers.ipopt.*`)."""

    IPOPT = IPOPT


# ============================================================================
# 2. SciPy `scipy.optimize.minimize` Backend (`fo.solvers.scipy`)
# ============================================================================

@dataclass
class _ScipySLSQP(SolverConfig):
    """SciPy SLSQP (Sequential Least Squares Programming) for smooth constrained NLPs."""

    _backend: ClassVar[str] = "scipy"
    _method: ClassVar[str] = "SLSQP"

    ftol: float = 1e-9
    maxiter: int = 500
    eps: Optional[float] = None
    disp: bool = False
    use_hessian: bool = False


@dataclass
class _ScipyTrustConstr(SolverConfig):
    """SciPy `trust-constr` interior-point / trust-region solver supporting exact Hessians."""

    _backend: ClassVar[str] = "scipy"
    _method: ClassVar[str] = "trust-constr"

    gtol: float = 1e-8
    xtol: float = 1e-8
    barrier_tol: float = 1e-8
    maxiter: int = 500
    initial_tr_radius: Optional[float] = None
    initial_constr_penalty: Optional[float] = None
    initial_barrier_parameter: Optional[float] = None
    initial_barrier_tolerance: Optional[float] = None
    factorization_method: Optional[Literal["NormalEquation", "AugmentedSystem", "QRFactorization", "SVDFactorization"]] = None
    verbose: Literal[0, 1, 2, 3] = 0
    use_hessian: bool = True


@dataclass
class _ScipyLBFGSB(SolverConfig):
    """SciPy L-BFGS-B for bound-constrained (unconstrained) smooth optimization."""

    _backend: ClassVar[str] = "scipy"
    _method: ClassVar[str] = "L-BFGS-B"

    ftol: float = 1e-9
    gtol: float = 1e-7
    maxcor: int = 20
    maxiter: int = 500
    maxfun: Optional[int] = None
    maxls: Optional[int] = None
    eps: Optional[float] = None
    disp: bool = False
    use_hessian: bool = False


@dataclass
class _ScipyBFGS(SolverConfig):
    """SciPy BFGS quasi-Newton solver for unconstrained smooth optimization."""

    _backend: ClassVar[str] = "scipy"
    _method: ClassVar[str] = "BFGS"

    gtol: float = 1e-7
    maxiter: int = 500
    norm: Optional[float] = None
    eps: Optional[float] = None
    xrtol: Optional[float] = None
    disp: bool = False
    use_hessian: bool = False


@dataclass
class _ScipyNewtonCG(SolverConfig):
    """SciPy Newton-CG solver using exact AD Hessian-vector products for unconstrained NLPs."""

    _backend: ClassVar[str] = "scipy"
    _method: ClassVar[str] = "Newton-CG"

    xtol: float = 1e-7
    maxiter: int = 500
    disp: bool = False
    use_hessian: bool = True


@dataclass
class _ScipyTNC(SolverConfig):
    """SciPy Truncated Newton (TNC) solver for bound-constrained optimization."""

    _backend: ClassVar[str] = "scipy"
    _method: ClassVar[str] = "TNC"

    ftol: Optional[float] = None
    gtol: Optional[float] = None
    xtol: Optional[float] = None
    maxfun: Optional[int] = None
    disp: bool = False
    use_hessian: bool = False


@dataclass
class _ScipyCG(SolverConfig):
    """SciPy Polak-Ribiere Conjugate Gradient (CG) solver for unconstrained optimization."""

    _backend: ClassVar[str] = "scipy"
    _method: ClassVar[str] = "CG"

    gtol: float = 1e-7
    maxiter: int = 500
    norm: Optional[float] = None
    eps: Optional[float] = None
    disp: bool = False
    use_hessian: bool = False


@dataclass
class _ScipyNelderMead(SolverConfig):
    """SciPy Nelder-Mead derivative-free simplex solver."""

    _backend: ClassVar[str] = "scipy"
    _method: ClassVar[str] = "Nelder-Mead"

    xatol: float = 1e-6
    fatol: float = 1e-6
    maxiter: Optional[int] = None
    maxfev: Optional[int] = None
    adaptive: bool = False
    disp: bool = False
    use_hessian: bool = False


@dataclass
class _ScipyPowell(SolverConfig):
    """SciPy Powell derivative-free conjugate direction solver."""

    _backend: ClassVar[str] = "scipy"
    _method: ClassVar[str] = "Powell"

    xtol: float = 1e-6
    ftol: float = 1e-6
    maxiter: Optional[int] = None
    maxfev: Optional[int] = None
    disp: bool = False
    use_hessian: bool = False


class scipy:
    """Backend namespace for `scipy.optimize` solver configurations (`fo.solvers.scipy.*`)."""

    SLSQP = _ScipySLSQP
    TrustConstr = _ScipyTrustConstr
    LBFGSB = _ScipyLBFGSB
    BFGS = _ScipyBFGS
    NewtonCG = _ScipyNewtonCG
    TNC = _ScipyTNC
    CG = _ScipyCG
    NelderMead = _ScipyNelderMead
    Powell = _ScipyPowell


# ============================================================================
# 3. Pymoo Evolutionary & Multi-Objective Backend (`fo.solvers.pymoo`)
# ============================================================================

@dataclass
class _PymooBase(SolverConfig):
    """Shared base configuration for pymoo evolutionary / multi-objective algorithms."""

    _backend: ClassVar[str] = "pymoo"

    pop_size: int = 100
    n_gen: int = 100
    seed: Optional[int] = 42
    verbose: bool = False
    num_threads: int = 0
    eq_tol: float = 1e-4
    use_native_eq: bool = False
    termination: Optional[Union[Tuple[str, Any], Any]] = None
    use_hessian: bool = False


@dataclass
class _PymooDE(_PymooBase):
    """Pymoo Differential Evolution (DE) for single-objective global optimization."""

    _method: ClassVar[str] = "DE"

    variant: Optional[str] = None
    CR: Optional[float] = None
    F: Optional[float] = None
    dither: Optional[Literal["vector", "scalar", "no"]] = None
    jitter: Optional[bool] = None


@dataclass
class _PymooGA(_PymooBase):
    """Pymoo Genetic Algorithm (GA) for single-objective global optimization."""

    _method: ClassVar[str] = "GA"


@dataclass
class _PymooPSO(_PymooBase):
    """Pymoo Particle Swarm Optimization (PSO) for single-objective global optimization."""

    _method: ClassVar[str] = "PSO"

    w: Optional[float] = None
    c1: Optional[float] = None
    c2: Optional[float] = None
    adaptive: Optional[bool] = None


@dataclass
class _PymooCMAES(_PymooBase):
    """Pymoo Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."""

    _method: ClassVar[str] = "CMAES"

    sigma: Optional[float] = None


@dataclass
class _PymooISRES(_PymooBase):
    """Pymoo Improved Stochastic Ranking Evolution Strategy (ISRES) for constrained problems."""

    _method: ClassVar[str] = "ISRES"


@dataclass
class _PymooSRES(_PymooBase):
    """Pymoo Stochastic Ranking Evolution Strategy (SRES) for constrained problems."""

    _method: ClassVar[str] = "SRES"


@dataclass
class _PymooPatternSearch(_PymooBase):
    """Pymoo Pattern Search derivative-free local/global solver."""

    _method: ClassVar[str] = "PatternSearch"


@dataclass
class _PymooNelderMead(_PymooBase):
    """Pymoo Nelder-Mead simplex solver."""

    _method: ClassVar[str] = "NelderMead"


@dataclass
class _PymooNSGA2(_PymooBase):
    """Pymoo NSGA-II for bi-/multi-objective Pareto optimization."""

    _method: ClassVar[str] = "NSGA2"


@dataclass
class _PymooNSGA3(_PymooBase):
    """Pymoo NSGA-III for many-objective Pareto optimization."""

    _method: ClassVar[str] = "NSGA3"

    n_partitions: Optional[int] = None


@dataclass
class _PymooMOEAD(_PymooBase):
    """Pymoo MOEA/D decomposition-based multi-objective optimization."""

    _method: ClassVar[str] = "MOEAD"

    n_neighbors: Optional[int] = None


@dataclass
class _PymooSMSEMOA(_PymooBase):
    """Pymoo SMS-EMOA hypervolume-indicator multi-objective optimization."""

    _method: ClassVar[str] = "SMSEMOA"


@dataclass
class _PymooAGEMOEA(_PymooBase):
    """Pymoo AGE-MOEA adaptive geometry estimation multi-objective optimization."""

    _method: ClassVar[str] = "AGEMOEA"


@dataclass
class _PymooAGEMOEA2(_PymooBase):
    """Pymoo AGE-MOEA-II multi-objective optimization."""

    _method: ClassVar[str] = "AGEMOEA2"


class pymoo:
    """Backend namespace for `pymoo` evolutionary & multi-objective configurations (`fo.solvers.pymoo.*`)."""

    DE = _PymooDE
    GA = _PymooGA
    PSO = _PymooPSO
    CMAES = _PymooCMAES
    ISRES = _PymooISRES
    SRES = _PymooSRES
    PatternSearch = _PymooPatternSearch
    NelderMead = _PymooNelderMead
    NSGA2 = _PymooNSGA2
    NSGA3 = _PymooNSGA3
    MOEAD = _PymooMOEAD
    SMSEMOA = _PymooSMSEMOA
    AGEMOEA = _PymooAGEMOEA
    AGEMOEA2 = _PymooAGEMOEA2


__all__ = [
    "SolverConfig",
    "ipopt",
    "scipy",
    "pymoo",
]
