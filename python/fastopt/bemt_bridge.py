"""
Joint Bridge Between `fast_bemt` and `fastopt` (`FastBEMTAdapter` & `BEMTSymbolicOutput`).

Supports two seamless execution channels (`use_c_abi="auto"`):
1. **Zero-GIL C-ABI `PyCapsule` Direct Channel (`fo.CBlackBox`)**:
   Connects C++ `fastopt::Evaluator` directly to C++ `bemt::BEMTSolver` via raw function pointers
   (`FastOptCBlackBox*`) with thread-local same-point flow-field caching, OpenMP multi-point
   parallelism (`eval_multipoint`), C++ inner auto-trim (`auto_trim_thrust`), 2D control-point
   abscissae (`chord_ctrl_r`, `twist_ctrl_r`), cyclic pitch (`pitch_1c`, `pitch_1s`), and root
   flapwise bending moment (`root_flap_moment`).
2. **Analytical-IFT `CustomOp` Channel (`fo.CustomOp`)**:
   Supports full 3-level chain-rule gradients for spanwise Kulfan CST airfoil parameters
   (`cst_params`) as well as fallback compatibility with older `fast_bemt` binaries.

All code comments are written in English.
"""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from . import _fastopt_impl as _core
from .blackbox import CBlackBox, CustomOp
from .expr import (
    ArrayLike,
    BSplineInterpolator2D,
    Expr,
    GraphBuilder,
    _coerce_with_graph,
    _find_graph,
    cos,
    sin,
    sqrt,
    sumsqr,
    trapz,
    vstack,
)


def _isa_atmosphere_from_altitude(altitude_m: float) -> Tuple[float, float, float]:
    """Return ISA (rho [kg/m^3], mu [Pa*s], a_sound [m/s]) at altitude_m in [0, 11000] m."""
    h = float(np.clip(altitude_m, 0.0, 11000.0))
    T0 = 288.15
    L = 0.0065
    g0 = 9.80665
    R_air = 287.05287
    T = T0 - L * h
    rho = 1.225 * (T / T0) ** (g0 / (R_air * L) - 1.0)
    mu = 1.458e-6 * (T ** 1.5) / (T + 110.4)
    a_sound = math.sqrt(1.4 * R_air * T)
    return float(rho), float(mu), float(a_sound)


def _cst_thickness_matrix(x_chk: Sequence[float], include_te: bool = False) -> np.ndarray:
    """
    Build the exact linear matrix M_thick (shape: (len(x_chk), 17 or 18)) mapping a Kulfan CST
    parameter vector [lower_weights (8), upper_weights (8), leading_edge_weight (1), (opt TE)]
    to relative thickness t/c at chordwise locations x_chk.
    """
    x = np.clip(np.asarray(x_chk, dtype=np.float64).ravel(), 1e-6, 1.0 - 1e-6)
    cx = (x ** 0.5) * (1.0 - x)
    bx = np.column_stack(
        [float(math.comb(7, k)) * (x ** k) * ((1.0 - x) ** (7 - k)) for k in range(8)]
    )
    cb = cx[:, None] * bx
    n_cst = 18 if include_te else 17
    M = np.zeros((x.size, n_cst), dtype=np.float64)
    # y_upper - y_lower = C(x) * sum_k (w_u_k - w_l_k) * B_k(x) + x * TE_thickness
    M[:, 0:8] = -cb
    M[:, 8:16] = cb
    if include_te:
        M[:, 17] = x
    return M


def _default_kulfan_polar_eval(
    kaf: Any,
    alpha_deg: np.ndarray,
    re: np.ndarray,
    ma: np.ndarray,
    *,
    n_crit: float = 9.0,
    xtr_upper: float = 1.0,
    xtr_lower: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate 3D airfoil polars (CL, CD) for a KulfanCSTAirfoil:
      - Uses `neuralfoil.get_aero_from_kulfan_parameters` if `neuralfoil` is installed;
      - Otherwise uses a smooth, physics-based compressible camber/thickness viscous airfoil
        model calibrated directly to the Kulfan CST geometry (camber line, thickness ratio, LE).
    """
    try:
        import aerosandbox as asb  # type: ignore

        asb_kaf = asb.KulfanAirfoil(
            name=getattr(kaf, "name", "KulfanAirfoil"),
            lower_weights=kaf.lower_weights,
            upper_weights=kaf.upper_weights,
            leading_edge_weight=kaf.leading_edge_weight,
            TE_thickness=kaf.TE_thickness,
        )
        aero = asb_kaf.get_aero_from_neuralfoil(
            alpha=alpha_deg,
            Re=re,
            mach=ma,
            n_crit=float(n_crit),
            xtr_upper=float(xtr_upper),
            xtr_lower=float(xtr_lower),
            model_size="large",
        )
        cl = np.asarray(aero["CL"], dtype=np.float64).ravel()
        cd = np.maximum(np.asarray(aero["CD"], dtype=np.float64).ravel(), 1e-4)
        return cl, cd
    except ImportError:
        pass

    try:
        import neuralfoil as nf  # type: ignore

        aero = nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=kaf.kulfan_parameters,
            alpha=alpha_deg,
            Re=re,
            n_crit=float(n_crit),
            xtr_upper=float(xtr_upper),
            xtr_lower=float(xtr_lower),
            model_size="large",
        )
        cl_inc = np.asarray(aero["CL"], dtype=np.float64).ravel()
        cd_inc = np.maximum(np.asarray(aero["CD"], dtype=np.float64).ravel(), 1e-4)
        # Apply smooth Prandtl-Glauert compressibility scaling when Mach > 0
        beta_pg = np.sqrt(np.maximum(1.0 - (0.8 * ma) ** 2, 0.25))
        cl = cl_inc / beta_pg
        cd = cd_inc
        return cl, cd
    except ImportError:
        pass

    # Built-in smooth compressible viscous Kulfan CST airfoil surrogate (zero external deps)
    x_c = np.linspace(0.02, 0.98, 25)
    cx = (x_c ** 0.5) * (1.0 - x_c)
    bx = np.column_stack(
        [float(math.comb(7, k)) * (x_c ** k) * ((1.0 - x_c) ** (7 - k)) for k in range(8)]
    )
    cb = cx[:, None] * bx
    lem = x_c * ((1.0 - x_c) ** 8.5)
    yu = cb @ kaf.upper_weights + kaf.leading_edge_weight * lem + 0.5 * x_c * kaf.TE_thickness
    yl = cb @ kaf.lower_weights + kaf.leading_edge_weight * lem - 0.5 * x_c * kaf.TE_thickness
    camber = 0.5 * (yu + yl)
    thick = np.maximum(yu - yl, 0.005)
    t_max = float(np.max(thick))

    # Thin-airfoil zero-lift angle integral approximation
    theta_c = np.arccos(1.0 - 2.0 * x_c)
    dz_dx = np.gradient(camber, x_c)
    alpha_0 = -float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(dz_dx * (np.cos(theta_c) - 1.0), theta_c) / np.pi)
    camber_max = float(np.max(np.abs(camber)))

    a_rad = np.deg2rad(alpha_deg)
    a_eff = a_rad - alpha_0
    pg = 1.0 / np.sqrt(np.maximum(1.0 - (0.8 * ma) ** 2, 0.15))

    # Thick sections have slightly lower lift slope and higher drag; moderate camber delays stall
    cl_slope = 2.0 * np.pi * (1.0 - 0.35 * max(t_max - 0.12, 0.0)) * pg
    a_stall = np.deg2rad(13.5 + 12.0 * min(t_max, 0.30) + 15.0 * min(camber_max, 0.06))
    stall_blend = 1.0 / (1.0 + (a_eff / a_stall) ** 6)
    cl = cl_slope * a_eff * stall_blend + 1.05 * np.sin(2.0 * a_eff) * (1.0 - stall_blend)

    re_scale = (np.maximum(re, 1e4) / 1.0e6) ** (-0.12)
    cd_0 = (0.0062 + 0.032 * (t_max ** 1.6) + 0.12 * (camber_max ** 2)) * re_scale
    cd_wave = 0.08 * np.maximum(ma - (0.78 - 0.5 * t_max), 0.0) ** 2
    cd = cd_0 + 0.022 * (a_eff / np.deg2rad(10.0)) ** 2 + 0.45 * (1.0 - stall_blend) * (np.sin(a_eff) ** 2) + cd_wave
    return cl, np.maximum(cd, 1e-4)


class BEMTSymbolicOutput:
    """
    Container of symbolic `fastopt.Expr` outputs returned by `FastBEMTAdapter(...)`.
    Provides both attribute access (`aero.thrust`, `aero.power`, `aero.solidity`,
    `aero.root_flap_bending_moment`, `aero.propulsive_efficiency`) and dict key access.
    """

    def __init__(
        self,
        raw_vec: Expr,
        geom: Any,
        rho: float,
        chord_ctrl_expr: Optional[Expr] = None,
        twist_ctrl_expr: Optional[Expr] = None,
        omega_expr: Optional[Expr] = None,
        v_inf_expr: Optional[Expr] = None,
        trimmed_pitch_0: Optional[Expr] = None,
        n_sec_aero: int = 0,
    ) -> None:
        """
        Initialize symbolic rotor aerodynamic output expressions from the 8-element BEMT output vector
        `[thrust, torque, power, fx, fy, mx, my, root_flap_moment]`.
        """
        self._raw_vec = raw_vec
        self._geom = geom
        self._rho = float(rho)

        self.thrust: Expr = raw_vec[0]
        self.torque: Expr = raw_vec[1]
        self.power: Expr = raw_vec[2]
        self.fx: Expr = raw_vec[3]
        self.fy: Expr = raw_vec[4]
        self.mx: Expr = raw_vec[5]
        self.my: Expr = raw_vec[6]
        self.root_flap_moment: Expr = raw_vec[7] if raw_vec.size >= 8 else raw_vec[0] * 0.0
        self.root_flap_bending_moment: Expr = self.root_flap_moment
        self.trimmed_pitch_0: Optional[Expr] = trimmed_pitch_0
        if n_sec_aero > 0 and raw_vec.size >= 8 + 2 * n_sec_aero:
            self.section_alpha_rad: Optional[Expr] = raw_vec[8 : 8 + n_sec_aero]
            self.section_alpha_deg: Optional[Expr] = self.section_alpha_rad * (180.0 / math.pi)
            self.section_re: Optional[Expr] = raw_vec[8 + n_sec_aero : 8 + 2 * n_sec_aero]
        else:
            self.section_alpha_rad = None
            self.section_alpha_deg = None
            self.section_re = None

        g = raw_vec.graph
        B_c = np.asarray(geom.chord_basis, dtype=np.float64)
        B_t = np.asarray(geom.twist_basis, dtype=np.float64)
        r_stations = np.asarray(geom.r, dtype=np.float64)

        if chord_ctrl_expr is not None:
            B_c_e = _coerce_with_graph(g, B_c)
            self.chord_r: Expr = B_c_e @ chord_ctrl_expr
        else:
            self.chord_r = _coerce_with_graph(g, np.asarray(geom.chord, dtype=np.float64))

        if twist_ctrl_expr is not None:
            B_t_e = _coerce_with_graph(g, B_t)
            self.twist_r: Expr = B_t_e @ twist_ctrl_expr
        else:
            self.twist_r = _coerce_with_graph(g, np.asarray(geom.twist, dtype=np.float64))

        # Symbolic single-blade planform area S_blade = int_{R_hub}^{R} c(r) dr via trapezoidal rule
        self.blade_area: Expr = trapz(self.chord_r, x=r_stations)
        disk_area = float(np.pi * (geom.radius ** 2))
        self.solidity: Expr = (float(geom.num_blades) / disk_area) * self.blade_area

        # Symbolic Hover Figure of Merit: FM = P_ideal / P = (T * sqrt(T / (2 * rho * pi * R^2))) / P
        t_pos = 0.5 * (self.thrust + sqrt(self.thrust * self.thrust + 1e-6))
        p_ideal = t_pos * sqrt(t_pos / (2.0 * self._rho * disk_area))
        p_safe = 0.5 * (self.power + sqrt(self.power * self.power + 1.0))
        self.figure_of_merit: Expr = p_ideal / p_safe

        # Symbolic non-dimensional rotor coefficients CT, CQ, CP and propulsive efficiency eta
        if omega_expr is not None:
            v_tip = omega_expr * float(geom.radius)
            q_disk = self._rho * disk_area * (v_tip * v_tip) + 1e-9
            self.ct: Expr = self.thrust / q_disk
            self.cq: Expr = self.torque / (q_disk * float(geom.radius))
            self.cp: Expr = self.power / (q_disk * v_tip + 1e-9)
        else:
            self.ct = self.thrust * 0.0
            self.cq = self.torque * 0.0
            self.cp = self.power * 0.0

        if v_inf_expr is not None:
            self.propulsive_efficiency: Expr = (self.thrust * v_inf_expr) / p_safe
        else:
            self.propulsive_efficiency = self.thrust * 0.0

    @property
    def raw_vec(self) -> Expr:
        """Return the raw `(8, 1)` or `(9, 1)` symbolic vector."""
        return self._raw_vec

    def __getitem__(self, key: str) -> Expr:
        """Access any symbolic aerodynamic or geometric output by string key."""
        if hasattr(self, key):
            val = getattr(self, key)
            if isinstance(val, Expr):
                return val
        raise KeyError(
            f"Unknown BEMTSymbolicOutput key '{key}'. Available: 'thrust', 'torque', 'power', "
            f"'fx', 'fy', 'mx', 'my', 'root_flap_moment', 'chord_r', 'twist_r', 'blade_area', "
            f"'solidity', 'figure_of_merit', 'ct', 'cq', 'cp', 'propulsive_efficiency'."
        )


_NEURALFOIL_BRIDGE_CACHE: Dict[Tuple[str, bool], Tuple[Any, Any]] = {}


def _resolve_cpp_neuralfoil(
    fb_mod: Any,
    neuralfoil_model: Any,
    use_float32: bool,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Resolve (fastopt.nn.NeuralFoilSurrogate, fast_bemt.NeuralFoilModel) with process-wide caching."""
    if neuralfoil_model is None or neuralfoil_model is False:
        return None, None

    from .nn import NeuralFoilSurrogate

    if isinstance(neuralfoil_model, NeuralFoilSurrogate):
        bemt_m = (
            neuralfoil_model.to_fast_bemt_model()
            if hasattr(fb_mod, "NeuralFoilModel")
            else None
        )
        return neuralfoil_model, bemt_m

    model_size = "large" if neuralfoil_model == "auto" else str(neuralfoil_model)
    cache_key = (model_size, bool(use_float32))
    if cache_key in _NEURALFOIL_BRIDGE_CACHE:
        return _NEURALFOIL_BRIDGE_CACHE[cache_key]

    try:
        nf_surr = NeuralFoilSurrogate.from_model_size(
            model_size, use_float32=bool(use_float32), slice_aero_outputs=True
        )
        bemt_m = (
            nf_surr.to_fast_bemt_model()
            if hasattr(fb_mod, "NeuralFoilModel")
            else None
        )
        _NEURALFOIL_BRIDGE_CACHE[cache_key] = (nf_surr, bemt_m)
        return nf_surr, bemt_m
    except (FileNotFoundError, ImportError):
        if neuralfoil_model != "auto":
            raise
        return None, None


class FastBEMTAdapter:
    """
    High-performance bridge connecting `fast_bemt` (`RotorGeometry`, `BEMTSolver`, `OperatingCondition`)
    with `fastopt` (`Opti`, `GraphBuilder`, `Evaluator`, `BSplineInterpolator2D`).

    Phase 14 Capabilities:
      - Zero-GIL C-ABI `PyCapsule` channel with thread-local same-point flow-field caching.
      - OpenMP parallel multi-point evaluation (`eval_multipoint`) with block-sparse Jacobian masks.
      - C++ inner damped-Newton auto-trim (`auto_trim_thrust`) eliminating `pitch_0` via IFT.
      - 2D control-point radial abscissae (`chord_ctrl_r`, `twist_ctrl_r`), cyclic pitch (`pitch_1c`, `pitch_1s`),
        and spanwise Kulfan CST airfoil optimization (`cst_params` + `cst_thickness`).
      - Root flapwise bending moment (`root_flap_moment`) and `add_planform_constraints(...)` helper.
    """

    def __init__(
        self,
        geom: Any,
        cfg: Optional[Any] = None,
        base_cond: Optional[Any] = None,
        *,
        altitude_m: Optional[float] = None,
        use_c_abi: Union[bool, str] = "auto",
        fd_step: float = 1e-5,
        fd_hessian: bool = True,
        section_airfoils_kulfan: Optional[Sequence[Any]] = None,
        polar_eval_fn: Optional[Callable[[Any, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
        neuralfoil_model: Any = "auto",
        neuralfoil_use_float32: bool = True,
        neuralfoil_compressibility: bool = True,
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        expose_section_aero: bool = False,
        name: str = "fast_bemt",
    ) -> None:
        """
        Initialize a `FastBEMTAdapter` for a given baseline `fast_bemt.RotorGeometry`.

        Args:
            geom: `fast_bemt.RotorGeometry` instance defining the baseline rotor and 3D airfoil polars.
            cfg: Optional `fast_bemt.SolverConfig` (defaults to `fb.SolverConfig()`).
            base_cond: Optional baseline `fast_bemt.OperatingCondition`.
            altitude_m: Optional ISA altitude [m] (0..11000 m) to set `(rho, mu, a_sound)` on `base_cond`.
            use_c_abi: `'auto'` (default), `True` (require C-ABI), or `False` (Python `CustomOp`).
            fd_step: Relative directional step size for 2nd-order HVP over the analytical 1st-order Jacobian.
            fd_hessian: Enable directional 2nd-order HVP over the analytical 1st-order Jacobian.
            section_airfoils_kulfan: Optional sequence of `KulfanCSTAirfoil` objects matching `geom.section_r_norm`.
            polar_eval_fn: Optional custom `(kulfan_airfoil, alpha_deg, Re, Ma) -> (CL, CD)` evaluator for CST mode.
            n_crit: Critical amplification factor for NeuralFoil boundary-layer transition (default 9.0; use ~4.0 for turbulent rotor flow).
            xtr_upper: Forced upper-surface transition location x/c in [0.0, 1.0] (1.0 = free transition; 0.05 = tripped near LE).
            xtr_lower: Forced lower-surface transition location x/c in [0.0, 1.0] (1.0 = free transition; 0.05 = tripped near LE).
            expose_section_aero: If True, expose differentiable local section `(section_alpha_rad, section_alpha_deg, section_re)` on `BEMTSymbolicOutput` in CST blade mode.
            name: Base identifier name for computation graph nodes.
        """
        import fast_bemt as fb

        self._fb = fb
        self._geom = geom
        self._cfg = cfg if cfg is not None else fb.SolverConfig()
        if base_cond is not None:
            self._base_cond = fb.OperatingCondition(
                vx=base_cond.vx,
                vy=base_cond.vy,
                vz=base_cond.vz,
                omega=base_cond.omega,
                pitch_0=base_cond.pitch_0,
                pitch_1c=base_cond.pitch_1c,
                pitch_1s=base_cond.pitch_1s,
                rho=base_cond.rho,
                mu=base_cond.mu,
                a_sound=base_cond.a_sound,
            )
        else:
            self._base_cond = fb.OperatingCondition()

        if altitude_m is not None:
            rho, mu, a_sound = _isa_atmosphere_from_altitude(altitude_m)
            self._base_cond.rho = rho
            self._base_cond.mu = mu
            self._base_cond.a_sound = a_sound

        self._solver = fb.BEMTSolver(self._geom, self._cfg)
        self._fd_step = float(fd_step)
        self._fd_hessian = bool(fd_hessian)
        self._name = str(name)
        self._custom_polar_eval = polar_eval_fn is not None
        self._polar_eval_fn = polar_eval_fn if polar_eval_fn is not None else _default_kulfan_polar_eval
        self._nf_compressibility = bool(neuralfoil_compressibility)
        if not (0.0 <= float(xtr_upper) <= 1.0) or not (0.0 <= float(xtr_lower) <= 1.0):
            raise ValueError(
                f"FastBEMTAdapter: xtr_upper ({xtr_upper}) and xtr_lower ({xtr_lower}) must be in [0.0, 1.0]."
            )
        if float(n_crit) < 0.0:
            raise ValueError(f"FastBEMTAdapter: n_crit ({n_crit}) must be non-negative.")
        self._nf_n_crit = float(n_crit)
        self._nf_xtr_upper = float(xtr_upper)
        self._nf_xtr_lower = float(xtr_lower)
        self._expose_section_aero = bool(expose_section_aero)
        if not self._custom_polar_eval:
            self._nf_surrogate, self._nf_bemt_model = _resolve_cpp_neuralfoil(
                fb, neuralfoil_model, neuralfoil_use_float32
            )
        else:
            self._nf_surrogate, self._nf_bemt_model = None, None

        if section_airfoils_kulfan is not None:
            self._kulfans = [
                fb.KulfanCSTAirfoil.from_aerosandbox(af)
                if not isinstance(af, fb.KulfanCSTAirfoil)
                else af
                for af in section_airfoils_kulfan
            ]
        else:
            self._kulfans = []

        has_c_abi = hasattr(self._solver, "fastopt_capsule") and hasattr(fb, "create_blade_design_capsule")
        if use_c_abi == "auto":
            self._use_c_abi = bool(has_c_abi)
        elif bool(use_c_abi):
            if not has_c_abi:
                raise RuntimeError(
                    "FastBEMTAdapter: use_c_abi=True requested, but the loaded fast_bemt binary "
                    "does not expose fastopt_capsule / create_blade_design_capsule."
                )
            self._use_c_abi = True
        else:
            self._use_c_abi = False

        self._cond_op: Optional[Union[CBlackBox, CustomOp]] = None
        self._blade_ops: Dict[bool, Union[CBlackBox, CustomOp]] = {}

    @property
    def geometry(self) -> Any:
        """Return the baseline `fast_bemt.RotorGeometry`."""
        return self._geom

    @property
    def config(self) -> Any:
        """Return the `fast_bemt.SolverConfig`."""
        return self._cfg

    @property
    def base_condition(self) -> Any:
        """Return the baseline `fast_bemt.OperatingCondition`."""
        return self._base_cond

    @property
    def solver(self) -> Any:
        """Return the persistent baseline `fast_bemt.BEMTSolver`."""
        return self._solver

    @property
    def uses_c_abi(self) -> bool:
        """Return `True` if bound via the zero-GIL C-ABI `PyCapsule` channel."""
        return self._use_c_abi

    @property
    def uses_cpp_neuralfoil(self) -> bool:
        """Return `True` if hybrid C++ NeuralFoil batch polar + analytical CST backprop is active."""
        return (self._nf_surrogate is not None) and (self._nf_bemt_model is not None)

    @property
    def n_crit(self) -> float:
        """Return configured NeuralFoil critical amplification factor `n_crit`."""
        return self._nf_n_crit

    @property
    def xtr_upper(self) -> float:
        """Return configured NeuralFoil upper-surface forced transition location `xtr_upper`."""
        return self._nf_xtr_upper

    @property
    def xtr_lower(self) -> float:
        """Return configured NeuralFoil lower-surface forced transition location `xtr_lower`."""
        return self._nf_xtr_lower

    @property
    def expose_section_aero(self) -> bool:
        """Return `True` if local section `(alpha, Re)` symbolic outputs are enabled."""
        return self._expose_section_aero

    def build_section_interp_matrix(self, sec_r_norm: Optional[Sequence[float]] = None) -> np.ndarray:
        """
        Build the piecewise-linear interpolation matrix `S_sec` of shape `(m_sec, n_radial)`
        mapping azimuth-averaged radial station quantities to airfoil control sections `sec_r_norm`
        (clamping endpoint sections to the nearest interior blade elements).
        """
        r_norm = np.asarray(self._geom.r, dtype=np.float64).ravel() / float(self._geom.radius)
        sec_r = (
            np.asarray(self._geom.section_r_norm, dtype=np.float64).ravel()
            if sec_r_norm is None
            else np.asarray(sec_r_norm, dtype=np.float64).ravel()
        )
        nr = r_norm.size
        S = np.zeros((sec_r.size, nr), dtype=np.float64)
        for s_idx, rs in enumerate(sec_r):
            if rs <= r_norm[0]:
                S[s_idx, 0] = 1.0
            elif rs >= r_norm[-1]:
                S[s_idx, nr - 1] = 1.0
            else:
                hi = int(np.searchsorted(r_norm, rs))
                lo = hi - 1
                w_hi = (rs - r_norm[lo]) / max(r_norm[hi] - r_norm[lo], 1e-12)
                S[s_idx, lo] = 1.0 - w_hi
                S[s_idx, hi] = w_hi
        return S

    @staticmethod
    def isa_atmosphere(altitude_m: float) -> Tuple[float, float, float]:
        """Return ISA `(rho [kg/m^3], mu [Pa*s], a_sound [m/s])` at `altitude_m` [m]."""
        return _isa_atmosphere_from_altitude(altitude_m)

    def cst_thickness(
        self,
        cst_params: Union[Expr, ArrayLike],
        x_chk: Sequence[float] = (0.08, 0.20, 0.30, 0.50, 0.75, 0.90),
        include_te: bool = False,
    ) -> Expr:
        """
        Compute exact symbolic airfoil relative thickness `t/c` at chordwise stations `x_chk`
        from a Kulfan CST parameter matrix `cst_params` of shape `(m_sec, 17)` or `(17, 1)`.
        """
        g = _find_graph(cst_params)
        cst_e = _coerce_with_graph(g, cst_params)
        M_t = _coerce_with_graph(g, _cst_thickness_matrix(x_chk, include_te=include_te))
        if cst_e.shape[0] in (17, 18) and cst_e.shape[1] == 1:
            return M_t @ cst_e
        return cst_e @ M_t.T

    def solve_condition(
        self,
        cond: Optional[Any] = None,
        *,
        v_inf: Optional[float] = None,
        alpha_disk_rad: float = 0.0,
        omega: Optional[float] = None,
        pitch_0: Optional[float] = None,
        pitch_1c: Optional[float] = None,
        pitch_1s: Optional[float] = None,
        beta_rad: float = 0.0,
        compute_gradients: bool = True,
    ) -> Any:
        """Directly solve a single operating condition using `fast_bemt.BEMTSolver`."""
        fb = self._fb
        if cond is not None:
            c_use = cond
        elif v_inf is not None:
            c_use = fb.OperatingCondition.from_freestream(
                v_inf=float(v_inf),
                alpha_disk_rad=float(alpha_disk_rad),
                omega_rad_s=float(omega if omega is not None else self._base_cond.omega),
                pitch_0_rad=float(pitch_0 if pitch_0 is not None else self._base_cond.pitch_0),
                pitch_1c_rad=float(pitch_1c if pitch_1c is not None else self._base_cond.pitch_1c),
                pitch_1s_rad=float(pitch_1s if pitch_1s is not None else self._base_cond.pitch_1s),
                beta_rad=float(beta_rad),
                rho=self._base_cond.rho,
                mu=self._base_cond.mu,
                a_sound=self._base_cond.a_sound,
            )
        else:
            c_use = fb.OperatingCondition(
                vx=self._base_cond.vx,
                vy=self._base_cond.vy,
                vz=self._base_cond.vz,
                omega=float(omega if omega is not None else self._base_cond.omega),
                pitch_0=float(pitch_0 if pitch_0 is not None else self._base_cond.pitch_0),
                pitch_1c=float(pitch_1c if pitch_1c is not None else self._base_cond.pitch_1c),
                pitch_1s=float(pitch_1s if pitch_1s is not None else self._base_cond.pitch_1s),
                rho=self._base_cond.rho,
                mu=self._base_cond.mu,
                a_sound=self._base_cond.a_sound,
            )
        return self._solver.solve(c_use, bool(compute_gradients))

    def _get_cond_operator(self) -> Union[CBlackBox, CustomOp]:
        if self._cond_op is not None:
            return self._cond_op

        if self._use_c_abi:
            cap, holder = self._solver.fastopt_capsule(self._base_cond)
            self._cond_op = CBlackBox.from_capsule(
                capsule=cap,
                shape=(8, 1),
                fd_step=self._fd_step,
                fd_hessian=self._fd_hessian,
                name=f"{self._name}_cond_c_abi",
                keep_alive=(self._solver, holder),
            )
            return self._cond_op

        fb = self._fb
        solver = self._solver
        base_c = self._base_cond
        cache: Dict[str, Any] = {"u": None, "res": None}

        def _eval_or_grad(u_vec: np.ndarray, need_grad: bool):
            u_flat = np.asarray(u_vec, dtype=np.float64).ravel()
            if (
                cache["u"] is not None
                and np.array_equal(cache["u"], u_flat)
                and (not need_grad or cache["res"].sensitivity.valid)
            ):
                return cache["res"]
            cond = fb.OperatingCondition(
                vx=float(u_flat[2]),
                vy=float(u_flat[3]),
                vz=float(u_flat[4]),
                omega=float(u_flat[0]),
                pitch_0=float(u_flat[1]),
                pitch_1c=float(u_flat[5]) if u_flat.size > 5 else base_c.pitch_1c,
                pitch_1s=float(u_flat[6]) if u_flat.size > 6 else base_c.pitch_1s,
                rho=base_c.rho,
                mu=base_c.mu,
                a_sound=base_c.a_sound,
            )
            res = solver.solve(cond, bool(need_grad))
            cache["u"] = u_flat.copy()
            cache["res"] = res
            return res

        def _fw(u_vec: np.ndarray) -> np.ndarray:
            r = _eval_or_grad(u_vec, False)
            return np.array(
                [r.thrust, r.torque, r.power, r.fx, r.fy, r.mx, r.my, getattr(r, "root_flap_moment", 0.0)],
                dtype=np.float64,
            )

        def _jac(u_vec: np.ndarray) -> np.ndarray:
            r = _eval_or_grad(u_vec, True)
            J = np.zeros((8, 7), dtype=np.float64)
            J[:7, :] = np.asarray(r.sensitivity.dForces_dCond, dtype=np.float64)
            if hasattr(r.sensitivity, "dMflap_dCond"):
                J[7, :] = np.asarray(r.sensitivity.dMflap_dCond, dtype=np.float64)
            return J

        self._cond_op = CustomOp(
            forward_fn=_fw,
            shape=(8, 1),
            jacobian_fn=_jac,
            fd_step=self._fd_step,
            fd_hessian=self._fd_hessian,
            name=f"{self._name}_cond_py",
        )
        return self._cond_op

    def _get_blade_operator(self, use_ctrl_r: bool = False) -> Union[CBlackBox, CustomOp]:
        if use_ctrl_r in self._blade_ops:
            return self._blade_ops[use_ctrl_r]

        if self._use_c_abi:
            cap, holder = self._fb.create_blade_design_capsule(
                self._geom, self._cfg, self._base_cond, use_ctrl_r=bool(use_ctrl_r)
            )
            op = CBlackBox.from_capsule(
                capsule=cap,
                shape=(8, 1),
                fd_step=self._fd_step,
                fd_hessian=self._fd_hessian,
                name=f"{self._name}_blade_c_abi",
                keep_alive=(self._geom, holder),
            )
            self._blade_ops[use_ctrl_r] = op
            return op

        fb = self._fb
        base_g = self._geom
        cfg = self._cfg
        base_c = self._base_cond
        n_c = int(base_g.chord_ctrl.size)
        n_t = int(base_g.twist_ctrl.size)
        cache: Dict[str, Any] = {"u": None, "res": None}

        def _eval_or_grad(u_vec: np.ndarray, need_grad: bool):
            u_flat = np.asarray(u_vec, dtype=np.float64).ravel()
            if (
                cache["u"] is not None
                and np.array_equal(cache["u"], u_flat)
                and (not need_grad or cache["res"].sensitivity.valid)
            ):
                return cache["res"]
            c_vals = u_flat[:n_c]
            t_vals = u_flat[n_c : n_c + n_t]
            off = n_c + n_t
            if use_ctrl_r:
                c_r = u_flat[off : off + n_c]
                t_r = u_flat[off + n_c : off + n_c + n_t]
                off += n_c + n_t
            else:
                c_r = base_g.chord_ctrl_r_norm
                t_r = base_g.twist_ctrl_r_norm
            u_cond = u_flat[off:]
            cc = fb.BSplineCurve(c_r, c_vals, int(base_g.chord_curve.degree))
            tc = fb.BSplineCurve(t_r, t_vals, int(base_g.twist_curve.degree))
            geom_i = fb.RotorGeometry(
                base_g.num_blades, base_g.radius, base_g.hub_radius, base_g.n_radial,
                cc, tc, base_g.section_airfoils, base_g.section_r_norm, base_g.spin_dir,
            )
            cond = fb.OperatingCondition(
                vx=float(u_cond[2]), vy=float(u_cond[3]), vz=float(u_cond[4]),
                omega=float(u_cond[0]), pitch_0=float(u_cond[1]),
                pitch_1c=float(u_cond[5]) if u_cond.size > 5 else base_c.pitch_1c,
                pitch_1s=float(u_cond[6]) if u_cond.size > 6 else base_c.pitch_1s,
                rho=base_c.rho, mu=base_c.mu, a_sound=base_c.a_sound,
            )
            res = fb.BEMTSolver(geom_i, cfg).solve(cond, bool(need_grad))
            cache["u"] = u_flat.copy()
            cache["res"] = res
            return res

        def _fw(u_vec: np.ndarray) -> np.ndarray:
            r = _eval_or_grad(u_vec, False)
            return np.array(
                [r.thrust, r.torque, r.power, r.fx, r.fy, r.mx, r.my, getattr(r, "root_flap_moment", 0.0)],
                dtype=np.float64,
            )

        def _jac(u_vec: np.ndarray) -> np.ndarray:
            r = _eval_or_grad(u_vec, True)
            s = r.sensitivity
            off = (2 * n_c + 2 * n_t) if use_ctrl_r else (n_c + n_t)
            J = np.zeros((8, off + 7), dtype=np.float64)
            J[0, :n_c] = s.dT_dchord_ctrl
            J[1, :n_c] = s.dQ_dchord_ctrl
            J[2, :n_c] = s.dP_dchord_ctrl
            J[3, :n_c] = s.dFx_dchord_ctrl
            J[4, :n_c] = s.dFy_dchord_ctrl
            J[5, :n_c] = s.dMx_dchord_ctrl
            J[6, :n_c] = s.dMy_dchord_ctrl
            J[7, :n_c] = s.dMflap_dchord_ctrl
            J[0, n_c : n_c + n_t] = s.dT_dtwist_ctrl
            J[1, n_c : n_c + n_t] = s.dQ_dtwist_ctrl
            J[2, n_c : n_c + n_t] = s.dP_dtwist_ctrl
            J[3, n_c : n_c + n_t] = s.dFx_dtwist_ctrl
            J[4, n_c : n_c + n_t] = s.dFy_dtwist_ctrl
            J[5, n_c : n_c + n_t] = s.dMx_dtwist_ctrl
            J[6, n_c : n_c + n_t] = s.dMy_dtwist_ctrl
            J[7, n_c : n_c + n_t] = s.dMflap_dtwist_ctrl
            if use_ctrl_r:
                oc = n_c + n_t
                ot = oc + n_c
                J[0, oc : oc + n_c] = s.dT_dchord_ctrl_r
                J[1, oc : oc + n_c] = s.dQ_dchord_ctrl_r
                J[2, oc : oc + n_c] = s.dP_dchord_ctrl_r
                J[3, oc : oc + n_c] = s.dFx_dchord_ctrl_r
                J[4, oc : oc + n_c] = s.dFy_dchord_ctrl_r
                J[5, oc : oc + n_c] = s.dMx_dchord_ctrl_r
                J[6, oc : oc + n_c] = s.dMy_dchord_ctrl_r
                J[7, oc : oc + n_c] = s.dMflap_dchord_ctrl_r
                J[0, ot : ot + n_t] = s.dT_dtwist_ctrl_r
                J[1, ot : ot + n_t] = s.dQ_dtwist_ctrl_r
                J[2, ot : ot + n_t] = s.dP_dtwist_ctrl_r
                J[3, ot : ot + n_t] = s.dFx_dtwist_ctrl_r
                J[4, ot : ot + n_t] = s.dFy_dtwist_ctrl_r
                J[5, ot : ot + n_t] = s.dMx_dtwist_ctrl_r
                J[6, ot : ot + n_t] = s.dMy_dtwist_ctrl_r
                J[7, ot : ot + n_t] = s.dMflap_dtwist_ctrl_r
            J[:7, off : off + 7] = s.dForces_dCond
            J[7, off : off + 7] = s.dMflap_dCond
            return J

        op_py = CustomOp(
            forward_fn=_fw, shape=(8, 1), jacobian_fn=_jac,
            fd_step=self._fd_step, fd_hessian=self._fd_hessian, name=f"{self._name}_blade_py",
        )
        self._blade_ops[use_ctrl_r] = op_py
        return op_py

    def _build_cst_blade_operator(
        self,
        m_sec: int,
        n_cst: int,
        use_ctrl_r: bool,
        include_te: bool,
    ) -> CustomOp:
        fb = self._fb
        base_g = self._geom
        cfg = self._cfg
        base_c = self._base_cond
        n_c = int(base_g.chord_ctrl.size)
        n_t = int(base_g.twist_ctrl.size)
        polar_fn = self._polar_eval_fn

        alpha_deg_grid = np.linspace(-25.0, 25.0, 51, dtype=np.float64)
        alpha_rad_grid = np.deg2rad(alpha_deg_grid)
        re_grid = np.array([1.0e5, 5.0e5, 1.5e6, 5.0e6], dtype=np.float64)
        ma_grid = np.array([0.0, 0.25, 0.50, 0.75], dtype=np.float64)
        A_deg, RE_g, MA_g = np.meshgrid(alpha_deg_grid, re_grid, ma_grid, indexing="ij")
        a_flat, re_flat, ma_flat = A_deg.ravel(), RE_g.ravel(), MA_g.ravel()

        # 2D (alpha, Re) grid for 4x faster C++ NeuralFoil batch evaluation + analytic Prandtl-Glauert broadcast
        A2_deg, RE2_g = np.meshgrid(alpha_deg_grid, re_grid, indexing="ij")
        a2_flat, re2_flat = A2_deg.ravel(), RE2_g.ravel()
        nf_surr = self._nf_surrogate
        nf_bemt = self._nf_bemt_model
        nf_comp = self._nf_compressibility
        nf_n_crit = self._nf_n_crit
        nf_xtr_u = self._nf_xtr_upper
        nf_xtr_l = self._nf_xtr_lower
        pg_inv_beta = (
            1.0 / np.sqrt(np.maximum(1.0 - (0.8 * ma_grid) ** 2, 0.25))
            if nf_comp
            else np.ones_like(ma_grid)
        )

        base_kulfans = (
            list(self._kulfans)
            if len(self._kulfans) == m_sec
            else [fb.KulfanCSTAirfoil.from_naca4("4412") for _ in range(m_sec)]
        )
        sec_r_norm = (
            np.asarray(base_g.section_r_norm, dtype=np.float64).ravel()
            if base_g.section_r_norm.size == m_sec
            else np.linspace(base_g.hub_radius / base_g.radius, 1.0, m_sec)
        )
        W_sec = fb.build_spanwise_section_weight_matrix(sec_r_norm, np.asarray(base_g.r) / base_g.radius)
        expose_sec = self._expose_section_aero
        S_sec = self.build_section_interp_matrix(sec_r_norm) if expose_sec else None
        out_dim = (8 + 2 * m_sec) if expose_sec else 8

        af_cache: Dict[bytes, Any] = {}
        cache: Dict[str, Any] = {"u": None, "geom": None, "cond": None, "res": None, "kulfans": None}

        use_native_nf = (
            (nf_bemt is not None)
            and (not self._custom_polar_eval)
            and hasattr(fb.Airfoil3D, "from_neuralfoil")
        )

        def _make_airfoil3d(kaf: Any) -> Any:
            key = kaf.to_vector(include_te=True).round(12).tobytes()
            if key in af_cache:
                return af_cache[key]
            if use_native_nf:
                # Direct native C++ NeuralFoil airfoil: zero 3D polar table discretization error
                # and exact parity with analytical CST backpropagation in _jac.
                af = fb.Airfoil3D.from_neuralfoil(
                    nf_bemt,
                    np.asarray(kaf.upper_weights, dtype=np.float64),
                    np.asarray(kaf.lower_weights, dtype=np.float64),
                    float(kaf.leading_edge_weight),
                    float(kaf.TE_thickness),
                    n_crit=nf_n_crit,
                    xtr_upper=nf_xtr_u,
                    xtr_lower=nf_xtr_l,
                    enable_compressibility=nf_comp,
                )
            else:
                if nf_surr is not None and not self._custom_polar_eval:
                    aero2d = nf_surr.get_aero_from_kulfan_parameters(
                        kaf.kulfan_parameters,
                        a2_flat,
                        re2_flat,
                        n_crit=nf_n_crit,
                        xtr_upper=nf_xtr_u,
                        xtr_lower=nf_xtr_l,
                    )
                    cl_2d = np.asarray(aero2d["CL"], dtype=np.float64).reshape(
                        (alpha_deg_grid.size, re_grid.size, 1)
                    )
                    cd_2d = np.maximum(
                        np.asarray(aero2d["CD"], dtype=np.float64), 1e-4
                    ).reshape((alpha_deg_grid.size, re_grid.size, 1))
                    cl_f = (cl_2d * pg_inv_beta[None, None, :]).ravel()
                    cd_f = np.broadcast_to(cd_2d, (alpha_deg_grid.size, re_grid.size, ma_grid.size)).ravel()
                elif not self._custom_polar_eval:
                    cl_f, cd_f = _default_kulfan_polar_eval(
                        kaf,
                        a_flat,
                        re_flat,
                        ma_flat,
                        n_crit=nf_n_crit,
                        xtr_upper=nf_xtr_u,
                        xtr_lower=nf_xtr_l,
                    )
                else:
                    cl_f, cd_f = polar_fn(kaf, a_flat, re_flat, ma_flat)
                af = fb.Airfoil3D(alpha_rad_grid, re_grid, ma_grid, cl_f, cd_f)
            if len(af_cache) > 512:
                af_cache.clear()
            af_cache[key] = af
            return af

        def _eval_or_grad(u_vec: np.ndarray, need_grad: bool):
            u_flat = np.asarray(u_vec, dtype=np.float64).ravel()
            if (
                cache["u"] is not None
                and np.array_equal(cache["u"], u_flat)
                and (not need_grad or cache["res"].sensitivity.valid)
            ):
                return cache["geom"], cache["cond"], cache["res"], cache["kulfans"]

            off = 0
            c_vals = u_flat[off : off + n_c]
            off += n_c
            t_vals = u_flat[off : off + n_t]
            off += n_t
            if use_ctrl_r:
                c_r = u_flat[off : off + n_c]
                off += n_c
                t_r = u_flat[off : off + n_t]
                off += n_t
            else:
                c_r = base_g.chord_ctrl_r_norm
                t_r = base_g.twist_ctrl_r_norm

            cst_flat = u_flat[off : off + m_sec * n_cst]
            off += m_sec * n_cst
            u_cond = u_flat[off : off + 7]

            cc = fb.BSplineCurve(c_r, c_vals, int(base_g.chord_curve.degree))
            tc = fb.BSplineCurve(t_r, t_vals, int(base_g.twist_curve.degree))
            kulfans = [
                base_kulfans[m].with_vector(cst_flat[m * n_cst : (m + 1) * n_cst], include_te=include_te)
                for m in range(m_sec)
            ]
            sec_afs = [_make_airfoil3d(kaf) for kaf in kulfans]
            geom_i = fb.RotorGeometry(
                base_g.num_blades,
                base_g.radius,
                base_g.hub_radius,
                base_g.n_radial,
                cc,
                tc,
                sec_afs,
                sec_r_norm,
                base_g.spin_dir,
            )
            cond = fb.OperatingCondition(
                vx=float(u_cond[2]),
                vy=float(u_cond[3]),
                vz=float(u_cond[4]),
                omega=float(u_cond[0]),
                pitch_0=float(u_cond[1]),
                pitch_1c=float(u_cond[5]),
                pitch_1s=float(u_cond[6]),
                rho=base_c.rho,
                mu=base_c.mu,
                a_sound=base_c.a_sound,
            )
            res = fb.BEMTSolver(geom_i, cfg).solve(cond, bool(need_grad))
            cache["u"] = u_flat.copy()
            cache["geom"] = geom_i
            cache["cond"] = cond
            cache["res"] = res
            cache["kulfans"] = kulfans
            return geom_i, cond, res, kulfans

        def _fw(u_vec: np.ndarray) -> np.ndarray:
            _, _, r, _ = _eval_or_grad(u_vec, False)
            base_out = np.array(
                [r.thrust, r.torque, r.power, r.fx, r.fy, r.mx, r.my, r.root_flap_moment],
                dtype=np.float64,
            )
            if not expose_sec:
                return base_out
            alpha_mean = np.mean(np.asarray(r.alpha, dtype=np.float64), axis=1)
            re_mean = np.mean(np.asarray(r.re, dtype=np.float64), axis=1)
            return np.concatenate([base_out, S_sec @ alpha_mean, S_sec @ re_mean])

        def _jac(u_vec: np.ndarray) -> np.ndarray:
            geom_i, _, res, kulfans = _eval_or_grad(u_vec, True)
            s = res.sensitivity
            geom_len = (2 * n_c + 2 * n_t) if use_ctrl_r else (n_c + n_t)
            cst_len = m_sec * n_cst
            J = np.zeros((out_dim, geom_len + cst_len + 7), dtype=np.float64)

            # 1. Chord & twist control points
            J[0, :n_c] = s.dT_dchord_ctrl
            J[1, :n_c] = s.dQ_dchord_ctrl
            J[2, :n_c] = s.dP_dchord_ctrl
            J[3, :n_c] = s.dFx_dchord_ctrl
            J[4, :n_c] = s.dFy_dchord_ctrl
            J[5, :n_c] = s.dMx_dchord_ctrl
            J[6, :n_c] = s.dMy_dchord_ctrl
            J[7, :n_c] = s.dMflap_dchord_ctrl

            J[0, n_c : n_c + n_t] = s.dT_dtwist_ctrl
            J[1, n_c : n_c + n_t] = s.dQ_dtwist_ctrl
            J[2, n_c : n_c + n_t] = s.dP_dtwist_ctrl
            J[3, n_c : n_c + n_t] = s.dFx_dtwist_ctrl
            J[4, n_c : n_c + n_t] = s.dFy_dtwist_ctrl
            J[5, n_c : n_c + n_t] = s.dMx_dtwist_ctrl
            J[6, n_c : n_c + n_t] = s.dMy_dtwist_ctrl
            J[7, n_c : n_c + n_t] = s.dMflap_dtwist_ctrl

            if use_ctrl_r:
                oc = n_c + n_t
                ot = oc + n_c
                J[0, oc : oc + n_c] = s.dT_dchord_ctrl_r
                J[1, oc : oc + n_c] = s.dQ_dchord_ctrl_r
                J[2, oc : oc + n_c] = s.dP_dchord_ctrl_r
                J[3, oc : oc + n_c] = s.dFx_dchord_ctrl_r
                J[4, oc : oc + n_c] = s.dFy_dchord_ctrl_r
                J[5, oc : oc + n_c] = s.dMx_dchord_ctrl_r
                J[6, oc : oc + n_c] = s.dMy_dchord_ctrl_r
                J[7, oc : oc + n_c] = s.dMflap_dchord_ctrl_r
                J[0, ot : ot + n_t] = s.dT_dtwist_ctrl_r
                J[1, ot : ot + n_t] = s.dQ_dtwist_ctrl_r
                J[2, ot : ot + n_t] = s.dP_dtwist_ctrl_r
                J[3, ot : ot + n_t] = s.dFx_dtwist_ctrl_r
                J[4, ot : ot + n_t] = s.dFy_dtwist_ctrl_r
                J[5, ot : ot + n_t] = s.dMx_dtwist_ctrl_r
                J[6, ot : ot + n_t] = s.dMy_dtwist_ctrl_r
                J[7, ot : ot + n_t] = s.dMflap_dtwist_ctrl_r

            if expose_sec:
                if not hasattr(s, "dalpha_dchord_ctrl"):
                    raise RuntimeError(
                        "FastBEMTAdapter: expose_section_aero=True requires a fast_bemt build "
                        "exposing BEMTSensitivity.dalpha_* and dre_*."
                    )
                r_a0, r_a1 = 8, 8 + m_sec
                r_r0, r_r1 = 8 + m_sec, 8 + 2 * m_sec
                J[r_a0:r_a1, :n_c] = S_sec @ np.asarray(s.dalpha_dchord_ctrl, dtype=np.float64)
                J[r_a0:r_a1, n_c : n_c + n_t] = S_sec @ np.asarray(s.dalpha_dtwist_ctrl, dtype=np.float64)
                J[r_r0:r_r1, :n_c] = S_sec @ np.asarray(s.dre_dchord_ctrl, dtype=np.float64)
                if use_ctrl_r:
                    oc = n_c + n_t
                    ot = oc + n_c
                    J[r_a0:r_a1, oc : oc + n_c] = S_sec @ np.asarray(s.dalpha_dchord_ctrl_r, dtype=np.float64)
                    J[r_a0:r_a1, ot : ot + n_t] = S_sec @ np.asarray(s.dalpha_dtwist_ctrl_r, dtype=np.float64)
                    J[r_r0:r_r1, oc : oc + n_c] = S_sec @ np.asarray(s.dre_dchord_ctrl_r, dtype=np.float64)
                c_off_sec = geom_len + cst_len
                J[r_a0:r_a1, c_off_sec : c_off_sec + 7] = S_sec @ np.asarray(s.dalpha_dCond, dtype=np.float64)
                J[r_r0:r_r1, c_off_sec : c_off_sec + 7] = S_sec @ np.asarray(s.dre_dCond, dtype=np.float64)

            # 2. 3-level CST chain rule: dJ/dCST_m = sum_{i,j} W_{i,m} * (dJ/dCl_{i,j} dCl/dCST_i + dJ/dCd_{i,j} dCd/dCST_i)
            d_cl_list = [s.dT_dcl, s.dQ_dcl, s.dP_dcl, s.dFx_dcl, s.dFy_dcl, s.dMx_dcl, s.dMy_dcl, s.dMflap_dcl]
            d_cd_list = [s.dT_dcd, s.dQ_dcd, s.dP_dcd, s.dFx_dcd, s.dFy_dcd, s.dMx_dcd, s.dMy_dcd, s.dMflap_dcd]
            nr_e, npsi_e = res.alpha.shape

            if use_native_nf:
                # Exact CST-space chain rule matching Airfoil3D::interpolate_section:
                # each radial station i evaluates NeuralFoil at CST_i = sum_m W_{i,m} * CST_m.
                sec_cst_full = np.vstack([k.to_vector(include_te=True) for k in kulfans])  # (m_sec, 18)
                elem_cst = W_sec @ sec_cst_full  # (nr_e, 18): [lower(8), upper(8), le(1), te(1)]
                alpha_deg_2d = np.rad2deg(res.alpha)
                re_2d = np.maximum(res.re, 100.0)
                ma_2d = np.asarray(res.ma, dtype=np.float64)
                d_cl_stack = np.stack(d_cl_list, axis=0)  # (8, nr_e, npsi_e)
                d_cd_stack = np.stack(d_cd_list, axis=0)  # (8, nr_e, npsi_e)
                da_dcl = np.asarray(s.dalpha_dcl, dtype=np.float64) if expose_sec else None
                da_dcd = np.asarray(s.dalpha_dcd, dtype=np.float64) if expose_sec else None

                for i in range(nr_e):
                    dcl_i, dcd_i = nf_bemt.eval_kulfan_cst_jac_batch(
                        elem_cst[i, 8:16],
                        elem_cst[i, 0:8],
                        float(elem_cst[i, 16]),
                        float(elem_cst[i, 17]),
                        alpha_deg_2d[i, :],
                        re_2d[i, :],
                        ma_2d[i, :],
                        enable_compressibility=nf_comp,
                        include_te=include_te,
                        n_crit=nf_n_crit,
                        xtr_upper=nf_xtr_u,
                        xtr_lower=nf_xtr_l,
                    )
                    # (8, npsi_e) @ (npsi_e, n_cst) -> (8, n_cst)
                    g_elem = d_cl_stack[:, i, :] @ dcl_i + d_cd_stack[:, i, :] @ dcd_i
                    g_alpha_i = (da_dcl[i, :] @ dcl_i + da_dcd[i, :] @ dcd_i) if expose_sec else None
                    for m in range(m_sec):
                        w_im = float(W_sec[i, m])
                        if w_im > 1e-12:
                            col_base = geom_len + m * n_cst
                            J[:8, col_base : col_base + n_cst] += w_im * g_elem
                            if expose_sec:
                                for sec_idx in range(m_sec):
                                    s_w = float(S_sec[sec_idx, i])
                                    if s_w > 1e-12:
                                        J[8 + sec_idx, col_base : col_base + n_cst] += (s_w * w_im) * g_alpha_i

                # 3. Operating condition sensitivities [omega, pitch_0, vx, vy, vz, pitch_1c, pitch_1s]
                c_off = geom_len + cst_len
                J[:7, c_off : c_off + 7] = s.dForces_dCond
                J[7, c_off : c_off + 7] = s.dMflap_dCond
                return J

            alpha_deg_elem = np.clip(np.rad2deg(res.alpha).ravel(), alpha_deg_grid[0], alpha_deg_grid[-1])
            re_elem = np.clip(res.re.ravel(), re_grid[0], re_grid[-1])
            ma_elem = np.clip(res.ma.ravel(), ma_grid[0], ma_grid[-1])
            eps_cst = 1e-5
            for m in range(m_sec):
                w_m = W_sec[:, m : m + 1]  # (nr_e, 1)
                c_vec = kulfans[m].to_vector(include_te=include_te)
                # Compute weighted sensitivities at local operating points
                dcl_dk = np.zeros((8, nr_e * npsi_e), dtype=np.float64)
                dcd_dk = np.zeros((8, nr_e * npsi_e), dtype=np.float64)
                for row_idx in range(8):
                    dcl_dk[row_idx] = (w_m * d_cl_list[row_idx]).ravel()
                    dcd_dk[row_idx] = (w_m * d_cd_list[row_idx]).ravel()

                col_base = geom_len + m * n_cst
                used_fast_batch_jac = False
                if nf_bemt is not None and not self._custom_polar_eval:
                    # Hybrid C++ Fast-Path: exact OpenMP analytical reverse-mode NeuralFoil backprop
                    # evaluated ONLY on active blade elements where W_sec[:, m] > 0!
                    w_flat = np.broadcast_to(w_m, (nr_e, npsi_e)).ravel()
                    act_mask = w_flat > 1e-12
                    if np.any(act_mask):
                        kaf_m = kulfans[m]
                        dcl_dcst_act, dcd_dcst_act = nf_bemt.eval_kulfan_cst_jac_batch(
                            kaf_m.upper_weights,
                            kaf_m.lower_weights,
                            float(kaf_m.leading_edge_weight),
                            float(kaf_m.TE_thickness),
                            alpha_deg_elem[act_mask],
                            re_elem[act_mask],
                            ma_elem[act_mask],
                            enable_compressibility=nf_comp,
                            include_te=include_te,
                            n_crit=nf_n_crit,
                            xtr_upper=nf_xtr_u,
                            xtr_lower=nf_xtr_l,
                        )
                        J[:8, col_base : col_base + n_cst] = (
                            dcl_dk[:, act_mask] @ dcl_dcst_act + dcd_dk[:, act_mask] @ dcd_dcst_act
                        )
                    used_fast_batch_jac = True
                elif polar_fn is _default_kulfan_polar_eval and hasattr(kulfans[m], "eval_aero_and_cst_jacobian"):
                    try:
                        import aerosandbox  # noqa: F401
                        _, _, dcl_dcst_m, dcd_dcst_m = kulfans[m].eval_aero_and_cst_jacobian(
                            alpha_deg=alpha_deg_elem,
                            Re=re_elem,
                            mach=ma_elem,
                            include_te=include_te,
                            fd_step=eps_cst,
                        )
                        J[:8, col_base : col_base + n_cst] = dcl_dk @ dcl_dcst_m + dcd_dk @ dcd_dcst_m
                        used_fast_batch_jac = True
                    except ImportError:
                        pass

                if not used_fast_batch_jac:
                    for p_idx in range(n_cst):
                        vp = c_vec.copy()
                        vm = c_vec.copy()
                        vp[p_idx] += eps_cst
                        vm[p_idx] -= eps_cst
                        kp = kulfans[m].with_vector(vp, include_te=include_te)
                        km = kulfans[m].with_vector(vm, include_te=include_te)
                        if not self._custom_polar_eval:
                            clp, cdp = _default_kulfan_polar_eval(
                                kp, alpha_deg_elem, re_elem, ma_elem,
                                n_crit=nf_n_crit, xtr_upper=nf_xtr_u, xtr_lower=nf_xtr_l,
                            )
                            clm, cdm = _default_kulfan_polar_eval(
                                km, alpha_deg_elem, re_elem, ma_elem,
                                n_crit=nf_n_crit, xtr_upper=nf_xtr_u, xtr_lower=nf_xtr_l,
                            )
                        else:
                            clp, cdp = polar_fn(kp, alpha_deg_elem, re_elem, ma_elem)
                            clm, cdm = polar_fn(km, alpha_deg_elem, re_elem, ma_elem)
                        dcl_dp = (clp - clm) / (2.0 * eps_cst)
                        dcd_dp = (cdp - cdm) / (2.0 * eps_cst)
                        J[:8, col_base + p_idx] = dcl_dk @ dcl_dp + dcd_dk @ dcd_dp

            # 3. 7-column condition Jacobian
            cond_start = geom_len + cst_len
            J[:7, cond_start : cond_start + 7] = s.dForces_dCond
            J[7, cond_start : cond_start + 7] = s.dMflap_dCond
            return J

        return CustomOp(
            forward_fn=_fw,
            shape=(out_dim, 1),
            jacobian_fn=_jac,
            fd_step=self._fd_step,
            fd_hessian=self._fd_hessian,
            name=f"{self._name}_cst_blade",
        )

    def _pack_condition_vector(
        self,
        g: Any,
        *,
        base_cond: Any,
        omega: Optional[Union[Expr, ArrayLike]] = None,
        pitch_0: Optional[Union[Expr, ArrayLike]] = None,
        pitch_1c: Optional[Union[Expr, ArrayLike]] = None,
        pitch_1s: Optional[Union[Expr, ArrayLike]] = None,
        vx: Optional[Union[Expr, ArrayLike]] = None,
        vy: Optional[Union[Expr, ArrayLike]] = None,
        vz: Optional[Union[Expr, ArrayLike]] = None,
        v_inf: Optional[Union[Expr, ArrayLike]] = None,
        alpha_disk_rad: Optional[Union[Expr, ArrayLike]] = None,
        beta_rad: Optional[Union[Expr, ArrayLike]] = None,
        include_pitch_0: bool = True,
    ) -> Tuple[Expr, Expr, Optional[Expr]]:
        omega_e = _coerce_with_graph(g, base_cond.omega if omega is None else omega)
        pitch_1c_e = _coerce_with_graph(g, base_cond.pitch_1c if pitch_1c is None else pitch_1c)
        pitch_1s_e = _coerce_with_graph(g, base_cond.pitch_1s if pitch_1s is None else pitch_1s)

        v_inf_e: Optional[Expr] = None
        if v_inf is not None or alpha_disk_rad is not None or beta_rad is not None:
            v_inf_default = float(np.sqrt(base_cond.vx ** 2 + base_cond.vy ** 2 + base_cond.vz ** 2))
            v_inf_e = _coerce_with_graph(g, v_inf_default if v_inf is None else v_inf)
            alpha_e = _coerce_with_graph(g, 0.0 if alpha_disk_rad is None else alpha_disk_rad)
            beta_e = _coerce_with_graph(g, 0.0 if beta_rad is None else beta_rad)

            v_inplane = v_inf_e * sin(alpha_e)
            vx_e = v_inplane * cos(beta_e)
            vy_e = v_inplane * sin(beta_e)
            vz_e = v_inf_e * cos(alpha_e)
        else:
            vx_e = _coerce_with_graph(g, base_cond.vx if vx is None else vx)
            vy_e = _coerce_with_graph(g, base_cond.vy if vy is None else vy)
            vz_e = _coerce_with_graph(g, base_cond.vz if vz is None else vz)
            v_inf_e = sqrt(vx_e * vx_e + vy_e * vy_e + vz_e * vz_e + 1e-12)

        if include_pitch_0:
            pitch_e = _coerce_with_graph(g, base_cond.pitch_0 if pitch_0 is None else pitch_0)
            cond_vec = vstack([omega_e, pitch_e, vx_e, vy_e, vz_e, pitch_1c_e, pitch_1s_e])
        else:
            cond_vec = vstack([omega_e, vx_e, vy_e, vz_e, pitch_1c_e, pitch_1s_e])
        return cond_vec, omega_e, v_inf_e

    def __call__(
        self,
        *,
        chord_ctrl: Optional[Union[Expr, ArrayLike]] = None,
        twist_ctrl: Optional[Union[Expr, ArrayLike]] = None,
        chord_ctrl_r: Optional[Union[Expr, ArrayLike]] = None,
        twist_ctrl_r: Optional[Union[Expr, ArrayLike]] = None,
        cst_params: Optional[Union[Expr, ArrayLike]] = None,
        omega: Optional[Union[Expr, ArrayLike]] = None,
        pitch_0: Optional[Union[Expr, ArrayLike]] = None,
        pitch_1c: Optional[Union[Expr, ArrayLike]] = None,
        pitch_1s: Optional[Union[Expr, ArrayLike]] = None,
        vx: Optional[Union[Expr, ArrayLike]] = None,
        vy: Optional[Union[Expr, ArrayLike]] = None,
        vz: Optional[Union[Expr, ArrayLike]] = None,
        v_inf: Optional[Union[Expr, ArrayLike]] = None,
        alpha_disk_rad: Optional[Union[Expr, ArrayLike]] = None,
        beta_rad: Optional[Union[Expr, ArrayLike]] = None,
        auto_trim_thrust: Optional[float] = None,
        pitch_lb: float = -0.35,
        pitch_ub: float = 0.65,
        include_te: bool = False,
    ) -> BEMTSymbolicOutput:
        """
        Bind symbolic `fastopt.Expr` (or numeric) inputs to `fast_bemt` and return a `BEMTSymbolicOutput`.
        """
        all_args = [
            chord_ctrl, twist_ctrl, chord_ctrl_r, twist_ctrl_r, cst_params,
            omega, pitch_0, pitch_1c, pitch_1s, vx, vy, vz, v_inf, alpha_disk_rad, beta_rad,
        ]
        g = _find_graph(*all_args)
        use_ctrl_r = (chord_ctrl_r is not None) or (twist_ctrl_r is not None)
        use_blade_mode = (
            (chord_ctrl is not None) or (twist_ctrl is not None) or use_ctrl_r
            or (cst_params is not None) or (auto_trim_thrust is not None)
        )

        cond_vec, omega_e, v_inf_e = self._pack_condition_vector(
            g,
            base_cond=self._base_cond,
            omega=omega,
            pitch_0=pitch_0,
            pitch_1c=pitch_1c,
            pitch_1s=pitch_1s,
            vx=vx,
            vy=vy,
            vz=vz,
            v_inf=v_inf,
            alpha_disk_rad=alpha_disk_rad,
            beta_rad=beta_rad,
            include_pitch_0=(auto_trim_thrust is None),
        )

        c_expr_opt: Optional[Expr] = None
        t_expr_opt: Optional[Expr] = None
        trimmed_p0: Optional[Expr] = None
        n_sec_aero = 0

        if use_blade_mode:
            c_expr_opt = _coerce_with_graph(g, self._geom.chord_ctrl if chord_ctrl is None else chord_ctrl)
            t_expr_opt = _coerce_with_graph(g, self._geom.twist_ctrl if twist_ctrl is None else twist_ctrl)
            parts: List[Expr] = [c_expr_opt, t_expr_opt]
            if use_ctrl_r:
                cr_e = _coerce_with_graph(
                    g, self._geom.chord_ctrl_r_norm if chord_ctrl_r is None else chord_ctrl_r
                )
                tr_e = _coerce_with_graph(
                    g, self._geom.twist_ctrl_r_norm if twist_ctrl_r is None else twist_ctrl_r
                )
                parts.extend([cr_e, tr_e])

            if cst_params is not None:
                if auto_trim_thrust is not None:
                    raise NotImplementedError(
                        "FastBEMTAdapter: combining `cst_params` with `auto_trim_thrust` is not supported; "
                        "pass `pitch_0` as an explicit decision variable instead."
                    )
                cst_e = _coerce_with_graph(g, cst_params)
                m_sec, n_cst = cst_e.shape
                if n_cst not in (17, 18):
                    cst_flat = cst_e.reshape(cst_e.size, 1)
                    n_cst = 18 if include_te else 17
                    if cst_e.size % n_cst != 0:
                        raise ValueError(f"FastBEMTAdapter: cst_params size ({cst_e.size}) is not divisible by {n_cst}.")
                    m_sec = cst_e.size // n_cst
                else:
                    include_te = (n_cst == 18)
                    # Row-major flatten so section m occupies [m*n_cst : (m+1)*n_cst]
                    cst_flat = vstack([cst_e[m, :].T for m in range(m_sec)])
                parts.append(cst_flat)
                parts.append(cond_vec)
                full_in = vstack(parts)
                op_cst = self._build_cst_blade_operator(m_sec, n_cst, use_ctrl_r, include_te)
                raw_out = op_cst(full_in)
                if self._expose_section_aero:
                    n_sec_aero = m_sec
            elif auto_trim_thrust is not None:
                parts.append(cond_vec)
                full_in = vstack(parts)
                cap, holder = self._fb.create_autotrim_blade_capsule(
                    self._geom,
                    self._cfg,
                    self._base_cond,
                    float(auto_trim_thrust),
                    float(pitch_lb),
                    float(pitch_ub),
                    bool(use_ctrl_r),
                )
                op_trim = CBlackBox.from_capsule(
                    capsule=cap,
                    shape=(9, 1),
                    fd_step=self._fd_step,
                    fd_hessian=self._fd_hessian,
                    name=f"{self._name}_autotrim_c_abi",
                    keep_alive=(self._geom, holder),
                )
                raw_out = op_trim(full_in)
                trimmed_p0 = raw_out[8]
            else:
                parts.append(cond_vec)
                full_in = vstack(parts)
                op = self._get_blade_operator(use_ctrl_r=use_ctrl_r)
                raw_out = op(full_in)
        else:
            op = self._get_cond_operator()
            raw_out = op(cond_vec)

        assert isinstance(raw_out, Expr)
        return BEMTSymbolicOutput(
            raw_vec=raw_out,
            geom=self._geom,
            rho=self._base_cond.rho,
            chord_ctrl_expr=c_expr_opt,
            twist_ctrl_expr=t_expr_opt,
            omega_expr=omega_e,
            v_inf_expr=v_inf_e,
            trimmed_pitch_0=trimmed_p0,
            n_sec_aero=n_sec_aero,
        )

    def eval_multipoint(
        self,
        points: Sequence[Dict[str, Any]],
        *,
        chord_ctrl: Optional[Union[Expr, ArrayLike]] = None,
        twist_ctrl: Optional[Union[Expr, ArrayLike]] = None,
        chord_ctrl_r: Optional[Union[Expr, ArrayLike]] = None,
        twist_ctrl_r: Optional[Union[Expr, ArrayLike]] = None,
    ) -> List[BEMTSymbolicOutput]:
        """
        Evaluate K operating points simultaneously in parallel via C++ OpenMP
        `create_multipoint_blade_capsule`, sharing a single blade geometry unpack and
        passing a block-sparse Jacobian mask to `fastopt`.
        """
        if not points:
            raise ValueError("eval_multipoint requires at least one operating point dictionary.")

        all_vals: List[Any] = [chord_ctrl, twist_ctrl, chord_ctrl_r, twist_ctrl_r]
        for pt in points:
            all_vals.extend(pt.values())
        g = _find_graph(*all_vals)

        use_ctrl_r = (chord_ctrl_r is not None) or (twist_ctrl_r is not None)
        c_e = _coerce_with_graph(g, self._geom.chord_ctrl if chord_ctrl is None else chord_ctrl)
        t_e = _coerce_with_graph(g, self._geom.twist_ctrl if twist_ctrl is None else twist_ctrl)
        parts: List[Expr] = [c_e, t_e]
        if use_ctrl_r:
            parts.append(
                _coerce_with_graph(g, self._geom.chord_ctrl_r_norm if chord_ctrl_r is None else chord_ctrl_r)
            )
            parts.append(
                _coerce_with_graph(g, self._geom.twist_ctrl_r_norm if twist_ctrl_r is None else twist_ctrl_r)
            )

        cfgs: List[Any] = []
        base_conds: List[Any] = []
        omega_exprs: List[Expr] = []
        vinf_exprs: List[Optional[Expr]] = []
        for pt in points:
            cfgs.append(pt.get("cfg", self._cfg))
            bc = pt.get("base_cond", self._base_cond)
            base_conds.append(bc)
            c_vec, om_e, vi_e = self._pack_condition_vector(
                g,
                base_cond=bc,
                omega=pt.get("omega"),
                pitch_0=pt.get("pitch_0"),
                pitch_1c=pt.get("pitch_1c"),
                pitch_1s=pt.get("pitch_1s"),
                vx=pt.get("vx"),
                vy=pt.get("vy"),
                vz=pt.get("vz"),
                v_inf=pt.get("v_inf"),
                alpha_disk_rad=pt.get("alpha_disk_rad"),
                beta_rad=pt.get("beta_rad"),
                include_pitch_0=True,
            )
            parts.append(c_vec)
            omega_exprs.append(om_e)
            vinf_exprs.append(vi_e)

        full_in = vstack(parts)
        n_pts = len(points)
        n_geom = c_e.size + t_e.size + ((c_e.size + t_e.size) if use_ctrl_r else 0)
        n_total = n_geom + 7 * n_pts
        m_total = 8 * n_pts

        # Block-sparse Jacobian mask: shared geometry affects all K points; cond_k affects only block k
        jac_mask = np.zeros((m_total, n_total), dtype=np.int32)
        jac_mask[:, :n_geom] = 1
        for k in range(n_pts):
            jac_mask[8 * k : 8 * (k + 1), n_geom + 7 * k : n_geom + 7 * (k + 1)] = 1

        cap, holder = self._fb.create_multipoint_blade_capsule(
            self._geom, cfgs, base_conds, use_ctrl_r=bool(use_ctrl_r)
        )
        op = CBlackBox.from_capsule(
            capsule=cap,
            shape=(m_total, 1),
            fd_step=self._fd_step,
            fd_hessian=self._fd_hessian,
            jac_sparsity=jac_mask,
            name=f"{self._name}_multipoint_c_abi",
            keep_alive=(self._geom, holder),
        )
        raw_all = op(full_in)
        assert isinstance(raw_all, Expr)

        outputs: List[BEMTSymbolicOutput] = []
        for k in range(n_pts):
            seg = raw_all[8 * k : 8 * (k + 1)]
            outputs.append(
                BEMTSymbolicOutput(
                    raw_vec=seg,
                    geom=self._geom,
                    rho=base_conds[k].rho,
                    chord_ctrl_expr=c_e,
                    twist_ctrl_expr=t_e,
                    omega_expr=omega_exprs[k],
                    v_inf_expr=vinf_exprs[k],
                )
            )
        return outputs

    @staticmethod
    def add_planform_constraints(
        opti: Any,
        *,
        chord_ctrl: Optional[Expr] = None,
        twist_ctrl: Optional[Expr] = None,
        solidity: Optional[Expr] = None,
        max_solidity: Optional[float] = None,
        root_chord: Optional[float] = None,
        min_tip_chord: Optional[float] = None,
        chord_taper: bool = True,
        twist_washout: bool = True,
        smoothness_weight_chord: float = 0.0,
        smoothness_weight_twist: float = 0.0,
    ) -> Expr:
        """
        Register standard rotorcraft blade planform/twist constraints (`root_chord`, `min_tip_chord`,
        `max_solidity`, `chord_taper`, `twist_washout`) on `opti` and return the smoothness
        regularization `Expr`.
        """
        g = _find_graph(chord_ctrl, twist_ctrl, solidity)
        reg: Expr = _coerce_with_graph(g, 0.0)

        if chord_ctrl is not None:
            if root_chord is not None:
                opti.subject_to(chord_ctrl[0] == float(root_chord), scale=float(root_chord), name="root_chord")
            if min_tip_chord is not None:
                opti.subject_to(chord_ctrl[-1] >= float(min_tip_chord), scale=float(min_tip_chord), name="min_tip_chord")
            if chord_taper and chord_ctrl.shape[0] >= 3:
                opti.subject_to(chord_ctrl[2:] <= chord_ctrl[1:-1] + 0.005, name="chord_taper")
            if smoothness_weight_chord > 0.0 and chord_ctrl.shape[0] >= 3:
                d2_c = chord_ctrl[2:] - 2.0 * chord_ctrl[1:-1] + chord_ctrl[:-2]
                reg = reg + float(smoothness_weight_chord) * sumsqr(d2_c)

        if twist_ctrl is not None:
            if twist_washout and twist_ctrl.shape[0] >= 2:
                opti.subject_to(twist_ctrl[1:] <= twist_ctrl[:-1], name="twist_washout")
            if smoothness_weight_twist > 0.0 and twist_ctrl.shape[0] >= 3:
                d2_t = twist_ctrl[2:] - 2.0 * twist_ctrl[1:-1] + twist_ctrl[:-2]
                reg = reg + float(smoothness_weight_twist) * sumsqr(d2_t)

        if solidity is not None and max_solidity is not None:
            opti.subject_to(solidity <= float(max_solidity), scale=float(max_solidity), name="max_solidity")

        return reg

    def build_polar_surrogate_2d(
        self,
        v_inf_grid: Sequence[float],
        pitch_0_grid: Sequence[float],
        *,
        alpha_disk_rad: float = 0.0,
        omega: Optional[float] = None,
        beta_rad: float = 0.0,
        degree: int = 3,
    ) -> Dict[str, Any]:
        """
        Evaluate a 2D operating envelope `(v_inf_grid, pitch_0_grid)` in parallel via C++ OpenMP
        `Evaluator.eval_batch`, and return raw 2D tables and `BSplineInterpolator2D` surrogates.
        """
        vg = np.asarray(v_inf_grid, dtype=np.float64).ravel()
        pg = np.asarray(pitch_0_grid, dtype=np.float64).ravel()
        nv, np_pitch = vg.size, pg.size
        om_val = float(self._base_cond.omega if omega is None else omega)

        gb = GraphBuilder()
        v_var = gb.variable(init=float(vg[0]), name="v_inf")
        p_var = gb.variable(init=float(pg[0]), name="pitch_0")
        aero = self(v_inf=v_var, pitch_0=p_var, alpha_disk_rad=float(alpha_disk_rad), omega=om_val, beta_rad=float(beta_rad))
        ev = gb.build_evaluator(aero.raw_vec)

        VV, PP = np.meshgrid(vg, pg, indexing="ij")
        X_pop = np.column_stack([VV.ravel(), PP.ravel()])
        Y_batch = ev.eval_batch(X_pop, num_threads=0)[0]

        names = ["thrust", "torque", "power", "fx", "fy", "mx", "my", "root_flap_moment"]
        tables: Dict[str, np.ndarray] = {}
        splines: Dict[str, BSplineInterpolator2D] = {}
        for idx, nm in enumerate(names):
            tbl = Y_batch[:, idx].reshape((nv, np_pitch))
            tables[nm] = tbl
            splines[nm] = BSplineInterpolator2D(vg, pg, tbl, degree_x=degree, degree_y=degree)

        return {
            "v_inf_grid": vg,
            "pitch_0_grid": pg,
            "tables": tables,
            "splines": splines,
        }
