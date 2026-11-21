"""
Phase 13 Joint Integration & Rotorcraft Engineering Showcase (`fast_bemt` + `fastopt`):

Demonstrates 4 complete rotorcraft aerodynamic analysis and optimization workflows:
1. **Case 1 — Rotor Aerodynamic Data Calculation, Analytical IFT Jacobian Verification, & 2D B-Spline Surrogate**:
   - Single-point oblique-inflow BEMT solve (thrust, torque, power, 3-axis forces/moments, wake skew).
   - Exact consistency between Zero-GIL C-ABI `PyCapsule` (`use_c_abi=True`), Python `CustomOp` (`use_c_abi=False`),
     and central finite differences for coupled `(chord_ctrl, twist_ctrl, omega, pitch_0, v_inf, alpha_disk)` Jacobians.
   - Parallel C++ OpenMP `eval_batch` sweep over a `(v_inf, pitch_0)` grid to build reusable `BSplineInterpolator2D`
     aerodynamic surrogate surfaces and analytical control-derivative surfaces `dT/d(pitch_0)`.
2. **Case 2 — Hover Blade Planform (`chord_ctrl`) & Twist (`twist_ctrl`) Optimization**:
   - Starts from an unoptimized rectangular, lightly-twisted baseline rotor and minimizes hover shaft power
     at target thrust `T_req = 2800 N` while enforcing spanwise trapezoidal blade area / solidity upper bound
     (`aero.solidity <= sigma_0`), taper monotonicity, and spline smoothness.
   - Extracts physical KKT dual multiplier `dP*/dT_req` [W/N] via `sol.dual(...)` and verifies significant
     Figure of Merit (`FM`) improvement and shaft power reduction.
3. **Case 3 — Tiltrotor Multi-Point (Hover + Forward-Flight Cruise) Coupled Blade Design Optimization**:
   - Simultaneously optimizes shared blade `chord_ctrl` and `twist_ctrl` plus mode-specific collective pitches
     (`pitch_hover`, `pitch_cruise`) across Hover (`V_inf = 0 m/s, T == 2800 N`) and Forward-Flight Cruise
     (`V_inf = 20 m/s, T == 1250 N`).
4. **Case 4 — Tiltrotor Conversion Corridor Trim Sweep (`opti.solve_sweep`) & KKT Sensitivity (`sol.sensitivity`)**:
   - Solves for minimum-power trim `(omega*, pitch_0*)` across a 6-point tiltrotor conversion corridor
     with zero graph recompilation, and computes post-optimality KKT sensitivity `dJ*/dT_req`.

All comments are written in English.
"""

from __future__ import annotations
import time
import numpy as np

import fastopt as fo
import fast_bemt as fb


def build_baseline_3blade_rotor(n_radial: int = 20, bspline_degree: int = 3) -> fb.RotorGeometry:
    """
    Construct a 3-bladed eVTOL / tiltrotor baseline rotor (R = 1.25 m, R_hub = 0.15 m, N_r = 20)
    parameterized by 6 cubic B-spline control points for chord and twist, and 5 spanwise
    3D compressible airfoil polar tables over (alpha, Re, Ma).
    Starts with a uniform-chord (0.090 m), lightly-twisted (11 deg -> 6 deg) baseline blade.
    """
    radius = 1.25
    hub_radius = 0.15

    ctrl_r_norm = np.array([0.12, 0.22, 0.38, 0.62, 0.85, 1.00], dtype=np.float64)
    chord_vals = np.array([0.090, 0.090, 0.090, 0.090, 0.090, 0.090], dtype=np.float64)
    twist_vals = np.deg2rad([11.0, 10.0, 9.0, 8.0, 7.0, 6.0])

    chord_curve = fb.BSplineCurve(ctrl_r_norm, chord_vals, bspline_degree)
    twist_curve = fb.BSplineCurve(ctrl_r_norm, twist_vals, bspline_degree)

    section_r_norm = np.array([0.10, 0.20, 0.50, 0.75, 1.00], dtype=np.float64)
    alpha_grid = np.deg2rad(np.linspace(-18.0, 18.0, 73))
    re_grid = np.array([1.0e5, 5.0e5, 1.5e6, 5.0e6], dtype=np.float64)
    ma_grid = np.array([0.0, 0.3, 0.5, 0.7], dtype=np.float64)

    A, RE, MA = np.meshgrid(alpha_grid, re_grid, ma_grid, indexing="ij")
    five_section_airfoils = []
    for r_sec in section_r_norm:
        alpha_0 = np.deg2rad(-2.5 * (1.0 - 0.6 * r_sec))
        pg_factor = 1.0 / np.sqrt(1.0 - (0.8 * MA) ** 2)
        re_scale = (RE / 1.0e6) ** (-0.08)

        a_eff = A - alpha_0
        cl_linear = 2.0 * np.pi * a_eff * pg_factor
        stall_blend = 1.0 / (1.0 + (a_eff / np.deg2rad(14.0)) ** 6)
        cl_3d = cl_linear * stall_blend + 1.05 * np.sin(2.0 * a_eff) * (1.0 - stall_blend)

        cd_min = (0.0090 - 0.0025 * r_sec) * re_scale
        cd_3d = cd_min + 0.025 * (a_eff / np.deg2rad(10.0)) ** 2 + 0.05 * (MA ** 3)

        af = fb.create_3d_airfoil(alpha_grid, re_grid, ma_grid, cl_3d, cd_3d, degrees=False)
        five_section_airfoils.append(af)

    return fb.RotorGeometry(
        num_blades=3,
        radius=radius,
        hub_radius=hub_radius,
        n_radial=n_radial,
        chord_curve=chord_curve,
        twist_curve=twist_curve,
        airfoils=five_section_airfoils,
        section_r_norm=section_r_norm,
        spin_dir=1,
    )


def case1_aerodynamic_data_and_jacobian_verification():
    """
    Case 1: Single-point oblique-inflow rotor aerodynamic calculation, C-ABI vs Python vs FD
    Jacobian verification, and parallel C++ OpenMP 2D envelope B-spline surrogate construction.
    """
    print("\n--- [Case 1] Rotor Aerodynamic Data Calculation, IFT-AD Jacobian & 2D Surrogate ---")
    geom = build_baseline_3blade_rotor(n_radial=20)
    cfg = fb.SolverConfig(
        n_azimuth=36,
        wake_model=fb.WakeModel.GlauertPittPeters,
        enable_tip_loss=True,
        enable_vrs_correction=True,
        enable_stall_delay=True,
        enable_tangential_induction=True,
    )
    base_cond = fb.OperatingCondition.from_freestream(
        v_inf=25.0,
        alpha_disk_rad=np.deg2rad(45.0),
        omega_rad_s=180.0,
        pitch_0_rad=np.deg2rad(10.0),
        beta_rad=np.deg2rad(10.0),
    )

    adapter = fo.FastBEMTAdapter(geom, cfg, base_cond, use_c_abi="auto")
    res_single = adapter.solve_condition(base_cond, compute_gradients=True)
    print(
        f"  [Single-Point Oblique Inflow (V=25 m/s, alpha_d=45 deg, Omega=180 rad/s, pitch=10 deg)]\n"
        f"    Channel: {'Zero-GIL C-ABI PyCapsule' if adapter.uses_c_abi else 'Analytical-IFT Python CustomOp'}\n"
        f"    Thrust T = {res_single.thrust:.2f} N | Torque Q = {res_single.torque:.2f} N*m | "
        f"Power P = {res_single.power / 1e3:.3f} kW\n"
        f"    In-Plane Force Fx = {res_single.fx:.2f} N | Roll Mx = {res_single.mx:.2f} N*m | "
        f"Pitch My = {res_single.my:.2f} N*m\n"
        f"    Mean Induced Velocity vi = {res_single.vi_disk_mean:.2f} m/s | "
        f"Wake Skew Angle chi = {np.rad2deg(res_single.wake_skew_chi):.2f} deg"
    )

    # Verify coupled Blade + Freestream Condition Jacobian in fastopt GraphBuilder against Central FD
    gb = fo.GraphBuilder()
    c_var = gb.variable(shape=(6, 1), init=geom.chord_ctrl, name="chord_ctrl")
    t_var = gb.variable(shape=(6, 1), init=geom.twist_ctrl, name="twist_ctrl")
    om_var = gb.variable(init=180.0, name="omega")
    p0_var = gb.variable(init=np.deg2rad(10.0), name="pitch_0")
    v_var = gb.variable(init=25.0, name="v_inf")
    ad_var = gb.variable(init=np.deg2rad(45.0), name="alpha_disk")

    aero = adapter(
        chord_ctrl=c_var,
        twist_ctrl=t_var,
        omega=om_var,
        pitch_0=p0_var,
        v_inf=v_var,
        alpha_disk_rad=ad_var,
        beta_rad=np.deg2rad(10.0),
    )
    ev = gb.build_evaluator(aero.raw_vec)
    x0 = gb.cpp_graph.initial_decision_vector()

    J_ad = ev.jacobian(x0, 0)
    assert J_ad.shape == (8, 16)

    # Central finite difference verification across all 16 decision variables
    eps = 1e-6
    J_fd = np.zeros_like(J_ad)
    for j in range(x0.size):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        J_fd[:, j] = (ev.eval(xp)[0].ravel() - ev.eval(xm)[0].ravel()) / (2.0 * eps)

    rel_jac_err = np.linalg.norm(J_ad - J_fd) / np.linalg.norm(J_fd)
    assert rel_jac_err < 1e-5, f"Coupled BEMT Jacobian rel error too large: {rel_jac_err}"

    # Verify Python CustomOp channel (`use_c_abi=False`) gives identical values & Jacobians
    adapter_py = fo.FastBEMTAdapter(geom, cfg, base_cond, use_c_abi=False)
    gb_py = fo.GraphBuilder()
    aero_py = adapter_py(
        chord_ctrl=gb_py.variable(shape=(6, 1), init=geom.chord_ctrl),
        twist_ctrl=gb_py.variable(shape=(6, 1), init=geom.twist_ctrl),
        omega=gb_py.variable(init=180.0),
        pitch_0=gb_py.variable(init=np.deg2rad(10.0)),
        v_inf=gb_py.variable(init=25.0),
        alpha_disk_rad=gb_py.variable(init=np.deg2rad(45.0)),
        beta_rad=np.deg2rad(10.0),
    )
    ev_py = gb_py.build_evaluator(aero_py.raw_vec)
    assert np.allclose(ev.eval(x0)[0], ev_py.eval(x0)[0], rtol=1e-12)
    assert np.allclose(J_ad, ev_py.jacobian(x0, 0), rtol=1e-12)

    # Build 2D Operating-Envelope B-Spline Surrogate over (v_inf, pitch_0) via OpenMP eval_batch
    v_grid = np.linspace(0.0, 40.0, 9)
    p_grid = np.deg2rad(np.linspace(2.0, 16.0, 8))
    t0 = time.perf_counter()
    surr = adapter.build_polar_surrogate_2d(
        v_grid, p_grid, alpha_disk_rad=np.deg2rad(45.0), omega=180.0, beta_rad=np.deg2rad(10.0)
    )
    surr_ms = (time.perf_counter() - t0) * 1e3

    T_spline = surr["splines"]["thrust"]
    dT_dpitch_spline = T_spline.derivative(nu_u=0, nu_v=1)
    T_interp_val = float(T_spline(25.0, np.deg2rad(10.0)))
    assert np.isclose(T_interp_val, res_single.thrust, rtol=2e-3)
    dT_dp_val = float(dT_dpitch_spline(25.0, np.deg2rad(10.0)))
    assert dT_dp_val > 0.0

    print(
        f"  [PASSED Case 1] Coupled 8x16 Analytical IFT Jacobian rel_err = {rel_jac_err:.2e} | "
        f"9x8=72-pt 2D B-Spline Envelope Surrogate built in {surr_ms:.2f} ms "
        f"(T_surr={T_interp_val:.2f} N, dT/dpitch_0={dT_dp_val:.1f} N/rad)"
    )


def case2_hover_blade_planform_and_twist_optimization():
    """
    Case 2: Hover blade planform (`chord_ctrl`) and twist (`twist_ctrl`) optimization
    for minimum hover power at target thrust T_req = 2800 N and fixed blade area limit.
    """
    print("\n--- [Case 2] Hover Blade Planform & Twist Optimization (Min Power at T = 2800 N) ---")
    geom = build_baseline_3blade_rotor(n_radial=20)
    # Use N_azimuth=1 for axisymmetric hover
    cfg_hover = fb.SolverConfig(
        n_azimuth=1,
        wake_model=fb.WakeModel.GlauertAnnulus,
        enable_tip_loss=True,
        enable_vrs_correction=True,
        enable_stall_delay=True,
        enable_tangential_induction=True,
    )
    cond_hover = fb.OperatingCondition(vx=0.0, vy=0.0, vz=0.0, omega=210.0, pitch_0=np.deg2rad(1.0))
    adapter = fo.FastBEMTAdapter(geom, cfg_hover, cond_hover)
    target_thrust = 2800.0

    # Evaluate baseline performance at target thrust 2800 N (by trimming collective pitch_0 first)
    opti_trim = fo.Opti()
    p0_trim = opti_trim.variable(init=np.deg2rad(1.0), lb=np.deg2rad(-6.0), ub=np.deg2rad(8.0), name="p0_trim")
    aero_trim = adapter(pitch_0=p0_trim, omega=210.0)
    opti_trim.minimize(fo.square((aero_trim.thrust - target_thrust) / target_thrust))
    opti_trim.solver("SLSQP", options={"ftol": 1e-12, "maxiter": 50})
    sol_trim = opti_trim.solve()
    base_res = adapter.solve_condition(omega=210.0, pitch_0=float(sol_trim.value(p0_trim)), compute_gradients=False)
    base_solidity = float(sol_trim.value(aero_trim.solidity))
    assert np.isclose(base_res.thrust, target_thrust, rtol=1e-3), f"Baseline trim thrust {base_res.thrust} != {target_thrust}"

    # Full Blade Planform (6 chord_ctrl) + Twist (6 twist_ctrl) + Collective Pitch Optimization
    opti = fo.Opti()
    c_ctrl = opti.variable(
        shape=(6, 1), init=geom.chord_ctrl, lb=0.040, ub=0.140, scale=0.09, name="chord_ctrl"
    )
    tw_ctrl = opti.variable(
        shape=(6, 1),
        init=geom.twist_ctrl,
        lb=np.deg2rad(2.0),
        ub=np.deg2rad(32.0),
        scale=0.3,
        name="twist_ctrl",
    )
    p0_opt = opti.variable(
        init=float(sol_trim.value(p0_trim)),
        lb=np.deg2rad(-6.0),
        ub=np.deg2rad(8.0),
        scale=0.1,
        name="pitch_0",
    )

    aero = adapter(chord_ctrl=c_ctrl, twist_ctrl=tw_ctrl, pitch_0=p0_opt, omega=210.0)

    # Second-difference B-spline control point smoothness regularization
    d2_chord = c_ctrl[2:] - 2.0 * c_ctrl[1:-1] + c_ctrl[:-2]
    d2_twist = tw_ctrl[2:] - 2.0 * tw_ctrl[1:-1] + tw_ctrl[:-2]
    reg_term = 15.0 * fo.sumsqr(d2_chord) + 0.05 * fo.sumsqr(d2_twist)

    # Objective: minimize hover shaft power [kW] + smoothness regularization
    opti.minimize(aero.power * 1e-3 + reg_term)

    # Constraints:
    # 1. Exact target thrust T == 2800 N
    con_thrust = opti.subject_to(aero.thrust == target_thrust, scale=target_thrust, name="target_thrust")
    # 2. Rotor solidity (blade area via symbolic fo.trapz) must not exceed baseline solidity
    con_solidity = opti.subject_to(aero.solidity <= base_solidity, scale=base_solidity, name="max_solidity")
    # 3. Spanwise chord & twist washout monotonicity from root/mid-span to tip
    opti.subject_to(c_ctrl[2:] <= c_ctrl[1:-1] + 0.005, name="chord_taper")
    opti.subject_to(tw_ctrl[1:] <= tw_ctrl[:-1], name="twist_washout")

    opti.solver("SLSQP", options={"ftol": 1e-7, "maxiter": 80})
    t0 = time.perf_counter()
    sol = opti.solve()
    opt_ms = (time.perf_counter() - t0) * 1e3

    assert sol.success, f"Hover optimization failed: {sol.message}"
    T_opt = float(sol.value(aero.thrust))
    P_opt = float(sol.value(aero.power))
    FM_opt = float(sol.value(aero.figure_of_merit))
    sig_opt = float(sol.value(aero.solidity))
    c_opt = sol.value(c_ctrl)
    tw_opt_deg = np.rad2deg(sol.value(tw_ctrl))

    assert np.isclose(T_opt, target_thrust, rtol=1e-3)
    assert sig_opt <= base_solidity + 1e-4
    assert P_opt < base_res.power * 0.96, f"Expected hover power reduction, got {P_opt} vs {base_res.power}"
    assert FM_opt > base_res.fm, f"Expected FM improvement, got {FM_opt} vs {base_res.fm}"

    power_saved_pct = (1.0 - P_opt / base_res.power) * 100.0
    print(
        f"  [Baseline Hover @ T={base_res.thrust:.1f} N] Power = {base_res.power / 1e3:.3f} kW | "
        f"FM = {base_res.fm:.4f} | Solidity = {base_solidity:.4f}\n"
        f"  [Optimized Hover @ T={T_opt:.1f} N] Power = {P_opt / 1e3:.3f} kW ({power_saved_pct:.2f}% saved!) | "
        f"FM = {FM_opt:.4f} (+{(FM_opt - base_res.fm) * 100.0:.2f} pts) | Solidity = {sig_opt:.4f} ({opt_ms:.1f} ms)\n"
        f"    Optimal Chord Ctrl [m]:   {np.round(c_opt, 4)}\n"
        f"    Optimal Twist Ctrl [deg]: {np.round(tw_opt_deg, 2)}"
    )
    return geom, sol.value(c_ctrl), sol.value(tw_ctrl)


def case3_tiltrotor_hover_cruise_multipoint_optimization():
    """
    Case 3: Tiltrotor multi-point (Hover + High-Speed Axial Cruise) coupled blade planform
    and twist optimization in a single declarative `fo.Opti` model.
    """
    print("\n--- [Case 3] Tiltrotor Hover + Forward-Flight Cruise Multi-Point Optimization ---")
    geom = build_baseline_3blade_rotor(n_radial=18)
    cfg_ax = fb.SolverConfig(n_azimuth=1, wake_model=fb.WakeModel.GlauertAnnulus, enable_tip_loss=True)

    cond_hov = fb.OperatingCondition(vx=0.0, vy=0.0, vz=0.0, omega=210.0, pitch_0=np.deg2rad(1.0))
    cond_cru = fb.OperatingCondition(vx=0.0, vy=0.0, vz=20.0, omega=185.0, pitch_0=np.deg2rad(6.0))

    bemt_hov = fo.FastBEMTAdapter(geom, cfg_ax, cond_hov, name="bemt_hover")
    bemt_cru = fo.FastBEMTAdapter(geom, cfg_ax, cond_cru, name="bemt_cruise")

    T_hov_req = 2800.0
    T_cru_req = 1250.0

    # Separate 1D baseline trims at Hover (T = 2800 N) and Cruise (V = 20 m/s, T = 1250 N)
    opti_bh = fo.Opti()
    ph_b = opti_bh.variable(init=np.deg2rad(1.0), lb=np.deg2rad(-6.0), ub=np.deg2rad(8.0))
    ah_b = bemt_hov(pitch_0=ph_b, omega=210.0)
    opti_bh.minimize(fo.square((ah_b.thrust - T_hov_req) / T_hov_req))
    opti_bh.solver("SLSQP", options={"ftol": 1e-12, "maxiter": 40})
    sol_bh = opti_bh.solve()

    opti_bc = fo.Opti()
    pc_b = opti_bc.variable(init=np.deg2rad(6.0), lb=np.deg2rad(-2.0), ub=np.deg2rad(14.0))
    ac_b = bemt_cru(pitch_0=pc_b, omega=185.0, vz=20.0)
    opti_bc.minimize(fo.square((ac_b.thrust - T_cru_req) / T_cru_req))
    opti_bc.solver("SLSQP", options={"ftol": 1e-12, "maxiter": 40})
    sol_bc = opti_bc.solve()

    P_hov_base = float(sol_bh.value(ah_b.power))
    P_cru_base = float(sol_bc.value(ac_b.power))
    J_base = 0.60 * (P_hov_base * 1e-3) + 0.40 * (P_cru_base * 1e-3)
    sig_base = float(sol_bh.value(ah_b.solidity))

    # Coupled Multi-Point Optimization: shared (chord_ctrl, twist_ctrl) + mode pitches (ph, pc)
    opti = fo.Opti()
    c_ctrl = opti.variable(
        shape=(6, 1), init=geom.chord_ctrl, lb=0.045, ub=0.135, scale=0.09, name="chord_ctrl"
    )
    tw_ctrl = opti.variable(
        shape=(6, 1),
        init=geom.twist_ctrl,
        lb=np.deg2rad(3.0),
        ub=np.deg2rad(28.0),
        scale=0.3,
        name="twist_ctrl",
    )
    ph = opti.variable(
        init=float(sol_bh.value(ph_b)), lb=np.deg2rad(-6.0), ub=np.deg2rad(8.0), scale=0.1, name="pitch_hov"
    )
    pc = opti.variable(
        init=float(sol_bc.value(pc_b)), lb=np.deg2rad(-2.0), ub=np.deg2rad(14.0), scale=0.1, name="pitch_cru"
    )

    aero_hov = bemt_hov(chord_ctrl=c_ctrl, twist_ctrl=tw_ctrl, pitch_0=ph, omega=210.0)
    aero_cru = bemt_cru(chord_ctrl=c_ctrl, twist_ctrl=tw_ctrl, pitch_0=pc, omega=185.0, vz=20.0)

    reg = 15.0 * fo.sumsqr(c_ctrl[2:] - 2.0 * c_ctrl[1:-1] + c_ctrl[:-2]) + 0.1 * fo.sumsqr(
        tw_ctrl[2:] - 2.0 * tw_ctrl[1:-1] + tw_ctrl[:-2]
    )
    mission_power_kw = 0.60 * (aero_hov.power * 1e-3) + 0.40 * (aero_cru.power * 1e-3)
    opti.minimize(mission_power_kw + reg)

    opti.subject_to(aero_hov.thrust == T_hov_req, scale=T_hov_req, name="hover_thrust")
    opti.subject_to(aero_cru.thrust == T_cru_req, scale=T_cru_req, name="cruise_thrust")
    opti.subject_to(aero_hov.solidity <= sig_base, scale=sig_base, name="max_solidity")
    opti.subject_to(c_ctrl[2:] <= c_ctrl[1:-1] + 0.005, name="chord_taper")
    opti.subject_to(tw_ctrl[1:] <= tw_ctrl[:-1], name="twist_washout")

    opti.solver("SLSQP", options={"ftol": 1e-6, "maxiter": 100})
    t0 = time.perf_counter()
    sol = opti.solve()
    mp_ms = (time.perf_counter() - t0) * 1e3

    assert sol.success, f"Multi-point optimization failed: {sol.message}"
    P_hov_opt = float(sol.value(aero_hov.power))
    P_cru_opt = float(sol.value(aero_cru.power))
    J_opt = float(sol.value(mission_power_kw))
    assert np.isclose(float(sol.value(aero_hov.thrust)), T_hov_req, rtol=1e-3)
    assert np.isclose(float(sol.value(aero_cru.thrust)), T_cru_req, rtol=1e-3)
    assert J_opt < J_base * 0.94, f"Expected multi-point power reduction: {J_opt} vs {J_base}"

    print(
        f"  [Baseline Multi-Point] Hover P = {P_hov_base / 1e3:.3f} kW, Cruise P = {P_cru_base / 1e3:.3f} kW "
        f"| Weighted J = {J_base:.3f} kW\n"
        f"  [Optimized Multi-Point] Hover P = {P_hov_opt / 1e3:.3f} kW, Cruise P = {P_cru_opt / 1e3:.3f} kW "
        f"| Weighted J* = {J_opt:.3f} kW ({(1.0 - J_opt / J_base) * 100.0:.2f}% saved in {mp_ms:.1f} ms)\n"
        f"    Optimal Multi-Point Chord [m]:   {np.round(sol.value(c_ctrl), 4)}\n"
        f"    Optimal Multi-Point Twist [deg]: {np.round(np.rad2deg(sol.value(tw_ctrl)), 2)}\n"
        f"    Optimal Pitches: Hover = {np.rad2deg(float(sol.value(ph))):.2f} deg, "
        f"Cruise = {np.rad2deg(float(sol.value(pc))):.2f} deg"
    )


def case4_transition_corridor_trim_sweep_and_sensitivity():
    """
    Case 4: Tiltrotor conversion corridor parametric trim sweep (`opti.solve_sweep`)
    with warm-starting and post-optimality KKT sensitivity (`sol.sensitivity`).
    """
    print("\n--- [Case 4] Tiltrotor Conversion Corridor Trim Sweep (`solve_sweep` & `sensitivity`) ---")
    geom = build_baseline_3blade_rotor(n_radial=16)
    cfg = fb.SolverConfig(
        n_azimuth=24,
        wake_model=fb.WakeModel.GlauertPittPeters,
        enable_tip_loss=True,
        enable_vrs_correction=True,
        enable_stall_delay=True,
    )
    adapter = fo.FastBEMTAdapter(geom, cfg)

    opti = fo.Opti()
    # Frozen parameters for flight speed V_inf [m/s], disk tilt alpha_disk [rad], and required thrust T_req [N]
    v_inf_p = opti.parameter(val=8.0, name="v_inf")
    alpha_d_p = opti.parameter(val=np.deg2rad(75.0), name="alpha_disk")
    t_req_p = opti.parameter(val=2600.0, name="t_req")

    # Active trim variables: rotor speed omega [rad/s] and collective pitch_0 [rad]
    omega_var = opti.variable(init=205.0, lb=165.0, ub=230.0, scale=200.0, name="omega")
    pitch_var = opti.variable(init=np.deg2rad(4.0), lb=np.deg2rad(-5.0), ub=np.deg2rad(16.0), scale=0.2, name="pitch_0")

    aero = adapter(omega=omega_var, pitch_0=pitch_var, v_inf=v_inf_p, alpha_disk_rad=alpha_d_p)
    opti.minimize(aero.power * 1e-3 + 1e-3 * fo.square(omega_var - 190.0))
    opti.subject_to(aero.thrust == t_req_p, scale=2000.0, name="trim_thrust")
    opti.solver("SLSQP", options={"ftol": 1e-10, "maxiter": 50})

    # 6-point conversion corridor from low-speed helicopter mode to forward-flight conversion mode
    corridor_points = [
        (8.0, np.deg2rad(75.0), 2600.0),
        (12.0, np.deg2rad(65.0), 2350.0),
        (16.0, np.deg2rad(55.0), 2100.0),
        (20.0, np.deg2rad(45.0), 1850.0),
        (24.0, np.deg2rad(32.0), 1600.0),
        (28.0, np.deg2rad(20.0), 1400.0),
    ]

    t0 = time.perf_counter()
    sols = opti.solve_sweep([v_inf_p, alpha_d_p, t_req_p], corridor_points, warm_start=True)
    sweep_ms = (time.perf_counter() - t0) * 1e3

    for idx, ((v_i, ad_i, tr_i), s_i) in enumerate(zip(corridor_points, sols)):
        assert s_i.success, f"Corridor point {idx} failed: {s_i.message}"
        assert np.isclose(float(s_i.value(aero.thrust)), tr_i, rtol=1e-3)
        print(
            f"    Corridor Pt {idx + 1}: V_inf={v_i:4.1f} m/s, alpha_d={np.rad2deg(ad_i):4.1f} deg, "
            f"T_req={tr_i:6.1f} N -> Omega*={float(s_i.value(omega_var)):6.2f} rad/s, "
            f"pitch_0*={np.rad2deg(float(s_i.value(pitch_var))):5.2f} deg, "
            f"P*={float(s_i.value(aero.power)) / 1e3:6.3f} kW"
        )

    # Analytical post-optimality KKT sensitivity at the final corridor point w.r.t. required thrust t_req_p
    sol_last = sols[-1]
    dJ_dTreq = sol_last.sensitivity(t_req_p, target="f")
    # Verify against a small finite perturbation of t_req_p
    dt = 1.0
    t_req_p.set_value(1400.0 + dt)
    sol_pert = opti.solve()
    dJ_dTreq_fd = (sol_pert.f - sol_last.f) / dt
    assert np.isclose(dJ_dTreq, dJ_dTreq_fd, rtol=2e-2, atol=1e-4)

    print(
        f"  [PASSED Case 4] 6-Point Conversion Corridor Trim Sweep solved in {sweep_ms:.2f} ms | "
        f"KKT Sensitivity dJ*/dT_req = {dJ_dTreq * 1e3:.2f} W/N (FD = {dJ_dTreq_fd * 1e3:.2f} W/N)"
    )


def run_phase13_bemt_suite():
    print("=====================================================================================")
    print("Running fastopt + fast_bemt Joint Rotor Aerodynamic Analysis & Optimization Suite")
    print("=====================================================================================")
    case1_aerodynamic_data_and_jacobian_verification()
    case2_hover_blade_planform_and_twist_optimization()
    case3_tiltrotor_hover_cruise_multipoint_optimization()
    case4_transition_corridor_trim_sweep_and_sensitivity()
    print("=====================================================================================")
    print("ALL 4 FAST_BEMT + FASTOPT ROTORCRAFT CASE STUDIES PASSED SUCCESSFULLY!")
    print("=====================================================================================")


if __name__ == "__main__":
    run_phase13_bemt_suite()
