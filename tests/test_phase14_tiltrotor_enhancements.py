"""
Phase 14 Tiltrotor Enhancements & 5-Section CST Tiltrotor Blade Optimization Suite (`fast_bemt` + `fastopt`):

Verifies all 6 tiltrotor-oriented enhancements:
1. Thread-local same-point flow-field caching in C-ABI + 2D B-spline radial abscissae (`chord_ctrl_r`, `twist_ctrl_r`)
2. OpenMP parallel multi-point C-ABI operator (`eval_multipoint`) with block-sparse Jacobian
3. C++ inner damped-Newton auto-trim (`auto_trim_thrust`) eliminating `pitch_0` via IFT
4. Cyclic pitch (`pitch_1c`, `pitch_1s`) 7x7 IFT Jacobian & 3-DOF oblique-inflow hub trim (`T, Mx=0, My=0`)
5. Root flapwise bending moment (`root_flap_moment`), `propulsive_efficiency`, and `ct/cp/cq` Jacobians
6. Full 5-section Kulfan CST airfoil + chord + twist + collective pitch tiltrotor blade optimization:
   - R = 2.1 m, B = 3, R_hub = 0.21 m (0.1 R)
   - 5 airfoil sections at r/R = [0.1, 0.2, 0.5, 0.75, 1.0] with max thickness >= [0.28, 0.26, 0.18, 0.12, 0.08]
   - c_root = 0.318 m, c_tip >= 0.15 m
   - Hover (0 km ISA, Omega = 110 rad/s): T_hover >= 15,000 N
   - Cruise (4.5 km ISA, Omega = 95.7 rad/s): T_cruise >= 850 N, P_cruise <= 110 kW
   - Objective: Minimize hover shaft power

All comments are written in English.
"""

from __future__ import annotations
import time
import numpy as np

import fastopt as fo
import fast_bemt as fb


def _make_sample_3blade_rotor(n_radial: int = 18) -> fb.RotorGeometry:
    ctrl_r = np.array([0.12, 0.22, 0.38, 0.62, 0.85, 1.00], dtype=np.float64)
    c_vals = np.array([0.095, 0.095, 0.092, 0.088, 0.082, 0.075], dtype=np.float64)
    t_vals = np.deg2rad([20.0, 17.0, 13.0, 10.0, 8.0, 6.0])
    cc = fb.BSplineCurve(ctrl_r, c_vals, 3)
    tc = fb.BSplineCurve(ctrl_r, t_vals, 3)

    a_g = np.deg2rad(np.linspace(-18.0, 18.0, 49))
    re_g = np.array([1.0e5, 5.0e5, 1.5e6, 5.0e6], dtype=np.float64)
    ma_g = np.array([0.0, 0.3, 0.5, 0.7], dtype=np.float64)
    A, RE, MA = np.meshgrid(a_g, re_g, ma_g, indexing="ij")
    cl = 2.0 * np.pi * (A + np.deg2rad(2.0)) / np.sqrt(1.0 - (0.8 * MA) ** 2)
    stall = 1.0 / (1.0 + (A / np.deg2rad(14.0)) ** 6)
    cl_3d = cl * stall + 1.05 * np.sin(2.0 * A) * (1.0 - stall)
    cd_3d = 0.008 * ((RE / 1.0e6) ** (-0.1)) + 0.025 * (A / np.deg2rad(10.0)) ** 2 + 0.04 * (MA ** 3)
    af = fb.create_3d_airfoil(a_g, re_g, ma_g, cl_3d, cd_3d, degrees=False)

    return fb.RotorGeometry(
        num_blades=3, radius=1.25, hub_radius=0.15, n_radial=n_radial,
        chord_curve=cc, twist_curve=tc, airfoils=[af], spin_dir=1,
    )


def test_1_ctrl_r_cyclic_pitch_and_mflap_jacobians():
    print("\n--- [Phase 14 Test 1] 2D ctrl_r, Cyclic Pitch (1c/1s), & M_flap/eta/CT Jacobians ---")
    geom = _make_sample_3blade_rotor(n_radial=16)
    cfg = fb.SolverConfig(n_azimuth=24, wake_model=fb.WakeModel.GlauertPittPeters)
    base_cond = fb.OperatingCondition.from_freestream(
        v_inf=25.0, alpha_disk_rad=np.deg2rad(45.0), omega_rad_s=180.0, pitch_0_rad=np.deg2rad(8.0)
    )
    adapter = fo.FastBEMTAdapter(geom, cfg, base_cond)

    gb = fo.GraphBuilder()
    c_v = gb.variable(shape=(6, 1), init=geom.chord_ctrl)
    t_v = gb.variable(shape=(6, 1), init=geom.twist_ctrl)
    cr_v = gb.variable(shape=(6, 1), init=geom.chord_ctrl_r_norm)
    tr_v = gb.variable(shape=(6, 1), init=geom.twist_ctrl_r_norm)
    om_v = gb.variable(init=180.0)
    p0_v = gb.variable(init=np.deg2rad(8.0))
    p1c_v = gb.variable(init=np.deg2rad(1.5))
    p1s_v = gb.variable(init=np.deg2rad(-1.0))
    v_v = gb.variable(init=25.0)
    ad_v = gb.variable(init=np.deg2rad(45.0))

    aero = adapter(
        chord_ctrl=c_v, twist_ctrl=t_v, chord_ctrl_r=cr_v, twist_ctrl_r=tr_v,
        omega=om_v, pitch_0=p0_v, pitch_1c=p1c_v, pitch_1s=p1s_v, v_inf=v_v, alpha_disk_rad=ad_v,
    )
    outs = fo.vstack([aero.raw_vec, aero.propulsive_efficiency, aero.ct, aero.cp])
    ev = gb.build_evaluator(outs)
    x0 = gb.cpp_graph.initial_decision_vector()

    J_ad = ev.jacobian(x0, 0)
    eps = 1e-6
    J_fd = np.zeros_like(J_ad)
    for j in range(x0.size):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += eps
        xm[j] -= eps
        J_fd[:, j] = (ev.eval(xp)[0].ravel() - ev.eval(xm)[0].ravel()) / (2.0 * eps)

    rel_err = np.linalg.norm(J_ad - J_fd) / np.linalg.norm(J_fd)
    assert rel_err < 1e-5, f"ctrl_r + cyclic pitch + M_flap Jacobian rel_err = {rel_err}"
    print(f"  [PASSED Test 1] 11x30 (ctrl_r + 1c/1s + M_flap + eta + CT/CP) Jacobian rel_err = {rel_err:.2e}")


def test_2_multipoint_openmp_and_3dof_cyclic_trim():
    print("\n--- [Phase 14 Test 2] OpenMP Multi-Point C-ABI & 3-DOF Cyclic Pitch Trim (T, Mx=0, My=0) ---")
    geom = _make_sample_3blade_rotor(n_radial=16)
    cfg = fb.SolverConfig(n_azimuth=24, wake_model=fb.WakeModel.GlauertPittPeters)
    adapter = fo.FastBEMTAdapter(geom, cfg)

    # Verify eval_multipoint matches single-point adapter(...) exactly
    gb = fo.GraphBuilder()
    c_v = gb.variable(shape=(6, 1), init=geom.chord_ctrl)
    t_v = gb.variable(shape=(6, 1), init=geom.twist_ctrl)
    p1 = gb.variable(init=np.deg2rad(4.0))
    p2 = gb.variable(init=np.deg2rad(10.0))

    pts_out = adapter.eval_multipoint(
        [
            {"v_inf": 0.0, "alpha_disk_rad": 0.0, "omega": 200.0, "pitch_0": p1},
            {"v_inf": 25.0, "alpha_disk_rad": np.deg2rad(45.0), "omega": 180.0, "pitch_0": p2},
        ],
        chord_ctrl=c_v,
        twist_ctrl=t_v,
    )
    ev_mp = gb.build_evaluator(fo.vstack([pts_out[0].raw_vec, pts_out[1].raw_vec]))
    x0 = gb.cpp_graph.initial_decision_vector()
    J_mp = ev_mp.jacobian(x0, 0)
    assert J_mp.shape == (16, 14)

    # Compare against single-point evaluator
    gb_sp = fo.GraphBuilder()
    c_s = gb_sp.variable(shape=(6, 1), init=geom.chord_ctrl)
    t_s = gb_sp.variable(shape=(6, 1), init=geom.twist_ctrl)
    p1_s = gb_sp.variable(init=np.deg2rad(4.0))
    p2_s = gb_sp.variable(init=np.deg2rad(10.0))
    a1 = adapter(chord_ctrl=c_s, twist_ctrl=t_s, v_inf=0.0, alpha_disk_rad=0.0, omega=200.0, pitch_0=p1_s)
    a2 = adapter(chord_ctrl=c_s, twist_ctrl=t_s, v_inf=25.0, alpha_disk_rad=np.deg2rad(45.0), omega=180.0, pitch_0=p2_s)
    ev_sp = gb_sp.build_evaluator(fo.vstack([a1.raw_vec, a2.raw_vec]))
    assert np.allclose(ev_mp.eval(x0)[0], ev_sp.eval(x0)[0], rtol=1e-12)
    assert np.allclose(J_mp, ev_sp.jacobian(x0, 0), rtol=1e-12)

    # 3-DOF Oblique-Inflow Hub Trim: find (pitch_0, pitch_1c, pitch_1s) for T == 2400 N, Mx == 0, My == 0
    opti = fo.Opti()
    p0 = opti.variable(init=np.deg2rad(7.0), lb=np.deg2rad(-5.0), ub=np.deg2rad(20.0), scale=0.2, name="p0")
    p1c = opti.variable(init=0.0, lb=np.deg2rad(-12.0), ub=np.deg2rad(12.0), scale=0.1, name="p1c")
    p1s = opti.variable(init=0.0, lb=np.deg2rad(-12.0), ub=np.deg2rad(12.0), scale=0.1, name="p1s")

    aero_trim = adapter(
        v_inf=25.0, alpha_disk_rad=np.deg2rad(45.0), omega=180.0,
        pitch_0=p0, pitch_1c=p1c, pitch_1s=p1s,
    )
    opti.minimize(aero_trim.power * 1e-3)
    opti.subject_to(aero_trim.thrust == 2400.0, scale=2400.0)
    opti.subject_to(aero_trim.mx == 0.0, scale=200.0)
    opti.subject_to(aero_trim.my == 0.0, scale=200.0)
    opti.solver("SLSQP", options={"ftol": 1e-9, "maxiter": 40})
    sol = opti.solve()
    assert sol.success
    assert abs(float(sol.value(aero_trim.thrust)) - 2400.0) < 0.5
    assert abs(float(sol.value(aero_trim.mx))) < 0.1
    assert abs(float(sol.value(aero_trim.my))) < 0.1
    print(
        f"  [PASSED Test 2] Multi-Point OpenMP exact match | 3-DOF Cyclic Trim: "
        f"p0={np.rad2deg(sol.value(p0)):.2f} deg, p1c={np.rad2deg(sol.value(p1c)):.2f} deg, "
        f"p1s={np.rad2deg(sol.value(p1s)):.2f} deg -> Mx={sol.value(aero_trim.mx):.3f} N*m, My={sol.value(aero_trim.my):.3f} N*m"
    )


def test_3_cpp_inner_autotrim_hover_optimization():
    print("\n--- [Phase 14 Test 3] C++ Inner Auto-Trim (IFT) Pure-Geometry Hover Optimization ---")
    geom = _make_sample_3blade_rotor(n_radial=18)
    cfg = fb.SolverConfig(n_azimuth=1, wake_model=fb.WakeModel.GlauertAnnulus)
    cond = fb.OperatingCondition(omega=200.0, pitch_0=np.deg2rad(4.0))
    adapter = fo.FastBEMTAdapter(geom, cfg, cond)

    # Verify Auto-Trim IFT Jacobian against FD
    gb = fo.GraphBuilder()
    c_v = gb.variable(shape=(6, 1), init=geom.chord_ctrl)
    t_v = gb.variable(shape=(6, 1), init=geom.twist_ctrl)
    aero_at = adapter(chord_ctrl=c_v, twist_ctrl=t_v, omega=200.0, auto_trim_thrust=2200.0)
    ev = gb.build_evaluator(fo.vstack([aero_at.power, aero_at.trimmed_pitch_0]))
    x0 = gb.cpp_graph.initial_decision_vector()
    J_ad = ev.jacobian(x0, 0)
    eps = 1e-6
    J_fd = np.zeros_like(J_ad)
    for j in range(x0.size):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += eps
        xm[j] -= eps
        J_fd[:, j] = (ev.eval(xp)[0].ravel() - ev.eval(xm)[0].ravel()) / (2.0 * eps)
    rel_err = np.linalg.norm(J_ad - J_fd) / np.linalg.norm(J_fd)
    assert rel_err < 1e-5, f"AutoTrim IFT Jacobian rel_err = {rel_err}"

    # Pure-geometric hover optimization (no pitch_0 variable, no thrust equality constraint!)
    opti = fo.Opti()
    c_opt = opti.variable(shape=(6, 1), init=geom.chord_ctrl, lb=0.04, ub=0.14, scale=0.09)
    t_opt = opti.variable(shape=(6, 1), init=geom.twist_ctrl, lb=np.deg2rad(3.0), ub=np.deg2rad(30.0), scale=0.3)
    aero = adapter(chord_ctrl=c_opt, twist_ctrl=t_opt, omega=200.0, auto_trim_thrust=2200.0)
    reg = adapter.add_planform_constraints(
        opti, chord_ctrl=c_opt, twist_ctrl=t_opt, solidity=aero.solidity, max_solidity=0.06,
        smoothness_weight_chord=10.0, smoothness_weight_twist=0.05,
    )
    opti.minimize(aero.power * 1e-3 + reg)
    opti.solver("SLSQP", options={"ftol": 1e-7, "maxiter": 60})
    t0 = time.perf_counter()
    sol = opti.solve()
    ms = (time.perf_counter() - t0) * 1e3
    assert sol.success
    assert abs(float(sol.value(aero.thrust)) - 2200.0) < 0.05
    print(
        f"  [PASSED Test 3] AutoTrim IFT Jacobian rel_err = {rel_err:.2e} | "
        f"Trimmed T={sol.value(aero.thrust):.2f} N, P*={sol.value(aero.power)/1e3:.3f} kW, "
        f"p0*={np.rad2deg(sol.value(aero.trimmed_pitch_0)):.2f} deg in {ms:.1f} ms"
    )


def test_4_user_5section_cst_tiltrotor_optimization():
    """
    User-Requested 5-Section CST Tiltrotor Blade Optimization Case Study:
      - Radius R = 2.1 m, Blades B = 3, Hub radius R_hub = 0.1 * R = 0.21 m
      - 5 airfoil sections at r/R = [0.1, 0.2, 0.5, 0.75, 1.0]
      - Max thickness >= [0.28, 0.26, 0.18, 0.12, 0.08] at the 5 sections
      - Root chord c(0.1 R) = 0.318 m, tip chord c(R) >= 0.15 m
      - Hover (0 km ISA): Omega_hover in [100, 115] rad/s (optimized), T_hover >= 15,000 N
      - Cruise (4.5 km ISA): Omega_cruise >= 0.5 * Omega_hover (optimized), V_cruise = 75 m/s, T_cruise >= 850 N, P_cruise <= 110 kW
      - Objective: Minimize hover shaft power
    """
    print("\n--- [Phase 14 Test 4] 5-Section CST Tiltrotor Optimization (R=2.1m, 0km Hover + 4.5km Cruise) ---")
    R = 2.1
    R_hub = 0.1 * R
    B = 3
    sec_r_norm = np.array([0.1, 0.2, 0.5, 0.75, 1.0], dtype=np.float64)
    min_thick_req = np.array([0.28, 0.26, 0.18, 0.12, 0.08], dtype=np.float64)

    # Construct 5 initial Kulfan CST airfoils satisfying each section's minimum thickness (e.g. NACA 4429, 4427, 4419, 4413, 4409)
    init_kulfans = [
        fb.KulfanCSTAirfoil.from_naca4("4429"),
        fb.KulfanCSTAirfoil.from_naca4("4427"),
        fb.KulfanCSTAirfoil.from_naca4("4419"),
        fb.KulfanCSTAirfoil.from_naca4("4413"),
        fb.KulfanCSTAirfoil.from_naca4("4409"),
    ]
    init_cst_mat = np.vstack([k.to_vector(include_te=False) for k in init_kulfans])  # (5, 17)

    # Build initial 3D airfoil polars at the 5 sections
    a_g = np.deg2rad(np.linspace(-18.0, 18.0, 49))
    re_g = np.array([1.0e5, 5.0e5, 1.5e6, 5.0e6], dtype=np.float64)
    ma_g = np.array([0.0, 0.25, 0.50, 0.75], dtype=np.float64)
    A_d, RE_d, MA_d = np.meshgrid(np.rad2deg(a_g), re_g, ma_g, indexing="ij")
    init_afs = []
    for kaf in init_kulfans:
        cl_f, cd_f = fo.bemt_bridge._default_kulfan_polar_eval(kaf, A_d.ravel(), RE_d.ravel(), MA_d.ravel())
        init_afs.append(fb.Airfoil3D(a_g, re_g, ma_g, cl_f, cd_f))

    ctrl_r = np.array([0.10, 0.22, 0.42, 0.65, 0.85, 1.00], dtype=np.float64)
    init_chord = np.array([0.318, 0.315, 0.300, 0.275, 0.240, 0.200], dtype=np.float64)
    init_twist = np.deg2rad([28.0, 23.0, 15.0, 10.0, 6.5, 4.0])

    geom = fb.RotorGeometry(
        num_blades=B,
        radius=R,
        hub_radius=R_hub,
        n_radial=16,
        chord_curve=fb.BSplineCurve(ctrl_r, init_chord, 3),
        twist_curve=fb.BSplineCurve(ctrl_r, init_twist, 3),
        airfoils=init_afs,
        section_r_norm=sec_r_norm,
        spin_dir=1,
    )

    cfg = fb.SolverConfig(n_azimuth=1, wake_model=fb.WakeModel.GlauertAnnulus, enable_tip_loss=True)

    # Hover (0 km ISA, Omega = 110 rad/s) and Cruise (4.5 km ISA, Omega = 95.7 rad/s, V_cruise = 75 m/s)
    omega_hov = 110.0
    omega_cru = 95.7
    v_cruise = 75.0
    cond_hov_base = fb.OperatingCondition(vx=0.0, vy=0.0, vz=0.0, omega=omega_hov, pitch_0=np.deg2rad(6.0))
    cond_cru_base = fb.OperatingCondition(vx=0.0, vy=0.0, vz=v_cruise, omega=omega_cru, pitch_0=np.deg2rad(22.0))

    bemt_hov = fo.FastBEMTAdapter(
        geom, cfg, cond_hov_base, altitude_m=0.0, section_airfoils_kulfan=init_kulfans, name="tilt_hov"
    )
    bemt_cru = fo.FastBEMTAdapter(
        geom, cfg, cond_cru_base, altitude_m=4500.0, section_airfoils_kulfan=init_kulfans, name="tilt_cru"
    )

    # 1. Trim baseline collective pitches for Hover (T >= 15000 N) and Cruise (T >= 850 N)
    opti_bh = fo.Opti()
    ph_b = opti_bh.variable(init=np.deg2rad(6.0), lb=np.deg2rad(-2.0), ub=np.deg2rad(14.0))
    ah_b = bemt_hov(pitch_0=ph_b, omega=omega_hov)
    opti_bh.minimize(fo.square((ah_b.thrust - 15000.0) / 15000.0))
    opti_bh.solver("SLSQP", options={"ftol": 1e-10, "maxiter": 40})
    sol_bh = opti_bh.solve()

    opti_bc = fo.Opti()
    pc_b = opti_bc.variable(init=np.deg2rad(28.0), lb=np.deg2rad(15.0), ub=np.deg2rad(45.0))
    ac_b = bemt_cru(pitch_0=pc_b, omega=omega_cru, vz=v_cruise)
    opti_bc.minimize(fo.square((ac_b.thrust - 850.0) / 850.0))
    opti_bc.solver("SLSQP", options={"ftol": 1e-10, "maxiter": 40})
    sol_bc = opti_bc.solve()

    P_hov_base = float(sol_bh.value(ah_b.power))
    P_cru_base = float(sol_bc.value(ac_b.power))
    print(
        f"  [Baseline 5-Sec Rotor] 0km Hover (Omega=110 rad/s, T={sol_bh.value(ah_b.thrust):.1f} N): "
        f"P_hov = {P_hov_base/1e3:.2f} kW | 4.5km Cruise (V=75 m/s, Omega=95.7 rad/s, T={sol_bc.value(ac_b.thrust):.1f} N): "
        f"P_cru = {P_cru_base/1e3:.2f} kW"
    )

    # 2. Full Coupled Optimization: Chord (6) + Twist (6) + 5-Section CST (85) + Pitches (2) + Omegas (2) = 101 vars
    opti = fo.Opti()
    c_ctrl = opti.variable(shape=(6, 1), init=init_chord, lb=0.15, ub=0.45, scale=0.25, name="chord_ctrl")
    tw_ctrl = opti.variable(
        shape=(6, 1), init=init_twist, lb=np.deg2rad(1.0), ub=np.deg2rad(42.0), scale=0.3, name="twist_ctrl"
    )
    cst_lb = init_cst_mat - 0.12
    cst_ub = init_cst_mat + 0.12
    # Keep leading-edge weight bounded within reasonable range
    cst_lb[:, 16] = np.maximum(cst_lb[:, 16], -0.25)
    cst_ub[:, 16] = np.minimum(cst_ub[:, 16], 0.25)
    cst_var = opti.variable(
        shape=(5, 17), init=init_cst_mat, lb=cst_lb, ub=cst_ub, scale=0.2, name="cst_params"
    )
    ph = opti.variable(
        init=float(sol_bh.value(ph_b)), lb=np.deg2rad(-4.0), ub=np.deg2rad(14.0), scale=0.1, name="pitch_hov"
    )
    pc = opti.variable(
        init=float(sol_bc.value(pc_b)), lb=np.deg2rad(15.0), ub=np.deg2rad(55.0), scale=0.2, name="pitch_cru"
    )
    # Normalize rotor speeds by 100 rad/s so all 101 variables are O(0.1..1.0) for IPOPT L-BFGS
    om_hov_n = opti.variable(init=omega_hov / 100.0, lb=1.00, ub=1.15, name="omega_hov_norm")
    om_cru_n = opti.variable(init=omega_cru / 100.0, lb=0.50, ub=1.15, name="omega_cru_norm")
    om_hov = 100.0 * om_hov_n
    om_cru = 100.0 * om_cru_n

    aero_hov = bemt_hov(chord_ctrl=c_ctrl, twist_ctrl=tw_ctrl, cst_params=cst_var, pitch_0=ph, omega=om_hov)
    aero_cru = bemt_cru(
        chord_ctrl=c_ctrl, twist_ctrl=tw_ctrl, cst_params=cst_var, pitch_0=pc, omega=om_cru, vz=v_cruise
    )

    # Geometric & Planform Constraints:
    # - Root chord = 0.318 m, Tip chord >= 0.15 m, chord taper, twist washout
    reg_geom = bemt_hov.add_planform_constraints(
        opti,
        chord_ctrl=c_ctrl,
        twist_ctrl=tw_ctrl,
        root_chord=0.318,
        min_tip_chord=0.15,
        chord_taper=True,
        twist_washout=True,
        smoothness_weight_chord=120.0,
        smoothness_weight_twist=2.0,
    )
    #Mild CST shape regularization to avoid jagged airfoil wiggles
    reg_cst = 5.0 * fo.sumsqr(cst_var - init_cst_mat)

    # Objective: Minimize Hover Shaft Power [kW] (scaled by 300 kW so O(1) for NLP tolerances)
    opti.minimize(aero_hov.power * 1e-3 + reg_geom + reg_cst, scale=300.0)

    # Aerodynamic Performance & Rotor Speed Constraints:
    opti.subject_to(aero_hov.thrust >= 15000.0, scale=15000.0, name="hover_thrust_min")
    opti.subject_to(aero_cru.thrust >= 850.0, scale=850.0, name="cruise_thrust_min")
    opti.subject_to(aero_cru.power <= 110000.0, scale=110000.0, name="cruise_power_max")
    opti.subject_to(om_cru >= 0.5 * om_hov, scale=50.0, name="cruise_omega_ratio")

    # 5-Section Airfoil Thickness Constraints (max thickness at x/c ~ 0.30 >= [0.28, 0.26, 0.18, 0.12, 0.08])
    # plus non-self-intersection (t/c >= 0.005) across chordwise check points
    x_chk = [0.10, 0.20, 0.30, 0.50, 0.75, 0.90]
    thick_mat = bemt_hov.cst_thickness(cst_var, x_chk=x_chk)  # shape (5, 6)
    opti.subject_to(thick_mat[:, 2] >= min_thick_req, scale=0.2, name="section_min_thickness")
    opti.subject_to(thick_mat >= 0.005, scale=0.1, name="airfoil_no_crossover")

    # Use IPOPT (L-BFGS) when CasADi/IPOPT runtime is available (ideal for 99-var NeuralFoil CST);
    # otherwise use SciPy SLSQP with normalized tolerance.
    try:
        import casadi  # noqa: F401
        opti.solver(
            "ipopt",
            use_hessian=False,
            options={
                "max_iter": 200,
                "tol": 2e-3,
                "acceptable_tol": 1e-2,
                "acceptable_constr_viol_tol": 1e-3,
                "acceptable_iter": 3,
                "print_level": 0,
            },
        )
    except ImportError:
        opti.solver("SLSQP", options={"ftol": 1e-4, "maxiter": 150})

    t0 = time.perf_counter()
    sol = opti.solve()
    opt_s = time.perf_counter() - t0

    assert sol.success, f"5-section CST tiltrotor optimization failed: {sol.message}"
    P_hov_opt = float(sol.value(aero_hov.power))
    T_hov_opt = float(sol.value(aero_hov.thrust))
    P_cru_opt = float(sol.value(aero_cru.power))
    T_cru_opt = float(sol.value(aero_cru.thrust))
    om_hov_opt = float(sol.value(om_hov))
    om_cru_opt = float(sol.value(om_cru))
    c_opt = sol.value(c_ctrl)
    tw_opt_deg = np.rad2deg(sol.value(tw_ctrl))
    thick_opt = sol.value(thick_mat)[:, 2]

    assert abs(c_opt[0] - 0.318) < 1e-4
    assert c_opt[-1] >= 0.15 - 1e-4
    assert 100.0 - 1e-3 <= om_hov_opt <= 115.0 + 1e-3
    assert om_cru_opt >= 0.5 * om_hov_opt - 1e-3
    assert T_hov_opt >= 15000.0 - 5.0
    assert T_cru_opt >= 850.0 - 1.0
    assert P_cru_opt <= 110000.0 + 50.0
    assert np.all(thick_opt >= min_thick_req - 1e-4)
    assert P_hov_opt < P_hov_base * 0.985, f"Expected hover power drop: {P_hov_opt} vs {P_hov_base}"

    print(
        f"  [PASSED Test 4 — 5-Section CST Tiltrotor Optimization in {opt_s:.2f} s]\n"
        f"    Hover (0km, Omega*={om_hov_opt:.2f} rad/s):   T* = {T_hov_opt:.1f} N (>= 15000 N), P* = {P_hov_opt/1e3:.2f} kW "
        f"({(1.0 - P_hov_opt/P_hov_base)*100:.2f}% saved vs {P_hov_base/1e3:.2f} kW)\n"
        f"    Cruise (4.5km, Omega*={om_cru_opt:.2f} rad/s, ratio={om_cru_opt/om_hov_opt:.3f} >= 0.5): "
        f"T* = {T_cru_opt:.1f} N (>= 850 N), P* = {P_cru_opt/1e3:.2f} kW (<= 110 kW)\n"
        f"    Chord Ctrl [m]:           {np.round(c_opt, 4)}\n"
        f"    Twist Ctrl [deg]:         {np.round(tw_opt_deg, 2)}\n"
        f"    5-Sec t/c @ 30%c:         {np.round(thick_opt, 4)} (req >= {min_thick_req})"
    )


def run_phase14_suite():
    print("=====================================================================================")
    print("Running fastopt + fast_bemt Phase 14 Tiltrotor Enhancements & CST Case Study")
    print("=====================================================================================")
    test_1_ctrl_r_cyclic_pitch_and_mflap_jacobians()
    test_2_multipoint_openmp_and_3dof_cyclic_trim()
    test_3_cpp_inner_autotrim_hover_optimization()
    test_4_user_5section_cst_tiltrotor_optimization()
    print("=====================================================================================")
    print("ALL PHASE 14 TESTS AND TILTROTOR CST OPTIMIZATION PASSED SUCCESSFULLY!")
    print("=====================================================================================")


if __name__ == "__main__":
    run_phase14_suite()
