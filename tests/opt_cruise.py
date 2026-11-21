import time
import numpy as np

try:
    import python.fastopt as fo
except ImportError:
    import fastopt as fo
import fast_bemt as fb
import aerosandbox as asb


def _eval_kulfan_coords(cst_vec17, te_thickness, n_pts=161):
    """Evaluate smooth cosine-spaced upper and lower Kulfan CST airfoil coordinates."""
    import math
    beta = np.linspace(0.0, np.pi, n_pts)
    x = 0.5 * (1.0 - np.cos(beta))
    cx = (x ** 0.5) * (1.0 - x)
    bx = np.column_stack([float(math.comb(7, k)) * (x ** k) * ((1.0 - x) ** (7 - k)) for k in range(8)])
    lo_w = cst_vec17[:8]
    up_w = cst_vec17[8:16]
    le_w = float(cst_vec17[16])
    le_term = le_w * x * ((1.0 - x) ** 8)
    y_up = cx * (bx @ up_w) + le_term + 0.5 * float(te_thickness) * x
    y_lo = cx * (bx @ lo_w) + le_term - 0.5 * float(te_thickness) * x
    return x, y_up, y_lo


def _plot_cruise_results(
    ctrl_r, init_chord, opt_chord, init_twist_deg, opt_twist_deg,
    sec_r_norm, init_cst_mat, opt_cst_mat, te_thicknesses,
    thick_opt, x_tmax_loc, opt_conf_cru,
    min_tip_chord=0.04,
):
    """Plot baseline vs. optimized propeller chord/twist distributions and all 5 airfoil sections."""
    import matplotlib.pyplot as plt

    def _sample_bspline(cr, vals, n=200):
        u_d = np.linspace(0.0, 1.0, n)
        raw = fb.BSplineCurve(cr, vals, 3).eval_u_batch(u_d)
        if isinstance(raw, tuple) and len(raw) == 2:
            return np.asarray(raw[0], dtype=np.float64).ravel(), np.asarray(raw[1], dtype=np.float64).ravel()
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == n:
            return arr[0], arr[1]
        return arr[:, 0], arr[:, 1]

    r_ci, v_ci = _sample_bspline(ctrl_r, init_chord)
    r_co, v_co = _sample_bspline(ctrl_r, opt_chord)
    r_ti, v_ti = _sample_bspline(ctrl_r, init_twist_deg)
    r_to, v_to = _sample_bspline(ctrl_r, opt_twist_deg)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.0, 1.0], hspace=0.42, wspace=0.28)

    # 1. Spanwise Chord Distribution (Propeller Leaf / Mid-Span Bulge)
    ax_c = fig.add_subplot(gs[0, 0:2])
    ax_c.plot(r_ci, v_ci, "--", color="gray", lw=1.8, label="Baseline Chord")
    ax_c.plot(r_co, v_co, "-", color="#1f77b4", lw=2.4, label="Optimized Propeller Chord")
    ax_c.fill_between(r_co, 0.0, v_co, color="#1f77b4", alpha=0.10)
    ax_c.scatter(ctrl_r, opt_chord, color="#1f77b4", s=36, zorder=5, label="Opt Ctrl Pts")
    ax_c.axhline(min_tip_chord, color="#d62728", ls=":", lw=1.2, label=f"Min Chord ({min_tip_chord:.2f} m)")
    ax_c.set_title("Pure-Cruise Propeller Spanwise Chord Distribution c(r/R)")
    ax_c.set_xlabel("r / R")
    ax_c.set_ylabel("Chord [m]")
    ax_c.set_ylim(0.0, max(float(np.max(v_co)), float(np.max(v_ci))) * 1.15)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8, ncol=2)

    # 2. Spanwise Twist Distribution
    ax_t = fig.add_subplot(gs[0, 2])
    ax_t.plot(r_ti, v_ti, "--", color="gray", lw=1.8, label="Baseline Twist")
    ax_t.plot(r_to, v_to, "-", color="#ff7f0e", lw=2.4, label="Optimized Twist")
    ax_t.scatter(ctrl_r, opt_twist_deg, color="#ff7f0e", s=36, zorder=5)
    ax_t.set_title("Spanwise Twist (Tip Ref = 0 deg)")
    ax_t.set_xlabel("r / R")
    ax_t.set_ylabel("Twist [deg]")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=8)

    # 3. Five Airfoil Sections (Baseline vs. Optimized)
    slots = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    for s_idx, (gr, gc) in enumerate(slots):
        ax = fig.add_subplot(gs[gr, gc])
        x0, yu0, yl0 = _eval_kulfan_coords(init_cst_mat[s_idx], te_thicknesses[s_idx])
        x1, yu1, yl1 = _eval_kulfan_coords(opt_cst_mat[s_idx], te_thicknesses[s_idx])
        ax.plot(x0, yu0, "--", color="gray", lw=1.3, label="Baseline")
        ax.plot(x0, yl0, "--", color="gray", lw=1.3)
        ax.plot(x1, yu1, "-", color="#1f77b4", lw=2.0, label="Optimized")
        ax.plot(x1, yl1, "-", color="#1f77b4", lw=2.0)
        ax.fill_between(x1, yl1, yu1, color="#1f77b4", alpha=0.12)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.22, 0.28)
        ax.set_title(
            f"r/R={sec_r_norm[s_idx]:.2f} | t/c={thick_opt[s_idx]*100:.1f}% @{x_tmax_loc[s_idx]:.2f}c\n"
            f"Conf Cruise={opt_conf_cru[s_idx]:.3f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)
        if s_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    # 4. Thickness & Confidence Summary Panel (bottom-right)
    ax_sum = fig.add_subplot(gs[2, 2])
    ax_sum.plot(sec_r_norm, thick_opt * 100.0, "o-", color="#2ca02c", lw=2.0, label="Opt max t/c [%]")
    ax_sum.set_xlabel("r / R")
    ax_sum.set_ylabel("Max t/c [%]", color="#2ca02c")
    ax_sum.tick_params(axis="y", labelcolor="#2ca02c")
    ax_sum.grid(True, alpha=0.3)
    ax_sum2 = ax_sum.twinx()
    ax_sum2.plot(sec_r_norm, opt_conf_cru, "^--", color="#d62728", lw=1.6, label="Conf Cruise")
    ax_sum2.set_ylabel("NF Confidence", color="#d62728")
    ax_sum2.set_ylim(0.80, 1.01)
    ax_sum.set_title("Spanwise Max t/c & NF Confidence", fontsize=9)
    lines1, labels1 = ax_sum.get_legend_handles_labels()
    lines2, labels2 = ax_sum2.get_legend_handles_labels()
    ax_sum.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="lower left")

    fig.suptitle("Pure-Cruise Propeller Coupled Optimization (Betz-Goldstein Mid-Span Bulge Verification)", fontsize=12)
    plt.show()


def cruise_propeller_optimization():
    """
    Pure-Cruise Propeller Coupled Optimization (Betz-Goldstein Mid-Span Bulge Verification):
      - Radius R = 2.1 m, Blades B = 3, Hub radius R_hub = 0.1 * R = 0.21 m
      - Cruise only (4.5 km ISA, V_cruise = 400 km/h = 111.11 m/s)
      - Fixed moderate cruise RPM (65 rad/s, J = 2.56) and high cruise thrust (T_cru_req = 5,000 N)
        so the blade is sufficiently loaded for mid-span chord to bulge naturally to ~0.20-0.28 m
      - Low root & tip chord lower bounds (0.05 m / 0.04 m) so root & tip are free to narrow
        under Betz-Goldstein + Prandtl root/tip losses
      - Light chord smoothness penalty (10.0) so natural propeller curvature is not flattened
    """
    # =========================================================================
    # User-Configurable Propeller Cruise Parameters
    # =========================================================================
    R = 2.1
    R_hub = 0.1 * R
    B = 3
    A_disk = np.pi * (R ** 2)
    max_tip_mach = 0.70

    # Cruise mission target (high advance ratio J = V / (n * D) = 2.56, loaded propeller)
    rho_cru, mu_cru, a_sound_cru = 0.777, 1.64477e-5, 322.573
    v_cruise = 400.0 / 3.6
    omega_cru_base = 65.0
    freeze_cruise_omega = True
    T_cru_req = 5000.0

    # Low root/tip lower bounds so the optimizer can freely form a narrow shank & tip
    root_chord_lb, root_chord_ub = 0.05, 0.35
    min_tip_chord = 0.04
    max_mid_chord = 0.45

    sec_r_norm = np.array([0.1, 0.2, 0.5, 0.75, 1.0], dtype=np.float64)
    min_thick_req = np.array([0.28, 0.26, 0.18, 0.12, 0.08], dtype=np.float64)
    max_thick_req = np.array([0.305, 0.275, 0.195, 0.130, 0.090], dtype=np.float64)

    J_adv = v_cruise / ((omega_cru_base / (2.0 * np.pi)) * (2.0 * R))
    print(
        f"\n--- Pure-Cruise Propeller Optimization "
        f"(R={R}m, 4.5km V={v_cruise*3.6:.0f}km/h, Omega={omega_cru_base:.1f} rad/s, J={J_adv:.2f}, T>={T_cru_req:.0f}N) ---"
    )

    # 5 initial Kulfan CST airfoils
    init_kulfans = [
        fb.KulfanCSTAirfoil.from_aerosandbox(asb.Airfoil(coordinates="./tests/afs/naca64-228.dat").to_kulfan_airfoil()),
        fb.KulfanCSTAirfoil.from_aerosandbox(asb.Airfoil(coordinates="./tests/afs/naca64-226.dat").to_kulfan_airfoil()),
        fb.KulfanCSTAirfoil.from_aerosandbox(asb.Airfoil(coordinates="./tests/afs/naca64-218.dat").to_kulfan_airfoil()),
        fb.KulfanCSTAirfoil.from_aerosandbox(asb.Airfoil(coordinates="./tests/afs/naca64-212.dat").to_kulfan_airfoil()),
        fb.KulfanCSTAirfoil.from_aerosandbox(asb.Airfoil(coordinates="./tests/afs/naca64-108.dat").to_kulfan_airfoil()),
    ]
    init_cst_mat = np.vstack([k.to_vector(include_te=False) for k in init_kulfans])

    # Initial 3D polars for baseline RotorGeometry construction
    a_g = np.deg2rad(np.linspace(-18.0, 18.0, 49))
    re_g = np.array([1.0e5, 5.0e5, 1.5e6, 5.0e6], dtype=np.float64)
    ma_g = np.array([0.0, 0.25, 0.50, 0.70, 0.85], dtype=np.float64)
    A_d, RE_d, MA_d = np.meshgrid(np.rad2deg(a_g), re_g, ma_g, indexing="ij")
    init_afs = []
    for kaf in init_kulfans:
        cl_f, cd_f = fo.bemt_bridge._default_kulfan_polar_eval(kaf, A_d.ravel(), RE_d.ravel(), MA_d.ravel())
        init_afs.append(fb.Airfoil3D(a_g, re_g, ma_g, cl_f, cd_f))

    ctrl_r = np.array([0.10, 0.22, 0.42, 0.65, 0.85, 1.00], dtype=np.float64)
    # Start from a uniform 0.22 m baseline chord so any mid-span bulge is purely optimizer-discovered
    init_chord = np.array([0.22, 0.22, 0.22, 0.22, 0.22, 0.22], dtype=np.float64)
    init_twist = np.deg2rad([34.0, 26.0, 16.0, 8.0, 3.0, 0.0])

    geom = fb.RotorGeometry(
        num_blades=B,
        radius=R,
        hub_radius=R_hub,
        n_radial=24,
        chord_curve=fb.BSplineCurve(ctrl_r, init_chord, 3),
        twist_curve=fb.BSplineCurve(ctrl_r, init_twist, 3),
        airfoils=init_afs,
        section_r_norm=sec_r_norm,
        spin_dir=1,
    )

    cfg = fb.SolverConfig(
        n_azimuth=1,
        wake_model=fb.WakeModel.GlauertAnnulus,
        enable_tip_loss=True,
        enable_vrs_correction=True,
    )

    cond_cru_base = fb.OperatingCondition(
        vx=0.0, vy=0.0, vz=v_cruise, omega=omega_cru_base, pitch_0=np.deg2rad(42.0),
        rho=rho_cru, mu=mu_cru, a_sound=a_sound_cru,
    )

    bemt_cru = fo.FastBEMTAdapter(
        geom, cfg, cond_cru_base, section_airfoils_kulfan=init_kulfans, name="prop_cru"
    )

    # Representative Cruise (alpha, Re) per section for NeuralFoil confidence check
    c_sec_est = np.interp(sec_r_norm, ctrl_r, init_chord)
    alpha_cru_sec = np.array([4.0, 4.0, 3.5, 3.0, 2.5], dtype=np.float64)
    Re_cru_sec = np.maximum(rho_cru * np.hypot(omega_cru_base * sec_r_norm * R, v_cruise) * c_sec_est / mu_cru, 2.0e5)

    nf_surr = getattr(bemt_cru, "_nf_surrogate", None)
    min_conf_target = np.array([0.85, 0.85, 0.90, 0.90, 0.90], dtype=np.float64)
    init_conf_cru = np.zeros(5, dtype=np.float64)
    if nf_surr is not None:
        for s, kaf in enumerate(init_kulfans):
            init_conf_cru[s] = float(
                nf_surr.get_aero_from_kulfan_parameters(kaf.kulfan_parameters, alpha_cru_sec[s], Re_cru_sec[s])[
                    "analysis_confidence"
                ][0]
            )
        min_conf_cru = np.minimum(min_conf_target, np.maximum(init_conf_cru - 0.02, 0.75))
        print(f"  [Baseline NF Confidence] Cruise: {np.round(init_conf_cru, 4)} (req >= {np.round(min_conf_cru, 3)})")
    else:
        min_conf_cru = min_conf_target.copy()

    # 1. Trim baseline collective pitch for Cruise (T_cru_req)
    opti_bc = fo.Opti()
    pc_b = opti_bc.variable(init=np.deg2rad(42.0), lb=np.deg2rad(15.0), ub=np.deg2rad(68.0))
    ac_b = bemt_cru(pitch_0=pc_b, omega=omega_cru_base, vz=v_cruise)
    opti_bc.minimize(fo.square((ac_b.thrust - T_cru_req) / T_cru_req))
    opti_bc.solver("SLSQP", options={"ftol": 1e-10, "maxiter": 40})
    sol_bc = opti_bc.solve()

    T_cru_base = float(sol_bc.value(ac_b.thrust))
    P_cru_base = float(sol_bc.value(ac_b.power))
    eta_cru_base = (T_cru_base * v_cruise) / max(P_cru_base, 1.0)
    M_cru_base = np.hypot(omega_cru_base * R, v_cruise) / a_sound_cru
    print(
        f"  [Baseline Uniform Chord] 4.5km Cruise (V={v_cruise:.2f} m/s, Omega={omega_cru_base:.1f} rad/s, "
        f"M_tip={M_cru_base:.3f}, T={T_cru_base:.1f} N): P_cru = {P_cru_base/1e3:.2f} kW, eta = {eta_cru_base*100:.2f}%"
    )

    # 2. Pure-Cruise Coupled Optimization: Chord (6) + Twist (6) + 5-Section CST (85) + Pitch (1) + Omega (1)
    opti = fo.Opti()
    c_lb = np.array([root_chord_lb, min_tip_chord, min_tip_chord, min_tip_chord, min_tip_chord, min_tip_chord], dtype=np.float64)
    c_ub = np.array([root_chord_ub, max_mid_chord, max_mid_chord, max_mid_chord, max_mid_chord, max_mid_chord], dtype=np.float64)
    c_ctrl = opti.variable(shape=(6, 1), init=init_chord, lb=c_lb, ub=c_ub, scale=0.25, name="chord_ctrl")

    tw_ctrl = opti.variable(
        shape=(6, 1), init=init_twist, lb=np.deg2rad(-20.0), ub=np.deg2rad(50.0), scale=0.3, name="twist_ctrl"
    )
    cst_lb = init_cst_mat - 0.25
    cst_ub = init_cst_mat + 0.25
    cst_lb[:, 16] = np.maximum(cst_lb[:, 16], -0.25)
    cst_ub[:, 16] = np.minimum(cst_ub[:, 16], 0.25)
    cst_var = opti.variable(
        shape=(5, 17), init=init_cst_mat, lb=cst_lb, ub=cst_ub, scale=0.2, name="cst_params"
    )
    pc = opti.variable(
        init=float(sol_bc.value(pc_b)), lb=np.deg2rad(15.0), ub=np.deg2rad(68.0), scale=0.2, name="pitch_cru"
    )
    om_cru_n = opti.variable(
        init=omega_cru_base / 100.0, lb=0.45, ub=1.15, name="omega_cru_norm", freeze=freeze_cruise_omega
    )
    om_cru = 100.0 * om_cru_n

    aero_cru = bemt_cru(
        chord_ctrl=c_ctrl, twist_ctrl=tw_ctrl, cst_params=cst_var, pitch_0=pc, omega=om_cru, vz=v_cruise
    )

    # Light chord smoothness (10.0) so natural leaf-shaped mid-span curvature is preserved
    reg_geom = bemt_cru.add_planform_constraints(
        opti,
        chord_ctrl=c_ctrl,
        twist_ctrl=tw_ctrl,
        root_chord=None,
        min_tip_chord=min_tip_chord,
        chord_taper=False,
        twist_washout=True,
        smoothness_weight_chord=10.0,
        smoothness_weight_twist=1.0,
    )
    opti.subject_to(tw_ctrl[-1] == 0.0, scale=0.1, name="tip_twist_ref")
    # Allow first 4 chord control points (0.10, 0.22, 0.42, 0.65 R) to rise freely into a mid-span peak,
    # then enforce smooth outer taper toward the tip: c[3] >= c[4] >= c[5]
    opti.subject_to(c_ctrl[4:6] <= c_ctrl[3:5], scale=0.1, name="tip_chord_taper")

    # Internal CST Bernstein coefficient smoothness
    d2_cst_lo = cst_var[:, 2:8] - 2.0 * cst_var[:, 1:7] + cst_var[:, 0:6]
    d2_cst_up = cst_var[:, 10:16] - 2.0 * cst_var[:, 9:15] + cst_var[:, 8:14]
    reg_cst = 10.0 * (fo.sumsqr(d2_cst_lo) + fo.sumsqr(d2_cst_up))

    obj_scale_cru = max(P_cru_base * 1e-3, 100.0)
    opti.minimize(aero_cru.power * 1e-3 + reg_geom + reg_cst, scale=obj_scale_cru)

    M_tip_cru = fo.sqrt(fo.square(om_cru * R) + v_cruise ** 2) / a_sound_cru
    opti.subject_to(aero_cru.thrust >= T_cru_req, scale=T_cru_req, name="cruise_thrust_min")
    if not freeze_cruise_omega:
        opti.subject_to(M_tip_cru <= max_tip_mach, scale=max_tip_mach, name="cruise_tip_mach_max")

    # Airfoil non-self-intersection and bounded max relative thickness
    x_chk = [0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.88, 0.95, 0.98]
    thick_mat = bemt_cru.cst_thickness(cst_var, x_chk=x_chk)
    opti.subject_to(thick_mat >= 0.001, scale=0.1, name="airfoil_no_crossover_le_te")

    x_tmax = np.linspace(0.15, 0.55, 31)
    thick_zone = bemt_cru.cst_thickness(cst_var, x_chk=x_tmax)
    max_thick_ks_lo = fo.ks_max(thick_zone, rho=500.0, axis=1, conservative=False)
    max_thick_ks_hi = fo.ks_max(thick_zone, rho=1000.0, axis=1, conservative=True)
    opti.subject_to(max_thick_ks_lo >= min_thick_req, scale=0.2, name="section_max_thickness_min")
    opti.subject_to(max_thick_ks_hi <= max_thick_req, scale=0.2, name="section_max_thickness_max")

    # 5-Section Cruise NeuralFoil `analysis_confidence` hard constraints
    conf_cru_exprs = []
    if nf_surr is not None:
        for s in range(len(init_kulfans)):
            kp_sym = {
                "lower_weights": cst_var[s, 0:8].reshape(8, 1),
                "upper_weights": cst_var[s, 8:16].reshape(8, 1),
                "leading_edge_weight": cst_var[s, 16],
                "TE_thickness": float(init_kulfans[s].TE_thickness),
            }
            c_c = nf_surr.get_aero_from_kulfan_parameters(kp_sym, alpha_cru_sec[s], Re_cru_sec[s])[
                "analysis_confidence"
            ]
            conf_cru_exprs.append(c_c)
            opti.subject_to(c_c >= float(min_conf_cru[s]), scale=1.0, name=f"nf_conf_cru_sec{s}")

    try:
        import casadi  # noqa: F401
        opti.solver(
            "ipopt",
            use_hessian=False,
            options={
                "max_iter": 250,
                "tol": 1e-4,
                "acceptable_tol": 5e-3,
                "acceptable_constr_viol_tol": 1e-4,
                "acceptable_iter": 15,
                "limited_memory_max_history": 25,
                "print_level": 5,
            },
        )
    except ImportError:
        opti.solver("SLSQP", options={"ftol": 1e-5, "maxiter": 200})

    t0 = time.perf_counter()
    sol = opti.solve()
    opt_s = time.perf_counter() - t0
    assert sol.success, f"Pure-cruise propeller optimization failed: {sol.message}"

    P_cru_opt = float(sol.value(aero_cru.power))
    T_cru_opt = float(sol.value(aero_cru.thrust))
    sigma_opt = float(sol.value(aero_cru.solidity))
    om_cru_opt = float(sol.value(om_cru))
    M_cru_opt = float(sol.value(M_tip_cru))
    c_opt = np.asarray(sol.value(c_ctrl)).ravel()
    tw_opt_deg = np.rad2deg(np.asarray(sol.value(tw_ctrl)).ravel())
    pc_opt_deg = float(np.rad2deg(sol.value(pc)))
    thick_all_opt = np.asarray(sol.value(thick_mat))
    thick_zone_opt = np.asarray(sol.value(thick_zone))
    thick_opt = np.max(thick_zone_opt, axis=1)
    x_tmax_loc = x_tmax[np.argmax(thick_zone_opt, axis=1)]
    min_gap_opt = float(np.min(thick_all_opt))
    eta_cru = (T_cru_opt * v_cruise) / max(P_cru_opt, 1.0)
    opt_conf_cru = np.array([float(np.asarray(sol.value(e)).ravel()[0]) for e in conf_cru_exprs]) if conf_cru_exprs else np.zeros(5)

    assert T_cru_opt >= T_cru_req - 2.0
    assert M_cru_opt <= max_tip_mach + 1e-4
    assert np.all(thick_opt >= min_thick_req - 1e-4)
    assert np.all(thick_opt <= max_thick_req + 1e-4)
    assert min_gap_opt >= 0.001 - 1e-4

    print(
        f"\n  [Propeller Cruise Optimization Converged in {opt_s:.2f} s]\n"
        f"    Cruise (4.5km, V={v_cruise:.2f} m/s, Omega*={om_cru_opt:.2f} rad/s, M_tip*={M_cru_opt:.4f} <= {max_tip_mach}, "
        f"p0*={pc_opt_deg:.2f} deg): T* = {T_cru_opt:.1f} N (>= {T_cru_req:.0f} N), P* = {P_cru_opt/1e3:.2f} kW "
        f"({(1.0 - P_cru_opt/P_cru_base)*100:.2f}% saved vs {P_cru_base/1e3:.2f} kW), "
        f"eta = {eta_cru*100:.2f}% (vs {eta_cru_base*100:.2f}%), sigma = {sigma_opt:.4f}\n"
        f"    Chord Ctrl [m]:           {np.round(c_opt, 4)} (peak at r/R={ctrl_r[int(np.argmax(c_opt))]:.2f})\n"
        f"    Twist Ctrl [deg]:         {np.round(tw_opt_deg, 2)} (tip ref = 0 deg)\n"
        f"    5-Sec max t/c (x/c):      {np.round(thick_opt, 4)} @ {np.round(x_tmax_loc, 2)} "
        f"(req [{min_thick_req}, {max_thick_req}]) | min LE-TE gap={min_gap_opt:.4f}\n"
        f"    NF Conf Cruise (init->opt): {np.round(init_conf_cru, 4)} -> {np.round(opt_conf_cru, 4)} (req >= {np.round(min_conf_cru, 3)})"
    )

    cst_opt_mat = np.asarray(sol.value(cst_var))
    te_list = [float(k.TE_thickness) for k in init_kulfans]
    _plot_cruise_results(
        ctrl_r=ctrl_r,
        init_chord=init_chord,
        opt_chord=c_opt,
        init_twist_deg=np.rad2deg(init_twist),
        opt_twist_deg=tw_opt_deg,
        sec_r_norm=sec_r_norm,
        init_cst_mat=init_cst_mat,
        opt_cst_mat=cst_opt_mat,
        te_thicknesses=te_list,
        thick_opt=thick_opt,
        x_tmax_loc=x_tmax_loc,
        opt_conf_cru=opt_conf_cru,
        min_tip_chord=min_tip_chord,
    )


if __name__ == "__main__":
    cruise_propeller_optimization()
