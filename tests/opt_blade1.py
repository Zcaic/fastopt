import time
import numpy as np

try:
    import python.fastopt as fo
except ImportError:
    import fastopt as fo
import fast_bemt as fb
import aerosandbox as asb

def export_nx(cst_opt_mat,te_list,ctrl_r ,c_opt,tw_opt_deg):
    import json 
    from pathlib import Path

    templata = '''
import NXOpen
import NXOpen.Features
import json

def modify_expr(expr_name: str, value: float, session: NXOpen.Session, unit: str = None):
    part = session.Parts.Work
    expr = part.Expressions.FindObject(expr_name)
    unit = NXOpen.Unit.Null if unit is None else part.UnitCollection.FindObject(unit)
    part.Expressions.EditExpressionWithUnits(expr, unit, f"{value:.6f}")

data_file = "%(data_file)s"

with open(data_file, "r") as fin:
    data = json.load(fin)

session = NXOpen.Session.GetSession()
part = session.Parts.Work
markid = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "modify airfoils")

for name, coords in zip(data["afs_name"], data["afs"]):
    spline = part.Features.FindObject(name)
    splineEx = part.Features.CreateStudioSplineBuilderEx(spline)
    for idx,i in enumerate(coords):
         geoCon = splineEx.ConstraintManager.FindItem(idx)
         pt = geoCon.Point
         pt.SetCoordinates(NXOpen.Point3d(-i[1]*1000.0, 0.0, i[0]*1000.0))
    
    splineEx.Evaluate()
    spline= splineEx.Commit()
    splineEx.Destroy()

session.UpdateManager.DoUpdate(markid)

markid = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "modify expr")
expr_list=data["expr"]
for expr_name,(expr_val,expr_unit)in expr_list.items():
    modify_expr(expr_name, expr_val, session, expr_unit)
session.UpdateManager.DoUpdate(markid)
'''

    datas={}
    datas["afs_name"]=["SPLINE(1)","SPLINE(4)","SPLINE(7)","SPLINE(10)","SPLINE(13)"]
    afs_coords=[]
    for i in range(5):
        af=asb.KulfanAirfoil(lower_weights=cst_opt_mat[i,:8],upper_weights=cst_opt_mat[i,8:16],leading_edge_weight=cst_opt_mat[i,16],TE_thickness=te_list[i])
        af=af.to_airfoil(n_coordinates_per_side=100)
        afs_coords.append(af.coordinates.tolist())
    datas["afs"]=afs_coords

    expr_list={}
    expr_list["p2"]=(c_opt[0]*1000.,"MilliMeter")
    expr_list["p78"]=(c_opt[1]*1000.,"MilliMeter")
    expr_list["p81"]=(c_opt[2]*1000.,"MilliMeter")
    expr_list["p5"]=(c_opt[3]*1000.,"MilliMeter")
    expr_list["p6"]=(c_opt[4]*1000.,"MilliMeter")
    expr_list["p3"]=(c_opt[5]*1000.,"MilliMeter")

    expr_list["p11"]=(tw_opt_deg[0],"MilliMeter")
    expr_list["p15"]=(tw_opt_deg[1],"MilliMeter")
    expr_list["p85"]=(tw_opt_deg[2],"MilliMeter")
    expr_list["p86"]=(tw_opt_deg[3],"MilliMeter")
    expr_list["p16"]=(tw_opt_deg[4],"MilliMeter")
    expr_list["p12"]=(tw_opt_deg[5],"MilliMeter")

    datas["expr"]=expr_list

    js_file=Path("./runtime/data.json")
    script_file=Path("./runtime/nx.py")

    with open(js_file,"w") as fout:
        json.dump(datas,fout)
    script=templata%{"data_file":js_file.resolve().as_posix()}
    script_file.write_text(script)

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


def _plot_optimization_results(
    ctrl_r, init_chord, opt_chord, init_twist_deg, opt_twist_deg,
    sec_r_norm, init_cst_mat, opt_cst_mat, te_thicknesses,
    thick_opt, x_tmax_loc, opt_conf_hov, opt_conf_cru,
    root_chord_bounds=(0.30, 0.35), min_tip_chord=0.15,
):
    """Plot baseline vs. optimized chord/twist distributions and all 5 airfoil sections."""
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

    # 1. Spanwise Chord Distribution
    ax_c = fig.add_subplot(gs[0, 0:2])
    ax_c.plot(r_ci, v_ci, "--", color="gray", lw=1.8, label="Baseline Chord")
    ax_c.plot(r_co, v_co, "-", color="#1f77b4", lw=2.4, label="Optimized Chord")
    ax_c.scatter(ctrl_r, opt_chord, color="#1f77b4", s=36, zorder=5, label="Opt Ctrl Pts")
    ax_c.axhspan(
        root_chord_bounds[0], root_chord_bounds[1], xmin=0.0, xmax=0.06,
        color="#2ca02c", alpha=0.25,
        label=f"Root [{root_chord_bounds[0]:.2f}, {root_chord_bounds[1]:.2f}]",
    )
    ax_c.axhline(min_tip_chord, color="#d62728", ls=":", lw=1.2, label=f"Min Tip ({min_tip_chord:.2f} m)")
    ax_c.set_title("Spanwise Chord Distribution c(r/R)")
    ax_c.set_xlabel("r / R")
    ax_c.set_ylabel("Chord [m]")
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
            f"Conf H/C={opt_conf_hov[s_idx]:.3f}/{opt_conf_cru[s_idx]:.3f}",
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
    ax_sum2.plot(sec_r_norm, opt_conf_hov, "s--", color="#1f77b4", lw=1.6, label="Conf Hover")
    ax_sum2.plot(sec_r_norm, opt_conf_cru, "^--", color="#d62728", lw=1.6, label="Conf Cruise")
    ax_sum2.set_ylabel("NF Confidence", color="#1f77b4")
    ax_sum2.set_ylim(0.80, 1.01)
    ax_sum.set_title("Spanwise Max t/c & NF Confidence", fontsize=9)
    lines1, labels1 = ax_sum.get_legend_handles_labels()
    lines2, labels2 = ax_sum2.get_legend_handles_labels()
    ax_sum.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="lower left")

    fig.suptitle("Tiltrotor Blade Coupled Optimization: Planform, Twist & 5-Section CST Airfoils", fontsize=12)
    plt.savefig("./runtime/Figure_1.png")
    plt.show()

def _plot_performance_comparison(
    B, R, R_hub, n_radial, ctrl_r, sec_r_norm, cfg,
    init_chord, opt_chord, init_twist_rad, opt_twist_rad,
    init_cst_mat, opt_cst_mat, te_thicknesses, init_afs,
    nf_bemt, nf_n_crit, nf_xtr_upper, nf_xtr_lower,
    rho_hov, mu_hov, a_sound_hov, omega_hov_base, om_hov_opt, ph_base_deg, ph_opt_deg,
    T_hov_req, T_hov_base, P_hov_base, T_hov_opt, P_hov_opt,
    rho_cru, mu_cru, a_sound_cru, v_cruise, omega_cru_base, om_cru_opt, pc_base_deg, pc_opt_deg,
    T_cru_req, T_cru_base, P_cru_base, T_cru_opt, P_cru_opt,
    cruise_power_limit_kw=None,
    save_path="./runtime/Figure_2_performance.png",
):
    """
    Plot 2x2 aerodynamic performance comparison (Baseline vs. Optimized):
      1. Hover collective pitch [deg] vs. Thrust [kN]
      2. Hover Thrust [kN] vs. Power [kW] (with ideal power reference)
      3. Cruise collective pitch [deg] vs. Thrust [N]
      4. Cruise Thrust [N] vs. Power [kW] (with ideal propulsive power reference)
    """
    import matplotlib.pyplot as plt

    def _build_section_airfoils(cst_mat, fallback_afs=None):
        if nf_bemt is not None and hasattr(fb.Airfoil3D, "from_neuralfoil"):
            afs = []
            for s in range(cst_mat.shape[0]):
                afs.append(
                    fb.Airfoil3D.from_neuralfoil(
                        nf_bemt,
                        np.asarray(cst_mat[s, 8:16], dtype=np.float64),
                        np.asarray(cst_mat[s, 0:8], dtype=np.float64),
                        float(cst_mat[s, 16]),
                        float(te_thicknesses[s]),
                        n_crit=float(nf_n_crit),
                        xtr_upper=float(nf_xtr_upper),
                        xtr_lower=float(nf_xtr_lower),
                        enable_compressibility=True,
                    )
                )
            return afs
        if fallback_afs is not None:
            return fallback_afs
        a_g = np.deg2rad(np.linspace(-18.0, 18.0, 49))
        re_g = np.array([1.0e5, 5.0e5, 1.5e6, 5.0e6], dtype=np.float64)
        ma_g = np.array([0.0, 0.25, 0.50, 0.70, 0.85], dtype=np.float64)
        A_d, RE_d, MA_d = np.meshgrid(np.rad2deg(a_g), re_g, ma_g, indexing="ij")
        afs = []
        for s in range(cst_mat.shape[0]):
            kaf = fb.KulfanCSTAirfoil(
                lower_weights=np.asarray(cst_mat[s, 0:8], dtype=np.float64),
                upper_weights=np.asarray(cst_mat[s, 8:16], dtype=np.float64),
                leading_edge_weight=float(cst_mat[s, 16]),
                TE_thickness=float(te_thicknesses[s]),
            )
            cl_f, cd_f = fo.bemt_bridge._default_kulfan_polar_eval(
                kaf, A_d.ravel(), RE_d.ravel(), MA_d.ravel(),
                n_crit=nf_n_crit, xtr_upper=nf_xtr_upper, xtr_lower=nf_xtr_lower,
            )
            afs.append(fb.Airfoil3D(a_g, re_g, ma_g, cl_f, cd_f))
        return afs

    def _make_solver(chord_ctrl, twist_ctrl_rad, afs):
        g = fb.RotorGeometry(
            num_blades=B,
            radius=R,
            hub_radius=R_hub,
            n_radial=n_radial,
            chord_curve=fb.BSplineCurve(ctrl_r, chord_ctrl, 3),
            twist_curve=fb.BSplineCurve(ctrl_r, twist_ctrl_rad, 3),
            airfoils=afs,
            section_r_norm=sec_r_norm,
            spin_dir=1,
        )
        return fb.BEMTSolver(g, cfg)

    def _sweep_pitch(solver, pitch_deg_grid, omega, vz, rho, mu, a_sound):
        conds = [
            fb.OperatingCondition(
                vx=0.0, vy=0.0, vz=float(vz), omega=float(omega),
                pitch_0=float(np.deg2rad(p_deg)),
                rho=float(rho), mu=float(mu), a_sound=float(a_sound),
            )
            for p_deg in pitch_deg_grid
        ]
        results = solver.solve_batch(conds, False)
        t_arr = np.array([r.thrust for r in results], dtype=np.float64)
        p_arr = np.array([r.power for r in results], dtype=np.float64)
        return t_arr, p_arr

    def _pre_stall_positive(t_arr, p_arr):
        """Extract positive-thrust pre-stall branch up to peak thrust for clean T-P polar."""
        idx_peak = int(np.argmax(t_arr))
        t_sub = t_arr[: idx_peak + 1]
        p_sub = p_arr[: idx_peak + 1]
        mask = (t_sub >= 0.0) & (p_sub >= 0.0)
        return t_sub[mask], p_sub[mask]

    solver_base = _make_solver(init_chord, init_twist_rad, _build_section_airfoils(init_cst_mat, init_afs))
    solver_opt = _make_solver(opt_chord, opt_twist_rad, _build_section_airfoils(opt_cst_mat, None))

    # 1. Hover pitch sweep (0 km ISA)
    p_hov_lo = max(-4.0, min(ph_base_deg, ph_opt_deg) - 7.0)
    p_hov_hi = min(18.0, max(ph_base_deg, ph_opt_deg) + 4.5)
    p_hov_grid = np.linspace(p_hov_lo, p_hov_hi, 45)
    T_hb, P_hb = _sweep_pitch(solver_base, p_hov_grid, omega_hov_base, 0.0, rho_hov, mu_hov, a_sound_hov)
    T_ho, P_ho = _sweep_pitch(solver_opt, p_hov_grid, om_hov_opt, 0.0, rho_hov, mu_hov, a_sound_hov)

    # 2. Cruise pitch sweep (4.5 km ISA, V = v_cruise)
    p_cb_grid = np.linspace(max(10.0, pc_base_deg - 8.0), min(68.0, pc_base_deg + 8.0), 45)
    p_co_grid = np.linspace(max(10.0, pc_opt_deg - 8.0), min(68.0, pc_opt_deg + 8.0), 45)
    T_cb, P_cb = _sweep_pitch(solver_base, p_cb_grid, omega_cru_base, v_cruise, rho_cru, mu_cru, a_sound_cru)
    T_co, P_co = _sweep_pitch(solver_opt, p_co_grid, om_cru_opt, v_cruise, rho_cru, mu_cru, a_sound_cru)

    has_diff_cru_omega = abs(om_cru_opt - omega_cru_base) > 1.0
    if has_diff_cru_omega:
        T_cb_optom, P_cb_optom = _sweep_pitch(
            solver_base, p_co_grid, om_cru_opt, v_cruise, rho_cru, mu_cru, a_sound_cru
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    ax_hp, ax_htp = axes[0, 0], axes[0, 1]
    ax_cp, ax_ctp = axes[1, 0], axes[1, 1]

    # Panel 1: Hover Collective Pitch vs. Thrust
    ax_hp.plot(p_hov_grid, T_hb / 1e3, "--", color="gray", lw=2.0, label=f"Baseline (Omega={omega_hov_base:.0f} rad/s)")
    ax_hp.plot(p_hov_grid, T_ho / 1e3, "-", color="#1f77b4", lw=2.4, label=f"Optimized (Omega={om_hov_opt:.0f} rad/s)")
    ax_hp.scatter([ph_base_deg], [T_hov_base / 1e3], color="gray", s=55, zorder=5, label="Baseline Design Pt")
    ax_hp.scatter([ph_opt_deg], [T_hov_opt / 1e3], color="#1f77b4", marker="*", s=130, zorder=6, label="Opt Design Pt")
    ax_hp.axhline(T_hov_req / 1e3, color="#d62728", ls=":", lw=1.3, label=f"Target T = {T_hov_req/1e3:.2f} kN")
    ax_hp.set_title("Hover: Collective Pitch vs. Thrust")
    ax_hp.set_xlabel("Collective Pitch p0 [deg]")
    ax_hp.set_ylabel("Thrust T [kN]")
    ax_hp.grid(True, alpha=0.3)
    ax_hp.legend(fontsize=8.5, loc="upper left")

    # Panel 2: Hover Thrust vs. Power
    T_hb_s, P_hb_s = _pre_stall_positive(T_hb, P_hb)
    T_ho_s, P_ho_s = _pre_stall_positive(T_ho, P_ho)
    A_disk = np.pi * (R ** 2)
    t_ideal_h = np.linspace(0.0, max(np.max(T_hb_s), np.max(T_ho_s)), 100)
    p_ideal_h = (t_ideal_h ** 1.5) / np.sqrt(2.0 * rho_hov * A_disk)
    ax_htp.plot(t_ideal_h / 1e3, p_ideal_h / 1e3, ":", color="#2ca02c", lw=1.6, label="Rankine-Froude Ideal")
    ax_htp.plot(T_hb_s / 1e3, P_hb_s / 1e3, "--", color="gray", lw=2.0, label="Baseline")
    ax_htp.plot(T_ho_s / 1e3, P_ho_s / 1e3, "-", color="#1f77b4", lw=2.4, label="Optimized")
    ax_htp.scatter([T_hov_base / 1e3], [P_hov_base / 1e3], color="gray", s=55, zorder=5, label="Baseline Design Pt")
    ax_htp.scatter([T_hov_opt / 1e3], [P_hov_opt / 1e3], color="#1f77b4", marker="*", s=130, zorder=6, label="Opt Design Pt")
    ax_htp.axvline(T_hov_req / 1e3, color="#d62728", ls=":", lw=1.3, label=f"Target T = {T_hov_req/1e3:.2f} kN")
    ax_htp.set_title("Hover: Thrust vs. Power")
    ax_htp.set_xlabel("Thrust T [kN]")
    ax_htp.set_ylabel("Power P [kW]")
    ax_htp.grid(True, alpha=0.3)
    ax_htp.legend(fontsize=8.5, loc="upper left")

    # Panel 3: Cruise Collective Pitch vs. Thrust
    ax_cp.plot(p_cb_grid, T_cb, "--", color="gray", lw=2.0, label=f"Baseline (Omega={omega_cru_base:.1f} rad/s)")
    if has_diff_cru_omega:
        ax_cp.plot(p_co_grid, T_cb_optom, ":", color="#7f7f7f", lw=1.6, label=f"Baseline (Omega={om_cru_opt:.1f} rad/s)")
    ax_cp.plot(p_co_grid, T_co, "-", color="#ff7f0e", lw=2.4, label=f"Optimized (Omega={om_cru_opt:.1f} rad/s)")
    ax_cp.scatter([pc_base_deg], [T_cru_base], color="gray", s=55, zorder=5, label="Baseline Design Pt")
    ax_cp.scatter([pc_opt_deg], [T_cru_opt], color="#ff7f0e", marker="*", s=130, zorder=6, label="Opt Design Pt")
    ax_cp.axhline(T_cru_req, color="#d62728", ls=":", lw=1.3, label=f"Target T = {T_cru_req:.0f} N")
    ax_cp.axhline(0.0, color="black", lw=0.7, alpha=0.4)
    ax_cp.set_title(f"Cruise (V={v_cruise*3.6:.0f} km/h): Collective Pitch vs. Thrust")
    ax_cp.set_xlabel("Collective Pitch p0 [deg]")
    ax_cp.set_ylabel("Thrust T [N]")
    ax_cp.set_ylim(-500.0, max(3000.0, 3.0 * T_cru_req))
    ax_cp.grid(True, alpha=0.3)
    ax_cp.legend(fontsize=8.5, loc="upper left")

    # Panel 4: Cruise Thrust vs. Power
    t_cru_cap = max(2500.0, 2.8 * T_cru_req)
    m_cb = (T_cb >= 0.0) & (T_cb <= t_cru_cap) & (P_cb >= 0.0)
    m_co = (T_co >= 0.0) & (T_co <= t_cru_cap) & (P_co >= 0.0)
    t_ideal_c = np.linspace(0.0, t_cru_cap, 100)
    ax_ctp.plot(t_ideal_c, (t_ideal_c * v_cruise) / 1e3, ":", color="#2ca02c", lw=1.6, label="Ideal Propulsive Power (T*V)")
    ax_ctp.plot(T_cb[m_cb], P_cb[m_cb] / 1e3, "--", color="gray", lw=2.0, label=f"Baseline (Omega={omega_cru_base:.1f} rad/s)")
    if has_diff_cru_omega:
        m_cbo = (T_cb_optom >= 0.0) & (T_cb_optom <= t_cru_cap) & (P_cb_optom >= 0.0)
        ax_ctp.plot(T_cb_optom[m_cbo], P_cb_optom[m_cbo] / 1e3, ":", color="#7f7f7f", lw=1.6, label=f"Baseline (Omega={om_cru_opt:.1f} rad/s)")
    ax_ctp.plot(T_co[m_co], P_co[m_co] / 1e3, "-", color="#ff7f0e", lw=2.4, label=f"Optimized (Omega={om_cru_opt:.1f} rad/s)")
    ax_ctp.scatter([T_cru_base], [P_cru_base / 1e3], color="gray", s=55, zorder=5, label="Baseline Design Pt")
    ax_ctp.scatter([T_cru_opt], [P_cru_opt / 1e3], color="#ff7f0e", marker="*", s=130, zorder=6, label="Opt Design Pt")
    ax_ctp.axvline(T_cru_req, color="#d62728", ls=":", lw=1.3, label=f"Target T = {T_cru_req:.0f} N")
    if cruise_power_limit_kw is not None:
        ax_ctp.axhline(cruise_power_limit_kw, color="#9467bd", ls="-.", lw=1.2, label=f"Power Cap = {cruise_power_limit_kw:.0f} kW")
    ax_ctp.set_title(f"Cruise (V={v_cruise*3.6:.0f} km/h): Thrust vs. Power")
    ax_ctp.set_xlabel("Thrust T [N]")
    ax_ctp.set_ylabel("Power P [kW]")
    ax_ctp.grid(True, alpha=0.3)
    ax_ctp.legend(fontsize=8.5, loc="upper left")

    fig.suptitle(
        f"Tiltrotor Aerodynamic Performance Comparison (n_crit={nf_n_crit:.1f}, xtr={nf_xtr_upper:.2f}/{nf_xtr_lower:.2f})",
        fontsize=12.5,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()



def tiltrotor_optimization():
    """
    Engineering 5-Section CST Tiltrotor Blade Optimization:
      - Radius R = 2.1 m, Blades B = 3, Hub radius R_hub = 0.1 * R = 0.21 m
      - 5 airfoil sections at r/R = [0.1, 0.2, 0.5, 0.75, 1.0]
      - Max relative thickness over 15%c..55%c (via smooth ks_max) >= [0.28, 0.26, 0.18, 0.12, 0.08]
      - Densified LE-to-TE non-self-intersection check (11 chordwise stations from 2%c to 98%c)
      - 5-section Hover + Cruise NeuralFoil `analysis_confidence` hard constraints
      - First 3 chord control points free, monotonic non-increasing outer-span taper c[2] >= c[3] >= c[4] >= c[5]
      - Parameterized hover/cruise thrust targets (T_hov_req = T_hov_nominal * hover_thrust_factor, T_cru_req)
      - Supports hover-min (with P_cruise cap), cruise-min (with P_hover cap), and weighted modes
    """
    # =========================================================================
    # User-Configurable Mission & Design Parameters
    # =========================================================================
    R = 2.1
    R_hub = 0.1 * R
    B = 3
    A_disk = np.pi * (R ** 2)
    max_tip_mach = 0.70

    # Thrust targets [N] (use hover_thrust_factor > 1.0 to build in CFD & overload stall margin)
    T_hov_nominal = 15000.0
    hover_thrust_factor = 1.2
    T_hov_req = T_hov_nominal * hover_thrust_factor
    T_cru_req = 850.0

    # Planform chord bounds [m] & per-section relative CST deformation margin
    root_chord_lb, root_chord_ub = 0.30, 0.35
    min_tip_chord = 0.15
    max_mid_chord = 0.45
    cst_rel_margin = 0.2  # +/-50% of each section characteristic CST weight scale

    # Atmosphere & operating conditions: Hover (0 km ISA) and Cruise (4.5 km ISA, 400 km/h)
    rho_hov, mu_hov, a_sound_hov = 1.225, 1.789e-5, 340.294
    rho_cru, mu_cru, a_sound_cru = 0.777, 1.64477e-5, 322.573
    omega_hov_base = 110.0
    omega_cru_base = 90.0  # Tip Mach = 0.680 (< 0.70) at 400 km/h and 4.5 km ISA
    v_cruise = 400.0 / 3.6
    freeze_hover_omega = True
    freeze_cruise_omega = False

    # Optimization objective mode switch:
    #   "hover"      : minimize hover power subject to P_cruise <= cruise_power_limit_kw
    #   "cruise_min" : minimize cruise power subject to P_hover <= hover_power_limit_kw
    #   "weighted"   : minimize w_hover * P_hover + w_cruise * P_cruise
    opt_mode = "hover"
    cruise_power_limit_kw = 115.0
    hover_power_limit_kw = 580.0  # Ideal hover power at 19,500 N is 467.4 kW
    # Optional hover blade-loading cap CT/sigma at T_hov_req (set to None to disable; e.g., 0.175 at 1.2x overload -> 0.146 nominal)
    max_hover_ct_sigma = None
    w_hover, w_cruise = 0.5, 0.5

    sec_r_norm = np.array([0.1, 0.2, 0.5, 0.75, 1.0], dtype=np.float64)
    min_thick_req = np.array([0.28, 0.26, 0.18, 0.12, 0.08], dtype=np.float64)
    max_thick_req = np.array([0.305, 0.275, 0.195, 0.130, 0.090], dtype=np.float64)
    # Pitching moment CM envelope per section (tight |CM| <= 0.05 on outer sections prevents high-camber banana airfoils)
    enforce_cm_bounds = True
    cm_min_req = np.array([-0.15, -0.15, -0.08, -0.05, -0.05], dtype=np.float64)
    cm_max_req = np.array([ 0.05,  0.05,  0.05,  0.05,  0.05], dtype=np.float64)

    # Spanwise twist smoothness (dr-normalized curvature penalty) & convexity (d^2(theta)/dr^2 >= 0, no S-inflection)
    twist_curv_weight = 0.15
    enforce_twist_convexity = True

    # NeuralFoil boundary-layer transition settings:
    #   Forced turbulent near LE (n_crit=4.0, xtr=0.05) matches full-scale XV-15 OARF / STAR-CCM+ SST parity.
    #   Set (n_crit=9.0, xtr_upper=1.0, xtr_lower=1.0) for free-transition laminar baseline.
    nf_n_crit = 4.0
    nf_xtr_upper = 0.05
    nf_xtr_lower = 0.05

    print(
        f"\n--- Engineering 5-Section CST Tiltrotor Optimization "
        f"(R={R}m, 0km Hover T>={T_hov_req:.0f}N [{hover_thrust_factor:.2f}x] + "
        f"4.5km {v_cruise*3.6:.0f}km/h Cruise T>={T_cru_req:.0f}N | "
        f"NF n_crit={nf_n_crit:.1f}, xtr={nf_xtr_upper:.2f}/{nf_xtr_lower:.2f}) ---"
    )

    # Construct 5 initial Kulfan CST airfoils
    init_kulfans = [
        # fb.KulfanCSTAirfoil.from_aerosandbox(asb.Airfoil(coordinates="./tests/afs/DU-97-W-300.dat").to_kulfan_airfoil().set_TE_thickness(2e-3)),
        # fb.KulfanCSTAirfoil.from_aerosandbox(asb.Airfoil(coordinates="./tests/afs/DU-91-W2-250.dat").to_kulfan_airfoil().set_TE_thickness(2e-3)),
        fb.KulfanCSTAirfoil.from_aerosandbox(asb.Airfoil(coordinates="./tests/afs/naca64-228.dat").to_kulfan_airfoil().set_TE_thickness(2e-3)),
        fb.KulfanCSTAirfoil.from_aerosandbox(asb.Airfoil(coordinates="./tests/afs/naca64-226.dat").to_kulfan_airfoil().set_TE_thickness(2e-3)),
        fb.KulfanCSTAirfoil.from_aerosandbox(asb.Airfoil(coordinates="./tests/afs/naca64-218.dat").to_kulfan_airfoil().set_TE_thickness(2e-3)),
        fb.KulfanCSTAirfoil.from_aerosandbox(asb.Airfoil(coordinates="./tests/afs/naca64-212.dat").to_kulfan_airfoil().set_TE_thickness(2e-3)),
        fb.KulfanCSTAirfoil.from_aerosandbox(asb.Airfoil(coordinates="./tests/afs/naca64-108.dat").to_kulfan_airfoil().set_TE_thickness(2e-3)),
    ]
    init_cst_mat = np.vstack([k.to_vector(include_te=False) for k in init_kulfans])  # (5, 17)

    # Build initial 5-section airfoils using the exact same native C++ NeuralFoil backend
    # (with compressibility, Mach stall, and forced transition) as the coupled optimizer
    _, nf_bemt_init = fo.bemt_bridge._resolve_cpp_neuralfoil(fb, "auto", use_float32=False)
    init_afs = []
    if nf_bemt_init is not None and hasattr(fb.Airfoil3D, "from_neuralfoil"):
        for kaf in init_kulfans:
            init_afs.append(
                fb.Airfoil3D.from_neuralfoil(
                    nf_bemt_init,
                    np.asarray(kaf.upper_weights, dtype=np.float64),
                    np.asarray(kaf.lower_weights, dtype=np.float64),
                    float(kaf.leading_edge_weight),
                    float(kaf.TE_thickness),
                    n_crit=nf_n_crit,
                    xtr_upper=nf_xtr_upper,
                    xtr_lower=nf_xtr_lower,
                    enable_compressibility=True,
                )
            )
    else:
        a_g = np.deg2rad(np.linspace(-18.0, 18.0, 49))
        re_g = np.array([1.0e5, 5.0e5, 1.5e6, 5.0e6], dtype=np.float64)
        ma_g = np.array([0.0, 0.25, 0.50, 0.70, 0.85], dtype=np.float64)
        A_d, RE_d, MA_d = np.meshgrid(np.rad2deg(a_g), re_g, ma_g, indexing="ij")
        for kaf in init_kulfans:
            cl_f, cd_f = fo.bemt_bridge._default_kulfan_polar_eval(
                kaf, A_d.ravel(), RE_d.ravel(), MA_d.ravel(),
                n_crit=nf_n_crit, xtr_upper=nf_xtr_upper, xtr_lower=nf_xtr_lower,
            )
            init_afs.append(fb.Airfoil3D(a_g, re_g, ma_g, cl_f, cd_f))

    ctrl_r = np.array([0.10, 0.22, 0.42, 0.65, 0.85, 1.00], dtype=np.float64)
    init_chord = np.array([0.340, 0.320, 0.300, 0.275, 0.240, 0.200], dtype=np.float64)
    # Express geometric twist relative to blade tip (tip twist anchored at 0 deg to eliminate pitch/twist redundancy)
    init_twist = np.deg2rad([24.0, 19.0, 11.0, 6.0, 2.5, 0.0])

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

    cond_hov_base = fb.OperatingCondition(
        vx=0.0, vy=0.0, vz=0.0, omega=omega_hov_base, pitch_0=np.deg2rad(10.0),
        rho=rho_hov, mu=mu_hov, a_sound=a_sound_hov,
    )
    cond_cru_base = fb.OperatingCondition(
        vx=0.0, vy=0.0, vz=v_cruise, omega=omega_cru_base, pitch_0=np.deg2rad(45.0),
        rho=rho_cru, mu=mu_cru, a_sound=a_sound_cru,
    )

    bemt_hov = fo.FastBEMTAdapter(
        geom,
        cfg,
        cond_hov_base,
        section_airfoils_kulfan=init_kulfans,
        neuralfoil_use_float32=False,
        n_crit=nf_n_crit,
        xtr_upper=nf_xtr_upper,
        xtr_lower=nf_xtr_lower,
        expose_section_aero=True,
        name="tilt_hov",
    )
    bemt_cru = fo.FastBEMTAdapter(
        geom,
        cfg,
        cond_cru_base,
        section_airfoils_kulfan=init_kulfans,
        neuralfoil_use_float32=False,
        n_crit=nf_n_crit,
        xtr_upper=nf_xtr_upper,
        xtr_lower=nf_xtr_lower,
        expose_section_aero=True,
        name="tilt_cru",
    )

    # 1. Trim baseline collective pitches for Hover (T_hov_req) and Cruise (T_cru_req)
    # Keep hover baseline trim on the pre-stall branch (init=8 deg, ub=16 deg, tiny power tie-breaker)
    opti_bh = fo.Opti()
    ph_b = opti_bh.variable(init=np.deg2rad(8.0), lb=np.deg2rad(-5.0), ub=np.deg2rad(16.0))
    ah_b = bemt_hov(pitch_0=ph_b, omega=omega_hov_base)
    opti_bh.minimize(fo.square((ah_b.thrust - T_hov_req) / T_hov_req) + 1e-6 * (ah_b.power * 1e-6))
    opti_bh.solver(fo.solvers.scipy.SLSQP(ftol=1e-10, maxiter=40))
    sol_bh = opti_bh.solve()

    opti_bc = fo.Opti()
    pc_b = opti_bc.variable(init=np.deg2rad(45.0), lb=np.deg2rad(15.0), ub=np.deg2rad(65.0))
    ac_b = bemt_cru(pitch_0=pc_b, omega=omega_cru_base, vz=v_cruise)
    opti_bc.minimize(fo.square((ac_b.thrust - T_cru_req) / T_cru_req))
    opti_bc.solver(fo.solvers.scipy.SLSQP(ftol=1e-10, maxiter=40))
    sol_bc = opti_bc.solve()

    # Extract actual trimmed baseline 5-section (alpha_deg, Re) from BEMT flowfield (zero manual alpha_hov_sec)
    S_sec = bemt_hov.build_section_interp_matrix(sec_r_norm)
    res_bh_0 = bemt_hov.solve_condition(pitch_0=float(sol_bh.value(ph_b)), omega=omega_hov_base, compute_gradients=False)
    res_bc_0 = bemt_cru.solve_condition(pitch_0=float(sol_bc.value(pc_b)), omega=omega_cru_base, compute_gradients=False)
    alpha_hov_base = np.rad2deg(S_sec @ np.mean(np.asarray(res_bh_0.alpha, dtype=np.float64), axis=1))
    alpha_cru_base = np.rad2deg(S_sec @ np.mean(np.asarray(res_bc_0.alpha, dtype=np.float64), axis=1))
    re_hov_base = S_sec @ np.mean(np.asarray(res_bh_0.re, dtype=np.float64), axis=1)
    re_cru_base = S_sec @ np.mean(np.asarray(res_bc_0.re, dtype=np.float64), axis=1)

    nf_surr = getattr(bemt_hov, "_nf_surrogate", None)
    min_conf_target = np.array([0.85, 0.85, 0.85, 0.90, 0.90], dtype=np.float64)
    init_conf_hov = np.zeros(5, dtype=np.float64)
    init_conf_cru = np.zeros(5, dtype=np.float64)
    init_cm_hov = np.zeros(5, dtype=np.float64)
    init_cm_cru = np.zeros(5, dtype=np.float64)
    if nf_surr is not None:
        for s, kaf in enumerate(init_kulfans):
            ah_0 = nf_surr.get_aero_from_kulfan_parameters(
                kaf.kulfan_parameters, alpha_hov_base[s], re_hov_base[s],
                n_crit=nf_n_crit, xtr_upper=nf_xtr_upper, xtr_lower=nf_xtr_lower,
            )
            ac_0 = nf_surr.get_aero_from_kulfan_parameters(
                kaf.kulfan_parameters, alpha_cru_base[s], re_cru_base[s],
                n_crit=nf_n_crit, xtr_upper=nf_xtr_upper, xtr_lower=nf_xtr_lower,
            )
            init_conf_hov[s] = float(ah_0["analysis_confidence"][0])
            init_conf_cru[s] = float(ac_0["analysis_confidence"][0])
            init_cm_hov[s] = float(ah_0["CM"][0])
            init_cm_cru[s] = float(ac_0["CM"][0])
        # Guard against infeasible initial thick-root airfoils while enforcing target on all well-resolved sections
        min_conf_hov = np.minimum(min_conf_target, np.maximum(init_conf_hov - 0.02, 0.75))
        min_conf_cru = np.minimum(min_conf_target, np.maximum(init_conf_cru - 0.02, 0.75))
        cm_min_eff = np.minimum(cm_min_req, np.minimum(init_cm_hov, init_cm_cru) - 0.01)
        cm_max_eff = np.maximum(cm_max_req, np.maximum(init_cm_hov, init_cm_cru) + 0.01)
        print(
            f"  [Baseline Sec Alpha deg] Hover:  {np.round(alpha_hov_base, 2)} | Cruise: {np.round(alpha_cru_base, 2)}\n"
            f"  [Baseline NF Confidence] Hover:  {np.round(init_conf_hov, 4)} (req >= {np.round(min_conf_hov, 3)})\n"
            f"                           Cruise: {np.round(init_conf_cru, 4)} (req >= {np.round(min_conf_cru, 3)})\n"
            f"  [Baseline NF CM]         Hover:  {np.round(init_cm_hov, 4)} | Cruise: {np.round(init_cm_cru, 4)} "
            f"(req [{np.round(cm_min_eff, 3)}, {np.round(cm_max_eff, 3)}])"
        )
    else:
        min_conf_hov = min_conf_target.copy()
        min_conf_cru = min_conf_target.copy()
        cm_min_eff = cm_min_req.copy()
        cm_max_eff = cm_max_req.copy()

    T_hov_base = float(sol_bh.value(ah_b.thrust))
    P_hov_base = float(sol_bh.value(ah_b.power))
    T_cru_base = float(sol_bc.value(ac_b.thrust))
    P_cru_base = float(sol_bc.value(ac_b.power))
    P_ideal_base = (max(T_hov_base, 0.0) ** 1.5) / np.sqrt(2.0 * rho_hov * A_disk)
    FM_hov_base = P_ideal_base / max(P_hov_base, 1.0)
    eta_cru_base = (T_cru_base * v_cruise) / max(P_cru_base, 1.0)
    M_hov_base = (omega_hov_base * R) / a_sound_hov
    M_cru_base = np.hypot(omega_cru_base * R, v_cruise) / a_sound_cru
    print(
        f"  [Baseline 5-Sec Rotor] 0km Hover (Omega={omega_hov_base:.1f} rad/s, M_tip={M_hov_base:.3f}, "
        f"T={T_hov_base:.1f} N): P_hov = {P_hov_base/1e3:.2f} kW, FM = {FM_hov_base:.4f} | "
        f"4.5km Cruise (V={v_cruise:.2f} m/s, Omega={omega_cru_base:.1f} rad/s, M_tip={M_cru_base:.3f}, "
        f"T={T_cru_base:.1f} N): P_cru = {P_cru_base/1e3:.2f} kW, eta = {eta_cru_base*100:.2f}%"
    )

    # 2. Full Coupled Optimization: Chord (6) + Twist (6) + 5-Section CST (85) + Pitches (2) + Omegas (2)
    opti = fo.Opti()
    c_lb = np.array([root_chord_lb, min_tip_chord, min_tip_chord, min_tip_chord, min_tip_chord, min_tip_chord], dtype=np.float64)
    c_ub = np.array([root_chord_ub, max_mid_chord, max_mid_chord, max_mid_chord, max_mid_chord, max_mid_chord], dtype=np.float64)
    c_ctrl = opti.variable(shape=(6, 1), init=init_chord, lb=c_lb, ub=c_ub, scale=0.25, name="chord_ctrl")

    tw_ctrl = opti.variable(
        shape=(6, 1), init=init_twist, lb=np.deg2rad(-20.0), ub=np.deg2rad(42.0), scale=0.3, name="twist_ctrl"
    )
    # Per-section relative CST bounds scaled by each section characteristic Bernstein weight magnitude
    sec_cst_scale = np.maximum(np.max(np.abs(init_cst_mat[:, :16]), axis=1, keepdims=True), 0.05)
    cst_delta = cst_rel_margin * sec_cst_scale
    cst_lb = init_cst_mat - cst_delta
    cst_ub = init_cst_mat + cst_delta
    cst_lb[:, 16] = np.maximum(cst_lb[:, 16], -0.25)
    cst_ub[:, 16] = np.minimum(cst_ub[:, 16], 0.25)
    # Leading-edge sign guard: keep upper_weights[0] > 0 and lower_weights[0] < 0 so nose opens cleanly around (0, 0)
    cst_lb[:, 8] = np.maximum(cst_lb[:, 8], np.minimum(0.03, init_cst_mat[:, 8]))
    cst_ub[:, 0] = np.minimum(cst_ub[:, 0], np.maximum(-0.03, init_cst_mat[:, 0]))
    cst_var = opti.variable(
        shape=(5, 17), init=init_cst_mat, lb=cst_lb, ub=cst_ub, scale=0.2, name="cst_params"
    )
    ph = opti.variable(
        init=float(sol_bh.value(ph_b)), lb=np.deg2rad(-15.0), ub=np.deg2rad(28.0), scale=0.1, name="pitch_hov"
    )
    pc = opti.variable(
        init=float(sol_bc.value(pc_b)), lb=np.deg2rad(15.0), ub=np.deg2rad(68.0), scale=0.2, name="pitch_cru"
    )
    om_hov_n = opti.variable(
        init=omega_hov_base / 100.0, lb=1.00, ub=1.15, name="omega_hov_norm", freeze=freeze_hover_omega
    )
    om_cru_n = opti.variable(
        init=omega_cru_base / 100.0, lb=0.50, ub=1.15, name="omega_cru_norm", freeze=freeze_cruise_omega
    )
    om_hov = 100.0 * om_hov_n
    om_cru = 100.0 * om_cru_n

    aero_hov = bemt_hov(chord_ctrl=c_ctrl, twist_ctrl=tw_ctrl, cst_params=cst_var, pitch_0=ph, omega=om_hov)
    aero_cru = bemt_cru(
        chord_ctrl=c_ctrl, twist_ctrl=tw_ctrl, cst_params=cst_var, pitch_0=pc, omega=om_cru, vz=v_cruise
    )

    # Planform & twist constraints (chord_taper=False allows non-monotone mid-span widening)
    reg_geom = bemt_hov.add_planform_constraints(
        opti,
        chord_ctrl=c_ctrl,
        twist_ctrl=tw_ctrl,
        root_chord=None,
        min_tip_chord=min_tip_chord,
        chord_taper=False,
        twist_washout=True,
        smoothness_weight_chord=200.0,
        smoothness_weight_twist=0.0,
    )
    opti.subject_to(tw_ctrl[-1] == 0.0, scale=0.1, name="tip_twist_ref")

    # Non-uniform dr-normalized spanwise twist curvature d^2(theta)/dr^2
    dr_ctrl = np.diff(ctrl_r)
    dr_mid = 0.5 * (dr_ctrl[:-1] + dr_ctrl[1:])
    tw_slopes = (tw_ctrl[1:] - tw_ctrl[:-1]) / dr_ctrl.reshape(-1, 1)
    tw_curv = (tw_slopes[1:] - tw_slopes[:-1]) / dr_mid.reshape(-1, 1)
    if twist_curv_weight > 0.0:
        reg_geom = reg_geom + twist_curv_weight * fo.sumsqr(tw_curv)
    if enforce_twist_convexity:
        # Monotonically decreasing washout rate from root to tip (concave-up, eliminates outer-span S-inflection)
        opti.subject_to(tw_curv >= 0.0, scale=1.0, name="twist_convexity")
    # Allow first 3 chord control points (r/R = 0.10, 0.22, 0.42) to bulge freely,
    # then enforce monotonic non-increasing outer-span taper from ctrl[2] to tip: c[2] >= c[3] >= c[4] >= c[5]
    opti.subject_to(c_ctrl[3:6] <= c_ctrl[2:5], scale=0.1, name="outer_chord_monotone_taper")

    # Internal CST Bernstein coefficient smoothness (2nd difference on lower [0:8] and upper [8:16] surfaces)
    d2_cst_lo = cst_var[:, 2:8] - 2.0 * cst_var[:, 1:7] + cst_var[:, 0:6]
    d2_cst_up = cst_var[:, 10:16] - 2.0 * cst_var[:, 9:15] + cst_var[:, 8:14]
    reg_cst_smooth = 10.0 * (fo.sumsqr(d2_cst_lo) + fo.sumsqr(d2_cst_up))
    reg_cst_anchor = 0.0 * fo.sumsqr(cst_var - init_cst_mat)
    reg_cst = reg_cst_smooth + reg_cst_anchor

    obj_scale_hov = max(P_hov_base * 1e-3, 300.0)
    if opt_mode == "hover":
        opti.minimize(aero_hov.power * 1e-3 + reg_geom + reg_cst, scale=obj_scale_hov)
    elif opt_mode == "cruise_min":
        opti.minimize(aero_cru.power * 1e-3 + reg_geom + reg_cst, scale=100.0)
    elif opt_mode == "weighted":
        opti.minimize(
            w_hover * (aero_hov.power * 1e-3) + w_cruise * (aero_cru.power * 1e-3) + reg_geom + reg_cst,
            scale=w_hover * obj_scale_hov + w_cruise * 100.0,
        )
    else:
        raise ValueError(f"Unsupported opt_mode: {opt_mode}")

    M_tip_hov = (om_hov * R) / a_sound_hov
    M_tip_cru = fo.sqrt(fo.square(om_cru * R) + v_cruise ** 2) / a_sound_cru

    opti.subject_to(aero_hov.thrust >= T_hov_req, scale=T_hov_req, name="hover_thrust_min")
    opti.subject_to(aero_cru.thrust >= T_cru_req, scale=T_cru_req, name="cruise_thrust_min")
    if max_hover_ct_sigma is not None:
        opti.subject_to(
            aero_hov.ct / aero_hov.solidity <= float(max_hover_ct_sigma),
            scale=float(max_hover_ct_sigma),
            name="hover_ct_sigma_max",
        )
    if opt_mode == "hover":
        opti.subject_to(
            aero_cru.power <= cruise_power_limit_kw * 1e3,
            scale=cruise_power_limit_kw * 1e3,
            name="cruise_power_max",
        )
    if opt_mode == "cruise_min":
        opti.subject_to(
            aero_hov.power <= hover_power_limit_kw * 1e3,
            scale=hover_power_limit_kw * 1e3,
            name="hover_power_max",
        )
    if not (freeze_hover_omega and freeze_cruise_omega):
        opti.subject_to(om_cru >= 0.5 * om_hov, scale=50.0, name="cruise_omega_ratio")
    if not freeze_hover_omega:
        opti.subject_to(M_tip_hov <= max_tip_mach, scale=max_tip_mach, name="hover_tip_mach_max")
    if not freeze_cruise_omega:
        opti.subject_to(M_tip_cru <= max_tip_mach, scale=max_tip_mach, name="cruise_tip_mach_max")

    # Densified LE-to-TE non-self-intersection check (2%c to 98%c prevents LE pinch & TE fishtail)
    x_chk = [0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.88, 0.95, 0.98]
    thick_mat = bemt_hov.cst_thickness(cst_var, x_chk=x_chk)
    opti.subject_to(thick_mat >= 0.001, scale=0.1, name="airfoil_no_crossover_le_te")

    # Leading-edge nose symmetry constraint: keeps nose curvature balanced around (0, 0) for clean 3D CAD/NX lofting
    le_asym = cst_var[:, 8] + cst_var[:, 0]
    le_sym_tol = np.maximum(0.05, np.abs(init_cst_mat[:, 8] + init_cst_mat[:, 0]) + 0.01)
    opti.subject_to(le_asym <= le_sym_tol.reshape(-1, 1), scale=0.1, name="airfoil_le_sym_ub")
    opti.subject_to(le_asym >= -le_sym_tol.reshape(-1, 1), scale=0.1, name="airfoil_le_sym_lb")

    # Split thickness envelope for structural D-spar & 3D CFD fidelity:
    #   - Lower bound in spar box x in [22%c, 38%c]: anchors structural depth & prevents aft-loaded laminar bulge
    #   - Upper bound over full chord x in [5%c, 90%c]: forbids hidden thickness bulge anywhere aft of 38%c
    x_spar = np.linspace(0.22, 0.38, 17)
    x_full = np.linspace(0.05, 0.90, 41)
    thick_spar = bemt_hov.cst_thickness(cst_var, x_chk=x_spar)
    thick_full = bemt_hov.cst_thickness(cst_var, x_chk=x_full)
    max_thick_ks_lo = fo.ks_max(thick_spar, rho=500.0, axis=1, conservative=False)
    max_thick_ks_hi = fo.ks_max(thick_full, rho=1000.0, axis=1, conservative=True)
    opti.subject_to(max_thick_ks_lo >= min_thick_req, scale=0.2, name="section_spar_thickness_min")
    opti.subject_to(max_thick_ks_hi <= max_thick_req, scale=0.2, name="section_full_chord_thickness_max")

    # 5-Section Hover + Cruise NeuralFoil `analysis_confidence` & pitching-moment `CM` constraints (fused C++ MLPNode AD)
    conf_hov_exprs = []
    conf_cru_exprs = []
    cm_hov_exprs = []
    cm_cru_exprs = []
    if nf_surr is not None:
        for s in range(len(init_kulfans)):
            kp_sym = {
                "lower_weights": cst_var[s, 0:8].reshape(8, 1),
                "upper_weights": cst_var[s, 8:16].reshape(8, 1),
                "leading_edge_weight": cst_var[s, 16],
                "TE_thickness": float(init_kulfans[s].TE_thickness),
            }
            ah_sym = nf_surr.get_aero_from_kulfan_parameters(
                kp_sym, aero_hov.section_alpha_deg[s], aero_hov.section_re[s],
                n_crit=nf_n_crit, xtr_upper=nf_xtr_upper, xtr_lower=nf_xtr_lower,
            )
            ac_sym = nf_surr.get_aero_from_kulfan_parameters(
                kp_sym, aero_cru.section_alpha_deg[s], aero_cru.section_re[s],
                n_crit=nf_n_crit, xtr_upper=nf_xtr_upper, xtr_lower=nf_xtr_lower,
            )
            c_h, cm_h = ah_sym["analysis_confidence"], ah_sym["CM"]
            c_c, cm_c = ac_sym["analysis_confidence"], ac_sym["CM"]
            conf_hov_exprs.append(c_h)
            conf_cru_exprs.append(c_c)
            cm_hov_exprs.append(cm_h)
            cm_cru_exprs.append(cm_c)
            opti.subject_to(c_h >= float(min_conf_hov[s]), scale=1.0, name=f"nf_conf_hov_sec{s}")
            opti.subject_to(c_c >= float(min_conf_cru[s]), scale=1.0, name=f"nf_conf_cru_sec{s}")
            if enforce_cm_bounds:
                opti.subject_to(cm_h >= float(cm_min_eff[s]), scale=0.1, name=f"nf_cm_hov_min_sec{s}")
                opti.subject_to(cm_h <= float(cm_max_eff[s]), scale=0.1, name=f"nf_cm_hov_max_sec{s}")
                opti.subject_to(cm_c >= float(cm_min_eff[s]), scale=0.1, name=f"nf_cm_cru_min_sec{s}")
                opti.subject_to(cm_c <= float(cm_max_eff[s]), scale=0.1, name=f"nf_cm_cru_max_sec{s}")

    try:
        import casadi  # noqa: F401
        opti.solver(
            fo.solvers.ipopt.IPOPT.lbfgs(
                max_iter=400,
                tol=1e-4,
                acceptable_tol=1e-3,
                acceptable_constr_viol_tol=5e-4,
                acceptable_iter=25,
                limited_memory_max_history=30,
                mu_strategy="monotone",
                print_level=5,
            )
        )
    except ImportError:
        opti.solver(fo.solvers.scipy.SLSQP(ftol=1e-5, maxiter=200))

    t0 = time.perf_counter()
    sol = opti.solve()
    opt_s = time.perf_counter() - t0
    assert sol.success, f"5-section CST tiltrotor optimization failed: {sol.message}"

    P_hov_opt = float(sol.value(aero_hov.power))
    T_hov_opt = float(sol.value(aero_hov.thrust))
    P_cru_opt = float(sol.value(aero_cru.power))
    T_cru_opt = float(sol.value(aero_cru.thrust))
    sigma_opt = float(sol.value(aero_hov.solidity))
    om_hov_opt = float(sol.value(om_hov))
    om_cru_opt = float(sol.value(om_cru))
    M_hov_opt = float(sol.value(M_tip_hov))
    M_cru_opt = float(sol.value(M_tip_cru))
    c_opt = np.asarray(sol.value(c_ctrl)).ravel()
    tw_opt_deg = np.rad2deg(np.asarray(sol.value(tw_ctrl)).ravel())
    ph_opt_deg = float(np.rad2deg(sol.value(ph)))
    pc_opt_deg = float(np.rad2deg(sol.value(pc)))
    thick_all_opt = np.asarray(sol.value(thick_mat))
    thick_spar_opt = np.max(np.asarray(sol.value(thick_spar)), axis=1)
    thick_full_opt = np.asarray(sol.value(thick_full))
    thick_opt = np.max(thick_full_opt, axis=1)
    x_tmax_loc = x_full[np.argmax(thick_full_opt, axis=1)]
    min_gap_opt = float(np.min(thick_all_opt))
    P_ideal_opt = (max(T_hov_opt, 0.0) ** 1.5) / np.sqrt(2.0 * rho_hov * A_disk)
    FM_hov_opt = P_ideal_opt / max(P_hov_opt, 1.0)
    ct_hov_opt = T_hov_opt / (rho_hov * A_disk * ((om_hov_opt * R) ** 2))
    eta_cru = (T_cru_opt * v_cruise) / max(P_cru_opt, 1.0)

    opt_alpha_hov = np.asarray(sol.value(aero_hov.section_alpha_deg)).ravel()
    opt_alpha_cru = np.asarray(sol.value(aero_cru.section_alpha_deg)).ravel()
    opt_conf_hov = np.array([float(np.asarray(sol.value(e)).ravel()[0]) for e in conf_hov_exprs]) if conf_hov_exprs else np.zeros(5)
    opt_conf_cru = np.array([float(np.asarray(sol.value(e)).ravel()[0]) for e in conf_cru_exprs]) if conf_cru_exprs else np.zeros(5)
    opt_cm_hov = np.array([float(np.asarray(sol.value(e)).ravel()[0]) for e in cm_hov_exprs]) if cm_hov_exprs else np.zeros(5)
    opt_cm_cru = np.array([float(np.asarray(sol.value(e)).ravel()[0]) for e in cm_cru_exprs]) if cm_cru_exprs else np.zeros(5)

    assert root_chord_lb - 1e-4 <= c_opt[0] <= root_chord_ub + 1e-4
    assert c_opt[-1] >= min_tip_chord - 1e-4
    assert np.all(c_opt[3:] <= c_opt[2:-1] + 1e-4)
    assert M_hov_opt <= max_tip_mach + 1e-4
    assert M_cru_opt <= max_tip_mach + 1e-4
    assert T_hov_opt >= T_hov_req - 5.0
    assert T_cru_opt >= T_cru_req - 1.0
    if max_hover_ct_sigma is not None:
        assert (ct_hov_opt / sigma_opt) <= float(max_hover_ct_sigma) + 1e-3
    if opt_mode == "hover":
        assert P_cru_opt <= cruise_power_limit_kw * 1e3 + 50.0
    if opt_mode == "cruise_min":
        assert P_hov_opt <= hover_power_limit_kw * 1e3 + 50.0
    assert np.all(thick_spar_opt >= min_thick_req - 1e-4)
    assert np.all(thick_opt <= max_thick_req + 1e-4)
    assert min_gap_opt >= 0.001 - 1e-4
    if enforce_twist_convexity:
        tw_slopes_opt = np.diff(np.deg2rad(tw_opt_deg)) / dr_ctrl
        tw_curv_opt = np.diff(tw_slopes_opt) / dr_mid
        assert np.all(tw_curv_opt >= -1e-3), f"Twist convexity violated: {tw_curv_opt}"
    if enforce_cm_bounds and cm_hov_exprs:
        assert np.all(opt_cm_hov >= cm_min_eff - 1e-3) and np.all(opt_cm_hov <= cm_max_eff + 1e-3)
        assert np.all(opt_cm_cru >= cm_min_eff - 1e-3) and np.all(opt_cm_cru <= cm_max_eff + 1e-3)

    print(
        f"\n  [Optimization Converged in {opt_s:.2f} s | mode={opt_mode} | n_crit={nf_n_crit:.1f}, xtr={nf_xtr_upper:.2f}/{nf_xtr_lower:.2f} | "
        f"cm={'ON' if enforce_cm_bounds else 'OFF'} | ct_sigma<={f'{max_hover_ct_sigma:.3f}' if max_hover_ct_sigma is not None else 'None'}]\n"
        f"    Hover  (0km, Omega*={om_hov_opt:.2f} rad/s, M_tip*={M_hov_opt:.4f} <= {max_tip_mach}, "
        f"p0*={ph_opt_deg:.2f} deg): T* = {T_hov_opt:.1f} N (>= {T_hov_req:.0f} N), P* = {P_hov_opt/1e3:.2f} kW "
        f"({(1.0 - P_hov_opt/P_hov_base)*100:.2f}% saved vs {P_hov_base/1e3:.2f} kW), "
        f"FM = {FM_hov_opt:.4f} (vs {FM_hov_base:.4f}), sigma = {sigma_opt:.4f}, CT/sigma = {ct_hov_opt/sigma_opt:.4f}\n"
        f"    Cruise (4.5km, V={v_cruise:.2f} m/s, Omega*={om_cru_opt:.2f} rad/s, ratio={om_cru_opt/om_hov_opt:.3f} >= 0.5, "
        f"M_tip*={M_cru_opt:.4f} <= {max_tip_mach}, p0*={pc_opt_deg:.2f} deg): "
        f"T* = {T_cru_opt:.1f} N (>= {T_cru_req:.0f} N), P* = {P_cru_opt/1e3:.2f} kW, eta = {eta_cru*100:.2f}%\n"
        f"    Chord Ctrl [m]:           {np.round(c_opt, 4)} (root in [{root_chord_lb:.2f}, {root_chord_ub:.2f}])\n"
        f"    Twist Ctrl [deg]:         {np.round(tw_opt_deg, 2)} (tip ref = 0 deg)\n"
        f"    5-Sec max t/c (x/c):      {np.round(thick_opt, 4)} @ {np.round(x_tmax_loc, 2)} (req [{min_thick_req}, {max_thick_req}]) | min LE-TE gap={min_gap_opt:.4f}\n"
        f"    Sec Alpha Hover  [deg]:   {np.round(alpha_hov_base, 2)} -> {np.round(opt_alpha_hov, 2)}\n"
        f"    Sec Alpha Cruise [deg]:   {np.round(alpha_cru_base, 2)} -> {np.round(opt_alpha_cru, 2)}\n"
        f"    NF Conf Hover  (init->opt): {np.round(init_conf_hov, 4)} -> {np.round(opt_conf_hov, 4)} (req >= {np.round(min_conf_hov, 3)})\n"
        f"    NF Conf Cruise (init->opt): {np.round(init_conf_cru, 4)} -> {np.round(opt_conf_cru, 4)} (req >= {np.round(min_conf_cru, 3)})\n"
        f"    NF CM Hover    (init->opt): {np.round(init_cm_hov, 4)} -> {np.round(opt_cm_hov, 4)} (req [{np.round(cm_min_eff, 3)}, {np.round(cm_max_eff, 3)}])\n"
        f"    NF CM Cruise   (init->opt): {np.round(init_cm_cru, 4)} -> {np.round(opt_cm_cru, 4)} (req [{np.round(cm_min_eff, 3)}, {np.round(cm_max_eff, 3)}])"
    )

    cst_opt_mat = np.asarray(sol.value(cst_var))
    te_list = [float(k.TE_thickness) for k in init_kulfans]
    _plot_optimization_results(
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
        opt_conf_hov=opt_conf_hov,
        opt_conf_cru=opt_conf_cru,
        root_chord_bounds=(root_chord_lb, root_chord_ub),
        min_tip_chord=min_tip_chord,
    )

    export_nx(cst_opt_mat,te_list,ctrl_r,c_opt,tw_opt_deg)

    _plot_performance_comparison(
        B=B,
        R=R,
        R_hub=R_hub,
        n_radial=geom.n_radial,
        ctrl_r=ctrl_r,
        sec_r_norm=sec_r_norm,
        cfg=cfg,
        init_chord=init_chord,
        opt_chord=c_opt,
        init_twist_rad=init_twist,
        opt_twist_rad=np.deg2rad(tw_opt_deg),
        init_cst_mat=init_cst_mat,
        opt_cst_mat=cst_opt_mat,
        te_thicknesses=te_list,
        init_afs=init_afs,
        nf_bemt=getattr(bemt_hov, "_nf_bemt_model", None),
        nf_n_crit=nf_n_crit,
        nf_xtr_upper=nf_xtr_upper,
        nf_xtr_lower=nf_xtr_lower,
        rho_hov=rho_hov,
        mu_hov=mu_hov,
        a_sound_hov=a_sound_hov,
        omega_hov_base=omega_hov_base,
        om_hov_opt=om_hov_opt,
        ph_base_deg=float(np.rad2deg(sol_bh.value(ph_b))),
        ph_opt_deg=ph_opt_deg,
        T_hov_req=T_hov_req,
        T_hov_base=T_hov_base,
        P_hov_base=P_hov_base,
        T_hov_opt=T_hov_opt,
        P_hov_opt=P_hov_opt,
        rho_cru=rho_cru,
        mu_cru=mu_cru,
        a_sound_cru=a_sound_cru,
        v_cruise=v_cruise,
        omega_cru_base=omega_cru_base,
        om_cru_opt=om_cru_opt,
        pc_base_deg=float(np.rad2deg(sol_bc.value(pc_b))),
        pc_opt_deg=pc_opt_deg,
        T_cru_req=T_cru_req,
        T_cru_base=T_cru_base,
        P_cru_base=P_cru_base,
        T_cru_opt=T_cru_opt,
        P_cru_opt=P_cru_opt,
        cruise_power_limit_kw=(cruise_power_limit_kw if opt_mode == "hover" else None),
    )


if __name__ == "__main__":
    tiltrotor_optimization()
