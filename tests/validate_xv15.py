import time
from pathlib import Path
import numpy as np

import fast_bemt as fb


# =============================================================================
# 1. Exact XV-15 Full-Scale Reference Data
#    Hover: Franklin D. Harris (AHS 2018 / NASA TM-2017-219486), Table 1
#      - NASA Ames OARF (Felker et al., NASA TM-86833, 217 points, M_tip=0.60-0.73)
#      - CAMRAD II BEMT (Ki=1.15, C81 tables, Bell stall delay, M_tip=0.69)
#      - CAMRAD II Free Wake (C81 tables, Bell stall delay, M_tip=0.69)
#    Airplane Mode (Cruise, mu_z = 0.337, M_tip = 0.54):
#      - Icke et al. (AIAA 2020-3140, Table 1) citing NASA CR-114363 (1971) & SU2-URANS
# =============================================================================
XV15_SIGMA_REF = 0.0891
XV15_CT_EXP = np.array([
    0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010,
    0.011, 0.012, 0.013, 0.014, 0.015, 0.016,
], dtype=np.float64)

XV15_CP_OARF = np.array([
    0.0003106, 0.0003833, 0.0004657, 0.0005582, 0.0006614, 0.0007756, 0.0009005,
    0.0010359, 0.0011829, 0.0013457, 0.0015346, 0.0017686, 0.0020800,
], dtype=np.float64)

XV15_CP_CAMRAD_BEMT = np.array([
    0.0003211, 0.0003939, 0.0004792, 0.0005748, 0.0006794, 0.0007926, 0.0009151,
    0.0010484, 0.0011952, 0.0013595, 0.0015463, 0.0017626, 0.0020175,
], dtype=np.float64)

XV15_CP_CAMRAD_FW = np.array([
    0.0003290, 0.0003971, 0.0004756, 0.0005622, 0.0006575, 0.0007632, 0.0008824,
    0.0010183, 0.0011755, 0.0013600, 0.0015813, 0.0018535, 0.0021985,
], dtype=np.float64)

XV15_FM_OARF = (XV15_CT_EXP ** 1.5) / (np.sqrt(2.0) * XV15_CP_OARF)
XV15_FM_CAMRAD_BEMT = (XV15_CT_EXP ** 1.5) / (np.sqrt(2.0) * XV15_CP_CAMRAD_BEMT)
XV15_FM_CAMRAD_FW = (XV15_CT_EXP ** 1.5) / (np.sqrt(2.0) * XV15_CP_CAMRAD_FW)

# Airplane-mode anchor points (mu_z = 0.337, M_tip = 0.54; Icke et al. AIAA 2020-3140 Table 1)
XV15_CRUISE_REF_EXP = {"ct_sigma": 0.0372, "eta": 0.9319, "label": "NASA CR-114363 Exp (Table 1)"}
XV15_CRUISE_REF_CFD = {"ct_sigma": 0.0367, "eta": 0.8456, "label": "SU2-URANS CFD (Icke 2020)"}


# =============================================================================
# 2. Analytical NACA 64-Series Airfoil Coordinate Generator
# =============================================================================
def _naca64_coords(cl_design: float, max_tc: float, a_mean: float = 0.8, n_half: int = 81):
    """Generate Selig-ordered (TE -> upper -> LE -> lower -> TE) NACA 64-series coordinates."""
    x_ref = np.array([
        0.0000, 0.0050, 0.0075, 0.0125, 0.0250, 0.0500, 0.0750, 0.1000,
        0.1500, 0.2000, 0.2500, 0.3000, 0.3500, 0.4000, 0.4500, 0.5000,
        0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500, 0.9000,
        0.9500, 1.0000,
    ], dtype=np.float64)
    yt_ref_10 = np.array([
        0.00000, 0.00820, 0.00989, 0.01250, 0.01719, 0.02401, 0.02930, 0.03370,
        0.04059, 0.04545, 0.04858, 0.05012, 0.05012, 0.04870, 0.04574, 0.04151,
        0.03651, 0.03109, 0.02549, 0.01993, 0.01463, 0.00980, 0.00563, 0.00241,
        0.00053, 0.00000,
    ], dtype=np.float64)

    beta = np.linspace(0.0, np.pi, n_half)
    x = 0.5 * (1.0 - np.cos(beta))

    yt_base = np.interp(np.sqrt(x), np.sqrt(x_ref), yt_ref_10)
    yt = yt_base * (max_tc / (2.0 * np.max(yt_base)))

    eps = 1e-12
    xc = np.clip(x, eps, 1.0 - eps)
    a = float(np.clip(a_mean, 0.1, 0.999))
    g = (-1.0 / (1.0 - a)) * (0.5 * (a ** 2) * (0.5 * np.log(a) - 0.25) + 0.25)
    h = (1.0 / (1.0 - a)) * (0.5 * ((1.0 - a) ** 2) * np.log(1.0 - a) - 0.25 * ((1.0 - a) ** 2)) + g

    term1 = 0.5 * ((a - xc) ** 2) * np.log(np.abs(a - xc) + eps)
    term2 = -0.5 * ((1.0 - xc) ** 2) * np.log(1.0 - xc)
    term3 = 0.25 * ((1.0 - xc) ** 2) - 0.25 * ((a - xc) ** 2)
    yc = (cl_design / (2.0 * np.pi * (a + 1.0))) * (
        (1.0 / (1.0 - a)) * (term1 + term2 + term3) - xc * np.log(xc) + g - h * xc
    )
    yc[0] = 0.0
    yc[-1] = 0.0

    dyc_dx = np.gradient(yc, x, edge_order=2)
    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    xs = np.concatenate([xu[::-1], xl[1:]])
    ys = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack([xs, ys])


def _load_xv15_kulfan_sections(af_dir: Path | None = None):
    """Build the 5 XV-15 NACA 64-series KulfanCSTAirfoil sections."""
    import aerosandbox as asb

    specs = [
        ("naca64-935", 0.90, 0.35, 0.3),
        ("naca64-528", 0.50, 0.28, 0.8),
        ("naca64-118", 0.10, 0.18, 0.8),
        ("naca64-1512", 0.15, 0.12, 0.8),
        ("naca64-208", 0.20, 0.08, 0.3),
    ]
    kulfans = []
    for name, cli, tc, a_m in specs:
        dat_path = (af_dir / f"{name}.dat") if af_dir is not None else None
        if dat_path is not None and dat_path.exists():
            af = asb.Airfoil(coordinates=str(dat_path))
            src = f"file:{dat_path.name}"
        else:
            coords = _naca64_coords(cl_design=cli, max_tc=tc, a_mean=a_m)
            af = asb.Airfoil(name=name, coordinates=coords)
            src = f"analytical(CLi={cli}, t/c={tc:.2f})"
        kaf = fb.KulfanCSTAirfoil.from_aerosandbox(af.to_kulfan_airfoil())
        kulfans.append(kaf)
        print(f"    Section {name:12s}: {src}")
    return kulfans


def _load_native_neuralfoil_model(model_size: str = "large"):
    """Load native C++ fast_bemt.NeuralFoilModel via fastopt.nn.NeuralFoilSurrogate."""
    try:
        from python.fastopt.nn import NeuralFoilSurrogate
    except ImportError:
        from fastopt.nn import NeuralFoilSurrogate
    nf_surr = NeuralFoilSurrogate.from_model_size(model_size, use_float32=False, slice_aero_outputs=True)
    return nf_surr.to_fast_bemt_model()


def _build_xv15_geometry(section_airfoils, n_radial: int = 36):
    """Build full-scale XV-15 metal-blade RotorGeometry (NASA-TM-86833 / WOPWOP)."""
    R = 3.81                # 150 in (12.5 ft)
    R_hub = 0.0848 * R      # 1.06 ft root cutout (0.3231 m)
    B = 3

    # Bell XV-15 chord: 18.2 in at theoretical root -> 14.0 in at 0.25R -> constant 14.0 in to tip
    r_c = np.array([0.0848, 0.25, 1.00], dtype=np.float64)
    c_root_cut = (18.2 + (14.0 - 18.2) * (0.0848 / 0.25)) * 0.0254
    c_ref = 14.0 * 0.0254
    c_vals = np.array([c_root_cut, c_ref, c_ref], dtype=np.float64)

    # Bilinear XV-15 twist relative to r/R = 0.75 (so pitch_0 == theta_0.75)
    r_tw = np.array([0.0848, 0.25, 0.50, 0.75, 1.00], dtype=np.float64)
    tw_deg = np.array([32.30, 22.10, 9.35, 0.00, -8.40], dtype=np.float64)

    sec_r_norm = np.array([0.09, 0.17, 0.51, 0.80, 1.00], dtype=np.float64)

    return fb.RotorGeometry(
        num_blades=B,
        radius=R,
        hub_radius=R_hub,
        n_radial=n_radial,
        chord_curve=fb.BSplineCurve(r_c, c_vals, 1),
        twist_curve=fb.BSplineCurve(r_tw, np.deg2rad(tw_deg), 1),
        airfoils=section_airfoils,
        section_r_norm=sec_r_norm,
        spin_dir=1,
    )


def _get_ablation_cases():
    """Return the 7-case ablation matrix shared by Hover and Cruise validation."""
    return [
        ("1. Full Baseline (opt_blade)",  dict(tip=True,  stall=True,  tang=True,  comp=True,  ncrit=9.0, xtr=1.00), "#1f77b4", "-"),
        ("2. No Tip Loss",                dict(tip=False, stall=True,  tang=True,  comp=True,  ncrit=9.0, xtr=1.00), "#ff7f0e", "--"),
        ("3. No Stall Delay (Du-Selig)",  dict(tip=True,  stall=False, tang=True,  comp=True,  ncrit=9.0, xtr=1.00), "#2ca02c", "-."),
        ("4. No Wave Drag (Korn-Lock)",   dict(tip=True,  stall=True,  tang=True,  comp=False, ncrit=9.0, xtr=1.00), "#9467bd", ":"),
        ("5. No Tangential Ind",          dict(tip=True,  stall=True,  tang=False, comp=True,  ncrit=9.0, xtr=1.00), "#8c564b", ":"),
        ("6. Forced Turb (xtr=0.05)",     dict(tip=True,  stall=True,  tang=True,  comp=True,  ncrit=4.0, xtr=0.05), "#d62728", "-"),
        ("7. Forced Turb + No StallDel",  dict(tip=True,  stall=False, tang=True,  comp=True,  ncrit=4.0, xtr=0.05), "#e377c2", "--"),
    ]


def _build_case_solver(kulfans, nf_model, opts, n_radial: int = 36):
    """Construct RotorGeometry and SolverConfig for one ablation case."""
    afs = [
        fb.Airfoil3D.from_neuralfoil(
            model=nf_model,
            upper_weights=k.upper_weights,
            lower_weights=k.lower_weights,
            leading_edge_weight=float(k.leading_edge_weight),
            te_thickness=float(k.TE_thickness),
            n_crit=opts["ncrit"],
            xtr_upper=opts["xtr"],
            xtr_lower=opts["xtr"],
            aspect_ratio=10.71,
            enable_extrapolation=True,
            enable_compressibility=opts["comp"],
        )
        for k in kulfans
    ]
    geom = _build_xv15_geometry(afs, n_radial=n_radial)
    cfg = fb.SolverConfig(
        n_azimuth=1,
        wake_model=fb.WakeModel.GlauertAnnulus,
        enable_tip_loss=opts["tip"],
        enable_vrs_correction=True,
        enable_stall_delay=opts["stall"],
        enable_tangential_induction=opts["tang"],
    )
    return geom, cfg


def _sweep_axial(geom, cfg, pitch_deg_grid, M_tip: float, mu_z: float = 0.0):
    """Run collective sweep in hover (mu_z=0) or axial airplane mode (mu_z = V_inf / V_tip)."""
    rho = 1.225
    mu = 1.789e-5
    a_sound = 340.294
    R = geom.radius
    A_disk = np.pi * (R ** 2)
    V_tip = M_tip * a_sound
    omega = V_tip / R
    vz = mu_z * V_tip

    conds = [
        fb.OperatingCondition(
            vx=0.0, vy=0.0, vz=vz,
            omega=omega, pitch_0=float(np.deg2rad(p)),
            rho=rho, mu=mu, a_sound=a_sound,
        )
        for p in pitch_deg_grid
    ]
    solver = fb.BEMTSolver(geom, cfg)
    results = solver.solve_batch(conds, compute_gradients=False)

    ct_arr, cp_arr, fm_arr, eta_arr = [], [], [], []
    for res in results:
        T = float(res.thrust)
        P = float(res.power)
        ct = T / (rho * A_disk * (V_tip ** 2))
        cp = P / (rho * A_disk * (V_tip ** 3))
        fm = (max(ct, 0.0) ** 1.5) / (np.sqrt(2.0) * max(cp, 1e-12)) if ct > 1e-5 else 0.0
        eta = (ct * mu_z) / cp if (mu_z > 0.0 and ct > 1e-5 and cp > 1e-6) else np.nan
        ct_arr.append(ct)
        cp_arr.append(cp)
        fm_arr.append(fm)
        eta_arr.append(eta)

    return np.asarray(ct_arr), np.asarray(cp_arr), np.asarray(fm_arr), np.asarray(eta_arr)


def _interp_pre_stall(ct_sim, y_sim, ct_query):
    """Monotonically interpolate y(CT) along the pre-stall increasing-CT branch."""
    idx_max = int(np.argmax(ct_sim))
    ct_m = ct_sim[: idx_max + 1]
    y_m = y_sim[: idx_max + 1]
    order = np.argsort(ct_m)
    return np.interp(ct_query, ct_m[order], y_m[order], left=np.nan, right=np.nan)


def _ideal_axial_eta(ct_arr, mu_z: float):
    """Rankine-Froude ideal axial momentum propulsive efficiency upper bound."""
    tc = np.maximum(2.0 * ct_arr / (mu_z ** 2), 0.0)
    return 2.0 / (1.0 + np.sqrt(1.0 + tc))


# =============================================================================
# 3. Hover Validation & Ablation (M_tip = 0.69)
# =============================================================================
def _run_hover_section(case_solvers, cases, plot: bool, save_path: str):
    print("\n--- [Part 1] XV-15 Hover Mode Validation & Ablation (M_tip = 0.69) ---")
    pitch_grid = np.linspace(-2.0, 20.0, 45)
    nom_mask = (XV15_CT_EXP >= 0.008) & (XV15_CT_EXP <= 0.013)
    idx_design = int(np.argmin(np.abs(XV15_CT_EXP - 0.011)))

    print(
        f"\n  {'Case':<30s} | {'Peak FM':>7s} | {'FM@0.011':>8s} | {'dFM@0.011':>9s} | "
        f"{'CP Bias(nom)':>12s} | {'Mean|Err|(nom)':>14s} | {'Implied dKi':>11s}"
    )
    print("  " + "-" * 108)
    print(
        f"  {'[Exp] NASA OARF (TM-86833)':<30s} | {np.max(XV15_FM_OARF):7.4f} | "
        f"{XV15_FM_OARF[idx_design]:8.4f} | {'0.0000':>9s} | {'0.00%':>12s} | {'0.00%':>14s} | {'---':>11s}"
    )

    hover_store = []
    for (geom, cfg), (label, _, color, ls) in zip(case_solvers, cases):
        ct_s, cp_s, fm_s, _ = _sweep_axial(geom, cfg, pitch_grid, M_tip=0.69, mu_z=0.0)
        cp_q = _interp_pre_stall(ct_s, cp_s, XV15_CT_EXP)
        fm_q = (XV15_CT_EXP ** 1.5) / (np.sqrt(2.0) * cp_q)

        rel_err = (cp_q - XV15_CP_OARF) / XV15_CP_OARF * 100.0
        bias_nom = float(np.nanmean(rel_err[nom_mask]))
        mae_nom = float(np.nanmean(np.abs(rel_err[nom_mask])))
        fm_011 = float(fm_q[idx_design])
        dfm_011 = fm_011 - float(XV15_FM_OARF[idx_design])

        cp_ideal_011 = (0.011 ** 1.5) / np.sqrt(2.0)
        dki_011 = float((XV15_CP_OARF[idx_design] - cp_q[idx_design]) / cp_ideal_011)

        print(
            f"  {label:<30s} | {np.nanmax(fm_q):7.4f} | {fm_011:8.4f} | {dfm_011:+9.4f} | "
            f"{bias_nom:+11.2f}% | {mae_nom:13.2f}% | {dki_011:+11.4f}"
        )
        hover_store.append((label, ct_s, cp_s, fm_s, cp_q, fm_q, color, ls))

    if not plot:
        return hover_store

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    ax_fm, ax_cp, ax_rat, ax_bar = axes.ravel()

    ax_fm.plot(XV15_CT_EXP / XV15_SIGMA_REF, XV15_FM_OARF, "ko-", lw=2.2, ms=5, label="NASA OARF Exp (TM-86833)")
    ax_fm.plot(XV15_CT_EXP / XV15_SIGMA_REF, XV15_FM_CAMRAD_FW, "ks--", lw=1.3, ms=3.5, alpha=0.65, label="CAMRAD II Free Wake")
    ax_fm.plot(XV15_CT_EXP / XV15_SIGMA_REF, XV15_FM_CAMRAD_BEMT, "k^:", lw=1.3, ms=3.5, alpha=0.65, label="CAMRAD II BEMT (Ki=1.15)")
    for label, _, _, _, _, fm_q, color, ls in hover_store:
        ax_fm.plot(XV15_CT_EXP / XV15_SIGMA_REF, fm_q, ls=ls, color=color, lw=1.8, label=label)
    ax_fm.axvspan(0.008 / XV15_SIGMA_REF, 0.013 / XV15_SIGMA_REF, color="gray", alpha=0.10, label="Nominal Hover Range")
    ax_fm.set_xlabel(r"$C_T / \sigma$")
    ax_fm.set_ylabel("Figure of Merit (FM)")
    ax_fm.set_title("XV-15 Hover Figure of Merit vs. Blade Loading")
    ax_fm.set_ylim(0.50, 0.92)
    ax_fm.grid(True, alpha=0.3)
    ax_fm.legend(fontsize=7.2, loc="lower right", ncol=2)

    ax_cp.plot(XV15_CT_EXP, XV15_CP_OARF * 1e3, "ko-", lw=2.2, ms=5, label="NASA OARF Exp")
    ax_cp.plot(XV15_CT_EXP, XV15_CP_CAMRAD_FW * 1e3, "ks--", lw=1.2, ms=3.5, alpha=0.65, label="CAMRAD II Free Wake")
    for label, _, _, _, cp_q, _, color, ls in hover_store:
        ax_cp.plot(XV15_CT_EXP, cp_q * 1e3, ls=ls, color=color, lw=1.8, label=label)
    ax_cp.set_xlabel(r"Thrust Coefficient $C_T$")
    ax_cp.set_ylabel(r"Power Coefficient $C_P \times 10^3$")
    ax_cp.set_title(r"XV-15 Hover Power Polar ($C_P$ vs. $C_T$)")
    ax_cp.grid(True, alpha=0.3)
    ax_cp.legend(fontsize=7.2, loc="upper left")

    ax_rat.axhline(1.0, color="k", lw=1.2)
    ax_rat.axhspan(0.95, 1.05, color="#2ca02c", alpha=0.10, label="±5% Band")
    ax_rat.plot(XV15_CT_EXP, XV15_CP_OARF / XV15_CP_CAMRAD_FW, "ks--", lw=1.2, ms=3.5, alpha=0.65, label="CAMRAD II Free Wake")
    for label, _, _, _, cp_q, _, color, ls in hover_store:
        ax_rat.plot(XV15_CT_EXP, XV15_CP_OARF / cp_q, ls=ls, color=color, lw=1.8, label=label)
    ax_rat.set_xlabel(r"Thrust Coefficient $C_T$")
    ax_rat.set_ylabel(r"Exp $C_P$ / Theory $C_P$")
    ax_rat.set_title("Power Ratio vs. NASA OARF (>1.0 means Theory Under-predicts Power)")
    ax_rat.set_ylim(0.88, 1.25)
    ax_rat.grid(True, alpha=0.3)
    ax_rat.legend(fontsize=7.0, loc="upper right", ncol=2)

    fm_base_011 = float(hover_store[0][5][idx_design])
    ablation_labels = [
        "Tip Loss (Prandtl)",
        "Stall Delay (Du-Selig)",
        "Wave Drag + PG Comp",
        "Tangential Induction",
        "Transition (Free -> Turb)",
        "Remaining Gap vs OARF",
    ]
    ablation_deltas = [
        fm_base_011 - float(hover_store[1][5][idx_design]),
        fm_base_011 - float(hover_store[2][5][idx_design]),
        fm_base_011 - float(hover_store[3][5][idx_design]),
        fm_base_011 - float(hover_store[4][5][idx_design]),
        float(hover_store[5][5][idx_design]) - fm_base_011,
        float(XV15_FM_OARF[idx_design]) - float(hover_store[5][5][idx_design]),
    ]
    bar_colors = ["#1f77b4" if d >= 0 else "#d62728" for d in ablation_deltas]
    y_pos = np.arange(len(ablation_labels))
    bars = ax_bar.barh(y_pos, ablation_deltas, color=bar_colors, alpha=0.85)
    ax_bar.axvline(0.0, color="k", lw=1.0)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(ablation_labels, fontsize=8.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(min(ablation_deltas) - 0.010, max(ablation_deltas) + 0.008)
    ax_bar.set_xlabel(r"$\Delta \mathrm{FM}$ at $C_T = 0.011$ ($C_T/\sigma = 0.1235$)")
    ax_bar.set_title("Hover Sensitivity Breakdown of Each Correction Term")
    ax_bar.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, ablation_deltas):
        x_txt = val + (0.0012 if val >= 0 else -0.0012)
        ha = "left" if val >= 0 else "right"
        ax_bar.text(x_txt, bar.get_y() + bar.get_height() / 2.0, f"{val:+.4f}", va="center", ha=ha, fontsize=8)

    fig.suptitle(
        "fast_bemt Hover Validation Against Full-Scale XV-15 (NASA OARF TM-86833 & CAMRAD II)",
        fontsize=12.5,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160)
    print(f"  [Saved hover validation plot to {save_path}]")
    return hover_store


# =============================================================================
# 4. Airplane (Cruise) Mode Validation & Ablation (M_tip = 0.54, mu_z = 0.337 & 0.70)
# =============================================================================
def _run_cruise_section(case_solvers, cases, plot: bool, save_path: str):
    print("\n--- [Part 2] XV-15 Airplane (Cruise) Mode Validation & Ablation (M_tip = 0.54) ---")
    mu_med = 0.337  # Canonical NASA CR-114363 / Jia et al. (2022) / Flow360 benchmark (J = 1.059)
    mu_hi = 0.700   # High-speed cruise benchmark (Koning & Acree 2016 Fig. 9, J = 2.199)

    # Medium-speed sweep includes exact canonical collectives: 26.0, 27.0, 28.0, 28.8 deg
    pitch_med = np.unique(np.concatenate([
        np.linspace(19.0, 33.0, 29),
        np.array([26.0, 27.0, 28.0, 28.8]),
    ]))
    pitch_hi = np.linspace(41.0, 50.0, 37)

    ct_sig_ref_med = XV15_CRUISE_REF_CFD["ct_sigma"]  # 0.0367 (CT = 0.00327)
    ct_ref_med = ct_sig_ref_med * XV15_SIGMA_REF
    ct_sig_ref_hi = 0.030
    ct_ref_hi = ct_sig_ref_hi * XV15_SIGMA_REF

    eta_ideal_med_ref = float(_ideal_axial_eta(ct_ref_med, mu_med))
    eta_ideal_hi_ref = float(_ideal_axial_eta(ct_ref_hi, mu_hi))

    print(
        f"\n  [Medium-Speed Cruise: M_tip=0.54, mu_z={mu_med:.3f} (J={np.pi*mu_med:.3f}) | "
        f"High-Speed Cruise: mu_z={mu_hi:.2f} (J={np.pi*mu_hi:.3f})]"
    )
    print(
        f"  {'Case':<30s} | {'Peak eta(0.337)':>15s} | {'eta@CT/s=0.037':>14s} | {'vs SU2(0.8456)':>14s} | "
        f"{'CT@26/28.8deg':>15s} | {'eta@mu=0.70':>11s}"
    )
    print("  " + "-" * 112)
    print(
        f"  {'[CFD] SU2-URANS (Icke 2020)':<30s} | {'---':>15s} | {XV15_CRUISE_REF_CFD['eta']:14.4f} | {'0.0000':>14s} | "
        f"{'---':>15s} | {'---':>11s}"
    )
    print(
        f"  {'[Exp] NASA CR-114363 (Table 1)':<30s} | {'---':>15s} | {XV15_CRUISE_REF_EXP['eta']:14.4f} | "
        f"{XV15_CRUISE_REF_EXP['eta'] - XV15_CRUISE_REF_CFD['eta']:+14.4f} | {'---':>15s} | {'---':>11s}"
    )
    print(
        f"  {'[Theory] Rankine-Froude Ideal':<30s} | {'---':>15s} | {eta_ideal_med_ref:14.4f} | "
        f"{eta_ideal_med_ref - XV15_CRUISE_REF_CFD['eta']:+14.4f} | {'---':>15s} | {eta_ideal_hi_ref:11.4f}"
    )

    idx_26 = int(np.argmin(np.abs(pitch_med - 26.0)))
    idx_288 = int(np.argmin(np.abs(pitch_med - 28.8)))

    cruise_med_store = []
    cruise_hi_store = []
    for (geom, cfg), (label, _, color, ls) in zip(case_solvers, cases):
        ct_m, cp_m, _, eta_m = _sweep_axial(geom, cfg, pitch_med, M_tip=0.54, mu_z=mu_med)
        ct_h, cp_h, _, eta_h = _sweep_axial(geom, cfg, pitch_hi, M_tip=0.54, mu_z=mu_hi)

        valid_m = (ct_m > 0.0005) & np.isfinite(eta_m) & (eta_m > 0.0) & (eta_m < 1.0)
        peak_eta_m = float(np.nanmax(eta_m[valid_m])) if np.any(valid_m) else np.nan
        eta_at_ref_m = float(_interp_pre_stall(ct_m[valid_m], eta_m[valid_m], ct_ref_med)) if np.any(valid_m) else np.nan
        d_su2 = eta_at_ref_m - XV15_CRUISE_REF_CFD["eta"]

        valid_h = (ct_h > 0.0005) & np.isfinite(eta_h) & (eta_h > 0.0) & (eta_h < 1.0)
        eta_at_ref_h = float(_interp_pre_stall(ct_h[valid_h], eta_h[valid_h], ct_ref_hi)) if np.any(valid_h) else np.nan

        print(
            f"  {label:<30s} | {peak_eta_m:15.4f} | {eta_at_ref_m:14.4f} | {d_su2:+14.4f} | "
            f"{ct_m[idx_26]:.4f}/{ct_m[idx_288]:.4f} | {eta_at_ref_h:11.4f}"
        )
        cruise_med_store.append((label, ct_m, cp_m, eta_m, eta_at_ref_m, color, ls))
        cruise_hi_store.append((label, ct_h, cp_h, eta_h, eta_at_ref_h, color, ls))

    if not plot:
        return cruise_med_store, cruise_hi_store

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    ax_e1, ax_pq, ax_e2, ax_bar = axes.ravel()

    # 1. Medium-speed eta vs CT/sigma (mu_z = 0.337)
    ct_grid_m = np.linspace(0.0005, 0.0065, 100)
    ax_e1.plot(ct_grid_m / XV15_SIGMA_REF, _ideal_axial_eta(ct_grid_m, mu_med), "k:", lw=1.4, label="Rankine-Froude Ideal Limit")
    ax_e1.scatter([XV15_CRUISE_REF_CFD["ct_sigma"]], [XV15_CRUISE_REF_CFD["eta"]], color="k", marker="s", s=55, zorder=6, label=XV15_CRUISE_REF_CFD["label"])
    ax_e1.scatter([XV15_CRUISE_REF_EXP["ct_sigma"]], [XV15_CRUISE_REF_EXP["eta"]], color="k", marker="o", facecolors="none", lw=1.8, s=65, zorder=6, label=XV15_CRUISE_REF_EXP["label"])
    for label, ct_m, _, eta_m, _, color, ls in cruise_med_store:
        v = (ct_m > 0.0004) & (ct_m < 0.0070) & np.isfinite(eta_m)
        ax_e1.plot(ct_m[v] / XV15_SIGMA_REF, eta_m[v], ls=ls, color=color, lw=1.8, label=label)
    ax_e1.set_xlabel(r"$C_T / \sigma$")
    ax_e1.set_ylabel(r"Propulsive Efficiency $\eta$")
    ax_e1.set_title(r"XV-15 Airplane Mode Efficiency ($\mu_z = 0.337$, $M_{\mathrm{tip}} = 0.54$)")
    ax_e1.set_xlim(0.0, 0.075)
    ax_e1.set_ylim(0.50, 0.98)
    ax_e1.grid(True, alpha=0.3)
    ax_e1.legend(fontsize=7.2, loc="lower right", ncol=2)

    # 2. Medium-speed CQ vs CT polar (mu_z = 0.337)
    for label, ct_m, cp_m, _, _, color, ls in cruise_med_store:
        v = (ct_m > 0.0) & (ct_m < 0.0070)
        ax_pq.plot(ct_m[v] * 1e3, cp_m[v] * 1e3, ls=ls, color=color, lw=1.8, label=label)
    ax_pq.set_xlabel(r"$C_T \times 10^3$")
    ax_pq.set_ylabel(r"$C_Q \times 10^3$")
    ax_pq.set_title(r"XV-15 Airplane Mode Torque Polar ($\mu_z = 0.337$, $M_{\mathrm{tip}} = 0.54$)")
    ax_pq.grid(True, alpha=0.3)
    ax_pq.legend(fontsize=7.5, loc="upper left")

    # 3. High-speed cruise eta vs CT/sigma (mu_z = 0.70)
    ct_grid_h = np.linspace(0.0005, 0.0060, 100)
    ax_e2.plot(ct_grid_h / XV15_SIGMA_REF, _ideal_axial_eta(ct_grid_h, mu_hi), "k:", lw=1.4, label="Rankine-Froude Ideal Limit")
    for label, ct_h, _, eta_h, _, color, ls in cruise_hi_store:
        v = (ct_h > 0.0004) & (ct_h < 0.0065) & np.isfinite(eta_h)
        ax_e2.plot(ct_h[v] / XV15_SIGMA_REF, eta_h[v], ls=ls, color=color, lw=1.8, label=label)
    ax_e2.set_xlabel(r"$C_T / \sigma$")
    ax_e2.set_ylabel(r"Propulsive Efficiency $\eta$")
    ax_e2.set_title(r"XV-15 High-Speed Cruise Efficiency ($\mu_z = 0.70$, $M_{\mathrm{tip}} = 0.54$)")
    ax_e2.set_xlim(0.0, 0.070)
    ax_e2.set_ylim(0.50, 0.98)
    ax_e2.grid(True, alpha=0.3)
    ax_e2.legend(fontsize=7.2, loc="lower right", ncol=2)

    # 4. Cruise Ablation Delta-eta breakdown at mu_z = 0.337, CT/sigma = 0.0367
    eta_base = float(cruise_med_store[0][4])
    ablation_labels = [
        "Tip Loss (Prandtl)",
        "Stall Delay (Du-Selig)",
        "Wave Drag + PG Comp",
        "Tangential Induction",
        "Transition (Free -> Turb)",
        "Forced Turb vs SU2-URANS",
    ]
    ablation_deltas = [
        eta_base - float(cruise_med_store[1][4]),
        eta_base - float(cruise_med_store[2][4]),
        eta_base - float(cruise_med_store[3][4]),
        eta_base - float(cruise_med_store[4][4]),
        float(cruise_med_store[5][4]) - eta_base,
        float(cruise_med_store[5][4]) - XV15_CRUISE_REF_CFD["eta"],
    ]
    bar_colors = ["#1f77b4" if d >= 0 else "#d62728" for d in ablation_deltas]
    y_pos = np.arange(len(ablation_labels))
    bars = ax_bar.barh(y_pos, ablation_deltas, color=bar_colors, alpha=0.85)
    ax_bar.axvline(0.0, color="k", lw=1.0)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(ablation_labels, fontsize=8.5)
    ax_bar.invert_yaxis()
    pad = max(0.015, 0.25 * (max(ablation_deltas) - min(ablation_deltas)))
    ax_bar.set_xlim(min(ablation_deltas) - pad, max(ablation_deltas) + pad)
    ax_bar.set_xlabel(r"$\Delta \eta$ at $\mu_z = 0.337, C_T/\sigma = 0.0367$")
    ax_bar.set_title("Cruise Sensitivity Breakdown of Each Correction Term")
    ax_bar.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, ablation_deltas):
        x_txt = val + (0.0015 if val >= 0 else -0.0015)
        ha = "left" if val >= 0 else "right"
        ax_bar.text(x_txt, bar.get_y() + bar.get_height() / 2.0, f"{val:+.4f}", va="center", ha=ha, fontsize=8)

    fig.suptitle(
        "fast_bemt Airplane (Cruise) Mode Validation on Full-Scale XV-15 (NASA CR-114363 & SU2-URANS)",
        fontsize=12.5,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160)
    print(f"  [Saved cruise validation plot to {save_path}]")
    return cruise_med_store, cruise_hi_store


def run_xv15_full_validation(
    plot: bool = True,
    hover_save_path: str = "./runtime/xv15_hover_validation.png",
    cruise_save_path: str = "./runtime/xv15_cruise_validation.png",
):
    """Run unified Hover + Airplane (Cruise) Mode XV-15 validation and 7-case ablation study."""
    print("\n===============================================================================")
    print("  Full-Scale XV-15 Unified Validation: Hover (M_tip=0.69) + Cruise (M_tip=0.54)")
    print("===============================================================================")
    t_start = time.perf_counter()

    af_dir = Path("./tests/afs/xv15")
    kulfans = _load_xv15_kulfan_sections(af_dir if af_dir.exists() else None)
    nf_model = _load_native_neuralfoil_model("large")

    cases = _get_ablation_cases()
    case_solvers = [_build_case_solver(kulfans, nf_model, opts, n_radial=36) for _, opts, _, _ in cases]

    hover_res = _run_hover_section(case_solvers, cases, plot=plot, save_path=hover_save_path)
    cruise_res = _run_cruise_section(case_solvers, cases, plot=plot, save_path=cruise_save_path)

    print(f"\n=== Completed Unified Hover + Cruise Validation in {time.perf_counter() - t_start:.2f} s ===")
    if plot:
        import matplotlib.pyplot as plt
        plt.show()
    return hover_res, cruise_res


if __name__ == "__main__":
    run_xv15_full_validation(plot=True)
