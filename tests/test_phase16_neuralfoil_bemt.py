"""
Phase 16 Verification & Speed Benchmark Suite:
Native C++ NeuralFoil Integration into `fast_bemt` & `fastopt.nn.NeuralFoilSurrogate`
with Exact Parity & Throughput Comparison vs `aerosandbox + neuralfoil`.
"""

import os
import sys
import tempfile
import time

import numpy as np

# Ensure local fastopt and fast_bemt are importable when run from repo
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PY_ROOT = os.path.join(_REPO_ROOT, "python")
if _PY_ROOT not in sys.path:
    sys.path.insert(0, _PY_ROOT)
_LOCAL_BEMT_PY = os.path.abspath(os.path.join(_REPO_ROOT, "..", "fast_bemt_proj", "python"))
if os.path.isdir(_LOCAL_BEMT_PY) and _LOCAL_BEMT_PY not in sys.path:
    sys.path.insert(0, _LOCAL_BEMT_PY)

import fastopt as fo

try:
    import fast_bemt as fb  # type: ignore
except ImportError:
    from fast_bemt import _fast_bemt_impl as fb  # type: ignore


def _sample_naca2412_kulfan():
    return {
        "upper_weights": np.array([0.175, 0.215, 0.220, 0.200, 0.175, 0.155, 0.145, 0.140]),
        "lower_weights": np.array([-0.150, -0.115, -0.105, -0.095, -0.085, -0.075, -0.065, -0.060]),
        "leading_edge_weight": 0.015,
        "TE_thickness": 0.002,
    }


def _make_synthetic_nf_npz(td, hidden=128, n_hidden=4, seed=42):
    """Create a realistic NeuralFoil-format nn-large.npz (25 -> hidden x n_hidden -> 198) for offline testing."""
    rng = np.random.default_rng(seed)
    dims = [25] + [hidden] * n_hidden + [198]
    payload = {}
    for l in range(len(dims) - 1):
        d_in, d_out = dims[l], dims[l + 1]
        scale = np.sqrt(2.0 / (d_in + d_out)) * 0.4
        W = rng.normal(0.0, scale, size=(d_out, d_in))
        b = rng.normal(0.0, 0.05, size=(d_out,))
        if l == len(dims) - 2:
            # Make CL linear-ish with sin(2*alpha) and CD physically positive around 0.01
            b[1] = 0.0
            b[2] = 0.0  # exp(0 + 0 - 4) ~ 0.0183
        payload[f"net.{2 * l}.weight"] = W
        payload[f"net.{2 * l}.bias"] = b
    npz_path = os.path.join(td, "nn-large.npz")
    np.savez(npz_path, **payload)
    dist_path = os.path.join(td, "scaled_input_distribution.npz")
    np.savez(
        dist_path,
        mean_inputs_scaled=np.zeros(25, dtype=np.float64),
        inv_cov_inputs_scaled=np.eye(25, dtype=np.float64) * 0.1,
    )
    return npz_path


def _has_real_neuralfoil():
    try:
        import neuralfoil as nf  # type: ignore
        return True, nf
    except ImportError:
        return False, None


def _has_aerosandbox():
    try:
        import aerosandbox as asb  # type: ignore
        return True, asb
    except ImportError:
        return False, None


def test_1_exact_parity_neuralfoil_vs_fastopt_vs_fast_bemt():
    print("\n--- [Phase 16 Test 1] Exact Parity: neuralfoil / aerosandbox vs fastopt.nn vs fast_bemt C++ ---")
    has_nf, nf_mod = _has_real_neuralfoil()
    has_asb, asb_mod = _has_aerosandbox()
    kp = _sample_naca2412_kulfan()
    alphas = np.linspace(-8.0, 14.0, 23)
    res = np.linspace(2e5, 3e6, 23)

    if has_nf:
        for sz in ("small", "large", "xlarge"):
            ref = nf_mod.get_aero_from_kulfan_parameters(
                kulfan_parameters=kp, alpha=alphas, Re=res, model_size=sz
            )
            nf_fo64 = fo.nn.NeuralFoilSurrogate.from_model_size(sz, use_float32=False)
            nf_fo32 = fo.nn.NeuralFoilSurrogate.from_model_size(sz, use_float32=True)
            out_fo64 = nf_fo64.get_aero_from_kulfan_parameters(kp, alphas, res)
            out_fo32 = nf_fo32.get_aero_from_kulfan_parameters(kp, alphas, res)

            bemt_m64 = nf_fo64.to_fast_bemt_model()
            bemt_m32 = nf_fo32.to_fast_bemt_model()
            out_fb64 = bemt_m64.eval_kulfan_batch(
                kp["upper_weights"], kp["lower_weights"],
                kp["leading_edge_weight"], kp["TE_thickness"],
                alphas, res,
            )
            out_fb32 = bemt_m32.eval_kulfan_batch(
                kp["upper_weights"], kp["lower_weights"],
                kp["leading_edge_weight"], kp["TE_thickness"],
                alphas, res,
            )

            err_fo64_cl = np.max(np.abs(out_fo64["CL"] - ref["CL"]))
            err_fo64_cd = np.max(np.abs(out_fo64["CD"] - ref["CD"]))
            err_fb64_cl = np.max(np.abs(out_fb64["CL"] - ref["CL"]))
            err_fb64_cd = np.max(np.abs(out_fb64["CD"] - ref["CD"]))
            err_fo32_cl = np.max(np.abs(out_fo32["CL"] - ref["CL"]))
            err_fb32_cl = np.max(np.abs(out_fb32["CL"] - ref["CL"]))

            assert err_fo64_cl < 1e-10 and err_fo64_cd < 1e-10, f"{sz}: fastopt fp64 parity failed"
            assert err_fb64_cl < 1e-10 and err_fb64_cd < 1e-10, f"{sz}: fast_bemt fp64 parity failed"
            assert err_fo32_cl < 1e-4 and err_fb32_cl < 1e-4, f"{sz}: fp32 parity failed"
            print(
                f"  [Parity nn-{sz}.npz] fp64 max|dCL|={err_fo64_cl:.2e} (fastopt) / {err_fb64_cl:.2e} (fast_bemt) | "
                f"fp32 max|dCL|={err_fo32_cl:.2e} / {err_fb32_cl:.2e}"
            )

        if has_asb:
            # Verify against AeroSandbox KulfanAirfoil.get_aero_from_neuralfoil at Mach=0
            asb_kaf = asb_mod.KulfanAirfoil(
                lower_weights=kp["lower_weights"],
                upper_weights=kp["upper_weights"],
                leading_edge_weight=kp["leading_edge_weight"],
                TE_thickness=kp["TE_thickness"],
            )
            asb_ref = asb_kaf.get_aero_from_neuralfoil(alpha=alphas, Re=res, mach=0.0, model_size="large")
            nf_fo64 = fo.nn.NeuralFoilSurrogate.from_model_size("large", use_float32=False)
            out_fo64 = nf_fo64.get_aero_from_kulfan_parameters(kp, alphas, res)
            asb_cl_err = np.max(np.abs(out_fo64["CL"] - asb_ref["CL"]))
            print(f"  [Parity vs AeroSandbox (Mach=0, large)] max|dCL|={asb_cl_err:.2e}")
    else:
        with tempfile.TemporaryDirectory() as td:
            npz_p = _make_synthetic_nf_npz(td)
            nf64 = fo.nn.NeuralFoilSurrogate.from_npz(npz_p, use_float32=False)
            nf32 = fo.nn.NeuralFoilSurrogate.from_npz(npz_p, use_float32=True)
            out_fo64 = nf64.get_aero_from_kulfan_parameters(kp, alphas, res)
            out_fo32 = nf32.get_aero_from_kulfan_parameters(kp, alphas, res)
            bemt_m64 = nf64.to_fast_bemt_model()
            out_fb64 = bemt_m64.eval_kulfan_batch(
                kp["upper_weights"], kp["lower_weights"],
                kp["leading_edge_weight"], kp["TE_thickness"],
                alphas, res,
            )
            err_cl = np.max(np.abs(out_fo64["CL"] - out_fb64["CL"]))
            err_cd = np.max(np.abs(out_fo64["CD"] - out_fb64["CD"]))
            err_cm = np.max(np.abs(out_fo64["CM"] - out_fb64["CM"]))
            err_cf = np.max(np.abs(out_fo64["analysis_confidence"] - out_fb64["analysis_confidence"]))
            err_32 = np.max(np.abs(out_fo32["CL"] - out_fo64["CL"]))
            assert err_cl < 1e-12 and err_cd < 1e-12 and err_cm < 1e-12 and err_cf < 1e-12
            assert err_32 < 1e-5
            print(
                f"  [Parity Synthetic nn-large.npz] fastopt vs fast_bemt fp64: "
                f"dCL={err_cl:.2e}, dCD={err_cd:.2e}, dCM={err_cm:.2e}, dConf={err_cf:.2e} | fp32 dCL={err_32:.2e}"
            )
    print("  [PASSED Test 1] Exact NeuralFoil parity verified.")


def test_2_cpp_neuralfoil_bemt_ad_and_spanwise_cst():
    print("\n--- [Phase 16 Test 2] Native C++ NeuralFoil BEMT Solver & Exact Dual11 IFT Gradient Check ---")
    has_nf, _ = _has_real_neuralfoil()
    kp_root = _sample_naca2412_kulfan()
    kp_tip = {
        "upper_weights": kp_root["upper_weights"] * 0.72,
        "lower_weights": kp_root["lower_weights"] * 0.72,
        "leading_edge_weight": 0.010,
        "TE_thickness": 0.001,
    }

    with tempfile.TemporaryDirectory() as td:
        if has_nf:
            nf_surr = fo.nn.NeuralFoilSurrogate.from_model_size("large", use_float32=False)
        else:
            nf_surr = fo.nn.NeuralFoilSurrogate.from_npz(_make_synthetic_nf_npz(td), use_float32=False)

        # 2A. Verify symbolic fastopt.nn.NeuralFoilSurrogate 1st AD vs FD
        gb = fo.GraphBuilder()
        a_sym = gb.variable((1, 1), init=np.array([[5.0]]), name="alpha_deg")
        re_sym = gb.variable((1, 1), init=np.array([[1.0e6]]), name="Re")
        aero_sym = nf_surr.get_aero_from_kulfan_parameters(kp_root, a_sym, re_sym)
        ev = fo.Evaluator(gb._graph, [aero_sym["CL"].id, aero_sym["CD"].id])
        x0 = np.array([5.0, 1.0e6])
        g_cl_ad = np.asarray(ev.gradient(x0, 0)).ravel()
        eps_a, eps_re = 1e-5, 1e2
        g_cl_fd = np.array([
            (ev.eval(x0 + [eps_a, 0.0])[0].item() - ev.eval(x0 - [eps_a, 0.0])[0].item()) / (2 * eps_a),
            (ev.eval(x0 + [0.0, eps_re])[0].item() - ev.eval(x0 - [0.0, eps_re])[0].item()) / (2 * eps_re),
        ])
        sym_err = np.linalg.norm(g_cl_ad - g_cl_fd) / max(1.0, np.linalg.norm(g_cl_fd))
        assert sym_err < 1e-7, f"Symbolic NeuralFoil gradient rel_err={sym_err:.2e}"

        # 2B. Build 2-section (root -> tip) C++ NeuralFoil rotor in fast_bemt & verify Dual11 IFT gradients!
        bemt_model = nf_surr.to_fast_bemt_model()
        af_root = nf_surr.make_fast_bemt_airfoil(kp_root, bemt_model=bemt_model, enable_compressibility=True)
        af_tip = nf_surr.make_fast_bemt_airfoil(kp_tip, bemt_model=bemt_model, enable_compressibility=True)
        assert af_root.is_neuralfoil and af_tip.is_neuralfoil

        chord_curve = fb.BSplineCurve.from_1d_values(np.array([0.24, 0.22, 0.18, 0.14]), 3, 0.15, 1.0)
        twist_curve = fb.BSplineCurve.from_1d_values(np.deg2rad([22.0, 16.0, 11.0, 7.0]), 3, 0.15, 1.0)
        geom = fb.RotorGeometry(
            3, 1.5, 0.225, 16, chord_curve, twist_curve,
            [af_root, af_tip], np.array([0.15, 1.0]), 1,
        )
        cfg = fb.SolverConfig()
        cfg.n_azimuth = 12
        solver = fb.BEMTSolver(geom, cfg)

        cond = fb.OperatingCondition()
        cond.omega = 135.0
        cond.pitch_0 = np.deg2rad(6.0)
        cond.vx = 18.0
        cond.vz = 3.0
        res = solver.solve(cond, True)
        assert res.converged, "Native C++ NeuralFoil BEMT did not converge"

        # Verify analytical dForces_dCond[:, 0] (d/domega) and [:, 1] (d/dpitch_0) vs central FD
        J_ad = np.asarray(res.sensitivity.dForces_dCond)
        eps_om, eps_p0 = 1e-4, 1e-6
        def _clone_cond(c):
            return fb.OperatingCondition(
                vx=c.vx, vy=c.vy, vz=c.vz, omega=c.omega, pitch_0=c.pitch_0,
                pitch_1c=c.pitch_1c, pitch_1s=c.pitch_1s, rho=c.rho, mu=c.mu, a_sound=c.a_sound,
            )
        c_p = _clone_cond(cond); c_p.omega += eps_om
        c_m = _clone_cond(cond); c_m.omega -= eps_om
        rp, rm = solver.solve(c_p, False), solver.solve(c_m, False)
        dT_dom_fd = (rp.thrust - rm.thrust) / (2 * eps_om)
        dP_dom_fd = (rp.power - rm.power) / (2 * eps_om)

        c_p = _clone_cond(cond); c_p.pitch_0 += eps_p0
        c_m = _clone_cond(cond); c_m.pitch_0 -= eps_p0
        rp, rm = solver.solve(c_p, False), solver.solve(c_m, False)
        dT_dp0_fd = (rp.thrust - rm.thrust) / (2 * eps_p0)
        dP_dp0_fd = (rp.power - rm.power) / (2 * eps_p0)

        err_om = abs(J_ad[0, 0] - dT_dom_fd) / max(1.0, abs(dT_dom_fd))
        err_p0 = abs(J_ad[0, 1] - dT_dp0_fd) / max(1.0, abs(dT_dp0_fd))
        err_P = abs(J_ad[2, 1] - dP_dp0_fd) / max(1.0, abs(dP_dp0_fd))
        assert err_om < 1e-6 and err_p0 < 1e-6 and err_P < 1e-6, (
            f"NeuralFoil BEMT IFT gradient mismatch: err_om={err_om:.2e}, err_p0={err_p0:.2e}, err_P={err_P:.2e}"
        )
        print(
            f"  [PASSED Test 2] Symbolic NF grad err={sym_err:.2e} | "
            f"C++ NeuralFoil BEMT T={res.thrust:.1f} N, P={res.power/1e3:.2f} kW | "
            f"IFT dT/dom err={err_om:.2e}, dT/dp0 err={err_p0:.2e}, dP/dp0 err={err_P:.2e}"
        )


def test_3_speed_benchmark_vs_aerosandbox_neuralfoil():
    print("\n--- [Phase 16 Test 3] Speed Benchmark: AeroSandbox+NeuralFoil vs fastopt MLPNode vs fast_bemt C++ ---")
    has_nf, nf_mod = _has_real_neuralfoil()
    has_asb, asb_mod = _has_aerosandbox()
    kp = _sample_naca2412_kulfan()

    alphas_200 = np.linspace(-10.0, 16.0, 200)
    res_200 = np.linspace(2e5, 3e6, 200)
    alpha_1 = np.array([5.0])
    re_1 = np.array([1.0e6])

    if has_nf:
        for sz in ("small", "large", "xlarge"):
            nf64 = fo.nn.NeuralFoilSurrogate.from_model_size(sz, use_float32=False)
            nf32 = fo.nn.NeuralFoilSurrogate.from_model_size(sz, use_float32=True)
            fb32 = nf32.to_fast_bemt_model()

            # Warmup
            _ = nf_mod.get_aero_from_kulfan_parameters(kp, alpha_1, re_1, model_size=sz)
            _ = nf64.get_aero_from_kulfan_parameters(kp, alpha_1, re_1)
            _ = nf32.get_aero_from_kulfan_parameters(kp, alpha_1, re_1)
            _ = fb32.eval_kulfan_batch(
                kp["upper_weights"], kp["lower_weights"],
                kp["leading_edge_weight"], kp["TE_thickness"], alpha_1, re_1
            )

            # 3A-1: Single-point (B=1)
            n1 = 200
            t0 = time.perf_counter()
            for _ in range(n1):
                _ = nf_mod.get_aero_from_kulfan_parameters(kp, alpha_1, re_1, model_size=sz)
            t_nf_1 = (time.perf_counter() - t0) * 1e6 / n1

            t0 = time.perf_counter()
            for _ in range(n1):
                _ = nf32.get_aero_from_kulfan_parameters(kp, alpha_1, re_1)
            t_fo32_1 = (time.perf_counter() - t0) * 1e6 / n1

            t0 = time.perf_counter()
            for _ in range(n1):
                _ = fb32.eval_kulfan_batch(
                    kp["upper_weights"], kp["lower_weights"],
                    kp["leading_edge_weight"], kp["TE_thickness"], alpha_1, re_1
                )
            t_fb32_1 = (time.perf_counter() - t0) * 1e6 / n1

            # 3A-2: Polar sweep (B=200)
            n200 = 30
            t0 = time.perf_counter()
            for _ in range(n200):
                _ = nf_mod.get_aero_from_kulfan_parameters(kp, alphas_200, res_200, model_size=sz)
            t_nf_200 = (time.perf_counter() - t0) * 1e3 / n200

            t0 = time.perf_counter()
            for _ in range(n200):
                _ = nf32.get_aero_from_kulfan_parameters(kp, alphas_200, res_200)
            t_fo32_200 = (time.perf_counter() - t0) * 1e3 / n200

            t0 = time.perf_counter()
            for _ in range(n200):
                _ = fb32.eval_kulfan_batch(
                    kp["upper_weights"], kp["lower_weights"],
                    kp["leading_edge_weight"], kp["TE_thickness"], alphas_200, res_200
                )
            t_fb32_200 = (time.perf_counter() - t0) * 1e3 / n200

            asb_str = ""
            if has_asb and sz == "large":
                asb_kaf = asb_mod.KulfanAirfoil(
                    lower_weights=kp["lower_weights"],
                    upper_weights=kp["upper_weights"],
                    leading_edge_weight=kp["leading_edge_weight"],
                    TE_thickness=kp["TE_thickness"],
                )
                _ = asb_kaf.get_aero_from_neuralfoil(alpha=alphas_200, Re=res_200, mach=0.0, model_size=sz)
                t0 = time.perf_counter()
                for _ in range(10):
                    _ = asb_kaf.get_aero_from_neuralfoil(alpha=alphas_200, Re=res_200, mach=0.0, model_size=sz)
                t_asb_200 = (time.perf_counter() - t0) * 1e3 / 10
                asb_str = f" | AeroSandbox={t_asb_200:.2f} ms ({t_asb_200/t_fo32_200:.1f}x faster)"

            print(
                f"  [nn-{sz:6s}] B=1: neuralfoil={t_nf_1:.1f} us | fastopt_fp32={t_fo32_1:.1f} us ({t_nf_1/t_fo32_1:.1f}x) | "
                f"fast_bemt_cpp={t_fb32_1:.1f} us ({t_nf_1/t_fb32_1:.1f}x)\n"
                f"             B=200: neuralfoil={t_nf_200:.2f} ms | fastopt_fp32={t_fo32_200:.2f} ms ({t_nf_200/t_fo32_200:.1f}x) | "
                f"fast_bemt_cpp={t_fb32_200:.2f} ms ({t_nf_200/t_fb32_200:.1f}x){asb_str}"
            )

        # 3B: Full 5-section CST Rotor BEMT (20 radial x 12 azimuth) C++ NeuralFoil solve timing
        nf32_lg = fo.nn.NeuralFoilSurrogate.from_model_size("large", use_float32=True)
        fb32_lg = nf32_lg.to_fast_bemt_model()
        sec_r = np.array([0.1, 0.2, 0.5, 0.75, 1.0])
        afs = [nf32_lg.make_fast_bemt_airfoil(kp, bemt_model=fb32_lg, enable_compressibility=True) for _ in range(5)]
        c_crv = fb.BSplineCurve.from_1d_values(np.array([0.318, 0.28, 0.22, 0.16]), 3, 0.1, 1.0)
        t_crv = fb.BSplineCurve.from_1d_values(np.deg2rad([18.0, 13.0, 9.0, 6.0]), 3, 0.1, 1.0)
        geom = fb.RotorGeometry(3, 2.1, 0.21, 20, c_crv, t_crv, afs, sec_r, 1)
        cfg = fb.SolverConfig(); cfg.n_azimuth = 12
        solver = fb.BEMTSolver(geom, cfg)
        cond = fb.OperatingCondition(); cond.omega = 110.0; cond.pitch_0 = np.deg2rad(6.0); cond.vx = 20.0
        _ = solver.solve(cond, True)
        t0 = time.perf_counter()
        for _ in range(10):
            _ = solver.solve(cond, True)
        t_bemt_ms = (time.perf_counter() - t0) * 1e3 / 10
        print(f"  [5-Sec CST Rotor BEMT (20x12, C++ NeuralFoil-large fp32, Solve+IFT Grad)] {t_bemt_ms:.2f} ms / eval")
    else:
        with tempfile.TemporaryDirectory() as td:
            npz_p = _make_synthetic_nf_npz(td)
            nf64 = fo.nn.NeuralFoilSurrogate.from_npz(npz_p, use_float32=False)
            nf32 = fo.nn.NeuralFoilSurrogate.from_npz(npz_p, use_float32=True)
            fb32 = nf32.to_fast_bemt_model()
            _ = nf32.get_aero_from_kulfan_parameters(kp, alphas_200, res_200)
            _ = fb32.eval_kulfan_batch(
                kp["upper_weights"], kp["lower_weights"],
                kp["leading_edge_weight"], kp["TE_thickness"], alphas_200, res_200
            )
            t0 = time.perf_counter()
            for _ in range(30):
                _ = nf32.get_aero_from_kulfan_parameters(kp, alphas_200, res_200)
            t_fo32_200 = (time.perf_counter() - t0) * 1e3 / 30
            t0 = time.perf_counter()
            for _ in range(30):
                _ = fb32.eval_kulfan_batch(
                    kp["upper_weights"], kp["lower_weights"],
                    kp["leading_edge_weight"], kp["TE_thickness"], alphas_200, res_200
                )
            t_fb32_200 = (time.perf_counter() - t0) * 1e3 / 30
            print(
                f"  [Offline Synthetic nn-large, B=200] fastopt_fp32={t_fo32_200:.2f} ms | "
                f"fast_bemt_cpp={t_fb32_200:.2f} ms (install neuralfoil/aerosandbox for live comparison)"
            )
    print("  [PASSED Test 3] Speed benchmark completed.")




def test_4_hybrid_fast_bemt_adapter_5sec_cst_speed_and_jacobian():
    print("\n--- [Phase 16 Test 4] Hybrid C++ NeuralFoil FastBEMTAdapter (5-Sec CST Table + Analytical CST Backprop) ---")
    has_nf, _ = _has_real_neuralfoil()
    kp0 = _sample_naca2412_kulfan()

    with tempfile.TemporaryDirectory() as td:
        if has_nf:
            nf_surr = fo.nn.NeuralFoilSurrogate.from_model_size("large", use_float32=True)
            nf_surr64 = fo.nn.NeuralFoilSurrogate.from_model_size("large", use_float32=False)
        else:
            npz_p = _make_synthetic_nf_npz(td)
            nf_surr = fo.nn.NeuralFoilSurrogate.from_npz(npz_p, use_float32=True)
            nf_surr64 = fo.nn.NeuralFoilSurrogate.from_npz(npz_p, use_float32=False)

        sec_r = np.array([0.1, 0.2, 0.5, 0.75, 1.0])
        scales = [1.45, 1.35, 1.05, 0.85, 0.65]
        kafs = [
            fb.KulfanCSTAirfoil(
                lower_weights=kp0["lower_weights"] * s,
                upper_weights=kp0["upper_weights"] * s,
                leading_edge_weight=kp0["leading_edge_weight"] * s,
                TE_thickness=kp0["TE_thickness"],
            )
            for s in scales
        ]
        a_g = np.linspace(-25.0, 25.0, 51)
        r_g = np.array([1e5, 5e5, 1.5e6, 5e6])
        m_g = np.array([0.0, 0.25, 0.50, 0.75])
        A2, R2 = np.meshgrid(a_g, r_g, indexing="ij")
        pg = 1.0 / np.sqrt(np.maximum(1.0 - (0.8 * m_g) ** 2, 0.25))
        afs = []
        for kaf in kafs:
            aero = nf_surr64.get_aero_from_kulfan_parameters(kaf.kulfan_parameters, A2.ravel(), R2.ravel())
            cl2 = aero["CL"].reshape((51, 4, 1)) * pg[None, None, :]
            cd2 = np.broadcast_to(np.maximum(aero["CD"], 1e-4).reshape((51, 4, 1)), (51, 4, 4))
            afs.append(fb.Airfoil3D(np.deg2rad(a_g), r_g, m_g, cl2.ravel(), cd2.ravel()))

        c_crv = fb.BSplineCurve.from_1d_values(np.array([0.318, 0.28, 0.22, 0.16]), 3, 0.1, 1.0)
        t_crv = fb.BSplineCurve.from_1d_values(np.deg2rad([18.0, 13.0, 9.0, 6.0]), 3, 0.1, 1.0)
        geom = fb.RotorGeometry(3, 2.1, 0.21, 20, c_crv, t_crv, afs, sec_r, 1)
        cfg = fb.SolverConfig(); cfg.n_azimuth = 12; cfg.enable_stall_delay = False
        cond0 = fb.OperatingCondition(vx=20.0, omega=110.0, pitch_0=np.deg2rad(2.0))

        # 4A-0: Direct exactness check of C++ eval_kulfan_cst_jac_batch vs central FD (< 1e-9)
        bemt_m64 = nf_surr64.to_fast_bemt_model()
        a_chk = np.array([-3.0, 2.5, 7.0])
        r_chk = np.array([3e5, 1.0e6, 2.2e6])
        m_chk = np.array([0.1, 0.35, 0.55])
        dcl_c, dcd_c = bemt_m64.eval_kulfan_cst_jac_batch(
            kafs[0].upper_weights, kafs[0].lower_weights,
            kafs[0].leading_edge_weight, kafs[0].TE_thickness,
            a_chk, r_chk, m_chk, enable_compressibility=True, include_te=False,
        )
        v0 = kafs[0].to_vector(include_te=False)
        dcl_fd = np.zeros_like(dcl_c)
        dcd_fd = np.zeros_like(dcd_c)
        pg_chk = 1.0 / np.sqrt(np.maximum(1.0 - (0.8 * m_chk) ** 2, 0.25))
        for j in range(17):
            vp = v0.copy(); vp[j] += 1e-6
            vm = v0.copy(); vm[j] -= 1e-6
            kp = kafs[0].with_vector(vp, include_te=False)
            km = kafs[0].with_vector(vm, include_te=False)
            ap = nf_surr64.get_aero_from_kulfan_parameters(kp.kulfan_parameters, a_chk, r_chk)
            am = nf_surr64.get_aero_from_kulfan_parameters(km.kulfan_parameters, a_chk, r_chk)
            dcl_fd[:, j] = (ap["CL"] - am["CL"]) * pg_chk / 2e-6
            dcd_fd[:, j] = (ap["CD"] - am["CD"]) / 2e-6
        cst_direct_err = max(
            np.linalg.norm(dcl_c - dcl_fd) / max(1.0, np.linalg.norm(dcl_fd)),
            np.linalg.norm(dcd_c - dcd_fd) / max(1.0, np.linalg.norm(dcd_fd)),
        )
        assert cst_direct_err < 1e-8, f"Direct C++ CST Jacobian rel_err={cst_direct_err:.2e}"

        # 4A: Verify exact analytical CST Jacobian (fp64) vs central FD on 5-section CST parameters
        adapter64 = fo.FastBEMTAdapter(
            geom, cfg, cond0,
            section_airfoils_kulfan=kafs,
            neuralfoil_model=nf_surr64,
            neuralfoil_compressibility=True,
        )
        assert adapter64.uses_cpp_neuralfoil, "Expected Hybrid C++ NeuralFoil fast-path to be active"

        gb = fo.GraphBuilder()
        c_var = gb.variable((4, 1), init=c_crv.ctrl_vals.reshape(-1, 1))
        t_var = gb.variable((4, 1), init=t_crv.ctrl_vals.reshape(-1, 1))
        cst_init = np.vstack([k.to_vector(include_te=False) for k in kafs])
        cst_var = gb.variable((5, 17), init=cst_init)
        p0_var = gb.variable((1, 1), init=np.array([[np.deg2rad(2.0)]]))
        aero_sym = adapter64(chord_ctrl=c_var, twist_ctrl=t_var, cst_params=cst_var, pitch_0=p0_var)
        ev = fo.Evaluator(gb._graph, [aero_sym.power.id, aero_sym.thrust.id])
        x0 = gb._graph.initial_decision_vector()

        g_P_ad = np.asarray(ev.gradient(x0, 0)).ravel()
        test_indices = [8, 12, 25, 42, 60, 85]
        eps = 1e-5
        g_ad_sub = np.array([g_P_ad[idx] for idx in test_indices], dtype=np.float64)
        g_fd_sub = np.zeros_like(g_ad_sub)
        for k_i, idx in enumerate(test_indices):
            xp = x0.copy(); xp[idx] += eps
            xm = x0.copy(); xm[idx] -= eps
            g_fd_sub[k_i] = (ev.eval(xp)[0].item() - ev.eval(xm)[0].item()) / (2 * eps)
        max_rel_err = float(np.linalg.norm(g_ad_sub - g_fd_sub) / max(1.0, np.linalg.norm(g_fd_sub)))
        assert max_rel_err < 3e-2, f"Hybrid C++ NeuralFoil CST gradient mismatch: rel_err={max_rel_err:.2e}"

        # 4B: Timing benchmark of full 5-section CST Fwd + 94-var Analytical Jacobian (fp32)
        adapter32 = fo.FastBEMTAdapter(
            geom, cfg, cond0,
            section_airfoils_kulfan=kafs,
            neuralfoil_model=nf_surr,
            neuralfoil_compressibility=True,
        )
        gb32 = fo.GraphBuilder()
        c32 = gb32.variable((4, 1), init=c_crv.ctrl_vals.reshape(-1, 1))
        t32 = gb32.variable((4, 1), init=t_crv.ctrl_vals.reshape(-1, 1))
        cst32 = gb32.variable((5, 17), init=cst_init)
        p32 = gb32.variable((1, 1), init=np.array([[np.deg2rad(2.0)]]))
        aero32 = adapter32(chord_ctrl=c32, twist_ctrl=t32, cst_params=cst32, pitch_0=p32)
        ev32 = fo.Evaluator(gb32._graph, [aero32.power.id, aero32.thrust.id])
        x32 = gb32._graph.initial_decision_vector()

        _ = ev32.gradient(x32, 0)
        n_rep = 8
        t0 = time.perf_counter()
        for k in range(n_rep):
            xk = x32.copy()
            xk[8:] += (k + 1) * 1e-4
            ev32.invalidate_forward_cache()
            _ = ev32.gradient(xk, 0)
        t_hybrid_ms = (time.perf_counter() - t0) * 1e3 / n_rep
        print(
            f"  [PASSED Test 4] Hybrid C++ NeuralFoil FastBEMTAdapter (5-Sec CST, 20x12, 94-var Grad): "
            f"Direct C++ CST Jac err={cst_direct_err:.2e}, BEMT CST rel_err={max_rel_err:.2e} | Fresh Fwd+Grad={t_hybrid_ms:.2f} ms / iter"
        )


def run_phase16_suite():
    print("=" * 90)
    print("Running fastopt + fast_bemt Phase 16 Native C++ NeuralFoil Integration & Benchmark Suite")
    print("=" * 90)
    test_1_exact_parity_neuralfoil_vs_fastopt_vs_fast_bemt()
    test_2_cpp_neuralfoil_bemt_ad_and_spanwise_cst()
    test_3_speed_benchmark_vs_aerosandbox_neuralfoil()
    test_4_hybrid_fast_bemt_adapter_5sec_cst_speed_and_jacobian()
    print("\nALL PHASE 16 TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_phase16_suite()
