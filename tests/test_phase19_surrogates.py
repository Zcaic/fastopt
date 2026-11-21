"""
Phase 19 Verification Suite: Native C++17 Traditional & Gradient-Enhanced Surrogates (`fastopt.surrogate`).

Tests:
  1. RBF (Gaussian, Multiquadric, InvMultiquadric, Cubic, TPS + linear/quadratic poly tails, multi-output)
     exact interpolation & exact 0/1/2 AD.
  2. Ordinary & Universal Kriging (Matérn 5/2 & Gaussian, ARD MLE) exact 0/1/2 AD for BOTH mean AND variance,
     plus differentiable Lower Confidence Bound (LCB) and Expected Improvement (EI).
  3. Direct Gradient-Enhanced Kriging (GEK) vs value-only Kriging: value+gradient interpolation, accuracy gain,
     and exact 0/1/2 AD for BOTH mean AND variance.
  4. High-Dimensional KPLS & GEKPLS (10D active subspace) + Multi-Fidelity CoKriging (with & without gradients):
     accuracy gains and exact 0/1/2 AD for mean & variance.
  5. End-to-end fo.Opti constrained optimization on GEK surrogate + gradient-based EGO infill optimization.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PY_ROOT = os.path.join(_REPO_ROOT, "python")
if _PY_ROOT not in sys.path:
    sys.path.insert(0, _PY_ROOT)

import fastopt as fo
from fastopt import surrogate as sr


def _fd_grad(fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (fn(xp) - fn(xm)) / (2.0 * eps)
    return g


def _fd_hess(grad_fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = x.size
    H = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        xp = x.copy(); xp[j] += eps
        xm = x.copy(); xm[j] -= eps
        H[:, j] = (grad_fn(xp) - grad_fn(xm)) / (2.0 * eps)
    return 0.5 * (H + H.T)


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(1.0, np.linalg.norm(b)))


def _check_scalar_ad(fn_sym_builder, x0: np.ndarray, label: str, h_tol: float = 1e-5) -> tuple[float, float]:
    gb = fo.GraphBuilder()
    xv = gb.variable((x0.size, 1), init=x0.reshape(-1, 1), name="x")
    yv = fn_sym_builder(xv)
    ev = fo.Evaluator(gb._graph, [yv.id])
    g_ad = ev.gradient(x0, 0).ravel()
    g_fd = _fd_grad(lambda z: float(ev.eval(z)[0].ravel()[0]), x0)
    h_ad = ev.hessian(x0, 0)
    h_fd = _fd_hess(lambda z: ev.gradient(z, 0).ravel(), x0)
    g_err = _rel_err(g_ad, g_fd)
    h_err = _rel_err(h_ad, h_fd)
    assert g_err < 1e-6, f"{label} grad_err={g_err}"
    assert h_err < h_tol, f"{label} hess_err={h_err}"
    return g_err, h_err


def test_1_rbf_kernels_poly_tails_and_2nd_ad() -> None:
    print("\n--- [Phase 19 Test 1] RBF (5 Kernels, Multi-Output, Linear/Quadratic Tails) Exact 0/1/2 AD ---")
    rng = np.random.default_rng(1901)
    X_tr = rng.uniform(-1.5, 1.5, size=(18, 3))
    Y_tr = np.column_stack([
        np.sin(X_tr[:, 0]) + 0.4 * X_tr[:, 1] ** 2 - 0.3 * X_tr[:, 2],
        np.cos(X_tr[:, 1]) * np.exp(-0.2 * X_tr[:, 0] ** 2) + 0.5 * X_tr[:, 2],
    ])
    x_test = np.array([0.33, -0.47, 0.62], dtype=np.float64)

    for kern, poly in [
        ("gaussian", "constant"),
        ("multiquadric", "linear"),
        ("inv_multiquadric", "quadratic"),
        ("cubic", "linear"),
        ("tps", "quadratic"),
    ]:
        model = sr.RBF(X_tr, Y_tr, kernel=kern, length_scale=1.2, poly=poly, reg=1e-10)
        # Interpolation check on training set
        interp_err = float(np.max(np.abs(model.predict(X_tr) - Y_tr)))
        assert interp_err < 1e-6, f"RBF {kern} interp_err={interp_err}"

        # Symbolic vs numeric parity + exact 1st/2nd AD on scalarized multi-output
        y_num = model.predict(x_test)
        w = np.array([[0.7], [-0.4]])
        g_err, h_err = _check_scalar_ad(lambda xv: fo.sum(model(xv) * w), x_test, f"RBF-{kern}")
        # Verify numeric value matches symbolic
        gb = fo.GraphBuilder()
        xv = gb.variable((3, 1), init=x_test.reshape(3, 1))
        ev = fo.Evaluator(gb._graph, [model(xv).id])
        assert np.max(np.abs(ev.eval(x_test)[0].ravel() - y_num)) < 1e-13
        print(f"  [RBF {kern:16s} | poly={poly:9s}] interp={interp_err:.2e} | g_err={g_err:.2e} | h_err={h_err:.2e}")

    print("  [PASSED Test 1] All 5 RBF kernels & polynomial tails passed exact 0/1/2 AD.")


def test_2_kriging_mean_variance_and_infill_ad() -> None:
    print("\n--- [Phase 19 Test 2] Ordinary/Universal Kriging Mean, Variance, LCB & EI Exact 0/1/2 AD ---")
    rng = np.random.default_rng(1902)
    X_tr = rng.uniform(-2.0, 2.0, size=(20, 2))
    Y_tr = np.sin(X_tr[:, 0]) * np.cos(0.8 * X_tr[:, 1]) + 0.15 * (X_tr[:, 0] ** 2 + X_tr[:, 1] ** 2)
    x_q = np.array([0.42, -0.65], dtype=np.float64)

    for kern, poly in [("matern52", "constant"), ("gaussian", "quadratic")]:
        krg = sr.Kriging(X_tr, Y_tr, kernel=kern, poly=poly, nugget=1e-8, n_restarts=2)
        # Variance near zero at training points and strictly positive away from them
        v_tr = float(np.max(krg.predict_variance(X_tr)))
        v_q = float(krg.predict_variance(x_q))
        assert v_tr < 1e-5, f"Expected tiny training variance, got {v_tr}"
        assert v_q > 1e-6, f"Expected positive query variance, got {v_q}"

        mg_err, mh_err = _check_scalar_ad(lambda xv: krg(xv), x_q, f"Kriging-{kern}-mean")
        vg_err, vh_err = _check_scalar_ad(lambda xv: krg.variance(xv), x_q, f"Kriging-{kern}-var")
        y_best = float(np.min(Y_tr))
        eg_err, eh_err = _check_scalar_ad(lambda xv: sr.expected_improvement(krg, xv, y_best), x_q, f"Kriging-{kern}-EI")
        lg_err, lh_err = _check_scalar_ad(lambda xv: sr.lower_confidence_bound(krg, xv, kappa=2.0), x_q, f"Kriging-{kern}-LCB")

        print(
            f"  [Kriging {kern:8s} | {poly:9s}] mean(g={mg_err:.2e}, h={mh_err:.2e}) | "
            f"var(g={vg_err:.2e}, h={vh_err:.2e}) | EI(h={eh_err:.2e}) | LCB(h={lh_err:.2e})"
        )

    print("  [PASSED Test 2] Kriging mean, variance, LCB & EI passed exact 0/1/2 AD.")


def test_3_gek_vs_kriging_accuracy_and_2nd_ad() -> None:
    print("\n--- [Phase 19 Test 3] Direct Gradient-Enhanced Kriging (GEK) Accuracy & Exact 0/1/2 AD ---")
    rng = np.random.default_rng(1903)

    def _f_and_grad(X: np.ndarray):
        x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
        y = np.sin(x0) * np.cos(x1) + 0.5 * x2 ** 2 + 0.3 * x0 * x2
        dy = np.column_stack([
            np.cos(x0) * np.cos(x1) + 0.3 * x2,
            -np.sin(x0) * np.sin(x1),
            x2 + 0.3 * x0,
        ])
        return y, dy

    X_tr = rng.uniform(-1.5, 1.5, size=(10, 3))
    Y_tr, dY_tr = _f_and_grad(X_tr)
    X_te = rng.uniform(-1.5, 1.5, size=(100, 3))
    Y_te, _ = _f_and_grad(X_te)

    krg = sr.Kriging(X_tr, Y_tr, kernel="gaussian", poly="linear", nugget=1e-8)
    gek = sr.GEK(X_tr, Y_tr, dY_tr, kernel="gaussian", poly="linear", nugget=1e-8)

    rmse_krg = float(np.sqrt(np.mean((krg.predict(X_te) - Y_te) ** 2)))
    rmse_gek = float(np.sqrt(np.mean((gek.predict(X_te) - Y_te) ** 2)))
    assert rmse_gek < 0.35 * rmse_krg, f"Expected GEK ({rmse_gek}) << Kriging ({rmse_krg})"

    # Check exact value & gradient interpolation at training points
    gb = fo.GraphBuilder()
    xv = gb.variable((3, 1), init=X_tr[0].reshape(3, 1))
    ev = fo.Evaluator(gb._graph, [gek(xv).id])
    val_err_tr = abs(float(ev.eval(X_tr[0])[0].ravel()[0]) - Y_tr[0])
    grad_err_tr = float(np.linalg.norm(ev.gradient(X_tr[0], 0).ravel() - dY_tr[0]))
    assert val_err_tr < 1e-4 and grad_err_tr < 1e-4, f"GEK training interp err: val={val_err_tr}, grad={grad_err_tr}"

    # Check exact 0/1/2 AD for BOTH mean AND variance of GEK
    x_q = np.array([0.25, -0.55, 0.71], dtype=np.float64)
    mg_err, mh_err = _check_scalar_ad(lambda xv: gek(xv), x_q, "GEK-mean")
    vg_err, vh_err = _check_scalar_ad(lambda xv: gek.variance(xv), x_q, "GEK-var")

    print(
        f"  [PASSED Test 3] 10-pt 3D RMSE: Kriging={rmse_krg:.4f} -> GEK={rmse_gek:.4f} "
        f"({rmse_krg / max(rmse_gek, 1e-12):.1f}x more accurate) | "
        f"GEK mean(h={mh_err:.2e}), var(h={vh_err:.2e})"
    )


def test_4_high_dim_kpls_gekpls_and_multifidelity_cokriging() -> None:
    print("\n--- [Phase 19 Test 4] High-Dim 10D KPLS/GEKPLS & Multi-Fidelity CoKriging ---")
    rng = np.random.default_rng(1904)

    # 4a. 10D function with 2 dominant active directions
    w1 = np.array([0.6, -0.5, 0.4, 0.3, -0.2, 0.1, 0.05, -0.05, 0.02, -0.02])
    w2 = np.array([-0.2, 0.5, 0.6, -0.4, 0.3, -0.1, 0.05, 0.02, -0.02, 0.01])

    def _f10(X: np.ndarray):
        u1 = X @ w1
        u2 = X @ w2
        y = np.sin(u1) + 0.5 * u2 ** 2 + 0.25 * u1 * u2
        du1 = np.cos(u1) + 0.25 * u2
        du2 = u2 + 0.25 * u1
        dy = du1[:, None] * w1[None, :] + du2[:, None] * w2[None, :]
        return y, dy

    X_tr = rng.uniform(-1.0, 1.0, size=(16, 10))
    Y_tr, dY_tr = _f10(X_tr)
    X_te = rng.uniform(-1.0, 1.0, size=(100, 10))
    Y_te, _ = _f10(X_te)

    kpls = sr.KPLS(X_tr, Y_tr, n_comp=2, kernel="gaussian", poly="constant")
    gekpls = sr.GEKPLS(X_tr, Y_tr, dY_tr, n_comp=2, kernel="gaussian", poly="constant")

    rmse_kpls = float(np.sqrt(np.mean((kpls.predict(X_te) - Y_te) ** 2)))
    rmse_gekpls = float(np.sqrt(np.mean((gekpls.predict(X_te) - Y_te) ** 2)))
    assert rmse_gekpls < 0.3 * rmse_kpls, f"Expected GEKPLS ({rmse_gekpls}) << KPLS ({rmse_kpls})"

    x_q10 = rng.uniform(-0.8, 0.8, size=(10,))
    _, kpls_h = _check_scalar_ad(lambda xv: kpls(xv), x_q10, "KPLS-mean")
    _, gekpls_h = _check_scalar_ad(lambda xv: gekpls(xv), x_q10, "GEKPLS-mean")
    _, gekpls_vh = _check_scalar_ad(lambda xv: gekpls.variance(xv), x_q10, "GEKPLS-var")

    # 4b. Multi-Fidelity CoKriging (30 cheap LF points + 7 expensive HF points in 2D)
    def _f_lf(X: np.ndarray):
        x0, x1 = X[:, 0], X[:, 1]
        y = np.sin(2.0 * x0) + 0.6 * np.cos(2.0 * x1) + 0.2 * x0 * x1
        dy = np.column_stack([2.0 * np.cos(2.0 * x0) + 0.2 * x1, -1.2 * np.sin(2.0 * x1) + 0.2 * x0])
        return y, dy

    def _f_hf(X: np.ndarray):
        yl, dyl = _f_lf(X)
        x0, x1 = X[:, 0], X[:, 1]
        y = 1.35 * yl + 0.4 * x0 - 0.25 * x1 + 0.5
        dy = 1.35 * dyl + np.array([[0.4, -0.25]])
        return y, dy

    X_lf = rng.uniform(-1.5, 1.5, size=(30, 2))
    Y_lf, dY_lf = _f_lf(X_lf)
    X_hf = X_lf[:7].copy()
    Y_hf, dY_hf = _f_hf(X_hf)
    X_mf_te = rng.uniform(-1.5, 1.5, size=(100, 2))
    Y_mf_te, _ = _f_hf(X_mf_te)

    krg_hf_only = sr.Kriging(X_hf, Y_hf, kernel="matern52", poly="constant")
    cokrg = sr.CoKriging(X_lf, Y_lf, X_hf, Y_hf, kernel="matern52", poly="constant")
    cokrg_ge = sr.CoKriging(X_lf, Y_lf, X_hf, Y_hf, dY_lf=dY_lf, dY_hf=dY_hf, kernel="gaussian", poly="constant")

    rmse_hf = float(np.sqrt(np.mean((krg_hf_only.predict(X_mf_te) - Y_mf_te) ** 2)))
    rmse_co = float(np.sqrt(np.mean((cokrg.predict(X_mf_te) - Y_mf_te) ** 2)))
    rmse_coge = float(np.sqrt(np.mean((cokrg_ge.predict(X_mf_te) - Y_mf_te) ** 2)))
    assert rmse_co < 0.5 * rmse_hf, f"Expected CoKriging ({rmse_co}) << HF-only ({rmse_hf})"
    assert rmse_coge <= rmse_co + 1e-3

    x_q2 = np.array([0.35, -0.52], dtype=np.float64)
    _, co_mh = _check_scalar_ad(lambda xv: cokrg(xv), x_q2, "CoKriging-mean")
    _, co_vh = _check_scalar_ad(lambda xv: cokrg.variance(xv), x_q2, "CoKriging-var")
    _, coge_mh = _check_scalar_ad(lambda xv: cokrg_ge(xv), x_q2, "GE-CoKriging-mean")

    print(
        f"  [PASSED Test 4] 10D KPLS={rmse_kpls:.4f} -> GEKPLS={rmse_gekpls:.4f} (h={gekpls_h:.2e}, vh={gekpls_vh:.2e}) | "
        f"2D MF RMSE: HF-only={rmse_hf:.4f} -> CoKriging={rmse_co:.4f} -> GE-CoKriging={rmse_coge:.5f} "
        f"(rho={cokrg.rho[0]:.2f}, co_h={co_mh:.2e}, co_vh={co_vh:.2e})"
    )


def test_5_opti_surrogate_and_ego_infill() -> None:
    print("\n--- [Phase 19 Test 5] End-to-End fo.Opti on GEK Surrogate & Differentiable EGO Infill ---")
    rng = np.random.default_rng(1905)
    # 2D shifted quadratic + nonlinear perturbation: true unconstrained min near (0.5, -0.4)
    def _obj_and_grad(X: np.ndarray):
        dx0 = X[:, 0] - 0.5
        dx1 = X[:, 1] + 0.4
        y = dx0 ** 2 + 1.5 * dx1 ** 2 + 0.2 * np.sin(X[:, 0] * X[:, 1])
        dy = np.column_stack([
            2.0 * dx0 + 0.2 * X[:, 1] * np.cos(X[:, 0] * X[:, 1]),
            3.0 * dx1 + 0.2 * X[:, 0] * np.cos(X[:, 0] * X[:, 1]),
        ])
        return y, dy

    X_tr = rng.uniform(-1.5, 1.5, size=(12, 2))
    Y_tr, dY_tr = _obj_and_grad(X_tr)
    gek = sr.GEK(X_tr, Y_tr, dY_tr, kernel="gaussian", poly="quadratic", nugget=1e-8)

    # 5a. Constrained optimization directly on GEK surrogate with exact C++ 2nd-order Hessian
    opti = fo.Opti()
    x = opti.variable((2, 1), init=np.array([[0.0], [0.0]]), lb=-1.5, ub=1.5, name="x")
    opti.minimize(gek(x))
    opti.subject_to(x[0] + x[1] >= 0.3)
    opti.solver("trust-constr", {"maxiter": 100, "gtol": 1e-8})
    sol = opti.solve()
    assert sol.success, f"GEK Opti failed: {sol.message}"
    x_star = np.asarray(sol.value(x)).ravel()
    assert x_star[0] + x_star[1] >= 0.3 - 1e-6

    # 5b. Gradient-based EGO infill: minimize LCB(x) using exact 2nd-order AD on mean & variance
    opti_ego = fo.Opti()
    xe = opti_ego.variable((2, 1), init=np.array([[0.2], [-0.2]]), lb=-1.5, ub=1.5, name="x_infill")
    opti_ego.minimize(sr.lower_confidence_bound(gek, xe, kappa=2.0))
    opti_ego.solver("trust-constr", {"maxiter": 100, "gtol": 1e-7})
    sol_ego = opti_ego.solve()
    assert sol_ego.success, f"EGO LCB Opti failed: {sol_ego.message}"
    x_infill = np.asarray(sol_ego.value(xe)).ravel()

    print(
        f"  [PASSED Test 5] GEK Constrained Opti x*={np.round(x_star, 4)}, f*={float(gek.predict(x_star)):.5f} | "
        f"EGO LCB Infill x_next={np.round(x_infill, 4)}"
    )


def run_phase19_suite() -> None:
    print("=" * 90)
    print("Running fastopt Phase 19 Traditional & Gradient-Enhanced Surrogate Suite")
    print("=" * 90)
    test_1_rbf_kernels_poly_tails_and_2nd_ad()
    test_2_kriging_mean_variance_and_infill_ad()
    test_3_gek_vs_kriging_accuracy_and_2nd_ad()
    test_4_high_dim_kpls_gekpls_and_multifidelity_cokriging()
    test_5_opti_surrogate_and_ego_infill()
    print("\nALL PHASE 19 TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_phase19_suite()
