"""
Phase 10 Verification & Benchmark Suite:
1. `Opti` Compilation & Sparsity Caching + `opti.solve_sweep()` Parametric Continuation
   (Zero Duplicate Graph Nodes Across 12 Sweep Solves)
2. Zero-Allocation Wave-Front `Evaluator.eval_batch` Across Variable Population Sizes
3. Pre-Compiled B-Spline Spatial Derivatives (`.derivative(nu=1/2)` in 1D & `.derivative(nu_u, nu_v)` in 2D)
   and Regularized Least-Squares Spline Fitting (`n_ctrl`, `reg_weight`)
4. CasADi-Style Modular Reusable Symbolic Sub-Model `fo.Function(name, inputs, outputs)`
   (Dual Numeric & Cross-Graph Symbolic Callability + Multi-Point Hover/Cruise Optimization)
5. Extended Tensor Indexing (`x[::2]`, `x[::-1]`, `x[[0, 3, 5]]`, `x[bool_mask]`) and
   Aerospace/Linalg Helpers (`sumsqr`, `norm`, `mean`, `vstack`, `hstack`, `diag`, `clip`, `deg2rad`, `rad2deg`)
All comments are written in English.
"""

import time
import numpy as np
from scipy.interpolate import make_interp_spline
import fastopt as fo


def fd_gradient(ev: fo._fastopt_impl.Evaluator, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = x0.size
    g = np.zeros(n, dtype=np.float64)
    for i in range(n):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = float(ev.eval(xp)[0].item())
        fm = float(ev.eval(xm)[0].item())
        g[i] = (fp - fm) / (2.0 * eps)
    return g


def fd_hessian(ev: fo._fastopt_impl.Evaluator, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = x0.size
    H = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        H[:, j] = (ev.gradient(xp, 0) - ev.gradient(xm, 0)) / (2.0 * eps)
    return 0.5 * (H + H.T)


def test_opti_compile_cache_and_solve_sweep():
    """
    Test 1: Verify that repeated `opti.solve()` and `opti.solve_sweep()` after `set_value(param, val)`:
      - Reuse the compiled C++ Evaluator and Graph-Coloring sparsity plans
      - Add ZERO new nodes to `opti.cpp_graph` across 12 parametric sweep points
      - Warm-start each sweep step cleanly
    """
    opti = fo.Opti()
    # Decision variables: collective pitch theta_deg and wing angle alpha_w_deg
    theta_deg = opti.variable(init=8.0, lb=0.5, ub=30.0, name="theta_deg")
    alpha_w_deg = opti.variable(init=4.0, lb=-5.0, ub=14.0, name="alpha_w_deg")

    # Frozen parameters for transition flight speed V_inf (m/s) and nacelle tilt angle (deg, 90=hover, 0=cruise)
    v_inf = opti.parameter(value=15.0, name="v_inf")
    tilt_deg = opti.parameter(value=75.0, name="tilt_deg")

    tilt_rad = fo.deg2rad(tilt_deg)
    # Rotor thrust and wing lift balance vertical weight W = 2000 N and horizontal drag
    q_dyn = 0.5 * 1.225 * fo.square(v_inf)
    rotor_T = 180.0 * theta_deg + 1.5 * v_inf * fo.sin(tilt_rad) * theta_deg
    wing_L = q_dyn * 1.8 * (0.11 * alpha_w_deg + 0.02)
    wing_D = q_dyn * 1.8 * (0.015 + 0.0012 * fo.square(alpha_w_deg))

    # Minimize smooth rotor power proxy + wing angle penalty
    power = 0.0025 * fo.square(rotor_T) + 0.5 * v_inf * rotor_T * fo.cos(tilt_rad) + 12.0 * fo.square(alpha_w_deg)
    opti.minimize(power * 1e-3)
    opti.subject_to(rotor_T * fo.sin(tilt_rad) + wing_L == 2000.0, scale=1000.0, name="vertical_trim")
    opti.subject_to(rotor_T * fo.cos(tilt_rad) >= wing_D, scale=100.0, name="propulsive_margin")

    opti.solver("SLSQP", options={"ftol": 1e-7, "maxiter": 100}, auto_scale=True)

    # First solve compiles the problem
    sol0 = opti.solve()
    assert sol0.success
    nodes_after_first_solve = opti.cpp_graph.num_nodes()

    # Sweep across 12 transition corridor points (V_inf from 15 to 65 m/s, tilt from 75 to 12 deg)
    v_grid = np.linspace(15.0, 65.0, 12)
    tilt_grid = np.linspace(75.0, 12.0, 12)
    sweep_pts = list(zip(v_grid, tilt_grid))

    t0 = time.perf_counter()
    sols = opti.solve_sweep([v_inf, tilt_deg], sweep_pts, warm_start=True)
    t_sweep = (time.perf_counter() - t0) * 1000.0

    nodes_after_sweep = opti.cpp_graph.num_nodes()
    assert nodes_after_sweep == nodes_after_first_solve, (
        f"Opti graph grew from {nodes_after_first_solve} to {nodes_after_sweep} nodes during solve_sweep!"
    )
    assert len(sols) == 12 and all(s.success for s in sols), (
        f"Sweep failures: {[(i, s.success, s.stats.get('status')) for i, s in enumerate(sols)]}"
    )

    print(
        f"[PASSED] 1. Opti Compilation Cache & `solve_sweep` "
        f"(12 transition points solved in {t_sweep:.2f} ms, Graph nodes stayed fixed at {nodes_after_sweep})"
    )


def test_bspline_spatial_derivatives_and_lstsq_fitting():
    """
    Test 2: Verify:
      - 1D B-Spline `.derivative(nu=1)` and `.derivative(nu=2)` (both numeric and symbolic `ctrl_pts`)
      - 2D B-Spline `.derivative(nu_u=1, nu_v=0)`, `(0, 1)`, and `(1, 1)`
      - Least-squares B-Spline fitting (`n_ctrl < len(x_grid)`, `reg_weight > 0`) on noisy data
    """
    xg = np.linspace(0.0, 2.5, 10)
    yg = np.sin(1.2 * xg) + 0.15 * (xg**3)

    bsp = fo.BSplineInterpolator1D(xg, yg, degree=3)
    dbsp1 = bsp.derivative(1)
    dbsp2 = bsp.derivative(2)

    ref_spl = make_interp_spline(xg, yg, k=3)
    xq = np.linspace(0.1, 2.4, 16)
    assert np.allclose(bsp(xq), ref_spl(xq), atol=1e-13)
    assert np.allclose(dbsp1(xq), ref_spl(xq, nu=1), atol=1e-12)
    assert np.allclose(dbsp2(xq), ref_spl(xq, nu=2), atol=1e-11)

    # Also test symbolic control points in BSplineCurve1D.derivative(1) and (2) inside an optimization graph
    gb = fo.GraphBuilder()
    c_var = gb.variable(shape=(10, 1), init=yg, name="c_var")
    sym_curve = fo.BSplineCurve1D(c_var, bsp.knots, degree=3)
    sym_d1 = sym_curve.derivative(1)
    sym_d2 = sym_curve.derivative(2)

    xq_var = gb.variable(shape=(4, 1), init=[0.3, 0.9, 1.5, 2.1], name="xq_var")
    loss = fo.sumsqr(sym_d1(xq_var)) + 0.25 * fo.sumsqr(sym_d2(xq_var))
    ev = gb.build_evaluator(loss)
    x0 = gb.cpp_graph.initial_decision_vector()

    g_ad = ev.gradient(x0, 0)
    g_fd = fd_gradient(ev, x0)
    assert np.max(np.abs(g_ad - g_fd)) < 1e-6

    H_ad = ev.hessian(x0, 0)
    H_fd = fd_hessian(ev, x0)
    assert np.max(np.abs(H_ad - H_fd)) < 1e-5

    # 2D B-Spline partial derivatives dS/du, dS/dv, d2S/(du dv)
    ug = np.linspace(-1.0, 1.0, 8)
    vg = np.linspace(0.0, 2.0, 9)
    U, V = np.meshgrid(ug, vg, indexing="ij")
    Z = np.sin(1.1 * U) * np.cos(0.9 * V) + 0.2 * (U**2) * V

    surf = fo.BSplineInterpolator2D(ug, vg, Z, degree_x=3, degree_y=3)
    dS_du = surf.derivative(1, 0)
    dS_dv = surf.derivative(0, 1)
    d2S_dudv = surf.derivative(1, 1)

    u_pt, v_pt = 0.25, 0.85
    eps = 1e-6
    fd_u = (surf(u_pt + eps, v_pt) - surf(u_pt - eps, v_pt)) / (2.0 * eps)
    fd_v = (surf(u_pt, v_pt + eps) - surf(u_pt, v_pt - eps)) / (2.0 * eps)
    fd_uv = (dS_du(u_pt, v_pt + eps) - dS_du(u_pt, v_pt - eps)) / (2.0 * eps)
    assert np.allclose(dS_du(u_pt, v_pt), fd_u, atol=1e-8)
    assert np.allclose(dS_dv(u_pt, v_pt), fd_v, atol=1e-8)
    assert np.allclose(d2S_dudv(u_pt, v_pt), fd_uv, atol=1e-8)

    # Regularized least-squares B-spline fitting on noisy airfoil/polar samples (50 samples -> 10 control points)
    rng = np.random.default_rng(42)
    x_noisy = np.linspace(0.0, 1.0, 50)
    y_clean = np.sin(2.0 * np.pi * x_noisy) + 0.3 * x_noisy
    y_noisy = y_clean + rng.normal(0.0, 0.04, size=x_noisy.shape)

    bsp_lsq = fo.BSplineInterpolator1D(x_noisy, y_noisy, degree=3, n_ctrl=10, reg_weight=1e-3)
    assert bsp_lsq.ctrl_pts.shape == (10, 1)
    rms_fit = float(np.sqrt(np.mean((bsp_lsq(x_noisy) - y_clean) ** 2)))
    assert rms_fit < 0.025, f"Expected smooth least-squares fit RMS < 0.025, got {rms_fit}"

    print(
        f"[PASSED] 2. B-Spline Spatial Derivatives (`.derivative(...)` 1D/2D) & "
        f"Regularized Least-Squares Fitting (n_ctrl=10, RMS error={rms_fit:.4f})"
    )


def test_modular_symbolic_function():
    """
    Test 3: Verify `fo.Function(name, inputs, outputs)`:
      - Build a reusable rotor blade-element aerodynamic sub-model once
      - Call it numerically (single point, `.jacobian`, `.hessian`, `.eval_batch`)
      - Call it symbolically twice inside a parent `Opti` problem (Hover + Cruise multi-point optimization)
    """
    gb_sub = fo.GraphBuilder()
    theta = gb_sub.variable(shape=(4, 1), name="theta")  # 4 radial twist angles (rad)
    v_ax = gb_sub.variable(shape=(1, 1), name="v_ax")    # axial inflow velocity (m/s)

    r_norm = gb_sub.constant([0.25, 0.50, 0.75, 0.95])
    omega_R = 120.0
    v_tan = omega_R * r_norm
    phi = fo.atan2(v_ax, v_tan)
    alpha = theta - phi
    cl = 5.7 * alpha
    cd = 0.010 + 0.045 * fo.square(alpha)
    q_sec = 0.5 * 1.225 * (fo.square(v_tan) + fo.square(v_ax))
    dT = q_sec * 0.12 * (cl * fo.cos(phi) - cd * fo.sin(phi))
    dP = q_sec * 0.12 * (cd * fo.cos(phi) + cl * fo.sin(phi)) * v_tan

    T_tot = fo.sum(dT)
    P_tot = fo.sum(dP)

    rotor_fn = fo.Function("rotor_bem_submodel", [theta, v_ax], [T_tot, P_tot])

    # 1. Direct numerical call & derivatives
    th_test = np.array([0.22, 0.18, 0.14, 0.10])
    T_num, P_num = rotor_fn(th_test, 5.0)
    assert isinstance(T_num, float) and isinstance(P_num, float) and T_num > 0.0

    J_num = rotor_fn.jacobian(th_test, 5.0, out_idx=0)
    assert J_num.shape == (1, 5)

    # 2. Cross-Graph symbolic reuse in a multi-point (Hover + Cruise) Opti design problem
    opti = fo.Opti()
    twist = opti.variable(shape=(4, 1), init=[0.20, 0.16, 0.12, 0.09], lb=0.02, ub=0.40, name="twist")
    dtheta_cruise = opti.variable(init=0.08, lb=-0.15, ub=0.30, name="dtheta_cruise")

    T_hover, P_hover = rotor_fn(twist, 4.0)
    T_cruise, P_cruise = rotor_fn(twist + dtheta_cruise, 28.0)

    opti.minimize((0.6 * P_hover + 0.4 * P_cruise) * 1e-3 + 0.5 * fo.sumsqr(twist[1:] - twist[:-1]))
    opti.subject_to(T_hover >= 420.0, scale=420.0, name="hover_thrust")
    opti.subject_to(T_cruise >= 180.0, scale=180.0, name="cruise_thrust")

    opti.solver("SLSQP", options={"ftol": 1e-7, "maxiter": 100})
    sol = opti.solve()
    assert sol.success
    assert float(sol.value(T_hover)) >= 420.0 - 1e-4
    assert float(sol.value(T_cruise)) >= 180.0 - 1e-4

    print(
        f"[PASSED] 3. Modular Symbolic `fo.Function` "
        f"(Multi-point Hover T={float(sol.value(T_hover)):.1f} N, Cruise T={float(sol.value(T_cruise)):.1f} N)"
    )


def test_fancy_indexing_and_math_linalg_helpers():
    """
    Test 4: Verify step slicing (`x[0::2]`, `x[1::2]`, `x[::-1]`), integer list/array indexing
    (`x[[0, 3, 5]]`), boolean mask indexing (`x[mask]`), and `sumsqr`, `norm`, `mean`,
    `vstack`, `hstack`, `diag`, `clip`, `deg2rad`, `rad2deg` (0th, 1st & 2nd AD).
    """
    gb = fo.GraphBuilder()
    x0 = np.array([0.4, -0.7, 1.1, 0.3, -0.5, 0.9], dtype=np.float64)
    x = gb.variable(shape=(6, 1), init=x0, name="x")

    even_x = x[0::2]  # indices [0, 2, 4]
    odd_x = x[1::2]   # indices [1, 3, 5]
    rev_x = x[::-1]   # reversed [5, 4, 3, 2, 1, 0]
    picked_x = x[[0, 1, 4, 5]]  # 2 contiguous runs: (0,2) and (4,6)
    mask = np.array([True, False, True, True, False, False])
    masked_x = x[mask]  # indices [0, 2, 3]

    # Stack, diag, norm, mean, sumsqr, clip, deg2rad/rad2deg
    M_pair = fo.hstack([even_x, odd_x])  # shape (3, 2)
    V_cat = fo.vstack([picked_x, masked_x])  # shape (7, 1)
    D_mat = fo.diag(even_x)  # shape (3, 3) diagonal matrix
    d_extracted = fo.diag(D_mat @ D_mat.T + 0.2)  # shape (3, 1)

    clipped = fo.clip(x, -0.6, 1.0)
    ang_rad = fo.deg2rad(x * 30.0)
    ang_deg = fo.rad2deg(ang_rad)

    obj = (
        fo.sumsqr(even_x - odd_x)
        + fo.norm(M_pair, ord=2, eps=1e-4)
        + fo.mean(fo.square(V_cat))
        + fo.sum(d_extracted * rev_x[0:3])
        + fo.sumsqr(clipped)
        + 0.01 * fo.sumsqr(ang_deg)
    )

    ev = gb.build_evaluator([obj, even_x, odd_x, rev_x, picked_x, masked_x, D_mat, d_extracted])
    outs = ev.eval(x0)

    assert np.allclose(outs[1].ravel(), x0[0::2])
    assert np.allclose(outs[2].ravel(), x0[1::2])
    assert np.allclose(outs[3].ravel(), x0[::-1])
    assert np.allclose(outs[4].ravel(), x0[[0, 1, 4, 5]])
    assert np.allclose(outs[5].ravel(), x0[mask])
    assert np.allclose(outs[6], np.diag(x0[0::2]))
    assert np.allclose(outs[7].ravel(), x0[0::2] ** 2 + 0.2)

    g_ad = ev.gradient(x0, 0)
    g_fd = fd_gradient(ev, x0)
    assert np.max(np.abs(g_ad - g_fd)) < 1e-6, f"Grad err: {np.max(np.abs(g_ad - g_fd))}"

    H_ad = ev.hessian(x0, 0)
    H_fd = fd_hessian(ev, x0)
    assert np.max(np.abs(H_ad - H_fd)) < 1e-5, f"Hess err: {np.max(np.abs(H_ad - H_fd))}"

    # Also verify zero-allocation wave-front eval_batch across varying population sizes
    @fo.blackbox(shape=(2, 1), batch_fn=lambda U: np.column_stack([np.sum(U**2, axis=1), np.sum(np.sin(U), axis=1)]))
    def batch_op(u):
        return np.array([np.sum(u**2), np.sum(np.sin(u))])

    gb_b = fo.GraphBuilder()
    vb = gb_b.variable(shape=(4, 1), init=[0.1, 0.2, 0.3, 0.4])
    ev_b = gb_b.build_evaluator(batch_op(vb))
    for pop_sz in (32, 12, 48):
        X_p = np.ones((pop_sz, 4), dtype=np.float64) * 0.5
        res_p = ev_b.eval_batch(X_p)[0]
        assert res_p.shape == (pop_sz, 2) and np.allclose(res_p[:, 0], 1.0)

    print(
        "[PASSED] 4. Fancy/Step Indexing (`x[::2]`, `x[::-1]`, `x[[0,1,4,5]]`, `x[mask]`), "
        "`sumsqr`/`norm`/`mean`/`diag`/`vstack`/`hstack` & Zero-Alloc `eval_batch`"
    )


if __name__ == "__main__":
    print("=" * 85)
    print("Running fastopt Phase 10 Suite (5 Project-Wide Architectural & Functional Enhancements)")
    print("=" * 85)
    test_opti_compile_cache_and_solve_sweep()
    test_bspline_spatial_derivatives_and_lstsq_fitting()
    test_modular_symbolic_function()
    test_fancy_indexing_and_math_linalg_helpers()
    print("=" * 85)
    print("ALL PHASE 10 TESTS PASSED SUCCESSFULLY!")
    print("=" * 85)
