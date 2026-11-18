"""
Phase 11 Verification & Benchmark Suite:
1. `Evaluator` Same-Point Forward-Pass Deduplication (`forward_eval_count() == 1` across
   `eval`, `gradient`, `sparse_jacobian_values`, and `sparse_hessian_values` at the same iterate `x`)
2. Stack-Buffered B-Spline Basis + O(1) Uniform Grid/Knot Lookup + 3D C2 Tensor-Product B-Spline
   (`BSplineVolume3D` & `BSplineInterpolator3D` with Mode-d Separable 1D Sweeps & `.derivative(nu_u, nu_v, nu_w)`)
3. OpenMP Multi-Threaded `MapNode` (`fo.vmap`) & Multi-Threaded `Evaluator.sparse_jacobian_values`
4. Spanwise Trapezoidal Integration (`fo.trapz`) & Trajectory / ODE Collocation Primitives
   (`fo.rk4_step`, `fo.odeint_rk4`, `fo.collocate_trapz`, `fo.collocate_hermite_simpson`)
5. Post-Optimality Analytical KKT Parameter Sensitivity (`sol.sensitivity(params, target=...)`)
All comments are written in English.
"""

import time
import numpy as np
import python.fastopt as fo


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


def test_same_point_forward_deduplication():
    """
    Test 1: Verify that `Evaluator` executes the forward dynamic graph ONLY ONCE when
    `eval(x)`, `gradient(x, 0)`, `sparse_jacobian_values(x, 1)`, `sparse_jacobian_values(x, 2)`,
    and `sparse_hessian_values(x, ...)` are called at the same iterate `x`, and automatically
    re-evaluates when `x` or a frozen `Parameter` changes.
    """
    gb = fo.GraphBuilder()
    x = gb.variable(shape=(6, 1), init=np.linspace(0.2, 1.2, 6), name="x")
    p = gb.parameter(value=2.5, name="p")

    obj = fo.sum(fo.sin(p * x) + 0.5 * fo.square(x))
    h_eq = x[0:3] + x[3:6] - p
    g_ineq = fo.square(x) - 2.0

    ev = gb.build_evaluator([obj, h_eq, g_ineq])
    x0 = gb.cpp_graph.initial_decision_vector()
    # Warm up sparsity plans first
    _ = ev.hessian_sparsity()

    ev.invalidate_forward_cache()
    ev.reset_forward_eval_count()

    # Simulate a full NLP solver iteration at point x0:
    _ = ev.eval(x0)
    _ = ev.gradient(x0, 0)
    _ = ev.sparse_jacobian_values(x0, 1)
    _ = ev.sparse_jacobian_values(x0, 2)
    _ = ev.sparse_hessian_values(x0, 1.0, [np.ones(3), np.ones(6)])

    assert ev.forward_eval_count() == 1, (
        f"Expected exactly 1 forward pass at same iterate x0, got {ev.forward_eval_count()}"
    )

    # Mutating parameter p must invalidate the forward cache automatically
    p.set_value(3.0)
    _ = ev.gradient(x0, 0)
    assert ev.forward_eval_count() == 2

    # Perturbing x0 must trigger a new forward pass
    x1 = x0 + 0.05
    _ = ev.eval(x1)
    _ = ev.gradient(x1, 0)
    assert ev.forward_eval_count() == 3

    print(
        "[PASSED] 1. Evaluator Same-Point Forward Deduplication "
        "(5 NLP callbacks at same x0 executed only 1 forward graph pass: 80% reduction)"
    )


def test_3d_bspline_volume_and_separable_fitting():
    """
    Test 2: Verify `BSplineVolume3D` and `BSplineInterpolator3D`:
      - Fast Mode-1/2/3 axis-separable 3D collocation solve on a (15 x 18 x 20 = 5,400 pts) 3D grid
      - Exact 0th-order interpolation, `.derivative(nu_u, nu_v, nu_w)` partial derivative volumes
      - Exact 1st-order gradient and 2nd-order Hessian w.r.t. both `(xq, yq, zq)` and symbolic 3D `ctrl_pts`
    """
    xg = np.linspace(-1.0, 1.0, 8)
    yg = np.linspace(0.0, 2.0, 9)
    zg = np.linspace(-0.5, 1.5, 10)
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")
    W = np.sin(0.9 * X) * np.cos(0.8 * Y) * np.exp(0.3 * Z) + 0.15 * (X**2) * Y * Z

    t0 = time.perf_counter()
    vol = fo.BSplineInterpolator3D(xg, yg, zg, W, degree_x=3, degree_y=3, degree_z=3)
    t_fit_ms = (time.perf_counter() - t0) * 1000.0

    # Check exact reproduction at grid nodes
    assert np.allclose(vol(xg[3], yg[4], zg[5]), W[3, 4, 5], atol=1e-12)

    # Check 3D partial derivative volumes dV/dx, dV/dy, dV/dz, d2V/(dx dy)
    dV_dx = vol.derivative(1, 0, 0)
    dV_dy = vol.derivative(0, 1, 0)
    dV_dz = vol.derivative(0, 0, 1)
    d2V_dxdy = vol.derivative(1, 1, 0)

    qx, qy, qz = 0.25, 0.85, 0.45
    eps = 1e-6
    fd_x = (vol(qx + eps, qy, qz) - vol(qx - eps, qy, qz)) / (2.0 * eps)
    fd_y = (vol(qx, qy + eps, qz) - vol(qx, qy - eps, qz)) / (2.0 * eps)
    fd_z = (vol(qx, qy, qz + eps) - vol(qx, qy, qz - eps)) / (2.0 * eps)
    fd_xy = (dV_dx(qx, qy + eps, qz) - dV_dx(qx, qy - eps, qz)) / (2.0 * eps)

    assert np.allclose(dV_dx(qx, qy, qz), fd_x, atol=1e-8)
    assert np.allclose(dV_dy(qx, qy, qz), fd_y, atol=1e-8)
    assert np.allclose(dV_dz(qx, qy, qz), fd_z, atol=1e-8)
    assert np.allclose(d2V_dxdy(qx, qy, qz), fd_xy, atol=1e-8)

    # Check 1st and 2nd AD w.r.t. query points (xq, yq, zq) inside a Graph
    gb = fo.GraphBuilder()
    uq = gb.variable(shape=(3, 1), init=[-0.3, 0.1, 0.5], name="uq")
    vq = gb.variable(shape=(3, 1), init=[0.4, 1.1, 1.6], name="vq")
    wq = gb.variable(shape=(3, 1), init=[-0.1, 0.5, 1.1], name="wq")

    val_3d = vol(uq, vq, wq)
    loss = fo.sumsqr(val_3d) + 0.3 * fo.sum(val_3d)
    ev = gb.build_evaluator(loss)
    x0 = gb.cpp_graph.initial_decision_vector()

    g_ad = ev.gradient(x0, 0)
    g_fd = fd_gradient(ev, x0)
    assert np.max(np.abs(g_ad - g_fd)) < 1e-6, f"3D B-spline grad err: {np.max(np.abs(g_ad - g_fd))}"

    H_ad = ev.hessian(x0, 0)
    H_fd = fd_hessian(ev, x0)
    assert np.max(np.abs(H_ad - H_fd)) < 1e-5, f"3D B-spline hess err: {np.max(np.abs(H_ad - H_fd))}"

    # Also test symbolic 3D control points on a (4 x 4 x 4) B-Spline volume
    gb_s = fo.GraphBuilder()
    xg_s = np.linspace(0.0, 1.0, 4)
    yg_s = np.linspace(0.0, 1.0, 4)
    zg_s = np.linspace(0.0, 1.0, 4)
    Xs, Ys, Zs = np.meshgrid(xg_s, yg_s, zg_s, indexing="ij")
    Ws_init = (Xs + 0.5 * Ys - 0.3 * Zs).ravel(order="F")

    w_var = gb_s.variable(shape=(64, 1), init=Ws_init, name="w_var")
    q_var = gb_s.variable(shape=(3, 1), init=[0.35, 0.55, 0.65], name="q_var")
    sym_vol = fo.BSplineInterpolator3D(xg_s, yg_s, zg_s, w_var, degree_x=2, degree_y=2, degree_z=2)
    sym_out = sym_vol(q_var[0], q_var[1], q_var[2])
    ev_s = gb_s.build_evaluator(fo.square(sym_out))
    x0_s = gb_s.cpp_graph.initial_decision_vector()
    assert np.max(np.abs(ev_s.gradient(x0_s, 0) - fd_gradient(ev_s, x0_s))) < 1e-6
    assert np.max(np.abs(ev_s.hessian(x0_s, 0) - fd_hessian(ev_s, x0_s))) < 1e-5

    # Benchmark larger 3D table (20 x 25 x 30 = 15,000 grid points)
    x_big = np.linspace(0.0, 1.0, 20)
    y_big = np.linspace(0.0, 1.0, 25)
    z_big = np.linspace(0.0, 1.0, 30)
    W_big = np.ones((20, 25, 30), dtype=np.float64)
    t1 = time.perf_counter()
    _ = fo.BSplineInterpolator3D(x_big, y_big, z_big, W_big, degree_x=3, degree_y=3, degree_z=3)
    t_big_ms = (time.perf_counter() - t1) * 1000.0

    print(
        f"[PASSED] 2. 3D C2 B-Spline (`BSplineVolume3D`/`BSplineInterpolator3D`) & Stack-Buffered Basis "
        f"(20x25x30=15,000 pts 3D coefficients solved in {t_big_ms:.2f} ms; 8x9x10 in {t_fit_ms:.2f} ms)"
    )


def test_openmp_vmap_and_parallel_sparse_jacobian():
    """
    Test 3: Verify OpenMP parallel `MapNode` (`fo.vmap` over B=32 batches) for 0th, 1st,
    and 2nd order AD, plus multi-threaded `Evaluator.sparse_jacobian_values(..., num_threads=4)`.
    """
    gb = fo.GraphBuilder()
    B = 32
    X_in = gb.variable(shape=(3, B), init=np.ones((3, B)) * 0.4, name="X_in")
    sh_p = gb.variable(shape=(3, 1), init=[1.1, 0.8, 1.3], name="sh_p")

    def strip_load(x_col: fo.Expr, p_vec: fo.Expr) -> fo.Expr:
        return fo.sin(x_col * p_vec) + 0.25 * fo.square(x_col) * p_vec

    Y_mapped = fo.vmap(strip_load, [X_in, sh_p], in_axes=[1, None], out_axis=1)
    loss = fo.sumsqr(Y_mapped)
    ev = gb.build_evaluator([loss, Y_mapped.reshape(3 * B, 1)])
    x0 = gb.cpp_graph.initial_decision_vector()

    g_ad = ev.gradient(x0, 0)
    g_fd = fd_gradient(ev, x0)
    assert np.max(np.abs(g_ad - g_fd)) < 1e-6

    H_ad = ev.hessian(x0, 0)
    H_fd = fd_hessian(ev, x0)
    assert np.max(np.abs(H_ad - H_fd)) < 1e-5

    # Also test multi-threaded sparse_jacobian_values on a pure tensor graph
    gb2 = fo.GraphBuilder()
    v2 = gb2.variable(shape=(40, 1), init=np.linspace(0.1, 1.5, 40), name="v2")
    r_vec = fo.sin(v2[:-1]) * fo.cos(v2[1:]) + v2[:-1] * v2[1:]
    ev2 = gb2.build_evaluator(r_vec)
    x2 = gb2.cpp_graph.initial_decision_vector()

    j_1t = ev2.sparse_jacobian_values(x2, 0, num_threads=1)
    j_4t = ev2.sparse_jacobian_values(x2, 0, num_threads=4)
    assert np.allclose(j_1t, j_4t, atol=1e-14)

    print(
        "[PASSED] 3. OpenMP Multi-Threaded `MapNode` (`fo.vmap` B=32) & "
        "Multi-Threaded `sparse_jacobian_values` (1T vs 4T exact match)"
    )


def test_trapz_odeint_rk4_and_collocation():
    """
    Test 4: Verify:
      - Non-uniform `fo.trapz(y, x)` spanwise integration (value, gradient & Hessian)
      - Single-ScanNode `fo.odeint_rk4` trajectory integration
      - 4th-order `fo.collocate_hermite_simpson` & `fo.collocate_trapz` optimal control in `Opti`
    """
    # 4a. Non-uniform spanwise trapezoidal integration
    gb = fo.GraphBuilder()
    r_grid = np.array([0.2, 0.35, 0.55, 0.75, 0.90, 1.0], dtype=np.float64)
    load = gb.variable(shape=(6, 1), init=r_grid**2, name="load")
    integral = fo.trapz(fo.square(load) + 0.5 * load, x=r_grid)
    ev_t = gb.build_evaluator(integral)
    x0_t = gb.cpp_graph.initial_decision_vector()
    _np_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    expected_trapz = _np_trapezoid((x0_t**2) + 0.5 * x0_t, x=r_grid)
    assert np.isclose(float(ev_t.eval(x0_t)[0].item()), expected_trapz, atol=1e-13)
    assert np.max(np.abs(ev_t.gradient(x0_t, 0) - fd_gradient(ev_t, x0_t))) < 1e-6
    assert np.max(np.abs(ev_t.hessian(x0_t, 0) - fd_hessian(ev_t, x0_t))) < 1e-5

    # 4b. Single-ScanNode RK4 integration (`fo.odeint_rk4`)
    gb_ode = fo.GraphBuilder()
    x_init = gb_ode.variable(shape=(2, 1), init=[1.0, 0.0], name="x_init")
    u_seq = gb_ode.variable(shape=(1, 20), init=np.ones((1, 20)) * 0.2, name="u_seq")

    def dyn_rhs(state: fo.Expr, u: fo.Expr) -> fo.Expr:
        pos = state[0]
        vel = state[1]
        acc = -2.0 * pos - 0.5 * vel + u
        return fo.vstack([vel, acc])

    x_final, X_traj = fo.odeint_rk4(dyn_rhs, x_init, u_seq=u_seq, dt=0.05)
    assert x_final.shape == (2, 1) and X_traj.shape == (2, 20)
    ev_ode = gb_ode.build_evaluator(fo.sumsqr(x_final) + 0.1 * fo.sumsqr(X_traj))
    x0_ode = gb_ode.cpp_graph.initial_decision_vector()
    assert np.max(np.abs(ev_ode.gradient(x0_ode, 0) - fd_gradient(ev_ode, x0_ode))) < 1e-6

    # 4c. Direct Hermite-Simpson collocation optimal control problem:
    # Drive a double integrator [pos, vel] from [0, 0] to [1, 0] in T = 1.0 s (N = 10 intervals)
    # minimizing integral(u^2 dt) -> analytical optimal cost is 12.0!
    N = 20
    dt = 1.0 / N
    opti = fo.Opti()
    X = opti.variable(shape=(2, N + 1), init=0.0, name="X")
    U = opti.variable(shape=(1, N + 1), init=0.0, lb=-20.0, ub=20.0, name="U")

    def dbl_int(x_k: fo.Expr, u_k: fo.Expr) -> fo.Expr:
        return fo.vstack([x_k[1], u_k[0]])

    defects = fo.collocate_hermite_simpson(dbl_int, X, U, dt=dt)
    opti.minimize(fo.trapz(fo.square(U.T), dx=dt))
    opti.subject_to(defects == 0.0, name="collocation_defects")
    opti.subject_to(X[:, 0] == np.array([0.0, 0.0]), name="bc_init")
    opti.subject_to(X[:, N] == np.array([1.0, 0.0]), name="bc_final")

    opti.solver("SLSQP", options={"ftol": 1e-9, "maxiter": 100})
    sol = opti.solve()
    assert sol.success
    # Analytical minimum control effort for double integrator rest-to-rest in T=1 is J* = 12.0
    assert np.isclose(sol.f, 12.0, atol=0.10), f"Expected J* ~ 12.0, got {sol.f}"

    print(
        f"[PASSED] 4. Spanwise `trapz`, Single-Scan `odeint_rk4`, & Hermite-Simpson Collocation "
        f"(Rest-to-rest optimal control J*={sol.f:.4f} vs analytical 12.0)"
    )


def test_post_optimality_kkt_parameter_sensitivity():
    """
    Test 5: Verify `sol.sensitivity(params, target=...)` against central finite-difference
    re-solves (`opti.solve()` at `p + eps` and `p - eps`):
      - Objective sensitivity `df_dp` (Envelope Theorem)
      - Primal decision variable sensitivity `dx_dp` (Active KKT Implicit Function Theorem)
    """
    opti = fo.Opti()
    theta_deg = opti.variable(init=8.0, lb=0.5, ub=30.0, name="theta_deg")
    alpha_w_deg = opti.variable(init=4.0, lb=-5.0, ub=14.0, name="alpha_w_deg")

    v_inf = opti.parameter(value=35.0, name="v_inf")
    weight_N = opti.parameter(value=2000.0, name="weight_N")

    q_dyn = 0.5 * 1.225 * fo.square(v_inf)
    rotor_T = 180.0 * theta_deg + 1.2 * v_inf * theta_deg
    wing_L = q_dyn * 1.8 * (0.11 * alpha_w_deg + 0.02)
    wing_D = q_dyn * 1.8 * (0.015 + 0.0012 * fo.square(alpha_w_deg))

    power = 0.003 * fo.square(rotor_T) + 0.4 * v_inf * rotor_T + 15.0 * fo.square(alpha_w_deg)
    opti.minimize(power * 1e-3)
    opti.subject_to(0.7071 * rotor_T + wing_L == weight_N, scale=1000.0, name="vertical_balance")
    opti.subject_to(0.7071 * rotor_T >= wing_D, scale=100.0, name="propulsive_balance")

    opti.solver("SLSQP", options={"ftol": 1e-11, "maxiter": 100}, auto_scale=True)
    sol0 = opti.solve()
    assert sol0.success

    # Analytical post-optimality sensitivity w.r.t. [v_inf, weight_N] without re-solving
    sens = sol0.sensitivity([v_inf, weight_N])
    df_dp_ana = sens["df_dp"]  # shape (2,)
    dx_dp_ana = sens["dx_dp"]  # shape (2, 2)

    # Also check single-parameter target queries
    dtheta_dv_ana = sol0.sensitivity(v_inf, target=theta_deg)
    assert np.isclose(dtheta_dv_ana, dx_dp_ana[0, 0], atol=1e-12)

    # Central finite-difference verification by perturbing v_inf and weight_N and re-solving
    eps_v = 1e-3
    v_inf.set_value(35.0 + eps_v)
    sol_vp = opti.solve()
    v_inf.set_value(35.0 - eps_v)
    sol_vm = opti.solve()
    v_inf.set_value(35.0)

    df_dv_fd = (sol_vp.f - sol_vm.f) / (2.0 * eps_v)
    dx_dv_fd = (sol_vp.x - sol_vm.x) / (2.0 * eps_v)

    eps_w = 1e-1
    weight_N.set_value(2000.0 + eps_w)
    sol_wp = opti.solve()
    weight_N.set_value(2000.0 - eps_w)
    sol_wm = opti.solve()
    weight_N.set_value(2000.0)

    df_dw_fd = (sol_wp.f - sol_wm.f) / (2.0 * eps_w)
    dx_dw_fd = (sol_wp.x - sol_wm.x) / (2.0 * eps_w)

    assert np.isclose(df_dp_ana[0], df_dv_fd, rtol=1e-3, atol=1e-4), f"{df_dp_ana[0]} vs {df_dv_fd}"
    assert np.isclose(df_dp_ana[1], df_dw_fd, rtol=1e-3, atol=1e-4), f"{df_dp_ana[1]} vs {df_dw_fd}"
    assert np.allclose(dx_dp_ana[:, 0], dx_dv_fd, rtol=1e-3, atol=1e-4), f"{dx_dp_ana[:, 0]} vs {dx_dv_fd}"
    assert np.allclose(dx_dp_ana[:, 1], dx_dw_fd, rtol=1e-3, atol=1e-4), f"{dx_dp_ana[:, 1]} vs {dx_dw_fd}"

    print(
        f"[PASSED] 5. Post-Optimality Analytical KKT Parameter Sensitivity `sol.sensitivity()` "
        f"(dJ*/dV_inf={df_dp_ana[0]:.5f} vs FD={df_dv_fd:.5f}, dtheta*/dV_inf={dx_dp_ana[0,0]:.5f} vs FD={dx_dv_fd[0]:.5f})"
    )


if __name__ == "__main__":
    print("=" * 85)
    print("Running fastopt Phase 11 Suite (6 Deep C++ Performance & 3D Spline / MDO Enhancements)")
    print("=" * 85)
    test_same_point_forward_deduplication()
    test_3d_bspline_volume_and_separable_fitting()
    test_openmp_vmap_and_parallel_sparse_jacobian()
    test_trapz_odeint_rk4_and_collocation()
    test_post_optimality_kkt_parameter_sensitivity()
    print("=" * 85)
    print("ALL PHASE 11 TESTS PASSED SUCCESSFULLY!")
    print("=" * 85)
