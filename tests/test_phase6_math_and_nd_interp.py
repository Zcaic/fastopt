"""
Phase 6 Verification Suite:
1. Inverse Trigonometric & Hyperbolic Operators (asin, acos, atan, sinh, cosh) + 4-Quadrant atan2(y, x)
2. Overflow-Safe KS Aggregation (ks_max, ks_min, logsumexp) & Smooth C2 Ops (smooth_abs, smooth_max, smooth_min, smooth_clip)
3. 2D Bilinear (interp_linear_2d) & 2D Tensor-Product B-Spline Surface (interp_bspline_2d, bspline_eval_2d) Interpolation
4. 3D Trilinear Regular-Grid Interpolation (interp_linear_3d) with 1st/2nd-Order AD & Extrapolation Control
5. Spanwise Aerodynamic Optimization Combining atan2, 2D B-Spline Polar Table, and KS Envelope Constraint
All comments are written in English.
"""

import numpy as np
import python.fastopt as fo


def fd_gradient(eval_fn, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient for scalar function eval_fn(x)."""
    n = x0.size
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (float(np.asarray(eval_fn(xp)).item()) - float(np.asarray(eval_fn(xm)).item())) / (2.0 * eps)
    return grad


def fd_hessian_from_grad(grad_fn, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference Hessian computed from exact analytical gradients."""
    n = x0.size
    hess = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        gp = grad_fn(xp)
        gm = grad_fn(xm)
        hess[:, j] = (gp - gm) / (2.0 * eps)
    return 0.5 * (hess + hess.T)


def test_inverse_trig_hyperbolic_and_atan2():
    gb = fo.GraphBuilder()
    # x in (-0.8, 0.8) so asin and acos are strictly inside (-1, 1)
    x0 = np.array([-0.65, -0.25, 0.35, 0.72], dtype=np.float64)
    y0 = np.array([0.80, -0.55, -0.90, 0.45], dtype=np.float64)

    x = gb.variable(shape=(4, 1), init=x0, name="x")
    y = gb.variable(shape=(4, 1), init=y0, name="y")

    # Combine asin, acos, atan, sinh, cosh, and 4-quadrant atan2 with broadcasting
    phi = fo.atan2(y, x) + fo.atan2(y, 1.25)
    term_trig = fo.asin(x) * 1.3 + fo.acos(x) * 0.7 + fo.atan(y) * 1.1
    term_hyp = fo.sinh(x * 0.8) - 0.5 * fo.cosh(y * 0.9)
    obj = fo.sum(fo.sin(phi) * term_trig + term_hyp * fo.cos(phi))

    ev = gb.build_evaluator(obj)
    xy0 = np.concatenate([x0, y0])

    # Verify 0th-order value against NumPy
    phi_np = np.arctan2(y0, x0) + np.arctan2(y0, 1.25)
    trig_np = np.arcsin(x0) * 1.3 + np.arccos(x0) * 0.7 + np.arctan(y0) * 1.1
    hyp_np = np.sinh(x0 * 0.8) - 0.5 * np.cosh(y0 * 0.9)
    expected_val = float(np.sum(np.sin(phi_np) * trig_np + hyp_np * np.cos(phi_np)))
    val = float(ev.eval(xy0)[0].item())
    assert np.isclose(val, expected_val, atol=1e-13), f"Value mismatch: {val} vs {expected_val}"

    # Verify 1st-order gradient and 2nd-order Hessian against FD
    g_ad = ev.gradient(xy0)
    g_fd = fd_gradient(lambda z: ev.eval(z), xy0)
    assert np.max(np.abs(g_ad - g_fd)) < 1e-7, f"Grad err: {np.max(np.abs(g_ad - g_fd))}"

    h_ad = ev.lagrangian_hessian(xy0, 1.0, np.zeros(0))
    h_fd = fd_hessian_from_grad(lambda z: ev.gradient(z), xy0)
    assert np.max(np.abs(h_ad - h_fd)) < 1e-6, f"Hess err: {np.max(np.abs(h_ad - h_fd))}"
    print("[PASSED] 1. Inverse Trig (asin, acos, atan), Hyperbolic (sinh, cosh) & 4-Quadrant atan2 (1st/2nd AD)")


def test_ks_aggregation_and_smooth_ops():
    gb = fo.GraphBuilder()
    X0 = np.array(
        [
            [0.20, -0.40, 1.15],
            [0.85, 0.55, -0.30],
        ],
        dtype=np.float64,
    )
    X = gb.variable(shape=(2, 3), init=X0, name="X")

    # Test ks_max / ks_min over full tensor (axis=None), along axis=0, and along axis=1
    ks_all = fo.ks_max(X, rho=25.0, axis=None, conservative=False)
    ks_all_cons = fo.ks_max(X, rho=25.0, axis=None, conservative=True)
    ks_min_all = fo.ks_min(X, rho=20.0, axis=None, conservative=False)
    ks_min_cons = fo.ks_min(X, rho=20.0, axis=None, conservative=True)
    ks_ax0 = fo.ks_max(X, rho=15.0, axis=0)  # shape (1, 3)
    ks_ax1 = fo.ks_min(X, rho=18.0, axis=1)  # shape (2, 1)
    lse_all = fo.logsumexp(X, rho=5.0, axis=None)

    # Smooth element-wise ops
    s_abs = fo.smooth_abs(X, eps=1e-2)
    s_max = fo.smooth_max(X, 0.3, eps=1e-2)
    s_min = fo.smooth_min(X, 0.6, eps=1e-2)
    s_clip = fo.smooth_clip(X, -0.2, 0.7, eps=1e-2)

    obj = (
        1.2 * ks_all
        + 0.5 * ks_all_cons
        - 0.8 * ks_min_all
        + 0.3 * fo.sum(ks_ax0 * ks_ax0)
        + 0.4 * fo.sum(ks_ax1)
        + 0.2 * lse_all
        + fo.sum(s_abs + s_max - s_min + s_clip * s_clip)
    )

    ev = gb.build_evaluator([obj, ks_all, ks_all_cons, ks_min_all, ks_min_cons])
    x_flat = X0.ravel(order="F")
    res = ev.eval(x_flat)

    # Check KS mathematical bounding properties:
    # max(X) - ln(N)/rho <= ks_max(conservative=False) <= max(X) <= ks_max(conservative=True)
    max_true = float(np.max(X0))
    min_true = float(np.min(X0))
    n_elem = X0.size
    ks_norm_val = float(res[1].item())
    ks_cons_val = float(res[2].item())
    ks_min_norm_val = float(res[3].item())
    ks_min_cons_val = float(res[4].item())

    assert max_true - np.log(n_elem) / 25.0 - 1e-12 <= ks_norm_val <= max_true + 1e-12
    assert ks_cons_val >= max_true - 1e-12
    assert min_true - 1e-12 <= ks_min_norm_val <= min_true + np.log(n_elem) / 20.0 + 1e-12
    assert ks_min_cons_val <= min_true + 1e-12

    # Check 1st and 2nd order AD of combined objective
    ev_obj = gb.build_evaluator(obj)
    g_ad = ev_obj.gradient(x_flat)
    g_fd = fd_gradient(lambda z: ev_obj.eval(z), x_flat)
    assert np.max(np.abs(g_ad - g_fd)) < 1e-7, f"KS Grad err: {np.max(np.abs(g_ad - g_fd))}"

    h_ad = ev_obj.lagrangian_hessian(x_flat, 1.0, np.zeros(0))
    h_fd = fd_hessian_from_grad(lambda z: ev_obj.gradient(z), x_flat)
    assert np.max(np.abs(h_ad - h_fd)) < 1e-5, f"KS Hess err: {np.max(np.abs(h_ad - h_fd))}"

    # Overflow-resistance test: rho = 500.0 with values around 100.0 (rho * x = 50,000 >> 709 exp overflow limit)
    gb_ovf = fo.GraphBuilder()
    v_large_init = np.array([98.5, 100.2, 99.7, 100.1], dtype=np.float64)
    v_large = gb_ovf.variable(shape=(4, 1), init=v_large_init)
    ks_hi = fo.ks_max(v_large, rho=500.0) + fo.ks_min(v_large, rho=500.0)
    ev_ovf = gb_ovf.build_evaluator(ks_hi)
    val_ovf = float(ev_ovf.eval(v_large_init)[0].item())
    grad_ovf = ev_ovf.gradient(v_large_init)
    hess_ovf = ev_ovf.lagrangian_hessian(v_large_init, 1.0, np.zeros(0))
    assert np.isfinite(val_ovf) and np.all(np.isfinite(grad_ovf)) and np.all(np.isfinite(hess_ovf))
    assert np.isclose(val_ovf, 100.2 + 98.5, atol=1e-2)
    print("[PASSED] 2. Overflow-Safe KS (ks_max, ks_min, logsumexp) & Smooth Ops (smooth_abs/max/min/clip) (1st/2nd AD)")


def test_2d_linear_and_bspline_interpolation():
    xg = np.linspace(-1.0, 2.0, 7)
    yg = np.linspace(0.5, 3.5, 8)
    XX, YY = np.meshgrid(xg, yg, indexing="ij")

    # Channel 0: cubic polynomial, Channel 1: trigonometric surface
    Z0 = 0.4 * XX**3 - 0.7 * XX * YY**2 + 1.2 * YY + 0.5
    Z1 = np.sin(0.8 * XX) * np.cos(0.6 * YY) + 0.3 * XX * YY
    Z_table = np.stack([Z0, Z1], axis=-1)  # shape (7, 8, 2)

    gb = fo.GraphBuilder()
    xq0 = np.array([-0.35, 0.85, 1.45], dtype=np.float64)
    yq0 = np.array([1.15, 2.25, 3.05], dtype=np.float64)
    xq = gb.variable(shape=(3, 1), init=xq0, name="xq")
    yq = gb.variable(shape=(3, 1), init=yq0, name="yq")
    # Also test symbolic Z table sensitivity on a (7, 8) single-channel surface
    Z_sym = gb.variable(shape=(7, 8), init=Z0, name="Z_sym")

    lin2d_multi = fo.interp_linear_2d(xq, yq, xg, yg, Z_table, extrapolate=False)
    bsp2d_multi = fo.interp_bspline_2d(xq, yq, xg, yg, Z_table, degree_x=3, degree_y=3, extrapolate=False)
    bsp2d_sym = fo.interp_bspline_2d(xq, yq, xg, yg, Z_sym, degree_x=3, degree_y=3, extrapolate=False)
    lin2d_sym = fo.interp_linear_2d(xq, yq, xg, yg, Z_sym, extrapolate=False)

    # Check exact cubic polynomial reproduction of 2D cubic B-spline at interior query points
    ev_bsp = gb.build_evaluator([bsp2d_multi, bsp2d_sym])
    x_all = np.concatenate([xq0, yq0, Z0.ravel(order="F")])
    bsp_vals, bsp_sym_vals = ev_bsp.eval(x_all)
    exact_cubic = 0.4 * xq0**3 - 0.7 * xq0 * yq0**2 + 1.2 * yq0 + 0.5
    assert np.max(np.abs(bsp_vals[:, 0] - exact_cubic)) < 1e-12
    assert np.max(np.abs(bsp_sym_vals.ravel() - exact_cubic)) < 1e-12

    # Check 1st and 2nd order AD w.r.t. BOTH query points (xq, yq) and table heights Z_sym
    obj = fo.sum(lin2d_multi * bsp2d_multi) + fo.sum(bsp2d_sym * lin2d_sym)
    ev_obj = gb.build_evaluator(obj)
    g_ad = ev_obj.gradient(x_all)
    g_fd = fd_gradient(lambda z: ev_obj.eval(z), x_all)
    assert np.max(np.abs(g_ad - g_fd)) < 1e-6, f"2D Interp Grad err: {np.max(np.abs(g_ad - g_fd))}"

    h_ad = ev_obj.lagrangian_hessian(x_all, 1.0, np.zeros(0))
    h_fd = fd_hessian_from_grad(lambda z: ev_obj.gradient(z), x_all)
    assert np.max(np.abs(h_ad - h_fd)) < 1e-5, f"2D Interp Hess err: {np.max(np.abs(h_ad - h_fd))}"

    # Verify extrapolate=False raises ValueError/RuntimeError on out-of-bounds query
    x_oob = x_all.copy()
    x_oob[0] = -2.5  # below xg[0] = -1.0
    raised = False
    try:
        ev_bsp.eval(x_oob)
    except Exception:
        raised = True
    assert raised, "interp_bspline_2d(extrapolate=False) must raise on out-of-bounds query."

    # Verify extrapolate=True succeeds on out-of-bounds query
    gb_ext = fo.GraphBuilder()
    xq_e = gb_ext.variable(shape=(1, 1), init=[-1.5])
    yq_e = gb_ext.variable(shape=(1, 1), init=[4.0])
    out_e = fo.interp_bspline_2d(xq_e, yq_e, xg, yg, Z0, extrapolate=True)
    ev_ext = gb_ext.build_evaluator(out_e)
    val_ext = float(ev_ext.eval(np.array([-1.5, 4.0]))[0].item())
    expected_ext = 0.4 * (-1.5) ** 3 - 0.7 * (-1.5) * (4.0) ** 2 + 1.2 * 4.0 + 0.5
    assert np.isclose(val_ext, expected_ext, atol=1e-10)
    print("[PASSED] 3. 2D Bilinear & 2D Cubic B-Spline Surface Interpolation (Exact Reproduction & 1st/2nd AD)")


def test_3d_trilinear_interpolation():
    xg = np.linspace(0.0, 2.0, 4)
    yg = np.linspace(-1.0, 1.0, 5)
    zg = np.linspace(0.2, 0.8, 3)
    XX, YY, ZZ = np.meshgrid(xg, yg, zg, indexing="ij")

    # Exact trilinear function: f(x, y, z) = 1.5*x*y*z - 0.8*x*y + 0.4*y*z - 0.3*x + 2.0
    W0 = 1.5 * XX * YY * ZZ - 0.8 * XX * YY + 0.4 * YY * ZZ - 0.3 * XX + 2.0
    W1 = -0.6 * XX * ZZ + 1.1 * YY * ZZ + 0.5 * XX * YY * ZZ
    W_table = np.stack([W0, W1], axis=-1)  # shape (4, 5, 3, 2)

    gb = fo.GraphBuilder()
    xq0 = np.array([0.45, 1.35], dtype=np.float64)
    yq0 = np.array([-0.35, 0.55], dtype=np.float64)
    zq0 = np.array([0.32, 0.68], dtype=np.float64)

    xq = gb.variable(shape=(2, 1), init=xq0)
    yq = gb.variable(shape=(2, 1), init=yq0)
    zq = gb.variable(shape=(2, 1), init=zq0)
    W_sym = gb.variable(shape=(4 * 5 * 3, 1), init=W0.ravel(order="F").reshape(-1, 1))

    w_multi = fo.interp_linear_3d(xq, yq, zq, xg, yg, zg, W_table, extrapolate=False)
    w_sym = fo.interp_linear_3d(xq, yq, zq, xg, yg, zg, W_sym, extrapolate=False)

    ev = gb.build_evaluator([w_multi, w_sym])
    x_all = np.concatenate([xq0, yq0, zq0, W0.ravel(order="F")])
    wm_val, ws_val = ev.eval(x_all)

    exact_w0 = 1.5 * xq0 * yq0 * zq0 - 0.8 * xq0 * yq0 + 0.4 * yq0 * zq0 - 0.3 * xq0 + 2.0
    exact_w1 = -0.6 * xq0 * zq0 + 1.1 * yq0 * zq0 + 0.5 * xq0 * yq0 * zq0
    assert np.max(np.abs(wm_val[:, 0] - exact_w0)) < 1e-13
    assert np.max(np.abs(wm_val[:, 1] - exact_w1)) < 1e-13
    assert np.max(np.abs(ws_val.ravel() - exact_w0)) < 1e-13

    # Verify 1st and 2nd order AD (including cross terms dx*dy, dy*dz, dx*dW)
    obj = fo.sum(w_multi[:, 0:1] * w_multi[:, 1:2] + w_sym * w_sym)
    ev_obj = gb.build_evaluator(obj)
    g_ad = ev_obj.gradient(x_all)
    g_fd = fd_gradient(lambda z: ev_obj.eval(z), x_all)
    assert np.max(np.abs(g_ad - g_fd)) < 1e-6, f"3D Interp Grad err: {np.max(np.abs(g_ad - g_fd))}"

    h_ad = ev_obj.lagrangian_hessian(x_all, 1.0, np.zeros(0))
    h_fd = fd_hessian_from_grad(lambda z: ev_obj.gradient(z), x_all)
    assert np.max(np.abs(h_ad - h_fd)) < 1e-5, f"3D Interp Hess err: {np.max(np.abs(h_ad - h_fd))}"
    print("[PASSED] 4. 3D Trilinear Regular-Grid Interpolation (Exact Trilinear Reproduction & 1st/2nd AD)")


def test_aerodynamic_blade_polar_ks_optimization():
    """
    Spanwise blade twist optimization combining:
      - `fo.atan2(V_axial, Omega * r)` for inflow angle phi(r)
      - `fo.interp_bspline_2d(alpha_deg, Mach, alpha_grid, mach_grid, polar_table)` for C_L and C_D
      - `fo.ks_max(C_L, rho=80.0, conservative=True) <= C_L_max` stall margin envelope constraint
    """
    n_sec = 8
    r_sec = np.linspace(0.25, 0.95, n_sec)
    omega = 120.0
    v_axial = 18.0 + 4.0 * r_sec
    v_tan = omega * r_sec
    v_rel = np.sqrt(v_axial**2 + v_tan**2)
    mach_sec = v_rel / 340.0
    chord_sec = 0.12 * (1.0 - 0.3 * r_sec)
    dr = (0.95 - 0.25) / (n_sec - 1)

    # 2D Airfoil Polar Table over alpha_deg in [-4, 14] and Mach in [0.05, 0.45]
    alpha_grid = np.linspace(-4.0, 14.0, 10)
    mach_grid = np.linspace(0.05, 0.45, 6)
    AA, MM = np.meshgrid(alpha_grid, mach_grid, indexing="ij")
    prandtl_glauert = 1.0 / np.sqrt(1.0 - MM**2)
    CL_tab = 0.11 * AA * prandtl_glauert - 0.0015 * (AA**2)
    CD_tab = (0.008 + 0.00045 * (AA - 1.5) ** 2) * (1.0 + 0.25 * MM**2)
    polar_tab = np.stack([CL_tab, CD_tab], axis=-1)  # shape (10, 6, 2)

    opti = fo.Opti()
    # Pitch angle theta(r) in degrees at each radial station
    theta_init = np.degrees(np.arctan2(v_axial, v_tan)) + 4.0
    theta_deg = opti.variable(shape=(n_sec, 1), init=theta_init, lb=0.0, ub=28.0, name="theta_deg")

    v_ax_p = opti.parameter(shape=(n_sec, 1), value=v_axial)
    v_tan_p = opti.parameter(shape=(n_sec, 1), value=v_tan)
    mach_p = opti.parameter(shape=(n_sec, 1), value=mach_sec)
    q_dyn_c = opti.constant(0.5 * 1.225 * (v_rel**2) * chord_sec * dr, shape=(n_sec, 1))

    # Inflow angle via 4-quadrant atan2
    phi_rad = fo.atan2(v_ax_p, v_tan_p)
    phi_deg = phi_rad * (180.0 / np.pi)
    alpha_deg = theta_deg - phi_deg

    # Smooth 2D B-spline aerodynamic lookup
    polar = fo.interp_bspline_2d(alpha_deg, mach_p, alpha_grid, mach_grid, polar_tab, extrapolate=True)
    cl_sec = polar[:, 0:1]
    cd_sec = polar[:, 1:2]

    # Sectional thrust and torque proxy
    dT = q_dyn_c * (cl_sec * fo.cos(phi_rad) - cd_sec * fo.sin(phi_rad))
    dP = q_dyn_c * (cl_sec * fo.sin(phi_rad) + cd_sec * fo.cos(phi_rad)) * v_tan_p

    total_thrust = fo.sum(dT)
    total_power = fo.sum(dP)

    # Minimize power + pitch smoothness regularization subject to minimum thrust and KS stall envelope
    d_theta = theta_deg[1:n_sec] - theta_deg[0 : n_sec - 1]
    opti.minimize(total_power * 1e-3 + 0.05 * fo.sum(fo.square(d_theta)))
    opti.subject_to(total_thrust >= 190.0)
    # Single scalar KS constraint replacing n_sec pointwise CL <= 0.85 constraints
    cl_ks_max = fo.ks_max(cl_sec, rho=80.0, conservative=True)
    opti.subject_to(cl_ks_max <= 0.85)

    opti.solver("trust-constr", {"gtol": 1e-9, "xtol": 1e-10, "maxiter": 300})
    sol = opti.solve()
    assert sol.success, f"trust-constr failed: {sol.message}"

    thrust_opt = float(sol.value(total_thrust))
    cl_vec_opt = sol.value(cl_sec).ravel()
    cl_ks_opt = float(sol.value(cl_ks_max))

    assert thrust_opt >= 190.0 - 1e-5, f"Thrust constraint violated: {thrust_opt}"
    assert cl_ks_opt <= 0.85 + 1e-5, f"KS max CL constraint violated: {cl_ks_opt}"
    assert np.max(cl_vec_opt) <= cl_ks_opt + 1e-9, "Conservative KS max must upper-bound true max(CL)"
    print(
        f"[PASSED] 5. Spanwise Blade Optimization with atan2 + 2D B-Spline Polar + KS Stall Constraint "
        f"(T={thrust_opt:.2f} N, max(CL)={np.max(cl_vec_opt):.4f} <= KS={cl_ks_opt:.4f})"
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Running fastopt Phase 6 Verification Suite (Math/KS Ops & 2D/3D Interpolation)")
    print("=" * 80)
    test_inverse_trig_hyperbolic_and_atan2()
    test_ks_aggregation_and_smooth_ops()
    test_2d_linear_and_bspline_interpolation()
    test_3d_trilinear_interpolation()
    test_aerodynamic_blade_polar_ks_optimization()
    print("=" * 80)
    print("ALL PHASE 6 TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
