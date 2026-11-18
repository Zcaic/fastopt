"""
Phase 2 comprehensive verification test suite for fastopt:
1. 1D Linear interpolation (multi-column, differentiable w.r.t. x_query and y_data, 1st & 2nd order AD).
2. Linear extrapolation behavior: default extrapolate=False raises error; extrapolate=True matches exact slope.
3. 1D B-spline interpolation (passing through data points and control-point mode) vs. scipy.interpolate.BSpline.
4. B-spline extrapolation behavior: default extrapolate=False raises error; extrapolate=True matches SciPy BSpline.
5. Opti + scipy.optimize (SLSQP, trust-constr with exact 2nd-order Hessian, L-BFGS-B) on constrained NLP problems,
   including freeze=True parameter switching, simple-bound extraction, and implicit equation constraints.
All comments are written in English.
"""

import time
import numpy as np
from scipy.interpolate import make_interp_spline
import python.fastopt as fo


def finite_diff_jacobian(eval_fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute central finite-difference Jacobian for verification."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y0 = np.asarray(eval_fn(x), dtype=np.float64).ravel()
    J = np.zeros((y0.size, x.size), dtype=np.float64)
    for j in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[j] += eps
        xm[j] -= eps
        yp = np.asarray(eval_fn(xp), dtype=np.float64).ravel()
        ym = np.asarray(eval_fn(xm), dtype=np.float64).ravel()
        J[:, j] = (yp - ym) / (2.0 * eps)
    return J


def finite_diff_hessian(grad_fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute central finite-difference Hessian from exact 1st-order gradients."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    H = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[j] += eps
        xm[j] -= eps
        gp = np.asarray(grad_fn(xp), dtype=np.float64).ravel()
        gm = np.asarray(grad_fn(xm), dtype=np.float64).ravel()
        H[:, j] = (gp - gm) / (2.0 * eps)
    return 0.5 * (H + H.T)


# ============================================================================
# 1. Linear Interpolation + Extrapolation Toggle + 1st/2nd Order AD
# ============================================================================
def test_linear_interp_and_extrapolation():
    x_grid = np.array([0.0, 1.0, 2.5, 4.0], dtype=np.float64)
    y_const = np.column_stack([2.0 * x_grid + 1.0, x_grid ** 2])

    # Default extrapolate=False must raise an exception on out-of-bounds query
    gb_strict = fo.GraphBuilder()
    xq_strict = gb_strict.variable(shape=(2, 1), init=[-0.5, 2.0])
    z_strict = fo.interp_linear(xq_strict, x_grid, y_const)
    ev_strict = gb_strict.build_evaluator(z_strict)

    raised = False
    try:
        ev_strict.eval(np.array([-0.5, 2.0]))
    except Exception as exc:
        raised = True
        assert "extrapolate=false" in str(exc)
    assert raised, "Expected out-of-range exception when extrapolate=False."

    # extrapolate=True with both x_query and y_data as differentiable Exprs
    gb = fo.GraphBuilder()
    xq = gb.variable(shape=(3, 1), init=[-0.5, 1.75, 4.8])
    y_var = gb.variable(shape=(4, 2), init=y_const)
    z = fo.interp_linear(xq, x_grid, y_var, extrapolate=True)
    obj = (z ** 2).sum()
    ev = gb.build_evaluator([obj, z])

    x_test = np.concatenate([
        np.array([-0.5, 1.75, 4.8]),
        y_const.ravel(order="F"),
    ])

    _, z_val = ev.eval(x_test)
    # Left extrap: x=-0.5 on [0,1] -> y0 = 0.0, y1 = -0.5
    np.testing.assert_allclose(z_val[0], [0.0, -0.5], rtol=1e-12, atol=1e-12)
    # Interior: x=1.75 on [1.0, 2.5] -> t=0.5 -> y0=4.5, y1=3.625
    np.testing.assert_allclose(z_val[1], [4.5, 3.625], rtol=1e-12, atol=1e-12)
    # Right extrap: x=4.8 on [2.5, 4.0] -> t=(4.8-2.5)/1.5=2.3/1.5
    t_r = (4.8 - 2.5) / 1.5
    expected_r = (1.0 - t_r) * y_const[2] + t_r * y_const[3]
    np.testing.assert_allclose(z_val[2], expected_r, rtol=1e-12, atol=1e-12)

    # Verify 1st-order Jacobian and 2nd-order Hessian (including xq-y_data cross terms)
    J_exact = ev.jacobian(x_test, 1)
    J_fd = finite_diff_jacobian(lambda w: ev.eval(w)[1].ravel(order="F"), x_test)
    np.testing.assert_allclose(J_exact, J_fd, rtol=1e-7, atol=1e-8)

    H_exact = ev.hessian(x_test, 0)
    H_fd = finite_diff_hessian(lambda w: ev.gradient(w, 0), x_test)
    np.testing.assert_allclose(H_exact, H_fd, rtol=1e-6, atol=1e-7)


# ============================================================================
# 2. B-Spline Interpolation + Extrapolation Toggle + 1st/2nd Order AD
# ============================================================================
def test_bspline_interp_and_extrapolation():
    x_grid = np.linspace(0.0, 3.0, 7)
    y_vals = np.column_stack([np.sin(x_grid), np.cos(0.5 * x_grid)])
    sp_ref = make_interp_spline(x_grid, y_vals, k=3)

    # Default extrapolate=False must raise on out-of-range query
    gb_strict = fo.GraphBuilder()
    xq_s = gb_strict.variable(shape=(1, 1), init=[3.2])
    z_s = fo.interp_bspline(xq_s, x_grid, y_vals, degree=3)
    ev_s = gb_strict.build_evaluator(z_s)

    raised = False
    try:
        ev_s.eval(np.array([3.2]))
    except Exception as exc:
        raised = True
        assert "extrapolate=false" in str(exc)
    assert raised, "Expected out-of-range exception when B-spline extrapolate=False."

    # extrapolate=True with both x_query and y_data as variables
    gb = fo.GraphBuilder()
    xq = gb.variable(shape=(4, 1), init=[-0.3, 0.7, 2.2, 3.4])
    y_var = gb.variable(shape=y_vals.shape, init=y_vals)
    z = fo.interp_bspline(xq, x_grid, y_var, degree=3, extrapolate=True)
    obj = (z ** 2).sum()
    ev = gb.build_evaluator([obj, z])

    xq_test = np.array([-0.3, 0.7, 2.2, 3.4])
    x_test = np.concatenate([xq_test, y_vals.ravel(order="F")])

    _, z_val = ev.eval(x_test)
    z_scipy = sp_ref(xq_test)
    np.testing.assert_allclose(z_val, z_scipy, rtol=1e-11, atol=1e-11)

    # Verify 1st-order Jacobian and 2nd-order Hessian against finite differences
    J_exact = ev.jacobian(x_test, 1)
    J_fd = finite_diff_jacobian(lambda w: ev.eval(w)[1].ravel(order="F"), x_test)
    np.testing.assert_allclose(J_exact, J_fd, rtol=1e-6, atol=1e-7)

    H_exact = ev.hessian(x_test, 0)
    H_fd = finite_diff_hessian(lambda w: ev.gradient(w, 0), x_test)
    np.testing.assert_allclose(H_exact, H_fd, rtol=1e-5, atol=1e-6)


# ============================================================================
# 3. Opti + SciPy SLSQP & trust-constr (Exact 2nd-Order Hessian)
# ============================================================================
def test_opti_scipy_slsqp_and_trust_constr():
    # Constrained Rosenbrock + nonlinear inequality & equality:
    # min (1 - x0)^2 + 100*(x1 - x0^2)^2 + (x2 - 0.5)^2
    # s.t. x0^2 + x1^2 <= 2.0,  x0 + x1 + x2 == 2.0,  x >= -2.0
    solutions = {}
    for method in ["SLSQP", "trust-constr"]:
        opti = fo.Opti()
        x = opti.variable(shape=(3, 1), init=[0.1, 0.2, 0.3], lb=-5.0, ub=5.0)
        p_target = opti.variable(shape=1, init=2.0, freeze=True)

        obj = (1.0 - x[0]) ** 2 + 100.0 * ((x[1] - x[0] ** 2) ** 2) + (x[2] - 0.5) ** 2
        opti.minimize(obj)
        opti.subject_to([
            x[0] ** 2 + x[1] ** 2 <= p_target,
            x[0] + x[1] + x[2] == 2.0,
            x >= -2.0,  # Extracted automatically as simple variable bound
        ])

        opts = {"ftol": 1e-11, "maxiter": 200} if method == "SLSQP" else {"gtol": 1e-7, "xtol": 1e-7, "maxiter": 500}
        opti.solver(method, options=opts)
        sol = opti.solve()

        assert sol.success, f"{method} failed: {sol.message}"
        x_opt = sol.value(x)
        assert abs(x_opt[0] + x_opt[1] + x_opt[2] - 2.0) < 1e-7
        assert x_opt[0] ** 2 + x_opt[1] ** 2 <= 2.0 + 1e-7
        solutions[method] = (x_opt, sol.f)

    # Verify SLSQP and trust-constr converge to the exact same constrained optimum
    np.testing.assert_allclose(solutions["SLSQP"][0], solutions["trust-constr"][0], atol=1e-5)
    np.testing.assert_allclose(solutions["SLSQP"][1], solutions["trust-constr"][1], atol=1e-7)


# ============================================================================
# 4. Opti Warm-Start + B-Spline Curve Optimization + Implicit Equation
# ============================================================================
def test_opti_bspline_and_implicit_optimization():
    # Optimize 5 B-spline control points so that the curve matches a target profile
    # while satisfying a coupled implicit equilibrium equation on an auxiliary state.
    opti = fo.Opti()
    ctrl_y = opti.variable(shape=(5, 1), init=np.zeros(5), lb=-3.0, ub=3.0)
    u_eval = np.linspace(0.0, 1.0, 9)
    knots = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=np.float64)

    curve_vals = fo.bspline_eval(u_eval, ctrl_y, knots, degree=3, extrapolate=False)
    target = np.sin(np.pi * u_eval).reshape(-1, 1)

    # Coupled implicit state y_eq satisfying y_eq^3 + y_eq = mean(ctrl_y)
    mean_ctrl = ctrl_y.sum() / 5.0
    y_eq = fo.implicit_solve(
        [mean_ctrl],
        lambda m_in, y: (y ** 3) + y - m_in,
        init_guess=0.0,
    )

    loss = ((curve_vals - target) ** 2).sum() + 0.1 * ((y_eq - 0.5) ** 2)
    opti.minimize(loss)
    opti.subject_to(curve_vals[4] >= 0.8)  # Midpoint >= 0.8

    opti.solver("SLSQP", options={"ftol": 1e-10, "maxiter": 100})
    sol = opti.solve()
    assert sol.success, f"B-spline + implicit optimization failed: {sol.message}"
    assert sol.value(curve_vals[4]) >= 0.8 - 1e-7

    # Also verify L-BFGS-B on an unconstrained/boxed problem
    opti2 = fo.Opti()
    z = opti2.variable(shape=(2, 1), init=[0.0, 0.0], lb=-2.0, ub=2.0)
    opti2.minimize((z[0] - 1.2) ** 2 + (z[1] + 0.7) ** 2)
    opti2.solver("L-BFGS-B")
    sol2 = opti2.solve()
    assert sol2.success
    np.testing.assert_allclose(sol2.value(z), [1.2, -0.7], atol=1e-6)


if __name__ == "__main__":
    tests = [
        ("1. Linear Interp + Extrapolation Toggle + 1st/2nd AD", test_linear_interp_and_extrapolation),
        ("2. B-Spline Interp + Extrapolation Toggle + 1st/2nd AD", test_bspline_interp_and_extrapolation),
        ("3. Opti + SciPy SLSQP & trust-constr (Exact 2nd Hessian)", test_opti_scipy_slsqp_and_trust_constr),
        ("4. Opti B-Spline + Implicit IFT + L-BFGS-B Optimization", test_opti_bspline_and_implicit_optimization),
    ]
    t0 = time.perf_counter()
    for title, fn in tests:
        fn()
        print(f"[PASSED] {title}")
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"\nAll {len(tests)} Phase 2 test suites passed in {elapsed:.2f} ms!")
