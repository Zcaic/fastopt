"""
Phase 1 comprehensive unit and finite-difference verification test suite for fastopt.
Tests:
1. Unified variable(..., freeze=False/True) routing, contiguous memory layout, and parameter folding.
2. Elementary unary and binary tensor operators with broadcasting (0th, 1st, and 2nd order AD).
3. Matrix multiplication, dot product, transpose, slicing, concatenation, and neural network layers.
4. Conditional branching: hard where() with NaN-safe gradient/Hessian isolation, smooth_where(), and short-circuit cond().
5. Nested ImplicitSolverNode (R(x, y) = 0) with exact 1st-order and 2nd-order Implicit Function Theorem (IFT) derivatives.
6. OpenMP multi-threaded population batch evaluation (eval_batch).
All comments are written in English.
"""

import time
import numpy as np
import fastopt as fo


# ============================================================================
# Finite-Difference Reference Helpers
# ============================================================================
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
    """Compute central finite-difference Hessian from exact analytical 1st-order gradients."""
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
# 1. Unified Variable / Parameter Routing via freeze=False/True
# ============================================================================
def test_unified_freeze_variable_and_parameter():
    gb = fo.GraphBuilder()
    # Active optimization variable (freeze=False)
    x = gb.variable(shape=(3, 1), init=[1.0, 2.0, 3.0], lb=-5.0, ub=5.0, freeze=False, name="x")
    # Frozen parameter via the exact same interface (freeze=True ignores lb/ub)
    p = gb.variable(shape=(3, 1), init=[10.0, 20.0, 30.0], lb=-5.0, ub=5.0, freeze=True, name="p")

    assert not x.is_frozen and x.requires_grad
    assert p.is_frozen and not p.requires_grad

    # Pure parameter subgraph should be marked is_param_only
    p_scaled = fo.sin(p) * 2.0
    assert p_scaled.is_param_only and not p_scaled.requires_grad

    f = fo.dot(x, p_scaled) + (x ** 2).sum()
    ev = gb.build_evaluator(f)

    # Decision vector only contains active (un-frozen) variables
    assert ev.num_vars() == 3
    x0 = gb.cpp_graph.initial_decision_vector()
    np.testing.assert_allclose(x0, [1.0, 2.0, 3.0])

    grad1 = ev.gradient(x0)
    expected_grad1 = 2.0 * np.sin([10.0, 20.0, 30.0]) + 2.0 * x0
    np.testing.assert_allclose(grad1, expected_grad1, rtol=1e-12, atol=1e-12)

    # Mutate frozen parameter at runtime without rebuilding the graph
    p.set_value([0.5, 1.5, 2.5])
    grad2 = ev.gradient(x0)
    expected_grad2 = 2.0 * np.sin([0.5, 1.5, 2.5]) + 2.0 * x0
    np.testing.assert_allclose(grad2, expected_grad2, rtol=1e-12, atol=1e-12)


# ============================================================================
# 2. Elementary Operators, Broadcasting, 1st-Order Grad & 2nd-Order Hessian
# ============================================================================
def test_elementary_ops_and_second_order_hessian():
    gb = fo.GraphBuilder()
    x = gb.variable(shape=(4, 1), init=[0.4, 0.8, 1.2, 1.6])

    # Nonlinear expression combining trig, exp, log, sqrt, tanh, pow, and broadcasting
    a = x[0:2]
    b = x[2:4]
    expr_vec = fo.sin(a) * fo.exp(0.5 * b) + fo.log(a + 1.5) / fo.sqrt(b + 1.0) + fo.tanh(a * b) + (a ** 2.5)
    obj = expr_vec.sum() + (x[0] * x[3] - x[1] / (x[2] + 2.0)) ** 2

    ev = gb.build_evaluator([obj, expr_vec])
    x_test = np.array([0.35, 0.75, 1.15, 1.55], dtype=np.float64)

    # Verify 1st-order gradient and Jacobian against central finite differences
    g_exact = ev.gradient(x_test, 0)
    g_fd = finite_diff_jacobian(lambda z: ev.eval(z)[0], x_test).ravel()
    np.testing.assert_allclose(g_exact, g_fd, rtol=1e-7, atol=1e-8)

    J_exact = ev.jacobian(x_test, 1)
    J_fd = finite_diff_jacobian(lambda z: ev.eval(z)[1].ravel(order="F"), x_test)
    np.testing.assert_allclose(J_exact, J_fd, rtol=1e-7, atol=1e-8)

    # Verify 2nd-order Lagrangian Hessian against finite differences of exact gradients
    lam = np.array([0.7, -1.3], dtype=np.float64)
    sigma = 1.4
    H_exact = ev.lagrangian_hessian(x_test, sigma, [lam])

    def lag_grad(z):
        return sigma * ev.gradient(z, 0) + ev.jacobian(z, 1).T @ lam

    H_fd = finite_diff_hessian(lag_grad, x_test)
    np.testing.assert_allclose(H_exact, H_fd, rtol=1e-6, atol=1e-7)


# ============================================================================
# 3. Matrix Multiplication, Slicing, Concatenation & Neural Network Layer
# ============================================================================
def test_neural_network_and_tensor_ops():
    rng = np.random.default_rng(42)
    W1_val = rng.normal(0.0, 0.5, size=(8, 4))
    b1_val = rng.normal(0.0, 0.2, size=(8, 1))
    W2_val = rng.normal(0.0, 0.5, size=(3, 8))
    b2_val = rng.normal(0.0, 0.2, size=(3, 1))

    gb = fo.GraphBuilder()
    x1 = gb.variable(shape=(2, 1), init=[0.2, -0.3])
    x2 = gb.variable(shape=(2, 1), init=[0.5, 0.1])

    # Frozen neural network weights (zero adjoint memory overhead)
    W1 = gb.variable(shape=(8, 4), init=W1_val, freeze=True)
    b1 = gb.variable(shape=(8, 1), init=b1_val, freeze=True)
    W2 = gb.variable(shape=(3, 8), init=W2_val, freeze=True)
    b2 = gb.variable(shape=(3, 1), init=b2_val, freeze=True)

    x_in = fo.concat([x1, x2], axis=0)
    h1 = fo.tanh(W1 @ x_in + b1)
    y_out = W2 @ h1 + b2
    loss = fo.dot(y_out, y_out) + (x_in.T @ x_in)[0, 0] * 0.1

    ev = gb.build_evaluator([loss, y_out])
    x_vec = np.array([0.25, -0.35, 0.45, 0.15], dtype=np.float64)

    # Check forward value against NumPy
    val_loss, val_y = ev.eval(x_vec)
    np_h1 = np.tanh(W1_val @ x_vec.reshape(4, 1) + b1_val)
    np_y = W2_val @ np_h1 + b2_val
    np.testing.assert_allclose(val_y, np_y, rtol=1e-12, atol=1e-12)

    # Check 1st-order Jacobian and 2nd-order Hessian
    J_exact = ev.jacobian(x_vec, 1)
    J_fd = finite_diff_jacobian(lambda z: ev.eval(z)[1].ravel(order="F"), x_vec)
    np.testing.assert_allclose(J_exact, J_fd, rtol=1e-7, atol=1e-8)

    H_exact = ev.hessian(x_vec, 0)
    H_fd = finite_diff_hessian(lambda z: ev.gradient(z, 0), x_vec)
    np.testing.assert_allclose(H_exact, H_fd, rtol=1e-6, atol=1e-7)


# ============================================================================
# 4. Branching Operators: NaN-Safe where(), smooth_where(), and Short-Circuit cond()
# ============================================================================
def test_branching_nan_safety_and_smooth_where():
    gb = fo.GraphBuilder()
    x = gb.variable(shape=(2, 1), init=[-1.0, 2.0])

    # Inactive branch sqrt(x) and log(x) will produce NaN when x < 0!
    # Our C++ WhereNode must strictly isolate the inactive branch in forward, grad, and Hessian.
    safe_branch = fo.where(x > 0.0, fo.sqrt(x) + fo.log(x), x ** 2 + 3.0 * x)
    smooth_branch = fo.smooth_where(x, x ** 2, -x, threshold=0.0, sharpness=5.0)

    # Also test short-circuit subgraph cond()
    cond_out = fo.cond(
        x[1] > 0.0,
        [x],
        lambda v: (v ** 3).sum(),
        lambda v: fo.sin(v).sum(),
    )

    total_obj = safe_branch.sum() + smooth_branch.sum() + cond_out
    ev = gb.build_evaluator(total_obj)

    x_test = np.array([-1.5, 2.0], dtype=np.float64)
    val = ev.eval(x_test)[0]
    grad = ev.gradient(x_test)
    H = ev.hessian(x_test)

    assert np.all(np.isfinite(val))
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(H))

    grad_fd = finite_diff_jacobian(lambda z: ev.eval(z)[0], x_test).ravel()
    H_fd = finite_diff_hessian(lambda z: ev.gradient(z), x_test)
    np.testing.assert_allclose(grad, grad_fd, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(H, H_fd, rtol=1e-6, atol=1e-7)


# ============================================================================
# 5. Implicit Function Variable (Nested Newton + 1st & 2nd Order IFT)
# ============================================================================
def test_implicit_solver_node_ift_first_and_second_order():
    gb = fo.GraphBuilder()
    x = gb.variable(shape=(2, 1), init=[1.2, 0.8])

    # Coupled 2D nonlinear implicit residual system R(x, y) = 0:
    # R0 = y0^3 + y0 * x0 + 0.2 * y1 - x1 = 0
    # R1 = y1 + exp(0.5 * y1) - x0 * y0 - x1^2 = 0
    def residual_fn(x_in, y_state):
        r0 = (y_state[0] ** 3) + y_state[0] * x_in[0] + 0.2 * y_state[1] - x_in[1]
        r1 = y_state[1] + fo.exp(0.5 * y_state[1]) - x_in[0] * y_state[0] - (x_in[1] ** 2)
        return fo.concat([r0, r1], axis=0)

    y_sol = fo.implicit_solve(
        inputs=[x],
        residual_fn=residual_fn,
        init_guess=[0.5, 0.5],
        lb=[-10.0, -10.0],
        ub=[10.0, 10.0],
    )

    obj = (y_sol[0] ** 2) * x[0] + fo.sin(y_sol[1]) * x[1]
    ev = gb.build_evaluator([obj, y_sol])

    x_test = np.array([1.3, 0.9], dtype=np.float64)
    _, y_val = ev.eval(x_test)
    y0, y1 = float(y_val[0, 0]), float(y_val[1, 0])

    # Verify residual is solved to machine precision
    res0 = y0 ** 3 + y0 * x_test[0] + 0.2 * y1 - x_test[1]
    res1 = y1 + np.exp(0.5 * y1) - x_test[0] * y0 - x_test[1] ** 2
    assert abs(res0) < 1e-10 and abs(res1) < 1e-10

    # Verify 1st-order IFT gradient and Jacobian against finite differences
    grad_ift = ev.gradient(x_test, 0)
    grad_fd = finite_diff_jacobian(lambda z: ev.eval(z)[0], x_test).ravel()
    np.testing.assert_allclose(grad_ift, grad_fd, rtol=1e-6, atol=1e-7)

    J_ift = ev.jacobian(x_test, 1)
    J_fd = finite_diff_jacobian(lambda z: ev.eval(z)[1].ravel(order="F"), x_test)
    np.testing.assert_allclose(J_ift, J_fd, rtol=1e-6, atol=1e-7)

    # Verify 2nd-order IFT Hessian against finite differences of exact IFT gradients
    H_ift = ev.hessian(x_test, 0)
    H_fd = finite_diff_hessian(lambda z: ev.gradient(z, 0), x_test)
    np.testing.assert_allclose(H_ift, H_fd, rtol=1e-5, atol=1e-6)


# ============================================================================
# 6. OpenMP Population Batch Evaluation (eval_batch)
# ============================================================================
def test_openmp_eval_batch():
    gb = fo.GraphBuilder()
    x = gb.variable(shape=(5, 1), init=np.zeros(5))
    f = (fo.sin(x) + (x ** 2)).sum()
    g = x[0:3] * x[2:5]

    ev = gb.build_evaluator([f, g])
    rng = np.random.default_rng(123)
    X_pop = rng.uniform(-2.0, 2.0, size=(200, 5))

    batch_f, batch_g = ev.eval_batch(X_pop, num_threads=4)
    assert batch_f.shape == (200, 1)
    assert batch_g.shape == (200, 3)

    for i in range(20):
        fi, gi = ev.eval(X_pop[i])
        np.testing.assert_allclose(batch_f[i, 0], fi[0, 0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch_g[i], gi.ravel(order="F"), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    tests = [
        ("1. Unified freeze=False/True Variable/Parameter Routing", test_unified_freeze_variable_and_parameter),
        ("2. Elementary Tensor Ops, Broadcasting, 1st Grad & 2nd Hessian", test_elementary_ops_and_second_order_hessian),
        ("3. Matrix Mul, Slicing, Concat & Neural Network AD", test_neural_network_and_tensor_ops),
        ("4. Branching Ops (NaN-safe where, smooth_where, short-circuit cond)", test_branching_nan_safety_and_smooth_where),
        ("5. ImplicitSolverNode (Newton + 1st & 2nd Order IFT)", test_implicit_solver_node_ift_first_and_second_order),
        ("6. OpenMP Population Batch Evaluation (eval_batch)", test_openmp_eval_batch),
    ]
    t0 = time.perf_counter()
    for title, fn in tests:
        fn()
        print(f"[PASSED] {title}")
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"\nAll {len(tests)} Phase 1 test suites passed in {elapsed:.2f} ms!")
