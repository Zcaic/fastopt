"""
Phase 3 verification test suite for fastopt:
1. Verify Opti inherits from GraphBuilder, retaining .variable(), .parameter(), .constant(),
   and .build_evaluator() directly on Opti instances.
2. Verify CasADi IPOPT/MUMPS directory auto-discovery (Path A: find_casadi_ipopt_dir).
3. Verify C++17 native IPOPT solver interface on the classic Hock-Schittkowski HS071 benchmark
   (both with exact 2nd-order Lagrangian Hessian and with limited-memory L-BFGS).
4. Verify constrained nonlinear optimization solved via IPOPT and compare against SLSQP.
All comments are written in English.
"""

import os
import time
import numpy as np
import fastopt as fo
from fastopt.solvers.ipopt_solver import find_casadi_ipopt_dir


# ============================================================================
# 1. Verify Opti Inherits from GraphBuilder (.build_evaluator, .parameter, .constant)
# ============================================================================
def test_opti_inherits_graph_builder():
    assert issubclass(fo.Opti, fo.GraphBuilder)
    opti = fo.Opti()
    assert isinstance(opti, fo.GraphBuilder)

    x = opti.variable(shape=(3, 1), init=[1.0, 2.0, 3.0], lb=-10.0, ub=10.0)
    p = opti.parameter(shape=(3, 1), value=[0.5, 1.5, 2.5])
    c = opti.constant([2.0, 2.0, 2.0], shape=(3, 1))

    expr = fo.dot(x, p) + fo.dot(x, c) + (x ** 2).sum()

    # Directly call build_evaluator() inherited from GraphBuilder
    ev = opti.build_evaluator(expr)
    x0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    val = ev.eval(x0)[0]
    grad = ev.gradient(x0)
    hess = ev.hessian(x0)

    expected_val = (1.0 * 0.5 + 2.0 * 1.5 + 3.0 * 2.5) + (1.0 * 2.0 + 2.0 * 2.0 + 3.0 * 2.0) + 14.0
    expected_grad = np.array([0.5, 1.5, 2.5]) + np.array([2.0, 2.0, 2.0]) + 2.0 * x0
    np.testing.assert_allclose(val.item(), expected_val, rtol=1e-12)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-12)
    np.testing.assert_allclose(hess, 2.0 * np.eye(3), rtol=1e-12)


# ============================================================================
# 2. Verify CasADi IPOPT/MUMPS Directory Auto-Discovery (Path A)
# ============================================================================
def test_casadi_ipopt_dir_discovery():
    discovered_dir = find_casadi_ipopt_dir()
    assert isinstance(discovered_dir, str) and len(discovered_dir) > 0
    assert os.path.exists(discovered_dir)


# ============================================================================
# 3. Classic Hock-Schittkowski HS071 Benchmark via Native C++ IPOPT Solver
# ============================================================================
def test_ipopt_hs071_exact_hessian_and_lbfgs():
    # HS071 Benchmark:
    # min  x0 * x3 * (x0 + x1 + x2) + x2
    # s.t. x0 * x1 * x2 * x3 >= 25
    #      x0^2 + x1^2 + x2^2 + x3^2 == 40
    #      1 <= x0, x1, x2, x3 <= 5
    x_star_ref = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
    f_star_ref = 17.014017289

    for use_hess in [True, False]:
        opti = fo.Opti()
        x = opti.variable(shape=(4, 1), init=[1.0, 5.0, 5.0, 1.0], lb=1.0, ub=5.0)

        obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
        opti.minimize(obj)
        opti.subject_to([
            x[0] * x[1] * x[2] * x[3] >= 25.0,
            x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 == 40.0,
        ])

        opti.solver("ipopt", options={"tol": 1e-9, "max_iter": 200, "print_level": 0}, use_hessian=use_hess)
        sol = opti.solve()

        assert sol.success, f"IPOPT (use_hessian={use_hess}) failed: {sol.message}"
        x_opt = sol.value(x)
        np.testing.assert_allclose(x_opt, x_star_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(sol.f, f_star_ref, rtol=1e-6, atol=1e-6)


# ============================================================================
# 4. IPOPT vs. SLSQP Consistency on Constrained NLP Problem
# ============================================================================
def test_ipopt_vs_slsqp_consistency():
    results = {}
    for solver_name in ["SLSQP", "ipopt"]:
        opti = fo.Opti()
        x = opti.variable(shape=(3, 1), init=[0.2, 0.3, 0.4], lb=-3.0, ub=3.0)
        p_rhs = opti.parameter(shape=1, value=2.0)

        obj = (1.0 - x[0]) ** 2 + 50.0 * ((x[1] - x[0] ** 2) ** 2) + (x[2] - 0.5) ** 2
        opti.minimize(obj)
        opti.subject_to([
            x[0] ** 2 + x[1] ** 2 <= p_rhs,
            x[0] + x[1] + x[2] == 2.0,
        ])

        if solver_name == "SLSQP":
            opti.solver("SLSQP", options={"ftol": 1e-11, "maxiter": 200})
        else:
            opti.solver("ipopt", options={"tol": 1e-10, "max_iter": 200, "print_level": 0})

        sol = opti.solve()
        assert sol.success, f"{solver_name} failed: {sol.message}"
        results[solver_name] = (sol.value(x), sol.f)

    np.testing.assert_allclose(results["SLSQP"][0], results["ipopt"][0], atol=1e-5)
    np.testing.assert_allclose(results["SLSQP"][1], results["ipopt"][1], atol=1e-7)


if __name__ == "__main__":
    t0 = time.perf_counter()
    test_opti_inherits_graph_builder()
    print("[PASSED] 1. Opti Inherits from GraphBuilder (.build_evaluator, .parameter, .constant)")

    test_casadi_ipopt_dir_discovery()
    print("[PASSED] 2. CasADi IPOPT/MUMPS Directory Auto-Discovery (Path A)")

    test_ipopt_hs071_exact_hessian_and_lbfgs()
    print("[PASSED] 3. IPOPT HS071 Benchmark (Exact 2nd-Order Hessian & L-BFGS)")

    test_ipopt_vs_slsqp_consistency()
    print("[PASSED] 4. IPOPT vs. SLSQP Constrained NLP Consistency")

    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"\nAll 4 Phase 3 IPOPT test suites passed in {elapsed:.2f} ms!")
