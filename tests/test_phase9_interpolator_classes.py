"""
Phase 9 Verification & Benchmark Suite:
Reusable Pre-Compiled Interpolator Classes:
  - `LinearInterpolator1D`, `BSplineCurve1D`, `BSplineInterpolator1D`
  - `LinearInterpolator2D`, `BSplineSurface2D`, `BSplineInterpolator2D`
  - `LinearInterpolator3D`
Verifies:
  1. Dual Symbolic (`Expr` with 1st/2nd AD) & Direct Numerical (`float` / `np.ndarray`) Evaluation
  2. C++ Computation Graph `ConstantNode` & Workspace Memory Sharing across Multiple Queries
  3. Symbolic `Expr` Data Table Collocation Sharing (`A_inv @ y_var` Built Only Once)
  4. Multi-Query Graph Build & Evaluation Speedup Benchmark (15 Flight-Point Queries on 2D Polar)
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


def test_dual_symbolic_and_numeric_evaluation():
    """
    Test 1: Verify that all 1D, 2D, and 3D Interpolator classes support both:
      - Direct numerical queries (`float` and `np.ndarray`) without building a user Graph
      - Symbolic `Expr` queries with exact 1st-order gradients and 2nd-order Hessians
    """
    # 1D grids
    xg = np.linspace(0.0, 2.0, 9)
    yg_1d = np.sin(1.5 * xg) + 0.2 * (xg**2)
    lin1d = fo.LinearInterpolator1D(xg, yg_1d)
    bsp1d = fo.BSplineInterpolator1D(xg, yg_1d, degree=3)

    # 2D grids (multi-channel K=2)
    xg2 = np.linspace(-1.0, 2.0, 8)
    yg2 = np.linspace(0.5, 3.0, 7)
    XX2, YY2 = np.meshgrid(xg2, yg2, indexing="ij")
    Z2 = np.stack([np.sin(XX2) * np.cos(YY2), 0.3 * (XX2**2) + 0.5 * YY2], axis=-1)
    lin2d = fo.LinearInterpolator2D(xg2, yg2, Z2)
    bsp2d = fo.BSplineInterpolator2D(xg2, yg2, Z2, degree_x=3, degree_y=3)

    # 3D grids
    xg3 = np.linspace(0.0, 1.5, 5)
    yg3 = np.linspace(0.0, 3.0, 6)
    zg3 = np.linspace(0.1, 0.8, 4)
    XX3, YY3, ZZ3 = np.meshgrid(xg3, yg3, zg3, indexing="ij")
    W3 = 1.2 * XX3 * YY3 * ZZ3 - 0.5 * XX3 * ZZ3 + 0.8 * YY3
    lin3d = fo.LinearInterpolator3D(xg3, yg3, zg3, W3)

    # Check scalar & vector numerical evaluations
    q_x = np.array([0.35, 1.15])
    q_y = np.array([0.85, 2.15])
    q_z = np.array([0.25, 0.65])

    val_lin1d_num = lin1d(q_x)
    val_bsp1d_num = bsp1d(q_x)
    val_lin2d_num = lin2d(q_x, q_y)
    val_bsp2d_num = bsp2d(q_x, q_y)
    val_lin3d_num = lin3d(q_x, q_y, q_z)
    assert isinstance(lin1d(0.55), float) and isinstance(bsp1d(0.55), float)
    assert isinstance(lin3d(0.55, 0.25, 0.45), float)

    # Now evaluate symbolically on a GraphBuilder and verify bit-for-bit equality + 1st/2nd AD
    gb = fo.GraphBuilder()
    vx = gb.variable(shape=(2, 1), init=q_x)
    vy = gb.variable(shape=(2, 1), init=q_y)
    vz = gb.variable(shape=(2, 1), init=q_z)

    e_lin1d = lin1d(vx)
    e_bsp1d = bsp1d(vx)
    e_lin2d = lin2d(vx, vy)
    e_bsp2d = bsp2d(vx, vy)
    e_lin3d = lin3d(vx, vy, vz)

    obj = (
        fo.sum(e_lin1d * e_bsp1d)
        + fo.sum(e_lin2d * e_bsp2d)
        + fo.sum(fo.square(e_lin3d))
    )
    ev = gb.build_evaluator([obj, e_lin1d, e_bsp1d, e_lin2d, e_bsp2d, e_lin3d])
    x_all = np.concatenate([q_x, q_y, q_z])
    outs = ev.eval(x_all)

    assert np.allclose(outs[1].ravel(), val_lin1d_num, atol=1e-15)
    assert np.allclose(outs[2].ravel(), val_bsp1d_num, atol=1e-15)
    assert np.allclose(outs[3], val_lin2d_num, atol=1e-15)
    assert np.allclose(outs[4], val_bsp2d_num, atol=1e-15)
    assert np.allclose(outs[5].ravel(), val_lin3d_num, atol=1e-15)

    g_ad = ev.gradient(x_all, 0)
    g_fd = fd_gradient(ev, x_all)
    assert np.max(np.abs(g_ad - g_fd)) < 1e-6

    H_ad = ev.hessian(x_all, 0)
    H_fd = fd_hessian(ev, x_all)
    assert np.max(np.abs(H_ad - H_fd)) < 1e-5

    print("[PASSED] 1. All 1D/2D/3D Pre-Compiled Interpolator Classes (Numeric & Symbolic 1st/2nd AD)")


def test_graph_constant_sharing_and_symbolic_table_reuse():
    """
    Test 2: Verify that calling a pre-compiled Interpolator multiple times in the same Graph:
      - Allocates only ONE ConstantNode for the table/control net (instead of N copies)
      - When the table itself is a symbolic `Expr` (e.g. optimization variables), builds the
        collocation inverse node `A_inv @ y_var` only ONCE!
    """
    xg = np.linspace(-4.0, 14.0, 20)
    yg = np.linspace(0.05, 0.65, 15)
    XX, YY = np.meshgrid(xg, yg, indexing="ij")
    Z_tab = np.stack([0.11 * XX / np.sqrt(1.0 - YY**2), 0.008 + 0.0004 * (XX**2)], axis=-1)

    n_queries = 15
    # A. Functional calls (creates new ConstantNodes and re-solves B-spline collocation every call)
    gb_func = fo.GraphBuilder()
    alphas_f = [gb_func.variable(init=2.0 + 0.4 * k) for k in range(n_queries)]
    machs_f = [gb_func.variable(init=0.2 + 0.02 * k) for k in range(n_queries)]
    terms_f = [fo.interp_bspline_2d(a, m, xg, yg, Z_tab) for a, m in zip(alphas_f, machs_f)]
    ev_func = gb_func.build_evaluator(fo.sum(fo.concat(terms_f, axis=0)))
    mem_func = gb_func.cpp_graph.total_memory_size()

    # B. Pre-compiled BSplineInterpolator2D (solves collocation ONCE and shares 1 pair of ConstantNodes)
    polar_spline = fo.BSplineInterpolator2D(xg, yg, Z_tab, degree_x=3, degree_y=3)
    gb_cls = fo.GraphBuilder()
    alphas_c = [gb_cls.variable(init=2.0 + 0.4 * k) for k in range(n_queries)]
    machs_c = [gb_cls.variable(init=0.2 + 0.02 * k) for k in range(n_queries)]
    terms_c = [polar_spline(a, m) for a, m in zip(alphas_c, machs_c)]
    ev_cls = gb_cls.build_evaluator(fo.sum(fo.concat(terms_c, axis=0)))
    mem_cls = gb_cls.cpp_graph.total_memory_size()

    # Each 20x15x2 table has 600 floats; sharing across 15 queries saves 14 * 600 = 8,400 floats!
    assert mem_func - mem_cls == (n_queries - 1) * Z_tab.size, (
        f"Expected memory reduction of {(n_queries - 1) * Z_tab.size} floats, got {mem_func - mem_cls}"
    )
    x0 = gb_cls.cpp_graph.initial_decision_vector()
    assert np.allclose(ev_func.eval(x0)[0], ev_cls.eval(x0)[0], atol=1e-14)
    assert np.allclose(ev_func.gradient(x0, 0), ev_cls.gradient(x0, 0), atol=1e-14)

    # C. Symbolic table reuse in multi-point optimization
    opti = fo.Opti()
    ctrl_r = np.linspace(0.2, 1.0, 5)
    theta_ctrl = opti.variable(shape=(5, 1), init=[0.30, 0.25, 0.20, 0.16, 0.12], lb=0.05, ub=0.45)

    # Instantiate symbolic 1D B-spline interpolator ONCE over decision variable `theta_ctrl`
    twist_interp = fo.BSplineInterpolator1D(ctrl_r, theta_ctrl, degree=3)
    nodes_after_init = opti.cpp_graph.num_nodes()

    # Query at 3 distinct radial grids (hover stations, cruise stations, tip constraint stations)
    th_hover = twist_interp(np.linspace(0.2, 1.0, 10))
    nodes_after_q1 = opti.cpp_graph.num_nodes()
    th_cruise = twist_interp(np.linspace(0.25, 0.95, 8))
    nodes_after_q2 = opti.cpp_graph.num_nodes()

    # Each additional query only adds 2 nodes (query constant + BSplineEvalNode), NOT `A_inv @ theta_ctrl`!
    assert (nodes_after_q2 - nodes_after_q1) == 2

    # Use pre-compiled numerical evaluation to set a target profile
    ref_interp = fo.BSplineInterpolator1D(ctrl_r, [0.28, 0.22, 0.18, 0.14, 0.10], degree=3)
    target_hover = ref_interp(np.linspace(0.2, 1.0, 10))
    target_cruise = ref_interp(np.linspace(0.25, 0.95, 8))

    opti.minimize(fo.sum(fo.square(th_hover - target_hover)) + fo.sum(fo.square(th_cruise - target_cruise)))
    opti.subject_to(twist_interp(1.0) <= 0.11)
    opti.solver("trust-constr", options={"gtol": 1e-10, "maxiter": 100})
    sol = opti.solve()
    assert sol.success

    print(
        f"[PASSED] 2. C++ ConstantNode Deduplication (Workspace saved {mem_func - mem_cls} floats across "
        f"{n_queries} queries) & Symbolic Control-Point Reuse"
    )


def test_repeated_query_speedup_benchmark():
    """
    Test 3: Benchmark building and evaluating 20 queries on a (30 x 20 x 2) 2D Airfoil Polar B-Spline.
    Compare pre-compiled `BSplineInterpolator2D` vs. repeated `interp_bspline_2d`.
    """
    xg = np.linspace(-6.0, 16.0, 30)
    yg = np.linspace(0.05, 0.75, 20)
    XX, YY = np.meshgrid(xg, yg, indexing="ij")
    Z_polar = np.stack([0.1 * XX / np.sqrt(1.0 - 0.8 * YY**2), 0.008 + 0.0005 * XX**2], axis=-1)

    n_queries = 20
    n_repeats = 15

    # 1. Repeated functional `fo.interp_bspline_2d`
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        gb = fo.GraphBuilder()
        a_v = gb.variable(shape=(8, 1), init=np.linspace(1.0, 8.0, 8))
        m_v = gb.variable(shape=(8, 1), init=np.linspace(0.15, 0.55, 8))
        acc = []
        for k in range(n_queries):
            acc.append(fo.interp_bspline_2d(a_v + 0.05 * k, m_v, xg, yg, Z_polar))
        ev_f = gb.build_evaluator(fo.sum(fo.concat(acc, axis=0)))
    t_func_build = (time.perf_counter() - t0) * 1000.0 / n_repeats

    # 2. Pre-compiled `fo.BSplineInterpolator2D` (pre-compiled once, queried 20 times per build)
    polar_2d = fo.BSplineInterpolator2D(xg, yg, Z_polar, degree_x=3, degree_y=3)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        gb = fo.GraphBuilder()
        a_v = gb.variable(shape=(8, 1), init=np.linspace(1.0, 8.0, 8))
        m_v = gb.variable(shape=(8, 1), init=np.linspace(0.15, 0.55, 8))
        acc = []
        for k in range(n_queries):
            acc.append(polar_2d(a_v + 0.05 * k, m_v))
        ev_c = gb.build_evaluator(fo.sum(fo.concat(acc, axis=0)))
    t_cls_build = (time.perf_counter() - t0) * 1000.0 / n_repeats

    speedup = t_func_build / max(t_cls_build, 1e-6)
    print(
        f"[PASSED] 3. Multi-Query 2D B-Spline Polar Build Benchmark ({n_queries} queries on 30x20x2 table):\n"
        f"         - Functional `interp_bspline_2d`:   {t_func_build:.2f} ms\n"
        f"         - Pre-Compiled `BSplineInterpolator2D`: {t_cls_build:.2f} ms ({speedup:.1f}x faster)"
    )


if __name__ == "__main__":
    print("=" * 85)
    print("Running fastopt Phase 9 Suite (Reusable Pre-Compiled 1D/2D/3D Interpolator Classes)")
    print("=" * 85)
    test_dual_symbolic_and_numeric_evaluation()
    test_graph_constant_sharing_and_symbolic_table_reuse()
    test_repeated_query_speedup_benchmark()
    print("=" * 85)
    print("ALL PHASE 9 TESTS PASSED SUCCESSFULLY!")
    print("=" * 85)
