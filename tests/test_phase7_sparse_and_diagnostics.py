"""
Phase 7 Verification Suite:
  Direction 3: Symbolic Sparsity Detection & Greedy Graph Coloring for Sparse Jacobian & Sparse Hessian
               (including fo.vmap / fo.scan sparsity propagation and large-scale N=400 benchmark)
  Direction 4: Variable/Constraint/Objective Auto-Scaling, KKT Dual Multiplier Extraction (sol.dual,
               sol.bound_dual_lb/ub), Per-Iteration Callbacks & Convergence History, and
               Infeasibility Diagnostics (opti.debug_infeasibility)
All comments are written in English.
"""

import time
import numpy as np
import fastopt as fo


def fd_jacobian(ev: fo._fastopt_impl.Evaluator, x0: np.ndarray, out_idx: int, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference Jacobian for verification."""
    m = ev.output_size(out_idx)
    n = x0.size
    J = np.zeros((m, n), dtype=np.float64)
    for j in range(n):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        fp = ev.eval(xp)[out_idx].ravel(order="F")
        fm = ev.eval(xm)[out_idx].ravel(order="F")
        J[:, j] = (fp - fm) / (2.0 * eps)
    return J


def fd_lagrangian_hessian(
    ev: fo._fastopt_impl.Evaluator,
    x0: np.ndarray,
    obj_scale: float,
    mults: list,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central finite-difference Lagrangian Hessian computed from analytical gradients/Jacobians."""
    n = x0.size
    H = np.zeros((n, n), dtype=np.float64)

    def grad_L(x: np.ndarray) -> np.ndarray:
        g = obj_scale * ev.gradient(x, 0)
        for k, lam in enumerate(mults):
            if lam.size > 0:
                g = g + ev.jacobian(x, k + 1).T @ lam
        return g

    for j in range(n):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        H[:, j] = (grad_L(xp) - grad_L(xm)) / (2.0 * eps)
    return 0.5 * (H + H.T)


def test_symbolic_sparsity_and_graph_coloring_exactness():
    """
    Test 1: Banded Chained Rosenbrock + Neighbor-Coupled Constraints (N=60, M=59).
      - Verify Jacobian has exact bidiagonal sparsity (nnz = 2 * 59 = 118 out of 3540, 3.3% density)
        and requires only 2 Graph-Coloring passes instead of 59!
      - Verify Lagrangian Hessian has exact tridiagonal sparsity (nnz_tril = 60 + 59 = 119 out of 1830)
        and requires only 3 Graph-Coloring HVP passes instead of 60!
      - Verify values match central finite differences to < 1e-7.
    """
    n = 60
    gb = fo.GraphBuilder()
    rng = np.random.default_rng(42)
    x0 = rng.uniform(-1.2, 1.5, size=n)
    x = gb.variable(shape=(n, 1), init=x0, name="x")

    # Chained Rosenbrock objective: sum_{i=0}^{n-2} (10*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    x_curr = x[0 : n - 1]
    x_next = x[1:n]
    obj = fo.sum(10.0 * fo.square(x_next - fo.square(x_curr)) + fo.square(1.0 - x_curr))

    # Neighbor-coupled nonlinear constraints: g_i(x) = sin(x_i) * x_{i+1} + x_i^2 <= 1.0
    g_con = fo.sin(x_curr) * x_next + fo.square(x_curr) - 1.0

    ev = gb.build_evaluator([obj, g_con])

    j_rows, j_cols = ev.jacobian_sparsity(1)
    j_colors = ev.num_jacobian_colors(1)
    assert len(j_rows) == 2 * (n - 1), f"Expected {2 * (n - 1)} Jacobian nonzeros, got {len(j_rows)}"
    assert j_colors == 2, f"Bidiagonal Jacobian must color in 2 passes, got {j_colors}"

    h_rows, h_cols = ev.hessian_sparsity()
    h_colors = ev.num_hessian_colors()
    assert len(h_rows) == n + (n - 1), f"Expected {2 * n - 1} tril Hessian nonzeros, got {len(h_rows)}"
    assert h_colors == 3, f"Tridiagonal Hessian must color in 3 passes, got {h_colors}"

    # Compare sparse/colored Jacobian and Hessian against finite differences
    J_csc = fo.sparse_jacobian(ev, x0, out_idx=1)
    J_ad = ev.jacobian(x0, 1)
    J_fd = fd_jacobian(ev, x0, out_idx=1)
    assert np.max(np.abs(J_csc.toarray() - J_fd)) < 1e-7
    assert np.max(np.abs(J_ad - J_fd)) < 1e-7

    lam = rng.uniform(0.1, 1.0, size=n - 1)
    H_csc = fo.sparse_hessian(ev, x0, obj_scale=1.2, constraint_multipliers=[lam])
    H_ad = ev.lagrangian_hessian(x0, 1.2, [lam])
    H_fd = fd_lagrangian_hessian(ev, x0, 1.2, [lam])
    assert np.max(np.abs(H_csc.toarray() - H_fd)) < 1e-6
    assert np.max(np.abs(H_ad - H_fd)) < 1e-6

    print(
        f"[PASSED] 1. Symbolic Sparsity & Graph Coloring Exactness "
        f"(N={n}: Jac colors={j_colors}/{n-1}, Hess colors={h_colors}/{n})"
    )


def test_vmap_and_scan_sparsity_preservation():
    """
    Test 2: Verify that `fo.vmap` and `fo.scan` preserve exact block/banded sparsity
    instead of Conservative All-to-All coupling.
    """
    gb = fo.GraphBuilder()
    B = 30
    u0 = np.linspace(0.2, 1.5, B)
    v0 = np.linspace(-0.5, 0.8, B)
    u = gb.variable(shape=(B, 1), init=u0, name="u")
    v = gb.variable(shape=(B, 1), init=v0, name="v")

    # vmap over B independent sections: each output row b only depends on (u_b, v_b)
    mapped = fo.vmap(lambda ui, vi: fo.sin(ui * vi) + fo.square(ui) * fo.exp(vi), [u, v], in_axes=[0, 0])
    obj_vmap = fo.sum(mapped)
    ev_vmap = gb.build_evaluator([obj_vmap, mapped])

    # Jacobian of `mapped` (30 x 60) has exactly 2 nonzeros per row (u_b and v_b) -> 1 color!
    jr, jc = ev_vmap.jacobian_sparsity(1)
    assert len(jr) == 2 * B, f"Expected {2 * B} nonzeros for vmap Jacobian, got {len(jr)}"
    assert ev_vmap.num_jacobian_colors(1) == 1, f"Disjoint vmap rows should need 1 color, got {ev_vmap.num_jacobian_colors(1)}"
    # Hessian of `obj_vmap` (60 x 60) consists of B disjoint 2x2 blocks -> only 2 colors!
    hr, hc = ev_vmap.hessian_sparsity()
    assert len(hr) == 3 * B, f"Expected {3 * B} tril Hessian nonzeros for 2x2 blocks, got {len(hr)}"
    assert ev_vmap.num_hessian_colors() == 2, f"2x2 block-diagonal Hessian should need 2 colors, got {ev_vmap.num_hessian_colors()}"

    uv0 = np.concatenate([u0, v0])
    J_fd = fd_jacobian(ev_vmap, uv0, 1)
    assert np.max(np.abs(ev_vmap.jacobian(uv0, 1) - J_fd)) < 1e-7
    H_fd = fd_lagrangian_hessian(ev_vmap, uv0, 1.0, [np.ones(B) * 0.5])
    assert np.max(np.abs(ev_vmap.lagrangian_hessian(uv0, 1.0, [np.ones(B) * 0.5]) - H_fd)) < 1e-6

    # Also test scan sparsity: state c_{t+1} = 0.9 * c_t + u_t^2 (nonlinear in u_t, linear in c_t!)
    gb_s = fo.GraphBuilder()
    T = 25
    ctrl0 = np.linspace(0.1, 0.9, T)
    ctrl = gb_s.variable(shape=(T, 1), init=ctrl0, name="ctrl")
    c_init = gb_s.constant([0.0], shape=(1, 1))
    c_final, c_traj = fo.scan(lambda c, ut: (0.9 * c + fo.square(ut), c + ut), c_init, xs=ctrl)
    ev_scan = gb_s.build_evaluator([c_final, c_traj])

    # Because c_{t+1} = 0.9*c_t + u_t^2 has zero cross-terms u_t * u_k for t != k,
    # the Hessian of c_final w.r.t. ctrl is strictly DIAGONAL (only 1 color for T=25!)
    hs_r, hs_c = ev_scan.hessian_sparsity()
    assert len(hs_r) == T and ev_scan.num_hessian_colors() == 1
    # And the trajectory Jacobian is lower-triangular!
    js_r, js_c = ev_scan.jacobian_sparsity(1)
    assert len(js_r) == T * (T + 1) // 2
    J_scan_fd = fd_jacobian(ev_scan, ctrl0, 1)
    assert np.max(np.abs(ev_scan.jacobian(ctrl0, 1) - J_scan_fd)) < 1e-7

    print(
        f"[PASSED] 2. Sparsity Preservation Through fo.vmap (B={B}: Jac=1 color, Hess=2 colors) "
        f"& fo.scan (T={T}: Hess=1 color)"
    )


def test_large_scale_graph_coloring_benchmark():
    """
    Test 3: Large-Scale Banded / Multi-Station Benchmark (N=400 variables, M=399 constraints).
    Compare Graph-Colored sparse evaluation vs. uncompressed N-pass evaluation.
    """
    n = 400
    gb = fo.GraphBuilder()
    x0 = np.linspace(0.2, 1.2, n)
    x = gb.variable(shape=(n, 1), init=x0, name="x")

    x_i = x[0 : n - 1]
    x_ip1 = x[1:n]
    obj = fo.sum(fo.sin(x_i * x_ip1) + 0.5 * fo.square(x_ip1 - x_i) + 0.1 * fo.exp(x_i))
    g_con = fo.square(x_i) + fo.cos(x_ip1) - 1.2

    ev = gb.build_evaluator([obj, g_con])
    j_colors = ev.num_jacobian_colors(1)
    h_colors = ev.num_hessian_colors()
    j_nnz = len(ev.jacobian_sparsity(1)[0])
    h_nnz = len(ev.hessian_sparsity()[0])
    lam = np.ones(n - 1, dtype=np.float64) * 0.3

    # Warm up
    _ = ev.sparse_jacobian_values(x0, 1)
    _ = ev.sparse_hessian_values(x0, 1.0, [lam], num_threads=1)
    _ = ev.hessian_vec_prod(x0, 1.0, [lam], np.ones(n))

    n_rep = 20
    # 1. Measure Graph-Colored Sparse Jacobian (2 passes)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        s_jac = ev.sparse_jacobian_values(x0, 1)
    t_colored_jac = (time.perf_counter() - t0) * 1000.0 / n_rep

    # 2. Measure Graph-Colored Sparse Hessian (3 passes)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        s_hess = ev.sparse_hessian_values(x0, 1.0, [lam], num_threads=1)
    t_colored_hess = (time.perf_counter() - t0) * 1000.0 / n_rep

    # 3. Measure Uncompressed N-Pass HVP Assembly (400 passes)
    t0 = time.perf_counter()
    e_vec = np.zeros(n, dtype=np.float64)
    for j in range(n):
        e_vec[j] = 1.0
        _ = ev.hessian_vec_prod(x0, 1.0, [lam], e_vec)
        e_vec[j] = 0.0
    t_uncompressed_hess = (time.perf_counter() - t0) * 1000.0

    hess_speedup = t_uncompressed_hess / max(t_colored_hess, 1e-6)
    assert s_jac.size == j_nnz and s_hess.size == h_nnz
    print(
        f"[PASSED] 3. Large-Scale Graph-Coloring Benchmark (N={n}, M={n-1}):\n"
        f"         - Jacobian: nnz={j_nnz}/{n*(n-1)} ({100.0*j_nnz/(n*(n-1)):.2f}%), "
        f"Colors={j_colors} vs {n-1} rows, Eval={t_colored_jac:.3f} ms\n"
        f"         - Hessian:  nnz_tril={h_nnz}/{n*(n+1)//2} ({100.0*h_nnz/(n*(n+1)//2):.2f}%), "
        f"Colors={h_colors} vs {n} cols, Colored={t_colored_hess:.3f} ms vs Uncompressed={t_uncompressed_hess:.3f} ms "
        f"({hess_speedup:.1f}x speedup)"
    )


def test_auto_scaling_and_kkt_duals_and_diagnostics():
    """
    Test 4: Direction 4 Engineering Features:
      - Multi-scale optimization with `scale=...` and `auto_scale=True` (mixing 1e-3 m thickness
        and 1e4 N thrust constraint)
      - KKT Dual Multiplier accuracy (`sol.dual(con)`, `sol.bound_dual_lb(var)`, `sol.bound_dual_ub(var)`)
        verified against analytical perturbation sensitivity d(f*)/d(rhs)
      - Iteration callback & `sol.history`
      - `opti.debug_infeasibility()` on an infeasible initial point
    """
    def solve_multiscale_problem(thrust_target: float, use_trust_constr: bool = False):
        opti = fo.Opti()
        # Variable 1: chord thickness t in [0.008, 0.030] m (order 1e-2)
        t_m = opti.variable(init=0.012, lb=0.008, ub=0.020, scale=0.01, name="thickness_m")
        # Variable 2: collective pitch angle theta in [2.0, 18.0] deg (order 10)
        theta_deg = opti.variable(init=6.0, lb=2.0, ub=18.0, scale=10.0, name="theta_deg")
        # Variable 3: rotor RPM omega in [800.0, 2200.0] (order 1e3)
        rpm = opti.variable(init=1100.0, lb=800.0, ub=2200.0, scale=1000.0, name="rpm")

        # Surrogate power (W, order 1e5) and thrust (N, order 1e4)
        thrust_N = 1.8e-3 * (rpm**2) * (0.12 * theta_deg + 5.0 * t_m)
        power_W = 4.5e-5 * (rpm**3) * (0.015 + 0.002 * (theta_deg**2) + 80.0 * ((t_m - 0.014) ** 2))
        bending_stress_MPa = 0.035 * thrust_N / (t_m * 100.0)

        opti.minimize(power_W)
        c_thrust = opti.subject_to(thrust_N >= thrust_target, scale=10000.0, name="min_thrust")
        c_stress = opti.subject_to(bending_stress_MPa <= 280.0, scale=100.0, name="max_stress")

        cb_log = []
        method = "trust-constr" if use_trust_constr else "SLSQP"
        opts = {"gtol": 1e-10, "xtol": 1e-11, "maxiter": 300} if use_trust_constr else {"ftol": 1e-11, "maxiter": 200}
        opti.solver(method, options=opts, auto_scale=True, callback=lambda info: cb_log.append(info))
        sol = opti.solve()
        return opti, sol, t_m, theta_deg, rpm, c_thrust, c_stress, cb_log

    target_T = 15000.0
    opti, sol, t_m, theta_deg, rpm, c_thrust, c_stress, cb_log = solve_multiscale_problem(
        target_T, use_trust_constr=False
    )
    assert sol.success, f"Auto-scaled SLSQP failed: {sol.message}"
    assert len(sol.history) > 0 and len(cb_log) == len(sol.history)

    # Check feasibility at optimal solution via sol.debug_infeasibility()
    diag_opt = sol.debug_infeasibility(tol=1e-4)
    assert diag_opt["is_feasible"], f"Optimal solution reported infeasible:\n{diag_opt['report']}"

    # Verify KKT dual multiplier of `c_thrust` against finite-difference perturbation of thrust_target
    delta_T = 1.0  # +1 N perturbation on 15,000 N constraint
    _, sol_pert, *_ = solve_multiscale_problem(target_T + delta_T, use_trust_constr=False)
    df_dT_fd = (sol_pert.f - sol.f) / delta_T
    lam_thrust_slsqp = sol.dual(c_thrust)
    assert np.isclose(lam_thrust_slsqp, df_dT_fd, rtol=1e-2), (
        f"SLSQP KKT dual mismatch: dual={lam_thrust_slsqp:.6f} vs FD={df_dT_fd:.6f}"
    )

    # Also verify with trust-constr
    _, sol_tc, t_tc, _, _, c_thrust_tc, _, _ = solve_multiscale_problem(target_T, use_trust_constr=True)
    assert sol_tc.success
    lam_thrust_tc = sol_tc.dual(c_thrust_tc)
    assert np.isclose(lam_thrust_tc, df_dT_fd, rtol=1e-2), (
        f"trust-constr KKT dual mismatch: dual={lam_thrust_tc:.6f} vs FD={df_dT_fd:.6f}"
    )
    # At the optimal solution, thickness t_m hits its upper bound 0.020 m to reduce bending stress -> ub dual > 0!
    assert np.isclose(sol.value(t_m), 0.020, atol=1e-5)
    assert sol.bound_dual_ub(t_m) > 0.0 and sol.bound_dual_lb(t_m) == 0.0

    # Test debug_infeasibility on a deliberately bad point (violating bounds and constraints)
    bad_x = np.array([0.005, 1.0, 600.0])  # below lb for all 3 variables, and thrust << 15000 N
    diag_bad = opti.debug_infeasibility(x=bad_x, tol=1e-5)
    assert not diag_bad["is_feasible"]
    assert len(diag_bad["violated_bounds"]) == 3
    assert len(diag_bad["violated_constraints"]) >= 1

    print(
        f"[PASSED] 4. Auto-Scaling, KKT Dual Multipliers (dual={lam_thrust_slsqp:.4f} W/N vs FD={df_dT_fd:.4f} W/N), "
        f"Callbacks ({len(sol.history)} iters) & debug_infeasibility()"
    )


if __name__ == "__main__":
    print("=" * 85)
    print("Running fastopt Phase 7 Suite (Graph-Coloring Sparsity, Auto-Scaling & Diagnostics)")
    print("=" * 85)
    test_symbolic_sparsity_and_graph_coloring_exactness()
    test_vmap_and_scan_sparsity_preservation()
    test_large_scale_graph_coloring_benchmark()
    test_auto_scaling_and_kkt_duals_and_diagnostics()
    print("=" * 85)
    print("ALL PHASE 7 TESTS PASSED SUCCESSFULLY!")
    print("=" * 85)
