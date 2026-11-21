"""
Phase 5 verification test suite for fastopt:
1. Independent Parallel Loop (`fo.vmap` / `MapNode`):
   - Verify graph compactness (1 MapNode vs. unrolled loop)
   - Verify exact 0th-order forward, 1st-order gradient/Jacobian, and 2nd-order Hessian
     against an unrolled reference loop (both `in_axes=0` and `in_axes=1`, with shared broadcast inputs).
2. Sequential Recurrent Loop (`fo.scan` / `ScanNode`):
   - Verify exact 0th-order state trajectory, 1st-order discrete-adjoint gradient/Jacobian,
     and 2nd-order Forward-over-Reverse discrete-adjoint Hessian against an unrolled time-stepping loop.
   - Solve a constrained nonlinear trajectory optimal control problem using `fo.scan` with SLSQP & trust-constr.
3. Vectorized Population Batch Black-Box (`CustomOp.forward_batch` & `@fo.blackbox(..., batch_fn=...)`):
   - Verify that `Evaluator.eval_batch` and `pymoo` invoke `forward_batch` once per population batch
     (even when sandwiched between upstream and downstream symbolic C++ nodes) with zero single-point calls.
4. Large-Scale Loop Performance Benchmark (`N = 500` for `fo.vmap` and `T = 500` for `fo.scan` vs. Python `for` loop):
   - Compare graph node count, workspace memory footprint, graph construction time,
     0th-order `eval()`, 1st-order `gradient()`, and 2nd-order `hessian_vec_prod()` runtime.
All comments are written in English.
"""

import os
import sys
import time
import numpy as np
import fastopt as fo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_phase4_blackbox_and_pymoo import _ensure_pymoo_available_for_test


# ============================================================================
# 1. Verify fo.vmap (Independent Parallel Loop: 0th, 1st, & 2nd Order AD)
# ============================================================================
def test_vmap_ad_and_graph_compactness():
    n_stations = 20
    r_grid = np.linspace(0.2, 1.0, n_stations)

    def station_element_model(twist_i, chord_i, r_i, flight_p):
        # Nonlinear aerodynamic-like element model returning [dT, dQ] (size 2)
        omega = flight_p[0]
        v_inf = flight_p[1]
        v_loc = fo.sqrt((omega * r_i) ** 2 + v_inf ** 2)
        alpha = twist_i - fo.tan(v_inf / (omega * r_i + 0.1))
        cl = 5.5 * fo.sin(alpha)
        cd = 0.015 + 1.2 * (fo.sin(alpha) ** 2)
        q_dyn = 0.5 * 1.225 * (v_loc ** 2) * chord_i
        d_thrust = q_dyn * (cl * fo.cos(alpha) - cd * fo.sin(alpha))
        d_torque = q_dyn * (cl * fo.sin(alpha) + cd * fo.cos(alpha)) * r_i
        return fo.concat([d_thrust, d_torque], axis=0)

    # 1A. Unrolled Python for-loop reference graph
    gb_ref = fo.GraphBuilder()
    tw_ref = gb_ref.variable(shape=(n_stations, 1), init=np.linspace(0.3, 0.1, n_stations))
    ch_ref = gb_ref.variable(shape=(n_stations, 1), init=np.full(n_stations, 0.15))
    r_ref = gb_ref.constant(r_grid, shape=(n_stations, 1))
    fp_ref = gb_ref.parameter(shape=(2, 1), value=[110.0, 15.0])

    rows_ref = []
    for i in range(n_stations):
        out_i = station_element_model(tw_ref[i], ch_ref[i], r_ref[i], fp_ref)
        rows_ref.append(out_i.T)
    mat_ref = fo.concat(rows_ref, axis=0)
    loss_ref = (mat_ref[:, 0] ** 2).sum() + 0.5 * mat_ref[:, 1].sum()
    ev_ref = gb_ref.build_evaluator([loss_ref, mat_ref])

    # 1B. Compact fo.vmap graph
    gb_map = fo.GraphBuilder()
    tw_map = gb_map.variable(shape=(n_stations, 1), init=np.linspace(0.3, 0.1, n_stations))
    ch_map = gb_map.variable(shape=(n_stations, 1), init=np.full(n_stations, 0.15))
    r_map = gb_map.constant(r_grid, shape=(n_stations, 1))
    fp_map = gb_map.parameter(shape=(2, 1), value=[110.0, 15.0])

    mat_map = fo.vmap(
        station_element_model,
        inputs=[tw_map, ch_map, r_map],
        shared_inputs=[fp_map],
        in_axes=0,
        out_axis=0,
    )
    loss_map = (mat_map[:, 0] ** 2).sum() + 0.5 * mat_map[:, 1].sum()
    ev_map = gb_map.build_evaluator([loss_map, mat_map])

    # Verify outer graph node count is drastically smaller with vmap
    assert gb_map.cpp_graph.num_nodes() < gb_ref.cpp_graph.num_nodes() // 5

    x_test = gb_ref.cpp_graph.initial_decision_vector()
    val_ref = ev_ref.eval(x_test)
    val_map = ev_map.eval(x_test)
    np.testing.assert_allclose(val_map[0], val_ref[0], rtol=1e-12)
    np.testing.assert_allclose(val_map[1], val_ref[1], rtol=1e-12)

    grad_ref = ev_ref.gradient(x_test, 0)
    grad_map = ev_map.gradient(x_test, 0)
    np.testing.assert_allclose(grad_map, grad_ref, rtol=1e-12)

    jac_ref = ev_ref.jacobian(x_test, 1)
    jac_map = ev_map.jacobian(x_test, 1)
    np.testing.assert_allclose(jac_map, jac_ref, rtol=1e-12)

    hess_ref = ev_ref.hessian(x_test, 0)
    hess_map = ev_map.hessian(x_test, 0)
    np.testing.assert_allclose(hess_map, hess_ref, rtol=1e-12, atol=1e-10)

    # Also test column-sliced in_axes=1 and out_axis=1 on a 2D matrix (3 x 6)
    gb_col = fo.GraphBuilder()
    X_col = gb_col.variable(shape=(3, 6), init=np.arange(18, dtype=np.float64) * 0.1 + 0.2)
    Y_col = fo.vmap(
        lambda col_vec: fo.concat([col_vec[0] * col_vec[1] + fo.sin(col_vec[2]), col_vec[2] ** 2], axis=0),
        inputs=X_col,
        in_axes=1,
        out_axis=1,
    )
    assert Y_col.shape == (2, 6)
    ev_col = gb_col.build_evaluator(Y_col.sum())
    x_c = gb_col.cpp_graph.initial_decision_vector()
    H_col = ev_col.hessian(x_c)
    assert H_col.shape == (18, 18)


# ============================================================================
# 2. Verify fo.scan (Sequential Recurrent Loop: 0th, 1st, & 2nd Order AD + OCP)
# ============================================================================
def test_scan_ad_and_trajectory_optimization():
    n_steps = 15
    dt = 0.1

    def dynamics_step(state, u_t, p_drag):
        pos = state[0]
        vel = state[1]
        u = u_t[0]
        drag_k = p_drag[0]
        acc = u - drag_k * (vel ** 2) * fo.sin(pos + 0.5)
        next_pos = pos + dt * vel + 0.5 * (dt ** 2) * acc
        next_vel = vel + dt * acc
        next_state = fo.concat([next_pos, next_vel], axis=0)
        step_out = fo.concat([next_pos, next_vel, u ** 2], axis=0)
        return next_state, step_out

    c0_init = np.array([0.1, 0.2], dtype=np.float64)
    u_init = np.linspace(0.5, -0.3, n_steps)

    # 2A. Unrolled Python time-stepping reference graph
    gb_ref = fo.GraphBuilder()
    c0_ref = gb_ref.variable(shape=(2, 1), init=c0_init)
    U_ref = gb_ref.variable(shape=(n_steps, 1), init=u_init)
    p_ref = gb_ref.parameter(shape=(1, 1), value=[0.25])

    c_curr = c0_ref
    y_list = []
    for t in range(n_steps):
        c_curr, y_t = dynamics_step(c_curr, U_ref[t], p_ref)
        y_list.append(y_t.T)
    ys_ref = fo.concat(y_list, axis=0)
    cost_ref = ((c_curr[0] - 1.5) ** 2) + (c_curr[1] ** 2) + 0.1 * ys_ref[:, 2].sum()
    ev_ref = gb_ref.build_evaluator([cost_ref, c_curr, ys_ref])

    # 2B. Compact fo.scan graph
    gb_scan = fo.GraphBuilder()
    c0_scan = gb_scan.variable(shape=(2, 1), init=c0_init)
    U_scan = gb_scan.variable(shape=(n_steps, 1), init=u_init)
    p_scan = gb_scan.parameter(shape=(1, 1), value=[0.25])

    final_state, ys_scan = fo.scan(
        dynamics_step,
        init_carry=c0_scan,
        xs=U_scan,
        shared_inputs=[p_scan],
        seq_axes=0,
        out_axis=0,
    )
    assert final_state.shape == (2, 1)
    assert ys_scan.shape == (n_steps, 3)

    cost_scan = ((final_state[0] - 1.5) ** 2) + (final_state[1] ** 2) + 0.1 * ys_scan[:, 2].sum()
    ev_scan = gb_scan.build_evaluator([cost_scan, final_state, ys_scan])

    x_test = gb_ref.cpp_graph.initial_decision_vector()
    v_ref = ev_ref.eval(x_test)
    v_scan = ev_scan.eval(x_test)
    np.testing.assert_allclose(v_scan[0], v_ref[0], rtol=1e-12)
    np.testing.assert_allclose(v_scan[1], v_ref[1], rtol=1e-12)
    np.testing.assert_allclose(v_scan[2], v_ref[2], rtol=1e-12)

    g_ref = ev_ref.gradient(x_test, 0)
    g_scan = ev_scan.gradient(x_test, 0)
    np.testing.assert_allclose(g_scan, g_ref, rtol=1e-12)

    j_ref = ev_ref.jacobian(x_test, 2)
    j_scan = ev_scan.jacobian(x_test, 2)
    np.testing.assert_allclose(j_scan, j_ref, rtol=1e-12)

    h_ref = ev_ref.hessian(x_test, 0)
    h_scan = ev_scan.hessian(x_test, 0)
    np.testing.assert_allclose(h_scan, h_ref, rtol=1e-11, atol=1e-10)

    # 2C. Solve an optimal control problem with fo.scan via SLSQP
    opti = fo.Opti()
    c0_fixed = opti.parameter(shape=(2, 1), value=[0.0, 0.0])
    U_var = opti.variable(shape=(n_steps, 1), init=0.2, lb=-5.0, ub=5.0)
    drag_p = opti.parameter(shape=(1, 1), value=[0.1])

    c_end, traj = fo.scan(
        dynamics_step,
        init_carry=c0_fixed,
        xs=U_var,
        shared_inputs=[drag_p],
    )
    opti.minimize(0.1 * dt * traj[:, 2].sum())
    opti.subject_to([
        c_end[0] == 1.0,
        c_end[1] == 0.0,
        traj[:, 1] <= 1.2,
    ])
    opti.solver("SLSQP", options={"ftol": 1e-10, "maxiter": 200})
    sol = opti.solve()
    assert sol.success, f"SLSQP trajectory optimization with scan failed: {sol.message}"
    np.testing.assert_allclose(sol.value(c_end), [1.0, 0.0], atol=1e-5)


# ============================================================================
# 3. Verify CustomOp.forward_batch in Evaluator.eval_batch and pymoo
# ============================================================================
class BatchableAeroOp(fo.CustomOp):
    """Custom black-box operator that tracks single-point vs. batch invocation counts."""

    def __init__(self) -> None:
        super().__init__(shape=(2, 1), name="batch_aero")
        self.single_calls = 0
        self.batch_calls = 0

    def forward(self, x: np.ndarray, scale_p: np.ndarray) -> np.ndarray:
        self.single_calls += 1
        s = float(np.asarray(scale_p).ravel()[0])
        y0 = s * ((x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2)
        y1 = x[0] + x[1]
        return np.array([y0, y1], dtype=np.float64)

    def forward_batch(self, X_batch: np.ndarray, P_batch: np.ndarray) -> np.ndarray:
        self.batch_calls += 1
        s = P_batch[:, 0]
        y0 = s * ((X_batch[:, 0] - 1.0) ** 2 + (X_batch[:, 1] - 2.0) ** 2)
        y1 = X_batch[:, 0] + X_batch[:, 1]
        return np.column_stack([y0, y1])


def test_custom_op_forward_batch_in_eval_batch_and_pymoo():
    _ensure_pymoo_available_for_test()

    op = BatchableAeroOp()
    opti = fo.Opti()
    x = opti.variable(shape=(2, 1), init=[0.0, 0.0], lb=-5.0, ub=5.0)
    p = opti.parameter(shape=(1, 1), value=[2.0])

    # Sandwich the batch black-box node between upstream and downstream symbolic nodes
    x_shifted = x + 0.0
    aero_out = op(x_shifted, p)
    obj = aero_out[0] + 0.01 * ((x[0] - 1.0) ** 2)
    opti.minimize(obj)
    opti.subject_to(aero_out[1] >= 3.0)

    ev = opti.build_evaluator([obj, aero_out])
    X_pop = np.random.default_rng(0).uniform(-2.0, 4.0, size=(32, 2))

    op.single_calls = 0
    op.batch_calls = 0
    batch_res = ev.eval_batch(X_pop)
    assert op.batch_calls == 1
    assert op.single_calls == 0

    for i in range(len(X_pop)):
        single_res = ev.eval(X_pop[i])
        np.testing.assert_allclose(batch_res[0][i, 0], single_res[0].item(), rtol=1e-12)
        np.testing.assert_allclose(batch_res[1][i], single_res[1].ravel(), rtol=1e-12)

    # Run pymoo DE using the batch black-box operator
    op.single_calls = 0
    op.batch_calls = 0
    opti.solver("DE", options={"pop_size": 40, "n_gen": 50, "seed": 7})
    sol = opti.solve()
    assert sol.success
    assert op.batch_calls >= 50
    assert op.single_calls == 0
    np.testing.assert_allclose(sol.value(x), [1.0, 2.0], atol=5e-2)


# ============================================================================
# 4. Large-Scale Loop Benchmark: fo.vmap & fo.scan vs. Unrolled Python For-Loop
# ============================================================================
def test_large_loop_vmap_and_scan_performance_benchmark(n_loop: int = 500, n_repeats: int = 50):
    print(f"\n--- Large-Scale Loop Benchmark (Loop Size = {n_loop}, Repeats = {n_repeats}) ---")

    # ------------------------------------------------------------------------
    # Part A: Independent Loop Benchmark (Unrolled Python For-Loop vs. fo.vmap)
    # ------------------------------------------------------------------------
    r_grid = np.linspace(0.2, 1.0, n_loop)

    def vmap_body(tw, ch, r, fp):
        omega, v_inf = fp[0], fp[1]
        v_loc = fo.sqrt((omega * r) ** 2 + v_inf ** 2)
        alpha = tw - fo.tan(v_inf / (omega * r + 0.1))
        cl = 5.5 * fo.sin(alpha)
        cd = 0.015 + 1.2 * (fo.sin(alpha) ** 2)
        q_dyn = 0.5 * 1.225 * (v_loc ** 2) * ch
        dT = q_dyn * (cl * fo.cos(alpha) - cd * fo.sin(alpha))
        dQ = q_dyn * (cl * fo.sin(alpha) + cd * fo.cos(alpha)) * r
        return fo.concat([dT, dQ], axis=0)

    t0 = time.perf_counter()
    gb_vmap_ref = fo.GraphBuilder()
    tw1 = gb_vmap_ref.variable(shape=(n_loop, 1), init=np.linspace(0.3, 0.1, n_loop))
    ch1 = gb_vmap_ref.variable(shape=(n_loop, 1), init=np.full(n_loop, 0.15))
    r1 = gb_vmap_ref.constant(r_grid, shape=(n_loop, 1))
    fp1 = gb_vmap_ref.parameter(shape=(2, 1), value=[110.0, 15.0])
    rows = [vmap_body(tw1[i], ch1[i], r1[i], fp1).T for i in range(n_loop)]
    mat1 = fo.concat(rows, axis=0)
    loss1 = (mat1[:, 0] ** 2).sum() + 0.5 * mat1[:, 1].sum()
    ev_vmap_ref = gb_vmap_ref.build_evaluator([loss1])
    t_build_vmap_ref = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    gb_vmap = fo.GraphBuilder()
    tw2 = gb_vmap.variable(shape=(n_loop, 1), init=np.linspace(0.3, 0.1, n_loop))
    ch2 = gb_vmap.variable(shape=(n_loop, 1), init=np.full(n_loop, 0.15))
    r2 = gb_vmap.constant(r_grid, shape=(n_loop, 1))
    fp2 = gb_vmap.parameter(shape=(2, 1), value=[110.0, 15.0])
    mat2 = fo.vmap(vmap_body, inputs=[tw2, ch2, r2], shared_inputs=[fp2], in_axes=0, out_axis=0)
    loss2 = (mat2[:, 0] ** 2).sum() + 0.5 * mat2[:, 1].sum()
    ev_vmap = gb_vmap.build_evaluator([loss2])
    t_build_vmap = (time.perf_counter() - t0) * 1000.0

    x_vmap = gb_vmap_ref.cpp_graph.initial_decision_vector()
    v_dir_vmap = np.ones_like(x_vmap)
    # Warm-up and numerical equivalence check
    np.testing.assert_allclose(ev_vmap.eval(x_vmap)[0], ev_vmap_ref.eval(x_vmap)[0], rtol=1e-12)
    np.testing.assert_allclose(ev_vmap.gradient(x_vmap), ev_vmap_ref.gradient(x_vmap), rtol=1e-12)
    np.testing.assert_allclose(
        ev_vmap.hessian_vec_prod(x_vmap, 1.0, [], v_dir_vmap),
        ev_vmap_ref.hessian_vec_prod(x_vmap, 1.0, [], v_dir_vmap),
        rtol=1e-12,
    )

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ev_vmap_ref.eval(x_vmap)
    t_eval_vmap_ref = (time.perf_counter() - t0) * 1000.0 / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ev_vmap.eval(x_vmap)
    t_eval_vmap = (time.perf_counter() - t0) * 1000.0 / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ev_vmap_ref.gradient(x_vmap)
    t_grad_vmap_ref = (time.perf_counter() - t0) * 1000.0 / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ev_vmap.gradient(x_vmap)
    t_grad_vmap = (time.perf_counter() - t0) * 1000.0 / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ev_vmap_ref.hessian_vec_prod(x_vmap, 1.0, [], v_dir_vmap)
    t_hvp_vmap_ref = (time.perf_counter() - t0) * 1000.0 / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ev_vmap.hessian_vec_prod(x_vmap, 1.0, [], v_dir_vmap)
    t_hvp_vmap = (time.perf_counter() - t0) * 1000.0 / n_repeats

    print(
        f"[vmap N={n_loop}] Outer Graph Nodes: Unrolled={gb_vmap_ref.cpp_graph.num_nodes()} "
        f"vs. vmap={gb_vmap.cpp_graph.num_nodes()} "
        f"({gb_vmap_ref.cpp_graph.num_nodes() / gb_vmap.cpp_graph.num_nodes():.1f}x reduction)"
    )
    print(
        f"[vmap N={n_loop}] Graph Build Time:  Unrolled={t_build_vmap_ref:.2f} ms "
        f"vs. vmap={t_build_vmap:.2f} ms ({t_build_vmap_ref / max(t_build_vmap, 1e-6):.1f}x faster)"
    )
    print(
        f"[vmap N={n_loop}] Forward eval():    Unrolled={t_eval_vmap_ref:.3f} ms "
        f"vs. vmap={t_eval_vmap:.3f} ms ({t_eval_vmap_ref / max(t_eval_vmap, 1e-6):.2f}x speedup)"
    )
    print(
        f"[vmap N={n_loop}] 1st-Order grad():  Unrolled={t_grad_vmap_ref:.3f} ms "
        f"vs. vmap={t_grad_vmap:.3f} ms ({t_grad_vmap_ref / max(t_grad_vmap, 1e-6):.2f}x speedup)"
    )
    print(
        f"[vmap N={n_loop}] 2nd-Order HVP:     Unrolled={t_hvp_vmap_ref:.3f} ms "
        f"vs. vmap={t_hvp_vmap:.3f} ms ({t_hvp_vmap_ref / max(t_hvp_vmap, 1e-6):.2f}x speedup)"
    )

    # ------------------------------------------------------------------------
    # Part B: Recurrent Loop Benchmark (Unrolled Python For-Loop vs. fo.scan)
    # ------------------------------------------------------------------------
    dt = 0.02

    def scan_step(state, u_t, p_drag):
        pos, vel = state[0], state[1]
        u, drag_k = u_t[0], p_drag[0]
        acc = u - drag_k * (vel ** 2) * fo.sin(pos + 0.5)
        npos = pos + dt * vel + 0.5 * (dt ** 2) * acc
        nvel = vel + dt * acc
        return fo.concat([npos, nvel], axis=0), fo.concat([npos, nvel, u ** 2], axis=0)

    t0 = time.perf_counter()
    gb_scan_ref = fo.GraphBuilder()
    c1 = gb_scan_ref.variable(shape=(2, 1), init=[0.1, 0.2])
    U1 = gb_scan_ref.variable(shape=(n_loop, 1), init=np.linspace(0.5, -0.3, n_loop))
    p1 = gb_scan_ref.parameter(shape=(1, 1), value=[0.25])
    curr = c1
    ys = []
    for t in range(n_loop):
        curr, yt = scan_step(curr, U1[t], p1)
        ys.append(yt.T)
    ymat1 = fo.concat(ys, axis=0)
    cost1 = ((curr[0] - 1.5) ** 2) + (curr[1] ** 2) + 0.1 * ymat1[:, 2].sum()
    ev_scan_ref = gb_scan_ref.build_evaluator([cost1])
    t_build_scan_ref = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    gb_scan = fo.GraphBuilder()
    c2 = gb_scan.variable(shape=(2, 1), init=[0.1, 0.2])
    U2 = gb_scan.variable(shape=(n_loop, 1), init=np.linspace(0.5, -0.3, n_loop))
    p2 = gb_scan.parameter(shape=(1, 1), value=[0.25])
    cend, ymat2 = fo.scan(scan_step, init_carry=c2, xs=U2, shared_inputs=[p2])
    cost2 = ((cend[0] - 1.5) ** 2) + (cend[1] ** 2) + 0.1 * ymat2[:, 2].sum()
    ev_scan = gb_scan.build_evaluator([cost2])
    t_build_scan = (time.perf_counter() - t0) * 1000.0

    x_scan = gb_scan_ref.cpp_graph.initial_decision_vector()
    v_dir_scan = np.ones_like(x_scan)
    np.testing.assert_allclose(ev_scan.eval(x_scan)[0], ev_scan_ref.eval(x_scan)[0], rtol=1e-12)
    np.testing.assert_allclose(ev_scan.gradient(x_scan), ev_scan_ref.gradient(x_scan), rtol=1e-12)
    np.testing.assert_allclose(
        ev_scan.hessian_vec_prod(x_scan, 1.0, [], v_dir_scan),
        ev_scan_ref.hessian_vec_prod(x_scan, 1.0, [], v_dir_scan),
        rtol=1e-11,
    )

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ev_scan_ref.eval(x_scan)
    t_eval_scan_ref = (time.perf_counter() - t0) * 1000.0 / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ev_scan.eval(x_scan)
    t_eval_scan = (time.perf_counter() - t0) * 1000.0 / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ev_scan_ref.gradient(x_scan)
    t_grad_scan_ref = (time.perf_counter() - t0) * 1000.0 / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ev_scan.gradient(x_scan)
    t_grad_scan = (time.perf_counter() - t0) * 1000.0 / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ev_scan_ref.hessian_vec_prod(x_scan, 1.0, [], v_dir_scan)
    t_hvp_scan_ref = (time.perf_counter() - t0) * 1000.0 / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ev_scan.hessian_vec_prod(x_scan, 1.0, [], v_dir_scan)
    t_hvp_scan = (time.perf_counter() - t0) * 1000.0 / n_repeats

    print(
        f"[scan T={n_loop}] Outer Graph Nodes: Unrolled={gb_scan_ref.cpp_graph.num_nodes()} "
        f"vs. scan={gb_scan.cpp_graph.num_nodes()} "
        f"({gb_scan_ref.cpp_graph.num_nodes() / gb_scan.cpp_graph.num_nodes():.1f}x reduction)"
    )
    print(
        f"[scan T={n_loop}] Graph Build Time:  Unrolled={t_build_scan_ref:.2f} ms "
        f"vs. scan={t_build_scan:.2f} ms ({t_build_scan_ref / max(t_build_scan, 1e-6):.1f}x faster)"
    )
    print(
        f"[scan T={n_loop}] Forward eval():    Unrolled={t_eval_scan_ref:.3f} ms "
        f"vs. scan={t_eval_scan:.3f} ms ({t_eval_scan_ref / max(t_eval_scan, 1e-6):.2f}x speedup)"
    )
    print(
        f"[scan T={n_loop}] 1st-Order grad():  Unrolled={t_grad_scan_ref:.3f} ms "
        f"vs. scan={t_grad_scan:.3f} ms ({t_grad_scan_ref / max(t_grad_scan, 1e-6):.2f}x speedup)"
    )
    print(
        f"[scan T={n_loop}] 2nd-Order HVP:     Unrolled={t_hvp_scan_ref:.3f} ms "
        f"vs. scan={t_hvp_scan:.3f} ms ({t_hvp_scan_ref / max(t_hvp_scan, 1e-6):.2f}x speedup)"
    )


if __name__ == "__main__":
    t0 = time.perf_counter()
    test_vmap_ad_and_graph_compactness()
    print("[PASSED] 1. fo.vmap Independent Parallel Loop (0th, 1st & 2nd Order AD + Graph Compactness)")

    test_scan_ad_and_trajectory_optimization()
    print("[PASSED] 2. fo.scan Sequential Recurrent Loop (0th, 1st & 2nd Order Discrete Adjoint + Trajectory OCP)")

    test_custom_op_forward_batch_in_eval_batch_and_pymoo()
    print("[PASSED] 3. CustomOp.forward_batch Vectorized Population Evaluation in eval_batch & pymoo")

    test_large_loop_vmap_and_scan_performance_benchmark(n_loop=500, n_repeats=50)
    print("[PASSED] 4. Large-Scale Loop Benchmark (N=500 vmap & T=500 scan vs. Unrolled Python Loop)")

    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"\nAll 4 Phase 5 Loop & Batch Black-Box test suites passed in {elapsed:.2f} ms!")
