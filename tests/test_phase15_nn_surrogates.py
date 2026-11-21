"""
Phase 15 Verification & CPU Throughput Benchmark Suite for `fastopt.nn`
(Fused C++17 MLPNode, float32/float64 Mixed Precision, ResNet, LayerNorm,
Column-Wise Batched Sparsity, NeuralFoilSurrogate, & Opti Integration).
"""

import os
import sys
import tempfile
import time

import numpy as np

# Ensure local fastopt package is importable when run directly
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PY_ROOT = os.path.join(_REPO_ROOT, "python")
if _PY_ROOT not in sys.path:
    sys.path.insert(0, _PY_ROOT)

import fastopt as fo


def _fd_jacobian(fn, x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    y0 = np.asarray(fn(x), dtype=np.float64).ravel()
    J = np.zeros((y0.size, x.size), dtype=np.float64)
    for j in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[j] += eps
        xm[j] -= eps
        J[:, j] = (np.asarray(fn(xp), dtype=np.float64).ravel() -
                   np.asarray(fn(xm), dtype=np.float64).ravel()) / (2.0 * eps)
    return J


def _fd_hessian_from_grad(grad_fn, x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    H = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        xp = x.copy(); xp[j] += eps
        xm = x.copy(); xm[j] -= eps
        H[:, j] = (np.asarray(grad_fn(xp), dtype=np.float64).ravel() -
                   np.asarray(grad_fn(xm), dtype=np.float64).ravel()) / (2.0 * eps)
    return 0.5 * (H + H.T)


def test_1_exact_ad_all_activations_resnet_layernorm():
    print("\n--- [Phase 15 Test 1] Exact 0th/1st/2nd AD (All 8 Activations + ResNet + LayerNorm + Norm) ---")
    rng = np.random.default_rng(123)
    x_test = rng.normal(0.0, 0.7, size=(5,))
    # Avoid kink at 0 for ReLU/ELU FD check by using well-conditioned random weights
    activations = ["identity", "tanh", "silu", "gelu", "relu", "softplus", "elu", "sin"]

    for act in activations:
        mlp = fo.nn.MLP.random(
            [5, 12, 12, 3],
            activation=act,
            residual=True,
            layer_norm=True,
            use_float32=False,
            seed=100 + activations.index(act),
            name=f"mlp_{act}",
        )
        # Attach input/output affine normalization
        in_mean = rng.normal(0.0, 0.2, size=(5,))
        in_std = rng.uniform(0.5, 1.5, size=(5,))
        out_mean = rng.normal(0.0, 0.3, size=(3,))
        out_std = rng.uniform(0.7, 1.3, size=(3,))
        mlp_norm = fo.nn.MLP(
            mlp._layer_configs,
            use_float32=False,
            in_mean=in_mean,
            in_std=in_std,
            out_mean=out_mean,
            out_std=out_std,
            name=f"mlp_{act}_norm",
        )

        gb = fo.GraphBuilder()
        x = gb.variable((5, 1), init=x_test, name="x")
        y = mlp_norm(x)
        # Weighted nonlinear scalar for full 2nd-order Hessian check
        w_vec = gb.constant(np.array([[1.2], [-0.7], [0.5]]))
        f = fo.dot(w_vec, y) + 0.3 * fo.sumsqr(y)
        ev = fo.Evaluator(gb._graph, [f.id, y.id])

        y_ad = ev.eval(x_test)[1].ravel()
        y_dir = mlp_norm(x_test)
        assert np.allclose(y_ad, y_dir, rtol=1e-12, atol=1e-12), f"{act}: direct eval mismatch"

        J_ad = np.asarray(ev.jacobian(x_test, 1))
        J_fd = _fd_jacobian(lambda xv: ev.eval(xv)[1].ravel(order="F"), x_test, eps=1e-6)
        j_err = np.linalg.norm(J_ad - J_fd) / max(1.0, np.linalg.norm(J_fd))
        assert j_err < 1e-7, f"{act}: Jacobian rel_err={j_err:.2e}"

        H_ad = np.asarray(ev.hessian(x_test, 0))
        H_fd = _fd_hessian_from_grad(lambda xv: ev.gradient(xv, 0), x_test, eps=1e-6)
        h_err = np.linalg.norm(H_ad - H_fd) / max(1.0, np.linalg.norm(H_fd))
        assert h_err < 1e-6, f"{act}: Hessian rel_err={h_err:.2e}"

    # Also verify float32 mixed-precision AD accuracy against float64 reference
    mlp_f64 = fo.nn.MLP.random([5, 16, 16, 3], activation="silu", residual=True, layer_norm=True, use_float32=False, seed=7)
    mlp_f32 = fo.nn.MLP(mlp_f64._layer_configs, use_float32=True, name="mlp_fp32")
    gb64 = fo.GraphBuilder(); x64 = gb64.variable((5, 1), init=x_test); y64 = mlp_f64(x64); f64 = fo.sumsqr(y64)
    gb32 = fo.GraphBuilder(); x32 = gb32.variable((5, 1), init=x_test); y32 = mlp_f32(x32); f32 = fo.sumsqr(y32)
    ev64 = fo.Evaluator(gb64._graph, [f64.id, y64.id])
    ev32 = fo.Evaluator(gb32._graph, [f32.id, y32.id])
    val_diff = np.linalg.norm(ev32.eval(x_test)[1] - ev64.eval(x_test)[1]) / np.linalg.norm(ev64.eval(x_test)[1])
    jac_diff = np.linalg.norm(ev32.jacobian(x_test, 1) - ev64.jacobian(x_test, 1)) / np.linalg.norm(ev64.jacobian(x_test, 1))
    hess_diff = np.linalg.norm(ev32.hessian(x_test, 0) - ev64.hessian(x_test, 0)) / np.linalg.norm(ev64.hessian(x_test, 0))
    assert val_diff < 1e-5 and jac_diff < 1e-5 and hess_diff < 1e-4, (
        f"fp32 vs fp64 mismatch: val={val_diff:.2e}, jac={jac_diff:.2e}, hess={hess_diff:.2e}"
    )
    print(f"  [PASSED Test 1] All 8 activations (ResNet+LN+Norm) exact 1st/2nd AD | fp32 vs fp64: val={val_diff:.2e}, jac={jac_diff:.2e}, hess={hess_diff:.2e}")


def test_2_batched_column_sparsity_and_weight_sharing():
    print("\n--- [Phase 15 Test 2] Multi-Column Batched MLPNode Sparsity & Zero-Copy Weight Sharing ---")
    rng = np.random.default_rng(99)
    d_in, d_out, B = 6, 3, 8
    mlp = fo.nn.MLP.random([d_in, 16, 16, d_out], activation="silu", residual=True, use_float32=False, seed=11)

    gb = fo.GraphBuilder()
    X = gb.variable((d_in, B), init=rng.normal(0.0, 0.5, size=(d_in, B)), name="X")
    Y = mlp(X)
    f = fo.sumsqr(Y)
    ev = fo.Evaluator(gb._graph, [f.id, Y.id])

    x0 = gb._graph.initial_decision_vector()
    J_coo = fo.sparse_jacobian(ev, x0, 1)
    expected_nnz = B * d_in * d_out
    assert J_coo.nnz == expected_nnz, f"Expected block-diagonal nnz={expected_nnz}, got {J_coo.nnz}"

    # Verify sparse Jacobian values match dense FD
    J_sp = J_coo.toarray()
    J_fd = _fd_jacobian(lambda xv: ev.eval(xv)[1].ravel(order="F"), x0, eps=1e-6)
    j_err = np.linalg.norm(J_sp - J_fd) / max(1.0, np.linalg.norm(J_fd))
    assert j_err < 1e-8, f"Batched sparse Jacobian rel_err={j_err:.2e}"

    # Verify sparse Hessian is block-diagonal (only B blocks of d_in x d_in lower triangle)
    H_coo = fo.sparse_hessian(ev, x0, 1.0)
    expected_h_nnz = B * d_in * d_in
    assert H_coo.nnz == expected_h_nnz, f"Expected Hessian nnz={expected_h_nnz}, got {H_coo.nnz}"
    H_sp = H_coo.toarray()
    H_fd = _fd_hessian_from_grad(lambda xv: ev.gradient(xv, 0), x0, eps=1e-6)
    # H_sp is lower-triangular or full symmetric depending on helper; compare lower triangle
    H_sp_full = np.tril(H_sp) + np.tril(H_sp, -1).T
    h_err = np.linalg.norm(H_sp_full - H_fd) / max(1.0, np.linalg.norm(H_fd))
    assert h_err < 1e-6, f"Batched sparse Hessian rel_err={h_err:.2e}"

    # Verify zero-copy weight sharing across 2 distinct symbolic call sites
    gb2 = fo.GraphBuilder()
    x_a = gb2.variable((d_in, 1), init=rng.normal(0.0, 0.5, size=(d_in, 1)))
    x_b = gb2.variable((d_in, 1), init=rng.normal(0.0, 0.5, size=(d_in, 1)))
    y_sum = mlp(x_a) + mlp(x_b)
    ev2 = fo.Evaluator(gb2._graph, [y_sum.id])
    x2_0 = gb2._graph.initial_decision_vector()
    y_ref = mlp(x2_0[:d_in]) + mlp(x2_0[d_in:])
    assert np.allclose(ev2.eval(x2_0)[0].ravel(), y_ref, rtol=1e-12, atol=1e-12)
    print(f"  [PASSED Test 2] Block-diagonal J_nnz={J_coo.nnz}/{d_out*B*d_in*B}, H_nnz={H_coo.nnz} (rel_err J={j_err:.2e}, H={h_err:.2e})")


def test_3_npz_roundtrip_and_neuralfoil_loader():
    print("\n--- [Phase 15 Test 3] .npz Round-Trip, ResNetMLP, & NeuralFoilSurrogate Loader ---")
    rng = np.random.default_rng(2025)
    mlp = fo.nn.MLP.random([6, 24, 24, 2], activation="gelu", residual=True, layer_norm=True, use_float32=False, seed=5)
    x_q = rng.normal(0.0, 0.6, size=(6, 10))
    y_orig = mlp(x_q)

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "test_mlp.npz")
        mlp.save_npz(p)
        mlp_loaded = fo.nn.MLP.from_npz(p)
        y_loaded = mlp_loaded(x_q)
        assert np.allclose(y_orig, y_loaded, rtol=1e-14, atol=1e-14), "npz round-trip mismatch"

        # Synthetic NeuralFoil-format .npz (`net.0.weight`, `net.0.bias`, `net.2.weight`, ...)
        nf_path = os.path.join(td, "nf_synth.npz")
        W0 = rng.normal(0.0, 0.3, size=(32, 19))
        b0 = rng.normal(0.0, 0.05, size=(32,))
        W2 = rng.normal(0.0, 0.25, size=(32, 32))
        b2 = rng.normal(0.0, 0.05, size=(32,))
        W4 = rng.normal(0.0, 0.25, size=(4, 32))
        b4 = rng.normal(0.0, 0.05, size=(4,))
        np.savez(nf_path, **{
            "net.0.weight": W0, "net.0.bias": b0,
            "net.2.weight": W2, "net.2.bias": b2,
            "net.4.weight": W4, "net.4.bias": b4,
        })
        nf = fo.nn.NeuralFoilSurrogate.from_npz(nf_path, use_float32=False)
        x_nf = rng.normal(0.0, 0.4, size=(19,))
        gb = fo.GraphBuilder()
        x_sym = gb.variable((19, 1), init=x_nf)
        y_sym = nf(x_sym)
        ev = fo.Evaluator(gb._graph, [y_sym.id])
        J_ad = np.asarray(ev.jacobian(x_nf, 0))
        J_fd = _fd_jacobian(lambda xv: nf(xv), x_nf, eps=1e-6)
        err = np.linalg.norm(J_ad - J_fd) / max(1.0, np.linalg.norm(J_fd))
        assert err < 1e-8, f"NeuralFoilSurrogate Jacobian rel_err={err:.2e}"

    print(f"  [PASSED Test 3] .npz round-trip exact | NeuralFoilSurrogate C++ AD Jac rel_err={err:.2e}")


def test_4_opti_surrogate_optimization():
    print("\n--- [Phase 15 Test 4] End-to-End fo.Opti Optimization with Fused MLP Surrogate ---")
    # Construct a smooth 2-layer surrogate MLP where y = [CL, CD] and minimize CD + 0.1*||x||^2 s.t. CL >= target
    rng = np.random.default_rng(77)
    W1 = rng.normal(0.0, 0.5, size=(32, 4))
    b1 = rng.normal(0.0, 0.1, size=(32,))
    W2 = rng.normal(0.0, 0.3, size=(2, 32))
    # Bias output so CL ~ 0.6 + ..., CD ~ 0.02 + ...
    b2 = np.array([0.6, 0.02])
    surrogate = fo.nn.MLP.from_weights([W1, W2], [b1, b2], activations="silu", use_float32=False, name="aero_nn")

    opti = fo.Opti()
    x = opti.variable((4, 1), init=np.zeros((4, 1)), lb=-1.5, ub=1.5, name="design")
    y = surrogate(x)
    CL = y[0, 0]
    CD = y[1, 0]
    opti.minimize(CD + 0.05 * fo.sumsqr(x))
    opti.subject_to(CL >= 0.85)
    opti.solver("trust-constr", {"maxiter": 200, "gtol": 1e-7})
    sol = opti.solve()
    assert sol.success, f"Opti MLP optimization failed: {sol.message}"
    y_opt = surrogate(sol.value(x).ravel())
    assert y_opt[0] >= 0.85 - 1e-5, f"CL constraint violated: {y_opt[0]}"
    print(f"  [PASSED Test 4] Opti + Fused MLP converged in {getattr(sol, 'iterations', getattr(sol, 'iter_count', '?'))} iters: CL={y_opt[0]:.4f} (>=0.85), CD={y_opt[1]:.5f}")


def test_5_large_cpu_nn_benchmark():
    print("\n--- [Phase 15 Test 5] Large CPU Neural Network Throughput Benchmark (Fused fp32 vs fp64 vs Unfused) ---")
    rng = np.random.default_rng(2026)

    # 5A. NeuralFoil-L scale: 16 -> 128 x 4 -> 4 (SiLU)
    dims_med = [16, 128, 128, 128, 128, 4]
    mlp_med_f64 = fo.nn.MLP.random(dims_med, activation="silu", use_float32=False, seed=1)
    mlp_med_f32 = fo.nn.MLP(mlp_med_f64._layer_configs, use_float32=True)

    # Build unfused graph for comparison (MatMul + Add + x/(1+exp(-x)))
    gb_unf = fo.GraphBuilder()
    x_u = gb_unf.variable((16, 1), init=np.zeros((16, 1)))
    h = x_u
    for l, cfg in enumerate(mlp_med_f64._layer_configs):
        z = gb_unf.constant(cfg.W) @ h + gb_unf.constant(cfg.b.reshape(-1, 1))
        if l < len(mlp_med_f64._layer_configs) - 1:
            h = z / (1.0 + fo.exp(-z))
        else:
            h = z
    ev_unf = fo.Evaluator(gb_unf._graph, [fo.sum(h).id])

    gb_f64 = fo.GraphBuilder(); x64 = gb_f64.variable((16, 1), init=np.zeros((16, 1)))
    ev_f64 = fo.Evaluator(gb_f64._graph, [fo.sum(mlp_med_f64(x64)).id])

    gb_f32 = fo.GraphBuilder(); x32 = gb_f32.variable((16, 1), init=np.zeros((16, 1)))
    ev_f32 = fo.Evaluator(gb_f32._graph, [fo.sum(mlp_med_f32(x32)).id])

    x_1 = rng.normal(0.0, 0.5, size=(16,))
    # Warmup
    for _ in range(20):
        ev_unf.gradient(x_1, 0); ev_f64.gradient(x_1, 0); ev_f32.gradient(x_1, 0)

    n_rep = 300
    t0 = time.perf_counter()
    for _ in range(n_rep):
        ev_unf.invalidate_forward_cache(); ev_unf.gradient(x_1, 0)
    dt_unf_us = (time.perf_counter() - t0) * 1e6 / n_rep

    t0 = time.perf_counter()
    for _ in range(n_rep):
        ev_f64.invalidate_forward_cache(); ev_f64.gradient(x_1, 0)
    dt_f64_us = (time.perf_counter() - t0) * 1e6 / n_rep

    t0 = time.perf_counter()
    for _ in range(n_rep):
        ev_f32.invalidate_forward_cache(); ev_f32.gradient(x_1, 0)
    dt_f32_us = (time.perf_counter() - t0) * 1e6 / n_rep

    print(
        f"  [Med 4x128 SiLU, B=1 Fwd+Grad] Unfused={dt_unf_us:.1f} us | "
        f"Fused fp64={dt_f64_us:.1f} us ({dt_unf_us/dt_f64_us:.2f}x) | "
        f"Fused fp32={dt_f32_us:.1f} us ({dt_unf_us/dt_f32_us:.2f}x)"
    )

    # 5B. Large Surrogate: 32 -> 512 x 6 -> 8 (~1.33M params, ResNet + SiLU)
    dims_lg = [32, 512, 512, 512, 512, 512, 512, 8]
    mlp_lg_f64 = fo.nn.MLP.random(dims_lg, activation="silu", residual=True, use_float32=False, seed=2)
    mlp_lg_f32 = fo.nn.MLP(mlp_lg_f64._layer_configs, use_float32=True)

    X_batch = np.asfortranarray(rng.normal(0.0, 0.5, size=(32, 1024)))
    # Warmup
    _ = mlp_lg_f64.eval(X_batch)
    _ = mlp_lg_f32.eval(X_batch)

    t0 = time.perf_counter()
    for _ in range(5):
        _ = mlp_lg_f64.eval(X_batch)
    dt_lg64_ms = (time.perf_counter() - t0) * 1e3 / 5.0

    t0 = time.perf_counter()
    for _ in range(5):
        _ = mlp_lg_f32.eval(X_batch)
    dt_lg32_ms = (time.perf_counter() - t0) * 1e3 / 5.0

    # Compute effective GFLOP/s (2 * params * B)
    n_params = sum(dims_lg[i] * dims_lg[i + 1] for i in range(len(dims_lg) - 1))
    gflops_64 = (2.0 * n_params * 1024) / (dt_lg64_ms * 1e-3) / 1e9
    gflops_32 = (2.0 * n_params * 1024) / (dt_lg32_ms * 1e-3) / 1e9
    print(
        f"  [Large 6x512 ResNet ({n_params/1e6:.2f}M params), B=1024 Fwd] "
        f"fp64={dt_lg64_ms:.2f} ms ({gflops_64:.1f} GFLOP/s) | "
        f"fp32={dt_lg32_ms:.2f} ms ({gflops_32:.1f} GFLOP/s, {dt_lg64_ms/dt_lg32_ms:.2f}x vs fp64)"
    )
    assert dt_f32_us < dt_unf_us, "Expected fused fp32 to outperform unfused graph"
    print("  [PASSED Test 5] CPU throughput benchmark verified.")




def test_6_jax_cpu_comparison():
    print("\n--- [Phase 15 Test 6] Head-to-Head CPU Benchmark: fastopt (fp32) vs JAX CPU JIT (fp32) ---")
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("  [SKIPPED Test 6] jax is not installed in this Python environment.")
        return

    cpu_devs = jax.devices("cpu")
    dev = cpu_devs[0]
    rng = np.random.default_rng(2026)

    def _build_jax_mlp(mlp_fo):
        params = []
        for cfg in mlp_fo._layer_configs:
            W = jax.device_put(jnp.asarray(cfg.W, dtype=jnp.float32), dev)
            b = jax.device_put(jnp.asarray(cfg.b, dtype=jnp.float32), dev)
            params.append((W, b, bool(cfg.residual), str(cfg.activation).lower()))

        @jax.jit
        def fwd_batch(X_col):
            # X_col has shape (d_in, B), matching fastopt column-major convention
            h = X_col
            for W, b, res, act in params:
                z = jnp.dot(W, h) + b[:, None]
                if act == "silu":
                    a = jax.nn.silu(z)
                elif act == "tanh":
                    a = jnp.tanh(z)
                elif act == "relu":
                    a = jax.nn.relu(z)
                else:
                    a = z
                h = (a + h) if res else a
            return h

        @jax.jit
        def val_and_grad_batch(X_col):
            def loss_fn(x_):
                return jnp.sum(fwd_batch(x_))
            return jax.value_and_grad(loss_fn)(X_col)

        return fwd_batch, val_and_grad_batch

    # Case A: Medium 4x128 SiLU, B=1 (Forward + Gradient, typical optimization loop)
    mlp_med = fo.nn.MLP.random([16, 128, 128, 128, 128, 4], activation="silu", use_float32=True, seed=10)
    gb_med = fo.GraphBuilder()
    x_med_sym = gb_med.variable((16, 1), init=np.zeros((16, 1)))
    ev_med = fo.Evaluator(gb_med._graph, [fo.sum(mlp_med(x_med_sym)).id])
    _, jax_vg_med = _build_jax_mlp(mlp_med)

    x1_np = rng.normal(0.0, 0.5, size=(16,))
    x1_jnp = jax.device_put(jnp.asarray(x1_np.reshape(16, 1), dtype=jnp.float32), dev)
    # Warmup
    for _ in range(20):
        ev_med.invalidate_forward_cache(); ev_med.gradient(x1_np, 0)
        v_j, g_j = jax_vg_med(x1_jnp); g_j.block_until_ready()

    g_fo = np.asarray(ev_med.gradient(x1_np, 0)).ravel()
    g_jx = np.asarray(g_j).ravel()
    err_a = np.linalg.norm(g_fo - g_jx) / max(1.0, np.linalg.norm(g_jx))

    n_a = 300
    t0 = time.perf_counter()
    for _ in range(n_a):
        ev_med.invalidate_forward_cache(); ev_med.gradient(x1_np, 0)
    fo_a_us = (time.perf_counter() - t0) * 1e6 / n_a

    t0 = time.perf_counter()
    for _ in range(n_a):
        _, g_j = jax_vg_med(x1_jnp); g_j.block_until_ready()
    jx_a_us = (time.perf_counter() - t0) * 1e6 / n_a

    print(
        f"  [Case A: 4x128 SiLU, B=1 Fwd+Grad] fastopt={fo_a_us:.1f} us | "
        f"JAX-CPU={jx_a_us:.1f} us | speedup={jx_a_us/fo_a_us:.2f}x | grad_diff={err_a:.2e}"
    )

    # Case B & C & D: Large 6x512 ResNet-SiLU (1.33M params)
    mlp_lg = fo.nn.MLP.random([32, 512, 512, 512, 512, 512, 512, 8], activation="silu", residual=True, use_float32=True, seed=20)
    jax_fwd_lg, jax_vg_lg = _build_jax_mlp(mlp_lg)

    # Case B: Large 6x512, B=1 Fwd+Grad
    gb_lg1 = fo.GraphBuilder()
    x_lg1_sym = gb_lg1.variable((32, 1), init=np.zeros((32, 1)))
    ev_lg1 = fo.Evaluator(gb_lg1._graph, [fo.sum(mlp_lg(x_lg1_sym)).id])
    x32_np = rng.normal(0.0, 0.5, size=(32,))
    x32_jnp = jax.device_put(jnp.asarray(x32_np.reshape(32, 1), dtype=jnp.float32), dev)
    for _ in range(5):
        ev_lg1.invalidate_forward_cache(); ev_lg1.gradient(x32_np, 0)
        _, g_j = jax_vg_lg(x32_jnp); g_j.block_until_ready()

    n_b = 40
    t0 = time.perf_counter()
    for _ in range(n_b):
        ev_lg1.invalidate_forward_cache(); ev_lg1.gradient(x32_np, 0)
    fo_b_ms = (time.perf_counter() - t0) * 1e3 / n_b

    t0 = time.perf_counter()
    for _ in range(n_b):
        _, g_j = jax_vg_lg(x32_jnp); g_j.block_until_ready()
    jx_b_ms = (time.perf_counter() - t0) * 1e3 / n_b

    print(
        f"  [Case B: 6x512 ResNet (1.33M), B=1 Fwd+Grad] fastopt={fo_b_ms:.3f} ms | "
        f"JAX-CPU={jx_b_ms:.3f} ms | speedup={jx_b_ms/fo_b_ms:.2f}x"
    )

    # Case C: Large 6x512, B=1024 Forward
    X_1024 = np.asfortranarray(rng.normal(0.0, 0.5, size=(32, 1024)))
    X_1024_jnp = jax.device_put(jnp.asarray(X_1024, dtype=jnp.float32), dev)
    y_fo_c = mlp_lg.eval(X_1024)
    y_jx_c = jax_fwd_lg(X_1024_jnp); y_jx_c.block_until_ready()
    err_c = np.linalg.norm(y_fo_c - np.asarray(y_jx_c)) / max(1.0, np.linalg.norm(np.asarray(y_jx_c)))

    n_c = 5
    t0 = time.perf_counter()
    for _ in range(n_c):
        _ = mlp_lg.eval(X_1024)
    fo_c_ms = (time.perf_counter() - t0) * 1e3 / n_c

    t0 = time.perf_counter()
    for _ in range(n_c):
        y_jx_c = jax_fwd_lg(X_1024_jnp); y_jx_c.block_until_ready()
    jx_c_ms = (time.perf_counter() - t0) * 1e3 / n_c

    print(
        f"  [Case C: 6x512 ResNet (1.33M), B=1024 Fwd] fastopt={fo_c_ms:.2f} ms | "
        f"JAX-CPU={jx_c_ms:.2f} ms | speedup={jx_c_ms/fo_c_ms:.2f}x | val_diff={err_c:.2e}"
    )

    # Case D: Large 6x512, B=256 Forward + Gradient
    gb_lg256 = fo.GraphBuilder()
    X_256_init = np.asfortranarray(rng.normal(0.0, 0.5, size=(32, 256)))
    X_256_sym = gb_lg256.variable((32, 256), init=X_256_init)
    ev_lg256 = fo.Evaluator(gb_lg256._graph, [fo.sum(mlp_lg(X_256_sym)).id])
    x256_flat = gb_lg256._graph.initial_decision_vector()
    X_256_jnp = jax.device_put(jnp.asarray(X_256_init, dtype=jnp.float32), dev)

    ev_lg256.invalidate_forward_cache(); g_fo_d = np.asarray(ev_lg256.gradient(x256_flat, 0)).reshape((32, 256), order="F")
    _, g_jx_d = jax_vg_lg(X_256_jnp); g_jx_d.block_until_ready()
    err_d = np.linalg.norm(g_fo_d - np.asarray(g_jx_d)) / max(1.0, np.linalg.norm(np.asarray(g_jx_d)))

    n_d = 5
    t0 = time.perf_counter()
    for _ in range(n_d):
        ev_lg256.invalidate_forward_cache(); ev_lg256.gradient(x256_flat, 0)
    fo_d_ms = (time.perf_counter() - t0) * 1e3 / n_d

    t0 = time.perf_counter()
    for _ in range(n_d):
        _, g_jx_d = jax_vg_lg(X_256_jnp); g_jx_d.block_until_ready()
    jx_d_ms = (time.perf_counter() - t0) * 1e3 / n_d

    print(
        f"  [Case D: 6x512 ResNet (1.33M), B=256 Fwd+Grad] fastopt={fo_d_ms:.2f} ms | "
        f"JAX-CPU={jx_d_ms:.2f} ms | speedup={jx_d_ms/fo_d_ms:.2f}x | grad_diff={err_d:.2e}"
    )
    print("  [PASSED Test 6] fastopt vs JAX CPU comparison completed.")


def run_phase15_suite():
    print("=" * 88)
    print("Running fastopt Phase 15 CPU Neural Network Surrogate Engine Verification Suite")
    print("=" * 88)
    test_1_exact_ad_all_activations_resnet_layernorm()
    test_2_batched_column_sparsity_and_weight_sharing()
    test_3_npz_roundtrip_and_neuralfoil_loader()
    test_4_opti_surrogate_optimization()
    test_5_large_cpu_nn_benchmark()
    test_6_jax_cpu_comparison()
    print("\nALL PHASE 15 TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_phase15_suite()
