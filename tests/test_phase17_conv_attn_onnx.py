"""
Phase 17 Verification & Benchmark Suite:
Native C++17 `Conv1D`, `MultiHeadAttention`, `TransformerBlock`, `Sequential`,
and Zero-Dependency ONNX Model Importer (`fo.nn.from_onnx`) with Automatic C++ Fusion.
"""

import os
import sys
import tempfile
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PY_ROOT = os.path.join(_REPO_ROOT, "python")
if _PY_ROOT not in sys.path:
    sys.path.insert(0, _PY_ROOT)

import fastopt as fo
from fastopt.onnx_import import write_minimal_onnx


def _fd_jacobian(fn, x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    y0 = np.asarray(fn(x), dtype=np.float64).ravel(order="F")
    J = np.zeros((y0.size, x.size), dtype=np.float64)
    for j in range(x.size):
        xp = x.copy(); xp[j] += eps
        xm = x.copy(); xm[j] -= eps
        J[:, j] = (np.asarray(fn(xp), dtype=np.float64).ravel(order="F") -
                   np.asarray(fn(xm), dtype=np.float64).ravel(order="F")) / (2.0 * eps)
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


def test_1_conv1d_exact_ad_and_banded_sparsity():
    print("\n--- [Phase 17 Test 1] Native C++ Conv1D Exact 0th/1st/2nd AD & Banded Sparsity ---")
    rng = np.random.default_rng(101)
    c_in, c_out, l_in = 3, 4, 10
    X0 = rng.normal(0.0, 0.5, size=(c_in, l_in))

    for act in ("identity", "silu", "tanh", "gelu"):
        conv = fo.nn.Conv1D.random(
            c_in, c_out, kernel_size=3, stride=2, padding=1, dilation=2,
            activation=act, use_float32=False, seed=10,
        )
        gb = fo.GraphBuilder()
        X = gb.variable((c_in, l_in), init=X0, name="X")
        Y = conv(X)
        f = 0.5 * fo.sumsqr(Y) + 0.2 * fo.sum(Y)
        ev = fo.Evaluator(gb._graph, [f.id, Y.id])
        x0 = gb._graph.initial_decision_vector()

        # Numeric vs symbolic forward parity
        Y_dir = conv(X0)
        Y_sym = ev.eval(x0)[1]
        assert np.allclose(Y_dir, Y_sym, rtol=1e-12, atol=1e-12), f"{act}: Conv1D direct/sym mismatch"

        # Exact 1st-order sparse Jacobian vs FD
        J_coo = fo.sparse_jacobian(ev, x0, 1)
        J_fd = _fd_jacobian(lambda xv: ev.eval(xv)[1], x0, eps=1e-6)
        j_err = np.linalg.norm(J_coo.toarray() - J_fd) / max(1.0, np.linalg.norm(J_fd))
        assert j_err < 1e-8, f"{act}: Conv1D Jacobian rel_err={j_err:.2e}"
        # Banded sparsity strictly smaller than dense
        dense_sz = Y_sym.size * X0.size
        assert J_coo.nnz < dense_sz * 0.5, f"Expected banded sparsity, got nnz={J_coo.nnz}/{dense_sz}"

        # Exact 2nd-order Hessian vs FD
        H_ad = np.asarray(ev.hessian(x0, 0))
        H_fd = _fd_hessian_from_grad(lambda xv: ev.gradient(xv, 0), x0, eps=1e-6)
        h_err = np.linalg.norm(H_ad - H_fd) / max(1.0, np.linalg.norm(H_fd))
        assert h_err < 1e-7, f"{act}: Conv1D Hessian rel_err={h_err:.2e}"

    print(
        f"  [PASSED Test 1] Conv1D (K=3, S=2, P=1, D=2) exact 1st/2nd AD | "
        f"J_err={j_err:.2e}, H_err={h_err:.2e}, banded J_nnz={J_coo.nnz}/{dense_sz}"
    )


def test_2_attention_and_transformer_exact_ad():
    print("\n--- [Phase 17 Test 2] Native C++ MultiHeadAttention & TransformerBlock Exact 0th/1st/2nd AD ---")
    rng = np.random.default_rng(202)
    d_model, seq_len, n_heads = 8, 5, 2
    X0 = rng.normal(0.0, 0.5, size=(d_model, seq_len))

    for causal in (False, True):
        mha = fo.nn.MultiHeadAttention.random(
            d_model, num_heads=n_heads, causal=causal, residual=True,
            use_float32=False, seed=33,
        )
        gb = fo.GraphBuilder()
        X = gb.variable((d_model, seq_len), init=X0, name="X")
        Y = mha(X)
        f = 0.5 * fo.sumsqr(Y) + 0.1 * fo.sum(Y)
        ev = fo.Evaluator(gb._graph, [f.id, Y.id])
        x0 = gb._graph.initial_decision_vector()

        assert np.allclose(mha(X0), ev.eval(x0)[1], rtol=1e-12, atol=1e-12)

        J_coo = fo.sparse_jacobian(ev, x0, 1)
        J_fd = _fd_jacobian(lambda xv: ev.eval(xv)[1], x0, eps=1e-6)
        j_err = np.linalg.norm(J_coo.toarray() - J_fd) / max(1.0, np.linalg.norm(J_fd))
        assert j_err < 1e-8, f"causal={causal}: MHA Jacobian rel_err={j_err:.2e}"

        if causal:
            # Causal attention Jacobian has block lower-triangular sparsity: L*(L+1)/2 * d_model^2
            expected_nnz = (seq_len * (seq_len + 1) // 2) * d_model * d_model
            assert J_coo.nnz == expected_nnz, f"Expected causal nnz={expected_nnz}, got {J_coo.nnz}"

        H_ad = np.asarray(ev.hessian(x0, 0))
        H_fd = _fd_hessian_from_grad(lambda xv: ev.gradient(xv, 0), x0, eps=1e-6)
        h_err = np.linalg.norm(H_ad - H_fd) / max(1.0, np.linalg.norm(H_fd))
        assert h_err < 1e-7, f"causal={causal}: MHA Hessian rel_err={h_err:.2e}"

    # Verify full TransformerBlock (Attention + LayerNorm FFN MLP) exact 2nd AD
    tb = fo.nn.TransformerBlock.random(d_model, num_heads=2, ffn_hidden=16, activation="gelu", seed=55)
    gb_tb = fo.GraphBuilder()
    X_tb = gb_tb.variable((d_model, seq_len), init=X0)
    Y_tb = tb(X_tb)
    f_tb = fo.sumsqr(Y_tb)
    ev_tb = fo.Evaluator(gb_tb._graph, [f_tb.id, Y_tb.id])
    x_tb = gb_tb._graph.initial_decision_vector()
    H_tb = np.asarray(ev_tb.hessian(x_tb, 0))
    H_tb_fd = _fd_hessian_from_grad(lambda xv: ev_tb.gradient(xv, 0), x_tb, eps=1e-6)
    tb_h_err = np.linalg.norm(H_tb - H_tb_fd) / max(1.0, np.linalg.norm(H_tb_fd))
    assert tb_h_err < 1e-7, f"TransformerBlock Hessian rel_err={tb_h_err:.2e}"

    print(
        f"  [PASSED Test 2] MultiHeadAttention (non-causal & causal nnz={J_coo.nnz}/{X0.size**2}) "
        f"J_err={j_err:.2e}, H_err={h_err:.2e} | TransformerBlock H_err={tb_h_err:.2e}"
    )


def test_3_onnx_importer_auto_fusion_and_ad():
    print("\n--- [Phase 17 Test 3] Zero-Dependency ONNX Importer (Auto-MLP & Conv1D Fusion + Exact 2nd AD) ---")
    rng = np.random.default_rng(303)

    with tempfile.TemporaryDirectory() as td:
        # 3A: Binary .onnx ResNet-SiLU MLP exported in PyTorch Gemm + Sigmoid/Mul + Add format
        W0 = rng.normal(0.0, 0.4, size=(12, 5)).astype(np.float64)
        b0 = rng.normal(0.0, 0.05, size=(12,)).astype(np.float64)
        W1 = rng.normal(0.0, 0.3, size=(12, 12)).astype(np.float64)
        b1 = rng.normal(0.0, 0.05, size=(12,)).astype(np.float64)
        W2 = rng.normal(0.0, 0.3, size=(3, 12)).astype(np.float64)
        b2 = rng.normal(0.0, 0.05, size=(3,)).astype(np.float64)

        onnx_mlp_path = os.path.join(td, "resnet_silu.onnx")
        write_minimal_onnx(
            onnx_mlp_path,
            nodes=[
                ("Gemm", ["x", "W0", "b0"], ["z0"], {"transB": 1}),
                ("Sigmoid", ["z0"], ["s0"], {}),
                ("Mul", ["z0", "s0"], ["h0"], {}),
                ("Gemm", ["h0", "W1", "b1"], ["z1"], {"transB": 1}),
                ("Sigmoid", ["z1"], ["s1"], {}),
                ("Mul", ["z1", "s1"], ["a1"], {}),
                ("Add", ["a1", "h0"], ["h1"], {}),  # residual skip
                ("Gemm", ["h1", "W2", "b2"], ["y"], {"transB": 1}),
            ],
            initializers={"W0": W0, "b0": b0, "W1": W1, "b1": b1, "W2": W2, "b2": b2},
            input_names=["x"],
            output_names=["y"],
        )

        model = fo.nn.from_onnx(onnx_mlp_path, use_float32=False, fuse_mlp=True)
        assert model.fused_mlp_count == 1, f"Expected 1 fused C++ MLP, got {model.fused_mlp_count}"

        x_in = rng.normal(0.0, 0.6, size=(5, 1))
        # Manual NumPy reference
        def _silu(u): return u / (1.0 + np.exp(-u))
        h0_ref = _silu(W0 @ x_in + b0[:, None])
        h1_ref = _silu(W1 @ h0_ref + b1[:, None]) + h0_ref
        y_ref = W2 @ h1_ref + b2[:, None]

        y_onnx = model(x_in)
        assert np.allclose(y_onnx, y_ref, rtol=1e-12, atol=1e-12), "ONNX MLP numeric mismatch"

        # Symbolic 1st & 2nd AD through ONNX-imported model
        gb = fo.GraphBuilder()
        x_sym = gb.variable((5, 1), init=x_in)
        y_sym = model(x_sym)
        f_sym = fo.sumsqr(y_sym)
        ev = fo.Evaluator(gb._graph, [f_sym.id, y_sym.id])
        H_ad = np.asarray(ev.hessian(x_in.ravel(), 0))
        H_fd = _fd_hessian_from_grad(lambda xv: ev.gradient(xv, 0), x_in.ravel(), eps=1e-6)
        h_err = np.linalg.norm(H_ad - H_fd) / max(1.0, np.linalg.norm(H_fd))
        assert h_err < 1e-8, f"ONNX MLP Hessian rel_err={h_err:.2e}"

        # 3B: Binary .onnx Conv1D + Tanh model
        Wc = rng.normal(0.0, 0.4, size=(4, 3, 3)).astype(np.float64)
        bc = rng.normal(0.0, 0.05, size=(4,)).astype(np.float64)
        onnx_conv_path = os.path.join(td, "conv1d.onnx")
        write_minimal_onnx(
            onnx_conv_path,
            nodes=[
                ("Conv", ["x", "Wc", "bc"], ["z"], {"strides": [1], "pads": [1, 1], "dilations": [1]}),
                ("Tanh", ["z"], ["y"], {}),
            ],
            initializers={"Wc": Wc, "bc": bc},
            input_names=["x"],
            output_names=["y"],
        )
        cmodel = fo.nn.from_onnx(onnx_conv_path, use_float32=False)
        assert cmodel.fused_conv1d_count == 1
        X_c = rng.normal(0.0, 0.5, size=(3, 8))
        ref_c = fo.nn.Conv1D(Wc, bc, stride=1, padding=1, dilation=1, activation="tanh")(X_c)
        assert np.allclose(cmodel(X_c), ref_c, rtol=1e-12, atol=1e-12)

        print(
            f"  [PASSED Test 3] ONNX ResNet-SiLU auto-fused to {model.fused_mlp_count} C++ MLPNode "
            f"(H_err={h_err:.2e}) | ONNX Conv1D+Tanh auto-fused to {cmodel.fused_conv1d_count} C++ Conv1DNode"
        )


def test_4_opti_with_conv1d_transformer_surrogate():
    print("\n--- [Phase 17 Test 4] End-to-End fo.Opti Optimization with Sequential(Conv1D, TransformerBlock) ---")
    net = fo.nn.Sequential(
        fo.nn.Conv1D.random(4, 8, kernel_size=3, stride=1, padding=1, activation="silu", seed=1),
        fo.nn.TransformerBlock.random(8, num_heads=2, ffn_hidden=16, activation="gelu", seed=2),
    )
    opti = fo.Opti()
    X = opti.variable((4, 6), init=np.zeros((4, 6)), lb=-1.0, ub=1.0, name="field")
    Y = net(X)
    # Minimize L2 norm of output + input regularization subject to mean(Y) >= 0.25
    opti.minimize(0.5 * fo.sumsqr(Y) + 0.1 * fo.sumsqr(X))
    opti.subject_to(fo.mean(Y) >= 0.25)
    opti.solver("trust-constr", {"maxiter": 150, "gtol": 1e-6})
    sol = opti.solve()
    assert sol.success, f"Conv1D+Transformer Opti failed: {sol.message}"
    Y_opt = net(sol.value(X))
    assert float(np.mean(Y_opt)) >= 0.25 - 1e-4
    print(f"  [PASSED Test 4] Opti + Conv1D + TransformerBlock converged: mean(Y*)={np.mean(Y_opt):.4f} (>=0.25)")


def run_phase17_suite():
    print("=" * 90)
    print("Running fastopt Phase 17 Conv1D, MultiHeadAttention, Transformer & ONNX Importer Suite")
    print("=" * 90)
    test_1_conv1d_exact_ad_and_banded_sparsity()
    test_2_attention_and_transformer_exact_ad()
    test_3_onnx_importer_auto_fusion_and_ad()
    test_4_opti_with_conv1d_transformer_surrogate()
    print("\nALL PHASE 17 TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_phase17_suite()
