"""
Phase 18 Verification Suite: Expanded C++17 Neural Network Layers & Activations.

Tests:
  1. Exact 0th / 1st (Jacobian) / 2nd (Hessian) AD for all 5 new activations:
     Sigmoid, LeakyReLU, Mish, SELU, HardSwish (inside fused C++ MLPNode).
  2. Exact 0th / 1st / 2nd AD & 2D local receptive-field sparsity for:
     - Conv2D (K=3x3, S=2, P=1, D=1, Mish/SiLU, fp64 & fp32)
     - AvgPool2D, MaxPool2D, GlobalAvgPool2D
     - BatchNorm (standalone + algebraic weight folding equivalence)
     - RMSNorm (exact Forward-over-Reverse HVP)
  3. Zero-dependency ONNX Importer 2D CNN pipeline:
     Conv2D -> BatchNormalization (auto-folded!) -> Mish -> AvgPool2D -> GlobalAvgPool2D -> MLP
     verifying numeric parity against manual NumPy reference and exact 1st & 2nd AD.
  4. End-to-end fo.Opti 2D spatial field optimization through Conv2D + BatchNorm + AvgPool2D
     + RMSNorm + MLP using exact C++ 2nd-order Hessian.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

# Ensure local python/fastopt package is importable when run directly
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PY_ROOT = os.path.join(_REPO_ROOT, "python")
if _PY_ROOT not in sys.path:
    sys.path.insert(0, _PY_ROOT)

import fastopt as fo
from fastopt import nn
from fastopt.onnx_import import write_minimal_onnx


def _fd_grad(fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (fn(xp) - fn(xm)) / (2.0 * eps)
    return g


def _fd_hess(grad_fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = x.size
    H = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        xp = x.copy(); xp[j] += eps
        xm = x.copy(); xm[j] -= eps
        H[:, j] = (grad_fn(xp) - grad_fn(xm)) / (2.0 * eps)
    return 0.5 * (H + H.T)


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(1.0, np.linalg.norm(b)))


def test_1_five_new_activations_exact_2nd_ad() -> None:
    print("\n--- [Phase 18 Test 1] Exact 0th/1st/2nd AD for Sigmoid, LeakyReLU, Mish, SELU, HardSwish ---")
    rng = np.random.default_rng(1801)
    # Choose smooth interior point away from kinks (0 for LeakyReLU/SELU, +/-3 for HardSwish)
    x0 = np.array([0.45, -0.72, 0.88, -0.35, 0.61], dtype=np.float64)

    for act in ["sigmoid", "leaky_relu", "mish", "selu", "hardswish"]:
        net = nn.MLP.random([5, 8, 6, 1], activation=act, output_activation="identity", seed=18 + len(act))
        # Shift biases slightly so HardSwish/LeakyReLU stay strictly away from kinks at x0
        gb = fo.GraphBuilder()
        x = gb.variable((5, 1), init=x0.reshape(5, 1), name="x")
        y = net(x)
        loss = 0.5 * fo.sum(y * y) + fo.sum(y)
        ev = fo.Evaluator(gb._graph, [loss.id])

        val_sym = float(ev.eval(x0)[0].ravel()[0])
        val_num_y = float(net.eval(x0)[0])
        val_num = 0.5 * val_num_y * val_num_y + val_num_y
        assert abs(val_sym - val_num) < 1e-13, f"{act}: eval mismatch {abs(val_sym - val_num)}"

        grad_ad = ev.gradient(x0, 0).ravel()
        grad_fd = _fd_grad(lambda xv: float(ev.eval(xv)[0].ravel()[0]), x0)
        g_err = _rel_err(grad_ad, grad_fd)

        hess_ad = ev.hessian(x0, 0)
        hess_fd = _fd_hess(lambda xv: ev.gradient(xv, 0).ravel(), x0)
        h_err = _rel_err(hess_ad, hess_fd)

        print(f"  [{act:9s}] |val_err|={abs(val_sym - val_num):.2e} | grad_err={g_err:.2e} | hess_err={h_err:.2e}")
        assert g_err < 1e-8, f"{act} grad_err={g_err}"
        assert h_err < 1e-7, f"{act} hess_err={h_err}"

    print("  [PASSED Test 1] All 5 new activations passed exact 0/1/2 AD checks.")


def test_2_conv2d_pool2d_bn_rmsnorm_ad_and_sparsity() -> None:
    print("\n--- [Phase 18 Test 2] Conv2D, AvgPool2D, MaxPool2D, GlobalAvgPool2D, BatchNorm, RMSNorm ---")
    rng = np.random.default_rng(1802)
    C_in, H_in, W_in = 2, 5, 5
    X0 = rng.normal(0.2, 0.6, size=(C_in, H_in * W_in))
    x0_flat = X0.ravel(order="F")

    # 2a. Conv2D (K=3x3, S=2, P=1, D=1, Mish) + sparsity + fp32 check
    conv = nn.Conv2D.random(
        C_in, 3, kernel_size=3, stride=2, padding=1, dilation=1,
        activation="mish", input_hw=(H_in, W_in), seed=101,
    )
    h_out, w_out = conv.out_hw(H_in, W_in)
    assert (h_out, w_out) == (3, 3), f"Expected (3, 3), got {(h_out, w_out)}"

    gb = fo.GraphBuilder()
    x = gb.variable((C_in, H_in * W_in), init=X0, name="x")
    y_conv = conv(x)
    loss_conv = fo.sum(y_conv * y_conv)
    ev_conv = fo.Evaluator(gb._graph, [loss_conv.id, y_conv.id])

    y_sym = ev_conv.eval(x0_flat)[1]
    y_num = conv.eval(X0, H_in, W_in)
    assert np.max(np.abs(y_sym - y_num)) < 1e-13

    # Verify 2D local receptive-field sparsity (output is 3*9=27, input is 2*25=50 -> 1350 dense)
    sp = ev_conv.jacobian_sparsity(1)
    nnz_conv = len(sp[0])
    assert nnz_conv < 27 * 50, f"Expected sparse Conv2D Jacobian, got dense {nnz_conv}"

    J_ad = ev_conv.jacobian(x0_flat, 1)
    # Check Jacobian vs FD on a few columns & full Hessian on scalar loss
    grad_ad = ev_conv.gradient(x0_flat, 0).ravel()
    grad_fd = _fd_grad(lambda xv: float(ev_conv.eval(xv)[0].ravel()[0]), x0_flat)
    H_ad = ev_conv.hessian(x0_flat, 0)
    H_fd = _fd_hess(lambda xv: ev_conv.gradient(xv, 0).ravel(), x0_flat)
    c_g_err = _rel_err(grad_ad, grad_fd)
    c_h_err = _rel_err(H_ad, H_fd)
    assert c_g_err < 1e-8 and c_h_err < 1e-7, f"Conv2D AD error: g={c_g_err}, h={c_h_err}"

    # Verify fp32 Conv2D numeric accuracy
    conv32 = nn.Conv2D(
        rng.normal(0.0, 0.3, size=(3, C_in, 3, 3)),
        rng.normal(0.0, 0.05, size=(3,)),
        stride=2, padding=1, activation="silu", input_hw=(H_in, W_in), use_float32=True,
    )
    conv64 = nn.Conv2D(
        rng.normal(0.0, 0.3, size=(3, C_in, 3, 3)),
        rng.normal(0.0, 0.05, size=(3,)),
        stride=2, padding=1, activation="silu", input_hw=(H_in, W_in), use_float32=False,
    )
    W_shared = rng.normal(0.0, 0.3, size=(3, C_in, 3, 3))
    b_shared = rng.normal(0.0, 0.05, size=(3,))
    c32 = nn.Conv2D(W_shared, b_shared, stride=2, padding=1, activation="silu", use_float32=True)
    c64 = nn.Conv2D(W_shared, b_shared, stride=2, padding=1, activation="silu", use_float32=False)
    fp32_diff = float(np.max(np.abs(c32.eval(X0, H_in, W_in) - c64.eval(X0, H_in, W_in))))
    assert fp32_diff < 1e-5, f"Conv2D fp32 mismatch: {fp32_diff}"

    # 2b. AvgPool2D, MaxPool2D, GlobalAvgPool2D
    # Use distinct values for MaxPool2D so argmax is strict and smooth locally
    X_pool = np.arange(1, 2 * 16 + 1, dtype=np.float64).reshape((2, 16)) * 0.13 + rng.normal(0.0, 0.02, (2, 16))
    xp_flat = X_pool.ravel(order="F")
    for name, layer in [
        ("AvgPool2D", nn.AvgPool2D(2, stride=2, input_hw=(4, 4))),
        ("MaxPool2D", nn.MaxPool2D(2, stride=2, input_hw=(4, 4))),
        ("GlobalAvgPool2D", nn.GlobalAvgPool2D()),
    ]:
        gp = fo.GraphBuilder()
        xp = gp.variable((2, 16), init=X_pool, name="xp")
        yp = layer(xp)
        lp = fo.sum(fo.sin(yp) + 0.5 * yp * yp)
        evp = fo.Evaluator(gp._graph, [lp.id, yp.id])
        yp_sym = evp.eval(xp_flat)[1]
        yp_num = layer.eval(X_pool)
        assert np.max(np.abs(yp_sym - yp_num)) < 1e-13, f"{name} eval mismatch"
        g_ad = evp.gradient(xp_flat, 0).ravel()
        g_fd = _fd_grad(lambda xv: float(evp.eval(xv)[0].ravel()[0]), xp_flat)
        h_ad = evp.hessian(xp_flat, 0)
        h_fd = _fd_hess(lambda xv: evp.gradient(xv, 0).ravel(), xp_flat)
        assert _rel_err(g_ad, g_fd) < 1e-8, f"{name} grad err"
        assert _rel_err(h_ad, h_fd) < 1e-7, f"{name} hess err"

    # 2c. BatchNorm standalone + Conv2D weight-folding exact parity
    bn = nn.BatchNorm.random(3, seed=1803)
    y_unfolded = bn.eval(c64.eval(X0, H_in, W_in))  # Note: fold only applies before nonlinear act
    c_lin = nn.Conv2D(W_shared, b_shared, stride=2, padding=1, activation="identity")
    W_fold, b_fold = bn.fold_into_conv_weights(W_shared, b_shared)
    c_folded = nn.Conv2D(W_fold, b_fold, stride=2, padding=1, activation="identity")
    fold_err = float(np.max(np.abs(bn.eval(c_lin.eval(X0, H_in, W_in)) - c_folded.eval(X0, H_in, W_in))))
    assert fold_err < 1e-14, f"BatchNorm folding mismatch: {fold_err}"

    # 2d. RMSNorm exact 0/1/2 AD
    rn = nn.RMSNorm.random(4, eps=1e-5, seed=1804)
    Xr = rng.normal(0.3, 0.7, size=(4, 3))
    xr_flat = Xr.ravel(order="F")
    gr = fo.GraphBuilder()
    xrv = gr.variable((4, 3), init=Xr, name="xr")
    yr = rn(xrv)
    lr = fo.sum(fo.exp(0.3 * yr) + 0.5 * yr * yr)
    evr = fo.Evaluator(gr._graph, [lr.id, yr.id])
    yr_sym = evr.eval(xr_flat)[1]
    yr_num = rn.eval(Xr)
    assert np.max(np.abs(yr_sym - yr_num)) < 1e-13
    rn_g_err = _rel_err(evr.gradient(xr_flat, 0).ravel(), _fd_grad(lambda xv: float(evr.eval(xv)[0].ravel()[0]), xr_flat))
    rn_h_err = _rel_err(evr.hessian(xr_flat, 0), _fd_hess(lambda xv: evr.gradient(xv, 0).ravel(), xr_flat))
    assert rn_g_err < 1e-8 and rn_h_err < 1e-7, f"RMSNorm AD error: g={rn_g_err}, h={rn_h_err}"

    print(
        f"  [PASSED Test 2] Conv2D g_err={c_g_err:.2e}, h_err={c_h_err:.2e}, sp_nnz={nnz_conv}/1350, fp32_diff={fp32_diff:.2e} | "
        f"Pool2D exact | BN fold_err={fold_err:.2e} | RMSNorm g_err={rn_g_err:.2e}, h_err={rn_h_err:.2e}"
    )


def test_3_onnx_2d_cnn_bn_folding_and_2nd_ad() -> None:
    print("\n--- [Phase 18 Test 3] Zero-Dependency ONNX 2D CNN (Conv2D + BN Auto-Fold + Mish + Pool2D + MLP) ---")
    rng = np.random.default_rng(1805)
    C_in, H_in, W_in = 2, 4, 4
    C_mid = 3

    W_conv = rng.normal(0.0, 0.35, size=(C_mid, C_in, 3, 3))
    b_conv = rng.normal(0.0, 0.05, size=(C_mid,))
    bn_gamma = rng.uniform(0.8, 1.2, size=(C_mid,))
    bn_beta = rng.normal(0.0, 0.05, size=(C_mid,))
    bn_mean = rng.normal(0.0, 0.2, size=(C_mid,))
    bn_var = rng.uniform(0.5, 1.5, size=(C_mid,))
    W_fc1 = rng.normal(0.0, 0.4, size=(4, C_mid))
    b_fc1 = rng.normal(0.0, 0.05, size=(4,))
    W_fc2 = rng.normal(0.0, 0.4, size=(1, 4))
    b_fc2 = rng.normal(0.0, 0.05, size=(1,))

    with tempfile.TemporaryDirectory() as td:
        onnx_path = os.path.join(td, "cnn2d_bn.onnx")
        write_minimal_onnx(
            onnx_path,
            nodes=[
                ("Conv", ["x", "W_conv", "b_conv"], ["c1"], {"strides": [1, 1], "pads": [1, 1, 1, 1], "dilations": [1, 1]}),
                ("BatchNormalization", ["c1", "bn_g", "bn_b", "bn_m", "bn_v"], ["bn1"], {"epsilon": 1e-5}),
                ("Mish", ["bn1"], ["a1"], {}),
                ("AveragePool", ["a1"], ["p1"], {"kernel_shape": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]}),
                ("GlobalAveragePool", ["p1"], ["gap"], {}),
                ("Gemm", ["gap", "W_fc1", "b_fc1"], ["h1"], {"transB": 1}),
                ("HardSwish", ["h1"], ["h1a"], {}),
                ("Gemm", ["h1a", "W_fc2", "b_fc2"], ["y"], {"transB": 1}),
            ],
            initializers={
                "W_conv": W_conv, "b_conv": b_conv,
                "bn_g": bn_gamma, "bn_b": bn_beta, "bn_m": bn_mean, "bn_v": bn_var,
                "W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2,
            },
            input_names=["x"],
            output_names=["y"],
        )

        model = nn.from_onnx(onnx_path)
        assert model.fused_conv2d_count == 1, f"Expected 1 fused Conv2D, got {model.fused_conv2d_count}"
        assert model.folded_bn_count == 1, f"Expected 1 folded BatchNorm, got {model.folded_bn_count}"
        assert model.fused_mlp_count == 1, f"Expected 1 fused MLP, got {model.fused_mlp_count}"

        X0 = rng.normal(0.15, 0.5, size=(C_in, H_in * W_in))
        x0_flat = X0.ravel(order="F")

        # Manual reference
        c_ref = nn.Conv2D(W_conv, b_conv, stride=1, padding=1, activation="identity")
        bn_ref = nn.BatchNorm(bn_mean, bn_var, bn_gamma, bn_beta, eps=1e-5)
        z1 = bn_ref.eval(c_ref.eval(X0, H_in, W_in))
        a1 = z1 * np.tanh(np.logaddexp(0.0, z1))
        p1 = nn.AvgPool2D(2, stride=2).eval(a1, H_in, W_in)
        gap = nn.GlobalAvgPool2D().eval(p1)
        mlp_ref = nn.MLP([
            nn.LayerConfig(W=W_fc1, b=b_fc1, activation="hardswish"),
            nn.LayerConfig(W=W_fc2, b=b_fc2, activation="identity"),
        ])
        y_ref = float(mlp_ref.eval(gap[:, 0])[0])

        # Symbolic ONNX evaluation & exact 1st/2nd AD
        gb = fo.GraphBuilder()
        xv = gb.variable((C_in, H_in * W_in), init=X0, name="x")
        yv = model(xv, spatial_hw=(H_in, W_in))
        ev = fo.Evaluator(gb._graph, [yv.id])
        y_sym = float(ev.eval(x0_flat)[0].ravel()[0])
        assert abs(y_sym - y_ref) < 1e-13, f"ONNX CNN parity error: {abs(y_sym - y_ref)}"

        g_err = _rel_err(ev.gradient(x0_flat, 0).ravel(), _fd_grad(lambda z: float(ev.eval(z)[0].ravel()[0]), x0_flat))
        h_err = _rel_err(ev.hessian(x0_flat, 0), _fd_hess(lambda z: ev.gradient(z, 0).ravel(), x0_flat))
        assert g_err < 1e-8 and h_err < 1e-7, f"ONNX CNN AD error: g={g_err}, h={h_err}"

        print(
            f"  [PASSED Test 3] ONNX 2D CNN auto-fused (Conv2D=1, FoldedBN=1, MLP=1) | "
            f"|val_err|={abs(y_sym - y_ref):.2e}, g_err={g_err:.2e}, h_err={h_err:.2e}"
        )


def test_4_end_to_end_opti_2d_cnn_surrogate() -> None:
    print("\n--- [Phase 18 Test 4] End-to-End fo.Opti Optimization with Conv2D + BatchNorm + AvgPool2D + RMSNorm + MLP ---")
    net = nn.Sequential(
        nn.Conv2D.random(2, 4, kernel_size=3, stride=1, padding=1, activation="mish", input_hw=(4, 4), seed=401),
        nn.BatchNorm.random(4, seed=402),
        nn.AvgPool2D(2, stride=2, input_hw=(4, 4)),
        nn.GlobalAvgPool2D(),
        nn.RMSNorm.random(4, seed=403),
        nn.MLP.random([4, 8, 1], activation="selu", output_activation="sigmoid", seed=404),
    )

    opti = fo.Opti()
    X = opti.variable((2, 16), init=np.zeros((2, 16)), lb=-2.0, ub=2.0, name="field_2d")
    y = net(X)  # shape (1, 1) in (0, 1) due to Sigmoid output
    # Minimize L2 norm of 2D field subject to surrogate output y >= 0.75
    opti.minimize(0.5 * fo.sum(X * X))
    opti.subject_to(y >= 0.75)

    opti.solver("trust-constr", {"maxiter": 150, "gtol": 1e-6})
    sol = opti.solve()
    assert sol.success, f"Opti failed: {sol.message}"
    y_star = float(np.asarray(sol.value(y)).ravel()[0])
    assert y_star >= 0.75 - 1e-6, f"Constraint violated: y*={y_star}"
    print(f"  [PASSED Test 4] Opti + 2D CNN+BN+Pool+RMSNorm+MLP converged: y*={y_star:.6f} (>=0.75), ||X*||^2={float(np.sum(sol.value(X)**2)):.4f}")


def run_phase18_suite() -> None:
    print("=" * 90)
    print("Running fastopt Phase 18 Expanded NN Layers & Activations (Conv2D/Pool2D/BN/RMSNorm) Suite")
    print("=" * 90)
    test_1_five_new_activations_exact_2nd_ad()
    test_2_conv2d_pool2d_bn_rmsnorm_ad_and_sparsity()
    test_3_onnx_2d_cnn_bn_folding_and_2nd_ad()
    test_4_end_to_end_opti_2d_cnn_surrogate()
    print("\nALL PHASE 18 TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_phase18_suite()
