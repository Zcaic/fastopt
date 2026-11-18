"""
Phase 8 Verification & Benchmark Suite:
Direction 5 — Zero-GIL C-ABI Direct Black-Box Bridge (`fastopt/c_abi.h`, `CBlackBox`, `PyCapsule` & `from_dll`)
for `fast_bemt` and `fastgeo` C++ solvers:
  1. Stateful C++ `MockFastBemtSolver` Connected via `PyCapsule` (`FASTOPT_C_BLACKBOX_CAPSULE_NAME`)
     and via `CBlackBox.from_dll(...)` (0th, 1st Analytical IFT Jac, 2nd Directional FD Hessian, & Batch Eval)
  2. Zero-GIL C++ Central Finite-Difference Mode (`jac=NULL`) & Custom Sparsity Mask (`jac_sparsity`)
     for `fastgeo`-Style Spanwise Geometric Constraints (1-Color Graph Coloring)
  3. B-Spline Parameterized Rotor Blade Optimization Coupling C-ABI `fast_bemt` Kernel + `ks_max` Envelope
  4. Performance Benchmark: Zero-GIL C-ABI `CBlackBox` vs. Python Callback `CustomOp`
     (Forward `eval`, `eval_batch`, 1st-Order `gradient`, and OpenMP Parallel 2nd-Order `lagrangian_hessian`)

Note: The mock C-ABI source (`mock_bemt_c_abi.cpp`) and compiled shared library
(`libmock_bemt_c_abi.dll` / `libmock_bemt_c_abi.so`) are placed directly inside `<project_root>/build/`
for easy inspection and debugging.
All comments are written in English.
"""

import ctypes
import os
import subprocess
import sys
import sysconfig
import time
import numpy as np
import python.fastopt as fo


_MOCK_BEMT_CPP_SOURCE = r"""
#include "fastopt/c_abi.h"
#include <Python.h>
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Stateful C++ rotor solver mimicking fast_bemt::Solver
struct MockBemtSolver {
    int n_sec;
    double omega;
    double v_axial;
    double rho;
    double r_hub;
    double r_tip;
    std::vector<double> r_sec;
    std::vector<double> chord_sec;
    double dr;
    FastOptCBlackBox desc;

    MockBemtSolver(int n, double om, double va, double air_rho)
        : n_sec(n), omega(om), v_axial(va), rho(air_rho), r_hub(0.25), r_tip(1.05) {
        r_sec.resize(n_sec);
        chord_sec.resize(n_sec);
        dr = (r_tip - r_hub) / static_cast<double>(n_sec - 1);
        for (int i = 0; i < n_sec; ++i) {
            r_sec[i] = r_hub + i * dr;
            chord_sec[i] = 0.12 * (1.0 - 0.25 * (r_sec[i] / r_tip));
        }
    }
};

extern "C" {

// Evaluate [Thrust, Power, CL_0, ..., CL_{n_sec-1}] from pitch angles u = theta_rad[0..n_sec-1]
int bemt_eval(const double* u, int32_t n_in, double* y, int32_t m_out, void* user_data) {
    const auto* s = static_cast<const MockBemtSolver*>(user_data);
    if (!s || n_in != s->n_sec || m_out != (2 + s->n_sec)) return -1;

    double total_T = 0.0;
    double total_P = 0.0;
    for (int i = 0; i < s->n_sec; ++i) {
        const double r = s->r_sec[i];
        const double vt = s->omega * r;
        const double va = s->v_axial * (1.0 + 0.15 * (r / s->r_tip));
        const double v2 = va * va + vt * vt;
        const double phi = std::atan2(va, vt);
        const double alpha = u[i] - phi;
        const double q_c_dr = 0.5 * s->rho * v2 * s->chord_sec[i] * s->dr * 3.0; // 3 blades

        // Nonlinear compressible polar with mild cubic stall nonlinearity
        const double cl = 5.7 * alpha - 4.2 * (alpha * alpha * alpha);
        const double cd = 0.0085 + 0.045 * alpha * alpha + 0.12 * (alpha * alpha * alpha * alpha);

        const double c_phi = std::cos(phi);
        const double s_phi = std::sin(phi);
        const double dT = q_c_dr * (cl * c_phi - cd * s_phi);
        const double dP = q_c_dr * (cl * s_phi + cd * c_phi) * vt;

        total_T += dT;
        total_P += dP;
        y[2 + i] = cl;
    }
    y[0] = total_T;
    y[1] = total_P;
    return 0;
}

// Analytical 1st-order Jacobian J (m_out x n_in) in column-major layout: J[row + col * m_out]
int bemt_jac(const double* u, int32_t n_in, double* J_col_major, int32_t m_out, void* user_data) {
    const auto* s = static_cast<const MockBemtSolver*>(user_data);
    if (!s || n_in != s->n_sec || m_out != (2 + s->n_sec)) return -1;

    for (int32_t idx = 0; idx < m_out * n_in; ++idx) {
        J_col_major[idx] = 0.0;
    }

    for (int i = 0; i < s->n_sec; ++i) {
        const double r = s->r_sec[i];
        const double vt = s->omega * r;
        const double va = s->v_axial * (1.0 + 0.15 * (r / s->r_tip));
        const double v2 = va * va + vt * vt;
        const double phi = std::atan2(va, vt);
        const double alpha = u[i] - phi;
        const double q_c_dr = 0.5 * s->rho * v2 * s->chord_sec[i] * s->dr * 3.0;

        const double dcl_dtheta = 5.7 - 12.6 * (alpha * alpha);
        const double dcd_dtheta = 0.090 * alpha + 0.48 * (alpha * alpha * alpha);

        const double c_phi = std::cos(phi);
        const double s_phi = std::sin(phi);
        const double ddT_dtheta = q_c_dr * (dcl_dtheta * c_phi - dcd_dtheta * s_phi);
        const double ddP_dtheta = q_c_dr * (dcl_dtheta * s_phi + dcd_dtheta * c_phi) * vt;

        J_col_major[0 + i * m_out] = ddT_dtheta;
        J_col_major[1 + i * m_out] = ddP_dtheta;
        J_col_major[(2 + i) + i * m_out] = dcl_dtheta;
    }
    return 0;
}

// OpenMP parallel population batch evaluation: U_col_major (n_pop x n_in) -> Y_col_major (n_pop x m_out)
int bemt_eval_batch(const double* U_col_major, int32_t n_pop, int32_t n_in,
                    double* Y_col_major, int32_t m_out, void* user_data) {
    const auto* s = static_cast<const MockBemtSolver*>(user_data);
    if (!s || n_in != s->n_sec || m_out != (2 + s->n_sec)) return -1;

    #pragma omp parallel for schedule(static)
    for (int32_t p = 0; p < n_pop; ++p) {
        std::vector<double> u_loc(n_in);
        std::vector<double> y_loc(m_out);
        for (int32_t j = 0; j < n_in; ++j) {
            u_loc[j] = U_col_major[p + j * n_pop];
        }
        bemt_eval(u_loc.data(), n_in, y_loc.data(), m_out, user_data);
        for (int32_t k = 0; k < m_out; ++k) {
            Y_col_major[p + k * n_pop] = y_loc[k];
        }
    }
    return 0;
}

// Spanwise local geometric thickness constraint mimicking fastgeo (1-to-1 diagonal dependency)
int fastgeo_thickness_eval(const double* u, int32_t n_in, double* y, int32_t m_out, void* /*user_data*/) {
    if (n_in != m_out) return -1;
    for (int32_t i = 0; i < n_in; ++i) {
        y[i] = 0.14 * std::sin(1.5 * u[i]) + 0.05 * u[i] * u[i];
    }
    return 0;
}

void* create_mock_bemt_solver(int n_sec, double omega, double v_axial, double rho) {
    auto* s = new MockBemtSolver(n_sec, omega, v_axial, rho);
    s->desc.abi_version = FASTOPT_C_ABI_VERSION;
    s->desc.n_in = n_sec;
    s->desc.m_out = 2 + n_sec;
    s->desc.is_thread_safe = 1;
    s->desc.user_data = s;
    s->desc.eval = &bemt_eval;
    s->desc.jac = &bemt_jac;
    s->desc.hvp = nullptr; // Let fastopt compute directional FD over analytical `bemt_jac` in C++!
    s->desc.eval_batch = &bemt_eval_batch;
    s->desc.destroy = nullptr;
    return s;
}

void destroy_mock_bemt_solver(void* ptr) {
    delete static_cast<MockBemtSolver*>(ptr);
}

// Return a Python PyCapsule wrapping `&s->desc` (exactly as a nanobind method in fast_bemt would)
PyObject* create_mock_bemt_capsule(void* ptr) {
    auto* s = static_cast<MockBemtSolver*>(ptr);
    return PyCapsule_New(&s->desc, FASTOPT_C_BLACKBOX_CAPSULE_NAME, nullptr);
}

} // extern "C"
"""


def _compile_mock_bemt_shared_lib() -> str:
    """
    Compile the C++17 mock fast_bemt/fastgeo shared library directly into `<project_root>/build/`
    with static MinGW runtime linking on Windows so it has zero external DLL dependencies.
    """
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(proj_root, "build")
    os.makedirs(build_dir, exist_ok=True)

    inc_dir = os.path.join(proj_root, "include")
    py_inc = sysconfig.get_path("include")

    cpp_path = os.path.join(build_dir, "mock_bemt_c_abi.cpp")
    ext = ".dll" if os.name == "nt" else ".so"
    lib_path = os.path.abspath(os.path.join(build_dir, f"libmock_bemt_c_abi{ext}"))

    with open(cpp_path, "w", encoding="utf-8") as f:
        f.write(_MOCK_BEMT_CPP_SOURCE)

    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-fopenmp",
        f"-I{inc_dir}",
        f"-I{py_inc}",
        cpp_path,
        "-o",
        lib_path,
    ]
    if os.name == "nt":
        # Link against python3x.dll on Windows and statically link MinGW C++/OpenMP/winpthread
        # runtimes (matching CMakeLists.txt) so ctypes.CDLL never fails on missing MinGW DLLs.
        py_lib_dir = os.path.join(sys.base_prefix, "libs")
        py_ver = f"python{sys.version_info.major}{sys.version_info.minor}"
        if os.path.isdir(py_lib_dir):
            cmd.extend([f"-L{py_lib_dir}", f"-l{py_ver}"])
        cmd.extend([
            "-static-libgcc",
            "-static-libstdc++",
            "-Wl,-Bstatic",
            "-lstdc++",
            "-lgomp",
            "-lwinpthread",
            "-lgcc_eh",
            "-lgcc",
            "-Wl,-Bdynamic",
        ])

    subprocess.run(cmd, check=True)
    return lib_path


def fd_gradient(ev: fo._fastopt_impl.Evaluator, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = x0.size
    g = np.zeros(n, dtype=np.float64)
    for j in range(n):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        fp = float(ev.eval(xp)[0].item())
        fm = float(ev.eval(xm)[0].item())
        g[j] = (fp - fm) / (2.0 * eps)
    return g


def fd_hessian(ev: fo._fastopt_impl.Evaluator, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = x0.size
    H = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        gp = ev.gradient(xp, 0)
        gm = ev.gradient(xm, 0)
        H[:, j] = (gp - gm) / (2.0 * eps)
    return 0.5 * (H + H.T)


def run_phase8_suite():
    lib_path = _compile_mock_bemt_shared_lib()
    print(f"[Build] Compiled C-ABI shared library: {lib_path}")
    lib = ctypes.CDLL(lib_path)

    lib.create_mock_bemt_solver.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.create_mock_bemt_solver.restype = ctypes.c_void_p
    lib.destroy_mock_bemt_solver.argtypes = [ctypes.c_void_p]
    lib.destroy_mock_bemt_solver.restype = None

    py_dll = ctypes.PyDLL(lib_path)
    py_dll.create_mock_bemt_capsule.argtypes = [ctypes.c_void_p]
    py_dll.create_mock_bemt_capsule.restype = ctypes.py_object

    n_sec = 12
    solver_ptr = lib.create_mock_bemt_solver(n_sec, 125.0, 22.0, 1.225)
    try:
        # =========================================================================
        # Test 1: PyCapsule C-ABI Direct Connection & From-DLL Connection (1st/2nd AD)
        # =========================================================================
        capsule = py_dll.create_mock_bemt_capsule(solver_ptr)
        bemt_cap_op = fo.CBlackBox.from_capsule(capsule, shape=(2 + n_sec, 1), name="bemt_capsule")
        bemt_dll_op = fo.CBlackBox.from_dll(
            lib_path,
            eval_symbol="bemt_eval",
            jac_symbol="bemt_jac",
            batch_symbol="bemt_eval_batch",
            user_data_ptr=solver_ptr,
            shape=(2 + n_sec, 1),
            is_thread_safe=True,
            name="bemt_dll",
        )

        gb = fo.GraphBuilder()
        theta0 = np.linspace(0.28, 0.12, n_sec)
        theta = gb.variable(shape=(n_sec, 1), init=theta0, name="theta")

        out_cap = bemt_cap_op(theta)
        out_dll = bemt_dll_op(theta)
        obj = 0.001 * out_cap[1] - 0.01 * out_cap[0] + fo.sum(fo.square(out_dll[2 : 2 + n_sec]))

        ev = gb.build_evaluator([obj, out_cap, out_dll])
        res = ev.eval(theta0)
        assert np.allclose(res[1], res[2], atol=1e-14), "PyCapsule and from_dll outputs must match identically"

        g_ad = ev.gradient(theta0, 0)
        g_fd = fd_gradient(ev, theta0)
        assert np.max(np.abs(g_ad - g_fd)) < 1e-6, f"C-ABI Grad error: {np.max(np.abs(g_ad - g_fd))}"

        H_ad = ev.hessian(theta0, 0, num_threads=2)
        H_fd = fd_hessian(ev, theta0)
        assert np.max(np.abs(H_ad - H_fd)) < 1e-4, f"C-ABI Hess error: {np.max(np.abs(H_ad - H_fd))}"

        # Test population batch evaluation via C-ABI `bemt_eval_batch`
        X_pop = np.vstack([theta0 + 0.01 * k for k in range(16)])
        Y_batch = ev.eval_batch(X_pop)
        for p in range(16):
            single = ev.eval(X_pop[p])[1].ravel()
            assert np.allclose(Y_batch[1][p], single, atol=1e-13)

        print(
            "[PASSED] 1. Zero-GIL PyCapsule & from_dll C-ABI Bridge "
            "(0th Eval, Batch OpenMP Eval, 1st Analytical IFT Jac & 2nd Directional FD Hessian)"
        )

        # =========================================================================
        # Test 2: Zero-GIL C++ Central FD (jac=NULL) + Custom Sparsity Mask (`jac_sparsity`)
        # =========================================================================
        diag_mask = np.eye(n_sec, dtype=np.int32)
        geo_c_op = fo.CBlackBox.from_dll(
            lib_path,
            eval_symbol="fastgeo_thickness_eval",
            shape=(n_sec, 1),
            jac_sparsity=diag_mask,
            is_thread_safe=True,
            name="fastgeo_thickness",
        )
        gb_geo = fo.GraphBuilder()
        u_geo = gb_geo.variable(shape=(n_sec, 1), init=theta0)
        t_out = geo_c_op(u_geo)
        ev_geo = gb_geo.build_evaluator([fo.sum(fo.square(t_out)), t_out])

        assert ev_geo.num_jacobian_colors(1) == 1
        assert ev_geo.num_hessian_colors() == 1
        assert len(ev_geo.jacobian_sparsity(1)[0]) == n_sec
        assert len(ev_geo.hessian_sparsity()[0]) == n_sec

        g_geo_ad = ev_geo.gradient(theta0, 0)
        g_geo_fd = fd_gradient(ev_geo, theta0)
        assert np.max(np.abs(g_geo_ad - g_geo_fd)) < 1e-7
        print(
            f"[PASSED] 2. Zero-GIL C++ Central FD (`jac=NULL`) & Custom Sparsity Mask "
            f"(N={n_sec}: Jac colors=1, Hess colors=1)"
        )

        # =========================================================================
        # Test 3: B-Spline Parameterized Rotor Optimization with C-ABI `fast_bemt` + `ks_max`
        # =========================================================================
        opti = fo.Opti()
        n_ctrl = 5
        ctrl_init = np.linspace(0.32, 0.14, n_ctrl)
        theta_ctrl = opti.variable(shape=(n_ctrl, 1), init=ctrl_init, lb=0.02, ub=0.55, name="theta_ctrl")

        r_norm = np.linspace(0.0, 1.0, n_sec)
        ctrl_grid = np.linspace(0.0, 1.0, n_ctrl)
        theta_span = fo.interp_bspline(r_norm, ctrl_grid, theta_ctrl, degree=3)

        bemt_out = bemt_cap_op(theta_span)
        thrust = bemt_out[0]
        power = bemt_out[1]
        cl_vec = bemt_out[2 : 2 + n_sec]

        opti.minimize(power * 1e-3)
        c_t = opti.subject_to(thrust >= 380.0, scale=100.0, name="thrust_req")
        opti.subject_to(fo.ks_max(cl_vec, rho=60.0, conservative=True) <= 0.92, name="stall_margin")

        opti.solver("trust-constr", options={"gtol": 1e-9, "xtol": 1e-10, "maxiter": 200}, auto_scale=True)
        sol = opti.solve()
        assert sol.success, f"Optimization with C-ABI BEMT failed: {sol.message}"
        t_opt = float(sol.value(thrust))
        cl_max_opt = float(np.max(sol.value(cl_vec)))
        assert t_opt >= 380.0 - 1e-4 and cl_max_opt <= 0.92 + 1e-4
        print(
            f"[PASSED] 3. B-Spline Rotor Blade Optimization with Zero-GIL C-ABI `fast_bemt` "
            f"(T={t_opt:.2f} N, max(CL)={cl_max_opt:.4f} <= 0.92, dual_T={sol.dual(c_t):.4f})"
        )

        # =========================================================================
        # Test 4: Performance Benchmark — Zero-GIL C-ABI vs. Python Callback CustomOp
        # =========================================================================
        r_sec = np.linspace(0.25, 1.05, n_sec)
        dr = (1.05 - 0.25) / (n_sec - 1)
        chord_sec = 0.12 * (1.0 - 0.25 * (r_sec / 1.05))
        vt_sec = 125.0 * r_sec
        va_sec = 22.0 * (1.0 + 0.15 * (r_sec / 1.05))
        v2_sec = va_sec**2 + vt_sec**2
        phi_sec = np.arctan2(va_sec, vt_sec)
        q_c_dr_sec = 0.5 * 1.225 * v2_sec * chord_sec * dr * 3.0
        c_phi_sec = np.cos(phi_sec)
        s_phi_sec = np.sin(phi_sec)

        def py_bemt_eval(u):
            alpha = u - phi_sec
            cl = 5.7 * alpha - 4.2 * (alpha**3)
            cd = 0.0085 + 0.045 * (alpha**2) + 0.12 * (alpha**4)
            dT = q_c_dr_sec * (cl * c_phi_sec - cd * s_phi_sec)
            dP = q_c_dr_sec * (cl * s_phi_sec + cd * c_phi_sec) * vt_sec
            return np.concatenate([[np.sum(dT), np.sum(dP)], cl])

        def py_bemt_jac(u):
            alpha = u - phi_sec
            dcl = 5.7 - 12.6 * (alpha**2)
            dcd = 0.090 * alpha + 0.48 * (alpha**3)
            ddT = q_c_dr_sec * (dcl * c_phi_sec - dcd * s_phi_sec)
            ddP = q_c_dr_sec * (dcl * s_phi_sec + dcd * c_phi_sec) * vt_sec
            J = np.zeros((2 + n_sec, n_sec), dtype=np.float64)
            J[0, :] = ddT
            J[1, :] = ddP
            J[2:, :] = np.diag(dcl)
            return J

        py_bb_op = fo.CustomOp(forward_fn=py_bemt_eval, jacobian_fn=py_bemt_jac, shape=(2 + n_sec, 1))

        gb_c = fo.GraphBuilder()
        th_c = gb_c.variable(shape=(n_sec, 1), init=theta0)
        y_c = bemt_cap_op(th_c)
        ev_c = gb_c.build_evaluator([0.001 * y_c[1] + fo.sum(fo.square(y_c[2 : 2 + n_sec])), y_c])

        gb_py = fo.GraphBuilder()
        th_py = gb_py.variable(shape=(n_sec, 1), init=theta0)
        y_py = py_bb_op(th_py)
        ev_py = gb_py.build_evaluator([0.001 * y_py[1] + fo.sum(fo.square(y_py[2 : 2 + n_sec])), y_py])

        n_rep = 200
        t0 = time.perf_counter()
        for _ in range(n_rep):
            ev_py.eval(theta0)
        t_py_eval = (time.perf_counter() - t0) * 1000.0 / n_rep

        t0 = time.perf_counter()
        for _ in range(n_rep):
            ev_c.eval(theta0)
        t_c_eval = (time.perf_counter() - t0) * 1000.0 / n_rep

        t0 = time.perf_counter()
        for _ in range(n_rep):
            ev_py.gradient(theta0, 0)
        t_py_grad = (time.perf_counter() - t0) * 1000.0 / n_rep

        t0 = time.perf_counter()
        for _ in range(n_rep):
            ev_c.gradient(theta0, 0)
        t_c_grad = (time.perf_counter() - t0) * 1000.0 / n_rep

        t0 = time.perf_counter()
        for _ in range(50):
            ev_py.hessian(theta0, 0)
        t_py_hess = (time.perf_counter() - t0) * 1000.0 / 50

        t0 = time.perf_counter()
        for _ in range(50):
            ev_c.hessian(theta0, 0)
        t_c_hess = (time.perf_counter() - t0) * 1000.0 / 50

        X_pop_200 = np.tile(theta0, (200, 1))
        t0 = time.perf_counter()
        for _ in range(20):
            ev_py.eval_batch(X_pop_200)
        t_py_batch = (time.perf_counter() - t0) * 1000.0 / 20

        t0 = time.perf_counter()
        for _ in range(20):
            ev_c.eval_batch(X_pop_200)
        t_c_batch = (time.perf_counter() - t0) * 1000.0 / 20

        print(
            f"[PASSED] 4. Benchmark — Zero-GIL C-ABI `CBlackBox` vs. Python `@blackbox`:\n"
            f"         - Forward eval():       Python={t_py_eval*1000:.2f} us vs C-ABI={t_c_eval*1000:.2f} us "
            f"({t_py_eval/max(t_c_eval, 1e-9):.1f}x speedup)\n"
            f"         - 1st-Order gradient(): Python={t_py_grad*1000:.2f} us vs C-ABI={t_c_grad*1000:.2f} us "
            f"({t_py_grad/max(t_c_grad, 1e-9):.1f}x speedup)\n"
            f"         - 2nd-Order hessian():  Python={t_py_hess*1000:.2f} us vs C-ABI={t_c_hess*1000:.2f} us "
            f"({t_py_hess/max(t_c_hess, 1e-9):.1f}x speedup)\n"
            f"         - Batch eval (P=200):   Python={t_py_batch:.3f} ms vs C-ABI={t_c_batch:.3f} ms "
            f"({t_py_batch/max(t_c_batch, 1e-9):.1f}x speedup)"
        )
    finally:
        lib.destroy_mock_bemt_solver(solver_ptr)


if __name__ == "__main__":
    print("=" * 85)
    print("Running fastopt Phase 8 Suite (Zero-GIL C-ABI BlackBox Bridge for fast_bemt/fastgeo)")
    print("=" * 85)
    run_phase8_suite()
    print("=" * 85)
    print("ALL PHASE 8 TESTS PASSED SUCCESSFULLY!")
    print("=" * 85)
