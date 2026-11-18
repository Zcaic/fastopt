"""
Phase 4 verification test suite for fastopt:
1. Custom Black-Box Function Nodes (`CustomOp` subclass & `@fo.blackbox` decorator):
   - Mode A: Analytical 1st-order Jacobian + Analytical 2nd-order Hessian/HVP (returning ONLY active variable derivatives, omitting parameter derivatives!)
   - Mode B: Analytical 1st-order Jacobian (with `(J_x, None)` parameter block) + Automatic directional FD 2nd-order HVP
   - Mode C: Pure 0th-order forward black-box + Automatic central FD 1st-order Jacobian & 2nd-order HVP (automatically skipping `freeze=True` / `parameter` inputs)
2. Constrained NLP optimization with embedded CustomOp black-box nodes (SLSQP & trust-constr).
3. Single-objective constrained evolutionary optimization via `pymoo` (`DE`, `GA`) + hybrid DE -> SLSQP warm-start.
4. Multi-objective constrained Pareto front optimization via `pymoo` (`NSGA2`) + `sol.pareto_x`, `sol.pareto_f`, and `sol.value_batch`.
All comments are written in English.
"""

import sys
import time
import types
import numpy as np
import python.fastopt as fo


def _ensure_pymoo_available_for_test() -> None:
    """
    If `pymoo` is installed in the current environment, use the real `pymoo` package.
    Otherwise, register a lightweight evolutionary solver shim matching the `pymoo` API
    so the test suite can also run in offline sandbox environments without `pymoo`.
    """
    try:
        import pymoo  # noqa: F401
        return
    except ImportError:
        pass

    pymoo_mod = types.ModuleType("pymoo")
    core_mod = types.ModuleType("pymoo.core")
    prob_mod = types.ModuleType("pymoo.core.problem")
    opt_mod = types.ModuleType("pymoo.optimize")
    alg_mod = types.ModuleType("pymoo.algorithms")
    soo_mod = types.ModuleType("pymoo.algorithms.soo")
    nonconvex_mod = types.ModuleType("pymoo.algorithms.soo.nonconvex")
    de_mod = types.ModuleType("pymoo.algorithms.soo.nonconvex.de")
    ga_mod = types.ModuleType("pymoo.algorithms.soo.nonconvex.ga")
    pso_mod = types.ModuleType("pymoo.algorithms.soo.nonconvex.pso")
    moo_mod = types.ModuleType("pymoo.algorithms.moo")
    nsga2_mod = types.ModuleType("pymoo.algorithms.moo.nsga2")

    class Problem:
        def __init__(self, n_var, n_obj=1, n_ieq_constr=0, n_eq_constr=0, xl=None, xu=None, **kwargs):
            self.n_var = int(n_var)
            self.n_obj = int(n_obj)
            self.n_ieq_constr = int(n_ieq_constr)
            self.n_eq_constr = int(n_eq_constr)
            self.xl = np.asarray(xl, dtype=np.float64)
            self.xu = np.asarray(xu, dtype=np.float64)

        def evaluate(self, X, return_values_of="auto", **kwargs):
            out = {}
            self._evaluate(X, out)
            if return_values_of == "auto":
                if self.n_ieq_constr > 0 or self.n_eq_constr > 0:
                    return out["F"], out.get("G", None)
                return out["F"]
            if isinstance(return_values_of, (list, tuple)):
                vals = [out.get(k, None) for k in return_values_of]
                return vals[0] if len(vals) == 1 else tuple(vals)
            return out.get(return_values_of, None)

    class _BaseAlg:
        def __init__(self, name, pop_size=100, **kwargs):
            self.name = name
            self.pop_size = int(pop_size)
            self.kwargs = kwargs
            self.n_gen = 0

    class DE(_BaseAlg):
        def __init__(self, pop_size=100, **kwargs):
            super().__init__("DE", pop_size=pop_size, **kwargs)

    class GA(_BaseAlg):
        def __init__(self, pop_size=100, **kwargs):
            super().__init__("GA", pop_size=pop_size, **kwargs)

    class PSO(_BaseAlg):
        def __init__(self, pop_size=100, **kwargs):
            super().__init__("PSO", pop_size=pop_size, **kwargs)

    class NSGA2(_BaseAlg):
        def __init__(self, pop_size=100, **kwargs):
            super().__init__("NSGA2", pop_size=pop_size, **kwargs)

    class _Result:
        def __init__(self, X, F, alg):
            self.X = X
            self.F = F
            self.algorithm = alg
            self.pop = None

    def minimize(problem, algorithm, termination=None, seed=42, verbose=False, **kwargs):
        rng = np.random.default_rng(seed if seed is not None else 42)
        n_gen = 100
        if isinstance(termination, tuple) and len(termination) == 2:
            n_gen = int(termination[1])
        pop_size = algorithm.pop_size
        n_var = problem.n_var
        xl, xu = problem.xl, problem.xu

        X = xl + rng.random((pop_size, n_var)) * (xu - xl)
        out = {}
        problem._evaluate(X, out)
        F = out["F"]
        G = out.get("G", np.zeros((pop_size, 0)))
        CV = np.sum(np.maximum(0.0, G), axis=1) if G.size > 0 else np.zeros(pop_size)

        for gen in range(n_gen):
            idx_a = rng.integers(0, pop_size, size=pop_size)
            idx_b = rng.integers(0, pop_size, size=pop_size)
            idx_c = rng.integers(0, pop_size, size=pop_size)
            if problem.n_obj == 1:
                best_idx = int(np.lexsort((F[:, 0], CV))[0])
                mutant = X[best_idx] + 0.6 * (X[idx_a] - X[idx_b]) + 0.3 * (X[idx_c] - X)
            else:
                mutant = X[idx_a] + 0.5 * (X[idx_b] - X[idx_c])
            mask = rng.random((pop_size, n_var)) < 0.8
            trial = np.clip(np.where(mask, mutant, X), xl, xu)

            out_t = {}
            problem._evaluate(trial, out_t)
            F_t = out_t["F"]
            G_t = out_t.get("G", np.zeros((pop_size, 0)))
            CV_t = np.sum(np.maximum(0.0, G_t), axis=1) if G_t.size > 0 else np.zeros(pop_size)

            for i in range(pop_size):
                if CV_t[i] < CV[i] - 1e-12:
                    X[i], F[i], CV[i] = trial[i], F_t[i], CV_t[i]
                elif abs(CV_t[i] - CV[i]) <= 1e-12:
                    if problem.n_obj == 1 and F_t[i, 0] <= F[i, 0]:
                        X[i], F[i], CV[i] = trial[i], F_t[i], CV_t[i]
                    elif problem.n_obj > 1 and np.all(F_t[i] <= F[i]) and np.any(F_t[i] < F[i]):
                        X[i], F[i], CV[i] = trial[i], F_t[i], CV_t[i]

        algorithm.n_gen = n_gen
        if problem.n_obj == 1:
            best_idx = int(np.lexsort((F[:, 0], CV))[0])
            return _Result(X[best_idx], F[best_idx], algorithm)

        feas_mask = CV <= 1e-6
        X_cand = X[feas_mask] if np.any(feas_mask) else X
        F_cand = F[feas_mask] if np.any(feas_mask) else F
        nd_mask = np.ones(len(X_cand), dtype=bool)
        for i in range(len(X_cand)):
            for j in range(len(X_cand)):
                if i != j and np.all(F_cand[j] <= F_cand[i]) and np.any(F_cand[j] < F_cand[i]):
                    nd_mask[i] = False
                    break
        return _Result(X_cand[nd_mask], F_cand[nd_mask], algorithm)

    prob_mod.Problem = Problem
    de_mod.DE = DE
    ga_mod.GA = GA
    pso_mod.PSO = PSO
    nsga2_mod.NSGA2 = NSGA2
    opt_mod.minimize = minimize

    sys.modules["pymoo"] = pymoo_mod
    sys.modules["pymoo.core"] = core_mod
    sys.modules["pymoo.core.problem"] = prob_mod
    sys.modules["pymoo.optimize"] = opt_mod
    sys.modules["pymoo.algorithms"] = alg_mod
    sys.modules["pymoo.algorithms.soo"] = soo_mod
    sys.modules["pymoo.algorithms.soo.nonconvex"] = nonconvex_mod
    sys.modules["pymoo.algorithms.soo.nonconvex.de"] = de_mod
    sys.modules["pymoo.algorithms.soo.nonconvex.ga"] = ga_mod
    sys.modules["pymoo.algorithms.soo.nonconvex.pso"] = pso_mod
    sys.modules["pymoo.algorithms.moo"] = moo_mod
    sys.modules["pymoo.algorithms.moo.nsga2"] = nsga2_mod


# ============================================================================
# 1. Custom Black-Box Node (Analytical & Finite-Difference 1st/2nd Order AD)
# ============================================================================
class NonlinearAeroBlackBox(fo.CustomOp):
    """
    Test black-box operator mapping (x in R^2 [active variable], p in R^1 [frozen parameter]) -> y in R^2:
      y0 = x0^2 * sin(x1) + p0 * x0 * x1
      y1 = exp(0.5 * x0) + x1^3 * p0
    Demonstrates that `jacobian` and `hessian` only need to return derivatives w.r.t.
    the active variable `x` (2 columns / 2x2 matrix) without computing derivatives for `p`!
    """

    def __init__(self, provide_jac: bool = True, provide_hess: bool = True) -> None:
        super().__init__(shape=(2, 1), fd_step=1e-7, fd_hessian=True, name="aero_bb")
        self._provide_jac = provide_jac
        self._provide_hess = provide_hess

    def forward(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        x0, x1 = float(x[0]), float(x[1])
        p0 = float(np.asarray(p).ravel()[0])
        y0 = (x0 ** 2) * np.sin(x1) + p0 * x0 * x1
        y1 = np.exp(0.5 * x0) + (x1 ** 3) * p0
        return np.array([y0, y1], dtype=np.float64)

    def jacobian(self, x: np.ndarray, p: np.ndarray):
        x0, x1 = float(x[0]), float(x[1])
        p0 = float(np.asarray(p).ravel()[0])
        # Return ONLY the 2x2 Jacobian w.r.t. active variable x (p is a frozen parameter!)
        J_x = np.array(
            [
                [2.0 * x0 * np.sin(x1) + p0 * x1, (x0 ** 2) * np.cos(x1) + p0 * x0],
                [0.5 * np.exp(0.5 * x0), 3.0 * (x1 ** 2) * p0],
            ],
            dtype=np.float64,
        )
        return J_x

    def hessian(self, x: np.ndarray, p: np.ndarray, y_bar: np.ndarray) -> np.ndarray:
        x0, x1 = float(x[0]), float(x[1])
        p0 = float(np.asarray(p).ravel()[0])
        yb0, yb1 = float(y_bar[0]), float(y_bar[1])

        # Return ONLY the 2x2 Hessian w.r.t. active variable x!
        H0_x = np.array(
            [
                [2.0 * np.sin(x1), 2.0 * x0 * np.cos(x1) + p0],
                [2.0 * x0 * np.cos(x1) + p0, -(x0 ** 2) * np.sin(x1)],
            ],
            dtype=np.float64,
        )
        H1_x = np.array(
            [
                [0.25 * np.exp(0.5 * x0), 0.0],
                [0.0, 6.0 * x1 * p0],
            ],
            dtype=np.float64,
        )
        return yb0 * H0_x + yb1 * H1_x

    def _has_custom_jacobian(self) -> bool:
        return self._provide_jac

    def _has_custom_hessian(self) -> bool:
        return self._provide_hess


def test_blackbox_analytical_and_fd_derivatives():
    # Reference symbolic graph
    gb_ref = fo.GraphBuilder()
    x_ref = gb_ref.variable(shape=(2, 1), init=[1.2, 0.7])
    p_ref = gb_ref.parameter(shape=(1, 1), value=[1.5])
    y0_ref = (x_ref[0] ** 2) * fo.sin(x_ref[1]) + p_ref[0] * x_ref[0] * x_ref[1]
    y1_ref = fo.exp(0.5 * x_ref[0]) + (x_ref[1] ** 3) * p_ref[0]
    y_ref = fo.concat([y0_ref, y1_ref], axis=0)
    loss_ref = (y_ref[0] ** 2) + 2.5 * y_ref[1]
    ev_ref = gb_ref.build_evaluator([loss_ref, y_ref])

    x_test = np.array([1.2, 0.7], dtype=np.float64)
    val_ref = ev_ref.eval(x_test)
    grad_ref = ev_ref.gradient(x_test, 0)
    jac_ref = ev_ref.jacobian(x_test, 1)
    hess_ref = ev_ref.hessian(x_test, 0)

    # Test Mode A (Analytical Jac + Analytical Hess), Mode B (Analytical Jac + FD Hess), Mode C (Pure FD)
    for provide_jac, provide_hess, rtol_hess in [
        (True, True, 1e-12),
        (True, False, 1e-6),
        (False, False, 1e-4),
    ]:
        gb = fo.GraphBuilder()
        x = gb.variable(shape=(2, 1), init=[1.2, 0.7])
        p = gb.parameter(shape=(1, 1), value=[1.5])

        bb_op = NonlinearAeroBlackBox(provide_jac=provide_jac, provide_hess=provide_hess)
        num_out = bb_op(x_test, np.array([1.5]))
        np.testing.assert_allclose(num_out.ravel(), val_ref[1].ravel(), rtol=1e-12)

        y = bb_op(x, p)
        loss = (y[0] ** 2) + 2.5 * y[1]
        ev = gb.build_evaluator([loss, y])

        val = ev.eval(x_test)
        grad = ev.gradient(x_test, 0)
        jac = ev.jacobian(x_test, 1)
        hess = ev.hessian(x_test, 0)

        np.testing.assert_allclose(val[0], val_ref[0], rtol=1e-12)
        np.testing.assert_allclose(val[1], val_ref[1], rtol=1e-12)
        np.testing.assert_allclose(grad, grad_ref, rtol=1e-6 if not provide_jac else 1e-12)
        np.testing.assert_allclose(jac, jac_ref, rtol=1e-6 if not provide_jac else 1e-12)
        np.testing.assert_allclose(hess, hess_ref, rtol=rtol_hess, atol=1e-5)


# ============================================================================
# 2. Constrained Optimization with @fo.blackbox Decorator (SLSQP & trust-constr)
# ============================================================================
def test_blackbox_decorator_in_opti_optimization():
    @fo.blackbox(
        shape=(1, 1),
        grad_inputs=[0],
        jac=lambda x, p_offset: np.array([[2.0 * (x[0] - p_offset[0]), 4.0 * (x[1] + 0.5)]], dtype=np.float64),
    )
    def blackbox_drag(x: np.ndarray, p_offset: np.ndarray) -> np.ndarray:
        return np.array([(x[0] - p_offset[0]) ** 2 + 2.0 * ((x[1] + 0.5) ** 2)], dtype=np.float64)

    for method in ["SLSQP", "trust-constr"]:
        opti = fo.Opti()
        x = opti.variable(shape=(2, 1), init=[0.0, 0.0], lb=-5.0, ub=5.0)
        p_offset = opti.parameter(shape=(1, 1), value=[1.5])
        drag = blackbox_drag(x, p_offset)

        opti.minimize(drag + 0.5 * (x[0] ** 2))
        opti.subject_to(x[0] + x[1] == 1.0)
        opti.solver(method)
        sol = opti.solve()

        assert sol.success, f"{method} with @blackbox failed: {sol.message}"
        np.testing.assert_allclose(sol.value(x), [9.0 / 7.0, -2.0 / 7.0], atol=1e-5)


# ============================================================================
# 3. Single-Objective Constrained Evolutionary Optimization via pymoo (DE -> SLSQP)
# ============================================================================
def test_pymoo_single_objective_and_hybrid_warmstart():
    _ensure_pymoo_available_for_test()

    opti = fo.Opti()
    x = opti.variable(shape=(2, 1), init=[3.0, -3.0], lb=-5.0, ub=5.0)

    # Objective with local ripples + global minimum near (1.0, 1.0) subject to x0 + x1 >= 2.0
    obj = (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2 + 0.05 * (1.0 - fo.cos(2.0 * np.pi * (x[0] - 1.0)))
    opti.minimize(obj)
    opti.subject_to(x[0] + x[1] >= 2.0)

    # Verify direct export of FastOptPymooProblem (when constraints exist, request return_values_of=["F", "G"])
    pymoo_prob = opti.to_pymoo_problem()
    F_batch, G_batch = pymoo_prob.evaluate(
        np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
        return_values_of=["F", "G"],
    )
    assert F_batch.shape == (2, 1)
    assert G_batch.shape == (2, 1)
    np.testing.assert_allclose(F_batch[0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(G_batch[:, 0], [0.0, -2.0], atol=1e-12)

    # Stage 1: Global search via pymoo DE
    opti.solver("DE", options={"pop_size": 60, "n_gen": 80, "seed": 42})
    sol_de = opti.solve()
    assert sol_de.success
    np.testing.assert_allclose(sol_de.value(x), [1.0, 1.0], atol=1e-2)

    # Stage 2: Local gradient refinement via SLSQP (automatically warm-started from DE solution!)
    opti.solver("SLSQP", options={"ftol": 1e-12})
    sol_local = opti.solve()
    assert sol_local.success
    np.testing.assert_allclose(sol_local.value(x), [1.0, 1.0], atol=1e-6)


# ============================================================================
# 4. Multi-Objective Constrained Pareto Optimization via pymoo (NSGA2)
# ============================================================================
def test_pymoo_multi_objective_nsga2():
    _ensure_pymoo_available_for_test()

    opti = fo.Opti()
    x = opti.variable(shape=(2, 1), init=[0.5, 0.5], lb=0.0, ub=1.0)

    # Bi-objective problem:
    # f1(x) = x0^2 + x1^2
    # f2(x) = (x0 - 1)^2 + (x1 - 1)^2
    # s.t. x0 + x1 >= 0.4
    f1 = x[0] ** 2 + x[1] ** 2
    f2 = (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2

    opti.minimize([f1, f2])
    opti.subject_to(x[0] + x[1] >= 0.4)
    opti.solver("NSGA2", options={"pop_size": 60, "n_gen": 60, "seed": 123})

    sol = opti.solve()
    assert sol.success
    assert sol.pareto_x is not None and sol.pareto_f is not None
    assert sol.pareto_x.ndim == 2 and sol.pareto_x.shape[1] == 2
    assert sol.pareto_f.ndim == 2 and sol.pareto_f.shape[1] == 2
    assert len(sol.pareto_x) >= 3

    # Verify value_batch evaluates arbitrary symbolic expressions across the entire Pareto front
    sum_x_front = sol.value_batch(x[0] + x[1]).ravel()
    assert len(sum_x_front) == len(sol.pareto_x)
    assert np.all(sum_x_front >= 0.4 - 1e-4)


if __name__ == "__main__":
    t0 = time.perf_counter()
    test_blackbox_analytical_and_fd_derivatives()
    print("[PASSED] 1. CustomOp Black-Box (Analytical & Finite-Difference 1st/2nd Order AD)")

    test_blackbox_decorator_in_opti_optimization()
    print("[PASSED] 2. @fo.blackbox Decorator in Constrained SLSQP & trust-constr Optimization")

    test_pymoo_single_objective_and_hybrid_warmstart()
    print("[PASSED] 3. pymoo Single-Objective DE + Hybrid DE -> SLSQP Warm-Start")

    test_pymoo_multi_objective_nsga2()
    print("[PASSED] 4. pymoo Multi-Objective NSGA2 Pareto Front & value_batch()")

    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"\nAll 4 Phase 4 Black-Box & pymoo test suites passed in {elapsed:.2f} ms!")
