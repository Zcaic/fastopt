# fastopt

High-performance C++17 tensor-level automatic differentiation and optimization framework with `Eigen3`, `OpenMP`, and `nanobind`.

## Architecture Overview

- **`include/fastopt/core.hpp` & `src/graph.cpp`**:
  - `Shape`: 2D tensor shape metadata `(rows, cols)` stored in column-major layout matching `Eigen::MatrixXd`.
  - `Workspace`: Thread-local tiered numerical memory buffers (`val` for 0th-order forward values, `bar` for 1st-order reverse adjoints, `dot` and `dot_bar` for 2nd-order Forward-over-Reverse HVP).
  - `Node`: Stateless operator node base class with 4-channel virtual execution methods (`forward`, `backward`, `forward_tangent`, `backward_hessian`).
  - `Graph`: Three-segment contiguous memory layout separating active `VariableNode`s, static `ParameterNode`s/`ConstantNode`s (with parameter folding), and dynamic operator nodes.
- **`include/fastopt/ops.hpp`, `src/ops.cpp`, & `src/interp.cpp`**:
  - Native tensor operators: unary (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, `abs`, `square`), binary with broadcasting (`+`, `-`, `*`, `/`, `**`, `atan2`), `matmul`, `dot`, `sum`, `sumsqr`, `mean`, `norm`, `diag`, `transpose`, `slice` (including step slices `x[::2]`, integer index arrays `x[[0, 3, 5]]`, and boolean masks), `concat` (`vstack`, `hstack`), `reshape`, `clip`, `deg2rad`, `rad2deg`.
  - Overflow-safe Kreisselmeier-Steinhauser (KS) & LogSumExp constraint aggregation (`KSLogSumExpNode` / `ks_max`, `ks_min`, `logsumexp`) and $C^2$ smooth approximations (`smooth_abs`, `smooth_max`, `smooth_min`, `smooth_clip`).
  - **Multi-Dimensional Interpolation & Reusable Pre-Compiled 1D/2D/3D Interpolator Classes**:
    - Reusable classes (`LinearInterpolator1D`, `BSplineCurve1D`, `BSplineInterpolator1D`, `LinearInterpolator2D`, `BSplineSurface2D`, `BSplineInterpolator2D`, `LinearInterpolator3D`, `BSplineVolume3D`, `BSplineInterpolator3D`) that pre-compute B-spline knots and collocation/least-squares (`n_ctrl`, `reg_weight`) control points once via axis-separable Mode-$d$ sweeps, provide exact spatial derivative splines (`.derivative(...)`), use zero-allocation C++ stack buffers (`BasisDersStack`) and $O(1)$ uniform grid/knot lookup, share a single `ConstantNode` across multiple queries in the same `Graph`, and support both symbolic `Expr` queries (1st/2nd AD) and direct numerical `float`/`np.ndarray` queries.
    - Functional wrappers: `interp_linear`, `bspline_eval`, `interp_bspline`, `interp_linear_2d`, `bspline_eval_2d`, `interp_bspline_2d`, `interp_linear_3d`, `bspline_eval_3d`, `interp_bspline_3d`.
  - **Spanwise Integration & Trajectory / ODE Collocation Primitives**:
    - Composite trapezoidal rule integration `fo.trapz(y, x=None, dx=1.0, axis=0)`.
    - Single-step `fo.rk4_step` and multi-step `fo.odeint_rk4` compiled into a single C++ `ScanNode`.
    - Parallel direct collocation defect generators `fo.collocate_trapz` and 4th-order `fo.collocate_hermite_simpson` powered by `fo.vmap`.
  - **Modular Symbolic Function (`fo.Function`)**: CasADi-style reusable sub-model wrapper supporting structured numerical evaluation, direct derivative queries (`.gradient`, `.jacobian`, `.hessian`), and cross-Graph symbolic embedding via C++ `MapNode`.
  - Conditional branching (`WhereNode` with strict NaN-safe derivative masking, `SmoothWhereNode` with analytical `tanh` transition derivatives, and short-circuit `CondNode`).
  - `ImplicitSolverNode`: Embedded damped Newton-Raphson solver with exact 1st-order and 2nd-order Implicit Function Theorem (IFT) sensitivities.
  - `BlackBoxNode` (`CustomOp` / `@blackbox` & **`CBlackBox` / `c_abi.h`**):
    - Python `CustomOp` / `@blackbox` supporting multi-input/multi-output functions with analytical or automatic central finite-difference 1st-order Jacobians, 2nd-order Hessian-vector products (`HVP`), and vectorized population evaluation (`forward_batch` / `batch_fn`).
    - **Zero-GIL C-ABI Direct Bridge (`include/fastopt/c_abi.h` & `fo.CBlackBox`)**: Connects external C/C++ solvers (`fast_bemt`, `fastgeo`, or shared `.dll`/`.so`) via `PyCapsule` (`FastOptCBlackBox*`), raw function pointers, or `CBlackBox.from_dll(...)`, executing 0th/1st/2nd-order AD in 100% native C++ without acquiring the Python GIL and unlocking multi-core OpenMP parallelism.
  - `MapNode` (`fo.vmap`) & `ScanNode` (`fo.scan`): Structured loop operators compiling single-step subgraphs once and executing OpenMP multi-threaded parallel batches (`vmap`) or sequential recurrent time-marching (`scan`) in native C++ with exact 1st- and 2nd-order derivatives.
- **`include/fastopt/evaluator.hpp` & `src/evaluator.cpp`**:
  - Automatic same-point forward evaluation deduplication (`ensure_forward_evaluated`), zero-allocation OpenMP multi-threaded & wave-front population `eval_batch`, 1st-order `gradient` & OpenMP `jacobian`, OpenMP parallel 2nd-order `lagrangian_hessian`, and exact analytical parameter sensitivity `parameter_sensitivities`.
  - **Symbolic Sparsity Propagation & Greedy Graph Coloring**: Compile-time bitset propagation of exact 1st-order Jacobian sparsity (`jacobian_sparsity`) and 2nd-order symmetric Hessian sparsity (`hessian_sparsity`) through all tensor, 1D/2D/3D interpolation, `BlackBoxNode` (with optional `jac_sparsity` mask), and `vmap`/`scan`/`Function` subgraph nodes, paired with Largest-Degree-First greedy Graph Coloring (`sparse_jacobian_values`, `sparse_hessian_values`, `fo.sparse_jacobian`, `fo.sparse_hessian`).
- **`include/fastopt/ipopt_solver.hpp`, `src/ipopt_solver.cpp`, `python/fastopt/opti.py`, & `python/fastopt/solvers/`**:
  - **IPOPT**: Native C++17 `IpStdCInterface` bridge that automatically discovers and loads prebuilt `IPOPT` and `MUMPS` sparse linear solver shared libraries (`libipopt` / `libcoinmumps`) from the Python `casadi` package (`pip install casadi`) and feeds exact symbolic sparse `(iRow, jCol)` triplets and graph-colored derivative values.
  - **SciPy**: `SLSQP`, `trust-constr` (with exact 2nd-order sparse CSC Lagrangian Hessian), `L-BFGS-B`, etc.
  - **pymoo**: Vectorized evolutionary solver adapter (`DE`, `GA`, `PSO`, `CMAES`, `ISRES`, `SRES`, `NSGA2`, `NSGA3`, `MOEAD`, `SMSEMOA`, `AGEMOEA`) powered by C++ OpenMP `eval_batch`.
  - **Compilation Caching, Parametric Sweeps, Post-Optimality Sensitivity & Diagnostics**: Automatic `Evaluator` & sparsity caching across repeated `opti.solve()` and `opti.solve_sweep(params, values)` calls, analytical KKT parameter sensitivity `sol.sensitivity(params, target=...)`, variable/constraint/objective scaling (`scale=...`, `auto_scale=True`), physical-unit KKT Lagrange multiplier queries (`sol.dual(con)`, `sol.bound_dual_lb(var)`, `sol.bound_dual_ub(var)`), iteration callbacks & history (`opti.callback(fn)`, `sol.history`), and infeasibility diagnostics (`opti.debug_infeasibility(x)`).

## Prerequisites & Build Instructions (Windows MSYS2 ucrt64 / Linux)

```bash
pip install numpy scipy casadi pymoo
cmake --preset gcc
cmake --build build -j
python tests/test_phase1_ad_core.py
python tests/test_phase2_interp_and_scipy.py
python tests/test_phase3_ipopt.py
python tests/test_phase4_blackbox_and_pymoo.py
python tests/test_phase5_loops_and_batch_bb.py
python tests/test_phase6_math_and_nd_interp.py
python tests/test_phase7_sparse_and_diagnostics.py
python tests/test_phase8_c_abi_bridge.py
python tests/test_phase9_interpolator_classes.py
python tests/test_phase10_enhancements.py
python tests/test_phase11_perf_and_3d_mdo.py
```

---

# `fastopt` 完整使用指南 (User Guide & API Reference)

本指南涵盖 `fastopt` 从基础声明式优化建模、张量与平滑算子、1D/2D/3D 预编译插值器、模块化子模型与隐式方程、结构化循环与轨迹配点、Python/C-ABI 黑盒直连、多目标进化算法，到参数扫描与 KKT 灵敏度分析的完整用法。

---

## 目录 (Table of Contents)

1. [快速上手：声明式优化建模 (`Opti`)](#1-快速上手声明式优化建模-opti)
2. [张量算子、高级切片索引与平滑聚合函数](#2-张量算子高级切片索引与平滑聚合函数)
3. [一维 / 二维 / 三维预编译插值器与 B 样条拟合](#3-一维--二维--三维预编译插值器与-b-样条拟合)
4. [模块化符号子模型 (`fo.Function`) 与隐式方程 (`fo.implicit_solve`)](#4-模块化符号子模型-fofunction-与隐式方程-foimplicit_solve)
5. [结构化循环 (`vmap` / `scan`)、展向积分 (`trapz`) 与轨迹配点 (`odeint_rk4` / `collocate_*`)](#5-结构化循环-vmap--scan展向积分-trapz-与轨迹配点-odeint_rk4--collocate_)
6. [自定义黑盒算子 (`@blackbox`) 与零 GIL C-ABI 直连 (`fo.CBlackBox`)](#6-自定义黑盒算子-blackbox-与零-gil-c-abi-直连-focblackbox)
7. [多目标与进化算法求解器 (`pymoo` 后端与 `eval_batch`)](#7-多目标与进化算法求解器-pymoo-后端与-eval_batch)
8. [参数扫描 (`solve_sweep`)、缩放、KKT 乘子、解析灵敏度 (`sol.sensitivity`) 与诊断](#8-参数扫描-solve_sweep缩放kkt-乘子解析灵敏度-solsensitivity-与诊断)
9. [底层计算图与稀疏导数引擎接口 (`GraphBuilder` & `Evaluator`)](#9-底层计算图与稀疏导数引擎接口-graphbuilder--evaluator)

---

## 1. 快速上手：声明式优化建模 (`Opti`)

`fo.Opti` 提供与 CasADi `Opti` 风格一致的声明式建模接口，同时继承自 `fo.GraphBuilder`。

### 1.1 变量、冻结参数与常量的声明

- **设计变量 (`Variable`)**：参与求导与优化更新的未知量（`freeze=False`）。
- **冻结参数 (`Parameter`)**：在单次优化中保持固定、但可在多次求解之间通过 `.set_value(...)` 动态修改的工况参数（无需重新编译计算图）。
- **静态常量 (`Constant`)**：完全固定不变的数值或矩阵。

```python
import numpy as np
import fastopt as fo

opti = fo.Opti()

# 1. Declare decision variables (scalar or 2D/column vector)
x = opti.variable(shape=(3, 1), init=[1.0, 2.0, 3.0], lb=0.0, ub=10.0, name="x")
pitch_deg = opti.variable(init=5.0, lb=-5.0, ub=25.0, scale=10.0, name="pitch_deg")

# 2. Declare frozen parameters (can also use opti.variable(..., freeze=True))
v_inf = opti.parameter(value=30.0, name="v_inf")

# 3. Declare static constants
weights = opti.constant([0.2, 0.3, 0.5], name="weights")

# 4. Define objective and constraints
obj = fo.sumsqr(x - v_inf * 0.1) + 0.5 * fo.square(pitch_deg)
opti.minimize(obj)

opti.subject_to(fo.dot(weights, x) + 0.1 * pitch_deg == 2.5, name="balance_eq")
opti.subject_to(x[0] + x[1] <= 5.0, name="sum_ub")

# 5. Select solver ("ipopt", "SLSQP", "trust-constr", "L-BFGS-B", etc.) and solve
opti.solver("ipopt", options={"tol": 1e-8, "print_level": 0}, auto_scale=True)
sol = opti.solve()

print("Converged:", sol.success, "Optimal objective:", sol.f)
print("Optimal x:", sol.value(x))
print("Optimal pitch (deg):", sol.value(pitch_deg))
```

### 1.2 支持的梯度类求解器后端

| 求解器名称 (`opti.solver(...)`) | 底层实现 | 适用场景与特点 |
| :--- | :--- | :--- |
| `"ipopt"` | C++17 原生 `IpStdCInterface` 直连（自动发现 `casadi` 内置的 `libipopt` 与 `MUMPS`） | **大规模稀疏非线性规划首选**；零 Python 回调开销，直接消费图染色稀疏雅可比与精确拉格朗日海森矩阵 |
| `"SLSQP"` | `scipy.optimize.minimize` (SLSQP) | **中小规模稠密约束优化首选**（数十至数百变量）；启动开销极低，适合毫秒级快速配平 |
| `"trust-constr"` | `scipy.optimize.minimize` (Trust-Region Interior Point) | 支持传入 `scipy.sparse.csc_matrix` 稀疏雅可比与稀疏海森矩阵，无需安装 `casadi` 即可求解中大规模问题 |
| `"L-BFGS-B"` / `"BFGS"` | `scipy.optimize.minimize` | 无约束或仅含简单上下界 (`lb` / `ub`) 的拟牛顿优化 |

---

## 2. 张量算子、高级切片索引与平滑聚合函数

### 2.1 基础数学、线性代数与宇航辅助函数

| 类别 | 函数列表 |
| :--- | :--- |
| **逐元素初等函数** | `fo.sin`, `fo.cos`, `fo.tan`, `fo.asin`, `fo.acos`, `fo.atan`, `fo.atan2(y, x)`, `fo.sinh`, `fo.cosh`, `fo.tanh`, `fo.exp`, `fo.log`, `fo.sqrt`, `fo.abs`, `fo.square` |
| **角度转换与裁剪** | `fo.deg2rad(x)`, `fo.rad2deg(x)`, `fo.clip(x, lb, ub)` |
| **线性代数与拼接** | `A @ B` (`fo.matmul(A, B)`), `fo.dot(a, b)`, `fo.diag(x)`（向量转对角阵 / 方阵提取主对角线）, `x.T` (`fo.transpose(x)`), `x.reshape(r, c)`, `fo.vstack([...])`, `fo.hstack([...])`, `fo.concat([...], axis=0/1)` |
| **归约与范数算子** | `fo.sum(x, axis=None/0/1)`, `fo.sumsqr(x)`（单节点 `DotNode(x, x)`，无中间平方张量）, `fo.mean(x, axis=None/0/1)`, `fo.norm(x, ord=2/1, axis=None/0/1, eps=0.0)` |

### 2.2 高级切片与花式索引 (`Expr.__getitem__`)

`Expr` 支持连续切片、步长切片、倒序切片、整数数组索引及布尔掩码（自动将相邻索引合并为最少数量的底层 `SliceNode`）：

```python
gb = fo.GraphBuilder()
x = gb.variable(shape=(8, 1), init=np.arange(8, dtype=float))

sub_contig = x[2:6]                  # Contiguous slice -> single C++ SliceNode
even_elems = x[0::2]                 # Step slice: indices [0, 2, 4, 6]
rev_elems  = x[::-1]                 # Reversed vector: indices [7, 6, ..., 0]
picked     = x[[0, 1, 5, 6]]         # Fancy integer indexing (coalesced into 2 runs: 0:2 and 5:7)
mask       = np.array([True, False, True, True, False, False, False, True])
masked     = x[mask]                 # Boolean mask indexing
```

### 2.3 $C^2$ 平滑近似、KS 包络聚合与条件分支

当优化模型中包含非光滑的 `max`、`min`、`|x|` 或分段物理公式（如失速前后气动切换）时，使用平滑算子可保证二阶海森矩阵连续，避免梯度求解器振荡：

- **Kreisselmeier-Steinhauser (KS) 约束聚合**（适合将成百上千个局部应力/过载约束聚合成单个标量约束）：
  - `fo.ks_max(x, rho=50.0, axis=None)`：平滑上包络（保守上界，$\ge \max(x)$）；
  - `fo.ks_min(x, rho=50.0, axis=None)`：平滑下包络（保守下界，$\le \min(x)$）；
  - `fo.logsumexp(x, rho=50.0, axis=None)`：非保守平滑最大值近似。
- **逐元素平滑算子**：
  - `fo.smooth_abs(x, eps=1e-6)`：$\sqrt{x^2 + \varepsilon^2}$；
  - `fo.smooth_max(a, b, eps=1e-4)` / `fo.smooth_min(a, b, eps=1e-4)`；
  - `fo.smooth_clip(x, lb, ub, eps=1e-4)`。
- **条件分支算子**：
  - `fo.where(cond_expr, true_val, false_val)`：硬分支选择（具备严格的 NaN 安全反向导数掩码，未命中分支若产生 `log(-1)` 或 `1/0` 不会污染命中分支的梯度与海森）；
  - `fo.smooth_where(x, true_val, false_val, threshold=0.0, sharpness=10.0)`：基于 `tanh` 的 $C^\infty$ 平滑过渡分支；
  - `fo.cond(cond_expr, true_fn, false_fn, operands)`：短路子图分支，前向与反向传播仅执行命中的那个子图。

---

## 3. 一维 / 二维 / 三维预编译插值器与 B 样条拟合

在旋翼/机翼气动优化中，同一个翼型极曲线表或高维气动数据库往往会在不同展向站位、不同方位角被重复查询数十次。**强烈推荐使用预编译插值器类**，其在初始化时仅求解一次样条控制顶点（3D 采用 Mode-1/2/3 轴向分离极速扫掠），并在同一个计算图中自动复用同一个底层 `ConstantNode`。

### 3.1 预编译插值器类一览

| 维度 | 线性插值器类 | $C^2$ B 样条插值器类（输入网格数据自动算控制点） | B 样条几何评估器类（直接输入控制顶点与节点向量） |
| :--- | :--- | :--- | :--- |
| **1D** | `fo.LinearInterpolator1D` | `fo.BSplineInterpolator1D` | `fo.BSplineCurve1D` |
| **2D** | `fo.LinearInterpolator2D` | `fo.BSplineInterpolator2D` | `fo.BSplineSurface2D` |
| **3D** | `fo.LinearInterpolator3D` | `fo.BSplineInterpolator3D` | `fo.BSplineVolume3D` |

### 3.2 预编译插值器的构建、数值/符号双模调用与空间导数 (`.derivative`)

```python
import numpy as np
import fastopt as fo

# 1. Pre-compile a 2D airfoil polar table (alpha_grid x re_grid -> [Cl, Cd]) ONCE
alpha_grid = np.linspace(-10.0, 20.0, 31)
re_grid    = np.linspace(1e5, 2e6, 20)
A, R = np.meshgrid(alpha_grid, re_grid, indexing="ij")
cl_table = 0.11 * A * (1.0 + 0.05 * np.log10(R / 1e5))
cd_table = 0.008 + 0.0004 * (A**2)
polar_data = np.stack([cl_table, cd_table], axis=-1)  # shape (31, 20, 2)

polar_interp = fo.BSplineInterpolator2D(
    alpha_grid, re_grid, polar_data, degree_x=3, degree_y=3, extrapolate=True
)

# 2. Direct numerical evaluation (outside any Graph)
cl_cd_num = polar_interp(4.5, 8.5e5)  # returns 1D array [Cl, Cd]

# 3. Exact spatial partial derivative surface d(Cl, Cd) / d(alpha)
dpolar_dalpha = polar_interp.derivative(nu_u=1, nu_v=0)
dcl_da_num = dpolar_dalpha(4.5, 8.5e5)

# 4. Symbolic evaluation across multiple blade stations inside Opti (shares a single ConstantNode!)
opti = fo.Opti()
alpha_stations = opti.variable(shape=(12, 1), init=4.0, lb=-5.0, ub=15.0)
re_stations    = opti.constant(np.linspace(3e5, 1.8e6, 12))

aero_coeffs = polar_interp(alpha_stations, re_stations)  # shape (12, 2)
cl_vec = aero_coeffs[:, 0]
cd_vec = aero_coeffs[:, 1]
```

### 3.3 含噪数据正则化最小二乘 B 样条拟合与三维极速样条

当实验或 CFD 采样数据含有噪声时，可通过指定控制点个数 `n_ctrl < len(grid)` 和二阶差分粗糙度惩罚权重 `reg_weight > 0` 生成平滑样条：

```python
# Smooth 1D least-squares B-spline fitting (50 noisy points -> 12 control points)
x_samples = np.linspace(0.0, 1.0, 50)
y_noisy   = np.sin(2.0 * np.pi * x_samples) + 0.03 * np.random.randn(50)
smooth_spl = fo.BSplineInterpolator1D(
    x_samples, y_noisy, degree=3, n_ctrl=12, reg_weight=1e-3
)
dy_dx_spl = smooth_spl.derivative(1)   # 1st derivative spline
d2y_dx2   = smooth_spl.derivative(2)   # 2nd derivative (curvature) spline

# 3D C2 B-Spline aerodynamic table (Mach x Re x Alpha -> Cl)
# Uses Mode-1/2/3 separable 1D sweeps: a 20x25x30 (15,000 pts) grid solves in ~1.4 ms!
mach_g  = np.linspace(0.1, 0.8, 20)
re_g    = np.linspace(1e5, 5e6, 25)
alpha_g = np.linspace(-8.0, 18.0, 30)
cl_3d_data = np.ones((20, 25, 30), dtype=np.float64)

aero_3d = fo.BSplineInterpolator3D(
    mach_g, re_g, alpha_g, cl_3d_data, degree_x=2, degree_y=2, degree_z=3
)
dcl_dalpha_3d = aero_3d.derivative(0, 0, 1)
```

> **提示（几何优化 / FFD 控制点优化）**：`BSplineCurve1D`、`BSplineSurface2D`、`BSplineVolume3D` 以及各 `BSplineInterpolator*` 类的 `ctrl_pts` / `y_data` 参数均支持直接传入符号变量 `Expr`（例如将翼型或机翼的 B 样条控制点设为 `opti.variable(...)`），C++ 内核会同时计算关于查询坐标和控制点变量的精确一阶与二阶导数。

---

## 4. 模块化符号子模型 (`fo.Function`) 与隐式方程 (`fo.implicit_solve`)

### 4.1 模块化可复用子模型 (`fo.Function`)

当一个复杂的学科子模型（如旋翼叶素气动模型、机翼结构刚度模型）需要在悬停、过渡、巡航等多个设计点重复调用时，使用 `fo.Function` 可将其独立编译一次，随后在任意 `Opti` 或 `GraphBuilder` 中直接调用（底层通过单批次 `MapNode` 嵌入，无需重新做 Python 符号追踪，且完整保留符号稀疏性）：

```python
# Step 1: Build and compile the standalone sub-model once
gb_sub = fo.GraphBuilder()
twist = gb_sub.variable(shape=(4, 1), name="twist")
v_ax  = gb_sub.variable(shape=(1, 1), name="v_ax")

r_norm = gb_sub.constant([0.25, 0.50, 0.75, 0.95])
v_tan  = 120.0 * r_norm
phi    = fo.atan2(v_ax, v_tan)
alpha  = twist - phi
q_dyn  = 0.5 * 1.225 * (fo.square(v_tan) + fo.square(v_ax))
thrust = fo.sum(q_dyn * 0.12 * (5.7 * alpha * fo.cos(phi)))
power  = fo.sum(q_dyn * 0.12 * (0.012 + 0.04 * fo.square(alpha)) * v_tan)

rotor_model = fo.Function("rotor_bem", [twist, v_ax], [thrust, power])

# Direct numerical call & derivatives:
T_val, P_val = rotor_model([0.22, 0.18, 0.14, 0.10], 5.0)
J_T = rotor_model.jacobian([0.22, 0.18, 0.14, 0.10], 5.0, out_idx=0)

# Step 2: Reuse symbolically across multiple flight conditions in Opti
opti = fo.Opti()
twist_design = opti.variable(shape=(4, 1), init=0.15, lb=0.02, ub=0.40)
dtheta_cruise = opti.variable(init=0.05, lb=-0.15, ub=0.25)

T_hover, P_hover   = rotor_model(twist_design, 4.0)
T_cruise, P_cruise = rotor_model(twist_design + dtheta_cruise, 28.0)

opti.minimize(0.6 * P_hover + 0.4 * P_cruise)
opti.subject_to(T_hover >= 400.0)
opti.subject_to(T_cruise >= 160.0)
```

### 4.2 内嵌隐式残差方程求解 (`fo.implicit_solve`)

对于诱导速度平衡方程 $R(x, v_i) = 0$ 或配平残差方程，无需将内部状态提升为外部优化器的决策变量，`fo.implicit_solve` 会在前向传播时用阻尼牛顿法自动收敛，并在反向/二阶传播时用一阶与二阶隐函数定理（IFT）给出精确解析导数：

```python
gb = fo.GraphBuilder()
ct_ hover = gb.variable(init=0.008, lb=0.001, ub=0.05)

# Solve 2 * v_i * sqrt(v_i^2 + v_inf^2) - ct = 0 for induced velocity v_i
v_i_sol = fo.implicit_solve(
    residual_fn=lambda ct, vi: 2.0 * vi * fo.sqrt(fo.square(vi) + 0.01) - ct,
    inputs=[ct_hover],
    init_guess=0.06,
    lb=1e-4,
    ub=1.0,
)
```

---

## 5. 结构化循环 (`vmap` / `scan`)、展向积分 (`trapz`) 与轨迹配点 (`odeint_rk4` / `collocate_*`)

### 5.1 并行批映射 (`fo.vmap`) 与时序递推 (`fo.scan`)

避免使用 Python `for` 循环展开成千上万个重复图节点：
- `fo.vmap(fn, inputs, in_axes=..., shared_inputs=..., out_axis=...)`：将单步函数编译为单个 `MapNode`，在 C++ 内利用 OpenMP 多线程并行计算所有独立批次（如 $N$ 个展向叶素或 $N$ 个气动条带）。
- `fo.scan(step_fn, init_carry, xs=..., shared_inputs=..., length=..., seq_axes=..., out_axis=...)`：将递推步编译为单个 `ScanNode`，在 C++ 内顺序推进时序状态并自动保存前向轨迹以供一阶/二阶离散伴随求导。

### 5.2 展向非均匀梯形积分 (`fo.trapz`)

```python
# Integrate spanwise sectional thrust dT/dr over non-uniform radial stations r_stations
r_stations = np.array([0.20, 0.35, 0.55, 0.75, 0.90, 0.98, 1.00])
total_thrust = fo.trapz(dT_dr_expr, x=r_stations, axis=0)
```

### 5.3 四阶龙格-库塔积分 (`fo.odeint_rk4`) 与直接配点最优控制 (`fo.collocate_*`)

`fastopt` 内置面向飞行轨迹优化的常微分方程积分与高阶配点缺陷生成算子：

```python
import numpy as np
import fastopt as fo

# 1. Single-ScanNode RK4 trajectory shooting (compiles N steps into 1 ScanNode)
def ode_rhs(state: fo.Expr, u: fo.Expr) -> fo.Expr:
    pos, vel = state[0], state[1]
    return fo.vstack([vel, -0.5 * vel + u[0]])

# 2. Direct 4th-order Hermite-Simpson collocation in Opti (parallelized via vmap)
N = 30
dt = 2.0 / N
opti = fo.Opti()
X = opti.variable(shape=(2, N + 1), init=0.0, name="X")
U = opti.variable(shape=(1, N + 1), init=0.0, lb=-10.0, ub=10.0, name="U")

# Generate (2, N) collocation defect matrix (also supports fo.collocate_trapz)
defects = fo.collocate_hermite_simpson(ode_rhs, X, U, dt=dt)

opti.minimize(fo.trapz(fo.square(U.T), dx=dt))
opti.subject_to(defects == 0.0, name="dynamics_defects")
opti.subject_to(X[:, 0] == np.array([0.0, 0.0]), name="initial_state")
opti.subject_to(X[:, N] == np.array([1.0, 0.0]), name="target_state")

opti.solver("SLSQP")
sol = opti.solve()
```

---

## 6. 自定义黑盒算子 (`@blackbox`) 与零 GIL C-ABI 直连 (`fo.CBlackBox`)

### 6.1 Python 黑盒装饰器 (`@fo.blackbox` / `fo.CustomOp`)

用于接入外部 Python 数值计算函数。若不提供 `jac_fn` 或 `hvp_fn`，C++ 内核会自动使用中心有限差分计算一阶雅可比与二阶海森-向量积（HVP）；若提供 `batch_fn`，进化算法种群求值时会自动触发波前批量回调：

```python
@fo.blackbox(
    shape=(2, 1),
    jac_fn=lambda u: np.array([[2.0 * u[0], 0.0], [0.0, np.cos(u[1])]]),
    batch_fn=lambda U_pop: np.column_stack([U_pop[:, 0] ** 2, np.sin(U_pop[:, 1])]),
    fd_step=1e-7,
    fd_hessian=True,
)
def external_aero_op(u: np.ndarray) -> np.ndarray:
    return np.array([u[0] ** 2, np.sin(u[1])])
```

### 6.2 零 GIL C-ABI 原生动态库/指针直连 (`include/fastopt/c_abi.h` & `fo.CBlackBox`)

针对 `fast_bemt`、`fastgeo` 等 C/C++ 编写的高性能求解器，通过 `#include "fastopt/c_abi.h"` 导出标准 C 函数或 `PyCapsule`，即可在 C++ 层实现**零 Python GIL、零内存拷贝、多核 OpenMP 并行**的0阶/1阶/2阶求导直连（比 Python `@blackbox` 快 **10～40 倍**）：

```python
# Load directly from a compiled C/C++ shared library (.dll / .so)
bemt_op = fo.CBlackBox.from_dll(
    dll_path="build/libfast_bemt_bridge.dll",
    shape=(2, 1),                      # Output shape, e.g. [Thrust, Torque]
    eval_symbol="bemt_eval_c",         # Required 0th-order C function symbol
    jac_symbol="bemt_jac_c",           # Optional analytical 1st-order Jacobian symbol (or None for C++ FD)
    hvp_symbol="bemt_hvp_c",           # Optional analytical 2nd-order HVP symbol (or None for C++ FD)
    eval_batch_symbol="bemt_batch_c",  # Optional OpenMP population batch symbol
    is_thread_safe=True,               # Unlocks multi-core OpenMP Hessian & batch parallelism
)
```

---

## 7. 多目标与进化算法求解器 (`pymoo` 后端与 `eval_batch`)

同一个 `Opti` 问题无需修改任何建模代码，只需将 `opti.solver(...)` 切换为 `pymoo` 算法名称，即可利用 C++ OpenMP `eval_batch` 批量求解单目标全局优化或多目标帕累托前沿（Pareto Front）：

- **单目标进化算法**：`"DE"`（差分进化）、`"GA"`（遗传算法）、`"PSO"`（粒子群）、`"CMAES"`、`"ISRES"`、`"SRES"`、`"NelderMead"`、`"PatternSearch"`。
- **多目标进化算法**（此时 `opti.minimize(fo.vstack([obj1, obj2]))` 可直接传入向量目标）：`"NSGA2"`、`"NSGA3"`、`"MOEAD"`、`"SMSEMOA"`、`"AGEMOEA"`。

```python
opti = fo.Opti()
x = opti.variable(shape=(4, 1), init=0.5, lb=0.0, ub=1.0)

# Bi-objective optimization: minimize [f1, f2] simultaneously
f1 = x[0]
g  = 1.0 + 9.0 * fo.mean(x[1:])
f2 = g * (1.0 - fo.sqrt(f1 / g))
opti.minimize(fo.vstack([f1, f2]))
opti.subject_to(fo.sum(x) <= 2.5)

opti.solver("NSGA2", options={"pop_size": 80, "n_gen": 60, "seed": 42})
sol = opti.solve()

# Query Pareto-optimal decision set (N_pareto, 4) and objective front (N_pareto, 2)
pareto_X = sol.pareto_x
pareto_F = sol.pareto_f
# Evaluate any intermediate expression across the entire Pareto set in parallel:
g_on_pareto = sol.value_batch(g)
```

---

## 8. 参数扫描 (`solve_sweep`)、缩放、KKT 乘子、解析灵敏度 (`sol.sensitivity`) 与诊断

### 8.1 零重编译包络参数扫描 (`opti.solve_sweep`)

在计算旋翼/倾转旋翼全包线配平走廊时，`opti.solve_sweep` 会自动缓存并复用首次编译的 `Evaluator` 与图染色稀疏结构（计算图节点数保持恒定），并默认使用上一状态点的最优解对下一状态点做热启动（`warm_start=True`）：

```python
# Sweep across 15 flight speeds and nacelle tilt angles without recompiling the graph
v_grid    = np.linspace(15.0, 75.0, 15)
tilt_grid = np.linspace(80.0, 0.0, 15)
solutions = opti.solve_sweep([v_inf_param, tilt_deg_param], list(zip(v_grid, tilt_grid)), warm_start=True)
```

### 8.2 变量/约束缩放、KKT 对偶乘子与最优点参数解析灵敏度 (`sol.sensitivity`)

```python
# 1. Query physical-unit KKT Lagrange multipliers (dJ* / d(constraint_violation))
lam_trim = sol.dual(trim_constraint_handle)
lam_lb   = sol.bound_dual_lb(pitch_deg)
lam_ub   = sol.bound_dual_ub(pitch_deg)

# 2. Analytical post-optimality parameter sensitivity WITHOUT re-solving the NLP:
#    Computes exact dJ*/dp (Envelope Theorem) and dx*/dp (Active KKT Implicit Function Theorem)
sens = sol.sensitivity([v_inf_param, weight_param])
df_dp = sens["df_dp"]  # Total derivative of optimal objective w.r.t. [v_inf, weight]
dx_dp = sens["dx_dp"]  # Total derivative matrix of optimal decision vector x* w.r.t. [v_inf, weight]

# Or query sensitivity of a specific decision variable or objective w.r.t. a single parameter:
dtheta_dv = sol.sensitivity(v_inf_param, target=pitch_deg)
```

### 8.3 迭代回调与不可行诊断 (`opti.callback` & `opti.debug_infeasibility`)

```python
# Register per-iteration callback
opti.callback(lambda info: print(f"Iter {info['iter']}: obj={info['obj']:.4e}, viol={info['constr_viol']:.2e}"))

# Diagnose which bounds or constraints are violated if a solve fails to converge
report = opti.debug_infeasibility(tol=1e-5, print_report=True)
```

---

## 9. 底层计算图与稀疏导数引擎接口 (`GraphBuilder` & `Evaluator`)

如果不需要使用优化器，而是仅将 `fastopt` 作为高速一阶/二阶自动微分与稀疏雅可比/海森计算引擎使用，可直接使用 `fo.GraphBuilder` 与 `Evaluator`：

```python
import numpy as np
import fastopt as fo

gb = fo.GraphBuilder()
x = gb.variable(shape=(100, 1), init=np.linspace(0.1, 1.0, 100), name="x")

# Tridiagonal nonlinear residual and scalar objective
res = fo.sin(x[:-1]) - 0.5 * fo.cos(x[1:])
obj = fo.sumsqr(res)

ev = gb.build_evaluator([obj, res])
x0 = gb.cpp_graph.initial_decision_vector()

# 1. Forward evaluation & OpenMP batch evaluation
f_val, r_val = ev.eval(x0)
X_pop = np.random.uniform(0.1, 1.0, size=(256, 100))
f_batch, r_batch = ev.eval_batch(X_pop, num_threads=4)

# 2. First-order gradient & Graph-Colored sparse/dense Jacobian
g = ev.gradient(x0, out_idx=0)
J_dense  = ev.jacobian(x0, out_idx=1, num_threads=4)
J_sparse = fo.sparse_jacobian(ev, x0, out_idx=1)  # returns scipy.sparse.csc_matrix
print("Jacobian colors required:", ev.num_jacobian_colors(out_idx=1))  # 2 colors instead of 99 rows!

# 3. Second-order HVP & Graph-Colored sparse/dense Lagrangian Hessian
v = np.ones(100)
hvp = ev.hessian_vec_prod(x0, obj_scale=1.0, constraint_multipliers=[np.zeros(99)], v=v)
H_sparse = fo.sparse_hessian(ev, x0, obj_scale=1.0, constraint_multipliers=[np.zeros(99)], num_threads=4)
print("Hessian colors required:", ev.num_hessian_colors())             # 3 colors instead of 100 cols!
```


### 十、全量函数与类参数提示速查表 (Complete API Parameter Hint Reference)

本项目已在 `python/fastopt/` 目录下内置了符合 PEP 561 规范的 `py.typed` 标记文件、C++ 扩展类型存根 `_fastopt_impl.pyi` 以及全量 Python 类型注解与文档字符串，在 VS Code (Pylance/Pyright) 或 PyCharm 中悬停任意函数/方法即可实时弹出参数提示。同时也可在交互式终端或脚本中直接调用 `fo.help_api()`（或 `fo.help_api(fo.BSplineInterpolator3D)`）输出任意接口的完整参数签名。

#### 10.1 优化建模与求解 (`Opti` / `OptiSol` / `Function`)

| 接口 / 方法签名 | 参数说明与默认值 | 返回值 |
| :--- | :--- | :--- |
| `opti.variable(shape=(1, 1), init=0.0, lb=None, ub=None, freeze=False, scale=None, name="")` | `shape`: 二维形状 `(rows, cols)` 或标量长度；`init`: 初值；`lb`/`ub`: 上下界；`freeze=True`: 冻结为零开销 `Parameter`；`scale`: 变量特征量纲 | `Expr` |
| `opti.parameter(shape=(1, 1), val=0.0, name="")` | `shape`: 参数形状；`val`: 初始数值；支持在多次 `solve()` 之间调用 `set_value` 动态更新且不触发图重编译 | `Expr` |
| `opti.constant(val, shape=None, name="")` | `val`: 不可变常数标量或数组；`shape`: 可选显式二维形状 | `Expr` |
| `opti.minimize(obj, scale=None)` | `obj`: 标量目标 `Expr`（单目标）或目标列表 `[f1, f2]`（`pymoo` 多目标）；`scale`: 目标函数特征缩放尺度 | `None` |
| `opti.subject_to(constraint, scale=None, name="")` | `constraint`: 单个或列表约束（`g <= ub`, `g >= lb`, `h == target`）；`scale`: 约束特征尺度 | 约束句柄 `ComparisonExpr` |
| `opti.solver(name="scipy", method=None, options=None, use_hessian=True, auto_scale=False, callback=None)` | `name`: `'ipopt'`, `'scipy'` (`'SLSQP'`, `'trust-constr'`, `'L-BFGS-B'`), `'pymoo'` (`'DE'`, `'GA'`, `'PSO'`, `'CMAES'`, `'ISRES'`, `'NSGA2'`)；`options`: 求解器字典；`use_hessian`: 是否启用精确二阶海森；`auto_scale`: 自动量纲归一化 | `None` |
| `opti.solve(auto_scale=None, callback=None)` | 编译（或复用缓存的 `Evaluator`）并调用求解器，解算完成后自动将最优解回写为变量初值供下一阶段热启动 | `OptiSol` |
| `opti.solve_sweep(params, values, warm_start=True, auto_scale=None, step_callback=None)` | `params`: 单个或序列 `Parameter`；`values`: 扫描工况序列；`warm_start=True`: 自动沿用上一工况最优解热启动 | `List[OptiSol]` |
| `opti.solve_multistart(n_starts=8, sampler='lhs', seed=42, include_current_init=True, batch_screen_factor=4, penalty_weight=1e4, default_bound_span=10.0, auto_scale=None, return_all=False)` | 基于 C++ OpenMP `eval_batch` 并行筛查拉丁超立方候选集并从最优候选点群执行多起点局部 NLP 求解 | `OptiSol` 或 `(OptiSol, List[OptiSol])` |
| `sol.to_dict()` / `opti.load_state(state)` | 导出最优解状态字典（含变量命名映射、目标值与 KKT 乘子），或从字典/`OptiSol` 恢复变量初值实现跨会话热启动 | `Dict[str, Any]` / `None` |
| `fo.solve(A, b)` / `fo.cross(a, b)` / `fo.trace(A)` | 符号线性方程组求解 $A Y = b$（一阶/二阶 LU 伴随）、三维向量叉乘 $\mathbf{a}\times\mathbf{b}$ 与方阵迹 | `Expr` |
| `opti.debug_infeasibility(x=None, tol=1e-5, print_report=False)` | 检查点 `x`（默认当前初值/解）处所有变量边界与非线性约束违反量并按降序排列 | `Dict[str, Any]` |
| `sol.value(expr)` | 提取任意决策变量、参数或中间符号表达式 `expr` 在最优解 `x*` 处的数值 | `float` 或 `np.ndarray` |
| `sol.dual(constraint)` / `sol.bound_dual_lb(var)` / `sol.bound_dual_ub(var)` | 查询不等式/等式约束或变量上下界在最优解处的物理量纲 KKT 拉格朗日乘子 | `float` 或 `np.ndarray` |
| `sol.sensitivity(params, target="f", reg=1e-9, active_tol=1e-6)` | 基于包络定理与紧约束 KKT 隐函数定理计算解析参数灵敏度：`target="f"` 返回 `df*/dp`，`target="x"` 返回 `(df*/dp, dx*/dp)`，传入 `Expr` 返回 `dg(x*, p)/dp` | `np.ndarray` 或 `Tuple` |
| `fo.Function(name, inputs, outputs)` | 将符号子图封装为可复用模块；支持符号调用 `fn(e1, e2)`、数值调用 `fn(v1, v2)`、`fn.gradient(...)`、`fn.jacobian(...)`、`fn.hessian(...)`、`fn.eval_batch(X_pop)` | `Function` |

#### 10.2 可复用预编译插值器类体系 (`1D` / `2D` / `3D`)

| 类与方法签名 | 参数说明与默认值 | 返回值 |
| :--- | :--- | :--- |
| `fo.LinearInterpolator1D(x_grid, y_data, extrapolate=False)` | `x_grid`: 严格递增一维网格 `(N,)`；`y_data`: `(N,)` 或多通道 `(N, K)` 数组或符号 `Expr` | 可调用插值器 `lerp(xq)` |
| `fo.BSplineInterpolator1D(x_grid, y_data, degree=3, extrapolate=False, n_ctrl=None, reg_weight=0.0)` | 构造时一次性完成 Not-a-Knot 插值或带正则化最小二乘拟合（`n_ctrl < N`, `reg_weight >= 0`）；支持 `.derivative(nu=1)` 导出精确导数曲线 | 可调用样条 `sp(xq)` |
| `fo.LinearInterpolator2D(x_grid, y_grid, z_data, extrapolate=False)` | `x_grid`: `(Nx,)`, `y_grid`: `(Ny,)`, `z_data`: `(Nx, Ny)` 或多通道 `(Nx, Ny, K)` | 可调用插值器 `lerp2d(xq, yq)` |
| `fo.BSplineInterpolator2D(x_grid, y_grid, z_data, degree_x=3, degree_y=3, extrapolate=False, n_ctrl_x=None, n_ctrl_y=None, reg_weight=0.0)` | 二维张量积 $C^2$ B 样条曲面预编译插值器；支持 `.derivative(nu_u=1, nu_v=0)` 导出偏导曲面 | 可调用曲面 `surf(xq, yq)` |
| `fo.LinearInterpolator3D(x_grid, y_grid, z_grid, w_data, extrapolate=False)` | `x_grid`: `(Nx,)`, `y_grid`: `(Ny,)`, `z_grid`: `(Nz,)`, `w_data`: `(Nx, Ny, Nz)` 或 `(Nx, Ny, Nz, K)` | 可调用插值器 `lerp3d(xq, yq, zq)` |
| `fo.BSplineInterpolator3D(x_grid, y_grid, z_grid, w_data, degree_x=3, degree_y=3, degree_z=3, extrapolate=False, n_ctrl_x=None, n_ctrl_y=None, n_ctrl_z=None, reg_weight=0.0)` | 基于 Mode-1/2/3 轴向分离扫掠的毫秒级三维 $C^2$ B 样条体插值器；支持 `.derivative(nu_u, nu_v, nu_w)` 导出偏导数体 | 可调用样条体 `vol(xq, yq, zq)` |

#### 10.3 轨迹配点、展向积分、隐式求解与结构化循环

| 函数签名 | 参数说明与默认值 | 返回值 |
| :--- | :--- | :--- |
| `fo.trapz(y, x=None, dx=1.0, axis=0)` / `fo.trapezoid(...)` | 沿 `axis=0` 或 `1` 的符号复合梯形积分；`x` 支持非均匀网格数组或符号坐标 `Expr` | `Expr` |
| `fo.rk4_step(rhs_fn, x, u=None, dt=0.01)` | 单步显式四阶龙格-库塔积分：`rhs_fn(x)` 或 `rhs_fn(x, u)` 返回状态导数 `dx/dt` | 下一时刻状态 `Expr` |
| `fo.odeint_rk4(rhs_fn, x0, u_seq=None, dt=0.01, n_steps=None)` | 编译为单个 C++ `ScanNode` 的多步 RK4 轨迹积分，图规模恒定 $O(1)$ | `(x_final, X_history)` |
| `fo.collocate_trapz(rhs_fn, X, U=None, dt=0.01)` | 二阶梯形直接配点缺陷残差矩阵 `(n_states, N-1)`，直接用于 `opti.subject_to(defects == 0)` | `Expr` |
| `fo.collocate_hermite_simpson(rhs_fn, X, U=None, dt=0.01)` | 四阶压缩 Hermite-Simpson 直接配点缺陷残差矩阵 `(n_states, N-1)` | `Expr` |
| `fo.implicit_solve(residual_fn, z_init, inputs, lb=None, ub=None, max_iter=50, tol=1e-11)` | C++ 内层阻尼牛顿法求解 $R(z^*, x)=0$，并通过一阶/二阶隐函数定理自动传播解析导数 | 隐式解 `Expr` |
| `fo.vmap(body_fn, inputs, in_axes=0, out_axis=0, name="vmap")` | 向量化批处理映射（编译为单个 OpenMP 并行 `MapNode`） | `Expr` |
| `fo.scan(step_fn, init_carry, xs, seq_axes=1, shared_inputs=(), name="scan")` | 时序递推与 BPTT 二阶海森回传（编译为单个 `ScanNode`） | `(final_carry, ys_stacked)` |

#### 10.4 黑盒算子 (`@blackbox` 与零 GIL C-ABI `CBlackBox`)

| 接口签名 | 参数说明与默认值 | 返回值 |
| :--- | :--- | :--- |
| `@fo.blackbox(shape=(1, 1), jacobian=None, hessian=None, hvp=None, batch=None, grad_inputs=None, jac_sparsity=None, fd_step=1e-7, fd_hessian=True, unpack_scalars=False, name="")` | Python 黑盒算子装饰器；`grad_inputs`: 仅对指定输入参数下标求导；`jac_sparsity`: 局部稀疏掩码矩阵 | `CustomOp` |
| `fo.CBlackBox.from_capsule(capsule, shape=(1, 1), *, fd_step=1e-7, fd_hessian=True, jac_sparsity=None, name="c_blackbox", keep_alive=None)` | 从封装了 `FastOptCBlackBox*` 的 `PyCapsule` 构建零 GIL C-ABI 黑盒算子 | `CBlackBox` |
| `fo.CBlackBox.from_pointers(eval_ptr, shape=(1, 1), *, jac_ptr=0, hvp_ptr=0, batch_ptr=0, user_data_ptr=0, is_thread_safe=True, fd_step=1e-7, fd_hessian=True, jac_sparsity=None, name="c_blackbox", keep_alive=None)` | 从原始函数指针地址 (`uintptr_t`) 或 `ctypes` 函数对象构建零 GIL C-ABI 黑盒算子 | `CBlackBox` |
| `fo.CBlackBox.from_dll(dll_path, eval_symbol, shape=(1, 1), *, jac_symbol=None, hvp_symbol=None, batch_symbol=None, user_data_ptr=0, is_thread_safe=True, fd_step=1e-7, fd_hessian=True, jac_sparsity=None, name="", keep_alive=None)` | 直接加载编译好的 `.dll` / `.so` / `.dylib` 动态库并绑定导出的 C-ABI 符号 | `CBlackBox` |
