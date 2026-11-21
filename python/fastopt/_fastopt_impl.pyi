"""
Type stub file (.pyi) for the compiled C++17 extension module `fastopt._fastopt_impl`.
Provides full parameter names, type hints, default values, and docstrings for IDEs
(VS Code Pylance/Pyright, PyCharm, mypy).
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray

class Shape:
    """2D tensor shape metadata `(rows, cols)` in column-major (Eigen) storage."""

    rows: int
    cols: int

    def __init__(self, rows: int = 1, cols: int = 1) -> None:
        """
        Initialize a 2D `Shape(rows, cols)`.

        Args:
            rows: Number of rows (default: 1).
            cols: Number of columns (default: 1).
        """
        ...
    def size(self) -> int:
        """Return total number of scalar elements (`rows * cols`)."""
        ...
    def is_scalar(self) -> bool:
        """Return `True` if `rows == 1 and cols == 1`."""
        ...
    def is_vector(self) -> bool:
        """Return `True` if `rows == 1 or cols == 1`."""
        ...
    def __repr__(self) -> str: ...

class OpType(Enum):
    """Enumeration of C++ computation graph node operator types."""

    Variable = ...
    Parameter = ...
    Constant = ...
    ImplicitState = ...
    Neg = ...
    Sin = ...
    Cos = ...
    Tan = ...
    Asin = ...
    Acos = ...
    Atan = ...
    Sinh = ...
    Cosh = ...
    Tanh = ...
    Exp = ...
    Log = ...
    Sqrt = ...
    Abs = ...
    Square = ...
    Add = ...
    Sub = ...
    Mul = ...
    Div = ...
    Pow = ...
    Atan2 = ...
    Lt = ...
    Le = ...
    Gt = ...
    Ge = ...
    Eq = ...
    MatMul = ...
    LinearSolve = ...
    Dot = ...
    Sum = ...
    KSLogSumExp = ...
    Transpose = ...
    Slice = ...
    Concat = ...
    Reshape = ...
    LinearInterp = ...
    BSplineEval = ...
    LinearInterp2D = ...
    BSplineEval2D = ...
    LinearInterp3D = ...
    BSplineEval3D = ...
    Where = ...
    SmoothWhere = ...
    Cond = ...
    ImplicitSolver = ...
    BlackBox = ...
    Map = ...
    Scan = ...
    MLP = ...
    Conv1D = ...
    Attention = ...
    Conv2D = ...
    Pool2D = ...
    RMSNorm = ...
    Surrogate = ...

class Graph:
    """C++17 dynamic tensor computation graph container with 3-segment contiguous memory layout."""

    def __init__(self) -> None:
        """Construct an empty computation graph."""
        ...
    def add_variable(
        self,
        shape: Shape,
        init: NDArray[np.float64],
        lb: NDArray[np.float64],
        ub: NDArray[np.float64],
        name: str = "",
    ) -> int:
        """
        Add an active decision variable node to the graph.

        Args:
            shape: 2D `Shape(rows, cols)` of the decision variable.
            init: Initial guess flattened in Fortran (column-major) order.
            lb: Lower bounds flattened in Fortran order.
            ub: Upper bounds flattened in Fortran order.
            name: Optional variable identifier name.

        Returns:
            Integer node ID in the computation graph.
        """
        ...
    def add_parameter(
        self,
        shape: Shape,
        val: NDArray[np.float64],
        name: str = "",
    ) -> int:
        """
        Add a mutable frozen parameter node (`requires_grad=False`) to the graph.

        Args:
            shape: 2D `Shape(rows, cols)` of the parameter.
            val: Initial numeric value flattened in Fortran order.
            name: Optional parameter identifier name.

        Returns:
            Integer node ID in the computation graph.
        """
        ...
    def add_constant(
        self,
        shape: Shape,
        val: NDArray[np.float64],
        name: str = "",
    ) -> int:
        """
        Add an immutable constant tensor node (`requires_grad=False`) to the graph.

        Args:
            shape: 2D `Shape(rows, cols)` of the constant.
            val: Constant numeric value flattened in Fortran order.
            name: Optional constant identifier name.

        Returns:
            Integer node ID in the computation graph.
        """
        ...
    def add_scalar_constant(self, val: float) -> int:
        """Add a scalar `(1, 1)` constant node with value `val` and return its node ID."""
        ...
    def add_implicit_state(
        self,
        shape: Shape,
        guess: NDArray[np.float64],
        lb: NDArray[np.float64],
        ub: NDArray[np.float64],
        name: str = "",
    ) -> int:
        """Add an inner implicit state node for `implicit_solve` residual sub-graphs."""
        ...
    def set_parameter_value(self, node_id: int, val: NDArray[np.float64]) -> None:
        """Update the numeric value of an existing `Parameter` node in-place without recompiling."""
        ...
    def get_parameter_value(self, node_id: int) -> NDArray[np.float64]:
        """Return the current flattened Fortran-order numeric vector of a `Parameter` node."""
        ...
    def set_variable_init(self, node_id: int, val: NDArray[np.float64]) -> None:
        """Update the initial guess vector of an existing `Variable` node in-place."""
        ...
    def get_variable_init(self, node_id: int) -> NDArray[np.float64]:
        """Return the current flattened initial guess vector of a `Variable` node."""
        ...
    def compile(self, output_ids: Sequence[int] = ...) -> None:
        """Compile dead-code-eliminated topological execution orders and 3-segment workspace offsets."""
        ...
    def num_nodes(self) -> int:
        """Return total number of nodes registered in the graph."""
        ...
    def total_var_size(self) -> int:
        """Return total scalar decision variable dimension `N`."""
        ...
    def total_memory_size(self) -> int:
        """Return total scalar workspace memory size across all 3 segments."""
        ...
    def is_compiled(self) -> bool:
        """Return `True` if `compile()` has been called since the last node addition."""
        ...
    def set_cse_enabled(self, enabled: bool) -> None:
        """Enable or disable Common Subexpression Elimination (CSE) and constant folding."""
        ...
    def cse_enabled(self) -> bool:
        """Return `True` if CSE and constant folding are enabled."""
        ...
    def cse_hit_count(self) -> int:
        """Return cumulative number of duplicate nodes/constants eliminated by CSE."""
        ...
    def has_unsafe_blackbox(self) -> bool:
        """Return `True` if this graph or any nested subgraph contains a non-thread-safe BlackBox node."""
        ...
    def variable_ids(self) -> List[int]:
        """Return list of all active `Variable` node IDs."""
        ...
    def parameter_ids(self) -> List[int]:
        """Return list of all `Parameter` node IDs."""
        ...
    def initial_decision_vector(self) -> NDArray[np.float64]:
        """Assemble the concatenated initial decision vector `x0` of length `total_var_size()`."""
        ...
    def lower_bounds_vector(self) -> NDArray[np.float64]:
        """Assemble the concatenated variable lower bounds vector `lb` of length `total_var_size()`."""
        ...
    def upper_bounds_vector(self) -> NDArray[np.float64]:
        """Assemble the concatenated variable upper bounds vector `ub` of length `total_var_size()`."""
        ...
    def node_shape(self, id: int) -> Shape:
        """Return the 2D `Shape` of node `id`."""
        ...
    def node_op(self, id: int) -> OpType:
        """Return the `OpType` of node `id`."""
        ...
    def node_requires_grad(self, id: int) -> bool:
        """Return `True` if node `id` depends on any active decision variable."""
        ...
    def node_is_param_only(self, id: int) -> bool:
        """Return `True` if node `id` depends only on constants and parameters."""
        ...
    def node_offset(self, id: int) -> int:
        """Return the workspace memory offset of node `id` (valid after `compile()`)."""
        ...

class Evaluator:
    """
    High-performance C++17 0th/1st/2nd-order AD evaluator with same-point forward deduplication,
    64-bit bitset symbolic sparsity propagation, Largest-Degree-First graph coloring, and OpenMP parallelism.
    """

    def __init__(self, graph: Graph, output_ids: Sequence[int]) -> None:
        """
        Compile `graph` for `output_ids` and initialize tiered AD workspaces.

        Args:
            graph: C++ `Graph` instance.
            output_ids: Sequence of output node IDs (e.g., `[obj_id, eq_id, ineq_id]`).
        """
        ...
    def num_vars(self) -> int:
        """Return total number of scalar decision variables `N`."""
        ...
    def num_outputs(self) -> int:
        """Return number of registered output nodes."""
        ...
    def output_size(self, out_idx: int) -> int:
        """Return scalar element count of output `out_idx`."""
        ...
    def output_shape(self, out_idx: int) -> Shape:
        """Return 2D `Shape(rows, cols)` of output `out_idx`."""
        ...
    def forward_eval_count(self) -> int:
        """Return cumulative count of dynamic forward sweeps executed on the main workspace."""
        ...
    def reset_forward_eval_count(self) -> None:
        """Reset `forward_eval_count()` counter to zero."""
        ...
    def invalidate_forward_cache(self) -> None:
        """Force the next evaluation call to re-run the dynamic forward pass."""
        ...
    def eval(self, x: NDArray[np.float64]) -> List[NDArray[np.float64]]:
        """
        Evaluate all registered outputs at decision vector `x` (with same-point deduplication).

        Args:
            x: 1D decision vector of shape `(num_vars,)`.

        Returns:
            List of 2D NumPy arrays corresponding to each output node.
        """
        ...
    def eval_node(self, x: NDArray[np.float64], node_id: int) -> NDArray[np.float64]:
        """Evaluate any arbitrary node `node_id` in the graph at decision vector `x`."""
        ...
    def eval_batch(
        self,
        X_pop: NDArray[np.float64],
        num_threads: int = 0,
    ) -> List[NDArray[np.float64]]:
        """
        Evaluate a population matrix `X_pop` of shape `(n_pop, num_vars)` in parallel via OpenMP.

        Args:
            X_pop: 2D array of shape `(n_pop, num_vars)` where each row is a candidate decision vector.
            num_threads: Number of OpenMP threads (`0` = auto-detect all available CPU cores).

        Returns:
            List of 2D arrays where output `k` has shape `(n_pop, output_size(k))`.
        """
        ...
    def gradient(self, x: NDArray[np.float64], out_idx: int = 0) -> NDArray[np.float64]:
        """
        Compute exact reverse-mode AD gradient `df/dx` of scalar output `out_idx` at `x`.

        Args:
            x: 1D decision vector of shape `(num_vars,)`.
            out_idx: Output index (default: `0`). Must be a scalar output (`output_size(out_idx) == 1`).

        Returns:
            1D gradient vector of shape `(num_vars,)`.
        """
        ...
    def jacobian(
        self,
        x: NDArray[np.float64],
        out_idx: int = 0,
        num_threads: int = 0,
    ) -> NDArray[np.float64]:
        """
        Compute dense Jacobian matrix `J` of shape `(output_size(out_idx), num_vars)` using graph-colored AD.

        Args:
            x: 1D decision vector of shape `(num_vars,)`.
            out_idx: Output index (default: `0`).
            num_threads: Number of OpenMP threads (`0` = auto).

        Returns:
            2D Jacobian matrix of shape `(M, N)`.
        """
        ...
    def jacobian_sparsity(
        self, out_idx: int = 0
    ) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Return `(rows, cols)` 0-based COO sparsity pattern arrays for output `out_idx`."""
        ...
    def num_jacobian_colors(self, out_idx: int = 0) -> int:
        """Return number of graph-colored directional sweeps required to evaluate Jacobian `out_idx`."""
        ...
    def sparse_jacobian_values(
        self,
        x: NDArray[np.float64],
        out_idx: int = 0,
        num_threads: int = 0,
    ) -> NDArray[np.float64]:
        """Evaluate non-zero Jacobian entries matching `jacobian_sparsity(out_idx)` via graph coloring."""
        ...
    def hessian_vec_prod(
        self,
        x: NDArray[np.float64],
        obj_scale: float,
        constraint_multipliers: Sequence[NDArray[np.float64]],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute exact Forward-over-Reverse Lagrangian Hessian-vector product `H(x, lambda) @ v`.

        Args:
            x: 1D decision vector of shape `(num_vars,)`.
            obj_scale: Scalar multiplier sigma for output `0` (objective).
            constraint_multipliers: List of 1D dual multiplier vectors for outputs `1, 2, ...`.
            v: 1D tangent perturbation vector of shape `(num_vars,)`.

        Returns:
            1D Hessian-vector product of shape `(num_vars,)`.
        """
        ...
    def lagrangian_hessian(
        self,
        x: NDArray[np.float64],
        obj_scale: float,
        constraint_multipliers: Sequence[NDArray[np.float64]],
        num_threads: int = 0,
    ) -> NDArray[np.float64]:
        """Compute dense symmetric `N x N` Lagrangian Hessian matrix at `(x, obj_scale, constraint_multipliers)`."""
        ...
    def hessian(
        self,
        x: NDArray[np.float64],
        out_idx: int = 0,
        num_threads: int = 0,
    ) -> NDArray[np.float64]:
        """Compute dense `N x N` Hessian matrix of scalar output `out_idx` at `x`."""
        ...
    def hessian_sparsity(self) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Return lower-triangular (`row >= col`) 0-based COO sparsity pattern `(tril_rows, tril_cols)`."""
        ...
    def num_hessian_colors(self) -> int:
        """Return number of graph-colored HVP sweeps required to assemble the sparse Lagrangian Hessian."""
        ...
    def sparse_hessian_values(
        self,
        x: NDArray[np.float64],
        obj_scale: float,
        constraint_multipliers: Sequence[NDArray[np.float64]],
        num_threads: int = 0,
    ) -> NDArray[np.float64]:
        """Evaluate non-zero lower-triangular Lagrangian Hessian values matching `hessian_sparsity()`."""
        ...
    def parameter_sensitivities(
        self,
        x: NDArray[np.float64],
        param_ids: Sequence[int],
        obj_scale: float,
        constraint_multipliers: Sequence[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[NDArray[np.float64]]]:
        """
        Compute exact 1st- and 2nd-order parameter derivatives `(dL_dp, d2L_dxp, dout_dp)` at `x`.

        Args:
            x: Optimal decision vector of shape `(num_vars,)`.
            param_ids: Sequence of `Parameter` node IDs to differentiate with respect to.
            obj_scale: Objective multiplier (typically `1.0`).
            constraint_multipliers: List of KKT dual multiplier vectors for equality/inequality outputs.

        Returns:
            Tuple `(dL_dp, d2L_dxp, dout_dp)` where `dL_dp` has shape `(n_p,)`,
            `d2L_dxp` has shape `(num_vars, n_p)`, and `dout_dp[k]` has shape `(output_size(k), n_p)`.
        """
        ...

class IpoptResult:
    """Result container returned by the native C++17 `solve_ipopt_cpp` driver."""

    @property
    def x(self) -> NDArray[np.float64]: ...
    @property
    def mult_g(self) -> NDArray[np.float64]: ...
    @property
    def mult_x_L(self) -> NDArray[np.float64]: ...
    @property
    def mult_x_U(self) -> NDArray[np.float64]: ...
    @property
    def obj_val(self) -> float: ...
    @property
    def status(self) -> int: ...
    @property
    def success(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def elapsed_time(self) -> float: ...
    @property
    def message(self) -> str: ...
    @property
    def hist_iter(self) -> List[int]: ...
    @property
    def hist_obj(self) -> List[float]: ...
    @property
    def hist_inf_pr(self) -> List[float]: ...
    @property
    def hist_inf_du(self) -> List[float]: ...
    @property
    def hist_mu(self) -> List[float]: ...
    @property
    def hist_alpha_pr(self) -> List[float]: ...

def solve_ipopt_cpp(
    evaluator: Evaluator,
    x0: NDArray[np.float64],
    x_L: NDArray[np.float64],
    x_U: NDArray[np.float64],
    g_L: NDArray[np.float64],
    g_U: NDArray[np.float64],
    has_eq: bool,
    has_ineq: bool,
    use_exact_hessian: bool = True,
    str_options: Dict[str, str] = ...,
    num_options: Dict[str, float] = ...,
    int_options: Dict[str, int] = ...,
    lib_search_dir: str = "",
    iter_callback: Optional[Callable[[int, float, float, float, float, float], Optional[bool]]] = None,
) -> IpoptResult:
    """
    Solve an NLP in pure C++17 via COIN-OR IPOPT (`IpStdCInterface`) with zero Python callback overhead.

    Args:
        evaluator: Compiled `Evaluator` with outputs `[obj, (eq), (ineq)]`.
        x0: Initial decision vector of shape `(n,)`.
        x_L: Decision variable lower bounds of shape `(n,)`.
        x_U: Decision variable upper bounds of shape `(n,)`.
        g_L: Constraint lower bounds of shape `(m,)`.
        g_U: Constraint upper bounds of shape `(m,)`.
        has_eq: Whether output `1` is an equality constraint vector.
        has_ineq: Whether the last constraint output is an inequality constraint vector.
        use_exact_hessian: Use exact sparse Lagrangian Hessian (`True`) or L-BFGS (`False`).
        str_options: IPOPT string options (e.g. `{"linear_solver": "mumps"}`).
        num_options: IPOPT floating-point options (e.g. `{"tol": 1e-8}`).
        int_options: IPOPT integer options (e.g. `{"max_iter": 500, "print_level": 0}`).
        lib_search_dir: Optional directory path to search for `libipopt`.

    Returns:
        `IpoptResult` struct containing primal/dual solutions and solver statistics.
    """
    ...

def op_unary(graph: Graph, op: OpType, a: int) -> int: ...
def op_binary(graph: Graph, op: OpType, a: int, b: int) -> int: ...
def op_matmul(graph: Graph, a: int, b: int) -> int: ...
def op_linear_solve(graph: Graph, a: int, b: int) -> int: ...
def op_dot(graph: Graph, a: int, b: int) -> int: ...
def op_sum(graph: Graph, a: int, axis: int = -1) -> int: ...
def op_ks_logsumexp(
    graph: Graph, a: int, rho: float = 50.0, axis: int = -1, subtract_log_n: bool = False
) -> int: ...
def op_transpose(graph: Graph, a: int) -> int: ...
def op_slice(graph: Graph, a: int, r0: int, r1: int, c0: int, c1: int) -> int: ...
def op_concat(graph: Graph, inputs: Sequence[int], axis: int = 0) -> int: ...
def op_reshape(graph: Graph, a: int, new_shape: Shape) -> int: ...
def op_linear_interp(
    graph: Graph,
    x_query_id: int,
    y_data_id: int,
    x_grid: NDArray[np.float64],
    extrapolate: bool = False,
) -> int: ...
def op_bspline_eval(
    graph: Graph,
    u_query_id: int,
    ctrl_pts_id: int,
    knots: NDArray[np.float64],
    degree: int = 3,
    extrapolate: bool = False,
) -> int: ...
def op_linear_interp_2d(
    graph: Graph,
    x_query_id: int,
    y_query_id: int,
    z_data_id: int,
    x_grid: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    extrapolate: bool = False,
) -> int: ...
def op_bspline_eval_2d(
    graph: Graph,
    u_query_id: int,
    v_query_id: int,
    ctrl_pts_id: int,
    knots_u: NDArray[np.float64],
    knots_v: NDArray[np.float64],
    degree_u: int = 3,
    degree_v: int = 3,
    extrapolate: bool = False,
) -> int: ...
def op_linear_interp_3d(
    graph: Graph,
    x_query_id: int,
    y_query_id: int,
    z_query_id: int,
    v_data_id: int,
    x_grid: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    extrapolate: bool = False,
) -> int: ...
def op_bspline_eval_3d(
    graph: Graph,
    u_query_id: int,
    v_query_id: int,
    w_query_id: int,
    ctrl_pts_id: int,
    knots_u: NDArray[np.float64],
    knots_v: NDArray[np.float64],
    knots_w: NDArray[np.float64],
    degree_u: int = 3,
    degree_v: int = 3,
    degree_w: int = 3,
    extrapolate: bool = False,
) -> int: ...
def op_where(graph: Graph, cond_id: int, true_id: int, false_id: int) -> int: ...
def op_smooth_where(
    graph: Graph,
    x_id: int,
    true_id: int,
    false_id: int,
    threshold: float = 0.0,
    sharpness: float = 10.0,
) -> int: ...
def op_cond_subgraph(
    graph: Graph,
    cond_id: int,
    arg_ids: Sequence[int],
    true_graph: Graph,
    true_out_id: int,
    false_graph: Graph,
    false_out_id: int,
) -> int: ...
def op_implicit_solve(
    graph: Graph,
    outer_input_ids: Sequence[int],
    res_graph: Graph,
    res_out_id: int,
    inner_state_var_id: int,
    inner_param_ids: Sequence[int],
    max_iter: int = 50,
    tol: float = 1e-11,
) -> int: ...
def op_blackbox(
    graph: Graph,
    out_shape: Shape,
    in_ids: Sequence[int],
    eval_fn: Callable[..., Any],
    jac_fn: Optional[Callable[..., Any]] = None,
    hvp_fn: Optional[Callable[..., Any]] = None,
    fd_step: float = 1e-7,
    fd_hessian: bool = True,
    name: str = "",
    batch_eval_fn: Optional[Callable[..., Any]] = None,
    jac_sparsity_mask: NDArray[np.int32] = ...,
) -> int: ...
def op_c_blackbox_capsule(
    graph: Graph,
    out_shape: Shape,
    in_ids: Sequence[int],
    capsule: Any,
    fd_step: float = 1e-7,
    fd_hessian: bool = True,
    jac_sparsity_mask: NDArray[np.int32] = ...,
    name: str = "",
) -> int: ...
def op_c_blackbox_raw(
    graph: Graph,
    out_shape: Shape,
    in_ids: Sequence[int],
    eval_ptr: int,
    jac_ptr: int = 0,
    hvp_ptr: int = 0,
    batch_ptr: int = 0,
    user_data_ptr: int = 0,
    is_thread_safe: bool = True,
    fd_step: float = 1e-7,
    fd_hessian: bool = True,
    jac_sparsity_mask: NDArray[np.int32] = ...,
    name: str = "",
    keep_alive: Any = None,
) -> int: ...
def op_map_subgraph(
    graph: Graph,
    out_shape: Shape,
    outer_in_ids: Sequence[int],
    body_graph: Graph,
    body_out_id: int,
    inner_var_ids: Sequence[int],
    in_axes: Sequence[int],
    out_axis: int,
    num_batches: int,
    name: str = "",
) -> int: ...
def op_scan_subgraph(
    graph: Graph,
    out_shape: Shape,
    outer_in_ids: Sequence[int],
    step_graph: Graph,
    inner_carry_id: int,
    inner_seq_ids: Sequence[int],
    seq_axes: Sequence[int],
    inner_shared_ids: Sequence[int],
    next_carry_out_id: int,
    step_y_out_id: int,
    num_steps: int,
    name: str = "",
) -> int: ...


class NNActivation(Enum):
    Identity: NNActivation
    Tanh: NNActivation
    SiLU: NNActivation
    GELU: NNActivation
    ReLU: NNActivation
    Softplus: NNActivation
    ELU: NNActivation
    Sin: NNActivation
    Sigmoid: NNActivation
    LeakyReLU: NNActivation
    Mish: NNActivation
    SELU: NNActivation
    HardSwish: NNActivation

class MLPLayerSpec:
    W: NDArray[np.float64]
    b: NDArray[np.float64]
    activation: NNActivation
    residual: bool
    layer_norm: bool
    ln_gamma: NDArray[np.float64]
    ln_beta: NDArray[np.float64]
    ln_eps: float
    def __init__(self) -> None: ...

class CompiledMLP:
    def __init__(
        self,
        layers: Sequence[MLPLayerSpec],
        use_float32: bool = False,
        in_mean: NDArray[np.float64] = ...,
        in_inv_std: NDArray[np.float64] = ...,
        out_mean: NDArray[np.float64] = ...,
        out_std: NDArray[np.float64] = ...,
        name: str = "",
    ) -> None: ...
    def d_in(self) -> int: ...
    def d_out(self) -> int: ...
    def num_layers(self) -> int: ...
    def use_float32(self) -> bool: ...
    def eval(self, X: NDArray[np.float64], num_threads: int = 0) -> NDArray[np.float64]: ...

def op_mlp(graph: Graph, x_id: int, model: CompiledMLP, name: str = "") -> int: ...

class CompiledConv1D:
    def __init__(
        self,
        W: NDArray[np.float64],
        b: NDArray[np.float64],
        c_in: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        activation: NNActivation = ...,
        use_float32: bool = False,
        name: str = "",
    ) -> None: ...
    def c_in(self) -> int: ...
    def c_out(self) -> int: ...
    def kernel_size(self) -> int: ...
    def stride(self) -> int: ...
    def padding(self) -> int: ...
    def dilation(self) -> int: ...
    def use_float32(self) -> bool: ...
    def eval(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...

def op_conv1d(graph: Graph, x_id: int, model: CompiledConv1D, name: str = "") -> int: ...

class CompiledAttention:
    def __init__(
        self,
        Wq: NDArray[np.float64], bq: NDArray[np.float64],
        Wk: NDArray[np.float64], bk: NDArray[np.float64],
        Wv: NDArray[np.float64], bv: NDArray[np.float64],
        Wo: NDArray[np.float64], bo: NDArray[np.float64],
        num_heads: int = 1,
        causal: bool = False,
        residual: bool = False,
        use_float32: bool = False,
        name: str = "",
    ) -> None: ...
    def d_model(self) -> int: ...
    def num_heads(self) -> int: ...
    def causal(self) -> bool: ...
    def residual(self) -> bool: ...
    def use_float32(self) -> bool: ...
    def eval(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...

def op_attention(graph: Graph, x_id: int, model: CompiledAttention, name: str = "") -> int: ...

class CompiledConv2D:
    def __init__(
        self,
        W: NDArray[np.float64],
        b: NDArray[np.float64],
        c_in: int,
        kh: int,
        kw: int,
        sh: int = 1,
        sw: int = 1,
        ph: int = 0,
        pw: int = 0,
        dh: int = 1,
        dw: int = 1,
        activation: NNActivation = ...,
        use_float32: bool = False,
        name: str = "",
    ) -> None: ...
    def c_in(self) -> int: ...
    def c_out(self) -> int: ...
    def out_hw(self, h_in: int, w_in: int) -> Tuple[int, int]: ...
    def eval(self, X: NDArray[np.float64], h_in: int, w_in: int) -> NDArray[np.float64]: ...

def op_conv2d(graph: Graph, x_id: int, h_in: int, w_in: int, model: CompiledConv2D, name: str = "") -> int: ...

class Pool2DMode(Enum):
    Avg: Pool2DMode
    Max: Pool2DMode

def op_pool2d(
    graph: Graph,
    x_id: int,
    mode: Pool2DMode,
    h_in: int,
    w_in: int,
    kh: int,
    kw: int,
    sh: int = 2,
    sw: int = 2,
    ph: int = 0,
    pw: int = 0,
    name: str = "",
) -> int: ...

def op_rms_norm(
    graph: Graph,
    x_id: int,
    gamma: NDArray[np.float64] = ...,
    eps: float = 1e-6,
    name: str = "",
) -> int: ...

class SurrogateKernel(Enum):
    Gaussian: SurrogateKernel
    Multiquadric: SurrogateKernel
    InvMultiquadric: SurrogateKernel
    Cubic: SurrogateKernel
    ThinPlateSpline: SurrogateKernel
    Matern32: SurrogateKernel
    Matern52: SurrogateKernel

class SurrogateOutputMode(Enum):
    Mean: SurrogateOutputMode
    Variance: SurrogateOutputMode

class CompiledSurrogate:
    def __init__(
        self,
        kernel: SurrogateKernel,
        poly_degree: int,
        X_centers: NDArray[np.float64],
        theta: NDArray[np.float64],
        alpha: NDArray[np.float64],
        beta_grad: NDArray[np.float64] = ...,
        poly_coef: NDArray[np.float64] = ...,
        in_mean: NDArray[np.float64] = ...,
        in_inv_std: NDArray[np.float64] = ...,
        out_mean: NDArray[np.float64] = ...,
        out_std: NDArray[np.float64] = ...,
        sigma2: NDArray[np.float64] = ...,
        M_kk: NDArray[np.float64] = ...,
        M_kp: NDArray[np.float64] = ...,
        M_pp: NDArray[np.float64] = ...,
        r_eps: float = 1e-14,
        name: str = "",
    ) -> None: ...
    def d_in(self) -> int: ...
    def d_out(self) -> int: ...
    def n_centers(self) -> int: ...
    def has_grad_basis(self) -> bool: ...
    def has_variance(self) -> bool: ...
    def eval(self, X: NDArray[np.float64], num_threads: int = 0) -> NDArray[np.float64]: ...
    def eval_variance(self, X: NDArray[np.float64], num_threads: int = 0) -> NDArray[np.float64]: ...

def op_surrogate(
    graph: Graph,
    x_id: int,
    model: CompiledSurrogate,
    mode: SurrogateOutputMode = ...,
    name: str = "",
) -> int: ...
