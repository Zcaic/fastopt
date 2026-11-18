"""
Python expression wrapper, lazy ComparisonExpr, interpolation helpers,
structured loop primitives (vmap and scan), and GraphBuilder for fastopt.
All comments are written in English.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import sparse as _sp_sparse
from scipy.interpolate import BSpline

from . import _fastopt_impl as _core

ShapeLike = Union[int, Tuple[int, ...], _core.Shape]
ArrayLike = Union[float, int, np.ndarray, Sequence[float]]


def _normalize_shape(shape: ShapeLike) -> _core.Shape:
    """Convert an integer, 1-tuple, 2-tuple, or Shape into a 2D C++ Shape(rows, cols)."""
    if isinstance(shape, _core.Shape):
        return shape
    if isinstance(shape, (int, np.integer)):
        return _core.Shape(int(shape), 1)
    if isinstance(shape, (tuple, list)):
        if len(shape) == 0:
            return _core.Shape(1, 1)
        if len(shape) == 1:
            return _core.Shape(int(shape[0]), 1)
        if len(shape) == 2:
            return _core.Shape(int(shape[0]), int(shape[1]))
    raise ValueError(f"Unsupported shape specification: {shape}")


def _to_fortran_flat(val: ArrayLike, shape: _core.Shape, default_val: float = 0.0) -> np.ndarray:
    """Flatten scalar or array in column-major (Fortran/Eigen) order matching Shape."""
    n = shape.size()
    if val is None:
        return np.full(n, default_val, dtype=np.float64)
    arr = np.asarray(val, dtype=np.float64)
    if arr.size == 1:
        return np.full(n, float(arr.item()), dtype=np.float64)
    if arr.size != n:
        raise ValueError(f"Array size {arr.size} does not match target shape ({shape.rows}, {shape.cols}).")
    return np.asfortranarray(arr.reshape(shape.rows, shape.cols)).ravel(order="F")


def _find_graph(*items) -> _core.Graph:
    """Find the underlying C++ Graph from the first Expr argument."""
    for item in items:
        if isinstance(item, Expr):
            return item.graph
    raise ValueError("At least one argument must be a fastopt Expr.")


def _coerce_with_graph(g: _core.Graph, val: Union[Expr, ArrayLike]) -> Expr:
    """Convert a Python scalar/array or Expr into an Expr on Graph g."""
    if isinstance(val, Expr):
        if val.graph is not g:
            raise ValueError("Cannot combine Expr objects belonging to different Graph instances.")
        return val
    if isinstance(val, (int, float, np.number)):
        cid = g.add_scalar_constant(float(val))
        return Expr(g, cid)
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim == 0:
        cid = g.add_scalar_constant(float(arr.item()))
        return Expr(g, cid)
    if arr.ndim == 1:
        s = _core.Shape(arr.shape[0], 1)
    elif arr.ndim == 2:
        s = _core.Shape(arr.shape[0], arr.shape[1])
    else:
        raise ValueError("Only 1D or 2D NumPy arrays are supported as tensor constants.")
    flat = np.asfortranarray(arr.reshape(s.rows, s.cols)).ravel(order="F")
    cid = g.add_constant(s, flat, "")
    return Expr(g, cid)


class Expr:
    """Symbolic tensor expression node backed by the C++17 fastopt::Graph."""

    __array_priority__ = 1000.0

    def __init__(self, graph: _core.Graph, node_id: int) -> None:
        """Wrap node `node_id` in C++ computation graph `graph` as a symbolic `Expr`."""
        self._graph = graph
        self._id = int(node_id)

    def __hash__(self) -> int:
        return hash((id(self._graph), self._id))

    @property
    def id(self) -> int:
        """Integer node ID in the underlying C++ `Graph`."""
        return self._id

    @property
    def graph(self) -> _core.Graph:
        """Underlying C++ `Graph` instance."""
        return self._graph

    @property
    def shape(self) -> Tuple[int, int]:
        """2D tensor shape `(rows, cols)`."""
        s = self._graph.node_shape(self._id)
        return (s.rows, s.cols)

    @property
    def size(self) -> int:
        """Total number of scalar elements (`rows * cols`)."""
        return self._graph.node_shape(self._id).size()

    @property
    def op_type(self) -> _core.OpType:
        """C++ operator type (`OpType` enum)."""
        return self._graph.node_op(self._id)

    @property
    def requires_grad(self) -> bool:
        """`True` if this expression depends on any active decision variable (`freeze=False`)."""
        return self._graph.node_requires_grad(self._id)

    @property
    def is_param_only(self) -> bool:
        """`True` if this expression depends only on constants and frozen parameters."""
        return self._graph.node_is_param_only(self._id)

    @property
    def is_frozen(self) -> bool:
        """`True` if this expression is a frozen `Parameter` leaf node."""
        return self.op_type == _core.OpType.Parameter

    @property
    def T(self) -> Expr:
        """Return the 2D matrix transpose `(cols, rows)` of this expression."""
        return Expr(self._graph, _core.op_transpose(self._graph, self._id))

    def _coerce(self, other: Union[Expr, ArrayLike]) -> Expr:
        return _coerce_with_graph(self._graph, other)

    # Prevent accidental Python boolean evaluation on symbolic expressions
    def __bool__(self) -> bool:
        raise TypeError(
            "Symbolic Expr cannot be evaluated as a Python boolean in 'if/else'. "
            "Use fastopt.where(cond, true_val, false_val), fastopt.smooth_where(...), "
            "or fastopt.cond(...) for variable-dependent branching."
        )

    # Unary operators
    def __neg__(self) -> Expr:
        return Expr(self._graph, _core.op_unary(self._graph, _core.OpType.Neg, self._id))

    def __abs__(self) -> Expr:
        return Expr(self._graph, _core.op_unary(self._graph, _core.OpType.Abs, self._id))

    # Binary arithmetic operators
    def __add__(self, other: Union[Expr, ArrayLike]) -> Expr:
        rhs = self._coerce(other)
        return Expr(self._graph, _core.op_binary(self._graph, _core.OpType.Add, self._id, rhs.id))

    def __radd__(self, other: Union[Expr, ArrayLike]) -> Expr:
        lhs = self._coerce(other)
        return Expr(self._graph, _core.op_binary(self._graph, _core.OpType.Add, lhs.id, self._id))

    def __sub__(self, other: Union[Expr, ArrayLike]) -> Expr:
        rhs = self._coerce(other)
        return Expr(self._graph, _core.op_binary(self._graph, _core.OpType.Sub, self._id, rhs.id))

    def __rsub__(self, other: Union[Expr, ArrayLike]) -> Expr:
        lhs = self._coerce(other)
        return Expr(self._graph, _core.op_binary(self._graph, _core.OpType.Sub, lhs.id, self._id))

    def __mul__(self, other: Union[Expr, ArrayLike]) -> Expr:
        rhs = self._coerce(other)
        return Expr(self._graph, _core.op_binary(self._graph, _core.OpType.Mul, self._id, rhs.id))

    def __rmul__(self, other: Union[Expr, ArrayLike]) -> Expr:
        lhs = self._coerce(other)
        return Expr(self._graph, _core.op_binary(self._graph, _core.OpType.Mul, lhs.id, self._id))

    def __truediv__(self, other: Union[Expr, ArrayLike]) -> Expr:
        rhs = self._coerce(other)
        return Expr(self._graph, _core.op_binary(self._graph, _core.OpType.Div, self._id, rhs.id))

    def __rtruediv__(self, other: Union[Expr, ArrayLike]) -> Expr:
        lhs = self._coerce(other)
        return Expr(self._graph, _core.op_binary(self._graph, _core.OpType.Div, lhs.id, self._id))

    def __pow__(self, power: Union[Expr, float, int]) -> Expr:
        if isinstance(power, (int, float)) and float(power) == 2.0:
            return Expr(self._graph, _core.op_unary(self._graph, _core.OpType.Square, self._id))
        rhs = self._coerce(power)
        return Expr(self._graph, _core.op_binary(self._graph, _core.OpType.Pow, self._id, rhs.id))

    def __rpow__(self, base: Union[Expr, float, int]) -> Expr:
        lhs = self._coerce(base)
        return Expr(self._graph, _core.op_binary(self._graph, _core.OpType.Pow, lhs.id, self._id))

    # Matrix multiplication operator (@)
    def __matmul__(self, other: Union[Expr, ArrayLike]) -> Expr:
        rhs = self._coerce(other)
        return Expr(self._graph, _core.op_matmul(self._graph, self._id, rhs.id))

    def __rmatmul__(self, other: Union[Expr, ArrayLike]) -> Expr:
        lhs = self._coerce(other)
        return Expr(self._graph, _core.op_matmul(self._graph, lhs.id, self._id))

    # Lazy comparison operators: zero C++ overhead when used in Opti.subject_to()
    def __lt__(self, other: Union[Expr, ArrayLike]) -> ComparisonExpr:
        return ComparisonExpr(self, "<", other)

    def __le__(self, other: Union[Expr, ArrayLike]) -> ComparisonExpr:
        return ComparisonExpr(self, "<=", other)

    def __gt__(self, other: Union[Expr, ArrayLike]) -> ComparisonExpr:
        return ComparisonExpr(self, ">", other)

    def __ge__(self, other: Union[Expr, ArrayLike]) -> ComparisonExpr:
        return ComparisonExpr(self, ">=", other)

    def __eq__(self, other: Union[Expr, ArrayLike]) -> ComparisonExpr:  # type: ignore[override]
        return ComparisonExpr(self, "==", other)

    # Indexing and slicing: x[i], x[r0:r1], x[r0:r1:step], x[[i0, i1, ...]], x[bool_mask]
    def __getitem__(self, key: Any) -> Expr:
        """
        Index or slice this 2D symbolic tensor expression.
        Supports integer indices `x[i]`, contiguous slices `x[r0:r1, c0:c1]`,
        strided/reversed slices `x[::2]`, `x[::-1]`, integer fancy lists `x[[0, 3, 5]]`, and boolean masks.
        """
        rows, cols = self.shape

        def _parse_dim_runs(idx, max_len: int) -> List[Tuple[int, int]]:
            if isinstance(idx, (int, np.integer)):
                i = int(idx)
                if i < 0:
                    i += max_len
                if i < 0 or i >= max_len:
                    raise IndexError(f"Index {idx} out of bounds for dimension of size {max_len}.")
                return [(i, i + 1)]
            if isinstance(idx, slice):
                start, stop, step = idx.indices(max_len)
                if step == 1:
                    if start >= stop:
                        raise IndexError("Empty slice is not supported on Expr.")
                    return [(start, stop)]
                indices = list(range(start, stop, step))
            elif isinstance(idx, (list, tuple, np.ndarray)):
                arr = np.asarray(idx)
                if arr.dtype == bool:
                    if arr.ndim != 1 or arr.size != max_len:
                        raise IndexError(f"Boolean index mask length ({arr.size}) must match dimension ({max_len}).")
                    indices = [int(i) for i in np.flatnonzero(arr)]
                else:
                    indices = []
                    for raw_i in arr.ravel():
                        i = int(raw_i)
                        if i < 0:
                            i += max_len
                        if i < 0 or i >= max_len:
                            raise IndexError(f"Index {raw_i} out of bounds for dimension of size {max_len}.")
                        indices.append(i)
            else:
                raise TypeError(f"Unsupported slice index type: {type(idx)}")

            if not indices:
                raise IndexError("Empty index selection is not supported on Expr.")
            # Group consecutive ascending indices into contiguous [start, end) runs
            runs: List[Tuple[int, int]] = []
            run_s = indices[0]
            run_e = run_s + 1
            for k in indices[1:]:
                if k == run_e:
                    run_e += 1
                else:
                    runs.append((run_s, run_e))
                    run_s = k
                    run_e = k + 1
            runs.append((run_s, run_e))
            return runs

        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Only 1D or 2D indexing is supported on Expr.")
            r_runs = _parse_dim_runs(key[0], rows)
            c_runs = _parse_dim_runs(key[1], cols)
        else:
            if rows == 1 and cols > 1:
                r_runs = [(0, 1)]
                c_runs = _parse_dim_runs(key, cols)
            else:
                r_runs = _parse_dim_runs(key, rows)
                c_runs = [(0, cols)]

        # Slice along rows first
        if len(r_runs) == 1 and r_runs[0] == (0, rows):
            row_sliced_id = self._id
        elif len(r_runs) == 1:
            r0, r1 = r_runs[0]
            if len(c_runs) == 1:
                c0, c1 = c_runs[0]
                return Expr(self._graph, _core.op_slice(self._graph, self._id, r0, r1, c0, c1))
            row_sliced_id = _core.op_slice(self._graph, self._id, r0, r1, 0, cols)
        else:
            r_blocks = [_core.op_slice(self._graph, self._id, r0, r1, 0, cols) for r0, r1 in r_runs]
            row_sliced_id = _core.op_concat(self._graph, r_blocks, 0)

        # Slice along cols if needed
        new_rows = sum(r1 - r0 for r0, r1 in r_runs)
        if len(c_runs) == 1 and c_runs[0] == (0, cols):
            return Expr(self._graph, row_sliced_id)
        if len(c_runs) == 1:
            c0, c1 = c_runs[0]
            return Expr(self._graph, _core.op_slice(self._graph, row_sliced_id, 0, new_rows, c0, c1))
        c_blocks = [_core.op_slice(self._graph, row_sliced_id, 0, new_rows, c0, c1) for c0, c1 in c_runs]
        return Expr(self._graph, _core.op_concat(self._graph, c_blocks, 1))

    # Convenience tensor reduction and shape methods
    def sum(self, axis: Optional[int] = None) -> Expr:
        """Sum tensor elements over all entries (`axis=None`), down rows (`axis=0`), or across columns (`axis=1`)."""
        ax = -1 if axis is None else int(axis)
        return Expr(self._graph, _core.op_sum(self._graph, self._id, ax))

    def mean(self, axis: Optional[int] = None) -> Expr:
        """Compute arithmetic mean over all entries (`axis=None`), down rows (`axis=0`), or across columns (`axis=1`)."""
        return mean(self, axis=axis)

    def sumsqr(self) -> Expr:
        """Compute sum of squares `sum(x_i^2)` (`dot(x, x)`)."""
        return sumsqr(self)

    def norm(self, ord: int = 2, axis: Optional[int] = None, eps: float = 0.0) -> Expr:
        """Compute vector/matrix norm (`ord=2` Euclidean/Frobenius or `ord=1` L1 norm) with optional smoothing `eps`."""
        return norm(self, ord=ord, axis=axis, eps=eps)

    def ks_max(self, rho: float = 50.0, axis: Optional[int] = None, conservative: bool = False) -> Expr:
        """Compute smooth Kreisselmeier-Steinhauser maximum with aggregation parameter `rho`."""
        return ks_max(self, rho=rho, axis=axis, conservative=conservative)

    def ks_min(self, rho: float = 50.0, axis: Optional[int] = None, conservative: bool = False) -> Expr:
        """Compute smooth Kreisselmeier-Steinhauser minimum with aggregation parameter `rho`."""
        return ks_min(self, rho=rho, axis=axis, conservative=conservative)

    def reshape(self, rows: int, cols: int = 1) -> Expr:
        """Zero-copy column-major reshape to `(rows, cols)` where `rows * cols == self.size`."""
        return Expr(self._graph, _core.op_reshape(self._graph, self._id, _core.Shape(rows, cols)))

    def set_value(self, val: ArrayLike) -> None:
        """Update value of a frozen Parameter (or initial value of an active Variable)."""
        s = self._graph.node_shape(self._id)
        flat = _to_fortran_flat(val, s)
        if self.op_type == _core.OpType.Parameter:
            self._graph.set_parameter_value(self._id, flat)
        elif self.op_type in (_core.OpType.Variable, _core.OpType.ImplicitState):
            self._graph.set_variable_init(self._id, flat)
        else:
            raise TypeError("set_value() can only be called on Variable or Parameter leaf expressions.")

    def __repr__(self) -> str:
        return f"Expr(id={self._id}, op={self.op_type.name}, shape={self.shape}, grad={self.requires_grad})"


class ComparisonExpr(Expr):
    """
    Lazy symbolic comparison expression.
    - When passed to Opti.subject_to(), no C++ comparison mask node is created;
      instead, Opti extracts the continuous residual (lhs - rhs) directly.
    - When passed to where() or cond(), the C++ comparison mask node is materialized lazily.
    """

    _OP_MAP = {
        "<": _core.OpType.Lt,
        "<=": _core.OpType.Le,
        ">": _core.OpType.Gt,
        ">=": _core.OpType.Ge,
        "==": _core.OpType.Eq,
    }

    def __init__(self, lhs: Expr, op_str: str, rhs: Union[Expr, ArrayLike]) -> None:
        """Construct a lazy symbolic comparison expression `lhs op_str rhs`."""
        self._lhs = lhs
        self._op_str = op_str
        self._rhs_raw = rhs
        self._graph = lhs.graph
        self._materialized_id: Optional[int] = None

    @property
    def lhs(self) -> Expr:
        """Left-hand side symbolic `Expr`."""
        return self._lhs

    @property
    def op_str(self) -> str:
        """Comparison operator string (`'<='`, `'>='`, `'=='`, `'<'`, `'>'`)."""
        return self._op_str

    @property
    def rhs_raw(self) -> Union[Expr, ArrayLike]:
        """Right-hand side symbolic `Expr` or numeric array/scalar."""
        return self._rhs_raw

    @property
    def id(self) -> int:
        """Materialize and return the C++ comparison mask node ID (when used in `where`/`cond`)."""
        if self._materialized_id is None:
            rhs_expr = _coerce_with_graph(self._graph, self._rhs_raw)
            self._materialized_id = _core.op_binary(
                self._graph, self._OP_MAP[self._op_str], self._lhs.id, rhs_expr.id
            )
        return self._materialized_id

    def __bool__(self) -> bool:
        raise TypeError(
            "ComparisonExpr cannot be evaluated as a Python boolean in 'if/else'. "
            "Use fastopt.where(...) or opti.subject_to(...)."
        )

    def __repr__(self) -> str:
        return f"ComparisonExpr({self._lhs!r} {self._op_str} {self._rhs_raw!r})"


# ============================================================================
# Mathematical, Linear Algebra, Interpolation, Branching, Implicit & Loop Ops
# ============================================================================
def sin(x: Expr) -> Expr:
    """Element-wise sine `sin(x)` with exact 1st and 2nd derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Sin, x.id))


def cos(x: Expr) -> Expr:
    """Element-wise cosine `cos(x)` with exact 1st and 2nd derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Cos, x.id))


def tan(x: Expr) -> Expr:
    """Element-wise tangent `tan(x)` with exact 1st and 2nd derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Tan, x.id))


def tanh(x: Expr) -> Expr:
    """Element-wise hyperbolic tangent `tanh(x)` with exact 1st and 2nd derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Tanh, x.id))


def exp(x: Expr) -> Expr:
    """Element-wise exponential `exp(x)` with exact 1st and 2nd derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Exp, x.id))


def log(x: Expr) -> Expr:
    """Element-wise natural logarithm `ln(x)` with exact 1st and 2nd derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Log, x.id))


def sqrt(x: Expr) -> Expr:
    """Element-wise square root `sqrt(x)` with exact 1st and 2nd derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Sqrt, x.id))


def abs(x: Expr) -> Expr:
    """Element-wise absolute value `|x|` (for C2 smooth absolute value near 0, use `smooth_abs`)."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Abs, x.id))


def square(x: Expr) -> Expr:
    """Element-wise square `x^2` with exact 1st (`2x`) and 2nd (`2`) derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Square, x.id))


def asin(x: Expr) -> Expr:
    """Element-wise inverse sine (arcsin) with exact 1st and 2nd derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Asin, x.id))


def acos(x: Expr) -> Expr:
    """Element-wise inverse cosine (arccos) with exact 1st and 2nd derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Acos, x.id))


def atan(x: Expr) -> Expr:
    """Element-wise inverse tangent (arctan) with exact 1st and 2nd derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Atan, x.id))


def atan2(y: Union[Expr, ArrayLike], x: Union[Expr, ArrayLike]) -> Expr:
    """
    Element-wise four-quadrant inverse tangent atan2(y, x) in [-pi, pi]
    supporting broadcasting and exact 1st and 2nd derivatives.
    Essential for inflow angle phi = atan2(V_axial, V_tan) in rotorcraft BEMT and flight mechanics.
    """
    g = _find_graph(y, x)
    y_expr = _coerce_with_graph(g, y)
    x_expr = _coerce_with_graph(g, x)
    return Expr(g, _core.op_binary(g, _core.OpType.Atan2, y_expr.id, x_expr.id))


def sinh(x: Expr) -> Expr:
    """Element-wise hyperbolic sine with exact 1st and 2nd derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Sinh, x.id))


def cosh(x: Expr) -> Expr:
    """Element-wise hyperbolic cosine with exact 1st and 2nd derivatives."""
    return Expr(x.graph, _core.op_unary(x.graph, _core.OpType.Cosh, x.id))


def ks_max(
    x: Expr,
    rho: float = 50.0,
    axis: Optional[int] = None,
    conservative: bool = False,
) -> Expr:
    """
    Overflow-safe Kreisselmeier-Steinhauser (KS) smooth maximum reduction.
    - When conservative=False (default): computes the normalized KS function
        KS(x) = m + (1/rho) * ln( (1/N) * sum_i exp(rho * (x_i - m)) )
      which satisfies max(x) - ln(N)/rho <= KS(x) <= max(x) and equals c exactly when all x_i = c.
    - When conservative=True: computes the upper-bounding LogSumExp form
        LSE(x) = m + (1/rho) * ln( sum_i exp(rho * (x_i - m)) ) >= max(x).
    """
    if rho <= 0.0:
        raise ValueError("ks_max requires rho > 0.")
    ax = -1 if axis is None else int(axis)
    return Expr(x.graph, _core.op_ks_logsumexp(x.graph, x.id, float(rho), ax, not bool(conservative)))


def ks_min(
    x: Expr,
    rho: float = 50.0,
    axis: Optional[int] = None,
    conservative: bool = False,
) -> Expr:
    """
    Overflow-safe Kreisselmeier-Steinhauser (KS) smooth minimum reduction.
    - When conservative=False (default): satisfies min(x) <= ks_min(x) <= min(x) + ln(N)/rho.
    - When conservative=True: satisfies ks_min(x) <= min(x) (strict lower bound).
    """
    if rho <= 0.0:
        raise ValueError("ks_min requires rho > 0.")
    ax = -1 if axis is None else int(axis)
    return Expr(x.graph, _core.op_ks_logsumexp(x.graph, x.id, -float(rho), ax, not bool(conservative)))


def logsumexp(x: Expr, rho: float = 1.0, axis: Optional[int] = None) -> Expr:
    """Overflow-safe scaled LogSumExp: (1/rho) * ln(sum_i exp(rho * x_i))."""
    if rho == 0.0:
        raise ValueError("logsumexp requires non-zero rho.")
    ax = -1 if axis is None else int(axis)
    return Expr(x.graph, _core.op_ks_logsumexp(x.graph, x.id, float(rho), ax, False))


def smooth_abs(x: Expr, eps: float = 1e-6) -> Expr:
    """C2-smooth absolute value approximation: sqrt(x^2 + eps^2)."""
    if eps <= 0.0:
        raise ValueError("smooth_abs requires eps > 0.")
    return sqrt(square(x) + float(eps) * float(eps))


def smooth_max(
    a: Union[Expr, ArrayLike],
    b: Union[Expr, ArrayLike],
    eps: float = 1e-6,
) -> Expr:
    """Element-wise C2-smooth maximum of two expressions: 0.5 * (a + b + sqrt((a - b)^2 + eps^2))."""
    if eps <= 0.0:
        raise ValueError("smooth_max requires eps > 0.")
    g = _find_graph(a, b)
    a_expr = _coerce_with_graph(g, a)
    b_expr = _coerce_with_graph(g, b)
    diff = a_expr - b_expr
    return 0.5 * (a_expr + b_expr + sqrt(square(diff) + float(eps) * float(eps)))


def smooth_min(
    a: Union[Expr, ArrayLike],
    b: Union[Expr, ArrayLike],
    eps: float = 1e-6,
) -> Expr:
    """Element-wise C2-smooth minimum of two expressions: 0.5 * (a + b - sqrt((a - b)^2 + eps^2))."""
    if eps <= 0.0:
        raise ValueError("smooth_min requires eps > 0.")
    g = _find_graph(a, b)
    a_expr = _coerce_with_graph(g, a)
    b_expr = _coerce_with_graph(g, b)
    diff = a_expr - b_expr
    return 0.5 * (a_expr + b_expr - sqrt(square(diff) + float(eps) * float(eps)))


def smooth_clip(
    x: Expr,
    lb: Union[Expr, ArrayLike],
    ub: Union[Expr, ArrayLike],
    eps: float = 1e-6,
) -> Expr:
    """Element-wise C2-smooth saturation/clipping of x to [lb, ub]."""
    return smooth_min(smooth_max(x, lb, eps=eps), ub, eps=eps)


def matmul(a: Expr, b: Union[Expr, ArrayLike]) -> Expr:
    """Matrix multiplication `a @ b` between 2D symbolic tensors."""
    return a @ b


def solve(A: Union[Expr, ArrayLike], b: Union[Expr, ArrayLike]) -> Expr:
    """
    Solve the symbolic linear system `A @ Y = b` (`Y = A^{-1} b`) with exact 1st- and 2nd-order AD.
    Backed by C++ `LinearSolveNode` (`Eigen::PartialPivLU`), reusing the LU transpose factorization
    directly during reverse-mode adjoints and Forward-over-Reverse Hessian sweeps.

    Args:
        A: Square coefficient matrix `Expr` or numeric array of shape `(m, m)`.
        b: Right-hand side vector or matrix `Expr` or numeric array of shape `(m,)` or `(m, k)`.

    Returns:
        Symbolic solution `Expr` of shape `(m, k)` (or `(m, 1)` when `b` is a 1D vector).
    """
    g = _find_graph(A, b)
    A_e = _coerce_with_graph(g, A)
    b_e = _coerce_with_graph(g, b)
    return Expr(g, _core.op_linear_solve(g, A_e.id, b_e.id))


# Alias for symbolic linear system solver
linalg_solve = solve


def cross(a: Union[Expr, ArrayLike], b: Union[Expr, ArrayLike]) -> Expr:
    """
    Symbolic 3D vector cross product `a x b`.
    Supports 3D column vectors `(3, 1)`, 3D row vectors `(1, 3)`, or batched 3D column matrices `(3, N)`.

    Args:
        a: First 3D vector or `(3, N)` batch matrix (`Expr` or numeric array).
        b: Second 3D vector or `(3, N)` batch matrix (`Expr` or numeric array).

    Returns:
        Symbolic 3D cross product `Expr` matching the orientation of inputs.
    """
    g = _find_graph(a, b)
    ae = _coerce_with_graph(g, a)
    be = _coerce_with_graph(g, b)
    if ae.shape == (1, 3) and be.shape == (1, 3):
        return cross(ae.T, be.T).T
    if ae.shape[0] != 3 or be.shape[0] != 3:
        raise ValueError(
            f"cross requires 3-row vectors (3, 1), (3, N), or (1, 3), got {ae.shape} and {be.shape}."
        )
    ax, ay, az = ae[0:1, :], ae[1:2, :], ae[2:3, :]
    bx, by, bz = be[0:1, :], be[1:2, :], be[2:3, :]
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return vstack([cx, cy, cz])


def trace(A: Expr) -> Expr:
    """
    Compute the matrix trace `sum_i A[i, i]` of a square 2D symbolic matrix `A`.

    Args:
        A: Square symbolic matrix `Expr` of shape `(m, m)`.

    Returns:
        Scalar `(1, 1)` `Expr` representing the trace of `A`.
    """
    r, c = A.shape
    if r != c:
        raise ValueError(f"trace requires a square matrix (m, m), got {A.shape}.")
    return diag(A).sum()


def dot(a: Expr, b: Union[Expr, ArrayLike]) -> Expr:
    """Frobenius / Euclidean inner product `sum(a * b)` returning a `(1, 1)` scalar `Expr`."""
    rhs = a._coerce(b)
    return Expr(a.graph, _core.op_dot(a.graph, a.id, rhs.id))


import builtins as _builtins


def sum(x: Any, axis: Optional[int] = None) -> Any:
    """Sum symbolic `Expr` elements along `axis` (`None`, `0`, or `1`), or fallback to Python `builtins.sum`."""
    if isinstance(x, Expr):
        return x.sum(axis=axis)
    return _builtins.sum(x)


def transpose(x: Expr) -> Expr:
    """Return 2D matrix transpose `x.T`."""
    return x.T


def concat(exprs: Sequence[Union[Expr, ArrayLike]], axis: int = 0) -> Expr:
    """Concatenate a sequence of 2D expressions along `axis=0` (vertical) or `axis=1` (horizontal)."""
    if not exprs:
        raise ValueError("concat requires a non-empty list of Expr.")
    g = _find_graph(*exprs)
    coerced = [_coerce_with_graph(g, e) for e in exprs]
    ids = [e.id for e in coerced]
    return Expr(g, _core.op_concat(g, ids, int(axis)))


def vstack(exprs: Sequence[Union[Expr, ArrayLike]]) -> Expr:
    """Stack expressions vertically along axis=0 (row-wise stack)."""
    return concat(exprs, axis=0)


def hstack(exprs: Sequence[Union[Expr, ArrayLike]]) -> Expr:
    """Stack expressions horizontally along axis=1 (column-wise stack)."""
    return concat(exprs, axis=1)


def sumsqr(x: Expr) -> Expr:
    """Sum of squares `sum(x_i^2)` computed via a single C++ `DotNode(x, x)` with zero intermediate tensor."""
    return Expr(x.graph, _core.op_dot(x.graph, x.id, x.id))


def mean(x: Expr, axis: Optional[int] = None) -> Expr:
    """Arithmetic mean over all elements (`axis=None`) or along `axis=0` / `axis=1`."""
    if axis is None or int(axis) == -1:
        return x.sum(axis=None) * (1.0 / float(x.size))
    ax = int(axis)
    if ax == 0:
        return x.sum(axis=0) * (1.0 / float(x.shape[0]))
    if ax == 1:
        return x.sum(axis=1) * (1.0 / float(x.shape[1]))
    raise ValueError("mean: axis must be None, 0, or 1.")


def norm(x: Expr, ord: int = 2, axis: Optional[int] = None, eps: float = 0.0) -> Expr:
    """
    Vector/matrix norm (L2 Euclidean / Frobenius when `ord=2`, L1 when `ord=1`).
    Optional `eps > 0` provides C2-smooth regularization at zero (`sqrt(sumsqr(x) + eps^2)`).
    """
    if ord == 2:
        if axis is None or int(axis) == -1:
            ss = sumsqr(x)
            return sqrt(ss + float(eps) * float(eps)) if eps > 0.0 else sqrt(ss)
        ss_ax = square(x).sum(axis=int(axis))
        return sqrt(ss_ax + float(eps) * float(eps)) if eps > 0.0 else sqrt(ss_ax)
    if ord == 1:
        return abs(x).sum(axis=axis)
    raise ValueError("norm: supported `ord` values are 2 (L2/Frobenius) and 1 (L1).")


def diag(x: Expr) -> Expr:
    """
    If `x` is a vector of length N (`(N, 1)` or `(1, N)`), construct an `(N, N)` diagonal matrix.
    If `x` is a square `(N, N)` matrix (`N > 1`), extract its main diagonal as an `(N, 1)` column vector.
    """
    r, c = x.shape
    if r == 1 or c == 1:
        n = r * c
        col_vec = x if c == 1 else x.reshape(n, 1)
        eye_c = _coerce_with_graph(x.graph, np.eye(n, dtype=np.float64))
        return col_vec * eye_c
    if r == c:
        eye_c = _coerce_with_graph(x.graph, np.eye(r, dtype=np.float64))
        return (x * eye_c).sum(axis=1)
    raise ValueError(f"diag requires a vector or a square matrix, got shape {(r, c)}.")


def clip(x: Expr, lb: Union[Expr, ArrayLike], ub: Union[Expr, ArrayLike]) -> Expr:
    """Element-wise hard clipping of `x` to `[lb, ub]` via NaN-safe `where`."""
    return where(x < lb, lb, where(x > ub, ub, x))


def deg2rad(x: Union[Expr, ArrayLike]) -> Any:
    """Convert angles from degrees to radians (supports both symbolic `Expr` and numeric arrays)."""
    return x * (np.pi / 180.0)


def rad2deg(x: Union[Expr, ArrayLike]) -> Any:
    """Convert angles from radians to degrees (supports both symbolic `Expr` and numeric arrays)."""
    return x * (180.0 / np.pi)


def _not_a_knot_vector(x: np.ndarray, k: int) -> np.ndarray:
    """Construct clamped not-a-knot B-spline knot vector matching scipy.interpolate.make_interp_spline."""
    n = len(x)
    if n <= k:
        raise ValueError(f"Number of data points ({n}) must be > degree ({k}).")
    if k == 1:
        return np.concatenate([[x[0]], x, [x[-1]]])
    if k % 2 == 0:
        midpoints = 0.5 * (x[:-1] + x[1:])
        interior = midpoints[k // 2 : n - 1 - k // 2]
    else:
        m = (k - 1) // 2
        interior = x[m + 1 : n - 1 - m]
    return np.concatenate([np.repeat(x[0], k + 1), interior, np.repeat(x[-1], k + 1)])


def _clamped_lstsq_knot_vector(x: np.ndarray, n_ctrl: int, k: int) -> np.ndarray:
    """Construct clamped B-spline knot vector with `n_ctrl` basis functions using quantile interior knots."""
    if n_ctrl <= k or n_ctrl > len(x):
        raise ValueError(f"n_ctrl ({n_ctrl}) must satisfy degree + 1 ({k + 1}) <= n_ctrl <= len(x_grid) ({len(x)}).")
    if n_ctrl == len(x):
        return _not_a_knot_vector(x, k)
    n_int = n_ctrl - k - 1
    if n_int > 0:
        probs = np.linspace(0.0, 1.0, n_int + 2)[1:-1]
        interior = np.quantile(x, probs)
    else:
        interior = np.array([], dtype=np.float64)
    return np.concatenate([np.repeat(x[0], k + 1), interior, np.repeat(x[-1], k + 1)])


def _bspline_1d_diff_operator(knots: np.ndarray, degree: int, nu: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute the analytical linear control-point differentiation matrix D^{(nu)}
    such that P^{(nu)} = D^{(nu)} @ P, along with the reduced knot vector and reduced degree.
    """
    if nu < 0:
        raise ValueError("Derivative order `nu` must be >= 0.")
    cur_knots = knots.copy()
    cur_deg = int(degree)
    n_ctrl = cur_knots.size - cur_deg - 1
    D_total = np.eye(n_ctrl, dtype=np.float64)

    for _ in range(nu):
        if cur_deg < 1 or n_ctrl < 2:
            raise ValueError(
                f"Cannot differentiate B-spline of degree {degree} with {knots.size - degree - 1} control points "
                f"to order nu={nu}."
            )
        dt = cur_knots[cur_deg + 1 : n_ctrl + cur_deg] - cur_knots[1:n_ctrl]
        D_step = np.zeros((n_ctrl - 1, n_ctrl), dtype=np.float64)
        for i in range(n_ctrl - 1):
            if dt[i] > 0.0:
                scale = float(cur_deg) / float(dt[i])
                D_step[i, i] = -scale
                D_step[i, i + 1] = scale
        D_total = D_step @ D_total
        cur_knots = cur_knots[1:-1]
        cur_deg -= 1
        n_ctrl -= 1

    return D_total, cur_knots, cur_deg


def _solve_1d_spline_system(
    x_grid: np.ndarray,
    k: int,
    n_ctrl: Optional[int],
    reg_weight: float,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Build knot vector and either collocation matrix A (if exact square collocation)
    or regularized least-squares operator M = (A^T A + reg * R_2^T R_2)^{-1} A^T.
    Returns `(knots, A_or_M, is_exact_collocation)`.
    """
    n_pts = len(x_grid)
    n_c = n_pts if n_ctrl is None else int(n_ctrl)
    knots = _clamped_lstsq_knot_vector(x_grid, n_c, k)
    collocation = BSpline.design_matrix(x_grid, knots, k).toarray()

    if n_c == n_pts and float(reg_weight) == 0.0:
        return knots, collocation, True

    ata = collocation.T @ collocation
    if float(reg_weight) > 0.0 and n_c >= 3:
        r2 = np.zeros((n_c - 2, n_c), dtype=np.float64)
        for i in range(n_c - 2):
            r2[i, i] = 1.0
            r2[i, i + 1] = -2.0
            r2[i, i + 2] = 1.0
        ata = ata + float(reg_weight) * (r2.T @ r2)
    proj_op = np.linalg.solve(ata, collocation.T)
    return knots, proj_op, False


def _format_numeric_interp_output(
    mat: np.ndarray,
    is_scalar_query: bool,
    n_channels: int,
) -> Union[float, np.ndarray]:
    """Format numeric interpolation output cleanly for scalar vs. vector queries and 1 vs. K channels."""
    if is_scalar_query and n_channels == 1:
        return float(mat.item())
    if is_scalar_query or n_channels == 1:
        return mat.ravel().copy()
    return mat.copy()


def _get_or_create_cached_const_ids(
    cache: dict,
    g: _core.Graph,
    col_arrays: List[Tuple[_core.Shape, np.ndarray]],
) -> List[int]:
    """
    Retrieve or create ConstantNode IDs on Graph `g` for pre-compiled interpolation tables/control points.
    Stores only plain Python ints in `cache` (zero C++ Graph/Shape reference retention).
    """
    gid = id(g)
    cur_n = int(g.num_nodes())
    if gid in cache:
        cids, saved_n = cache[gid]
        if cur_n >= saved_n and all(cid < cur_n and g.node_op(cid) == _core.OpType.Constant for cid in cids):
            return cids

    cids = []
    for s, flat in col_arrays:
        cids.append(int(g.add_constant(s, flat, "")))
    cache[gid] = (cids, int(g.num_nodes()))
    return cids


# ============================================================================
# Reusable Pre-Compiled Interpolator Classes (1D, 2D & 3D)
# ============================================================================
class LinearInterpolator1D:
    """
    Reusable 1D piecewise-linear interpolator.
    - Pre-validates `x_grid` and pre-formats `y_data` ONCE on initialization.
    - Automatically shares a single C++ `ConstantNode` across multiple calls in the same `Graph`.
    - Callable on both symbolic `Expr` queries (1st & 2nd order AD) and numerical `float`/`np.ndarray` queries.
    """

    def __init__(
        self,
        x_grid: Sequence[float],
        y_data: Union[Expr, ArrayLike],
        extrapolate: bool = False,
    ) -> None:
        """
        Pre-compile a 1D piecewise-linear interpolator over `(x_grid, y_data)`.

        Args:
            x_grid: 1D strictly increasing grid coordinates of length `N >= 2`.
            y_data: Numeric array or symbolic `Expr` of shape `(N,)` or `(N, K)`.
            extrapolate: Linearly extrapolate beyond `[x_grid[0], x_grid[-1]]` if `True`, else clamp.
        """
        self._x_grid = np.asarray(x_grid, dtype=np.float64).ravel()
        if self._x_grid.size < 2 or np.any(np.diff(self._x_grid) <= 0.0):
            raise ValueError("LinearInterpolator1D: x_grid must be 1D strictly increasing with >= 2 points.")
        self._extrapolate = bool(extrapolate)
        self._graph_cache: dict = {}

        nx = self._x_grid.size
        if isinstance(y_data, Expr):
            if y_data.shape[0] != nx:
                raise ValueError(
                    f"LinearInterpolator1D: symbolic y_data.shape[0] ({y_data.shape[0]}) must match len(x_grid) ({nx})."
                )
            self._y_expr: Optional[Expr] = y_data
            self._y_val: Optional[np.ndarray] = None
            self._n_channels = int(y_data.shape[1])
            self._const_payload: List[Tuple[_core.Shape, np.ndarray]] = []
        else:
            y_arr = np.asarray(y_data, dtype=np.float64)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            if y_arr.ndim != 2 or y_arr.shape[0] != nx:
                raise ValueError(
                    f"LinearInterpolator1D: y_data shape {y_arr.shape} must be ({nx},) or ({nx}, K)."
                )
            self._y_expr = None
            self._y_val = y_arr.copy()
            self._n_channels = int(y_arr.shape[1])
            s = _core.Shape(nx, self._n_channels)
            flat = np.asfortranarray(self._y_val).ravel(order="F")
            self._const_payload = [(s, flat)]

    @property
    def x_grid(self) -> np.ndarray:
        """1D strictly increasing coordinate grid."""
        return self._x_grid

    @property
    def y_data(self) -> Union[Expr, np.ndarray]:
        """Underlying 1D/2D table values (`np.ndarray` or symbolic `Expr`)."""
        return self._y_expr if self._y_expr is not None else self._y_val  # type: ignore[return-value]

    @property
    def extrapolate(self) -> bool:
        """Whether out-of-domain queries extrapolate linearly (`True`) or clamp (`False`)."""
        return self._extrapolate

    def __call__(self, x_query: Union[Expr, ArrayLike]) -> Union[Expr, float, np.ndarray]:
        """Evaluate the 1D linear interpolator at symbolic `Expr` or numeric `x_query`."""
        if isinstance(x_query, Expr) or self._y_expr is not None:
            g = _find_graph(x_query, self._y_expr)
            xq_expr = _coerce_with_graph(g, x_query)
            if self._y_expr is not None:
                if self._y_expr.graph is not g:
                    raise ValueError("LinearInterpolator1D: x_query and symbolic y_data belong to different Graphs.")
                y_id = self._y_expr.id
            else:
                y_id = _get_or_create_cached_const_ids(self._graph_cache, g, self._const_payload)[0]
            nid = _core.op_linear_interp(g, xq_expr.id, y_id, self._x_grid, self._extrapolate)
            return Expr(g, nid)

        # Direct numerical evaluation via local ephemeral C++ Evaluator
        arr_q = np.asarray(x_query, dtype=np.float64)
        is_scalar = arr_q.ndim == 0 or arr_q.size == 1
        q_flat = arr_q.ravel()
        tmp_g = _core.Graph()
        q_id = tmp_g.add_variable(
            _core.Shape(q_flat.size, 1),
            q_flat,
            np.full(q_flat.size, -np.inf),
            np.full(q_flat.size, np.inf),
            "xq",
        )
        s, flat = self._const_payload[0]
        y_id = tmp_g.add_constant(s, flat, "")
        nid = _core.op_linear_interp(tmp_g, q_id, y_id, self._x_grid, self._extrapolate)
        ev = _core.Evaluator(tmp_g, [nid])
        res_mat = ev.eval(q_flat)[0]
        return _format_numeric_interp_output(res_mat, is_scalar, self._n_channels)


class BSplineCurve1D:
    """
    Reusable 1D B-Spline evaluator given pre-computed control points and knot vector.
    - Shares a single C++ `ConstantNode` across multiple calls in the same `Graph`.
    - Callable on both symbolic `Expr` queries and numerical `float`/`np.ndarray` queries.
    """

    def __init__(
        self,
        ctrl_pts: Union[Expr, ArrayLike],
        knots: Sequence[float],
        degree: int = 3,
        extrapolate: bool = False,
    ) -> None:
        """
        Pre-compile a 1D B-spline curve evaluator from explicit control points and knot vector.

        Args:
            ctrl_pts: Control points array or symbolic `Expr` of shape `(n_ctrl,)` or `(n_ctrl, K)`.
            knots: Non-decreasing 1D knot vector of length `n_ctrl + degree + 1`.
            degree: B-spline polynomial degree (default: `3` for C2 cubic).
            extrapolate: Extrapolate outside knot domain if `True`, else clamp.
        """
        self._knots = np.asarray(knots, dtype=np.float64).ravel()
        self._degree = int(degree)
        self._extrapolate = bool(extrapolate)
        self._graph_cache: dict = {}

        n_ctrl = self._knots.size - self._degree - 1
        if n_ctrl < 1:
            raise ValueError("BSplineCurve1D: invalid knot vector length for given degree.")

        if isinstance(ctrl_pts, Expr):
            if ctrl_pts.shape[0] != n_ctrl:
                raise ValueError(
                    f"BSplineCurve1D: symbolic ctrl_pts.shape[0] ({ctrl_pts.shape[0]}) != n_ctrl ({n_ctrl})."
                )
            self._ctrl_expr: Optional[Expr] = ctrl_pts
            self._ctrl_val: Optional[np.ndarray] = None
            self._n_channels = int(ctrl_pts.shape[1])
            self._const_payload: List[Tuple[_core.Shape, np.ndarray]] = []
        else:
            p_arr = np.asarray(ctrl_pts, dtype=np.float64)
            if p_arr.ndim == 1:
                p_arr = p_arr.reshape(-1, 1)
            if p_arr.ndim != 2 or p_arr.shape[0] != n_ctrl:
                raise ValueError(
                    f"BSplineCurve1D: ctrl_pts shape {p_arr.shape} must be ({n_ctrl},) or ({n_ctrl}, K)."
                )
            self._ctrl_expr = None
            self._ctrl_val = p_arr.copy()
            self._n_channels = int(p_arr.shape[1])
            s = _core.Shape(n_ctrl, self._n_channels)
            flat = np.asfortranarray(self._ctrl_val).ravel(order="F")
            self._const_payload = [(s, flat)]

    @property
    def knots(self) -> np.ndarray:
        """1D knot vector of length `n_ctrl + degree + 1`."""
        return self._knots

    @property
    def degree(self) -> int:
        """B-spline polynomial degree `k`."""
        return self._degree

    @property
    def ctrl_pts(self) -> Union[Expr, np.ndarray]:
        """Pre-compiled B-spline control points (`np.ndarray` or symbolic `Expr`)."""
        return self._ctrl_expr if self._ctrl_expr is not None else self._ctrl_val  # type: ignore[return-value]

    @property
    def extrapolate(self) -> bool:
        """Whether out-of-domain queries extrapolate (`True`) or clamp (`False`)."""
        return self._extrapolate

    def __call__(self, u_query: Union[Expr, ArrayLike]) -> Union[Expr, float, np.ndarray]:
        """Evaluate the 1D B-spline curve at symbolic `Expr` or numeric `u_query`."""
        if isinstance(u_query, Expr) or self._ctrl_expr is not None:
            g = _find_graph(u_query, self._ctrl_expr)
            uq_expr = _coerce_with_graph(g, u_query)
            if self._ctrl_expr is not None:
                if self._ctrl_expr.graph is not g:
                    raise ValueError("BSplineCurve1D: u_query and symbolic ctrl_pts belong to different Graphs.")
                p_id = self._ctrl_expr.id
            else:
                p_id = _get_or_create_cached_const_ids(self._graph_cache, g, self._const_payload)[0]
            nid = _core.op_bspline_eval(g, uq_expr.id, p_id, self._knots, self._degree, self._extrapolate)
            return Expr(g, nid)

        arr_q = np.asarray(u_query, dtype=np.float64)
        is_scalar = arr_q.ndim == 0 or arr_q.size == 1
        q_flat = arr_q.ravel()
        tmp_g = _core.Graph()
        q_id = tmp_g.add_variable(
            _core.Shape(q_flat.size, 1),
            q_flat,
            np.full(q_flat.size, -np.inf),
            np.full(q_flat.size, np.inf),
            "uq",
        )
        s, flat = self._const_payload[0]
        p_id = tmp_g.add_constant(s, flat, "")
        nid = _core.op_bspline_eval(tmp_g, q_id, p_id, self._knots, self._degree, self._extrapolate)
        ev = _core.Evaluator(tmp_g, [nid])
        res_mat = ev.eval(q_flat)[0]
        return _format_numeric_interp_output(res_mat, is_scalar, self._n_channels)

    def derivative(self, nu: int = 1) -> BSplineCurve1D:
        """
        Return a new pre-compiled `BSplineCurve1D` representing the `nu`-th spatial derivative
        $d^{nu} S(u) / du^{nu}$ (of degree `degree - nu`).
        Preserves full 1st and 2nd order AD w.r.t. both query points `u` and symbolic `ctrl_pts`!
        """
        if int(nu) == 0:
            return self
        D_mat, new_knots, new_deg = _bspline_1d_diff_operator(self._knots, self._degree, int(nu))
        if self._ctrl_expr is not None:
            D_expr = _coerce_with_graph(self._ctrl_expr.graph, D_mat)
            new_ctrl: Union[Expr, np.ndarray] = D_expr @ self._ctrl_expr
        else:
            assert self._ctrl_val is not None
            new_ctrl = D_mat @ self._ctrl_val
        return BSplineCurve1D(
            ctrl_pts=new_ctrl,
            knots=new_knots,
            degree=new_deg,
            extrapolate=self._extrapolate,
        )


class BSplineInterpolator1D(BSplineCurve1D):
    """
    Reusable 1D interpolating B-Spline passing through `(x_grid, y_data)`.
    - Solves the B-spline collocation system ONCE in `__init__` (even when `y_data` is a symbolic `Expr`!).
    - Subsequent calls `spline(xq)` reuse the solved control points with zero repeated linear solves
      and share a single node in the C++ computation graph.
    """

    def __init__(
        self,
        x_grid: Sequence[float],
        y_data: Union[Expr, ArrayLike],
        degree: int = 3,
        extrapolate: bool = False,
        n_ctrl: Optional[int] = None,
        reg_weight: float = 0.0,
    ) -> None:
        """
        Pre-compute 1D B-spline control points and knot vector from `(x_grid, y_data)` ONCE on initialization.

        Args:
            x_grid: 1D strictly increasing grid coordinates of length `N`.
            y_data: Data values of shape `(N,)` or `(N, K)` (numeric array or symbolic `Expr`).
            degree: B-spline polynomial degree (default: `3` for C2 cubic).
            extrapolate: Extrapolate outside `[x_grid[0], x_grid[-1]]` if `True`, else clamp.
            n_ctrl: Optional reduced control point count (`< N`) for least-squares spline regression.
            reg_weight: Second-difference smoothing regularization weight `>= 0.0`.
        """
        grid_arr = np.asarray(x_grid, dtype=np.float64).ravel()
        if grid_arr.ndim != 1 or len(grid_arr) < 2:
            raise ValueError("BSplineInterpolator1D: x_grid must be a 1D array with at least 2 points.")
        if np.any(np.diff(grid_arr) <= 0):
            raise ValueError("BSplineInterpolator1D: x_grid must be strictly increasing.")

        self._x_grid = grid_arr
        max_c = len(grid_arr) if n_ctrl is None else int(n_ctrl)
        k = min(int(degree), max_c - 1)
        knots, mat_op, is_exact = _solve_1d_spline_system(grid_arr, k, n_ctrl, float(reg_weight))

        if isinstance(y_data, Expr):
            op_mat = np.linalg.inv(mat_op) if is_exact else mat_op
            op_expr = _coerce_with_graph(y_data.graph, op_mat)
            ctrl_pts: Union[Expr, np.ndarray] = op_expr @ y_data
        else:
            y_arr = np.asarray(y_data, dtype=np.float64)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            ctrl_pts = np.linalg.solve(mat_op, y_arr) if is_exact else (mat_op @ y_arr)

        super().__init__(ctrl_pts=ctrl_pts, knots=knots, degree=k, extrapolate=extrapolate)

    @property
    def x_grid(self) -> np.ndarray:
        """1D strictly increasing coordinate grid."""
        return self._x_grid


class LinearInterpolator2D:
    """
    Reusable 2D bilinear regular-grid interpolator over `(x_grid, y_grid, z_data)`.
    - Pre-validates grids and table shapes ONCE on initialization.
    - Shares a single set of C++ `ConstantNode`s across multiple queries in the same `Graph`.
    - Supports both symbolic `Expr` queries and direct numerical `float`/`np.ndarray` queries.
    """

    def __init__(
        self,
        x_grid: Sequence[float],
        y_grid: Sequence[float],
        z_data: Union[Expr, ArrayLike],
        extrapolate: bool = False,
    ) -> None:
        """
        Pre-compile a 2D bilinear regular-grid interpolator over `(x_grid, y_grid, z_data)`.

        Args:
            x_grid: 1D strictly increasing X coordinates of length `Nx`.
            y_grid: 1D strictly increasing Y coordinates of length `Ny`.
            z_data: Table values of shape `(Nx, Ny)` or `(Nx, Ny, K)` (numeric array or symbolic `Expr`).
            extrapolate: Extrapolate outside grid bounds if `True`, else clamp.
        """
        self._x_grid = np.asarray(x_grid, dtype=np.float64).ravel()
        self._y_grid = np.asarray(y_grid, dtype=np.float64).ravel()
        nx, ny = self._x_grid.size, self._y_grid.size
        if nx < 2 or ny < 2 or np.any(np.diff(self._x_grid) <= 0) or np.any(np.diff(self._y_grid) <= 0):
            raise ValueError("LinearInterpolator2D: x_grid and y_grid must be strictly increasing with >= 2 points.")
        self._extrapolate = bool(extrapolate)
        self._graph_cache: dict = {}

        if isinstance(z_data, Expr):
            if z_data.shape != (nx, ny):
                raise ValueError(
                    f"LinearInterpolator2D: symbolic Expr z_data must have shape ({nx}, {ny}), got {z_data.shape}."
                )
            self._z_expr: Optional[Expr] = z_data
            self._z_val: Optional[np.ndarray] = None
            self._n_channels = 1
            self._const_payload: List[Tuple[_core.Shape, np.ndarray]] = []
        else:
            arr = np.asarray(z_data, dtype=np.float64)
            if arr.ndim == 2 and arr.shape == (nx, ny):
                arr_3d = arr[:, :, np.newaxis]
            elif arr.ndim == 3 and arr.shape[0] == nx and arr.shape[1] == ny:
                arr_3d = arr
            else:
                raise ValueError(
                    f"LinearInterpolator2D: z_data must have shape ({nx}, {ny}) or ({nx}, {ny}, K), got {arr.shape}."
                )
            self._z_expr = None
            self._z_val = arr.copy()
            self._n_channels = int(arr_3d.shape[2])
            s = _core.Shape(nx, ny)
            self._const_payload = [
                (s, np.asfortranarray(arr_3d[:, :, ch]).ravel(order="F"))
                for ch in range(self._n_channels)
            ]

    @property
    def x_grid(self) -> np.ndarray:
        """1D strictly increasing X coordinate grid."""
        return self._x_grid

    @property
    def y_grid(self) -> np.ndarray:
        """1D strictly increasing Y coordinate grid."""
        return self._y_grid

    @property
    def z_data(self) -> Union[Expr, np.ndarray]:
        """Underlying 2D/3D table data (`np.ndarray` or symbolic `Expr`)."""
        return self._z_expr if self._z_expr is not None else self._z_val  # type: ignore[return-value]

    @property
    def extrapolate(self) -> bool:
        """Whether out-of-domain queries extrapolate (`True`) or clamp (`False`)."""
        return self._extrapolate

    def __call__(
        self,
        x_query: Union[Expr, ArrayLike],
        y_query: Union[Expr, ArrayLike],
    ) -> Union[Expr, float, np.ndarray]:
        """Evaluate the 2D bilinear interpolator at `(x_query, y_query)`."""
        if isinstance(x_query, Expr) or isinstance(y_query, Expr) or self._z_expr is not None:
            g = _find_graph(x_query, y_query, self._z_expr)
            xq_expr = _coerce_with_graph(g, x_query)
            yq_expr = _coerce_with_graph(g, y_query)
            if self._z_expr is not None:
                if self._z_expr.graph is not g:
                    raise ValueError("LinearInterpolator2D: queries and symbolic z_data belong to different Graphs.")
                nid = _core.op_linear_interp_2d(
                    g, xq_expr.id, yq_expr.id, self._z_expr.id, self._x_grid, self._y_grid, self._extrapolate
                )
                return Expr(g, nid)

            cids = _get_or_create_cached_const_ids(self._graph_cache, g, self._const_payload)
            cols = [
                Expr(g, _core.op_linear_interp_2d(g, xq_expr.id, yq_expr.id, cid, self._x_grid, self._y_grid, self._extrapolate))
                for cid in cids
            ]
            return concat(cols, axis=1) if len(cols) > 1 else cols[0]

        arr_x = np.asarray(x_query, dtype=np.float64)
        arr_y = np.asarray(y_query, dtype=np.float64)
        is_scalar = (arr_x.ndim == 0 or arr_x.size == 1) and (arr_y.ndim == 0 or arr_y.size == 1)
        qx = arr_x.ravel()
        qy = arr_y.ravel()
        tmp_g = _core.Graph()
        x_id = tmp_g.add_variable(_core.Shape(qx.size, 1), qx, np.full(qx.size, -np.inf), np.full(qx.size, np.inf), "xq")
        y_id = tmp_g.add_variable(_core.Shape(qy.size, 1), qy, np.full(qy.size, -np.inf), np.full(qy.size, np.inf), "yq")
        out_exprs = []
        for s, flat in self._const_payload:
            cid = tmp_g.add_constant(s, flat, "")
            nid = _core.op_linear_interp_2d(tmp_g, x_id, y_id, cid, self._x_grid, self._y_grid, self._extrapolate)
            out_exprs.append(Expr(tmp_g, nid))
        final_expr = concat(out_exprs, axis=1) if len(out_exprs) > 1 else out_exprs[0]
        ev = _core.Evaluator(tmp_g, [final_expr.id])
        res_mat = ev.eval(np.concatenate([qx, qy]))[0]
        return _format_numeric_interp_output(res_mat, is_scalar, self._n_channels)


class BSplineSurface2D:
    """
    Reusable 2D tensor-product B-Spline surface evaluator given pre-computed control points and knot vectors.
    - Shares a single set of C++ `ConstantNode`s across multiple queries in the same `Graph`.
    - Supports both symbolic `Expr` queries and direct numerical `float`/`np.ndarray` queries.
    """

    def __init__(
        self,
        ctrl_pts: Union[Expr, ArrayLike],
        knots_u: Sequence[float],
        knots_v: Sequence[float],
        degree_u: int = 3,
        degree_v: int = 3,
        extrapolate: bool = False,
    ) -> None:
        """
        Pre-compile a 2D tensor-product B-spline surface from control net `ctrl_pts` and knot vectors.

        Args:
            ctrl_pts: Control net of shape `(Nu, Nv)` or `(Nu, Nv, K)` (numeric array or symbolic `Expr`).
            knots_u: Non-decreasing knot vector along U of length `Nu + degree_u + 1`.
            knots_v: Non-decreasing knot vector along V of length `Nv + degree_v + 1`.
            degree_u: B-spline degree along U (default: `3`).
            degree_v: B-spline degree along V (default: `3`).
            extrapolate: Extrapolate outside domain if `True`, else clamp.
        """
        self._knots_u = np.asarray(knots_u, dtype=np.float64).ravel()
        self._knots_v = np.asarray(knots_v, dtype=np.float64).ravel()
        self._degree_u = int(degree_u)
        self._degree_v = int(degree_v)
        self._extrapolate = bool(extrapolate)
        self._graph_cache: dict = {}

        nu = self._knots_u.size - self._degree_u - 1
        nv = self._knots_v.size - self._degree_v - 1
        if nu < 1 or nv < 1:
            raise ValueError("BSplineSurface2D: invalid knot vector lengths for given degrees.")

        if isinstance(ctrl_pts, Expr):
            if ctrl_pts.shape != (nu, nv):
                raise ValueError(
                    f"BSplineSurface2D: symbolic Expr ctrl_pts must have shape ({nu}, {nv}), got {ctrl_pts.shape}."
                )
            self._ctrl_expr: Optional[Expr] = ctrl_pts
            self._ctrl_val: Optional[np.ndarray] = None
            self._n_channels = 1
            self._const_payload: List[Tuple[_core.Shape, np.ndarray]] = []
        else:
            arr = np.asarray(ctrl_pts, dtype=np.float64)
            if arr.ndim == 2 and arr.shape == (nu, nv):
                arr_3d = arr[:, :, np.newaxis]
            elif arr.ndim == 3 and arr.shape[0] == nu and arr.shape[1] == nv:
                arr_3d = arr
            else:
                raise ValueError(
                    f"BSplineSurface2D: ctrl_pts must have shape ({nu}, {nv}) or ({nu}, {nv}, K), got {arr.shape}."
                )
            self._ctrl_expr = None
            self._ctrl_val = arr.copy()
            self._n_channels = int(arr_3d.shape[2])
            s = _core.Shape(nu, nv)
            self._const_payload = [
                (s, np.asfortranarray(arr_3d[:, :, ch]).ravel(order="F"))
                for ch in range(self._n_channels)
            ]

    @property
    def knots_u(self) -> np.ndarray:
        """Knot vector along axis U."""
        return self._knots_u

    @property
    def knots_v(self) -> np.ndarray:
        """Knot vector along axis V."""
        return self._knots_v

    @property
    def degree_u(self) -> int:
        """B-spline degree along axis U."""
        return self._degree_u

    @property
    def degree_v(self) -> int:
        """B-spline degree along axis V."""
        return self._degree_v

    @property
    def ctrl_pts(self) -> Union[Expr, np.ndarray]:
        """Pre-compiled 2D/3D B-spline control net (`np.ndarray` or symbolic `Expr`)."""
        return self._ctrl_expr if self._ctrl_expr is not None else self._ctrl_val  # type: ignore[return-value]

    @property
    def extrapolate(self) -> bool:
        """Whether out-of-domain queries extrapolate (`True`) or clamp (`False`)."""
        return self._extrapolate

    def __call__(
        self,
        u_query: Union[Expr, ArrayLike],
        v_query: Union[Expr, ArrayLike],
    ) -> Union[Expr, float, np.ndarray]:
        """Evaluate the 2D B-spline surface at `(u_query, v_query)`."""
        if isinstance(u_query, Expr) or isinstance(v_query, Expr) or self._ctrl_expr is not None:
            g = _find_graph(u_query, v_query, self._ctrl_expr)
            uq_expr = _coerce_with_graph(g, u_query)
            vq_expr = _coerce_with_graph(g, v_query)
            if self._ctrl_expr is not None:
                if self._ctrl_expr.graph is not g:
                    raise ValueError("BSplineSurface2D: queries and symbolic ctrl_pts belong to different Graphs.")
                nid = _core.op_bspline_eval_2d(
                    g, uq_expr.id, vq_expr.id, self._ctrl_expr.id,
                    self._knots_u, self._knots_v, self._degree_u, self._degree_v, self._extrapolate
                )
                return Expr(g, nid)

            cids = _get_or_create_cached_const_ids(self._graph_cache, g, self._const_payload)
            cols = [
                Expr(
                    g,
                    _core.op_bspline_eval_2d(
                        g, uq_expr.id, vq_expr.id, cid,
                        self._knots_u, self._knots_v, self._degree_u, self._degree_v, self._extrapolate
                    ),
                )
                for cid in cids
            ]
            return concat(cols, axis=1) if len(cols) > 1 else cols[0]

        arr_u = np.asarray(u_query, dtype=np.float64)
        arr_v = np.asarray(v_query, dtype=np.float64)
        is_scalar = (arr_u.ndim == 0 or arr_u.size == 1) and (arr_v.ndim == 0 or arr_v.size == 1)
        qu = arr_u.ravel()
        qv = arr_v.ravel()
        tmp_g = _core.Graph()
        u_id = tmp_g.add_variable(_core.Shape(qu.size, 1), qu, np.full(qu.size, -np.inf), np.full(qu.size, np.inf), "uq")
        v_id = tmp_g.add_variable(_core.Shape(qv.size, 1), qv, np.full(qv.size, -np.inf), np.full(qv.size, np.inf), "vq")
        out_exprs = []
        for s, flat in self._const_payload:
            cid = tmp_g.add_constant(s, flat, "")
            nid = _core.op_bspline_eval_2d(
                tmp_g, u_id, v_id, cid, self._knots_u, self._knots_v, self._degree_u, self._degree_v, self._extrapolate
            )
            out_exprs.append(Expr(tmp_g, nid))
        final_expr = concat(out_exprs, axis=1) if len(out_exprs) > 1 else out_exprs[0]
        ev = _core.Evaluator(tmp_g, [final_expr.id])
        res_mat = ev.eval(np.concatenate([qu, qv]))[0]
        return _format_numeric_interp_output(res_mat, is_scalar, self._n_channels)

    def derivative(self, nu_u: int = 0, nu_v: int = 0) -> BSplineSurface2D:
        r"""
        Return a new pre-compiled `BSplineSurface2D` representing the partial derivative surface
        $\partial^{nu_u + nu_v} S(u, v) / (\partial u^{nu_u} \partial v^{nu_v})$.
        """
        if int(nu_u) == 0 and int(nu_v) == 0:
            return self
        Du, new_ku, new_degu = _bspline_1d_diff_operator(self._knots_u, self._degree_u, int(nu_u))
        Dv, new_kv, new_degv = _bspline_1d_diff_operator(self._knots_v, self._degree_v, int(nu_v))

        if self._ctrl_expr is not None:
            g = self._ctrl_expr.graph
            Du_e = _coerce_with_graph(g, Du)
            Dv_t_e = _coerce_with_graph(g, Dv.T)
            new_ctrl: Union[Expr, np.ndarray] = Du_e @ self._ctrl_expr @ Dv_t_e
        else:
            assert self._ctrl_val is not None
            arr = self._ctrl_val
            if arr.ndim == 2:
                new_ctrl = Du @ arr @ Dv.T
            else:
                k_ch = arr.shape[2]
                new_ctrl = np.empty((Du.shape[0], Dv.shape[0], k_ch), dtype=np.float64)
                for ch in range(k_ch):
                    new_ctrl[:, :, ch] = Du @ arr[:, :, ch] @ Dv.T

        return BSplineSurface2D(
            ctrl_pts=new_ctrl,
            knots_u=new_ku,
            knots_v=new_kv,
            degree_u=new_degu,
            degree_v=new_degv,
            extrapolate=self._extrapolate,
        )


class BSplineInterpolator2D(BSplineSurface2D):
    """
    Reusable 2D C2-smooth tensor-product B-Spline surface interpolator over `(x_grid, y_grid, z_data)`.
    - Solves the 2D tensor-product collocation linear systems $A_x P A_y^T = Z$ ONCE in `__init__`!
    - Subsequent calls `surf(xq, yq)` reuse the solved control net with zero repeated linear solves
      and share a single set of `ConstantNode`s in the C++ computation graph.
    """

    def __init__(
        self,
        x_grid: Sequence[float],
        y_grid: Sequence[float],
        z_data: Union[Expr, ArrayLike],
        degree_x: int = 3,
        degree_y: int = 3,
        extrapolate: bool = False,
        n_ctrl_x: Optional[int] = None,
        n_ctrl_y: Optional[int] = None,
        reg_weight: float = 0.0,
    ) -> None:
        """
        Pre-compute 2D tensor-product B-spline surface control points from `(x_grid, y_grid, z_data)` ONCE.

        Args:
            x_grid: 1D strictly increasing X coordinates of length `Nx`.
            y_grid: 1D strictly increasing Y coordinates of length `Ny`.
            z_data: Grid values of shape `(Nx, Ny)` or `(Nx, Ny, K)` (numeric array or symbolic `Expr`).
            degree_x: B-spline degree along X (default: `3` for C2 cubic).
            degree_y: B-spline degree along Y (default: `3` for C2 cubic).
            extrapolate: Extrapolate outside grid bounds if `True`, else clamp.
            n_ctrl_x: Optional reduced control point count along X (`< Nx`) for least-squares fitting.
            n_ctrl_y: Optional reduced control point count along Y (`< Ny`) for least-squares fitting.
            reg_weight: Second-difference smoothing regularization weight `>= 0.0`.
        """
        xg = np.asarray(x_grid, dtype=np.float64).ravel()
        yg = np.asarray(y_grid, dtype=np.float64).ravel()
        nx, ny = len(xg), len(yg)
        if nx < 2 or ny < 2:
            raise ValueError("BSplineInterpolator2D: x_grid and y_grid must each have at least 2 points.")
        if np.any(np.diff(xg) <= 0) or np.any(np.diff(yg) <= 0):
            raise ValueError("BSplineInterpolator2D: x_grid and y_grid must be strictly increasing.")

        self._x_grid = xg
        self._y_grid = yg
        max_cx = nx if n_ctrl_x is None else int(n_ctrl_x)
        max_cy = ny if n_ctrl_y is None else int(n_ctrl_y)
        kx = min(int(degree_x), max_cx - 1)
        ky = min(int(degree_y), max_cy - 1)

        knots_x, op_x, exact_x = _solve_1d_spline_system(xg, kx, n_ctrl_x, float(reg_weight))
        knots_y, op_y, exact_y = _solve_1d_spline_system(yg, ky, n_ctrl_y, float(reg_weight))

        if isinstance(z_data, Expr):
            if z_data.shape != (nx, ny):
                raise ValueError(
                    f"BSplineInterpolator2D: symbolic z_data must have shape ({nx}, {ny}), got {z_data.shape}."
                )
            mx = np.linalg.inv(op_x) if exact_x else op_x
            my = np.linalg.inv(op_y) if exact_y else op_y
            ax_inv_e = _coerce_with_graph(z_data.graph, mx)
            ay_inv_t_e = _coerce_with_graph(z_data.graph, my.T)
            ctrl_pts: Union[Expr, np.ndarray] = ax_inv_e @ z_data @ ay_inv_t_e
        else:
            arr = np.asarray(z_data, dtype=np.float64)
            if arr.ndim == 2 and arr.shape == (nx, ny):
                temp = np.linalg.solve(op_x, arr) if exact_x else (op_x @ arr)
                ctrl_pts = (np.linalg.solve(op_y, temp.T) if exact_y else (op_y @ temp.T)).T
            elif arr.ndim == 3 and arr.shape[0] == nx and arr.shape[1] == ny:
                k_ch = arr.shape[2]
                nc_x = len(knots_x) - kx - 1
                nc_y = len(knots_y) - ky - 1
                ctrl_3d = np.empty((nc_x, nc_y, k_ch), dtype=np.float64)
                for ch in range(k_ch):
                    temp = np.linalg.solve(op_x, arr[:, :, ch]) if exact_x else (op_x @ arr[:, :, ch])
                    ctrl_3d[:, :, ch] = (np.linalg.solve(op_y, temp.T) if exact_y else (op_y @ temp.T)).T
                ctrl_pts = ctrl_3d
            else:
                raise ValueError(
                    f"BSplineInterpolator2D: z_data must have shape ({nx}, {ny}) or ({nx}, {ny}, K), got {arr.shape}."
                )

        super().__init__(
            ctrl_pts=ctrl_pts,
            knots_u=knots_x,
            knots_v=knots_y,
            degree_u=kx,
            degree_v=ky,
            extrapolate=extrapolate,
        )

    @property
    def x_grid(self) -> np.ndarray:
        """1D strictly increasing X coordinate grid."""
        return self._x_grid

    @property
    def y_grid(self) -> np.ndarray:
        """1D strictly increasing Y coordinate grid."""
        return self._y_grid


class LinearInterpolator3D:
    """
    Reusable 3D trilinear regular-grid interpolator over `(x_grid, y_grid, z_grid, w_data)`.
    - Pre-validates grids and pre-flattens `w_data` ONCE in `__init__`.
    - Shares a single set of C++ `ConstantNode`s across multiple queries in the same `Graph`.
    - Supports both symbolic `Expr` queries and direct numerical `float`/`np.ndarray` queries.
    """

    def __init__(
        self,
        x_grid: Sequence[float],
        y_grid: Sequence[float],
        z_grid: Sequence[float],
        w_data: Union[Expr, ArrayLike],
        extrapolate: bool = False,
    ) -> None:
        """
        Pre-compile a 3D trilinear regular-grid interpolator over `(x_grid, y_grid, z_grid, w_data)`.

        Args:
            x_grid: 1D strictly increasing X coordinates of length `Nx`.
            y_grid: 1D strictly increasing Y coordinates of length `Ny`.
            z_grid: 1D strictly increasing Z coordinates of length `Nz`.
            w_data: 3D/4D table of shape `(Nx, Ny, Nz)` or `(Nx, Ny, Nz, K)` (or symbolic `(Nx*Ny*Nz, 1)` `Expr`).
            extrapolate: Extrapolate outside grid bounds if `True`, else clamp.
        """
        self._x_grid = np.asarray(x_grid, dtype=np.float64).ravel()
        self._y_grid = np.asarray(y_grid, dtype=np.float64).ravel()
        self._z_grid = np.asarray(z_grid, dtype=np.float64).ravel()
        nx, ny, nz = self._x_grid.size, self._y_grid.size, self._z_grid.size
        if (
            nx < 2
            or ny < 2
            or nz < 2
            or np.any(np.diff(self._x_grid) <= 0)
            or np.any(np.diff(self._y_grid) <= 0)
            or np.any(np.diff(self._z_grid) <= 0)
        ):
            raise ValueError(
                "LinearInterpolator3D: x_grid, y_grid, and z_grid must be strictly increasing with >= 2 points."
            )
        self._extrapolate = bool(extrapolate)
        self._graph_cache: dict = {}
        n_grid = nx * ny * nz

        if isinstance(w_data, Expr):
            if w_data.size != n_grid:
                raise ValueError(
                    f"LinearInterpolator3D: symbolic Expr w_data must have total size {n_grid}, got shape {w_data.shape}."
                )
            self._w_expr: Optional[Expr] = w_data
            self._w_val: Optional[np.ndarray] = None
            self._n_channels = 1
            self._const_payload: List[Tuple[_core.Shape, np.ndarray]] = []
        else:
            arr = np.asarray(w_data, dtype=np.float64)
            if arr.ndim == 3 and arr.shape == (nx, ny, nz):
                arr_4d = arr[:, :, :, np.newaxis]
            elif arr.ndim == 4 and arr.shape[:3] == (nx, ny, nz):
                arr_4d = arr
            else:
                raise ValueError(
                    f"LinearInterpolator3D: w_data must have shape ({nx}, {ny}, {nz}) or ({nx}, {ny}, {nz}, K), got {arr.shape}."
                )
            self._w_expr = None
            self._w_val = arr.copy()
            self._n_channels = int(arr_4d.shape[3])
            s = _core.Shape(n_grid, 1)
            self._const_payload = [
                (s, arr_4d[:, :, :, ch].ravel(order="F"))
                for ch in range(self._n_channels)
            ]

    @property
    def x_grid(self) -> np.ndarray:
        """1D strictly increasing X coordinate grid."""
        return self._x_grid

    @property
    def y_grid(self) -> np.ndarray:
        """1D strictly increasing Y coordinate grid."""
        return self._y_grid

    @property
    def z_grid(self) -> np.ndarray:
        """1D strictly increasing Z coordinate grid."""
        return self._z_grid

    @property
    def w_data(self) -> Union[Expr, np.ndarray]:
        """Underlying 3D/4D table data (`np.ndarray` or symbolic `Expr`)."""
        return self._w_expr if self._w_expr is not None else self._w_val  # type: ignore[return-value]

    @property
    def extrapolate(self) -> bool:
        """Whether out-of-domain queries extrapolate (`True`) or clamp (`False`)."""
        return self._extrapolate

    def __call__(
        self,
        x_query: Union[Expr, ArrayLike],
        y_query: Union[Expr, ArrayLike],
        z_query: Union[Expr, ArrayLike],
    ) -> Union[Expr, float, np.ndarray]:
        """Evaluate the 3D trilinear interpolator at `(x_query, y_query, z_query)`."""
        if (
            isinstance(x_query, Expr)
            or isinstance(y_query, Expr)
            or isinstance(z_query, Expr)
            or self._w_expr is not None
        ):
            g = _find_graph(x_query, y_query, z_query, self._w_expr)
            xq_expr = _coerce_with_graph(g, x_query)
            yq_expr = _coerce_with_graph(g, y_query)
            zq_expr = _coerce_with_graph(g, z_query)
            if self._w_expr is not None:
                if self._w_expr.graph is not g:
                    raise ValueError("LinearInterpolator3D: queries and symbolic w_data belong to different Graphs.")
                nid = _core.op_linear_interp_3d(
                    g, xq_expr.id, yq_expr.id, zq_expr.id, self._w_expr.id,
                    self._x_grid, self._y_grid, self._z_grid, self._extrapolate
                )
                return Expr(g, nid)

            cids = _get_or_create_cached_const_ids(self._graph_cache, g, self._const_payload)
            cols = [
                Expr(
                    g,
                    _core.op_linear_interp_3d(
                        g, xq_expr.id, yq_expr.id, zq_expr.id, cid,
                        self._x_grid, self._y_grid, self._z_grid, self._extrapolate
                    ),
                )
                for cid in cids
            ]
            return concat(cols, axis=1) if len(cols) > 1 else cols[0]

        arr_x = np.asarray(x_query, dtype=np.float64)
        arr_y = np.asarray(y_query, dtype=np.float64)
        arr_z = np.asarray(z_query, dtype=np.float64)
        is_scalar = (
            (arr_x.ndim == 0 or arr_x.size == 1)
            and (arr_y.ndim == 0 or arr_y.size == 1)
            and (arr_z.ndim == 0 or arr_z.size == 1)
        )
        qx = arr_x.ravel()
        qy = arr_y.ravel()
        qz = arr_z.ravel()
        tmp_g = _core.Graph()
        x_id = tmp_g.add_variable(_core.Shape(qx.size, 1), qx, np.full(qx.size, -np.inf), np.full(qx.size, np.inf), "xq")
        y_id = tmp_g.add_variable(_core.Shape(qy.size, 1), qy, np.full(qy.size, -np.inf), np.full(qy.size, np.inf), "yq")
        z_id = tmp_g.add_variable(_core.Shape(qz.size, 1), qz, np.full(qz.size, -np.inf), np.full(qz.size, np.inf), "zq")
        out_exprs = []
        for s, flat in self._const_payload:
            cid = tmp_g.add_constant(s, flat, "")
            nid = _core.op_linear_interp_3d(
                tmp_g, x_id, y_id, z_id, cid, self._x_grid, self._y_grid, self._z_grid, self._extrapolate
            )
            out_exprs.append(Expr(tmp_g, nid))
        final_expr = concat(out_exprs, axis=1) if len(out_exprs) > 1 else out_exprs[0]
        ev = _core.Evaluator(tmp_g, [final_expr.id])
        res_mat = ev.eval(np.concatenate([qx, qy, qz]))[0]
        return _format_numeric_interp_output(res_mat, is_scalar, self._n_channels)


def _mode3d_apply_1d_ops_numeric(
    arr_4d: np.ndarray,
    op_x: np.ndarray,
    exact_x: bool,
    op_y: np.ndarray,
    exact_y: bool,
    op_z: np.ndarray,
    exact_z: bool,
) -> np.ndarray:
    """
    Apply 1D linear operators (either exact collocation solve `A^{-1} @ X` or matrix product `M @ X`)
    along axes 0, 1, and 2 of a 4D array `(nx, ny, nz, K)` via Mode-1, Mode-2, and Mode-3 tensor unfoldings.
    Complexity is strictly linear in total grid points: O((nx + ny + nz) * nx * ny * nz * K).
    """
    nx, ny, nz, k_ch = arr_4d.shape
    m0 = arr_4d.reshape(nx, ny * nz * k_ch)
    t0 = np.linalg.solve(op_x, m0) if exact_x else (op_x @ m0)
    nc_x = t0.shape[0]
    arr_1 = t0.reshape(nc_x, ny, nz, k_ch)

    m1 = np.transpose(arr_1, (1, 0, 2, 3)).reshape(ny, nc_x * nz * k_ch)
    t1 = np.linalg.solve(op_y, m1) if exact_y else (op_y @ m1)
    nc_y = t1.shape[0]
    arr_2 = np.transpose(t1.reshape(nc_y, nc_x, nz, k_ch), (1, 0, 2, 3))

    m2 = np.transpose(arr_2, (2, 0, 1, 3)).reshape(nz, nc_x * nc_y * k_ch)
    t2 = np.linalg.solve(op_z, m2) if exact_z else (op_z @ m2)
    nc_z = t2.shape[0]
    ctrl_4d = np.transpose(t2.reshape(nc_z, nc_x, nc_y, k_ch), (1, 2, 0, 3))
    return ctrl_4d


def _mode3d_apply_1d_ops_symbolic(
    w_expr: Expr,
    shape_3d: Tuple[int, int, int],
    mx: np.ndarray,
    my: np.ndarray,
    mz: np.ndarray,
) -> Expr:
    """
    Apply 1D matrices `(mx, my, mz)` along axes 0, 1, and 2 of a flattened column-major symbolic
    3D tensor `w_expr` of shape `(nx * ny * nz, 1)` using only 2D matrix multiplies, reshapes, and transposes.
    """
    g = w_expr.graph
    nx, ny, nz = shape_3d
    nc_x, nc_y, nc_z = mx.shape[0], my.shape[0], mz.shape[0]

    mx_e = _coerce_with_graph(g, mx)
    my_e = _coerce_with_graph(g, my)
    mz_e = _coerce_with_graph(g, mz)

    w_mat0 = w_expr.reshape(nx, ny * nz)
    t1 = mx_e @ w_mat0  # (nc_x, ny * nz)

    t1_t = t1.T.reshape(ny, nz * nc_x)
    t2 = my_e @ t1_t  # (nc_y, nz * nc_x)

    t2_t = t2.T.reshape(nz, nc_x * nc_y)
    t3 = mz_e @ t2_t  # (nc_z, nc_x * nc_y)

    ctrl_flat = t3.T.reshape(nc_x * nc_y * nc_z, 1)
    return ctrl_flat


class BSplineVolume3D:
    """
    Reusable 3D C2-smooth tensor-product B-Spline volume evaluator given pre-computed 3D control net
    and knot vectors `(knots_u, knots_v, knots_w)`.
    - Shares a single set of C++ `ConstantNode`s across multiple calls in the same `Graph`.
    - Callable on both symbolic `Expr` queries and numerical `float`/`np.ndarray` queries.
    - Supports exact analytical partial derivative volumes via `.derivative(nu_u, nu_v, nu_w)`.
    """

    def __init__(
        self,
        ctrl_pts: Union[Expr, ArrayLike],
        knots_u: Sequence[float],
        knots_v: Sequence[float],
        knots_w: Sequence[float],
        degree_u: int = 3,
        degree_v: int = 3,
        degree_w: int = 3,
        extrapolate: bool = False,
    ) -> None:
        """
        Pre-compile a 3D tensor-product B-spline volume from control net `ctrl_pts` and knot vectors.

        Args:
            ctrl_pts: Control net of shape `(Nu, Nv, Nw)` or `(Nu, Nv, Nw, K)` (or symbolic `(Nu*Nv*Nw, 1)` `Expr`).
            knots_u: Non-decreasing knot vector along U of length `Nu + degree_u + 1`.
            knots_v: Non-decreasing knot vector along V of length `Nv + degree_v + 1`.
            knots_w: Non-decreasing knot vector along W of length `Nw + degree_w + 1`.
            degree_u: B-spline degree along U (default: `3`).
            degree_v: B-spline degree along V (default: `3`).
            degree_w: B-spline degree along W (default: `3`).
            extrapolate: Extrapolate outside domain if `True`, else clamp.
        """
        self._knots_u = np.asarray(knots_u, dtype=np.float64).ravel()
        self._knots_v = np.asarray(knots_v, dtype=np.float64).ravel()
        self._knots_w = np.asarray(knots_w, dtype=np.float64).ravel()
        self._degree_u = int(degree_u)
        self._degree_v = int(degree_v)
        self._degree_w = int(degree_w)
        self._extrapolate = bool(extrapolate)
        self._graph_cache: dict = {}

        nu = self._knots_u.size - self._degree_u - 1
        nv = self._knots_v.size - self._degree_v - 1
        nw = self._knots_w.size - self._degree_w - 1
        if nu < 1 or nv < 1 or nw < 1:
            raise ValueError("BSplineVolume3D: invalid knot vector lengths for given degrees.")
        self._ctrl_shape_3d = (int(nu), int(nv), int(nw))
        n_grid = int(nu * nv * nw)

        if isinstance(ctrl_pts, Expr):
            if ctrl_pts.size != n_grid:
                raise ValueError(
                    f"BSplineVolume3D: symbolic ctrl_pts must have total size {n_grid}, got shape {ctrl_pts.shape}."
                )
            self._ctrl_expr: Optional[Expr] = (
                ctrl_pts if ctrl_pts.shape == (n_grid, 1) else ctrl_pts.reshape(n_grid, 1)
            )
            self._ctrl_val: Optional[np.ndarray] = None
            self._n_channels = 1
            self._const_payload: List[Tuple[_core.Shape, np.ndarray]] = []
        else:
            arr = np.asarray(ctrl_pts, dtype=np.float64)
            if arr.ndim == 3 and arr.shape == (nu, nv, nw):
                arr_4d = arr[:, :, :, np.newaxis]
            elif arr.ndim == 4 and arr.shape[:3] == (nu, nv, nw):
                arr_4d = arr
            else:
                raise ValueError(
                    f"BSplineVolume3D: ctrl_pts must have shape ({nu}, {nv}, {nw}) or ({nu}, {nv}, {nw}, K), got {arr.shape}."
                )
            self._ctrl_expr = None
            self._ctrl_val = arr.copy()
            self._n_channels = int(arr_4d.shape[3])
            s = _core.Shape(n_grid, 1)
            self._const_payload = [
                (s, arr_4d[:, :, :, ch].ravel(order="F"))
                for ch in range(self._n_channels)
            ]

    @property
    def knots_u(self) -> np.ndarray:
        """Knot vector along axis U."""
        return self._knots_u

    @property
    def knots_v(self) -> np.ndarray:
        """Knot vector along axis V."""
        return self._knots_v

    @property
    def knots_w(self) -> np.ndarray:
        """Knot vector along axis W."""
        return self._knots_w

    @property
    def degree_u(self) -> int:
        """B-spline degree along axis U."""
        return self._degree_u

    @property
    def degree_v(self) -> int:
        """B-spline degree along axis V."""
        return self._degree_v

    @property
    def degree_w(self) -> int:
        """B-spline degree along axis W."""
        return self._degree_w

    @property
    def ctrl_pts(self) -> Union[Expr, np.ndarray]:
        """Pre-compiled 3D/4D B-spline control net (`np.ndarray` or symbolic `Expr`)."""
        return self._ctrl_expr if self._ctrl_expr is not None else self._ctrl_val  # type: ignore[return-value]

    @property
    def extrapolate(self) -> bool:
        """Whether out-of-domain queries extrapolate (`True`) or clamp (`False`)."""
        return self._extrapolate

    def __call__(
        self,
        u_query: Union[Expr, ArrayLike],
        v_query: Union[Expr, ArrayLike],
        w_query: Union[Expr, ArrayLike],
    ) -> Union[Expr, float, np.ndarray]:
        """Evaluate the 3D B-spline volume at `(u_query, v_query, w_query)`."""
        if (
            isinstance(u_query, Expr)
            or isinstance(v_query, Expr)
            or isinstance(w_query, Expr)
            or self._ctrl_expr is not None
        ):
            g = _find_graph(u_query, v_query, w_query, self._ctrl_expr)
            uq_expr = _coerce_with_graph(g, u_query)
            vq_expr = _coerce_with_graph(g, v_query)
            wq_expr = _coerce_with_graph(g, w_query)
            if self._ctrl_expr is not None:
                if self._ctrl_expr.graph is not g:
                    raise ValueError("BSplineVolume3D: queries and symbolic ctrl_pts belong to different Graphs.")
                nid = _core.op_bspline_eval_3d(
                    g, uq_expr.id, vq_expr.id, wq_expr.id, self._ctrl_expr.id,
                    self._knots_u, self._knots_v, self._knots_w,
                    self._degree_u, self._degree_v, self._degree_w, self._extrapolate
                )
                return Expr(g, nid)

            cids = _get_or_create_cached_const_ids(self._graph_cache, g, self._const_payload)
            cols = [
                Expr(
                    g,
                    _core.op_bspline_eval_3d(
                        g, uq_expr.id, vq_expr.id, wq_expr.id, cid,
                        self._knots_u, self._knots_v, self._knots_w,
                        self._degree_u, self._degree_v, self._degree_w, self._extrapolate
                    ),
                )
                for cid in cids
            ]
            return concat(cols, axis=1) if len(cols) > 1 else cols[0]

        arr_u = np.asarray(u_query, dtype=np.float64)
        arr_v = np.asarray(v_query, dtype=np.float64)
        arr_w = np.asarray(w_query, dtype=np.float64)
        is_scalar = (
            (arr_u.ndim == 0 or arr_u.size == 1)
            and (arr_v.ndim == 0 or arr_v.size == 1)
            and (arr_w.ndim == 0 or arr_w.size == 1)
        )
        qu = arr_u.ravel()
        qv = arr_v.ravel()
        qw = arr_w.ravel()
        tmp_g = _core.Graph()
        u_id = tmp_g.add_variable(_core.Shape(qu.size, 1), qu, np.full(qu.size, -np.inf), np.full(qu.size, np.inf), "uq")
        v_id = tmp_g.add_variable(_core.Shape(qv.size, 1), qv, np.full(qv.size, -np.inf), np.full(qv.size, np.inf), "vq")
        w_id = tmp_g.add_variable(_core.Shape(qw.size, 1), qw, np.full(qw.size, -np.inf), np.full(qw.size, np.inf), "wq")
        out_exprs = []
        for s, flat in self._const_payload:
            cid = tmp_g.add_constant(s, flat, "")
            nid = _core.op_bspline_eval_3d(
                tmp_g, u_id, v_id, w_id, cid,
                self._knots_u, self._knots_v, self._knots_w,
                self._degree_u, self._degree_v, self._degree_w, self._extrapolate
            )
            out_exprs.append(Expr(tmp_g, nid))
        final_expr = concat(out_exprs, axis=1) if len(out_exprs) > 1 else out_exprs[0]
        ev = _core.Evaluator(tmp_g, [final_expr.id])
        res_mat = ev.eval(np.concatenate([qu, qv, qw]))[0]
        return _format_numeric_interp_output(res_mat, is_scalar, self._n_channels)

    def derivative(self, nu_u: int = 0, nu_v: int = 0, nu_w: int = 0) -> BSplineVolume3D:
        """
        Return a new pre-compiled `BSplineVolume3D` representing the partial derivative volume
        `d^(nu_u + nu_v + nu_w) V(u, v, w) / (du^nu_u dv^nu_v dw^nu_w)`.
        """
        if int(nu_u) == 0 and int(nu_v) == 0 and int(nu_w) == 0:
            return self
        Du, new_ku, new_degu = _bspline_1d_diff_operator(self._knots_u, self._degree_u, int(nu_u))
        Dv, new_kv, new_degv = _bspline_1d_diff_operator(self._knots_v, self._degree_v, int(nu_v))
        Dw, new_kw, new_degw = _bspline_1d_diff_operator(self._knots_w, self._degree_w, int(nu_w))

        if self._ctrl_expr is not None:
            new_ctrl: Union[Expr, np.ndarray] = _mode3d_apply_1d_ops_symbolic(
                self._ctrl_expr, self._ctrl_shape_3d, Du, Dv, Dw
            )
        else:
            assert self._ctrl_val is not None
            arr = self._ctrl_val
            arr_4d = arr[:, :, :, np.newaxis] if arr.ndim == 3 else arr
            ctrl_4d = _mode3d_apply_1d_ops_numeric(arr_4d, Du, False, Dv, False, Dw, False)
            new_ctrl = ctrl_4d[:, :, :, 0] if arr.ndim == 3 else ctrl_4d

        return BSplineVolume3D(
            ctrl_pts=new_ctrl,
            knots_u=new_ku,
            knots_v=new_kv,
            knots_w=new_kw,
            degree_u=new_degu,
            degree_v=new_degv,
            degree_w=new_degw,
            extrapolate=self._extrapolate,
        )


class BSplineInterpolator3D(BSplineVolume3D):
    """
    Reusable 3D C2-smooth tensor-product B-Spline volume interpolator over `(x_grid, y_grid, z_grid, w_data)`.
    - Solves the 3D collocation or regularized least-squares system ONCE in `__init__` via
      Mode-1, Mode-2, and Mode-3 axis-separable 1D matrix sweeps in O((nx + ny + nz) * nx * ny * nz) time!
    - Subsequent calls `vol(xq, yq, zq)` reuse the solved control net with zero repeated linear solves
      and share a single set of `ConstantNode`s in the C++ computation graph.
    """

    def __init__(
        self,
        x_grid: Sequence[float],
        y_grid: Sequence[float],
        z_grid: Sequence[float],
        w_data: Union[Expr, ArrayLike],
        degree_x: int = 3,
        degree_y: int = 3,
        degree_z: int = 3,
        extrapolate: bool = False,
        n_ctrl_x: Optional[int] = None,
        n_ctrl_y: Optional[int] = None,
        n_ctrl_z: Optional[int] = None,
        reg_weight: float = 0.0,
    ) -> None:
        """
        Pre-compute 3D tensor-product B-spline control net from `(x_grid, y_grid, z_grid, w_data)` ONCE
        using fast Mode-1/2/3 axis-separable 1D linear solves (`O(Nx*Ny*Nz)` instead of `O((Nx*Ny*Nz)^3)`).

        Args:
            x_grid: 1D strictly increasing X coordinates of length `Nx`.
            y_grid: 1D strictly increasing Y coordinates of length `Ny`.
            z_grid: 1D strictly increasing Z coordinates of length `Nz`.
            w_data: 3D/4D grid values of shape `(Nx, Ny, Nz)` or `(Nx, Ny, Nz, K)` (or symbolic `Expr`).
            degree_x: B-spline degree along X (default: `3` for C2 cubic).
            degree_y: B-spline degree along Y (default: `3` for C2 cubic).
            degree_z: B-spline degree along Z (default: `3` for C2 cubic).
            extrapolate: Extrapolate outside grid bounds if `True`, else clamp.
            n_ctrl_x: Optional reduced control point count along X (`< Nx`).
            n_ctrl_y: Optional reduced control point count along Y (`< Ny`).
            n_ctrl_z: Optional reduced control point count along Z (`< Nz`).
            reg_weight: Second-difference smoothing regularization weight `>= 0.0`.
        """
        xg = np.asarray(x_grid, dtype=np.float64).ravel()
        yg = np.asarray(y_grid, dtype=np.float64).ravel()
        zg = np.asarray(z_grid, dtype=np.float64).ravel()
        nx, ny, nz = len(xg), len(yg), len(zg)
        if nx < 2 or ny < 2 or nz < 2:
            raise ValueError("BSplineInterpolator3D: x_grid, y_grid, and z_grid must each have >= 2 points.")
        if np.any(np.diff(xg) <= 0) or np.any(np.diff(yg) <= 0) or np.any(np.diff(zg) <= 0):
            raise ValueError("BSplineInterpolator3D: x_grid, y_grid, and z_grid must be strictly increasing.")

        self._x_grid = xg
        self._y_grid = yg
        self._z_grid = zg

        max_cx = nx if n_ctrl_x is None else int(n_ctrl_x)
        max_cy = ny if n_ctrl_y is None else int(n_ctrl_y)
        max_cz = nz if n_ctrl_z is None else int(n_ctrl_z)
        kx = min(int(degree_x), max_cx - 1)
        ky = min(int(degree_y), max_cy - 1)
        kz = min(int(degree_z), max_cz - 1)

        knots_x, op_x, exact_x = _solve_1d_spline_system(xg, kx, n_ctrl_x, float(reg_weight))
        knots_y, op_y, exact_y = _solve_1d_spline_system(yg, ky, n_ctrl_y, float(reg_weight))
        knots_z, op_z, exact_z = _solve_1d_spline_system(zg, kz, n_ctrl_z, float(reg_weight))

        if isinstance(w_data, Expr):
            if w_data.size != nx * ny * nz:
                raise ValueError(
                    f"BSplineInterpolator3D: symbolic w_data must have total size {nx * ny * nz}, got shape {w_data.shape}."
                )
            mx = np.linalg.inv(op_x) if exact_x else op_x
            my = np.linalg.inv(op_y) if exact_y else op_y
            mz = np.linalg.inv(op_z) if exact_z else op_z
            w_col = w_data if w_data.shape == (nx * ny * nz, 1) else w_data.reshape(nx * ny * nz, 1)
            ctrl_pts: Union[Expr, np.ndarray] = _mode3d_apply_1d_ops_symbolic(
                w_col, (nx, ny, nz), mx, my, mz
            )
        else:
            arr = np.asarray(w_data, dtype=np.float64)
            if arr.ndim == 3 and arr.shape == (nx, ny, nz):
                arr_4d = arr[:, :, :, np.newaxis]
                ctrl_4d = _mode3d_apply_1d_ops_numeric(arr_4d, op_x, exact_x, op_y, exact_y, op_z, exact_z)
                ctrl_pts = ctrl_4d[:, :, :, 0]
            elif arr.ndim == 4 and arr.shape[:3] == (nx, ny, nz):
                ctrl_pts = _mode3d_apply_1d_ops_numeric(arr, op_x, exact_x, op_y, exact_y, op_z, exact_z)
            else:
                raise ValueError(
                    f"BSplineInterpolator3D: w_data must have shape ({nx}, {ny}, {nz}) or ({nx}, {ny}, {nz}, K), got {arr.shape}."
                )

        super().__init__(
            ctrl_pts=ctrl_pts,
            knots_u=knots_x,
            knots_v=knots_y,
            knots_w=knots_z,
            degree_u=kx,
            degree_v=ky,
            degree_w=kz,
            extrapolate=extrapolate,
        )

    @property
    def x_grid(self) -> np.ndarray:
        """1D strictly increasing X coordinate grid."""
        return self._x_grid

    @property
    def y_grid(self) -> np.ndarray:
        """1D strictly increasing Y coordinate grid."""
        return self._y_grid

    @property
    def z_grid(self) -> np.ndarray:
        """1D strictly increasing Z coordinate grid."""
        return self._z_grid


# ============================================================================
# Functional Interpolation Wrappers (Delegating to Pre-Compiled Classes)
# ============================================================================
def interp_linear(
    x_query: Union[Expr, ArrayLike],
    x_grid: Sequence[float],
    y_data: Union[Expr, ArrayLike],
    extrapolate: bool = False,
) -> Expr:
    """1D piecewise-linear interpolation (functional wrapper around `LinearInterpolator1D`)."""
    res = LinearInterpolator1D(x_grid, y_data, extrapolate=extrapolate)(x_query)
    assert isinstance(res, Expr)
    return res


def bspline_eval(
    u_query: Union[Expr, ArrayLike],
    ctrl_pts: Union[Expr, ArrayLike],
    knots: Sequence[float],
    degree: int = 3,
    extrapolate: bool = False,
) -> Expr:
    """1D B-spline curve evaluation (functional wrapper around `BSplineCurve1D`)."""
    res = BSplineCurve1D(ctrl_pts, knots, degree=degree, extrapolate=extrapolate)(u_query)
    assert isinstance(res, Expr)
    return res


def interp_bspline(
    x_query: Union[Expr, ArrayLike],
    x_grid: Sequence[float],
    y_data: Union[Expr, ArrayLike],
    degree: int = 3,
    extrapolate: bool = False,
) -> Expr:
    """1D interpolating B-spline (functional wrapper around `BSplineInterpolator1D`)."""
    res = BSplineInterpolator1D(x_grid, y_data, degree=degree, extrapolate=extrapolate)(x_query)
    assert isinstance(res, Expr)
    return res


def interp_linear_2d(
    x_query: Union[Expr, ArrayLike],
    y_query: Union[Expr, ArrayLike],
    x_grid: Sequence[float],
    y_grid: Sequence[float],
    z_data: Union[Expr, ArrayLike],
    extrapolate: bool = False,
) -> Expr:
    """2D bilinear interpolation (functional wrapper around `LinearInterpolator2D`)."""
    res = LinearInterpolator2D(x_grid, y_grid, z_data, extrapolate=extrapolate)(x_query, y_query)
    assert isinstance(res, Expr)
    return res


def bspline_eval_2d(
    u_query: Union[Expr, ArrayLike],
    v_query: Union[Expr, ArrayLike],
    ctrl_pts: Union[Expr, ArrayLike],
    knots_u: Sequence[float],
    knots_v: Sequence[float],
    degree_u: int = 3,
    degree_v: int = 3,
    extrapolate: bool = False,
) -> Expr:
    """2D tensor-product B-spline surface evaluation (functional wrapper around `BSplineSurface2D`)."""
    res = BSplineSurface2D(
        ctrl_pts, knots_u, knots_v, degree_u=degree_u, degree_v=degree_v, extrapolate=extrapolate
    )(u_query, v_query)
    assert isinstance(res, Expr)
    return res


def interp_bspline_2d(
    x_query: Union[Expr, ArrayLike],
    y_query: Union[Expr, ArrayLike],
    x_grid: Sequence[float],
    y_grid: Sequence[float],
    z_data: Union[Expr, ArrayLike],
    degree_x: int = 3,
    degree_y: int = 3,
    extrapolate: bool = False,
) -> Expr:
    """2D interpolating B-spline surface (functional wrapper around `BSplineInterpolator2D`)."""
    res = BSplineInterpolator2D(
        x_grid, y_grid, z_data, degree_x=degree_x, degree_y=degree_y, extrapolate=extrapolate
    )(x_query, y_query)
    assert isinstance(res, Expr)
    return res


def interp_linear_3d(
    x_query: Union[Expr, ArrayLike],
    y_query: Union[Expr, ArrayLike],
    z_query: Union[Expr, ArrayLike],
    x_grid: Sequence[float],
    y_grid: Sequence[float],
    z_grid: Sequence[float],
    w_data: Union[Expr, ArrayLike],
    extrapolate: bool = False,
) -> Expr:
    """3D trilinear interpolation (functional wrapper around `LinearInterpolator3D`)."""
    res = LinearInterpolator3D(x_grid, y_grid, z_grid, w_data, extrapolate=extrapolate)(x_query, y_query, z_query)
    assert isinstance(res, Expr)
    return res


def bspline_eval_3d(
    u_query: Union[Expr, ArrayLike],
    v_query: Union[Expr, ArrayLike],
    w_query: Union[Expr, ArrayLike],
    ctrl_pts: Union[Expr, ArrayLike],
    knots_u: Sequence[float],
    knots_v: Sequence[float],
    knots_w: Sequence[float],
    degree_u: int = 3,
    degree_v: int = 3,
    degree_w: int = 3,
    extrapolate: bool = False,
) -> Expr:
    """3D tensor-product B-spline volume evaluation (functional wrapper around `BSplineVolume3D`)."""
    res = BSplineVolume3D(
        ctrl_pts, knots_u, knots_v, knots_w,
        degree_u=degree_u, degree_v=degree_v, degree_w=degree_w, extrapolate=extrapolate
    )(u_query, v_query, w_query)
    assert isinstance(res, Expr)
    return res


def interp_bspline_3d(
    x_query: Union[Expr, ArrayLike],
    y_query: Union[Expr, ArrayLike],
    z_query: Union[Expr, ArrayLike],
    x_grid: Sequence[float],
    y_grid: Sequence[float],
    z_grid: Sequence[float],
    w_data: Union[Expr, ArrayLike],
    degree_x: int = 3,
    degree_y: int = 3,
    degree_z: int = 3,
    extrapolate: bool = False,
) -> Expr:
    """3D interpolating B-spline volume (functional wrapper around `BSplineInterpolator3D`)."""
    res = BSplineInterpolator3D(
        x_grid, y_grid, z_grid, w_data,
        degree_x=degree_x, degree_y=degree_y, degree_z=degree_z, extrapolate=extrapolate
    )(x_query, y_query, z_query)
    assert isinstance(res, Expr)
    return res


def where(cond: Expr, true_val: Union[Expr, ArrayLike], false_val: Union[Expr, ArrayLike]) -> Expr:
    """Element-wise hard conditional selection with NaN-safe backward/Hessian masking."""
    t_expr = _coerce_with_graph(cond.graph, true_val)
    f_expr = _coerce_with_graph(cond.graph, false_val)
    return Expr(cond.graph, _core.op_where(cond.graph, cond.id, t_expr.id, f_expr.id))


def smooth_where(
    x: Expr,
    true_val: Union[Expr, ArrayLike],
    false_val: Union[Expr, ArrayLike],
    threshold: float = 0.0,
    sharpness: float = 10.0,
) -> Expr:
    """Smooth C2-differentiable transition branch based on 0.5 * (1 + tanh(sharpness * (x - threshold)))."""
    t_expr = x._coerce(true_val)
    f_expr = x._coerce(false_val)
    return Expr(
        x.graph,
        _core.op_smooth_where(x.graph, x.id, t_expr.id, f_expr.id, float(threshold), float(sharpness)),
    )


def cond(
    scalar_cond: Expr,
    inputs: Sequence[Expr],
    true_fn: Callable[..., Expr],
    false_fn: Callable[..., Expr],
) -> Expr:
    """Short-circuit subgraph conditional node: evaluates and differentiates only the taken branch."""
    t_graph = _core.Graph()
    f_graph = _core.Graph()
    t_vars = []
    f_vars = []
    for inp in inputs:
        s = inp.graph.node_shape(inp.id)
        zeros = np.zeros(s.size(), dtype=np.float64)
        inf_b = np.full(s.size(), 1e20, dtype=np.float64)
        t_vars.append(Expr(t_graph, t_graph.add_variable(s, zeros, -inf_b, inf_b, "")))
        f_vars.append(Expr(f_graph, f_graph.add_variable(s, zeros, -inf_b, inf_b, "")))

    t_out = true_fn(*t_vars)
    f_out = false_fn(*f_vars)
    arg_ids = [inp.id for inp in inputs]
    nid = _core.op_cond_subgraph(
        scalar_cond.graph,
        scalar_cond.id,
        arg_ids,
        t_graph,
        t_out.id,
        f_graph,
        f_out.id,
    )
    return Expr(scalar_cond.graph, nid)


def implicit_solve(
    inputs: Sequence[Expr],
    residual_fn: Callable[..., Expr],
    init_guess: ArrayLike,
    shape: Optional[ShapeLike] = None,
    lb: Optional[ArrayLike] = None,
    ub: Optional[ArrayLike] = None,
    max_iter: int = 50,
    tol: float = 1e-11,
) -> Expr:
    """
    Create an embedded ImplicitSolverNode solving R(x_inputs, y) = 0 via C++ damped Newton-Raphson
    and computing exact 1st-order and 2nd-order derivatives via the Implicit Function Theorem (IFT).
    """
    if not inputs:
        raise ValueError("implicit_solve requires at least one outer input expression.")
    outer_graph = inputs[0].graph

    if shape is None:
        arr_guess = np.asarray(init_guess, dtype=np.float64)
        if arr_guess.ndim <= 1:
            s = _core.Shape(arr_guess.size, 1)
        else:
            s = _core.Shape(arr_guess.shape[0], arr_guess.shape[1])
    else:
        s = _normalize_shape(shape)

    guess_flat = _to_fortran_flat(init_guess, s, 0.0)
    lb_flat = _to_fortran_flat(lb, s, -1e20)
    ub_flat = _to_fortran_flat(ub, s, 1e20)

    res_graph = _core.Graph()
    inner_params = []
    for inp in inputs:
        in_s = outer_graph.node_shape(inp.id)
        zeros = np.zeros(in_s.size(), dtype=np.float64)
        inf_v = np.full(in_s.size(), 1e20, dtype=np.float64)
        pid = res_graph.add_variable(in_s, zeros, -inf_v, inf_v, "")
        inner_params.append(Expr(res_graph, pid))

    state_id = res_graph.add_implicit_state(s, guess_flat, lb_flat, ub_flat, "y_implicit")
    y_expr = Expr(res_graph, state_id)

    res_expr = residual_fn(*inner_params, y_expr)
    nid = _core.op_implicit_solve(
        outer_graph,
        [inp.id for inp in inputs],
        res_graph,
        res_expr.id,
        state_id,
        [p.id for p in inner_params],
        int(max_iter),
        float(tol),
    )
    return Expr(outer_graph, nid)


def vmap(
    fn: Callable[..., Expr],
    inputs: Union[Expr, Sequence[Union[Expr, ArrayLike]]],
    in_axes: Union[int, Sequence[Optional[int]]] = 0,
    shared_inputs: Optional[Sequence[Union[Expr, ArrayLike]]] = None,
    out_axis: int = 0,
    name: str = "",
) -> Expr:
    """
    Independent parallel loop operator (similar to CasADi `map` and JAX `vmap`).
    Traces `fn` ONCE into a compact C++ subgraph and applies it across all `B` slices
    in C++ with exact 1st-order and 2nd-order Forward-over-Reverse derivatives.

    Parameters:
      - fn: Callable taking `(slice_0, slice_1, ..., *shared_inputs)` and returning an `Expr`.
      - inputs: Single `Expr` or list/tuple of `Expr`s to map over.
      - in_axes: Int (`0` for row-slicing, `1` for column-slicing) or sequence of `0`/`1`/`None`
                 per input in `inputs` (`None` or `-1` means the input is broadcast unchanged).
      - shared_inputs: Optional sequence of unmapped `Expr`s appended after `inputs` with `in_axis = None`.
      - out_axis: `0` to stack slice outputs as rows `(B, M_out)`, or `1` to stack as columns `(M_out, B)`.
    """
    raw_list: List[Union[Expr, ArrayLike]] = (
        [inputs] if isinstance(inputs, Expr) or not isinstance(inputs, (list, tuple)) else list(inputs)
    )
    if shared_inputs is not None:
        shared_list = list(shared_inputs)
    else:
        shared_list = []

    all_raw = raw_list + shared_list
    outer_graph = _find_graph(*all_raw)
    expr_list = [_coerce_with_graph(outer_graph, item) for item in raw_list]
    shared_expr_list = [_coerce_with_graph(outer_graph, item) for item in shared_list]

    if isinstance(in_axes, (int, np.integer)):
        axes_list: List[int] = [int(in_axes)] * len(expr_list)
    elif in_axes is None:
        axes_list = [-1] * len(expr_list)
    else:
        if len(in_axes) != len(expr_list):
            raise ValueError(f"vmap: len(in_axes)={len(in_axes)} does not match len(inputs)={len(expr_list)}.")
        axes_list = [-1 if ax is None else int(ax) for ax in in_axes]

    axes_list.extend([-1] * len(shared_expr_list))
    all_exprs = expr_list + shared_expr_list

    num_batches: Optional[int] = None
    slice_shapes: List[_core.Shape] = []
    for e, ax in zip(all_exprs, axes_list):
        r, c = e.shape
        if ax == -1:
            slice_shapes.append(_core.Shape(r, c))
        elif ax == 0:
            if num_batches is None:
                num_batches = r
            elif num_batches != r:
                raise ValueError(f"vmap batch size mismatch along axis 0: expected {num_batches}, got {r}.")
            slice_shapes.append(_core.Shape(c, 1))
        elif ax == 1:
            if num_batches is None:
                num_batches = c
            elif num_batches != c:
                raise ValueError(f"vmap batch size mismatch along axis 1: expected {num_batches}, got {c}.")
            slice_shapes.append(_core.Shape(r, 1))
        else:
            raise ValueError(f"vmap: in_axes must be 0, 1, or None/-1, got {ax}.")

    if num_batches is None or num_batches <= 0:
        raise ValueError("vmap requires at least one mapped input with a positive batch dimension.")

    body_graph = _core.Graph()
    inner_vars: List[Expr] = []
    for s in slice_shapes:
        zeros = np.zeros(s.size(), dtype=np.float64)
        inf_v = np.full(s.size(), 1e20, dtype=np.float64)
        vid = body_graph.add_variable(s, zeros, -inf_v, inf_v, "")
        inner_vars.append(Expr(body_graph, vid))

    body_out = fn(*inner_vars)
    if not isinstance(body_out, Expr) or body_out.graph is not body_graph:
        body_out = _coerce_with_graph(body_graph, body_out)

    m_slice = body_out.size
    if int(out_axis) == 0:
        out_shape = _core.Shape(num_batches, m_slice)
    elif int(out_axis) == 1:
        out_shape = _core.Shape(m_slice, num_batches)
    else:
        raise ValueError("vmap: out_axis must be 0 or 1.")

    nid = _core.op_map_subgraph(
        outer_graph,
        out_shape,
        [e.id for e in all_exprs],
        body_graph,
        body_out.id,
        [v.id for v in inner_vars],
        axes_list,
        int(out_axis),
        int(num_batches),
        name,
    )
    return Expr(outer_graph, nid)


def scan(
    step_fn: Callable[..., Union[Expr, Tuple[Expr, Expr]]],
    init_carry: Union[Expr, ArrayLike],
    xs: Optional[Union[Expr, Sequence[Union[Expr, ArrayLike]]]] = None,
    shared_inputs: Optional[Sequence[Union[Expr, ArrayLike]]] = None,
    length: Optional[int] = None,
    seq_axes: Union[int, Sequence[int]] = 0,
    out_axis: int = 0,
    name: str = "",
) -> Tuple[Expr, Expr]:
    """
    Sequential recurrent loop operator (similar to JAX `lax.scan` and CasADi `fold`).
    Traces `step_fn` ONCE into a compact single-step C++ subgraph and executes:
        (c_{t+1}, y_t) = step_fn(c_t, *x_t, *shared_inputs)   for t = 0 .. T - 1
    in native C++ with exact 1st-order and 2nd-order Forward-over-Reverse discrete-adjoint sensitivities.

    Parameters:
      - step_fn: Callable returning either `(next_carry, y_t)` or `next_carry` (in which case `y_t = next_carry`).
      - init_carry: Initial state `Expr` $c_0$.
      - xs: Optional sequence input `Expr` (or list of `Expr`s) sliced at each step $t$ along `seq_axes`.
      - shared_inputs: Optional list of time-invariant `Expr`s passed to every step.
      - length: Number of steps $T$ (inferred automatically from `xs` if `xs` is provided).
      - seq_axes: `0` (row per step, shape `(T, dim)`) or `1` (column per step, shape `(dim, T)`).
      - out_axis: `0` to stack step outputs `ys` as `(T, n_y)`, or `1` as `(n_y, T)`.

    Returns:
      - `(final_carry, ys)` where `final_carry` has the shape of `init_carry` and `ys` stacks all $y_t$.
    """
    seq_raw: List[Union[Expr, ArrayLike]] = []
    if xs is not None:
        if isinstance(xs, Expr) or not isinstance(xs, (list, tuple)):
            seq_raw = [xs]
        else:
            seq_raw = list(xs)

    shared_raw: List[Union[Expr, ArrayLike]] = list(shared_inputs) if shared_inputs is not None else []
    outer_graph = _find_graph(init_carry, *seq_raw, *shared_raw)

    c0_expr = _coerce_with_graph(outer_graph, init_carry)
    seq_exprs = [_coerce_with_graph(outer_graph, item) for item in seq_raw]
    shared_exprs = [_coerce_with_graph(outer_graph, item) for item in shared_raw]

    if isinstance(seq_axes, (int, np.integer)):
        s_axes = [int(seq_axes)] * len(seq_exprs)
    else:
        if len(seq_axes) != len(seq_exprs):
            raise ValueError("scan: len(seq_axes) does not match len(xs).")
        s_axes = [int(ax) for ax in seq_axes]

    num_steps = int(length) if length is not None else None
    seq_slice_shapes: List[_core.Shape] = []
    for e, ax in zip(seq_exprs, s_axes):
        r, c = e.shape
        if ax == 0:
            if num_steps is None:
                num_steps = r
            elif num_steps != r:
                raise ValueError(f"scan sequence length mismatch along axis 0: expected {num_steps}, got {r}.")
            seq_slice_shapes.append(_core.Shape(c, 1))
        elif ax == 1:
            if num_steps is None:
                num_steps = c
            elif num_steps != c:
                raise ValueError(f"scan sequence length mismatch along axis 1: expected {num_steps}, got {c}.")
            seq_slice_shapes.append(_core.Shape(r, 1))
        else:
            raise ValueError("scan: seq_axes must be 0 or 1.")

    if num_steps is None or num_steps <= 0:
        raise ValueError("scan requires either non-empty `xs` or a positive `length` parameter.")

    step_graph = _core.Graph()
    c_shape = _core.Shape(*c0_expr.shape)
    c_zeros = np.zeros(c_shape.size(), dtype=np.float64)
    c_inf = np.full(c_shape.size(), 1e20, dtype=np.float64)
    inner_carry = Expr(step_graph, step_graph.add_variable(c_shape, c_zeros, -c_inf, c_inf, "carry"))

    inner_seqs: List[Expr] = []
    for s in seq_slice_shapes:
        zeros = np.zeros(s.size(), dtype=np.float64)
        inf_v = np.full(s.size(), 1e20, dtype=np.float64)
        inner_seqs.append(Expr(step_graph, step_graph.add_variable(s, zeros, -inf_v, inf_v, "x_t")))

    inner_shareds: List[Expr] = []
    for e in shared_exprs:
        s = _core.Shape(*e.shape)
        zeros = np.zeros(s.size(), dtype=np.float64)
        inf_v = np.full(s.size(), 1e20, dtype=np.float64)
        inner_shareds.append(Expr(step_graph, step_graph.add_variable(s, zeros, -inf_v, inf_v, "p_shared")))

    step_res = step_fn(inner_carry, *inner_seqs, *inner_shareds)
    if isinstance(step_res, (tuple, list)):
        if len(step_res) != 2:
            raise ValueError("scan step_fn must return either `next_carry` or `(next_carry, y_t)`.")
        next_carry_expr = _coerce_with_graph(step_graph, step_res[0])
        y_t_expr = _coerce_with_graph(step_graph, step_res[1])
    else:
        next_carry_expr = _coerce_with_graph(step_graph, step_res)
        y_t_expr = next_carry_expr

    if next_carry_expr.size != c0_expr.size:
        raise ValueError(
            f"scan: next_carry size ({next_carry_expr.size}) must match init_carry size ({c0_expr.size})."
        )

    n_c = c0_expr.size
    n_y = y_t_expr.size
    total_out_size = n_c + num_steps * n_y
    outer_in_ids = [c0_expr.id] + [e.id for e in seq_exprs] + [e.id for e in shared_exprs]

    scan_nid = _core.op_scan_subgraph(
        outer_graph,
        _core.Shape(total_out_size, 1),
        outer_in_ids,
        step_graph,
        inner_carry.id,
        [v.id for v in inner_seqs],
        s_axes,
        [v.id for v in inner_shareds],
        next_carry_expr.id,
        y_t_expr.id,
        int(num_steps),
        name,
    )
    raw_scan_expr = Expr(outer_graph, scan_nid)

    # Extract final_carry (first n_c elements) and stacked ys (remaining num_steps * n_y elements)
    final_carry = raw_scan_expr[0:n_c].reshape(c0_expr.shape[0], c0_expr.shape[1])
    # In C++ ScanNode, y_0, y_1, ..., y_{T-1} are stored consecutively in blocks of length n_y,
    # which in column-major layout corresponds to a matrix of shape (n_y, num_steps).
    ys_col_major = raw_scan_expr[n_c:total_out_size].reshape(n_y, num_steps)
    if int(out_axis) == 0:
        ys = ys_col_major.T
    elif int(out_axis) == 1:
        ys = ys_col_major
    else:
        raise ValueError("scan: out_axis must be 0 or 1.")

    return final_carry, ys


# ============================================================================
# GraphBuilder: Unified Variable/Parameter Declaration via freeze=False/True
# ============================================================================
class GraphBuilder:
    """
    Computation Graph Builder with unified variable(..., freeze=False/True) interface.
    When freeze=False (default), creates a C++ VariableNode (participates in AD and decision vector).
    When freeze=True, routes to a C++ ParameterNode (zero AD memory overhead, bounds silently ignored).
    """

    def __init__(self) -> None:
        """Initialize a new `GraphBuilder` with an empty C++ computation graph."""
        self._graph = _core.Graph()

    @property
    def cpp_graph(self) -> _core.Graph:
        """Return the underlying C++ `_fastopt_impl.Graph` instance."""
        return self._graph

    def variable(
        self,
        shape: ShapeLike = (1, 1),
        init: ArrayLike = 0.0,
        lb: Optional[ArrayLike] = None,
        ub: Optional[ArrayLike] = None,
        freeze: bool = False,
        name: str = "",
    ) -> Expr:
        """
        Declare an optimization variable (freeze=False) or fixed parameter (freeze=True).
        When freeze=True, lb and ub are silently ignored so switching requires flipping only freeze.
        """
        s = _normalize_shape(shape)
        init_flat = _to_fortran_flat(init, s, 0.0)
        if freeze:
            nid = self._graph.add_parameter(s, init_flat, name)
            return Expr(self._graph, nid)

        lb_flat = _to_fortran_flat(lb, s, -np.inf)
        ub_flat = _to_fortran_flat(ub, s, np.inf)
        nid = self._graph.add_variable(s, init_flat, lb_flat, ub_flat, name)
        return Expr(self._graph, nid)

    def parameter(self, shape: ShapeLike = (1, 1), value: ArrayLike = 0.0, name: str = "") -> Expr:
        """Alias for variable(..., init=value, freeze=True)."""
        return self.variable(shape=shape, init=value, freeze=True, name=name)

    def constant(self, value: ArrayLike, shape: Optional[ShapeLike] = None, name: str = "") -> Expr:
        """Create an immutable ConstantNode in the computation graph."""
        if shape is None:
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 0:
                return Expr(self._graph, self._graph.add_scalar_constant(float(arr.item())))
            elif arr.ndim == 1:
                s = _core.Shape(arr.shape[0], 1)
            elif arr.ndim == 2:
                s = _core.Shape(arr.shape[0], arr.shape[1])
            else:
                raise ValueError("Constant array must be 0D, 1D, or 2D.")
        else:
            s = _normalize_shape(shape)
        flat = _to_fortran_flat(value, s, 0.0)
        return Expr(self._graph, self._graph.add_constant(s, flat, name))

    def build_evaluator(self, *outputs: Union[Expr, Sequence[Expr]]) -> _core.Evaluator:
        """
        Compile the computation graph for the given output `Expr`(s) and return a C++ `Evaluator`.
        Accepts either a list/sequence `build_evaluator([f, g])` or variadic `build_evaluator(f, g)`.
        """
        out_ids: List[int] = []
        for item in outputs:
            if isinstance(item, Expr):
                out_ids.append(item.id)
            else:
                out_ids.extend(o.id for o in item)
        return _core.Evaluator(self._graph, out_ids)


def sparse_jacobian(
    ev: _core.Evaluator,
    x: ArrayLike,
    out_idx: int = 0,
    num_threads: int = 0,
) -> Any:
    """
    Evaluate the sparse Jacobian matrix of output `out_idx` at `x` using Graph Coloring,
    and return a `scipy.sparse.csc_matrix` of shape `(m, n)`.
    """
    import scipy.sparse as sp

    x_arr = np.asarray(x, dtype=np.float64).ravel()
    rows, cols = ev.jacobian_sparsity(int(out_idx))
    vals = ev.sparse_jacobian_values(x_arr, int(out_idx), int(num_threads))
    m = ev.output_size(int(out_idx))
    n = ev.num_vars()
    return sp.csc_matrix((vals, (rows, cols)), shape=(m, n))


def sparse_hessian(
    ev: _core.Evaluator,
    x: ArrayLike,
    obj_scale: float = 1.0,
    constraint_multipliers: Optional[Sequence[ArrayLike]] = None,
    lower_only: bool = False,
    num_threads: int = 0,
) -> Any:
    """
    Evaluate the sparse Lagrangian Hessian at `x` using distance-2 Graph Coloring,
    and return a `scipy.sparse.csc_matrix` of shape `(n, n)`.
    When `lower_only=True`, returns only the lower-triangular part (`row >= col`).
    """
    import scipy.sparse as sp

    x_arr = np.asarray(x, dtype=np.float64).ravel()
    mults = [np.asarray(m, dtype=np.float64).ravel() for m in (constraint_multipliers or [])]
    tril_rows, tril_cols = ev.hessian_sparsity()
    tril_vals = ev.sparse_hessian_values(x_arr, float(obj_scale), mults, int(num_threads))
    n = ev.num_vars()
    if lower_only or tril_rows.size == 0:
        return sp.csc_matrix((tril_vals, (tril_rows, tril_cols)), shape=(n, n))

    off_diag = tril_rows != tril_cols
    all_rows = np.concatenate([tril_rows, tril_cols[off_diag]])
    all_cols = np.concatenate([tril_cols, tril_rows[off_diag]])
    all_vals = np.concatenate([tril_vals, tril_vals[off_diag]])
    return sp.csc_matrix((all_vals, (all_rows, all_cols)), shape=(n, n))


class Function:
    """
    CasADi-style reusable modular symbolic function wrapper `Function(name, inputs, outputs)`.
    Compiles a standalone sub-model ONCE and allows calling it:
      1. Numerically (`fn(a_val, b_val, ...)`): packs inputs, runs C++ `Evaluator`, and returns
         structured outputs (`float` or `np.ndarray`).
      2. Symbolically across ANY `Graph` or `Opti` instance (`fn(a_expr, b_expr, ...)`):
         embeds the compiled C++ subgraph directly via a 1-batch `MapNode` (`op_map_subgraph`)
         with zero Python re-tracing, preserving exact 1st/2nd-order AD and exact symbolic
         Jacobian/Hessian sparsity!
      3. Directly for derivatives & population batches: `fn.gradient(...)`, `fn.jacobian(...)`,
         `fn.hessian(...)`, `fn.eval_batch(X_pop)`.
    """

    def __init__(
        self,
        name: str,
        inputs: Sequence[Expr],
        outputs: Union[Expr, Sequence[Expr]],
    ) -> None:
        """
        Compile a reusable modular symbolic sub-model `Function(name, inputs, outputs)`.

        Args:
            name: Function identifier name.
            inputs: Sequence of active `Variable` expressions (`freeze=False`) representing sub-model inputs.
            outputs: Single output `Expr` or sequence of output `Expr`s built from `inputs`.
        """
        if not inputs:
            raise ValueError("Function requires a non-empty list of input Exprs.")
        self._name = str(name)
        self._in_exprs = list(inputs)
        self._graph = self._in_exprs[0].graph

        for idx, inp in enumerate(self._in_exprs):
            if inp.graph is not self._graph:
                raise ValueError(f"Function input {idx} belongs to a different Graph.")
            if inp.op_type != _core.OpType.Variable:
                raise ValueError(
                    f"Function input {idx} must be an active Variable Expr (freeze=False), got {inp.op_type.name}."
                )

        if isinstance(outputs, Expr):
            self._single_output = True
            self._out_exprs = [outputs]
        else:
            self._single_output = False
            self._out_exprs = list(outputs)
            if not self._out_exprs:
                raise ValueError("Function outputs list cannot be empty.")

        for idx, out in enumerate(self._out_exprs):
            if out.graph is not self._graph:
                raise ValueError(f"Function output {idx} belongs to a different Graph.")

        self._in_shapes: List[Tuple[int, int]] = [e.shape for e in self._in_exprs]
        self._in_sizes: List[int] = [r * c for r, c in self._in_shapes]
        self._in_ids: List[int] = [e.id for e in self._in_exprs]
        self._out_shapes: List[Tuple[int, int]] = [o.shape for o in self._out_exprs]
        self._out_sizes: List[int] = [r * c for r, c in self._out_shapes]

        # Build a combined flat output node for single-node subgraph embedding
        if len(self._out_exprs) == 1:
            self._combined_out = self._out_exprs[0].reshape(self._out_sizes[0], 1)
        else:
            flat_outs = [o.reshape(sz, 1) for o, sz in zip(self._out_exprs, self._out_sizes)]
            self._combined_out = concat(flat_outs, axis=0)

        self._total_out_size = sum(self._out_sizes)
        self._evaluator = _core.Evaluator(
            self._graph,
            [o.id for o in self._out_exprs] + [self._combined_out.id],
        )

    @property
    def name(self) -> str:
        """Function identifier name."""
        return self._name

    @property
    def evaluator(self) -> _core.Evaluator:
        """Underlying compiled C++ `Evaluator` instance."""
        return self._evaluator

    def _pack_numeric_args(self, args: Sequence[Any]) -> np.ndarray:
        if len(args) == 1 and len(self._in_exprs) > 1:
            arr0 = np.asarray(args[0], dtype=np.float64).ravel()
            if arr0.size == self._graph.total_var_size():
                return arr0
        if len(args) != len(self._in_exprs):
            raise ValueError(f"Function {self._name} expected {len(self._in_exprs)} inputs, got {len(args)}.")

        x_full = self._graph.initial_decision_vector()
        for inp, val, (r, c) in zip(self._in_exprs, args, self._in_shapes):
            off = self._graph.node_offset(inp.id)
            sz = r * c
            flat = _to_fortran_flat(val, _core.Shape(r, c))
            x_full[off : off + sz] = flat
        return x_full

    def __call__(self, *args: Union[Expr, ArrayLike]) -> Any:
        """
        Invoke the sub-model on symbolic `Expr` arguments (embedding a single `MapNode(B=1)` in the outer graph)
        or on numerical `float`/`np.ndarray` arguments.
        """
        if len(args) != len(self._in_exprs):
            raise ValueError(f"Function {self._name} expected {len(self._in_exprs)} arguments, got {len(args)}.")

        if any(isinstance(a, Expr) for a in args):
            target_g = _find_graph(*args)
            outer_exprs: List[Expr] = []
            for a, (r, c) in zip(args, self._in_shapes):
                e = _coerce_with_graph(target_g, a)
                if e.size != r * c:
                    raise ValueError(
                        f"Function {self._name} argument size {e.size} does not match expected shape {(r, c)}."
                    )
                if e.shape != (r, c):
                    e = e.reshape(r, c)
                outer_exprs.append(e)

            nid = _core.op_map_subgraph(
                target_g,
                _core.Shape(self._total_out_size, 1),
                [e.id for e in outer_exprs],
                self._graph,
                self._combined_out.id,
                self._in_ids,
                [-1] * len(self._in_ids),
                1,
                1,
                self._name,
            )
            raw_out = Expr(target_g, nid)
            if self._single_output:
                r0, c0 = self._out_shapes[0]
                return raw_out if (r0, c0) == (self._total_out_size, 1) else raw_out.reshape(r0, c0)

            res_exprs: List[Expr] = []
            offset = 0
            for (r, c), sz in zip(self._out_shapes, self._out_sizes):
                seg = raw_out[offset : offset + sz]
                res_exprs.append(seg if (r, c) == (sz, 1) else seg.reshape(r, c))
                offset += sz
            return tuple(res_exprs)

        # Numerical evaluation
        x_vec = self._pack_numeric_args(args)
        raw_mats = self._evaluator.eval(x_vec)[:-1]
        formatted: List[Any] = []
        for mat, (r, c) in zip(raw_mats, self._out_shapes):
            if r == 1 and c == 1:
                formatted.append(float(mat.item()))
            elif c == 1:
                formatted.append(mat.ravel().copy())
            else:
                formatted.append(mat.copy())
        return formatted[0] if self._single_output else tuple(formatted)

    def gradient(self, *args: ArrayLike, out_idx: int = 0) -> np.ndarray:
        """Compute exact 1D gradient vector of scalar output `out_idx` at numeric `*args`."""
        x_vec = self._pack_numeric_args(args)
        return self._evaluator.gradient(x_vec, int(out_idx))

    def jacobian(self, *args: ArrayLike, out_idx: int = 0) -> np.ndarray:
        """Compute 2D Jacobian matrix of output `out_idx` at numeric `*args`."""
        x_vec = self._pack_numeric_args(args)
        return self._evaluator.jacobian(x_vec, int(out_idx))

    def hessian(self, *args: ArrayLike, out_idx: int = 0, num_threads: int = 0) -> np.ndarray:
        """Compute 2D Hessian matrix of scalar output `out_idx` at numeric `*args`."""
        x_vec = self._pack_numeric_args(args)
        return self._evaluator.hessian(x_vec, int(out_idx), int(num_threads))

    def eval_batch(self, X_pop: np.ndarray, num_threads: int = 0) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Evaluate a population matrix `X_pop` of shape `(P, total_in_size)` in parallel via OpenMP."""
        res = self._evaluator.eval_batch(np.ascontiguousarray(X_pop, dtype=np.float64), int(num_threads))[:-1]
        return res[0] if self._single_output else tuple(res)



# ============================================================================
# Spanwise Integration & Trajectory / ODE Collocation Primitives
# ============================================================================
def trapz(
    y: Expr,
    x: Optional[Union[Expr, ArrayLike]] = None,
    dx: float = 1.0,
    axis: int = 0,
) -> Expr:
    """
    Symbolic composite trapezoidal integration along `axis=0` (default) or `axis=1`.
    Supports both uniform step `dx` and non-uniform 1D coordinate vector `x` (numeric or symbolic `Expr`).
    """
    ax = int(axis)
    if ax == 1:
        return trapz(y.T, x=x, dx=dx, axis=0).T
    if ax != 0 and ax != -1:
        raise ValueError("trapz: axis must be 0 or 1.")

    n = y.shape[0]
    if n < 2:
        raise ValueError(f"trapz requires at least 2 points along integration axis, got {n}.")

    y_avg = 0.5 * (y[0 : n - 1, :] + y[1:n, :])
    if x is None:
        return y_avg.sum(axis=0) * float(dx)

    x_e = _coerce_with_graph(y.graph, x)
    if x_e.size != n:
        raise ValueError(f"trapz: coordinate vector x size ({x_e.size}) must match y dimension ({n}).")
    x_col = x_e if x_e.shape == (n, 1) else x_e.reshape(n, 1)
    dx_col = x_col[1:n, :] - x_col[0 : n - 1, :]
    return (y_avg * dx_col).sum(axis=0)


# NumPy 2.0+ compatible alias for trapezoidal integration
trapezoid = trapz


def _eval_ode_rhs(
    rhs_fn: Callable[..., Expr],
    state: Expr,
    control: Optional[Union[Expr, ArrayLike]] = None,
) -> Expr:
    res = rhs_fn(state) if control is None else rhs_fn(state, control)
    if not isinstance(res, Expr):
        res = _coerce_with_graph(state.graph, res)
    if res.size != state.size:
        raise ValueError(f"ODE rhs_fn output size ({res.size}) must match state size ({state.size}).")
    return res if res.shape == state.shape else res.reshape(state.shape[0], state.shape[1])


def rk4_step(
    rhs_fn: Callable[..., Expr],
    x: Expr,
    u: Optional[Union[Expr, ArrayLike]] = None,
    dt: Union[Expr, float] = 0.01,
) -> Expr:
    """Single explicit 4th-order Runge-Kutta (RK4) integration step for `dx/dt = rhs_fn(x[, u])`."""
    k1 = _eval_ode_rhs(rhs_fn, x, u)
    k2 = _eval_ode_rhs(rhs_fn, x + (0.5 * dt) * k1, u)
    k3 = _eval_ode_rhs(rhs_fn, x + (0.5 * dt) * k2, u)
    k4 = _eval_ode_rhs(rhs_fn, x + dt * k3, u)
    return x + (dt * (1.0 / 6.0)) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def odeint_rk4(
    rhs_fn: Callable[..., Expr],
    x0: Expr,
    u_seq: Optional[Union[Expr, ArrayLike]] = None,
    dt: Union[Expr, float] = 0.01,
    num_steps: Optional[int] = None,
    seq_axis: int = 1,
    out_axis: int = 1,
) -> Tuple[Expr, Expr]:
    """
    Integrate `dx/dt = rhs_fn(x[, u])` over a multi-step horizon using 4th-order Runge-Kutta
    compiled into a single C++ `ScanNode` (`fo.scan`) with zero loop unrolling overhead.
    Returns `(x_final, X_traj)` where `x_final` has shape `(n_x, 1)` and `X_traj` has shape `(n_x, T)`.
    """
    x_init = x0 if x0.shape[1] == 1 else x0.reshape(x0.size, 1)
    g = x_init.graph

    if u_seq is not None:
        u_e = _coerce_with_graph(g, u_seq)
        ax = int(seq_axis)
        if isinstance(dt, Expr):
            def _step_dt(carry: Expr, u_t: Expr, dt_s: Expr):
                xn = rk4_step(rhs_fn, carry, u_t, dt_s)
                return xn, xn
            return scan(_step_dt, init_carry=x_init, xs=[u_e], shared_inputs=[dt], seq_axes=ax, out_axis=out_axis)
        else:
            dt_f = float(dt)
            def _step_const_dt(carry: Expr, u_t: Expr):
                xn = rk4_step(rhs_fn, carry, u_t, dt_f)
                return xn, xn
            return scan(_step_const_dt, init_carry=x_init, xs=[u_e], seq_axes=ax, out_axis=out_axis)

    if num_steps is None or int(num_steps) < 1:
        raise ValueError("odeint_rk4 requires `num_steps >= 1` when `u_seq` is None.")
    if isinstance(dt, Expr):
        def _step_auto_dt(carry: Expr, dt_s: Expr):
            xn = rk4_step(rhs_fn, carry, None, dt_s)
            return xn, xn
        return scan(_step_auto_dt, init_carry=x_init, xs=None, shared_inputs=[dt], length=int(num_steps), out_axis=out_axis)
    else:
        dt_f = float(dt)
        def _step_auto(carry: Expr):
            xn = rk4_step(rhs_fn, carry, None, dt_f)
            return xn, xn
        return scan(_step_auto, init_carry=x_init, xs=None, length=int(num_steps), out_axis=out_axis)


def collocate_trapz(
    rhs_fn: Callable[..., Expr],
    X: Expr,
    U: Optional[Expr] = None,
    dt: Union[Expr, float] = 0.01,
) -> Expr:
    """
    Compute direct trapezoidal collocation defect residuals across state matrix `X` of shape `(n_x, N+1)`
    and optional control matrix `U` of shape `(n_u, N+1)` in parallel via `fo.vmap` (`MapNode`).
    Returns defect matrix of shape `(n_x, N)` ready for `opti.subject_to(defects == 0.0)`.
    """
    n_x, n_nodes = X.shape
    if n_nodes < 2:
        raise ValueError("collocate_trapz requires at least 2 trajectory knot columns in X.")

    if U is None:
        F = vmap(lambda x_k: _eval_ode_rhs(rhs_fn, x_k, None), [X], in_axes=[1], out_axis=1)
    else:
        if U.shape[1] != n_nodes:
            raise ValueError(f"collocate_trapz: U columns ({U.shape[1]}) must match X columns ({n_nodes}).")
        F = vmap(lambda x_k, u_k: _eval_ode_rhs(rhs_fn, x_k, u_k), [X, U], in_axes=[1, 1], out_axis=1)

    X_prev = X[:, 0 : n_nodes - 1]
    X_next = X[:, 1:n_nodes]
    F_prev = F[:, 0 : n_nodes - 1]
    F_next = F[:, 1:n_nodes]
    return X_next - X_prev - (0.5 * dt) * (F_prev + F_next)


def collocate_hermite_simpson(
    rhs_fn: Callable[..., Expr],
    X: Expr,
    U: Optional[Expr] = None,
    dt: Union[Expr, float] = 0.01,
    U_mid: Optional[Expr] = None,
) -> Expr:
    """
    Compute 4th-order compressed Hermite-Simpson direct collocation defect residuals across state
    matrix `X` of shape `(n_x, N+1)` and optional control matrix `U` of shape `(n_u, N+1)` via `fo.vmap`.
    Returns defect matrix of shape `(n_x, N)` ready for `opti.subject_to(defects == 0.0)`.
    """
    n_x, n_nodes = X.shape
    if n_nodes < 2:
        raise ValueError("collocate_hermite_simpson requires at least 2 trajectory knot columns in X.")

    if U is None:
        F = vmap(lambda x_k: _eval_ode_rhs(rhs_fn, x_k, None), [X], in_axes=[1], out_axis=1)
    else:
        if U.shape[1] != n_nodes:
            raise ValueError(f"collocate_hermite_simpson: U columns ({U.shape[1]}) must match X columns ({n_nodes}).")
        F = vmap(lambda x_k, u_k: _eval_ode_rhs(rhs_fn, x_k, u_k), [X, U], in_axes=[1, 1], out_axis=1)

    X_prev = X[:, 0 : n_nodes - 1]
    X_next = X[:, 1:n_nodes]
    F_prev = F[:, 0 : n_nodes - 1]
    F_next = F[:, 1:n_nodes]

    # Cubic Hermite midpoint state interpolation
    X_mid = 0.5 * (X_prev + X_next) + (0.125 * dt) * (F_prev - F_next)

    if U is None:
        F_mid = vmap(lambda xm: _eval_ode_rhs(rhs_fn, xm, None), [X_mid], in_axes=[1], out_axis=1)
    else:
        U_m = U_mid if U_mid is not None else 0.5 * (U[:, 0 : n_nodes - 1] + U[:, 1:n_nodes])
        F_mid = vmap(lambda xm, um: _eval_ode_rhs(rhs_fn, xm, um), [X_mid, U_m], in_axes=[1, 1], out_axis=1)

    return X_next - X_prev - (dt * (1.0 / 6.0)) * (F_prev + 4.0 * F_mid + F_next)



def help_api(target: Optional[Any] = None, print_output: bool = True) -> str:
    """
    Generate a formatted function/class parameter signature and docstring hint guide.
    - If `target` is a specific `fastopt` function, class, or method (e.g. `fo.help_api(fo.BSplineInterpolator3D)`),
      returns its full signature, parameter types, defaults, and docstring, plus all public methods if it is a class.
    - If `target` is `None` (`fo.help_api()`), returns a categorized parameter hint cheat-sheet of all public `fastopt` APIs.

    Args:
        target: Optional function, class, method, or string symbol name in `fastopt`.
        print_output: Print the formatted parameter hint guide to stdout (default: `True`).

    Returns:
        Formatted parameter hint string.
    """
    import inspect
    import fastopt as _fo

    def _fmt_callable(name: str, obj: Any, indent: str = "") -> List[str]:
        lines_out: List[str] = []
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = "(...)"
        doc = inspect.getdoc(obj) or ""
        first_line = doc.splitlines()[0] if doc else ""
        lines_out.append(f"{indent}{name}{sig}")
        if first_line:
            lines_out.append(f"{indent}    -> {first_line}")
        return lines_out

    if isinstance(target, str):
        if hasattr(_fo, target):
            target = getattr(_fo, target)
        else:
            raise ValueError(f"Unknown fastopt symbol '{target}'.")

    out_lines: List[str] = []
    if target is not None:
        name = getattr(target, "__qualname__", getattr(target, "__name__", str(target)))
        out_lines.append(f"=== fastopt Parameter Hint: {name} ===")
        if inspect.isclass(target):
            out_lines.extend(_fmt_callable(name, target))
            cdoc = inspect.getdoc(target)
            if cdoc:
                out_lines.append("")
                for dl in cdoc.splitlines():
                    out_lines.append(f"  {dl}")
            out_lines.append("\n  Public Methods & Properties:")
            for attr_name, attr_val in inspect.getmembers(target):
                if attr_name.startswith("_") and attr_name not in ("__init__", "__call__", "__getitem__"):
                    continue
                if isinstance(attr_val, property):
                    pdoc = inspect.getdoc(attr_val.fget) or ""
                    pfirst = pdoc.splitlines()[0] if pdoc else ""
                    out_lines.append(f"    .{attr_name} (property) -> {pfirst}")
                elif callable(attr_val):
                    out_lines.extend(_fmt_callable(f".{attr_name}", attr_val, indent="    "))
        else:
            out_lines.extend(_fmt_callable(name, target))
            fdoc = inspect.getdoc(target)
            if fdoc:
                out_lines.append("")
                for dl in fdoc.splitlines():
                    out_lines.append(f"  {dl}")
    else:
        categories = [
            ("1. Optimization & Problem Building (`Opti`, `OptiSol`, `GraphBuilder`, `Function`)", [
                "Opti", "OptiSol", "GraphBuilder", "Function", "sparse_jacobian", "sparse_hessian"
            ]),
            ("2. Pre-Compiled Reusable 1D/2D/3D Interpolators", [
                "LinearInterpolator1D", "BSplineCurve1D", "BSplineInterpolator1D",
                "LinearInterpolator2D", "BSplineSurface2D", "BSplineInterpolator2D",
                "LinearInterpolator3D", "BSplineVolume3D", "BSplineInterpolator3D",
            ]),
            ("3. Trajectory Collocation, RK4 ODE & Spanwise Integration", [
                "trapz", "trapezoid", "rk4_step", "odeint_rk4", "collocate_trapz", "collocate_hermite_simpson"
            ]),
            ("4. Implicit Solver, Structured Loops (`vmap`, `scan`) & Branching", [
                "implicit_solve", "vmap", "scan", "where", "smooth_where", "cond"
            ]),
            ("5. Black-Box Operators (Python `@blackbox` & Zero-GIL C-ABI `CBlackBox`)", [
                "CustomOp", "blackbox", "CBlackBox", "c_blackbox"
            ]),
            ("6. Smooth Aggregations, Tensor Indexing & Math Operators", [
                "ks_max", "ks_min", "logsumexp", "smooth_abs", "smooth_max", "smooth_min", "smooth_clip",
                "clip", "sumsqr", "norm", "mean", "matmul", "solve", "cross", "trace", "dot", "sum", "concat", "vstack", "hstack", "diag",
                "deg2rad", "rad2deg", "atan2"
            ]),
        ]
        out_lines.append("=====================================================================================")
        out_lines.append("fastopt Complete API Function & Class Parameter Hint Reference")
        out_lines.append("=====================================================================================")
        for cat_title, sym_names in categories:
            out_lines.append(f"\n[{cat_title}]")
            for sname in sym_names:
                obj = getattr(_fo, sname, None)
                if obj is None:
                    continue
                out_lines.extend(_fmt_callable(f"fo.{sname}", obj, indent="  "))
                if inspect.isclass(obj) and sname in ("Opti", "OptiSol", "Function", "CBlackBox"):
                    for mname in ("variable", "parameter", "constant", "minimize", "subject_to", "solver",
                                  "solve", "solve_sweep", "solve_multistart", "load_state", "debug_infeasibility", "value", "dual", "to_dict",
                                  "sensitivity", "from_capsule", "from_pointers", "from_dll", "__call__", "derivative"):
                        if hasattr(obj, mname):
                            mobj = getattr(obj, mname)
                            if callable(mobj):
                                out_lines.extend(_fmt_callable(f"{sname}.{mname}", mobj, indent="      "))

    report = "\n".join(out_lines)
    if print_output:
        print(report)
    return report
