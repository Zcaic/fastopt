"""
Custom Black-Box Operator (`CustomOp` / `BlackBox` / `@blackbox`) and
Zero-GIL C-ABI Direct Bridge (`CBlackBox` / `c_blackbox`) for fastopt.

1. `CustomOp` / `@blackbox`:
   Wraps Python callables with analytical or central finite-difference 1st/2nd-order AD
   and optional vectorized `forward_batch` (`batch_fn`) for `pymoo`.
2. `CBlackBox` / `c_blackbox` (Zero-GIL C-ABI Direct Connection for `fast_bemt` / `fastgeo`):
   Connects external C/C++17/C++20 libraries (`fast_bemt`, `fastgeo`, or shared `.dll`/`.so`)
   directly to `fastopt::BlackBoxNode` via `PyCapsule` (`FastOptCBlackBox`), raw C function
   pointers, or `ctypes.CDLL` symbol names. Executes 0th, 1st, and 2nd-order AD in 100% native
   C++ without acquiring the Python GIL or allocating Python/NumPy objects, and unlocks
   full multi-core OpenMP parallelism in `eval_batch` and `lagrangian_hessian`.

All comments are written in English.
"""

from __future__ import annotations
import ctypes
import os
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np

from . import _fastopt_impl as _core
from .expr import (
    ArrayLike,
    Expr,
    ShapeLike,
    _coerce_with_graph,
    _find_graph,
    _normalize_shape,
)


def _unpack_inputs(
    u_flat: np.ndarray,
    shapes: Sequence[Tuple[int, int]],
    unpack_scalars: bool = False,
) -> List[Any]:
    """Unpack a flattened concatenated vector u into individual input arguments matching shapes."""
    args: List[Any] = []
    offset = 0
    for rows, cols in shapes:
        sz = rows * cols
        seg = u_flat[offset : offset + sz]
        offset += sz
        if unpack_scalars and sz == 1:
            args.append(float(seg[0]))
        elif cols == 1:
            args.append(seg.copy())
        else:
            args.append(seg.reshape((rows, cols), order="F"))
    return args


def _unpack_batch_inputs(
    U_batch: np.ndarray,
    shapes: Sequence[Tuple[int, int]],
) -> List[np.ndarray]:
    """
    Unpack a 2D population batch matrix U_batch of shape (n_pop, total_in)
    into per-argument 2D batch matrices of shape (n_pop, sz_k).
    """
    batch_args: List[np.ndarray] = []
    offset = 0
    for rows, cols in shapes:
        sz = rows * cols
        batch_args.append(U_batch[:, offset : offset + sz])
        offset += sz
    return batch_args


def _flatten_output(out_val: Any, out_rows: int, out_cols: int) -> np.ndarray:
    """Convert user forward() return value into a 1D column-major float64 array of size out_rows * out_cols."""
    expected_size = out_rows * out_cols
    arr = np.asarray(out_val, dtype=np.float64)
    if arr.size != expected_size:
        raise ValueError(
            f"CustomOp forward() returned size {arr.size}, expected {expected_size} "
            f"for shape ({out_rows}, {out_cols})."
        )
    if arr.ndim <= 1:
        return np.ascontiguousarray(arr.ravel(), dtype=np.float64)
    return np.ascontiguousarray(
        np.asfortranarray(arr.reshape(out_rows, out_cols)).ravel(order="F"),
        dtype=np.float64,
    )


def _normalize_jac_sparsity_mask(
    jac_sparsity: Optional[ArrayLike],
    out_size: int,
    total_in: int,
) -> np.ndarray:
    """Convert optional user Jacobian sparsity mask into a Fortran-contiguous int32 matrix."""
    if jac_sparsity is None:
        return np.zeros((0, 0), dtype=np.int32, order="F")
    arr = np.asarray(jac_sparsity, dtype=np.int32)
    if arr.shape != (out_size, total_in):
        raise ValueError(
            f"jac_sparsity mask shape {arr.shape} does not match (out_size={out_size}, total_in={total_in})."
        )
    return np.asfortranarray((arr != 0).astype(np.int32))


class CustomOp:
    """
    Base class and wrapper for user-defined black-box operators in fastopt.
    """

    def __init__(
        self,
        forward_fn: Optional[Callable[..., Any]] = None,
        shape: ShapeLike = (1, 1),
        jacobian_fn: Optional[Callable[..., Any]] = None,
        hessian_fn: Optional[Callable[..., Any]] = None,
        hvp_fn: Optional[Callable[..., Any]] = None,
        batch_fn: Optional[Callable[..., Any]] = None,
        grad_inputs: Optional[Sequence[int]] = None,
        jac_sparsity: Optional[ArrayLike] = None,
        fd_step: float = 1e-7,
        fd_hessian: bool = True,
        unpack_scalars: bool = False,
        name: str = "",
    ) -> None:
        """
        Initialize a Python `CustomOp` / `BlackBox` operator wrapper.

        Args:
            forward_fn: Forward evaluation callback `(*inputs) -> float | np.ndarray`.
            shape: Output 2D shape `(rows, cols)` or scalar integer length.
            jacobian_fn: Optional analytical Jacobian callback `(*inputs) -> np.ndarray | Sequence[np.ndarray]`.
            hessian_fn: Optional weighted Hessian callback `(*inputs, y_bar) -> np.ndarray`.
            hvp_fn: Optional Hessian-vector product callback `(*inputs, y_bar, u_dot) -> np.ndarray`.
            batch_fn: Optional batch forward callback `(*batch_inputs) -> np.ndarray` for population solvers.
            grad_inputs: Optional sequence of input argument indices that require derivatives (e.g. `[0, 1]`).
            jac_sparsity: Optional boolean/integer sparsity mask of shape `(out_size, total_in_size)`.
            fd_step: Relative central finite-difference step size when analytical derivatives are omitted.
            fd_hessian: Whether to approximate 2nd-order HVP when `hessian_fn`/`hvp_fn` are omitted.
            unpack_scalars: Unpack `(1, 1)` scalar inputs as Python `float` before invoking callbacks.
            name: Optional node identifier in the computation graph.
        """
        s = _normalize_shape(shape)
        self._out_rows: int = int(s.rows)
        self._out_cols: int = int(s.cols)
        self._forward_fn = forward_fn
        self._jacobian_fn = jacobian_fn
        self._hessian_fn = hessian_fn
        self._hvp_fn = hvp_fn
        self._batch_fn = batch_fn
        self._grad_inputs: Optional[Tuple[int, ...]] = (
            tuple(int(i) for i in grad_inputs) if grad_inputs is not None else None
        )
        self._jac_sparsity = jac_sparsity
        self._fd_step = float(fd_step)
        self._fd_hessian = bool(fd_hessian)
        self._unpack_scalars = bool(unpack_scalars)
        self._name = str(name)
        self.input_requires_grad: Tuple[bool, ...] = ()

    @property
    def shape(self) -> Tuple[int, int]:
        """Output 2D tensor shape `(rows, cols)`."""
        return (self._out_rows, self._out_cols)

    def forward(self, *inputs: Any) -> Any:
        """Evaluate the black-box forward mapping on numeric inputs `*inputs`."""
        if self._forward_fn is not None:
            return self._forward_fn(*inputs)
        raise NotImplementedError("CustomOp subclasses must implement forward(self, *inputs).")

    def forward_batch(self, *batch_inputs: np.ndarray) -> Any:
        """Evaluate vectorized population batch inputs of shape `(P, dim_k)` -> `(P, out_size)`."""
        if self._batch_fn is not None:
            return self._batch_fn(*batch_inputs)
        return None

    def jacobian(self, *inputs: Any) -> Any:
        """Return analytical Jacobian `(out_size, total_in_size)` or per-input blocks at `*inputs`."""
        if self._jacobian_fn is not None:
            return self._jacobian_fn(*inputs)
        return None

    def hessian(self, *inputs: Any, y_bar: np.ndarray) -> Any:
        """Return weighted Hessian `sum_k y_bar[k] * d^2 y_k / du^2` of shape `(total_in, total_in)`."""
        if self._hessian_fn is not None:
            return self._hessian_fn(*inputs, y_bar=y_bar)
        return None

    def hvp(self, *inputs: Any, y_bar: np.ndarray, u_dot: np.ndarray) -> Any:
        """Return directional Hessian-vector product `(sum_k y_bar[k] * d^2 y_k / du^2) @ u_dot`."""
        if self._hvp_fn is not None:
            return self._hvp_fn(*inputs, y_bar=y_bar, u_dot=u_dot)
        return None

    def _has_custom_batch(self) -> bool:
        if self._batch_fn is not None:
            return True
        return type(self).forward_batch is not CustomOp.forward_batch

    def _has_custom_jacobian(self) -> bool:
        if self._jacobian_fn is not None:
            return True
        return type(self).jacobian is not CustomOp.jacobian

    def _has_custom_hessian(self) -> bool:
        if self._hessian_fn is not None or self._hvp_fn is not None:
            return True
        return (type(self).hessian is not CustomOp.hessian) or (type(self).hvp is not CustomOp.hvp)

    def __call__(self, *inputs: Union[Expr, ArrayLike]) -> Union[Expr, np.ndarray, float]:
        """
        Apply the black-box operator to symbolic `Expr` inputs (adding a `BlackBoxNode` to the graph)
        or directly evaluate numeric inputs.
        """
        has_expr = any(isinstance(inp, Expr) for inp in inputs)
        if not has_expr:
            return self.forward(*inputs)

        g = _find_graph(*inputs)
        expr_inputs: List[Expr] = [_coerce_with_graph(g, inp) for inp in inputs]
        n_args = len(expr_inputs)
        in_ids: List[int] = [e.id for e in expr_inputs]
        in_shapes: List[Tuple[int, int]] = [e.shape for e in expr_inputs]
        in_sizes: List[int] = [r * c for r, c in in_shapes]
        total_in = sum(in_sizes)

        if self._grad_inputs is not None:
            grad_set = {i if i >= 0 else n_args + i for i in self._grad_inputs}
            active_flags: Tuple[bool, ...] = tuple(
                (i in grad_set) and expr_inputs[i].requires_grad for i in range(n_args)
            )
            declared_grad_indices: List[int] = [i for i in range(n_args) if i in grad_set]
        else:
            active_flags = tuple(e.requires_grad for e in expr_inputs)
            declared_grad_indices = [i for i, flag in enumerate(active_flags) if flag]

        self.input_requires_grad = active_flags

        in_offsets: List[Tuple[int, int]] = []
        curr_off = 0
        for sz in in_sizes:
            in_offsets.append((curr_off, curr_off + sz))
            curr_off += sz

        active_cols: List[int] = []
        for idx in declared_grad_indices:
            c0, c1 = in_offsets[idx]
            active_cols.extend(range(c0, c1))
        n_active_cols = len(active_cols)

        out_rows = self._out_rows
        out_cols = self._out_cols
        out_size = out_rows * out_cols
        unpack_sc = self._unpack_scalars

        def _py_eval(u_flat: np.ndarray) -> np.ndarray:
            args = _unpack_inputs(u_flat, in_shapes, unpack_sc)
            res = self.forward(*args)
            return _flatten_output(res, out_rows, out_cols)

        _py_batch: Optional[Callable[[np.ndarray], np.ndarray]] = None
        if self._has_custom_batch():

            def _batch_wrapper(U_batch: np.ndarray) -> np.ndarray:
                n_pop = U_batch.shape[0]
                b_args = _unpack_batch_inputs(U_batch, in_shapes)
                y_batch = self.forward_batch(*b_args)
                y_arr = np.asarray(y_batch, dtype=np.float64)
                if y_arr.size != n_pop * out_size:
                    raise ValueError(
                        f"CustomOp.forward_batch returned array of size {y_arr.size}, "
                        f"expected {n_pop * out_size} for shape ({n_pop}, {out_size})."
                    )
                return np.asfortranarray(y_arr.reshape(n_pop, out_size), dtype=np.float64)

            _py_batch = _batch_wrapper

        _py_jac: Optional[Callable[[np.ndarray], np.ndarray]] = None
        if self._has_custom_jacobian():

            def _jac_wrapper(u_flat: np.ndarray) -> np.ndarray:
                args = _unpack_inputs(u_flat, in_shapes, unpack_sc)
                j_raw = self.jacobian(*args)
                J_full = np.zeros((out_size, total_in), dtype=np.float64)

                if isinstance(j_raw, (tuple, list)):
                    if len(j_raw) == n_args:
                        for idx, (blk, (c0, c1)) in enumerate(zip(j_raw, in_offsets)):
                            if blk is None or (isinstance(blk, (int, float)) and float(blk) == 0.0):
                                continue
                            if not active_flags[idx]:
                                continue
                            J_full[:, c0:c1] = np.asarray(blk, dtype=np.float64).reshape(out_size, c1 - c0)
                    elif len(j_raw) == len(declared_grad_indices):
                        for idx, blk in zip(declared_grad_indices, j_raw):
                            if blk is None or (isinstance(blk, (int, float)) and float(blk) == 0.0):
                                continue
                            if not active_flags[idx]:
                                continue
                            c0, c1 = in_offsets[idx]
                            J_full[:, c0:c1] = np.asarray(blk, dtype=np.float64).reshape(out_size, c1 - c0)
                    else:
                        raise ValueError(
                            f"CustomOp.jacobian returned {len(j_raw)} blocks, expected {n_args} (all inputs) "
                            f"or {len(declared_grad_indices)} (active differentiable inputs)."
                        )
                else:
                    j_arr = np.asarray(j_raw, dtype=np.float64)
                    if j_arr.size == out_size * total_in:
                        J_full[:, :] = j_arr.reshape(out_size, total_in)
                        if self._grad_inputs is not None:
                            for idx in range(n_args):
                                if not active_flags[idx]:
                                    c0, c1 = in_offsets[idx]
                                    J_full[:, c0:c1] = 0.0
                    elif j_arr.size == out_size * n_active_cols and n_active_cols > 0:
                        J_sub = j_arr.reshape(out_size, n_active_cols)
                        J_full[:, active_cols] = J_sub
                    else:
                        raise ValueError(
                            f"CustomOp.jacobian returned array of size {j_arr.size}, expected "
                            f"{out_size * total_in} (all inputs) or {out_size * n_active_cols} (active inputs)."
                        )

                return np.asfortranarray(J_full, dtype=np.float64)

            _py_jac = _jac_wrapper

        _py_hvp: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None
        if self._has_custom_hessian():

            def _hvp_wrapper(u_flat: np.ndarray, y_bar: np.ndarray, u_dot: np.ndarray) -> np.ndarray:
                args = _unpack_inputs(u_flat, in_shapes, unpack_sc)
                yb = np.asarray(y_bar, dtype=np.float64).ravel()
                ud = np.asarray(u_dot, dtype=np.float64).ravel()

                if self._hvp_fn is not None or (type(self).hvp is not CustomOp.hvp):
                    res = self.hvp(*args, y_bar=yb, u_dot=ud)
                    res_arr = np.asarray(res, dtype=np.float64).ravel()
                    if res_arr.size == total_in:
                        return np.ascontiguousarray(res_arr, dtype=np.float64)
                    if res_arr.size == n_active_cols:
                        full_hvp = np.zeros(total_in, dtype=np.float64)
                        full_hvp[active_cols] = res_arr
                        return full_hvp
                    raise ValueError(
                        f"CustomOp.hvp returned size {res_arr.size}, expected {total_in} or {n_active_cols}."
                    )

                H_weighted = self.hessian(*args, y_bar=yb)
                if H_weighted is None:
                    return np.zeros(total_in, dtype=np.float64)
                H_arr = np.asarray(H_weighted, dtype=np.float64)
                if H_arr.size == total_in * total_in:
                    H_mat = H_arr.reshape(total_in, total_in)
                    return np.ascontiguousarray(H_mat @ ud, dtype=np.float64)
                if H_arr.size == n_active_cols * n_active_cols and n_active_cols > 0:
                    H_sub = H_arr.reshape(n_active_cols, n_active_cols)
                    ud_sub = ud[active_cols]
                    full_hvp = np.zeros(total_in, dtype=np.float64)
                    full_hvp[active_cols] = H_sub @ ud_sub
                    return full_hvp
                raise ValueError(
                    f"CustomOp.hessian returned size {H_arr.size}, expected "
                    f"{total_in * total_in} (all inputs) or {n_active_cols * n_active_cols} (active inputs)."
                )

            _py_hvp = _hvp_wrapper

        jac_mask = _normalize_jac_sparsity_mask(self._jac_sparsity, out_size, total_in)

        nid = _core.op_blackbox(
            g,
            _core.Shape(out_rows, out_cols),
            in_ids,
            _py_eval,
            _py_jac,
            _py_hvp,
            self._fd_step,
            self._fd_hessian,
            self._name,
            _py_batch,
            jac_mask,
        )
        return Expr(g, nid)


# Alias for CustomOp
BlackBox = CustomOp


def blackbox(
    fn: Optional[Callable[..., Any]] = None,
    *,
    shape: ShapeLike = (1, 1),
    jac: Optional[Callable[..., Any]] = None,
    hess: Optional[Callable[..., Any]] = None,
    hvp: Optional[Callable[..., Any]] = None,
    batch_fn: Optional[Callable[..., Any]] = None,
    grad_inputs: Optional[Sequence[int]] = None,
    jac_sparsity: Optional[ArrayLike] = None,
    fd_step: float = 1e-7,
    fd_hessian: bool = True,
    unpack_scalars: bool = False,
    name: str = "",
) -> Union[CustomOp, Callable[[Callable[..., Any]], CustomOp]]:
    """
    Decorator or wrapper function to turn any Python/C++ callable into a differentiable fastopt CustomOp.
    """

    def _wrap(f: Callable[..., Any]) -> CustomOp:
        op_name = name if name else getattr(f, "__name__", "blackbox_op")
        return CustomOp(
            forward_fn=f,
            shape=shape,
            jacobian_fn=jac,
            hessian_fn=hess,
            hvp_fn=hvp,
            batch_fn=batch_fn,
            grad_inputs=grad_inputs,
            jac_sparsity=jac_sparsity,
            fd_step=fd_step,
            fd_hessian=fd_hessian,
            unpack_scalars=unpack_scalars,
            name=op_name,
        )

    if fn is not None:
        return _wrap(fn)
    return _wrap


# ============================================================================
# Direction 5: Zero-GIL C-ABI Direct Black-Box Bridge (`CBlackBox`)
# ============================================================================
def _extract_c_ptr(fn_or_int: Any) -> int:
    """Extract raw C function/void pointer address (`uintptr_t`) from int, ctypes function, or c_void_p."""
    if fn_or_int is None or fn_or_int == 0:
        return 0
    if isinstance(fn_or_int, (int, np.integer)):
        return int(fn_or_int)
    if isinstance(fn_or_int, ctypes.c_void_p):
        return int(fn_or_int.value or 0)
    try:
        return int(ctypes.cast(fn_or_int, ctypes.c_void_p).value or 0)
    except Exception as exc:
        raise TypeError(f"Cannot extract raw C pointer address from {type(fn_or_int)}: {exc}") from exc


class CBlackBox:
    """
    Zero-GIL C-ABI BlackBox operator connecting external C/C++ libraries (such as `fast_bemt`
    or `fastgeo`) directly into `fastopt::BlackBoxNode`.

    Three construction modes:
      1. `CBlackBox.from_capsule(capsule, shape=..., ...)` (or `CBlackBox(capsule, shape=...)`):
         Accepts a `PyCapsule` wrapping a `FastOptCBlackBox*` struct (defined in `fastopt/c_abi.h`).
      2. `CBlackBox.from_pointers(eval_ptr, shape=..., jac_ptr=..., hvp_ptr=..., batch_ptr=..., user_data_ptr=...)`:
         Accepts raw C function pointer addresses (`uintptr_t` or `ctypes` function objects) and an
         optional `user_data_ptr` pointing to a live C++ solver instance (e.g., `fast_bemt::Solver*`).
      3. `CBlackBox.from_dll(dll_path, eval_symbol=..., shape=..., jac_symbol=..., ...)`:
         Loads a compiled shared library (`.dll` / `.so` / `.dylib`) and binds its exported C functions
         with zero Python callback overhead during evaluation and optimization.
    """

    def __init__(
        self,
        capsule: Optional[Any] = None,
        shape: ShapeLike = (1, 1),
        *,
        eval_ptr: Any = 0,
        jac_ptr: Any = 0,
        hvp_ptr: Any = 0,
        batch_ptr: Any = 0,
        user_data_ptr: Any = 0,
        is_thread_safe: bool = True,
        fd_step: float = 1e-7,
        fd_hessian: bool = True,
        jac_sparsity: Optional[ArrayLike] = None,
        name: str = "c_blackbox",
        keep_alive: Any = None,
    ) -> None:
        """
        Initialize a zero-GIL C-ABI `CBlackBox` operator.

        Args:
            capsule: Optional `PyCapsule` wrapping a `FastOptCBlackBox*` descriptor struct.
            shape: Output 2D shape `(rows, cols)` or scalar integer length.
            eval_ptr: Raw C function pointer (`uintptr_t` or `ctypes` function) for `FastOptCEvalFn`.
            jac_ptr: Optional C function pointer for `FastOptCJacFn` (`0` = central finite difference).
            hvp_ptr: Optional C function pointer for `FastOptCHvpFn` (`0` = finite difference HVP).
            batch_ptr: Optional C function pointer for `FastOptCBatchEvalFn`.
            user_data_ptr: Opaque `void*` context pointer passed to all C callbacks.
            is_thread_safe: Whether the C callbacks are re-entrant and safe for OpenMP multi-threading.
            fd_step: Relative finite-difference step size when `jac_ptr` or `hvp_ptr` is omitted.
            fd_hessian: Enable finite-difference 2nd-order HVP when `hvp_ptr` is omitted.
            jac_sparsity: Optional boolean/integer Jacobian sparsity mask of shape `(out_size, in_size)`.
            name: Node identifier name in the computation graph.
            keep_alive: Optional Python object (e.g., `ctypes.CDLL` or solver instance) kept alive with the graph.
        """
        s = _normalize_shape(shape)
        self._out_rows: int = int(s.rows)
        self._out_cols: int = int(s.cols)
        self._capsule = capsule
        self._eval_ptr = _extract_c_ptr(eval_ptr)
        self._jac_ptr = _extract_c_ptr(jac_ptr)
        self._hvp_ptr = _extract_c_ptr(hvp_ptr)
        self._batch_ptr = _extract_c_ptr(batch_ptr)
        self._user_data_ptr = _extract_c_ptr(user_data_ptr)
        self._is_thread_safe = bool(is_thread_safe)
        self._fd_step = float(fd_step)
        self._fd_hessian = bool(fd_hessian)
        self._jac_sparsity = jac_sparsity
        self._name = str(name)
        self._keep_alive = keep_alive

        if self._capsule is None and self._eval_ptr == 0:
            raise ValueError(
                "CBlackBox requires either a PyCapsule (`FastOptCBlackBox*`) or a non-null `eval_ptr`."
            )

    @property
    def shape(self) -> Tuple[int, int]:
        """Output 2D tensor shape `(rows, cols)`."""
        return (self._out_rows, self._out_cols)

    @classmethod
    def from_capsule(
        cls,
        capsule: Any,
        shape: ShapeLike = (1, 1),
        *,
        fd_step: float = 1e-7,
        fd_hessian: bool = True,
        jac_sparsity: Optional[ArrayLike] = None,
        name: str = "c_blackbox",
        keep_alive: Any = None,
    ) -> CBlackBox:
        """Create a zero-GIL `CBlackBox` from a `PyCapsule` wrapping `FastOptCBlackBox*`."""
        return cls(
            capsule=capsule,
            shape=shape,
            fd_step=fd_step,
            fd_hessian=fd_hessian,
            jac_sparsity=jac_sparsity,
            name=name,
            keep_alive=keep_alive,
        )

    @classmethod
    def from_pointers(
        cls,
        eval_ptr: Any,
        shape: ShapeLike = (1, 1),
        *,
        jac_ptr: Any = 0,
        hvp_ptr: Any = 0,
        batch_ptr: Any = 0,
        user_data_ptr: Any = 0,
        is_thread_safe: bool = True,
        fd_step: float = 1e-7,
        fd_hessian: bool = True,
        jac_sparsity: Optional[ArrayLike] = None,
        name: str = "c_blackbox",
        keep_alive: Any = None,
    ) -> CBlackBox:
        """Create a zero-GIL `CBlackBox` from raw function/instance pointer addresses or ctypes objects."""
        return cls(
            capsule=None,
            shape=shape,
            eval_ptr=eval_ptr,
            jac_ptr=jac_ptr,
            hvp_ptr=hvp_ptr,
            batch_ptr=batch_ptr,
            user_data_ptr=user_data_ptr,
            is_thread_safe=is_thread_safe,
            fd_step=fd_step,
            fd_hessian=fd_hessian,
            jac_sparsity=jac_sparsity,
            name=name,
            keep_alive=keep_alive,
        )

    @classmethod
    def from_dll(
        cls,
        dll_path: str,
        eval_symbol: str,
        shape: ShapeLike = (1, 1),
        *,
        jac_symbol: Optional[str] = None,
        hvp_symbol: Optional[str] = None,
        batch_symbol: Optional[str] = None,
        user_data_ptr: Any = 0,
        is_thread_safe: bool = True,
        fd_step: float = 1e-7,
        fd_hessian: bool = True,
        jac_sparsity: Optional[ArrayLike] = None,
        name: str = "",
        keep_alive: Any = None,
    ) -> CBlackBox:
        """
        Load a shared library (`.dll` / `.so` / `.dylib`) and bind its exported C-ABI functions
        directly into a zero-GIL `CBlackBox`.
        """
        lib = ctypes.CDLL(os.path.abspath(str(dll_path)))
        eval_fn = getattr(lib, eval_symbol)
        jac_fn = getattr(lib, jac_symbol) if jac_symbol else None
        hvp_fn = getattr(lib, hvp_symbol) if hvp_symbol else None
        batch_fn = getattr(lib, batch_symbol) if batch_symbol else None

        op_name = name if name else eval_symbol
        return cls.from_pointers(
            eval_ptr=eval_fn,
            shape=shape,
            jac_ptr=jac_fn,
            hvp_ptr=hvp_fn,
            batch_ptr=batch_fn,
            user_data_ptr=user_data_ptr,
            is_thread_safe=is_thread_safe,
            fd_step=fd_step,
            fd_hessian=fd_hessian,
            jac_sparsity=jac_sparsity,
            name=op_name,
            keep_alive=(lib, keep_alive),
        )

    def __call__(self, *inputs: Union[Expr, ArrayLike]) -> Expr:
        """Bind symbolic `Expr` inputs to this zero-GIL C-ABI black-box operator and return the output `Expr`."""
        if not inputs:
            raise ValueError("CBlackBox requires at least one input argument.")
        g = _find_graph(*inputs)
        expr_inputs: List[Expr] = [_coerce_with_graph(g, inp) for inp in inputs]
        in_ids: List[int] = [e.id for e in expr_inputs]
        total_in = sum(e.size for e in expr_inputs)
        out_size = self._out_rows * self._out_cols

        jac_mask = _normalize_jac_sparsity_mask(self._jac_sparsity, out_size, total_in)
        out_shape = _core.Shape(self._out_rows, self._out_cols)

        if self._capsule is not None:
            nid = _core.op_c_blackbox_capsule(
                g,
                out_shape,
                in_ids,
                self._capsule,
                self._fd_step,
                self._fd_hessian,
                jac_mask,
                self._name,
            )
        else:
            nid = _core.op_c_blackbox_raw(
                g,
                out_shape,
                in_ids,
                self._eval_ptr,
                self._jac_ptr,
                self._hvp_ptr,
                self._batch_ptr,
                self._user_data_ptr,
                self._is_thread_safe,
                self._fd_step,
                self._fd_hessian,
                jac_mask,
                self._name,
                self._keep_alive,
            )
        return Expr(g, nid)


def c_blackbox(
    capsule_or_eval_ptr: Any,
    shape: ShapeLike = (1, 1),
    *,
    jac_ptr: Any = 0,
    hvp_ptr: Any = 0,
    batch_ptr: Any = 0,
    user_data_ptr: Any = 0,
    is_thread_safe: bool = True,
    fd_step: float = 1e-7,
    fd_hessian: bool = True,
    jac_sparsity: Optional[ArrayLike] = None,
    name: str = "c_blackbox",
    keep_alive: Any = None,
) -> CBlackBox:
    """
    Convenience builder for a zero-GIL C-ABI `CBlackBox` from either a `PyCapsule`
    (`FastOptCBlackBox*`) or a raw C function pointer / ctypes function.
    """
    if type(capsule_or_eval_ptr).__name__ == "PyCapsule":
        return CBlackBox.from_capsule(
            capsule=capsule_or_eval_ptr,
            shape=shape,
            fd_step=fd_step,
            fd_hessian=fd_hessian,
            jac_sparsity=jac_sparsity,
            name=name,
            keep_alive=keep_alive,
        )
    return CBlackBox.from_pointers(
        eval_ptr=capsule_or_eval_ptr,
        shape=shape,
        jac_ptr=jac_ptr,
        hvp_ptr=hvp_ptr,
        batch_ptr=batch_ptr,
        user_data_ptr=user_data_ptr,
        is_thread_safe=is_thread_safe,
        fd_step=fd_step,
        fd_hessian=fd_hessian,
        jac_sparsity=jac_sparsity,
        name=name,
        keep_alive=keep_alive,
    )
