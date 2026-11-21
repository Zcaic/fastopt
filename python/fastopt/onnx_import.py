"""
Zero-dependency ONNX model importer for `fastopt.nn`.

Parses standard binary `.onnx` protobuf files directly in pure Python (no `onnx`
or `onnxruntime` package required, though fully compatible with `torch.onnx.export`
and `onnx` models), automatically fuses consecutive linear + LayerNorm + activation
+ residual chains into a single C++ `MLPNode`, lowers 1D `Conv` into C++ `Conv1DNode`,
and compiles remaining element-wise / linear / reduction ops into native `fastopt`
symbolic graph nodes with exact 1st & 2nd-order AD.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .expr import Expr


# ============================================================================
# 1. Minimal Pure-Python ONNX Protobuf Wire-Format Reader (Zero Dependencies)
# ============================================================================
# ONNX TensorProto DataType codes (from onnx.proto3)
_ONNX_DTYPE_MAP: Dict[int, np.dtype] = {
    1: np.dtype("<f4"),   # FLOAT
    2: np.dtype("<u1"),   # UINT8
    3: np.dtype("<i1"),   # INT8
    4: np.dtype("<u2"),   # UINT16
    5: np.dtype("<i2"),   # INT16
    6: np.dtype("<i4"),   # INT32
    7: np.dtype("<i8"),   # INT64
    10: np.dtype("<f2"),  # FLOAT16
    11: np.dtype("<f8"),  # DOUBLE
    12: np.dtype("<u4"),  # UINT32
    13: np.dtype("<u8"),  # UINT64
}


def _read_varint(buf: memoryview, pos: int) -> Tuple[int, int]:
    val = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7


def _iter_pb_fields(buf: memoryview):
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:
            v, pos = _read_varint(buf, pos)
            yield field_num, wire_type, v
        elif wire_type == 1:
            v = buf[pos : pos + 8]
            pos += 8
            yield field_num, wire_type, v
        elif wire_type == 2:
            length, pos = _read_varint(buf, pos)
            v = buf[pos : pos + length]
            pos += length
            yield field_num, wire_type, v
        elif wire_type == 5:
            v = buf[pos : pos + 4]
            pos += 4
            yield field_num, wire_type, v
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")


def _decode_packed_varints(buf: memoryview) -> List[int]:
    out: List[int] = []
    pos = 0
    n = len(buf)
    while pos < n:
        v, pos = _read_varint(buf, pos)
        # Handle signed 64-bit two's complement
        if v >= (1 << 63):
            v -= 1 << 64
        out.append(v)
    return out


def _parse_tensor_proto(buf: memoryview) -> Tuple[str, np.ndarray]:
    dims: List[int] = []
    data_type = 1
    name = ""
    raw_data: Optional[bytes] = None
    float_data: List[float] = []
    double_data: List[float] = []
    int32_data: List[int] = []
    int64_data: List[int] = []

    for fnum, wtype, val in _iter_pb_fields(buf):
        if fnum == 1:  # dims (repeated int64)
            if wtype == 2:
                dims.extend(_decode_packed_varints(val))
            else:
                dims.append(int(val))
        elif fnum == 2:  # data_type
            data_type = int(val)
        elif fnum == 4:  # float_data
            if wtype == 2:
                float_data.extend(np.frombuffer(val, dtype="<f4").tolist())
            else:
                float_data.append(struct.unpack("<f", bytes(val))[0])
        elif fnum == 5:  # int32_data
            if wtype == 2:
                int32_data.extend(_decode_packed_varints(val))
            else:
                int32_data.append(int(val))
        elif fnum == 7:  # int64_data
            if wtype == 2:
                int64_data.extend(_decode_packed_varints(val))
            else:
                v = int(val)
                if v >= (1 << 63):
                    v -= 1 << 64
                int64_data.append(v)
        elif fnum == 8:  # name
            name = bytes(val).decode("utf-8")
        elif fnum == 9:  # raw_data
            raw_data = bytes(val)
        elif fnum == 10:  # double_data
            if wtype == 2:
                double_data.extend(np.frombuffer(val, dtype="<f8").tolist())
            else:
                double_data.append(struct.unpack("<d", bytes(val))[0])

    if raw_data is not None:
        dt = _ONNX_DTYPE_MAP.get(data_type, np.dtype("<f4"))
        arr = np.frombuffer(raw_data, dtype=dt).copy()
    elif double_data:
        arr = np.array(double_data, dtype=np.float64)
    elif float_data:
        arr = np.array(float_data, dtype=np.float32)
    elif int64_data:
        arr = np.array(int64_data, dtype=np.int64)
    elif int32_data:
        arr = np.array(int32_data, dtype=np.int32)
    else:
        arr = np.empty(0, dtype=np.float64)

    if dims:
        arr = arr.reshape(tuple(dims))
    return name, arr


def _parse_attr_proto(buf: memoryview) -> Tuple[str, Any]:
    name = ""
    attr_type = 0
    f_val: Optional[float] = None
    i_val: Optional[int] = None
    s_val: Optional[str] = None
    t_val: Optional[np.ndarray] = None
    floats: List[float] = []
    ints: List[int] = []
    for fnum, wtype, val in _iter_pb_fields(buf):
        if fnum == 1:
            name = bytes(val).decode("utf-8")
        elif fnum == 2:
            f_val = struct.unpack("<f", bytes(val))[0]
        elif fnum == 3:
            v = int(val)
            if v >= (1 << 63):
                v -= 1 << 64
            i_val = v
        elif fnum == 4:
            s_val = bytes(val).decode("utf-8", errors="replace")
        elif fnum == 5:
            _, t_val = _parse_tensor_proto(val)
        elif fnum == 7:
            if wtype == 2:
                floats.extend(np.frombuffer(val, dtype="<f4").tolist())
            else:
                floats.append(struct.unpack("<f", bytes(val))[0])
        elif fnum == 8:
            if wtype == 2:
                ints.extend(_decode_packed_varints(val))
            else:
                v = int(val)
                if v >= (1 << 63):
                    v -= 1 << 64
                ints.append(v)
        elif fnum == 20:
            attr_type = int(val)

    # AttributeType: 1=FLOAT, 2=INT, 3=STRING, 4=TENSOR, 6=FLOATS, 7=INTS
    if attr_type == 1 or (attr_type == 0 and f_val is not None):
        return name, float(f_val if f_val is not None else 0.0)
    if attr_type == 2 or (attr_type == 0 and i_val is not None):
        return name, int(i_val if i_val is not None else 0)
    if attr_type == 3 or (attr_type == 0 and s_val is not None):
        return name, s_val
    if attr_type == 4 or (attr_type == 0 and t_val is not None):
        return name, t_val
    if attr_type == 6 or floats:
        return name, floats
    if attr_type == 7 or ints:
        return name, ints
    return name, None


@dataclass
class ONNXNode:
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)
    name: str = ""


@dataclass
class ONNXGraph:
    nodes: List[ONNXNode]
    initializers: Dict[str, np.ndarray]
    input_names: List[str]
    output_names: List[str]


def _parse_node_proto(buf: memoryview) -> ONNXNode:
    inputs: List[str] = []
    outputs: List[str] = []
    name = ""
    op_type = ""
    attrs: Dict[str, Any] = {}
    for fnum, _, val in _iter_pb_fields(buf):
        if fnum == 1:
            inputs.append(bytes(val).decode("utf-8"))
        elif fnum == 2:
            outputs.append(bytes(val).decode("utf-8"))
        elif fnum == 3:
            name = bytes(val).decode("utf-8")
        elif fnum == 4:
            op_type = bytes(val).decode("utf-8")
        elif fnum == 5:
            aname, aval = _parse_attr_proto(val)
            attrs[aname] = aval
    return ONNXNode(op_type=op_type, inputs=inputs, outputs=outputs, attrs=attrs, name=name)


def _parse_value_info_name(buf: memoryview) -> str:
    for fnum, _, val in _iter_pb_fields(buf):
        if fnum == 1:
            return bytes(val).decode("utf-8")
    return ""


def parse_onnx_file(path: Union[str, os.PathLike]) -> ONNXGraph:
    """Parse a binary `.onnx` file into an `ONNXGraph` with zero external dependencies."""
    raw = open(path, "rb").read()
    mv = memoryview(raw)
    graph_buf: Optional[memoryview] = None
    for fnum, _, val in _iter_pb_fields(mv):
        if fnum == 7:  # ModelProto.graph
            graph_buf = val
            break
    if graph_buf is None:
        raise ValueError(f"Invalid ONNX file (missing GraphProto): {path}")

    nodes: List[ONNXNode] = []
    initializers: Dict[str, np.ndarray] = {}
    inputs: List[str] = []
    outputs: List[str] = []
    for fnum, _, val in _iter_pb_fields(graph_buf):
        if fnum == 1:  # node
            nodes.append(_parse_node_proto(val))
        elif fnum == 5:  # initializer
            tname, tarr = _parse_tensor_proto(val)
            initializers[tname] = tarr
        elif fnum == 11:  # input
            nm = _parse_value_info_name(val)
            if nm:
                inputs.append(nm)
        elif fnum == 12:  # output
            nm = _parse_value_info_name(val)
            if nm:
                outputs.append(nm)

    # Filter out initializer names from graph inputs (older ONNX opsets list weights in graph.input)
    true_inputs = [nm for nm in inputs if nm not in initializers]
    return ONNXGraph(
        nodes=nodes,
        initializers=initializers,
        input_names=true_inputs,
        output_names=outputs,
    )


# ============================================================================
# 2. Minimal Binary `.onnx` Writer (for testing & lightweight model export)
# ============================================================================
def _enc_varint(v: int) -> bytes:
    if v < 0:
        v += 1 << 64
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _enc_tag(fnum: int, wtype: int) -> bytes:
    return _enc_varint((fnum << 3) | wtype)


def _enc_bytes(fnum: int, data: bytes) -> bytes:
    return _enc_tag(fnum, 2) + _enc_varint(len(data)) + data


def _enc_str(fnum: int, s: str) -> bytes:
    return _enc_bytes(fnum, s.encode("utf-8"))


def _enc_tensor_proto(name: str, arr: np.ndarray) -> bytes:
    a = np.ascontiguousarray(arr)
    if a.dtype == np.float64:
        dtype_code = 11
    elif a.dtype == np.int64:
        dtype_code = 7
    else:
        a = a.astype(np.float32)
        dtype_code = 1
    b = bytearray()
    for d in a.shape:
        b += _enc_tag(1, 0) + _enc_varint(int(d))
    b += _enc_tag(2, 0) + _enc_varint(dtype_code)
    b += _enc_str(8, name)
    b += _enc_bytes(9, a.tobytes())
    return bytes(b)


def _enc_attr_proto(name: str, val: Any) -> bytes:
    b = bytearray(_enc_str(1, name))
    if isinstance(val, int):
        b += _enc_tag(3, 0) + _enc_varint(val)
        b += _enc_tag(20, 0) + _enc_varint(2)
    elif isinstance(val, float):
        b += _enc_tag(2, 5) + struct.pack("<f", val)
        b += _enc_tag(20, 0) + _enc_varint(1)
    elif isinstance(val, (list, tuple)) and all(isinstance(x, int) for x in val):
        for x in val:
            b += _enc_tag(8, 0) + _enc_varint(x)
        b += _enc_tag(20, 0) + _enc_varint(7)
    return bytes(b)


def _enc_node_proto(op_type: str, inputs: Sequence[str], outputs: Sequence[str], attrs: Optional[Dict[str, Any]] = None) -> bytes:
    b = bytearray()
    for inp in inputs:
        b += _enc_str(1, inp)
    for out in outputs:
        b += _enc_str(2, out)
    b += _enc_str(4, op_type)
    if attrs:
        for k, v in attrs.items():
            b += _enc_bytes(5, _enc_attr_proto(k, v))
    return bytes(b)


def write_minimal_onnx(
    path: Union[str, os.PathLike],
    *,
    nodes: Sequence[Tuple[str, Sequence[str], Sequence[str], Dict[str, Any]]],
    initializers: Dict[str, np.ndarray],
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> None:
    """Write a standard binary `.onnx` ModelProto file with zero external dependencies."""
    gb = bytearray()
    for op_type, inps, outs, attrs in nodes:
        gb += _enc_bytes(1, _enc_node_proto(op_type, inps, outs, attrs))
    for nm, arr in initializers.items():
        gb += _enc_bytes(5, _enc_tensor_proto(nm, arr))
    for nm in input_names:
        gb += _enc_bytes(11, _enc_str(1, nm))
    for nm in output_names:
        gb += _enc_bytes(12, _enc_str(1, nm))
    mb = bytearray()
    mb += _enc_tag(1, 0) + _enc_varint(8)  # ir_version = 8
    mb += _enc_bytes(7, bytes(gb))
    with open(path, "wb") as f:
        f.write(bytes(mb))


# ============================================================================
# 3. ONNX -> fastopt Compiler with Automatic C++ MLP & Conv1D Fusion
# ============================================================================
_ONNX_ACT_MAP: Dict[str, str] = {
    "Tanh": "tanh",
    "Relu": "relu",
    "Silu": "silu",
    "Swish": "silu",
    "Gelu": "gelu",
    "Softplus": "softplus",
    "Elu": "elu",
    "Sin": "sin",
    "Sigmoid": "sigmoid",
    "LeakyRelu": "leaky_relu",
    "Mish": "mish",
    "Selu": "selu",
    "HardSwish": "hardswish",
}


class ONNXModel:
    """
    Compiled ONNX model runnable on both symbolic `fastopt.Expr` (exact 1st & 2nd AD)
    and numeric `numpy.ndarray` inputs.

    Automatically fuses consecutive `Gemm` / `MatMul+Add` + activation (+ residual)
    chains into a single C++ `fastopt.nn.MLP` (`OpType.MLP`) and 1D `Conv` layers
    into C++ `fastopt.nn.Conv1D` (`OpType.Conv1D`).
    """

    def __init__(
        self,
        graph: ONNXGraph,
        *,
        use_float32: bool = False,
        fuse_mlp: bool = True,
        name: str = "onnx_model",
    ) -> None:
        self._graph = graph
        self.use_float32 = bool(use_float32)
        self.fuse_mlp = bool(fuse_mlp)
        self.name = str(name)
        self.input_names = list(graph.input_names)
        self.output_names = list(graph.output_names)
        self._compiled_steps: List[Tuple[str, Any, List[str], str]] = []
        self._fused_mlp_count = 0
        self._fused_conv1d_count = 0
        self._fused_conv2d_count = 0
        self._folded_bn_count = 0
        self._compile_ir()

    @property
    def fused_mlp_count(self) -> int:
        return self._fused_mlp_count

    @property
    def fused_conv1d_count(self) -> int:
        return self._fused_conv1d_count

    @property
    def fused_conv2d_count(self) -> int:
        return self._fused_conv2d_count

    @property
    def folded_bn_count(self) -> int:
        return self._folded_bn_count

    def _compile_ir(self) -> None:
        from .nn import (
            AvgPool2D, BatchNorm, Conv1D, Conv2D, GlobalAvgPool2D,
            LayerConfig, MaxPool2D, MLP, RMSNorm,
        )

        inits = dict(self._graph.initializers)
        nodes = list(self._graph.nodes)

        # Resolve Constant nodes into initializers
        filtered: List[ONNXNode] = []
        for nd in nodes:
            if nd.op_type == "Constant" and "value" in nd.attrs and nd.attrs["value"] is not None:
                inits[nd.outputs[0]] = np.asarray(nd.attrs["value"])
            else:
                filtered.append(nd)
        nodes = filtered

        # Count consumer uses of each tensor for safe fusion
        use_count: Dict[str, int] = {}
        for nd in nodes:
            for inp in nd.inputs:
                use_count[inp] = use_count.get(inp, 0) + 1
        for out_nm in self.output_names:
            use_count[out_nm] = use_count.get(out_nm, 0) + 1

        def _extract_linear(idx: int) -> Optional[Tuple[str, np.ndarray, np.ndarray, str, int]]:
            """Try to match Gemm(x, W, b) or MatMul(x, W) + Add(b). Returns (x_name, W(dout, din), b(dout), out_name, next_idx)."""
            nd = nodes[idx]
            if nd.op_type == "Gemm" and len(nd.inputs) >= 2 and nd.inputs[1] in inits:
                x_nm = nd.inputs[0]
                W = np.asarray(inits[nd.inputs[1]], dtype=np.float64)
                transB = int(nd.attrs.get("transB", 0))
                alpha = float(nd.attrs.get("alpha", 1.0))
                beta = float(nd.attrs.get("beta", 1.0))
                if transB == 0:
                    W = W.T  # ONNX Gemm row-vector: x @ W -> column-vector: W^T @ x
                W = alpha * W
                if len(nd.inputs) >= 3 and nd.inputs[2] in inits:
                    b = beta * np.asarray(inits[nd.inputs[2]], dtype=np.float64).ravel()
                else:
                    b = np.zeros(W.shape[0], dtype=np.float64)
                return x_nm, W, b, nd.outputs[0], idx + 1

            if nd.op_type == "MatMul" and len(nd.inputs) == 2 and nd.inputs[1] in inits:
                x_nm = nd.inputs[0]
                W = np.asarray(inits[nd.inputs[1]], dtype=np.float64).T  # (din, dout)^T -> (dout, din)
                out_mm = nd.outputs[0]
                if (
                    idx + 1 < len(nodes)
                    and nodes[idx + 1].op_type == "Add"
                    and use_count.get(out_mm, 0) == 1
                ):
                    nd_add = nodes[idx + 1]
                    other = None
                    if nd_add.inputs[0] == out_mm and nd_add.inputs[1] in inits:
                        other = nd_add.inputs[1]
                    elif nd_add.inputs[1] == out_mm and nd_add.inputs[0] in inits:
                        other = nd_add.inputs[0]
                    if other is not None:
                        b = np.asarray(inits[other], dtype=np.float64).ravel()
                        if b.size == W.shape[0]:
                            return x_nm, W, b, nd_add.outputs[0], idx + 2
                b = np.zeros(W.shape[0], dtype=np.float64)
                return x_nm, W, b, out_mm, idx + 1
            return None

        def _match_activation(in_name: str, idx: int) -> Optional[Tuple[str, str, int, int]]:
            """Match element-wise activation or PyTorch ONNX SiLU pattern (Sigmoid(z) * z).
            Returns (act_name, out_name, next_idx, needed_uses_of_in_name)."""
            if idx >= len(nodes):
                return None
            nd = nodes[idx]
            # Check PyTorch nn.SiLU() pattern (Sigmoid(z) -> Mul(z, sig)) before plain Sigmoid
            if (
                nd.op_type == "Sigmoid"
                and nd.inputs[0] == in_name
                and idx + 1 < len(nodes)
                and nodes[idx + 1].op_type == "Mul"
            ):
                sig_out = nd.outputs[0]
                mul_in = nodes[idx + 1].inputs
                if set(mul_in) == {in_name, sig_out} and use_count.get(sig_out, 0) == 1:
                    return "silu", nodes[idx + 1].outputs[0], idx + 2, 2
            if nd.op_type in _ONNX_ACT_MAP and nd.inputs[0] == in_name:
                return _ONNX_ACT_MAP[nd.op_type], nd.outputs[0], idx + 1, 1
            return None

        i = 0
        n_nodes = len(nodes)
        while i < n_nodes:
            # 1. Try fusing consecutive Linear (+ Activation + Residual) layers into a single C++ MLP
            if self.fuse_mlp:
                lin0 = _extract_linear(i)
                if lin0 is not None:
                    mlp_in_name = lin0[0]
                    layers: List[LayerConfig] = []
                    cur_in = mlp_in_name
                    cur_idx = i
                    while cur_idx < n_nodes:
                        lin = _extract_linear(cur_idx)
                        if lin is None or lin[0] != cur_in:
                            break
                        _, W, b, lin_out, nxt = lin
                        act_name = "identity"
                        out_nm = lin_out
                        # Try optional activation if lin_out is consumed only by it (or SiLU pair)
                        act_m = _match_activation(lin_out, nxt)
                        if act_m is not None:
                            cand_act, cand_out, cand_nxt, needed_uses = act_m
                            if use_count.get(lin_out, 0) == needed_uses:
                                act_name, out_nm, nxt = cand_act, cand_out, cand_nxt
                        # Try optional residual skip: Add(out_nm, cur_in) when W is square
                        is_res = False
                        if (
                            W.shape[0] == W.shape[1]
                            and nxt < n_nodes
                            and nodes[nxt].op_type == "Add"
                            and set(nodes[nxt].inputs) == {out_nm, cur_in}
                            and use_count.get(out_nm, 0) == 1
                        ):
                            is_res = True
                            out_nm = nodes[nxt].outputs[0]
                            nxt += 1

                        layers.append(LayerConfig(W=W, b=b, activation=act_name, residual=is_res))
                        cur_in = out_nm
                        cur_idx = nxt
                        # If output fans out to non-residual consumers, stop extending this MLP
                        if use_count.get(cur_in, 0) > 1:
                            # Allow continuing only if next op is Linear+Residual using cur_in twice
                            if not (cur_idx < n_nodes and _extract_linear(cur_idx) is not None):
                                break

                    if layers:
                        mlp_obj = MLP(
                            layers,
                            use_float32=self.use_float32,
                            name=f"{self.name}_mlp_{self._fused_mlp_count}",
                        )
                        self._fused_mlp_count += 1
                        self._compiled_steps.append(("mlp", mlp_obj, [mlp_in_name], cur_in))
                        i = cur_idx
                        continue

            def _extract_bn(bn_nd: ONNXNode) -> Optional[BatchNorm]:
                if bn_nd.op_type != "BatchNormalization" or len(bn_nd.inputs) < 5:
                    return None
                if not all(nm in inits for nm in bn_nd.inputs[1:5]):
                    return None
                return BatchNorm(
                    running_mean=inits[bn_nd.inputs[3]],
                    running_var=inits[bn_nd.inputs[4]],
                    gamma=inits[bn_nd.inputs[1]],
                    beta=inits[bn_nd.inputs[2]],
                    eps=float(bn_nd.attrs.get("epsilon", 1e-5)),
                    name=bn_nd.name or "onnx_bn",
                )

            # 2. Try lowering 1D/2D Conv (+ optional BatchNormalization auto-fold + optional activation)
            nd = nodes[i]
            if nd.op_type == "Conv" and len(nd.inputs) >= 2 and nd.inputs[1] in inits:
                W_raw = np.asarray(inits[nd.inputs[1]], dtype=np.float64)
                if W_raw.ndim in (3, 4):
                    c_out = W_raw.shape[0]
                    b_raw = (
                        np.asarray(inits[nd.inputs[2]], dtype=np.float64).ravel()
                        if (len(nd.inputs) >= 3 and nd.inputs[2] in inits)
                        else np.zeros(c_out, dtype=np.float64)
                    )
                    out_nm = nd.outputs[0]
                    nxt = i + 1
                    # Auto-fold immediately following BatchNormalization into Conv weights/bias
                    if nxt < n_nodes and use_count.get(out_nm, 0) == 1:
                        bn_cand = _extract_bn(nodes[nxt])
                        if bn_cand is not None and nodes[nxt].inputs[0] == out_nm:
                            W_raw, b_raw = bn_cand.fold_into_conv_weights(W_raw, b_raw)
                            out_nm = nodes[nxt].outputs[0]
                            nxt += 1
                            self._folded_bn_count += 1
                    # Fuse optional activation
                    act_name = "identity"
                    act_m = _match_activation(out_nm, nxt)
                    if act_m is not None:
                        cand_act, cand_out, cand_nxt, needed_uses = act_m
                        if use_count.get(out_nm, 0) == needed_uses:
                            act_name, out_nm, nxt = cand_act, cand_out, cand_nxt
                    if W_raw.ndim == 3:
                        strides = nd.attrs.get("strides", [1])
                        pads = nd.attrs.get("pads", [0, 0])
                        dilations = nd.attrs.get("dilations", [1])
                        conv_obj = Conv1D(
                            W_raw, b_raw,
                            stride=int(strides[0]) if strides else 1,
                            padding=int(pads[0]) if pads else 0,
                            dilation=int(dilations[0]) if dilations else 1,
                            activation=act_name, use_float32=self.use_float32,
                            name=f"{self.name}_conv1d_{self._fused_conv1d_count}",
                        )
                        self._fused_conv1d_count += 1
                        self._compiled_steps.append(("conv1d", conv_obj, [nd.inputs[0]], out_nm))
                        i = nxt
                        continue
                    else:
                        strides = nd.attrs.get("strides", [1, 1])
                        pads = nd.attrs.get("pads", [0, 0, 0, 0])
                        dilations = nd.attrs.get("dilations", [1, 1])
                        sh = int(strides[0]) if len(strides) >= 1 else 1
                        sw = int(strides[1]) if len(strides) >= 2 else sh
                        ph = int(pads[0]) if len(pads) >= 1 else 0
                        pw = int(pads[1]) if len(pads) >= 2 else ph
                        dh = int(dilations[0]) if len(dilations) >= 1 else 1
                        dw = int(dilations[1]) if len(dilations) >= 2 else dh
                        conv2d_obj = Conv2D(
                            W_raw, b_raw,
                            stride=(sh, sw), padding=(ph, pw), dilation=(dh, dw),
                            activation=act_name, use_float32=self.use_float32,
                            name=f"{self.name}_conv2d_{self._fused_conv2d_count}",
                        )
                        self._fused_conv2d_count += 1
                        self._compiled_steps.append(("conv2d", conv2d_obj, [nd.inputs[0]], out_nm))
                        i = nxt
                        continue

            # 3. Standalone BatchNormalization -> BatchNorm
            if nd.op_type == "BatchNormalization":
                bn_obj = _extract_bn(nd)
                if bn_obj is not None:
                    self._compiled_steps.append(("bn", bn_obj, [nd.inputs[0]], nd.outputs[0]))
                    i += 1
                    continue

            # 4. 2D Pooling (AveragePool / MaxPool / GlobalAveragePool)
            if nd.op_type in ("AveragePool", "MaxPool"):
                ks = nd.attrs.get("kernel_shape", [2, 2])
                st = nd.attrs.get("strides", ks)
                pd = nd.attrs.get("pads", [0, 0, 0, 0])
                kh, kw = int(ks[0]), int(ks[1] if len(ks) > 1 else ks[0])
                sh, sw = int(st[0]), int(st[1] if len(st) > 1 else st[0])
                ph, pw = int(pd[0]), int(pd[1] if len(pd) > 1 else pd[0])
                pool_cls = AvgPool2D if nd.op_type == "AveragePool" else MaxPool2D
                pool_obj = pool_cls(
                    kernel_size=(kh, kw), stride=(sh, sw), padding=(ph, pw),
                    name=nd.name or nd.op_type.lower(),
                )
                self._compiled_steps.append(("pool2d", pool_obj, [nd.inputs[0]], nd.outputs[0]))
                i += 1
                continue

            if nd.op_type == "GlobalAveragePool":
                self._compiled_steps.append(("global_avgpool2d", GlobalAvgPool2D(name=nd.name or "gap2d"), [nd.inputs[0]], nd.outputs[0]))
                i += 1
                continue

            # 5. RMSNormalization -> C++ RMSNorm
            if nd.op_type == "RMSNormalization":
                gamma = inits[nd.inputs[1]] if (len(nd.inputs) >= 2 and nd.inputs[1] in inits) else None
                eps = float(nd.attrs.get("epsilon", 1e-6))
                self._compiled_steps.append(("rmsnorm", RMSNorm(gamma=gamma, eps=eps, name=nd.name or "rmsnorm"), [nd.inputs[0]], nd.outputs[0]))
                i += 1
                continue

            # 6. Fallback general ONNX op
            self._compiled_steps.append(("op", nd, list(nd.inputs), nd.outputs[0]))
            i += 1

        self._inits = inits

    def __call__(
        self,
        *inputs: Union[Expr, np.ndarray],
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> Union[Expr, np.ndarray, Tuple[Any, ...]]:
        """Evaluate the ONNX model on symbolic `fastopt.Expr` or numeric `np.ndarray` inputs.

        For 2D CNN models with symbolic `(C, H*W)` inputs, pass `spatial_hw=(H, W)` so
        spatial dimensions are automatically tracked across `Conv2D` and `Pool2D` layers.
        """
        from . import expr as fo_expr

        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"ONNXModel '{self.name}' expects {len(self.input_names)} input(s) "
                f"({self.input_names}), got {len(inputs)}."
            )

        env: Dict[str, Any] = {}
        hw_env: Dict[str, Tuple[int, int]] = {}
        for nm, val in zip(self.input_names, inputs):
            env[nm] = val
            if spatial_hw is not None:
                hw_env[nm] = (int(spatial_hw[0]), int(spatial_hw[1]))
            elif isinstance(val, np.ndarray) and val.ndim == 3:
                hw_env[nm] = (int(val.shape[1]), int(val.shape[2]))

        def _get(nm: str) -> Any:
            if nm in env:
                return env[nm]
            if nm in self._inits:
                return np.asarray(self._inits[nm], dtype=np.float64)
            raise KeyError(f"ONNX tensor '{nm}' not found in environment or initializers.")

        for kind, obj, in_nms, out_nm in self._compiled_steps:
            if kind in ("mlp", "conv1d", "bn", "rmsnorm", "global_avgpool2d"):
                env[out_nm] = obj(_get(in_nms[0]))
                if kind in ("bn", "rmsnorm") and in_nms[0] in hw_env:
                    hw_env[out_nm] = hw_env[in_nms[0]]
                elif kind == "global_avgpool2d":
                    hw_env[out_nm] = (1, 1)
            elif kind in ("conv2d", "pool2d"):
                x_val = _get(in_nms[0])
                if in_nms[0] in hw_env:
                    hi, wi = hw_env[in_nms[0]]
                    env[out_nm] = obj(x_val, h_in=hi, w_in=wi)
                    hw_env[out_nm] = obj.out_hw(hi, wi)
                else:
                    env[out_nm] = obj(x_val)
            else:
                nd: ONNXNode = obj
                op = nd.op_type
                if op == "Add":
                    env[out_nm] = _get(in_nms[0]) + _get(in_nms[1])
                elif op == "Sub":
                    env[out_nm] = _get(in_nms[0]) - _get(in_nms[1])
                elif op == "Mul":
                    env[out_nm] = _get(in_nms[0]) * _get(in_nms[1])
                elif op == "Div":
                    env[out_nm] = _get(in_nms[0]) / _get(in_nms[1])
                elif op == "Tanh":
                    x = _get(in_nms[0])
                    env[out_nm] = fo_expr.tanh(x) if isinstance(x, Expr) else np.tanh(x)
                elif op == "Sin":
                    x = _get(in_nms[0])
                    env[out_nm] = fo_expr.sin(x) if isinstance(x, Expr) else np.sin(x)
                elif op == "Cos":
                    x = _get(in_nms[0])
                    env[out_nm] = fo_expr.cos(x) if isinstance(x, Expr) else np.cos(x)
                elif op == "Exp":
                    x = _get(in_nms[0])
                    env[out_nm] = fo_expr.exp(x) if isinstance(x, Expr) else np.exp(x)
                elif op == "Log":
                    x = _get(in_nms[0])
                    env[out_nm] = fo_expr.log(x) if isinstance(x, Expr) else np.log(x)
                elif op == "Sqrt":
                    x = _get(in_nms[0])
                    env[out_nm] = fo_expr.sqrt(x) if isinstance(x, Expr) else np.sqrt(x)
                elif op == "Sigmoid":
                    x = _get(in_nms[0])
                    env[out_nm] = (1.0 / (1.0 + fo_expr.exp(-x))) if isinstance(x, Expr) else (1.0 / (1.0 + np.exp(-x)))
                elif op == "Relu":
                    x = _get(in_nms[0])
                    env[out_nm] = fo_expr.where(x > 0.0, x, 0.0) if isinstance(x, Expr) else np.maximum(x, 0.0)
                elif op == "LeakyRelu":
                    x = _get(in_nms[0])
                    alpha = float(nd.attrs.get("alpha", 0.01))
                    env[out_nm] = fo_expr.where(x > 0.0, x, alpha * x) if isinstance(x, Expr) else np.where(x > 0.0, x, alpha * x)
                elif op == "Softplus":
                    x = _get(in_nms[0])
                    env[out_nm] = fo_expr.log(1.0 + fo_expr.exp(x)) if isinstance(x, Expr) else np.logaddexp(0.0, x)
                elif op == "Mish":
                    x = _get(in_nms[0])
                    if isinstance(x, Expr):
                        env[out_nm] = x * fo_expr.tanh(fo_expr.log(1.0 + fo_expr.exp(x)))
                    else:
                        env[out_nm] = x * np.tanh(np.logaddexp(0.0, x))
                elif op == "Selu":
                    x = _get(in_nms[0])
                    al = float(nd.attrs.get("alpha", 1.6732632423543772))
                    ga = float(nd.attrs.get("gamma", 1.0507009873554805))
                    if isinstance(x, Expr):
                        env[out_nm] = ga * fo_expr.where(x > 0.0, x, al * (fo_expr.exp(x) - 1.0))
                    else:
                        env[out_nm] = ga * np.where(x > 0.0, x, al * (np.exp(x) - 1.0))
                elif op == "HardSwish":
                    x = _get(in_nms[0])
                    if isinstance(x, Expr):
                        env[out_nm] = fo_expr.where(x <= -3.0, 0.0, fo_expr.where(x >= 3.0, x, x * (x + 3.0) / 6.0))
                    else:
                        env[out_nm] = np.where(x <= -3.0, 0.0, np.where(x >= 3.0, x, x * (x + 3.0) / 6.0))
                elif op == "Transpose":
                    x = _get(in_nms[0])
                    env[out_nm] = x.T
                elif op in ("Identity", "Dropout"):
                    env[out_nm] = _get(in_nms[0])
                elif op == "ReduceSum":
                    x = _get(in_nms[0])
                    env[out_nm] = fo_expr.sum(x) if isinstance(x, Expr) else np.sum(x)
                elif op == "ReduceMean":
                    x = _get(in_nms[0])
                    env[out_nm] = fo_expr.mean(x) if isinstance(x, Expr) else np.mean(x)
                elif op == "Flatten":
                    x = _get(in_nms[0])
                    if isinstance(x, Expr):
                        # Symbolic Conv2D/Pool2D outputs are (C, H*W) in column-major storage;
                        # transposing to (H*W, C) before column-major flatten matches ONNX row-major (C, H, W) Flatten
                        env[out_nm] = x.T.reshape(x.shape[0] * x.shape[1], 1)
                    else:
                        env[out_nm] = np.asarray(x).reshape((-1, 1), order="C")
                else:
                    raise NotImplementedError(f"ONNX operator '{op}' is not yet supported by fastopt.nn.from_onnx.")

        outs = [env[nm] for nm in self.output_names]
        return outs[0] if len(outs) == 1 else tuple(outs)


def from_onnx(
    path: Union[str, os.PathLike],
    *,
    use_float32: bool = False,
    fuse_mlp: bool = True,
    name: Optional[str] = None,
) -> ONNXModel:
    """
    Load an `.onnx` neural network model and compile it into a native `fastopt`
    executable surrogate (with automatic C++ `MLPNode` & `Conv1DNode` fusion).
    """
    g = parse_onnx_file(path)
    model_name = name or os.path.splitext(os.path.basename(os.fspath(path)))[0]
    return ONNXModel(g, use_float32=use_float32, fuse_mlp=fuse_mlp, name=model_name)


__all__ = [
    "ONNXNode",
    "ONNXGraph",
    "ONNXModel",
    "parse_onnx_file",
    "write_minimal_onnx",
    "from_onnx",
]
