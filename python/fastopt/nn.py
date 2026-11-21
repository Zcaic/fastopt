"""
High-performance CPU neural network surrogate module for fastopt.

All networks compile down to a single fused C++17 `MLPNode` (`OpType.MLP`) with:
- Zero Python interpreter overhead and zero heap allocations per AD evaluation
- Optional `float32` mixed-precision GEMM/activation kernels (2x AVX2/FMA SIMD throughput)
- Exact 0th, 1st (VJP + JVP), and 2nd-order (Forward-over-Reverse HVP) derivatives
- Support for feedforward MLPs, Residual (ResNet) skip blocks, LayerNorm, and
  fused input/output affine normalization
- Zero-copy weight sharing across multiple symbolic call sites in a computation graph
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._fastopt_impl import (
    CompiledAttention,
    CompiledConv1D,
    CompiledConv2D,
    CompiledMLP,
    MLPLayerSpec,
    NNActivation,
    Pool2DMode,
    op_attention,
    op_conv1d,
    op_conv2d,
    op_mlp,
    op_pool2d,
    op_rms_norm,
)
from .onnx_import import ONNXModel, from_onnx
from .expr import Expr


_ACTIVATION_MAP: Dict[str, NNActivation] = {
    "identity": NNActivation.Identity,
    "linear": NNActivation.Identity,
    "none": NNActivation.Identity,
    "tanh": NNActivation.Tanh,
    "silu": NNActivation.SiLU,
    "swish": NNActivation.SiLU,
    "gelu": NNActivation.GELU,
    "relu": NNActivation.ReLU,
    "softplus": NNActivation.Softplus,
    "elu": NNActivation.ELU,
    "sin": NNActivation.Sin,
    "sigmoid": NNActivation.Sigmoid,
    "leaky_relu": NNActivation.LeakyReLU,
    "leakyrelu": NNActivation.LeakyReLU,
    "mish": NNActivation.Mish,
    "selu": NNActivation.SELU,
    "hardswish": NNActivation.HardSwish,
    "hard_swish": NNActivation.HardSwish,
}

_ACTIVATION_INV_MAP: Dict[NNActivation, str] = {
    NNActivation.Identity: "identity",
    NNActivation.Tanh: "tanh",
    NNActivation.SiLU: "silu",
    NNActivation.GELU: "gelu",
    NNActivation.ReLU: "relu",
    NNActivation.Softplus: "softplus",
    NNActivation.ELU: "elu",
    NNActivation.Sin: "sin",
    NNActivation.Sigmoid: "sigmoid",
    NNActivation.LeakyReLU: "leaky_relu",
    NNActivation.Mish: "mish",
    NNActivation.SELU: "selu",
    NNActivation.HardSwish: "hardswish",
}


def resolve_activation(act: Union[str, NNActivation, None]) -> NNActivation:
    """Convert an activation name or enum into `NNActivation`."""
    if act is None:
        return NNActivation.Identity
    if isinstance(act, NNActivation):
        return act
    key = str(act).strip().lower()
    if key not in _ACTIVATION_MAP:
        raise ValueError(
            f"Unsupported activation '{act}'. Supported: {sorted(set(_ACTIVATION_MAP.keys()))}"
        )
    return _ACTIVATION_MAP[key]


@dataclass
class LayerConfig:
    """Python-side specification of a single dense/residual/LayerNorm MLP layer."""

    W: np.ndarray
    b: np.ndarray
    activation: Union[str, NNActivation] = "tanh"
    residual: bool = False
    layer_norm: bool = False
    ln_gamma: Optional[np.ndarray] = None
    ln_beta: Optional[np.ndarray] = None
    ln_eps: float = 1e-5

    def to_cpp_spec(self) -> MLPLayerSpec:
        spec = MLPLayerSpec()
        spec.W = np.asfortranarray(self.W, dtype=np.float64)
        spec.b = np.ascontiguousarray(self.b, dtype=np.float64).ravel()
        spec.activation = resolve_activation(self.activation)
        spec.residual = bool(self.residual)
        spec.layer_norm = bool(self.layer_norm)
        if self.ln_gamma is not None:
            spec.ln_gamma = np.ascontiguousarray(self.ln_gamma, dtype=np.float64).ravel()
        if self.ln_beta is not None:
            spec.ln_beta = np.ascontiguousarray(self.ln_beta, dtype=np.float64).ravel()
        spec.ln_eps = float(self.ln_eps)
        return spec


class MLP:
    """
    Fused C++17 Multi-Layer Perceptron (with optional ResNet skip connections and LayerNorm).

    Supports both symbolic `fastopt.Expr` inputs (emitting a single fused `MLPNode` with
    exact 0th/1st/2nd-order AD) and direct numeric `numpy.ndarray` evaluation (GIL-free,
    multi-column GEMM with automatic OpenMP column blocking for large batches).
    """

    def __init__(
        self,
        layers: Sequence[Union[LayerConfig, Dict[str, Any]]],
        *,
        use_float32: bool = False,
        in_mean: Optional[np.ndarray] = None,
        in_std: Optional[np.ndarray] = None,
        out_mean: Optional[np.ndarray] = None,
        out_std: Optional[np.ndarray] = None,
        name: str = "mlp",
    ) -> None:
        if len(layers) == 0:
            raise ValueError("MLP requires at least one layer.")

        self._layer_configs: List[LayerConfig] = []
        cpp_specs: List[MLPLayerSpec] = []
        for ly in layers:
            if isinstance(ly, LayerConfig):
                cfg = ly
            elif isinstance(ly, dict):
                cfg = LayerConfig(**ly)
            else:
                raise TypeError(f"Expected LayerConfig or dict, got {type(ly).__name__}")
            self._layer_configs.append(cfg)
            cpp_specs.append(cfg.to_cpp_spec())

        self.name = str(name)
        self._use_float32 = bool(use_float32)

        self._in_mean = (
            np.ascontiguousarray(in_mean, dtype=np.float64).ravel()
            if in_mean is not None
            else None
        )
        self._in_std = (
            np.ascontiguousarray(in_std, dtype=np.float64).ravel()
            if in_std is not None
            else None
        )
        self._out_mean = (
            np.ascontiguousarray(out_mean, dtype=np.float64).ravel()
            if out_mean is not None
            else None
        )
        self._out_std = (
            np.ascontiguousarray(out_std, dtype=np.float64).ravel()
            if out_std is not None
            else None
        )

        if (self._in_mean is None) != (self._in_std is None):
            raise ValueError("Both in_mean and in_std must be provided together.")
        if (self._out_mean is None) != (self._out_std is None):
            raise ValueError("Both out_mean and out_std must be provided together.")

        in_mean_vec = self._in_mean if self._in_mean is not None else np.empty(0, dtype=np.float64)
        in_inv_std_vec = (
            (1.0 / self._in_std) if self._in_std is not None else np.empty(0, dtype=np.float64)
        )
        out_mean_vec = (
            self._out_mean if self._out_mean is not None else np.empty(0, dtype=np.float64)
        )
        out_std_vec = (
            self._out_std if self._out_std is not None else np.empty(0, dtype=np.float64)
        )

        self._compiled = CompiledMLP(
            cpp_specs,
            use_float32=self._use_float32,
            in_mean=in_mean_vec,
            in_inv_std=in_inv_std_vec,
            out_mean=out_mean_vec,
            out_std=out_std_vec,
            name=self.name,
        )

    @property
    def d_in(self) -> int:
        return int(self._compiled.d_in())

    @property
    def d_out(self) -> int:
        return int(self._compiled.d_out())

    @property
    def num_layers(self) -> int:
        return int(self._compiled.num_layers())

    @property
    def use_float32(self) -> bool:
        return bool(self._compiled.use_float32())

    @property
    def compiled(self) -> CompiledMLP:
        return self._compiled

    @classmethod
    def from_weights(
        cls,
        weights: Sequence[np.ndarray],
        biases: Sequence[np.ndarray],
        activations: Union[str, NNActivation, Sequence[Union[str, NNActivation]]] = "tanh",
        *,
        output_activation: Union[str, NNActivation] = "identity",
        residual: Union[bool, Sequence[bool]] = False,
        layer_norm: Union[bool, Sequence[bool]] = False,
        ln_gammas: Optional[Sequence[Optional[np.ndarray]]] = None,
        ln_betas: Optional[Sequence[Optional[np.ndarray]]] = None,
        ln_eps: float = 1e-5,
        use_float32: bool = False,
        in_mean: Optional[np.ndarray] = None,
        in_std: Optional[np.ndarray] = None,
        out_mean: Optional[np.ndarray] = None,
        out_std: Optional[np.ndarray] = None,
        name: str = "mlp",
    ) -> "MLP":
        """Construct an `MLP` from lists of weight matrices `W_l (d_out, d_in)` and biases `b_l`."""
        L = len(weights)
        if L == 0 or len(biases) != L:
            raise ValueError("weights and biases must be non-empty sequences of the same length.")

        if isinstance(activations, (str, NNActivation)):
            act_list: List[Union[str, NNActivation]] = [activations] * (L - 1) + [output_activation]
        else:
            act_list = list(activations)
            if len(act_list) != L:
                raise ValueError(f"Expected {L} activations, got {len(act_list)}.")

        res_list = [bool(residual)] * L if isinstance(residual, bool) else [bool(r) for r in residual]
        ln_list = [bool(layer_norm)] * L if isinstance(layer_norm, bool) else [bool(x) for x in layer_norm]

        layers: List[LayerConfig] = []
        for l in range(L):
            layers.append(
                LayerConfig(
                    W=np.asarray(weights[l], dtype=np.float64),
                    b=np.asarray(biases[l], dtype=np.float64).ravel(),
                    activation=act_list[l],
                    residual=res_list[l],
                    layer_norm=ln_list[l],
                    ln_gamma=(None if ln_gammas is None else ln_gammas[l]),
                    ln_beta=(None if ln_betas is None else ln_betas[l]),
                    ln_eps=ln_eps,
                )
            )
        return cls(
            layers,
            use_float32=use_float32,
            in_mean=in_mean,
            in_std=in_std,
            out_mean=out_mean,
            out_std=out_std,
            name=name,
        )

    @classmethod
    def random(
        cls,
        dims: Sequence[int],
        activation: Union[str, NNActivation] = "silu",
        *,
        output_activation: Union[str, NNActivation] = "identity",
        residual: bool = False,
        layer_norm: bool = False,
        use_float32: bool = False,
        seed: int = 42,
        name: str = "mlp_rand",
    ) -> "MLP":
        """Create a randomly initialized MLP (Xavier/Glorot scaling) for testing or benchmarking."""
        if len(dims) < 2:
            raise ValueError("dims must contain at least [d_in, d_out].")
        rng = np.random.default_rng(seed)
        layers: List[LayerConfig] = []
        L = len(dims) - 1
        for l in range(L):
            d_in, d_out = int(dims[l]), int(dims[l + 1])
            scale = np.sqrt(2.0 / (d_in + d_out))
            W = rng.normal(0.0, scale, size=(d_out, d_in))
            b = rng.normal(0.0, 0.05, size=(d_out,))
            is_last = l == L - 1
            use_res = bool(residual and (not is_last) and (d_in == d_out))
            use_ln = bool(layer_norm and (not is_last))
            ln_g = rng.uniform(0.8, 1.2, size=(d_out,)) if use_ln else None
            ln_b = rng.normal(0.0, 0.05, size=(d_out,)) if use_ln else None
            layers.append(
                LayerConfig(
                    W=W,
                    b=b,
                    activation=(output_activation if is_last else activation),
                    residual=use_res,
                    layer_norm=use_ln,
                    ln_gamma=ln_g,
                    ln_beta=ln_b,
                )
            )
        return cls(layers, use_float32=use_float32, name=name)

    def eval(self, X: np.ndarray, num_threads: int = 0) -> np.ndarray:
        """
        Evaluate the compiled C++ MLP directly on numeric inputs.

        Args:
            X: 1D array of shape `(d_in,)` or 2D column-batch matrix of shape `(d_in, B)`.
            num_threads: OpenMP thread count for large batches (`0` = automatic).

        Returns:
            1D array of shape `(d_out,)` if input was 1D, or 2D `(d_out, B)` otherwise.
        """
        arr = np.asarray(X, dtype=np.float64)
        is_1d = arr.ndim == 1
        if is_1d:
            if arr.shape[0] != self.d_in:
                raise ValueError(f"Expected input length {self.d_in}, got {arr.shape[0]}.")
            X2 = np.asfortranarray(arr.reshape(self.d_in, 1))
            Y2 = self._compiled.eval(X2, int(num_threads))
            return np.asarray(Y2[:, 0], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != self.d_in:
            raise ValueError(f"Expected 2D input of shape ({self.d_in}, B), got {arr.shape}.")
        X2 = np.asfortranarray(arr)
        return np.asarray(self._compiled.eval(X2, int(num_threads)), dtype=np.float64)

    def __call__(self, x: Union[Expr, np.ndarray, Sequence[float]]) -> Union[Expr, np.ndarray]:
        """
        Apply the MLP to either a symbolic `fastopt.Expr` of shape `(d_in, B)`
        or a numeric `numpy.ndarray` of shape `(d_in,)` / `(d_in, B)`.
        """
        if isinstance(x, Expr):
            if x.shape[0] != self.d_in:
                raise ValueError(
                    f"MLP '{self.name}' expects symbolic input rows == {self.d_in}, "
                    f"got shape {x.shape}."
                )
            out_id = op_mlp(x.graph, x.id, self._compiled, self.name)
            return Expr(x.graph, out_id)
        return self.eval(np.asarray(x, dtype=np.float64))

    def save_npz(self, path: Union[str, os.PathLike]) -> None:
        """Serialize weights, activations, LayerNorm params, and normalization stats to `.npz`."""
        payload: Dict[str, Any] = {
            "num_layers": np.array(len(self._layer_configs), dtype=np.int64),
            "use_float32": np.array(int(self._use_float32), dtype=np.int64),
            "name": np.array(self.name),
        }
        if self._in_mean is not None and self._in_std is not None:
            payload["in_mean"] = self._in_mean
            payload["in_std"] = self._in_std
        if self._out_mean is not None and self._out_std is not None:
            payload["out_mean"] = self._out_mean
            payload["out_std"] = self._out_std

        for l, cfg in enumerate(self._layer_configs):
            payload[f"W_{l}"] = np.asarray(cfg.W, dtype=np.float64)
            payload[f"b_{l}"] = np.asarray(cfg.b, dtype=np.float64)
            act_enum = resolve_activation(cfg.activation)
            payload[f"act_{l}"] = np.array(_ACTIVATION_INV_MAP[act_enum])
            payload[f"res_{l}"] = np.array(int(cfg.residual), dtype=np.int64)
            payload[f"ln_{l}"] = np.array(int(cfg.layer_norm), dtype=np.int64)
            payload[f"ln_eps_{l}"] = np.array(float(cfg.ln_eps), dtype=np.float64)
            if cfg.ln_gamma is not None:
                payload[f"ln_gamma_{l}"] = np.asarray(cfg.ln_gamma, dtype=np.float64)
            if cfg.ln_beta is not None:
                payload[f"ln_beta_{l}"] = np.asarray(cfg.ln_beta, dtype=np.float64)
        np.savez(path, **payload)

    @classmethod
    def from_npz(
        cls,
        path: Union[str, os.PathLike],
        *,
        use_float32: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> "MLP":
        """Load an `MLP` previously saved with `save_npz()`."""
        data = np.load(path, allow_pickle=False)
        L = int(data["num_layers"])
        fp32 = bool(int(data["use_float32"])) if use_float32 is None else bool(use_float32)
        net_name = str(data["name"]) if name is None else str(name)
        in_mean = data["in_mean"] if "in_mean" in data else None
        in_std = data["in_std"] if "in_std" in data else None
        out_mean = data["out_mean"] if "out_mean" in data else None
        out_std = data["out_std"] if "out_std" in data else None

        layers: List[LayerConfig] = []
        for l in range(L):
            layers.append(
                LayerConfig(
                    W=data[f"W_{l}"],
                    b=data[f"b_{l}"],
                    activation=str(data[f"act_{l}"]),
                    residual=bool(int(data[f"res_{l}"])),
                    layer_norm=bool(int(data[f"ln_{l}"])),
                    ln_gamma=(data[f"ln_gamma_{l}"] if f"ln_gamma_{l}" in data else None),
                    ln_beta=(data[f"ln_beta_{l}"] if f"ln_beta_{l}" in data else None),
                    ln_eps=float(data[f"ln_eps_{l}"]) if f"ln_eps_{l}" in data else 1e-5,
                )
            )
        return cls(
            layers,
            use_float32=fp32,
            in_mean=in_mean,
            in_std=in_std,
            out_mean=out_mean,
            out_std=out_std,
            name=net_name,
        )

    @classmethod
    def from_pytorch(
        cls,
        module: Any,
        *,
        use_float32: bool = False,
        in_mean: Optional[np.ndarray] = None,
        in_std: Optional[np.ndarray] = None,
        out_mean: Optional[np.ndarray] = None,
        out_std: Optional[np.ndarray] = None,
        name: str = "torch_mlp",
    ) -> "MLP":
        """
        Convert a sequential PyTorch MLP (`torch.nn.Sequential` or `nn.Module`
        composed of `Linear`, optional `LayerNorm`, and element-wise activations)
        into a fused C++ `fastopt.nn.MLP`.
        """
        try:
            import torch
            import torch.nn as tnn
        except ImportError as exc:
            raise ImportError("PyTorch (`torch`) is required for MLP.from_pytorch().") from exc

        torch_act_map = {
            tnn.Identity: "identity",
            tnn.Tanh: "tanh",
            tnn.SiLU: "silu",
            tnn.GELU: "gelu",
            tnn.ReLU: "relu",
            tnn.Softplus: "softplus",
            tnn.ELU: "elu",
            tnn.Sigmoid: "sigmoid",
            tnn.LeakyReLU: "leaky_relu",
            tnn.SELU: "selu",
        }
        if hasattr(tnn, "Mish"):
            torch_act_map[tnn.Mish] = "mish"
        if hasattr(tnn, "Hardswish"):
            torch_act_map[tnn.Hardswish] = "hardswish" 

        children = list(module.children())
        if len(children) == 1 and isinstance(children[0], tnn.Sequential):
            children = list(children[0].children())

        layers: List[LayerConfig] = []
        for child in children:
            if isinstance(child, tnn.Linear):
                W = child.weight.detach().cpu().numpy().astype(np.float64)
                b = (
                    child.bias.detach().cpu().numpy().astype(np.float64)
                    if child.bias is not None
                    else np.zeros(W.shape[0], dtype=np.float64)
                )
                layers.append(LayerConfig(W=W, b=b, activation="identity"))
            elif isinstance(child, tnn.LayerNorm):
                if not layers:
                    raise ValueError("LayerNorm must follow a Linear layer in MLP.from_pytorch().")
                layers[-1].layer_norm = True
                layers[-1].ln_eps = float(child.eps)
                if child.weight is not None:
                    layers[-1].ln_gamma = child.weight.detach().cpu().numpy().astype(np.float64)
                if child.bias is not None:
                    layers[-1].ln_beta = child.bias.detach().cpu().numpy().astype(np.float64)
            else:
                matched = None
                for t_cls, act_name in torch_act_map.items():
                    if isinstance(child, t_cls):
                        matched = act_name
                        break
                if matched is None:
                    raise TypeError(
                        f"Unsupported PyTorch layer type '{type(child).__name__}' in MLP.from_pytorch()."
                    )
                if not layers:
                    raise ValueError("Activation layer appeared before any Linear layer.")
                layers[-1].activation = matched

        if not layers:
            raise ValueError("No torch.nn.Linear layers found in module.")
        return cls(
            layers,
            use_float32=use_float32,
            in_mean=in_mean,
            in_std=in_std,
            out_mean=out_mean,
            out_std=out_std,
            name=name,
        )


def ResNetMLP(
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    *,
    activation: Union[str, NNActivation] = "silu",
    output_activation: Union[str, NNActivation] = "identity",
    layer_norm: bool = True,
    ln_gammas: Optional[Sequence[Optional[np.ndarray]]] = None,
    ln_betas: Optional[Sequence[Optional[np.ndarray]]] = None,
    use_float32: bool = False,
    in_mean: Optional[np.ndarray] = None,
    in_std: Optional[np.ndarray] = None,
    out_mean: Optional[np.ndarray] = None,
    out_std: Optional[np.ndarray] = None,
    name: str = "resnet_mlp",
) -> MLP:
    """
    Convenience constructor for a Residual MLP: automatically enables residual skip
    connections (`h_{l+1} = act(LN(W_l h_l + b_l)) + h_l`) on all square hidden layers.
    """
    L = len(weights)
    res_flags = [
        (l < L - 1) and (weights[l].shape[0] == weights[l].shape[1]) for l in range(L)
    ]
    ln_flags = [bool(layer_norm and (l < L - 1)) for l in range(L)]
    return MLP.from_weights(
        weights,
        biases,
        activations=activation,
        output_activation=output_activation,
        residual=res_flags,
        layer_norm=ln_flags,
        ln_gammas=ln_gammas,
        ln_betas=ln_betas,
        use_float32=use_float32,
        in_mean=in_mean,
        in_std=in_std,
        out_mean=out_mean,
        out_std=out_std,
        name=name,
    )


class NeuralFoilSurrogate:
    """
    Exact C++17 compiler for NeuralFoil `.npz` airfoil aerodynamic surrogate weights.

    Implements 100% mathematical parity with `peterdsharpe/NeuralFoil` (`neuralfoil/main.py`):
    - Auto-discovers `nn_weights_and_biases/nn-<model_size>.npz` (and optional
      `scaled_input_distribution.npz` for `analysis_confidence`) from either a
      user-supplied directory or the installed `neuralfoil` package.
    - Slices the final output layer to rows `[0..3]` (`[conf, CL, CD, CM]`) by default,
      eliminating ~98% of final-layer FLOPs with zero accuracy loss on aerodynamic forces.
    - Evaluates normal + alpha-flipped orientations in a single fused C++ `MLPNode`
      batched pass (`2 * N` columns), supporting both numeric NumPy arrays and symbolic
      `fastopt.Expr` inputs with exact 1st & 2nd-order AD.
    - Can export the exact loaded weights directly into `fast_bemt.NeuralFoilModel` and
      `fast_bemt.Airfoil3D.from_neuralfoil(...)` for zero-GIL C++ rotor BEMT evaluation.
    """

    ALLOWABLE_MODEL_SIZES = (
        "xxsmall", "xsmall", "small", "medium",
        "large", "xlarge", "xxlarge", "xxxlarge",
    )

    def __init__(
        self,
        mlp: MLP,
        *,
        weights: Sequence[np.ndarray],
        biases: Sequence[np.ndarray],
        use_float32: bool = False,
        model_size: str = "custom",
        mean_inputs_scaled: Optional[np.ndarray] = None,
        inv_cov_inputs_scaled: Optional[np.ndarray] = None,
    ) -> None:
        self.mlp = mlp
        self.weights = [np.asarray(w, dtype=np.float64) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float64).ravel() for b in biases]
        self.use_float32 = bool(use_float32)
        self.model_size = str(model_size)
        self.mean_inputs_scaled = (
            np.asarray(mean_inputs_scaled, dtype=np.float64).ravel()
            if mean_inputs_scaled is not None
            else None
        )
        self.inv_cov_inputs_scaled = (
            np.asarray(inv_cov_inputs_scaled, dtype=np.float64)
            if inv_cov_inputs_scaled is not None
            else None
        )

    @staticmethod
    def find_weights_dir(weights_dir: Optional[Union[str, os.PathLike]] = None) -> str:
        """Locate NeuralFoil `nn_weights_and_biases` directory."""
        if weights_dir is not None:
            cand = os.path.abspath(os.fspath(weights_dir))
            if os.path.isdir(cand):
                return cand
            raise FileNotFoundError(f"NeuralFoil weights_dir not found: {cand}")
        try:
            import neuralfoil  # type: ignore

            nf_dir = os.path.join(os.path.dirname(neuralfoil.__file__), "nn_weights_and_biases")
            if os.path.isdir(nf_dir):
                return nf_dir
        except ImportError:
            pass
        raise FileNotFoundError(
            "Could not auto-locate NeuralFoil `nn_weights_and_biases`. "
            "Install `neuralfoil` or pass `weights_dir=` / `npz_path=` explicitly."
        )

    @classmethod
    def from_npz(
        cls,
        npz_path: Union[str, os.PathLike],
        *,
        activation: str = "silu",
        use_float32: bool = False,
        slice_aero_outputs: bool = True,
        distribution_npz_path: Optional[Union[str, os.PathLike]] = None,
        model_size: str = "custom",
        name: str = "neuralfoil",
    ) -> "NeuralFoilSurrogate":
        """Load a NeuralFoil-format `.npz` weight file (`net.0.weight`, `net.0.bias`, ...)."""
        raw = np.load(npz_path, allow_pickle=False)
        pattern = re.compile(r"^(?:net\.)?(\d+)\.weight$")
        indexed: List[Tuple[int, str, str]] = []
        for k in raw.files:
            m = pattern.match(k)
            if m:
                idx = int(m.group(1))
                b_key = k[:-6] + "bias"
                if b_key not in raw.files:
                    raise KeyError(f"Missing matching bias key \x27{b_key}\x27 for \x27{k}\x27 in {npz_path}.")
                indexed.append((idx, k, b_key))
        if not indexed:
            raise ValueError(
                f"No NeuralFoil weight keys (\x27net.<idx>.weight\x27) found in {npz_path}."
            )
        indexed.sort(key=lambda t: t[0])
        weights = [np.asarray(raw[wk], dtype=np.float64) for _, wk, _ in indexed]
        biases = [np.asarray(raw[bk], dtype=np.float64).ravel() for _, _, bk in indexed]

        if slice_aero_outputs and weights[-1].shape[0] > 4:
            weights[-1] = weights[-1][:4, :].copy()
            biases[-1] = biases[-1][:4].copy()

        mean_in = None
        inv_cov_in = None
        if distribution_npz_path is None:
            sibling = os.path.join(os.path.dirname(os.fspath(npz_path)), "scaled_input_distribution.npz")
            if os.path.isfile(sibling):
                distribution_npz_path = sibling
        if distribution_npz_path is not None and os.path.isfile(distribution_npz_path):
            dist = np.load(distribution_npz_path, allow_pickle=False)
            if "mean_inputs_scaled" in dist and "inv_cov_inputs_scaled" in dist:
                mean_in = np.asarray(dist["mean_inputs_scaled"], dtype=np.float64).ravel()
                inv_cov_in = np.asarray(dist["inv_cov_inputs_scaled"], dtype=np.float64)

        mlp = MLP.from_weights(
            weights,
            biases,
            activations=activation,
            output_activation="identity",
            use_float32=use_float32,
            name=name,
        )
        return cls(
            mlp=mlp,
            weights=weights,
            biases=biases,
            use_float32=use_float32,
            model_size=model_size,
            mean_inputs_scaled=mean_in,
            inv_cov_inputs_scaled=inv_cov_in,
        )

    @classmethod
    def from_model_size(
        cls,
        model_size: str = "large",
        *,
        weights_dir: Optional[Union[str, os.PathLike]] = None,
        use_float32: bool = False,
        slice_aero_outputs: bool = True,
        name: Optional[str] = None,
    ) -> "NeuralFoilSurrogate":
        """Load a standard NeuralFoil model (`xxsmall`..`xxxlarge`) from `nn_weights_and_biases`."""
        wdir = cls.find_weights_dir(weights_dir)
        npz_path = os.path.join(wdir, f"nn-{model_size}.npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"NeuralFoil weight file not found: {npz_path}")
        dist_path = os.path.join(wdir, "scaled_input_distribution.npz")
        return cls.from_npz(
            npz_path,
            use_float32=use_float32,
            slice_aero_outputs=slice_aero_outputs,
            distribution_npz_path=(dist_path if os.path.isfile(dist_path) else None),
            model_size=model_size,
            name=(name or f"neuralfoil_{model_size}"),
        )

    def get_aero_from_kulfan_parameters(
        self,
        kulfan_parameters: Dict[str, Any],
        alpha: Union[float, np.ndarray, Expr],
        Re: Union[float, np.ndarray, Expr],
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        num_threads: int = 0,
    ) -> Dict[str, Any]:
        """
        Evaluate NeuralFoil aerodynamics (`CL`, `CD`, `CM`, `analysis_confidence`) with
        exact parity to `neuralfoil.get_aero_from_kulfan_parameters`.

        Supports both numeric NumPy arrays (vectorized, GIL-free C++ `CompiledMLP`) and
        symbolic `fastopt.Expr` inputs (fused C++ `MLPNode` with exact 1st & 2nd AD).
        """
        uw = kulfan_parameters["upper_weights"]
        lw = kulfan_parameters["lower_weights"]
        le = kulfan_parameters["leading_edge_weight"]
        te = kulfan_parameters["TE_thickness"]

        is_sym = any(isinstance(v, Expr) for v in (uw, lw, le, te, alpha, Re))
        if is_sym:
            from . import expr as _expr

            # Symbolic single-point evaluation via fused 2-column [x, x_flipped] MLPNode
            def _to_col(val: Any, rows: int) -> Expr:
                if isinstance(val, Expr):
                    return val.reshape(rows, 1) if val.shape != (rows, 1) else val
                arr = np.asarray(val, dtype=np.float64).reshape((rows, 1))
                return g_const(arr)

            # Locate graph from first symbolic input
            g = next(v.graph for v in (uw, lw, le, te, alpha, Re) if isinstance(v, Expr))
            def g_const(arr: np.ndarray) -> Expr:
                from ._fastopt_impl import Shape
                cid = g.add_constant(Shape(arr.shape[0], arr.shape[1]), arr.ravel(order="F"))
                return Expr(g, cid)

            uw_e = _to_col(uw, 8)
            lw_e = _to_col(lw, 8)
            le_e = _to_col(le, 1)
            te_e = _to_col(te, 1) * 50.0
            a_rad = _to_col(alpha, 1) * (np.pi / 180.0)
            re_e = _to_col(Re, 1)
            sin2a = _expr.sin(2.0 * a_rad)
            cosa = _expr.cos(a_rad)
            omc2 = 1.0 - _expr.square(cosa)
            re_s = (_expr.log(re_e) - 12.5) / 3.5
            nc_s = g_const(np.array([[(float(n_crit) - 9.0) / 4.5]]))
            xu_s = g_const(np.array([[float(xtr_upper)]]))
            xl_s = g_const(np.array([[float(xtr_lower)]]))

            x_norm = _expr.vstack([uw_e, lw_e, le_e, te_e, sin2a, cosa, omc2, re_s, nc_s, xu_s, xl_s])
            x_flip = _expr.vstack([-lw_e, -uw_e, -le_e, te_e, -sin2a, cosa, omc2, re_s, nc_s, xl_s, xu_s])
            X_pair = _expr.hstack([x_norm, x_flip])
            Y_pair = self.mlp(X_pair)  # shape (>=4, 2)

            cl = 0.25 * (Y_pair[1, 0] - Y_pair[1, 1])
            cd = _expr.exp(Y_pair[2, 0] + Y_pair[2, 1] - 4.0)
            cm = 0.025 * (Y_pair[3, 0] - Y_pair[3, 1])
            y0_n = Y_pair[0, 0]
            y0_f = Y_pair[0, 1]
            if self.mean_inputs_scaled is not None and self.inv_cov_inputs_scaled is not None:
                mean_e = g_const(self.mean_inputs_scaled.reshape(25, 1))
                inv_cov_e = g_const(self.inv_cov_inputs_scaled)
                diff_n = x_norm - mean_e
                diff_f = x_flip - mean_e
                mah_n = _expr.sum((inv_cov_e @ diff_n) * diff_n) / 50.0
                mah_f = _expr.sum((inv_cov_e @ diff_f) * diff_f) / 50.0
                y0_n = y0_n - mah_n
                y0_f = y0_f - mah_f
            conf = 0.5 * (1.0 + _expr.tanh(0.25 * (y0_n + y0_f)))
            return {"CL": cl, "CD": cd, "CM": cm, "analysis_confidence": conf}

        # Numeric vectorized evaluation: stack [X_normal, X_flipped] into (25, 2*N)
        uw_a = np.asarray(uw, dtype=np.float64)
        lw_a = np.asarray(lw, dtype=np.float64)
        a_arr = np.atleast_1d(np.asarray(alpha, dtype=np.float64)).ravel()
        re_arr = np.atleast_1d(np.asarray(Re, dtype=np.float64)).ravel()
        N = max(a_arr.size, re_arr.size)
        if a_arr.size == 1 and N > 1:
            a_arr = np.full(N, a_arr[0], dtype=np.float64)
        if re_arr.size == 1 and N > 1:
            re_arr = np.full(N, re_arr[0], dtype=np.float64)

        a_rad = np.deg2rad(a_arr)
        sin2a = np.sin(2.0 * a_rad)
        cosa = np.cos(a_rad)
        omc2 = 1.0 - cosa * cosa
        re_s = (np.log(np.maximum(re_arr, 100.0)) - 12.5) / 3.5
        nc_s = np.full(N, (float(n_crit) - 9.0) / 4.5, dtype=np.float64)
        xu_s = np.full(N, float(xtr_upper), dtype=np.float64)
        xl_s = np.full(N, float(xtr_lower), dtype=np.float64)
        le_s = np.broadcast_to(np.asarray(le, dtype=np.float64), (N,))
        te_s = np.broadcast_to(np.asarray(te, dtype=np.float64) * 50.0, (N,))

        uw_m = np.broadcast_to(uw_a.reshape(8, -1), (8, N))
        lw_m = np.broadcast_to(lw_a.reshape(8, -1), (8, N))

        X = np.empty((25, 2 * N), dtype=np.float64, order="F")
        X[0:8, :N] = uw_m
        X[8:16, :N] = lw_m
        X[16, :N] = le_s
        X[17, :N] = te_s
        X[18, :N] = sin2a
        X[19, :N] = cosa
        X[20, :N] = omc2
        X[21, :N] = re_s
        X[22, :N] = nc_s
        X[23, :N] = xu_s
        X[24, :N] = xl_s

        X[0:8, N:] = -lw_m
        X[8:16, N:] = -uw_m
        X[16, N:] = -le_s
        X[17, N:] = te_s
        X[18, N:] = -sin2a
        X[19, N:] = cosa
        X[20, N:] = omc2
        X[21, N:] = re_s
        X[22, N:] = nc_s
        X[23, N:] = xl_s
        X[24, N:] = xu_s

        Y = self.mlp.eval(X, num_threads=num_threads)  # shape (>=4, 2*N)
        y0_n = Y[0, :N].copy()
        y0_f = Y[0, N:].copy()
        if self.mean_inputs_scaled is not None and self.inv_cov_inputs_scaled is not None:
            diff = X - self.mean_inputs_scaled[:, None]
            mah = np.sum((self.inv_cov_inputs_scaled @ diff) * diff, axis=0) / 50.0
            y0_n -= mah[:N]
            y0_f -= mah[N:]

        conf = 0.5 * (1.0 + np.tanh(0.25 * (y0_n + y0_f)))
        cl = 0.25 * (Y[1, :N] - Y[1, N:])
        cd = np.exp(Y[2, :N] + Y[2, N:] - 4.0)
        cm = 0.025 * (Y[3, :N] - Y[3, N:])
        return {"CL": cl, "CD": cd, "CM": cm, "analysis_confidence": conf}

    def to_fast_bemt_model(self) -> Any:
        """Construct a native C++ `fast_bemt.NeuralFoilModel` sharing these exact weights."""
        import fast_bemt as fb  # type: ignore

        return fb.NeuralFoilModel(
            self.weights,
            self.biases,
            use_float32=self.use_float32,
            slice_aero_outputs=True,
            mean_inputs_scaled=(
                self.mean_inputs_scaled if self.mean_inputs_scaled is not None else np.empty(0)
            ),
            inv_cov_inputs_scaled=(
                self.inv_cov_inputs_scaled
                if self.inv_cov_inputs_scaled is not None
                else np.empty((0, 0))
            ),
        )

    def make_fast_bemt_airfoil(
        self,
        kulfan_or_kaf: Any,
        *,
        bemt_model: Any = None,
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        aspect_ratio: float = 10.0,
        enable_extrapolation: bool = True,
        enable_compressibility: bool = False,
    ) -> Any:
        """Build a native C++ `fast_bemt.Airfoil3D.from_neuralfoil(...)` directly from CST params."""
        import fast_bemt as fb  # type: ignore

        if bemt_model is None:
            bemt_model = self.to_fast_bemt_model()
        kp = (
            kulfan_or_kaf.kulfan_parameters
            if hasattr(kulfan_or_kaf, "kulfan_parameters")
            else kulfan_or_kaf
        )
        return fb.Airfoil3D.from_neuralfoil(
            bemt_model,
            np.asarray(kp["upper_weights"], dtype=np.float64),
            np.asarray(kp["lower_weights"], dtype=np.float64),
            float(kp["leading_edge_weight"]),
            float(kp["TE_thickness"]),
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
            aspect_ratio=aspect_ratio,
            enable_extrapolation=enable_extrapolation,
            enable_compressibility=enable_compressibility,
        )

    def __call__(self, x: Union[Expr, np.ndarray]) -> Union[Expr, np.ndarray]:
        return self.mlp(x)

    def eval(self, X: np.ndarray, num_threads: int = 0) -> np.ndarray:
        return self.mlp.eval(X, num_threads=num_threads)




class Conv1D:
    """
    Fused C++17 1D Convolution (`OpType.Conv1D`, im2col + BLAS3 GEMM + fused activation)
    with exact 0th, 1st, and 2nd-order AD and local receptive-field banded sparsity.

    Input shape: `(c_in, l_in)` -> Output shape: `(c_out, l_out)`.
    """

    def __init__(
        self,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        *,
        c_in: Optional[int] = None,
        kernel_size: Optional[int] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        activation: Union[str, NNActivation] = "identity",
        use_float32: bool = False,
        name: str = "conv1d",
    ) -> None:
        W_arr = np.asarray(weight, dtype=np.float64)
        if W_arr.ndim == 3:
            # PyTorch / ONNX layout: (c_out, c_in, K) -> reorder to (c_out, K * c_in) matching im2col
            c_out, ci, K = W_arr.shape
            W_mat = np.transpose(W_arr, (0, 2, 1)).reshape((c_out, K * ci))
            c_in_use, k_use = int(ci), int(K)
        elif W_arr.ndim == 2:
            if c_in is None or kernel_size is None:
                raise ValueError("2D weight matrix requires explicit c_in and kernel_size.")
            c_out = W_arr.shape[0]
            W_mat = W_arr
            c_in_use, k_use = int(c_in), int(kernel_size)
        else:
            raise ValueError(f"Conv1D weight must be 3D (c_out, c_in, K) or 2D, got {W_arr.shape}.")

        b_vec = (
            np.zeros(c_out, dtype=np.float64)
            if bias is None
            else np.asarray(bias, dtype=np.float64).ravel()
        )
        self.name = str(name)
        self._compiled = CompiledConv1D(
            np.asfortranarray(W_mat, dtype=np.float64),
            np.ascontiguousarray(b_vec, dtype=np.float64),
            c_in_use,
            k_use,
            int(stride),
            int(padding),
            int(dilation),
            resolve_activation(activation),
            bool(use_float32),
            self.name,
        )

    @property
    def c_in(self) -> int:
        return int(self._compiled.c_in())

    @property
    def c_out(self) -> int:
        return int(self._compiled.c_out())

    @property
    def kernel_size(self) -> int:
        return int(self._compiled.kernel_size())

    @classmethod
    def random(
        cls,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        activation: Union[str, NNActivation] = "silu",
        use_float32: bool = False,
        seed: int = 42,
        name: str = "conv1d_rand",
    ) -> "Conv1D":
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (c_in * kernel_size + c_out))
        W = rng.normal(0.0, scale, size=(c_out, c_in, kernel_size))
        b = rng.normal(0.0, 0.05, size=(c_out,))
        return cls(
            W, b, stride=stride, padding=padding, dilation=dilation,
            activation=activation, use_float32=use_float32, name=name,
        )

    def eval(self, X: np.ndarray) -> np.ndarray:
        X2 = np.asfortranarray(X, dtype=np.float64)
        return np.asarray(self._compiled.eval(X2), dtype=np.float64)

    def __call__(self, x: Union[Expr, np.ndarray]) -> Union[Expr, np.ndarray]:
        if isinstance(x, Expr):
            out_id = op_conv1d(x.graph, x.id, self._compiled, self.name)
            return Expr(x.graph, out_id)
        return self.eval(np.asarray(x, dtype=np.float64))


class MultiHeadAttention:
    """
    Fused C++17 Multi-Head Self-Attention (`OpType.Attention`) with exact 0th, 1st,
    and 2nd-order Forward-over-Reverse AD, optional causal masking and residual skip.

    Input shape: `(d_model, seq_len)` -> Output shape: `(d_model, seq_len)`.
    """

    def __init__(
        self,
        Wq: np.ndarray,
        bq: np.ndarray,
        Wk: np.ndarray,
        bk: np.ndarray,
        Wv: np.ndarray,
        bv: np.ndarray,
        Wo: np.ndarray,
        bo: np.ndarray,
        *,
        num_heads: int = 1,
        causal: bool = False,
        residual: bool = False,
        use_float32: bool = False,
        name: str = "mha",
    ) -> None:
        self.name = str(name)
        self._compiled = CompiledAttention(
            np.asfortranarray(Wq, dtype=np.float64), np.ascontiguousarray(bq, dtype=np.float64).ravel(),
            np.asfortranarray(Wk, dtype=np.float64), np.ascontiguousarray(bk, dtype=np.float64).ravel(),
            np.asfortranarray(Wv, dtype=np.float64), np.ascontiguousarray(bv, dtype=np.float64).ravel(),
            np.asfortranarray(Wo, dtype=np.float64), np.ascontiguousarray(bo, dtype=np.float64).ravel(),
            int(num_heads), bool(causal), bool(residual), bool(use_float32), self.name,
        )

    @property
    def d_model(self) -> int:
        return int(self._compiled.d_model())

    @property
    def num_heads(self) -> int:
        return int(self._compiled.num_heads())

    @classmethod
    def random(
        cls,
        d_model: int,
        num_heads: int = 4,
        *,
        causal: bool = False,
        residual: bool = True,
        use_float32: bool = False,
        seed: int = 42,
        name: str = "mha_rand",
    ) -> "MultiHeadAttention":
        rng = np.random.default_rng(seed)
        s = np.sqrt(1.0 / d_model)
        return cls(
            rng.normal(0.0, s, (d_model, d_model)), rng.normal(0.0, 0.02, (d_model,)),
            rng.normal(0.0, s, (d_model, d_model)), rng.normal(0.0, 0.02, (d_model,)),
            rng.normal(0.0, s, (d_model, d_model)), rng.normal(0.0, 0.02, (d_model,)),
            rng.normal(0.0, s, (d_model, d_model)), rng.normal(0.0, 0.02, (d_model,)),
            num_heads=num_heads, causal=causal, residual=residual,
            use_float32=use_float32, name=name,
        )

    def eval(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._compiled.eval(np.asfortranarray(X, dtype=np.float64)), dtype=np.float64)

    def __call__(self, x: Union[Expr, np.ndarray]) -> Union[Expr, np.ndarray]:
        if isinstance(x, Expr):
            out_id = op_attention(x.graph, x.id, self._compiled, self.name)
            return Expr(x.graph, out_id)
        return self.eval(np.asarray(x, dtype=np.float64))


class TransformerBlock:
    """Fused Transformer Encoder/Decoder block: Residual Multi-Head Attention + Residual FFN MLP."""

    def __init__(self, attn: MultiHeadAttention, ffn: MLP, *, name: str = "transformer_block") -> None:
        self.attn = attn
        self.ffn = ffn
        self.name = str(name)

    @classmethod
    def random(
        cls,
        d_model: int,
        num_heads: int = 4,
        ffn_hidden: Optional[int] = None,
        *,
        activation: str = "gelu",
        causal: bool = False,
        use_float32: bool = False,
        seed: int = 42,
        name: str = "transformer_block",
    ) -> "TransformerBlock":
        h_dim = ffn_hidden or (2 * d_model)
        attn = MultiHeadAttention.random(
            d_model, num_heads=num_heads, causal=causal, residual=True,
            use_float32=use_float32, seed=seed, name=f"{name}_attn",
        )
        ffn = MLP.random(
            [d_model, h_dim, d_model], activation=activation, layer_norm=True,
            use_float32=use_float32, seed=seed + 1, name=f"{name}_ffn",
        )
        return cls(attn, ffn, name=name)

    def __call__(self, x: Union[Expr, np.ndarray]) -> Union[Expr, np.ndarray]:
        h = self.attn(x)
        return h + self.ffn(h)


class Sequential:
    """Chain arbitrary `fastopt.nn` modules (`MLP`, `Conv1D`, `MultiHeadAttention`, `TransformerBlock`, `ONNXModel`)."""

    def __init__(self, *modules: Any) -> None:
        self.modules = list(modules)

    def __call__(self, x: Union[Expr, np.ndarray]) -> Union[Expr, np.ndarray]:
        h = x
        for mod in self.modules:
            h = mod(h)
        return h


def _as_pair(v: Union[int, Tuple[int, int], Sequence[int]]) -> Tuple[int, int]:
    if isinstance(v, (int, np.integer)):
        iv = int(v)
        return iv, iv
    seq = list(v)
    if len(seq) == 1:
        iv = int(seq[0])
        return iv, iv
    if len(seq) != 2:
        raise ValueError(f"Expected int or 2-tuple, got {v!r}.")
    return int(seq[0]), int(seq[1])


class Conv2D:
    """
    Fused C++17 2D Convolution (`OpType.Conv2D`, im2col + BLAS3 GEMM + fused activation)
    with exact 0th, 1st, and 2nd-order AD and 2D local receptive-field sparsity.

    Symbolic input shape: `(c_in, h_in * w_in)` (row-major spatial flattening `r * w_in + c`)
    -> Symbolic output shape: `(c_out, h_out * w_out)`.
    Numeric input also accepts 3D `(c_in, h_in, w_in)`.
    """

    def __init__(
        self,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        *,
        c_in: Optional[int] = None,
        kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        activation: Union[str, NNActivation] = "identity",
        input_hw: Optional[Tuple[int, int]] = None,
        use_float32: bool = False,
        name: str = "conv2d",
    ) -> None:
        W_arr = np.asarray(weight, dtype=np.float64)
        if W_arr.ndim == 4:
            # PyTorch / ONNX layout: (c_out, c_in, kh, kw) -> (c_out, kh * kw * c_in) matching C++ im2col_2d
            c_out, ci, kh, kw = W_arr.shape
            W_mat = np.transpose(W_arr, (0, 2, 3, 1)).reshape((c_out, kh * kw * ci))
            c_in_use, kh_use, kw_use = int(ci), int(kh), int(kw)
        elif W_arr.ndim == 2:
            if c_in is None or kernel_size is None:
                raise ValueError("2D weight matrix requires explicit c_in and kernel_size.")
            c_out = W_arr.shape[0]
            kh_use, kw_use = _as_pair(kernel_size)
            c_in_use = int(c_in)
            W_mat = W_arr
        else:
            raise ValueError(f"Conv2D weight must be 4D (c_out, c_in, kh, kw) or 2D, got {W_arr.shape}.")

        b_vec = (
            np.zeros(c_out, dtype=np.float64)
            if bias is None
            else np.asarray(bias, dtype=np.float64).ravel()
        )
        sh, sw = _as_pair(stride)
        ph, pw = _as_pair(padding)
        dh, dw = _as_pair(dilation)
        self.kernel_size = (kh_use, kw_use)
        self.stride = (sh, sw)
        self.padding = (ph, pw)
        self.dilation = (dh, dw)
        self.input_hw = tuple(int(v) for v in input_hw) if input_hw is not None else None
        self.name = str(name)
        self._compiled = CompiledConv2D(
            np.asfortranarray(W_mat, dtype=np.float64),
            np.ascontiguousarray(b_vec, dtype=np.float64),
            c_in_use, kh_use, kw_use,
            sh, sw, ph, pw, dh, dw,
            resolve_activation(activation),
            bool(use_float32),
            self.name,
        )

    @property
    def c_in(self) -> int:
        return int(self._compiled.c_in())

    @property
    def c_out(self) -> int:
        return int(self._compiled.c_out())

    def out_hw(self, h_in: int, w_in: int) -> Tuple[int, int]:
        oh, ow = self._compiled.out_hw(int(h_in), int(w_in))
        return int(oh), int(ow)

    @classmethod
    def random(
        cls,
        c_in: int,
        c_out: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        *,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        activation: Union[str, NNActivation] = "silu",
        input_hw: Optional[Tuple[int, int]] = None,
        use_float32: bool = False,
        seed: int = 42,
        name: str = "conv2d_rand",
    ) -> "Conv2D":
        kh, kw = _as_pair(kernel_size)
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (c_in * kh * kw + c_out))
        W = rng.normal(0.0, scale, size=(c_out, c_in, kh, kw))
        b = rng.normal(0.0, 0.05, size=(c_out,))
        return cls(
            W, b, stride=stride, padding=padding, dilation=dilation,
            activation=activation, input_hw=input_hw, use_float32=use_float32, name=name,
        )

    def _resolve_hw(self, h_in: Optional[int], w_in: Optional[int]) -> Tuple[int, int]:
        if h_in is not None and w_in is not None:
            return int(h_in), int(w_in)
        if self.input_hw is not None:
            return self.input_hw
        raise ValueError(f"Conv2D '{self.name}' requires (h_in, w_in) or constructor input_hw=(H, W).")

    def eval(
        self,
        X: np.ndarray,
        h_in: Optional[int] = None,
        w_in: Optional[int] = None,
    ) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 3:
            ci, hi, wi = arr.shape
            X2 = np.asfortranarray(arr.reshape((ci, hi * wi)))
            Y2 = np.asarray(self._compiled.eval(X2, int(hi), int(wi)), dtype=np.float64)
            ho, wo = self.out_hw(hi, wi)
            return Y2.reshape((self.c_out, ho, wo))
        hi, wi = self._resolve_hw(h_in, w_in)
        X2 = np.asfortranarray(arr)
        return np.asarray(self._compiled.eval(X2, hi, wi), dtype=np.float64)

    def __call__(
        self,
        x: Union[Expr, np.ndarray],
        h_in: Optional[int] = None,
        w_in: Optional[int] = None,
    ) -> Union[Expr, np.ndarray]:
        if isinstance(x, Expr):
            hi, wi = self._resolve_hw(h_in, w_in)
            out_id = op_conv2d(x.graph, x.id, hi, wi, self._compiled, self.name)
            return Expr(x.graph, out_id)
        return self.eval(np.asarray(x, dtype=np.float64), h_in=h_in, w_in=w_in)


class _Pool2DBase:
    """Base class for C++17 2D Pooling layers (`AvgPool2D`, `MaxPool2D`) with exact 0/1/2 AD."""

    _mode: Pool2DMode

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        *,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        input_hw: Optional[Tuple[int, int]] = None,
        name: str = "pool2d",
    ) -> None:
        self.kernel_size = _as_pair(kernel_size)
        self.stride = _as_pair(kernel_size if stride is None else stride)
        self.padding = _as_pair(padding)
        self.input_hw = tuple(int(v) for v in input_hw) if input_hw is not None else None
        self.name = str(name)

    def out_hw(self, h_in: int, w_in: int) -> Tuple[int, int]:
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        ho = (int(h_in) + 2 * ph - kh) // sh + 1
        wo = (int(w_in) + 2 * pw - kw) // sw + 1
        if ho <= 0 or wo <= 0:
            raise ValueError(f"Pool2D '{self.name}' produced invalid output size ({ho}, {wo}).")
        return ho, wo

    def _resolve_hw(self, h_in: Optional[int], w_in: Optional[int]) -> Tuple[int, int]:
        if h_in is not None and w_in is not None:
            return int(h_in), int(w_in)
        if self.input_hw is not None:
            return self.input_hw
        raise ValueError(f"Pool2D '{self.name}' requires (h_in, w_in) or constructor input_hw=(H, W).")

    def eval(
        self,
        X: np.ndarray,
        h_in: Optional[int] = None,
        w_in: Optional[int] = None,
    ) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        is_3d = arr.ndim == 3
        if is_3d:
            C, hi, wi = arr.shape
            X2 = arr.reshape((C, hi * wi))
        else:
            hi, wi = self._resolve_hw(h_in, w_in)
            C = arr.shape[0]
            X2 = arr
        ho, wo = self.out_hw(hi, wi)
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        Y2 = np.empty((C, ho * wo), dtype=np.float64)
        is_avg = self._mode == Pool2DMode.Avg
        for r_out in range(ho):
            for c_out_i in range(wo):
                col_o = r_out * wo + c_out_i
                vals = []
                for dr in range(kh):
                    ri = r_out * sh - ph + dr
                    if ri < 0 or ri >= hi:
                        continue
                    for dc in range(kw):
                        ci = c_out_i * sw - pw + dc
                        if ci < 0 or ci >= wi:
                            continue
                        vals.append(X2[:, ri * wi + ci])
                if is_avg:
                    Y2[:, col_o] = np.mean(vals, axis=0) if vals else 0.0
                else:
                    Y2[:, col_o] = np.max(vals, axis=0) if vals else -np.inf
        return Y2.reshape((C, ho, wo)) if is_3d else Y2

    def __call__(
        self,
        x: Union[Expr, np.ndarray],
        h_in: Optional[int] = None,
        w_in: Optional[int] = None,
    ) -> Union[Expr, np.ndarray]:
        if isinstance(x, Expr):
            hi, wi = self._resolve_hw(h_in, w_in)
            kh, kw = self.kernel_size
            sh, sw = self.stride
            ph, pw = self.padding
            out_id = op_pool2d(
                x.graph, x.id, self._mode, hi, wi, kh, kw, sh, sw, ph, pw, self.name
            )
            return Expr(x.graph, out_id)
        return self.eval(np.asarray(x, dtype=np.float64), h_in=h_in, w_in=w_in)


class AvgPool2D(_Pool2DBase):
    """Native C++17 2D Average Pooling (`OpType.Pool2D`, Avg) with exact 0/1/2 AD."""
    _mode = Pool2DMode.Avg

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        *,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        input_hw: Optional[Tuple[int, int]] = None,
        name: str = "avgpool2d",
    ) -> None:
        super().__init__(kernel_size, stride=stride, padding=padding, input_hw=input_hw, name=name)


class MaxPool2D(_Pool2DBase):
    """Native C++17 2D Max Pooling (`OpType.Pool2D`, Max) with exact 0/1/2 AD."""
    _mode = Pool2DMode.Max

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        *,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        input_hw: Optional[Tuple[int, int]] = None,
        name: str = "maxpool2d",
    ) -> None:
        super().__init__(kernel_size, stride=stride, padding=padding, input_hw=input_hw, name=name)


class GlobalAvgPool2D:
    """
    Native C++17 Global Average Pooling over spatial positions: `(C, H * W) -> (C, 1)`.
    Works without requiring explicit `(H, W)` dimensions.
    """

    def __init__(self, name: str = "global_avgpool2d") -> None:
        self.name = str(name)

    def eval(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 3:
            return arr.mean(axis=(1, 2), keepdims=False).reshape((-1, 1))
        if arr.ndim == 2:
            return arr.mean(axis=1, keepdims=True)
        raise ValueError(f"GlobalAvgPool2D expects 2D (C, N) or 3D (C, H, W), got {arr.shape}.")

    def __call__(self, x: Union[Expr, np.ndarray]) -> Union[Expr, np.ndarray]:
        if isinstance(x, Expr):
            n_spatial = int(x.shape[1])
            out_id = op_pool2d(
                x.graph, x.id, Pool2DMode.Avg,
                1, n_spatial, 1, n_spatial, 1, n_spatial, 0, 0, self.name,
            )
            return Expr(x.graph, out_id)
        return self.eval(np.asarray(x, dtype=np.float64))


class BatchNorm:
    """
    Inference-mode Batch Normalization (channel-wise affine transform) with exact 0/1/2 AD:
        y[c, :] = (gamma[c] / sqrt(running_var[c] + eps)) * (x[c, :] - running_mean[c]) + beta[c]

    Also provides `fold_into_conv_weights(W, b)` to algebraically fold BatchNorm into
    a preceding `Conv1D`, `Conv2D`, or `Linear` layer with zero runtime cost.
    """

    def __init__(
        self,
        running_mean: np.ndarray,
        running_var: np.ndarray,
        gamma: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        *,
        eps: float = 1e-5,
        name: str = "batch_norm",
    ) -> None:
        mean = np.asarray(running_mean, dtype=np.float64).ravel()
        var = np.asarray(running_var, dtype=np.float64).ravel()
        c = mean.size
        if var.size != c:
            raise ValueError(f"BatchNorm running_mean ({c}) and running_var ({var.size}) mismatch.")
        g = np.ones(c, dtype=np.float64) if gamma is None else np.asarray(gamma, dtype=np.float64).ravel()
        b = np.zeros(c, dtype=np.float64) if beta is None else np.asarray(beta, dtype=np.float64).ravel()
        if g.size != c or b.size != c:
            raise ValueError("BatchNorm gamma/beta length must match num_features.")
        self.num_features = c
        self.eps = float(eps)
        self.name = str(name)
        self.scale = g / np.sqrt(var + self.eps)
        self.shift = b - mean * self.scale

    @classmethod
    def random(cls, num_features: int, *, eps: float = 1e-5, seed: int = 42, name: str = "bn_rand") -> "BatchNorm":
        rng = np.random.default_rng(seed)
        return cls(
            running_mean=rng.normal(0.0, 0.3, size=(num_features,)),
            running_var=rng.uniform(0.5, 2.0, size=(num_features,)),
            gamma=rng.uniform(0.7, 1.3, size=(num_features,)),
            beta=rng.normal(0.0, 0.1, size=(num_features,)),
            eps=eps,
            name=name,
        )

    def fold_into_conv_weights(
        self,
        W: np.ndarray,
        b: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fold this BatchNorm into preceding Linear `(C_out, D_in)`, Conv1D `(C_out, C_in, K)`, or Conv2D `(C_out, C_in, KH, KW)`."""
        W_arr = np.asarray(W, dtype=np.float64)
        c_out = W_arr.shape[0]
        if c_out != self.num_features:
            raise ValueError(f"Cannot fold BatchNorm({self.num_features}) into weights with c_out={c_out}.")
        b_arr = np.zeros(c_out, dtype=np.float64) if b is None else np.asarray(b, dtype=np.float64).ravel()
        reshape_dims = (c_out,) + (1,) * (W_arr.ndim - 1)
        W_folded = W_arr * self.scale.reshape(reshape_dims)
        b_folded = b_arr * self.scale + self.shift
        return W_folded, b_folded

    def eval(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            return arr * self.scale + self.shift
        reshape_dims = (self.num_features,) + (1,) * (arr.ndim - 1)
        return arr * self.scale.reshape(reshape_dims) + self.shift.reshape(reshape_dims)

    def __call__(self, x: Union[Expr, np.ndarray]) -> Union[Expr, np.ndarray]:
        if isinstance(x, Expr):
            c, n = x.shape
            if c != self.num_features:
                raise ValueError(f"BatchNorm '{self.name}' expects {self.num_features} channels, got {c}.")
            s_mat = np.broadcast_to(self.scale[:, None], (c, n)).copy()
            b_mat = np.broadcast_to(self.shift[:, None], (c, n)).copy()
            return x * s_mat + b_mat
        return self.eval(np.asarray(x, dtype=np.float64))


class RMSNorm:
    """
    Native C++17 Root-Mean-Square Layer Normalization (`OpType.RMSNorm`) with exact
    0th, 1st, and 2nd-order Forward-over-Reverse AD:
        y[:, j] = gamma * x[:, j] / sqrt(mean(x[:, j]^2) + eps)
    """

    def __init__(
        self,
        gamma: Optional[np.ndarray] = None,
        *,
        eps: float = 1e-6,
        name: str = "rms_norm",
    ) -> None:
        self.gamma = (
            np.empty(0, dtype=np.float64)
            if gamma is None
            else np.ascontiguousarray(gamma, dtype=np.float64).ravel()
        )
        self.eps = float(eps)
        self.name = str(name)

    @classmethod
    def random(cls, dim: int, *, eps: float = 1e-6, seed: int = 42, name: str = "rmsnorm_rand") -> "RMSNorm":
        rng = np.random.default_rng(seed)
        return cls(gamma=rng.uniform(0.8, 1.2, size=(dim,)), eps=eps, name=name)

    def eval(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        is_1d = arr.ndim == 1
        X2 = arr.reshape((-1, 1)) if is_1d else arr
        inv_rms = 1.0 / np.sqrt(np.mean(X2 * X2, axis=0, keepdims=True) + self.eps)
        Y2 = X2 * inv_rms
        if self.gamma.size > 0:
            Y2 = Y2 * self.gamma[:, None]
        return Y2[:, 0] if is_1d else Y2

    def __call__(self, x: Union[Expr, np.ndarray]) -> Union[Expr, np.ndarray]:
        if isinstance(x, Expr):
            out_id = op_rms_norm(x.graph, x.id, self.gamma, self.eps, self.name)
            return Expr(x.graph, out_id)
        return self.eval(np.asarray(x, dtype=np.float64))


__all__ = [
    "NNActivation",
    "LayerConfig",
    "MLP",
    "ResNetMLP",
    "NeuralFoilSurrogate",
    "Conv1D",
    "Conv2D",
    "AvgPool2D",
    "MaxPool2D",
    "GlobalAvgPool2D",
    "BatchNorm",
    "RMSNorm",
    "MultiHeadAttention",
    "TransformerBlock",
    "Sequential",
    "ONNXModel",
    "from_onnx",
    "resolve_activation",
]
