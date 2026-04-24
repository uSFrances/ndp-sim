from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from .layout import LayoutSpec


LayoutBuilder = Callable[[str], LayoutSpec]
ShapeResolver = Callable[[Dict[str, Dict[str, str]]], Dict[str, str]]


@dataclass(frozen=True)
class PortTemplate:
    memory_dtype: str
    layout_builder: LayoutBuilder

    def build(self) -> Dict[str, object]:
        return {"layout": self.layout_builder(self.memory_dtype)}


@dataclass(frozen=True)
class RegisteredOp:
    name: str
    input_ports: Dict[str, PortTemplate]
    output_ports: Dict[str, PortTemplate]
    ordered_inputs: Sequence[str]
    ordered_outputs: Sequence[str]
    shape_resolver: ShapeResolver

    def infer_output_tensor(self, output_name: str, input_shapes: Dict[str, Dict[str, str]]) -> Dict[str, object]:
        port = self.output_ports[output_name]
        return {
            "dtype": port.memory_dtype,
            "shape": dict(self.shape_resolver(input_shapes)),
        }


def make_layout(
    dtype: str,
    logical_shape: Dict[str, str],
    factors: List[Dict[str, object]],
    linear_order: List[str],
) -> LayoutSpec:
    return LayoutSpec.from_dict(
        {
            "dtype": dtype,
            "logical_shape": logical_shape,
            "factors": factors,
            "linear_order": linear_order,
        }
    )


def rowmajor_fp16_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "K": "K"},
        [
            {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
            {"name": "K", "parent_axis": "K", "extent": "K", "kind": "outer"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["M_outer8", "K", "m8"],
    )


def rowmajor_mn_fp16_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "N": "N"},
        [
            {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
            {"name": "N", "parent_axis": "N", "extent": "N", "kind": "outer"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["M_outer8", "N", "m8"],
    )


def v_add_rowmajor_mn_fp16_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "N": "N"},
        [
            {"name": "M", "parent_axis": "M", "extent": "M", "kind": "outer"},
            {"name": "N", "parent_axis": "N", "extent": "N", "kind": "outer"},
        ],
        ["M", "N"],
    )




def qkt_kt_view_fp16_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"K": "K", "N": "N"},
        [
            {"name": "N_outer8", "parent_axis": "N", "extent": "N//8", "kind": "outer"},
            {"name": "K", "parent_axis": "K", "extent": "K", "kind": "outer"},
            {"name": "n8", "parent_axis": "N", "extent": 8, "kind": "tile"},
        ],
        ["N_outer8", "K", "n8"],
    )


def sv_v_view_fp16_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"K": "K", "N": "N"},
        [
            {"name": "K_outer8", "parent_axis": "K", "extent": "K//8", "kind": "outer"},
            {"name": "N", "parent_axis": "N", "extent": "N", "kind": "outer"},
            {"name": "k8", "parent_axis": "K", "extent": 8, "kind": "tile"},
        ],
        ["K_outer8", "N", "k8"],
    )


def rowmajor_fp32_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "K": "K"},
        [
            {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
            {"name": "K", "parent_axis": "K", "extent": "K", "kind": "outer"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["M_outer8", "K", "m8"],
    )


def rowmajor_mn_fp32_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "N": "N"},
        [
            {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
            {"name": "N", "parent_axis": "N", "extent": "N", "kind": "outer"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["M_outer8", "N", "m8"],
    )


def vector_m_fp32_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M"},
        [
            {"name": "M", "parent_axis": "M", "extent": "M", "kind": "outer"},
        ],
        ["M"],
    )


def vector_n_fp32_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"N": "N"},
        [
            {"name": "N", "parent_axis": "N", "extent": "N", "kind": "outer"},
        ],
        ["N"],
    )


def ring_gemm_a_fp16_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "K": "K"},
        [
            {"name": "M_outer32", "parent_axis": "M", "extent": "M//32", "kind": "outer"},
            {"name": "K_outer2", "parent_axis": "K", "extent": "K//2", "kind": "outer"},
            {"name": "m4", "parent_axis": "M", "extent": 4, "kind": "tile"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
            {"name": "k2", "parent_axis": "K", "extent": 2, "kind": "tile"},
        ],
        ["M_outer32", "K_outer2", "m4", "m8", "k2"],
    )


def ring_gemm_b_fp16_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"K": "K", "N": "N"},
        [
            {"name": "N_outer32", "parent_axis": "N", "extent": "N//32", "kind": "outer"},
            {"name": "K_outer2", "parent_axis": "K", "extent": "K//2", "kind": "outer"},
            {"name": "n4", "parent_axis": "N", "extent": 4, "kind": "tile"},
            {"name": "n8", "parent_axis": "N", "extent": 8, "kind": "tile"},
            {"name": "k2", "parent_axis": "K", "extent": 2, "kind": "tile"},
        ],
        ["N_outer32", "K_outer2", "n4", "n8", "k2"],
    )


def ring_gemm_out_fp16_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "N": "N"},
        [
            {"name": "M_outer32", "parent_axis": "M", "extent": "M//32", "kind": "outer"},
            {"name": "N_outer32", "parent_axis": "N", "extent": "N//32", "kind": "outer"},
            {"name": "n4", "parent_axis": "N", "extent": 4, "kind": "tile"},
            {"name": "m4", "parent_axis": "M", "extent": 4, "kind": "tile"},
            {"name": "n8", "parent_axis": "N", "extent": 8, "kind": "tile"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["M_outer32", "N_outer32", "n4", "m4", "n8", "m8"],
    )


def bias_in_fp16_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "N": "N"},
        [
            {"name": "N_outer4", "parent_axis": "N", "extent": "N//4", "kind": "outer"},
            {"name": "n4", "parent_axis": "N", "extent": 4, "kind": "tile"},
            {"name": "M_outer64", "parent_axis": "M", "extent": "M//64", "kind": "outer"},
            {"name": "m8_a", "parent_axis": "M", "extent": 8, "kind": "tile"},
            {"name": "m8_b", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["N_outer4", "n4", "M_outer64", "m8_a", "m8_b"],
    )


def bias_out_fp32_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "N": "N"},
        [
            {"name": "N_outer4", "parent_axis": "N", "extent": "N//4", "kind": "outer"},
            {"name": "n4", "parent_axis": "N", "extent": 4, "kind": "tile"},
            {"name": "M_outer32", "parent_axis": "M", "extent": "M//32", "kind": "outer"},
            {"name": "m4", "parent_axis": "M", "extent": 4, "kind": "tile"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["N_outer4", "n4", "M_outer32", "m4", "m8"],
    )


def prefill_mul_mn_n_in_fp32_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "N": "N"},
        [
            {"name": "N_outer4", "parent_axis": "N", "extent": "N//4", "kind": "outer"},
            {"name": "n4", "parent_axis": "N", "extent": 4, "kind": "tile"},
            {"name": "M_outer32", "parent_axis": "M", "extent": "M//32", "kind": "outer"},
            {"name": "m4", "parent_axis": "M", "extent": 4, "kind": "tile"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["N_outer4", "n4", "M_outer32", "m4", "m8"],
    )


def prefill_mul_mn_n_out_fp16_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "N": "N"},
        [
            {"name": "N_outer4", "parent_axis": "N", "extent": "N//4", "kind": "outer"},
            {"name": "n4", "parent_axis": "N", "extent": 4, "kind": "tile"},
            {"name": "M_outer64", "parent_axis": "M", "extent": "M//64", "kind": "outer"},
            {"name": "m8_a", "parent_axis": "M", "extent": 8, "kind": "tile"},
            {"name": "m8_b", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["N_outer4", "n4", "M_outer64", "m8_a", "m8_b"],
    )


def prefill_mn_fp32_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "N": "N"},
        [
            {"name": "M_outer32", "parent_axis": "M", "extent": "M//32", "kind": "outer"},
            {"name": "N_outer16", "parent_axis": "N", "extent": "N//16", "kind": "outer"},
            {"name": "m4", "parent_axis": "M", "extent": 4, "kind": "tile"},
            {"name": "n16", "parent_axis": "N", "extent": 16, "kind": "tile"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["M_outer32", "N_outer16", "m4", "n16", "m8"],
    )


def prefill_mn_fp16_layout(dtype: str) -> LayoutSpec:
    return prefill_mn_fp32_layout(dtype)


def elementwise_mn_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "N": "N"},
        [
            {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
            {"name": "N", "parent_axis": "N", "extent": "N", "kind": "outer"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["M_outer8", "N", "m8"],
    )


def elementwise_m_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M"},
        [
            {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["M_outer8", "m8"],
    )


def prefill_silu_in_fp16_layout(dtype: str) -> LayoutSpec:
    return elementwise_mn_layout(dtype)


def prefill_silu_out_fp32_layout(dtype: str) -> LayoutSpec:
    return prefill_silu_in_fp16_layout(dtype)


def reduction_mn_fp32_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M", "N": "N"},
        [
            {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
            {"name": "N", "parent_axis": "N", "extent": "N", "kind": "outer"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["M_outer8", "N", "m8"],
    )


def reduction_mn_fp16_layout(dtype: str) -> LayoutSpec:
    return reduction_mn_fp32_layout(dtype)


def reduction_out_m_fp32_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"M": "M"},
        [
            {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
            {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
        ],
        ["M_outer8", "m8"],
    )


def reduction_out_c_fp32_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"C": "C"},
        [
            {"name": "C_outer8", "parent_axis": "C", "extent": "C//8", "kind": "outer"},
            {"name": "c8", "parent_axis": "C", "extent": 8, "kind": "tile"},
        ],
        ["C_outer8", "c8"],
    )


def reduction_in_cwh_fp32_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"C": "C", "H": "H", "W": "W"},
        [
            {"name": "C_outer8", "parent_axis": "C", "extent": "C//8", "kind": "outer"},
            {"name": "H", "parent_axis": "H", "extent": "H", "kind": "outer"},
            {"name": "W", "parent_axis": "W", "extent": "W", "kind": "outer"},
            {"name": "c8", "parent_axis": "C", "extent": 8, "kind": "tile"},
        ],
        ["C_outer8", "H", "W", "c8"],
    )


def reduction_in_cwh_uint8_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"C": "C", "H": "H", "W": "W"},
        [
            {"name": "C_outer4", "parent_axis": "C", "extent": "C//4", "kind": "outer"},
            {"name": "H", "parent_axis": "H", "extent": "H", "kind": "outer"},
            {"name": "W", "parent_axis": "W", "extent": "W", "kind": "outer"},
            {"name": "c4", "parent_axis": "C", "extent": 4, "kind": "tile"},
        ],
        ["C_outer4", "H", "W", "c4"],
    )


def reduction_out_c_uint8_layout(dtype: str) -> LayoutSpec:
    return make_layout(
        dtype,
        {"C": "C"},
        [
            {"name": "C_outer4", "parent_axis": "C", "extent": "C//4", "kind": "outer"},
            {"name": "c4", "parent_axis": "C", "extent": 4, "kind": "tile"},
        ],
        ["C_outer4", "c4"],
    )


def _port(memory_dtype: str, layout_builder: LayoutBuilder) -> PortTemplate:
    return PortTemplate(memory_dtype=memory_dtype, layout_builder=layout_builder)


def _same_shape_resolver(input_shapes: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    ordered = list(input_shapes.values())
    if not ordered:
        return {"M": "M", "N": "N"}
    first = ordered[0]
    for shape in ordered[1:]:
        if shape != first:
            raise ValueError(f"Expected identical input shapes, got {input_shapes}.")
    return dict(first)


def _mn_n_resolver(input_shapes: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    in_a = input_shapes["inA"]
    in_b = input_shapes["inB"]
    if set(in_a.keys()) != {"M", "N"}:
        raise ValueError(f"Expected inA to have axes M,N, got {in_a}.")
    if set(in_b.keys()) != {"N"}:
        raise ValueError(f"Expected inB to have axis N, got {in_b}.")
    return dict(in_a)


def _mn_m_resolver(input_shapes: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    in_a = input_shapes["inA"]
    in_b = input_shapes["inB"]
    if set(in_a.keys()) != {"M", "N"}:
        raise ValueError(f"Expected inA to have axes M,N, got {in_a}.")
    if set(in_b.keys()) != {"M"}:
        raise ValueError(f"Expected inB to have axis M, got {in_b}.")
    return dict(in_a)


def _mn_reduce_to_m(input_shapes: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    in_a = input_shapes["inA"]
    if set(in_a.keys()) != {"M", "N"}:
        raise ValueError(f"Expected inA to have axes M,N, got {in_a}.")
    return {"M": "M"}


def _gemm_mk_kn_to_mn_shape(input_shapes: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    in_a = input_shapes["inA"]
    in_b = input_shapes["inB"]
    if len(in_a) != 2:
        raise ValueError(f"Expected ring_gemm inA to have rank 2, got {in_a}.")
    if len(in_b) != 2:
        raise ValueError(f"Expected ring_gemm inB to have rank 2, got {in_b}.")
    a_axes = list(in_a.keys())
    b_axes = list(in_b.keys())
    return {
        a_axes[0]: in_a[a_axes[0]],
        b_axes[1]: in_b[b_axes[1]],
    }


def _qkt_kt_view_resolver(input_shapes: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    in_a = input_shapes["inA"]
    if set(in_a.keys()) != {"M", "N"}:
        raise ValueError(f"Expected inA to have axes M,N, got {in_a}.")
    return {
        "K": in_a["N"],
        "N": in_a["M"],
    }


def _sv_v_view_resolver(input_shapes: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    in_a = input_shapes["inA"]
    if set(in_a.keys()) != {"M", "N"}:
        raise ValueError(f"Expected inA to have axes M,N, got {in_a}.")
    return {
        "K": in_a["M"],
        "N": in_a["N"],
    }


def _single_input_same_resolver(expected_axes: Sequence[str]) -> ShapeResolver:
    expected = set(expected_axes)

    def resolver(input_shapes: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        in_a = input_shapes["inA"]
        if set(in_a.keys()) != expected:
            raise ValueError(f"Expected inA axes {sorted(expected)}, got {in_a}.")
        return dict(in_a)

    return resolver


def _reduce_resolver(input_axes: Sequence[str], output_axes: Sequence[str]) -> ShapeResolver:
    expected_in = set(input_axes)

    def resolver(input_shapes: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        in_a = input_shapes["inA"]
        if set(in_a.keys()) != expected_in:
            raise ValueError(f"Expected inA axes {sorted(expected_in)}, got {in_a}.")
        return {axis: axis for axis in output_axes}

    return resolver


def _register(
    registry: Dict[str, RegisteredOp],
    name: str,
    input_ports: Dict[str, PortTemplate],
    output_ports: Dict[str, PortTemplate],
    ordered_inputs: Sequence[str],
    ordered_outputs: Sequence[str],
    shape_resolver: ShapeResolver,
    aliases: Sequence[str] = (),
) -> None:
    op = RegisteredOp(
        name=name,
        input_ports=input_ports,
        output_ports=output_ports,
        ordered_inputs=ordered_inputs,
        ordered_outputs=ordered_outputs,
        shape_resolver=shape_resolver,
    )
    registry[name] = op
    for alias in aliases:
        registry[alias] = op


def build_default_registry() -> Dict[str, RegisteredOp]:
    registry: Dict[str, RegisteredOp] = {}

    _register(
        registry,
        "prefill_remote_sum_fp32_fp32_fp32",
        {"inA": _port("fp32", vector_m_fp32_layout), "inB": _port("fp32", vector_m_fp32_layout)},
        {"out": _port("fp32", vector_m_fp32_layout)},
        ("inA", "inB"),
        ("out",),
        _same_shape_resolver,
        aliases=("prefill_remote_sum__fp32_fp32_fp32", "remote_sum__fp32_fp32_fp32"),
    )
    _register(
        registry,
        "prefill_mul_MN_N_fp32_fp32_fp16",
        {"inA": _port("fp32", prefill_mul_mn_n_in_fp32_layout), "inB": _port("fp32", vector_n_fp32_layout)},
        {"out": _port("fp16", prefill_mul_mn_n_out_fp16_layout)},
        ("inA", "inB"),
        ("out",),
        _mn_n_resolver,
        aliases=("prefill_mul_MN_N__fp32_fp32_fp16", "mul_MN_N__fp32_fp32_fp16"),
    )
    _register(
        registry,
        "gemm_local_fp16_fp16_fp16",
        {"inA": _port("fp16", ring_gemm_a_fp16_layout), "inB": _port("fp16", ring_gemm_b_fp16_layout)},
        {"out": _port("fp16", ring_gemm_out_fp16_layout)},
        ("inA", "inB"),
        ("out",),
        _gemm_mk_kn_to_mn_shape,
        aliases=("gemm_local__fp16_fp16_fp16", "gemm_local"),
    )
    _register(
        registry,
        "ring_gemm_fp16_fp16_fp16",
        {"inA": _port("fp16", ring_gemm_a_fp16_layout), "inB": _port("fp16", ring_gemm_b_fp16_layout)},
        {"out": _port("fp16", ring_gemm_out_fp16_layout)},
        ("inA", "inB"),
        ("out",),
        _gemm_mk_kn_to_mn_shape,
        aliases=("ring_gemm__fp16_fp16_fp16", "ring_gemm"),
    )
    _register(
        registry,
        "prefill_qkt_kt_view_fp16_fp16",
        {"inA": _port("fp16", rowmajor_fp16_layout)},
        {"out": _port("fp16", qkt_kt_view_fp16_layout)},
        ("inA",),
        ("out",),
        _qkt_kt_view_resolver,
        aliases=("prefill_qkt_kt_view__fp16_fp16", "qkt_kt_view"),
    )
    _register(
        registry,
        "prefill_sv_v_view_fp16_fp16",
        {"inA": _port("fp16", bias_in_fp16_layout)},
        {"out": _port("fp16", sv_v_view_fp16_layout)},
        ("inA",),
        ("out",),
        _sv_v_view_resolver,
        aliases=("prefill_sv_v_view__fp16_fp16", "sv_v_view"),
    )
    _register(
        registry,
        "prefill_add_MN_N_fp16_fp32_fp32",
        {"inA": _port("fp16", bias_in_fp16_layout), "inB": _port("fp32", vector_n_fp32_layout)},
        {"out": _port("fp32", bias_out_fp32_layout)},
        ("inA", "inB"),
        ("out",),
        _mn_n_resolver,
        aliases=("prefill_add_MN_N__fp16_fp32_fp32", "add_MN_N__fp16_fp32_fp32", "bias"),
    )
    _register(
        registry,
        "prefill_add_MN_N_fp16_fp32_fp16",
        {"inA": _port("fp16", bias_in_fp16_layout), "inB": _port("fp32", vector_n_fp32_layout)},
        {"out": _port("fp16", bias_in_fp16_layout)},
        ("inA", "inB"),
        ("out",),
        _mn_n_resolver,
        aliases=("prefill_add_MN_N__fp16_fp32_fp16", "add_MN_N__fp16_fp32_fp16"),
    )
    _register(
        registry,
        "prefill_add_V_MN_N_fp16_fp32_fp16",
        {"inA": _port("fp16", v_add_rowmajor_mn_fp16_layout), "inB": _port("fp32", vector_n_fp32_layout)},
        {"out": _port("fp16", v_add_rowmajor_mn_fp16_layout)},
        ("inA", "inB"),
        ("out",),
        _mn_n_resolver,
        aliases=("prefill_add_V_MN_N__fp16_fp32_fp16", "add_V_MN_N__fp16_fp32_fp16"),
    )
    _register(
        registry,
        "prefill_mul_fp32MN_fp32MN_fp32MN",
        {"inA": _port("fp32", elementwise_mn_layout), "inB": _port("fp32", elementwise_mn_layout)},
        {"out": _port("fp32", elementwise_mn_layout)},
        ("inA", "inB"),
        ("out",),
        _same_shape_resolver,
    )
    _register(
        registry,
        "prefill_add_fp32MN_fp32MN_fp32MN",
        {"inA": _port("fp32", elementwise_mn_layout), "inB": _port("fp32", elementwise_mn_layout)},
        {"out": _port("fp32", elementwise_mn_layout)},
        ("inA", "inB"),
        ("out",),
        _same_shape_resolver,
    )
    _register(
        registry,
        "prefill_sub_SFU_fp32MN_fp32MN_fp32MN",
        {"inA": _port("fp32", elementwise_mn_layout), "inB": _port("fp32", vector_m_fp32_layout)},
        {"out": _port("fp32", elementwise_mn_layout)},
        ("inA", "inB"),
        ("out",),
        _mn_m_resolver,
    )
    _register(
        registry,
        "prefill_sum_SFU_fp32_fp32",
        {"inA": _port("fp32", rowmajor_fp32_layout)},
        {"out": _port("fp32", vector_m_fp32_layout)},
        ("inA",),
        ("out",),
        _mn_reduce_to_m,
        aliases=("prefill_sum_SFU__fp32_fp32", "sum_SFU__fp32_fp32"),
    )
    _register(
        registry,
        "prefill_mul_MN_MN_fp16_fp32_fp16",
        {"inA": _port("fp16", rowmajor_fp16_layout), "inB": _port("fp32", rowmajor_fp32_layout)},
        {"out": _port("fp16", rowmajor_fp16_layout)},
        ("inA", "inB"),
        ("out",),
        _same_shape_resolver,
        aliases=("prefill_mul_MN_MN__fp16_fp32_fp16", "mul_MN_MN__fp16_fp32_fp16"),
    )

    _register(
        registry,
        "prefill_mul_fp32MN_fp32M_fp32MN",
        {"inA": _port("fp32", elementwise_mn_layout), "inB": _port("fp32", elementwise_m_layout)},
        {"out": _port("fp32", elementwise_mn_layout)},
        ("inA", "inB"),
        ("out",),
        _mn_m_resolver,
    )
    _register(
        registry,
        "prefill_mul_fp32MN_fp32M_fp16MN",
        {"inA": _port("fp32", elementwise_mn_layout), "inB": _port("fp32", elementwise_m_layout)},
        {"out": _port("fp16", elementwise_mn_layout)},
        ("inA", "inB"),
        ("out",),
        _mn_m_resolver,
    )
    _register(
        registry,
        "prefill_add_fp32MN_fp16MN_fp32MN",
        {"inA": _port("fp32", elementwise_mn_layout), "inB": _port("fp16", elementwise_mn_layout)},
        {"out": _port("fp32", elementwise_mn_layout)},
        ("inA", "inB"),
        ("out",),
        _same_shape_resolver,
    )
    _register(
        registry,
        "prefill_add_fp32MN_fp32MN_fp16MN",
        {"inA": _port("fp32", elementwise_mn_layout), "inB": _port("fp32", elementwise_mn_layout)},
        {"out": _port("fp16", elementwise_mn_layout)},
        ("inA", "inB"),
        ("out",),
        _same_shape_resolver,
    )
    _register(
        registry,
        "prefill_mul_fp32MN_fp16MN_fp16MN",
        {"inA": _port("fp32", elementwise_mn_layout), "inB": _port("fp16", elementwise_mn_layout)},
        {"out": _port("fp16", elementwise_mn_layout)},
        ("inA", "inB"),
        ("out",),
        _same_shape_resolver,
    )
    _register(
        registry,
        "prefill_silu_fp16MN_fp32MN",
        {"inA": _port("fp16", prefill_silu_in_fp16_layout)},
        {"out": _port("fp32", prefill_silu_out_fp32_layout)},
        ("inA",),
        ("out",),
        _single_input_same_resolver(("M", "N")),
    )
    _register(
        registry,
        "prefill_sum_rec",
        {"inA": _port("fp32", reduction_mn_fp32_layout)},
        {"out": _port("fp32", reduction_out_m_fp32_layout)},
        ("inA",),
        ("out",),
        _reduce_resolver(("M", "N"), ("M",)),
    )
    _register(
        registry,
        "prefill_summac",
        {"inA": _port("fp32", reduction_mn_fp32_layout)},
        {"out": _port("fp32", reduction_out_m_fp32_layout)},
        ("inA",),
        ("out",),
        _reduce_resolver(("M", "N"), ("M",)),
    )
    _register(
        registry,
        "prefill_mac_SFU",
        {"inA": _port("fp32", reduction_out_m_fp32_layout)},
        {"out": _port("fp32", reduction_out_m_fp32_layout)},
        ("inA",),
        ("out",),
        _single_input_same_resolver(("M",)),
    )
    _register(
        registry,
        "prefill_max",
        {"inA": _port("fp32", reduction_mn_fp32_layout)},
        {"out": _port("fp32", reduction_out_m_fp32_layout)},
        ("inA",),
        ("out",),
        _reduce_resolver(("M", "N"), ("M",)),
    )
    _register(
        registry,
        "prefill_remote_sum_Mfp32_Mfp32",
        {"inA": _port("fp32", vector_m_fp32_layout)},
        {"out": _port("fp32", vector_m_fp32_layout)},
        ("inA",),
        ("out",),
        _single_input_same_resolver(("M",)),
    )
    _register(
        registry,
        "prefill_remote_sum_fp16MN_fp32MN",
        {"inA": _port("fp16", reduction_mn_fp16_layout)},
        {"out": _port("fp32", reduction_mn_fp32_layout)},
        ("inA",),
        ("out",),
        _single_input_same_resolver(("M", "N")),
    )
    _register(
        registry,
        "avgpool_fp32_fp32",
        {"inA": _port("fp32", reduction_in_cwh_fp32_layout)},
        {"out": _port("fp32", reduction_out_c_fp32_layout)},
        ("inA",),
        ("out",),
        _reduce_resolver(("C", "H", "W"), ("C",)),
        aliases=("avgpool__fp32_fp32", "avgpool"),
    )
    _register(
        registry,
        "maxpool_uint8_uint8",
        {"inA": _port("uint8", reduction_in_cwh_uint8_layout)},
        {"out": _port("uint8", reduction_out_c_uint8_layout)},
        ("inA",),
        ("out",),
        _reduce_resolver(("C", "H", "W"), ("C",)),
        aliases=("maxpool__uint8_uint8", "maxpool"),
    )

    return registry
