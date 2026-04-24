import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from address_remapping.addressing import AddressTransform, encode_physical_address
from address_remapping.graph import load_graph_file, solve_graph
from address_remapping.hardware import HardwareSpec
from address_remapping.layout import LayoutSpec
from address_remapping.model_parser import expand_model_spec, parse_call
from address_remapping.performance import (
    MODE_BASELINE,
    MODE_REMAP,
    MODE_REMAP_INTERLEAVE,
    PerformanceConfig,
    PhysicalRequest,
    _estimate_compute,
    analyze_graph_performance,
    load_performance_config,
    _analyze_request_stream,
    write_performance_outputs,
)
from address_remapping.registry import build_default_registry
from address_remapping.roofline import build_roofline_summary
from address_remapping.rmsnorm_bridge import (
    RmsNormBridgeError,
    build_expanded_graph_from_external_rmsnorm,
    fill_external_rmsnorm_remapping,
    normalize_graph_spec,
)
from address_remapping.solver import EdgeSolveResult, solve_edge
from address_remapping.validation import (
    _detect_ramulator_executable,
    _parse_reference_cycles,
    emit_trace_artifacts,
    run_validation,
)


def make_layout(dtype, logical_shape, factors, linear_order):
    return LayoutSpec.from_dict(
        {
            "dtype": dtype,
            "logical_shape": logical_shape,
            "factors": factors,
            "linear_order": linear_order,
        }
    )


def make_physical_request(hw, request_id, edge_name, ag_id, role, slice_id, bank_id, row_id, col_id):
    return PhysicalRequest(
        request_id=request_id,
        tensor_name="x",
        edge_name=edge_name,
        ag_id=ag_id,
        role=role,
        logical_addr=request_id,
        base_addr=0,
        address_transform=AddressTransform.identity(["addr_bit_0"], name="test_identity").to_dict(),
        physical_addr=encode_physical_address(
            slice_id=slice_id,
            bank_id=bank_id,
            row_id=row_id,
            col_id=col_id,
            hw=hw,
        ),
        slice_id=slice_id,
        bank_id=bank_id,
        row_id=row_id,
        col_id=col_id,
    )


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.hw = HardwareSpec()

    def test_parse_call_with_kwargs(self):
        lhs, op_name, args, kwargs = parse_call(
            "gemm_out_fp16 = ring_gemm_fp16_fp16_fp16(a_fp16, b_fp16, ring_scope=cluster)"
        )
        self.assertEqual(lhs, "gemm_out_fp16")
        self.assertEqual(op_name, "ring_gemm_fp16_fp16_fp16")
        self.assertEqual(args, ["a_fp16", "b_fp16"])
        self.assertEqual(kwargs, {"ring_scope": "cluster"})

    def test_registry_contains_documented_ops(self):
        registry = build_default_registry()
        self.assertIn("prefill_add_MN_N_fp16_fp32_fp16", registry)
        self.assertIn("prefill_add_V_MN_N_fp16_fp32_fp16", registry)
        self.assertIn("prefill_qkt_kt_view_fp16_fp16", registry)
        self.assertIn("prefill_mul_fp32MN_fp32M_fp32MN", registry)
        self.assertIn("prefill_mul_fp32MN_fp32M_fp16MN", registry)
        self.assertIn("prefill_mul_fp32MN_fp32MN_fp32MN", registry)
        self.assertIn("prefill_add_fp32MN_fp32MN_fp32MN", registry)
        self.assertIn("prefill_silu_fp16MN_fp32MN", registry)
        self.assertIn("prefill_sub_SFU_fp32MN_fp32MN_fp32MN", registry)
        self.assertIn("prefill_summac", registry)
        self.assertIn("prefill_sum_rec", registry)
        self.assertIn("prefill_sum_rec", registry)
        self.assertIn("prefill_remote_sum_fp16MN_fp32MN", registry)
        self.assertIn("prefill_remote_sum_Mfp32_Mfp32", registry)
        self.assertIn("avgpool_fp32_fp32", registry)
        self.assertIn("maxpool_uint8_uint8", registry)
        self.assertNotIn("quant_from_buffer_int32MN_uint8MN", registry)
        self.assertNotIn("add_dequant_uint8CWH_uint8CWH_fp32CWH", registry)
        self.assertNotIn("prefill_sum_mac_fp32_fp32", registry)
        self.assertNotIn("prefill_sum_rec_fp32_fp32", registry)
        self.assertNotIn("prefill_remote_sum_fp32_fp32", registry)
        self.assertNotIn("prefill_remote_sum_fp16_fp32", registry)
        self.assertNotIn("prefill_mul_MN_M_fp32_fp32_fp32", registry)
        self.assertNotIn("prefill_mul_MN_M_fp32_fp32_fp16", registry)
        self.assertNotIn("prefill_mul_MN_MN_fp32_fp32_fp32", registry)
        self.assertNotIn("prefill_add_MN_MN_fp32_fp32_fp32", registry)
        self.assertNotIn("prefill_add_MN_MN_fp32_fp16_fp32", registry)
        self.assertNotIn("prefill_add_MN_MN_fp32_fp32_fp16", registry)
        self.assertNotIn("prefill_silu_fp16_fp32", registry)

    def test_expand_model_spec_builds_local_views(self):
        spec = {
            "shape_bindings": {"sequence_length": 128, "hidden_size": 896},
            "params": {"slices_per_cluster": 4, "slices_num": 28},
            "tensors": {
                "a_fp16": {
                    "dtype": "fp16",
                    "shape": {"M": "sequence_length", "K": "hidden_size"},
                    "partition": {"K": {"by_scope": {"cluster": "slices_per_cluster", "global": "slices_num"}}},
                },
                "b_fp16": {
                    "dtype": "fp16",
                    "shape": {"K": "hidden_size", "N": "hidden_size"},
                    "partition": {"N": {"by_scope": {"cluster": "slices_num", "global": "slices_num"}}},
                },
                "bias_vec_fp32": {
                    "dtype": "fp32",
                    "shape": {"N": "hidden_size"},
                    "partition": {"N": {"follow": "inA:N"}},
                },
            },
            "model": [
                "gemm_out_fp16 = ring_gemm_fp16_fp16_fp16(a_fp16, b_fp16, ring_scope=cluster)",
                "bias_out_fp32 = prefill_add_MN_N_fp16_fp32_fp32(gemm_out_fp16, bias_vec_fp32)",
            ],
        }
        expanded = expand_model_spec(spec)
        ring_inputs = expanded["ops"]["ring_gemm_fp16_fp16_fp16_0"]["inputs"]
        self.assertEqual(ring_inputs["inA"]["resolved_shape"], {"M": 128, "K": 224})
        self.assertEqual(ring_inputs["inB"]["resolved_shape"], {"K": 896, "N": 32})
        self.assertEqual(expanded["tensors"]["gemm_out_fp16"]["resolved_shape"], {"M": 128, "N": 32})
        self.assertEqual(expanded["ops"]["prefill_add_MN_N_fp16_fp32_fp32_1"]["inputs"]["inB"]["resolved_shape"], {"N": 32})

    def test_expand_model_spec_global_scope(self):
        spec = {
            "shape_bindings": {"sequence_length": 128, "hidden_size": 896},
            "params": {"slices_per_cluster": 4, "slices_num": 28},
            "tensors": {
                "a_fp16": {
                    "dtype": "fp16",
                    "shape": {"M": "sequence_length", "K": "hidden_size"},
                    "partition": {"K": {"by_scope": {"cluster": "slices_per_cluster", "global": "slices_num"}}},
                },
                "b_fp16": {
                    "dtype": "fp16",
                    "shape": {"K": "hidden_size", "N": "hidden_size"},
                    "partition": {"N": {"by_scope": {"cluster": "slices_num", "global": "slices_num"}}},
                },
            },
            "model": [
                "gemm_out_fp16 = ring_gemm_fp16_fp16_fp16(a_fp16, b_fp16, ring_scope=global)",
            ],
        }
        expanded = expand_model_spec(spec)
        ring_inputs = expanded["ops"]["ring_gemm_fp16_fp16_fp16_0"]["inputs"]
        self.assertEqual(ring_inputs["inA"]["resolved_shape"], {"M": 128, "K": 32})
        self.assertEqual(ring_inputs["inB"]["resolved_shape"], {"K": 896, "N": 32})
        self.assertEqual(expanded["tensors"]["gemm_out_fp16"]["resolved_shape"], {"M": 128, "N": 32})

    def test_expand_prefill_elementwise_and_reduction_ops(self):
        spec = {
            "shape_bindings": {"M": 128, "N": 128},
            "tensors": {
                "x_fp32": {"dtype": "fp32", "shape": {"M": "M", "N": "N"}},
                "y_fp32": {"dtype": "fp32", "shape": {"M": "M", "N": "N"}},
                "vec_fp32": {"dtype": "fp32", "shape": {"M": "M"}},
            },
            "model": [
                "mul_out_fp32 = prefill_mul_fp32MN_fp32M_fp32MN(x_fp32, vec_fp32)",
                "mul_out_fp16 = prefill_mul_fp32MN_fp32M_fp16MN(x_fp32, vec_fp32)",
                "sum_out_fp32 = prefill_sum_rec(x_fp32)"
            ],
        }
        expanded = expand_model_spec(spec)
        self.assertEqual(expanded["tensors"]["mul_out_fp32"]["shape"], {"M": "M", "N": "N"})
        self.assertEqual(expanded["tensors"]["mul_out_fp16"]["shape"], {"M": "M", "N": "N"})
        self.assertEqual(expanded["tensors"]["sum_out_fp32"]["shape"], {"M": "M"})

    def test_expand_prefill_fp32mn_fp32mn_fp32mn_ops(self):
        spec = {
            "shape_bindings": {"M": 128, "N": 128},
            "tensors": {
                "x_fp32": {"dtype": "fp32", "shape": {"M": "M", "N": "N"}},
                "y_fp32": {"dtype": "fp32", "shape": {"M": "M", "N": "N"}},
            },
            "model": [
                "mul_out_fp32 = prefill_mul_fp32MN_fp32MN_fp32MN(x_fp32, y_fp32)",
                "add_out_fp32 = prefill_add_fp32MN_fp32MN_fp32MN(x_fp32, y_fp32)",
            ],
        }
        expanded = expand_model_spec(spec)
        self.assertEqual(expanded["tensors"]["mul_out_fp32"]["dtype"], "fp32")
        self.assertEqual(expanded["tensors"]["mul_out_fp32"]["shape"], {"M": "M", "N": "N"})
        self.assertEqual(expanded["tensors"]["add_out_fp32"]["dtype"], "fp32")
        self.assertEqual(expanded["tensors"]["add_out_fp32"]["shape"], {"M": "M", "N": "N"})

    def test_elementwise_layouts_match_updated_document(self):
        registry = build_default_registry()

        mn_layout = registry["prefill_add_fp32MN_fp16MN_fp32MN"].input_ports["inA"].build()["layout"]
        self.assertEqual(mn_layout.logical_shape, {"M": "M", "N": "N"})
        self.assertEqual(
            [
                {
                    "name": factor.name,
                    "parent_axis": factor.parent_axis,
                    "extent": factor.extent_expr,
                    "kind": factor.kind,
                }
                for factor in mn_layout.factors
            ],
            [
                {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
                {"name": "N", "parent_axis": "N", "extent": "N", "kind": "outer"},
                {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
            ],
        )
        self.assertEqual(list(mn_layout.linear_order), ["M_outer8", "N", "m8"])

        m_layout = registry["prefill_mul_fp32MN_fp32M_fp32MN"].input_ports["inB"].build()["layout"]
        self.assertEqual(m_layout.logical_shape, {"M": "M"})
        self.assertEqual(
            [
                {
                    "name": factor.name,
                    "parent_axis": factor.parent_axis,
                    "extent": factor.extent_expr,
                    "kind": factor.kind,
                }
                for factor in m_layout.factors
            ],
            [
                {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
                {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
            ],
        )
        self.assertEqual(list(m_layout.linear_order), ["M_outer8", "m8"])

        sub_sfu = registry["prefill_sub_SFU_fp32MN_fp32MN_fp32MN"]
        self.assertEqual(sub_sfu.input_ports["inA"].memory_dtype, "fp32")
        self.assertEqual(sub_sfu.input_ports["inB"].memory_dtype, "fp32")
        self.assertEqual(sub_sfu.output_ports["out"].memory_dtype, "fp32")

        add_fp16 = registry["prefill_add_MN_N_fp16_fp32_fp16"]
        self.assertEqual(add_fp16.input_ports["inA"].memory_dtype, "fp16")
        self.assertEqual(add_fp16.input_ports["inB"].memory_dtype, "fp32")
        self.assertEqual(add_fp16.output_ports["out"].memory_dtype, "fp16")
        self.assertEqual(
            add_fp16.output_ports["out"].build()["layout"],
            add_fp16.input_ports["inA"].build()["layout"],
        )
        add_v_fp16 = registry["prefill_add_V_MN_N_fp16_fp32_fp16"]
        self.assertEqual(add_v_fp16.input_ports["inA"].memory_dtype, "fp16")
        self.assertEqual(add_v_fp16.input_ports["inB"].memory_dtype, "fp32")
        self.assertEqual(add_v_fp16.output_ports["out"].memory_dtype, "fp16")
        self.assertEqual(
            add_v_fp16.output_ports["out"].build()["layout"],
            add_v_fp16.input_ports["inA"].build()["layout"],
        )

        silu = registry["prefill_silu_fp16MN_fp32MN"]
        self.assertEqual(silu.input_ports["inA"].memory_dtype, "fp16")
        self.assertEqual(silu.output_ports["out"].memory_dtype, "fp32")

        summac = registry["prefill_summac"]
        self.assertEqual(summac.input_ports["inA"].memory_dtype, "fp32")
        self.assertEqual(summac.output_ports["out"].memory_dtype, "fp32")

        remote_sum_fp32 = registry["prefill_remote_sum_Mfp32_Mfp32"]
        self.assertEqual(remote_sum_fp32.input_ports["inA"].memory_dtype, "fp32")
        self.assertEqual(remote_sum_fp32.output_ports["out"].memory_dtype, "fp32")
        self.assertEqual(
            remote_sum_fp32.input_ports["inA"].build()["layout"],
            remote_sum_fp32.output_ports["out"].build()["layout"],
        )

        remote_sum_fp16 = registry["prefill_remote_sum_fp16MN_fp32MN"]
        self.assertEqual(remote_sum_fp16.input_ports["inA"].memory_dtype, "fp16")
        self.assertEqual(remote_sum_fp16.output_ports["out"].memory_dtype, "fp32")

        sub_sfu = registry["prefill_sub_SFU_fp32MN_fp32MN_fp32MN"]
        self.assertEqual(sub_sfu.input_ports["inA"].memory_dtype, "fp32")
        self.assertEqual(sub_sfu.input_ports["inB"].memory_dtype, "fp32")
        self.assertEqual(sub_sfu.output_ports["out"].memory_dtype, "fp32")

        mul_mn_n = registry["prefill_mul_MN_N_fp32_fp32_fp16"]
        self.assertEqual(mul_mn_n.input_ports["inA"].memory_dtype, "fp32")
        self.assertEqual(mul_mn_n.input_ports["inB"].memory_dtype, "fp32")
        self.assertEqual(mul_mn_n.output_ports["out"].memory_dtype, "fp16")
        mul_mn_m = registry["prefill_mul_fp32MN_fp32M_fp32MN"]
        self.assertEqual(
            mul_mn_m.input_ports["inA"].build()["layout"].logical_shape,
            {"M": "M", "N": "N"},
        )
        self.assertEqual(
            mul_mn_m.output_ports["out"].build()["layout"].logical_shape,
            {"M": "M", "N": "N"},
        )
        in_a_layout = mul_mn_n.input_ports["inA"].build()["layout"]
        out_layout = mul_mn_n.output_ports["out"].build()["layout"]
        self.assertEqual(
            list(in_a_layout.logical_shape.keys()),
            ["M", "N"],
        )
        self.assertEqual(
            [factor.name for factor in in_a_layout.factors],
            ["N_outer4", "n4", "M_outer32", "m4", "m8"],
        )
        self.assertEqual(
            [factor.extent_expr for factor in in_a_layout.factors],
            ["N//4", 4, "M//32", 4, 8],
        )
        self.assertEqual(
            [factor.name for factor in out_layout.factors],
            ["N_outer4", "n4", "M_outer64", "m8_a", "m8_b"],
        )
        self.assertEqual(
            [factor.extent_expr for factor in out_layout.factors],
            ["N//4", 4, "M//64", 8, 8],
        )

    def test_gemm_example_produces_expected_swap(self):
        producer = make_layout(
            "fp32",
            {"M": "M", "K": "K"},
            [
                {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
                {"name": "K", "parent_axis": "K", "extent": "K", "kind": "outer"},
                {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
            ],
            ["M_outer8", "K", "m8"],
        )
        consumer = make_layout(
            "fp32",
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
        result = solve_edge(producer, consumer, {"M": 128, "K": 64}, memory_dtype="fp32", hw_cfg=self.hw)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.permutation[:9], [0, 6, 7, 1, 2, 3, 4, 5, 8])

    def test_flexible_refinement_supports_split_factorization(self):
        registry = build_default_registry()
        producer = registry["prefill_mul_fp32MN_fp32M_fp32MN"].output_ports["out"].build()["layout"]
        consumer = registry["prefill_mul_MN_N_fp32_fp32_fp16"].input_ports["inA"].build()["layout"]
        result = solve_edge(
            producer,
            consumer,
            {"M": 128, "N": 32},
            memory_dtype="fp32",
            hw_cfg=self.hw,
        )
        self.assertEqual(result.status, "ok")

    def test_write_reg_relayout_supports_gemm_input(self):
        registry = build_default_registry()
        producer = registry["prefill_mul_MN_N_fp32_fp32_fp16"].output_ports["out"].build()["layout"]
        consumer = registry["ring_gemm_fp16_fp16_fp16"].input_ports["inA"].build()["layout"]
        result = solve_edge(
            producer,
            consumer,
            {"M": 128, "N": 32, "K": 32},
            memory_dtype="fp16",
            hw_cfg=self.hw,
            producer_axis_aliases={"M": "M", "N": "N"},
            consumer_axis_aliases={"M": "M", "K": "N"},
        )
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.write_reg_required)
        self.assertIn("reorder(", result.write_reg_hint)
        self.assertIn("m8", result.write_reg_hint)
        self.assertIn("n2", result.write_reg_hint)
        self.assertNotEqual(result.permutation[:7], list(range(7)))
        self.assertEqual(result.shape_bindings, {"M": 128, "N": 32, "K": 32})
        self.assertEqual(result.producer_bound_layout["dtype"], "fp16")
        self.assertEqual(result.consumer_bound_layout["dtype"], "fp16")
        self.assertEqual(
            result.producer_bound_layout["ordered_factors"][0],
            {"name": "N_outer4", "parent_axis": "N", "extent": 8, "kind": "outer", "bits": 3},
        )
        self.assertEqual(
            result.consumer_bound_layout["ordered_factors"][0],
            {"name": "M_outer32", "parent_axis": "M", "extent": 4, "kind": "outer", "bits": 2},
        )
        self.assertEqual(result.consumer_axis_aliases, {"M": "M", "K": "N"})
        self.assertEqual(
            result.consumer_visible_outer_bits,
            [
                result.producer_visible_outer_bits[index]
                for index in result.permutation[: len(result.consumer_visible_outer_bits)]
            ],
        )
        self.assertEqual(
            result.producer_visible_outer_bits,
            ["M:bit4", "M:bit5", "M:bit6", "M:bit7", "N:bit1", "N:bit2", "N:bit3", "N:bit4", "N:bit5"],
        )
        self.assertEqual(
            result.consumer_visible_outer_bits,
            ["N:bit1", "M:bit4", "M:bit5", "N:bit2", "N:bit3", "N:bit4", "N:bit5", "M:bit6", "M:bit7"],
        )

    def test_ring_gemm_b_default_layout_uses_n8_then_k2(self):
        registry = build_default_registry()
        layout = registry["ring_gemm_fp16_fp16_fp16"].input_ports["inB"].build()["layout"]
        self.assertEqual(
            [factor.name for factor in layout.factors],
            ["N_outer32", "K_outer2", "n4", "n8", "k2"],
        )

    def test_write_reg_relayout_supports_gemm_b_input(self):
        registry = build_default_registry()
        producer = registry["prefill_qkt_kt_view_fp16_fp16"].output_ports["out"].build()["layout"]
        consumer = registry["ring_gemm_fp16_fp16_fp16"].input_ports["inB"].build()["layout"]
        result = solve_edge(
            producer,
            consumer,
            {"K": 32, "N": 128},
            memory_dtype="fp16",
            hw_cfg=self.hw,
        )
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.write_reg_required)
        self.assertEqual(result.write_reg_hint, "reorder(n8,k2)->(k2,n8)")
        self.assertEqual(
            result.consumer_bound_layout["ordered_factors"],
            [
                {"name": "N_outer32", "parent_axis": "N", "extent": 4, "kind": "outer", "bits": 2},
                {"name": "K_outer2", "parent_axis": "K", "extent": 16, "kind": "outer", "bits": 4},
                {"name": "n4", "parent_axis": "N", "extent": 4, "kind": "tile", "bits": 2},
                {"name": "n8", "parent_axis": "N", "extent": 8, "kind": "tile", "bits": 3},
                {"name": "k2", "parent_axis": "K", "extent": 2, "kind": "tile", "bits": 1},
            ],
        )

    def test_identity_mapping(self):
        layout = make_layout(
            "fp32",
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
        result = solve_edge(layout, layout, {"M": 128, "K": 64}, memory_dtype="fp32", hw_cfg=self.hw)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.permutation, list(range(self.hw.remap_bits)))

    def test_multi_dtype_support(self):
        self.assertEqual(self.hw.block_elements("fp16"), 8)
        self.assertEqual(self.hw.block_elements("int8"), 16)
        self.assertEqual(self.hw.block_elements("fp32"), 4)

    def test_tensor_memory_dtype_mismatch_fails(self):
        producer = make_layout(
            "fp32",
            {"M": "M", "K": "K"},
            [
                {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
                {"name": "K", "parent_axis": "K", "extent": "K", "kind": "outer"},
                {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
            ],
            ["M_outer8", "K", "m8"],
        )
        consumer = make_layout(
            "fp32",
            {"M": "M", "K": "K"},
            [
                {"name": "M_outer8", "parent_axis": "M", "extent": "M//8", "kind": "outer"},
                {"name": "K", "parent_axis": "K", "extent": "K", "kind": "outer"},
                {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
            ],
            ["M_outer8", "K", "m8"],
        )
        result = solve_edge(producer, consumer, {"M": 128, "K": 64}, memory_dtype="fp16", hw_cfg=self.hw)
        self.assertEqual(result.status, "unimplemented")
        self.assertEqual(result.reason_code, "dtype/block packing")

    def test_outer_factor_mismatch_fails(self):
        producer = make_layout(
            "fp32",
            {"M": "M", "K": "K"},
            [
                {"name": "M_outer16", "parent_axis": "M", "extent": "M//16", "kind": "outer"},
                {"name": "K", "parent_axis": "K", "extent": "K", "kind": "outer"},
                {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
            ],
            ["M_outer16", "K", "m8"],
        )
        consumer = make_layout(
            "fp32",
            {"M": "M", "K": "K"},
            [
                {"name": "M_outer32", "parent_axis": "M", "extent": "M//32", "kind": "outer"},
                {"name": "K_outer4", "parent_axis": "K", "extent": "K//4", "kind": "outer"},
                {"name": "m4", "parent_axis": "M", "extent": 4, "kind": "tile"},
                {"name": "m8", "parent_axis": "M", "extent": 8, "kind": "tile"},
                {"name": "k4", "parent_axis": "K", "extent": 4, "kind": "tile"},
            ],
            ["M_outer32", "K_outer4", "m4", "m8", "k4"],
        )
        result = solve_edge(producer, consumer, {"M": 128, "K": 64}, memory_dtype="fp32", hw_cfg=self.hw)
        self.assertEqual(result.status, "unimplemented")
        self.assertEqual(result.reason_code, "factor mismatch")

    def test_non_power_of_two_factor_fails(self):
        producer = make_layout(
            "fp32",
            {"M": "M"},
            [
                {"name": "M_outer6", "parent_axis": "M", "extent": "M//6", "kind": "outer"},
                {"name": "m6", "parent_axis": "M", "extent": 6, "kind": "tile"},
            ],
            ["M_outer6", "m6"],
        )
        result = solve_edge(producer, producer, {"M": 96}, memory_dtype="fp32", hw_cfg=self.hw)
        self.assertEqual(result.status, "unimplemented")
        self.assertEqual(result.reason_code, "unsupported extent")

    def test_graph_level_supports_compact_model(self):
        graph = {
            "shape_bindings": {"M": 128, "K": 64, "N": 64},
            "tensors": {
                "x_fp16": {"dtype": "fp16", "shape": {"M": "M", "K": "K"}},
                "w_fp16": {"dtype": "fp16", "shape": {"K": "K", "N": "N"}},
                "bias_fp32": {"dtype": "fp32", "shape": {"N": "N"}},
            },
            "model": [
                "post_gemm_fp16 = gemm_local_fp16_fp16_fp16(x_fp16, w_fp16)",
                "out_fp32 = prefill_add_MN_N_fp16_fp32_fp32(post_gemm_fp16, bias_fp32)",
            ],
        }
        results = solve_graph(graph, self.hw)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "ok")

    def test_rope_slice_exchange_marks_write_reg_base_addr_swap(self):
        graph = json.loads(Path("examples/graphs/transformer_layer_single_slice.json").read_text(encoding="utf-8-sig"))
        results = solve_graph(graph, self.hw)
        by_edge = {
            (result.producer, result.consumer, result.tensor_name): result
            for result in results
        }

        q_rope_exchange = next(
            result
            for result in results
            if result.tensor_name == "q_sin_fp32"
            and result.consumer.startswith("prefill_add_fp32MN_fp32MN_fp16MN_")
        )
        self.assertEqual(q_rope_exchange.status, "ok")
        self.assertTrue(q_rope_exchange.write_reg_required)
        self.assertIn("cross_slice_base_addr_exchange", q_rope_exchange.write_reg_hint)

        k_rope_exchange = next(
            result
            for result in results
            if result.tensor_name == "k_sin_fp32"
            and result.consumer.startswith("prefill_add_fp32MN_fp32MN_fp16MN_")
        )
        self.assertEqual(k_rope_exchange.status, "ok")
        self.assertTrue(k_rope_exchange.write_reg_required)
        self.assertIn("cross_slice_base_addr_exchange", k_rope_exchange.write_reg_hint)

    def test_qkt_k_route_exposes_explicit_kt_view_and_inb_remap(self):
        graph = json.loads(Path("examples/graphs/transformer_layer_single_slice.json").read_text(encoding="utf-8-sig"))
        results = solve_graph(graph, self.hw)

        by_tensor = {}
        for result in results:
            by_tensor.setdefault(result.tensor_name, []).append(result)

        self.assertIn("k_rope_fp16", by_tensor)
        self.assertIn("k_rope_t_fp16", by_tensor)

        kt_view_edge = next(
            result for result in by_tensor["k_rope_fp16"] if result.consumer.startswith("prefill_qkt_kt_view_fp16_fp16_")
        )
        self.assertEqual(kt_view_edge.status, "ok")

        gemm_inb_edge = next(
            result for result in by_tensor["k_rope_t_fp16"] if result.consumer.startswith("ring_gemm_fp16_fp16_fp16_")
        )
        self.assertEqual(gemm_inb_edge.status, "ok")
        self.assertIsNotNone(gemm_inb_edge.producer_visible_outer_bits)
        self.assertIsNotNone(gemm_inb_edge.consumer_visible_outer_bits)
        self.assertEqual(
            gemm_inb_edge.consumer_visible_outer_bits,
            [
                gemm_inb_edge.producer_visible_outer_bits[index]
                for index in gemm_inb_edge.permutation[: len(gemm_inb_edge.consumer_visible_outer_bits)]
            ],
        )

    def test_sv_consumes_v_fp16_directly_and_checks_inb_remap(self):
        graph = json.loads(Path("examples/graphs/transformer_layer_single_slice.json").read_text(encoding="utf-8-sig"))
        results = solve_graph(graph, self.hw)

        by_tensor = {}
        for result in results:
            by_tensor.setdefault(result.tensor_name, []).append(result)

        self.assertIn("v_fp16", by_tensor)
        sv_gemm_inb_edge = next(
            result
            for result in by_tensor["v_fp16"]
            if result.consumer.startswith("ring_gemm_fp16_fp16_fp16_")
            and result.tensor_name == "v_fp16"
        )
        self.assertEqual(sv_gemm_inb_edge.status, "ok")
        self.assertIsNotNone(sv_gemm_inb_edge.producer_visible_outer_bits)
        self.assertIsNotNone(sv_gemm_inb_edge.consumer_visible_outer_bits)
        self.assertEqual(
            sv_gemm_inb_edge.consumer_visible_outer_bits,
            [
                sv_gemm_inb_edge.producer_visible_outer_bits[index]
                for index in sv_gemm_inb_edge.permutation[: len(sv_gemm_inb_edge.consumer_visible_outer_bits)]
            ],
        )

    def test_v_proj_uses_row_writeback_annotation(self):
        graph = json.loads(Path("examples/graphs/transformer_layer_single_slice.json").read_text(encoding="utf-8-sig"))
        results = solve_graph(graph, self.hw)
        v_proj_edge = next(
            result
            for result in results
            if result.tensor_name == "v_proj_fp16"
            and result.consumer.startswith("prefill_add_V_MN_N_fp16_fp32_fp16_")
        )
        self.assertEqual(v_proj_edge.status, "ok")
        self.assertTrue(v_proj_edge.write_reg_required)
        self.assertIn("row_writeback", v_proj_edge.write_reg_hint)
        self.assertEqual(
            [factor["name"] for factor in v_proj_edge.producer_bound_layout["ordered_factors"]],
            ["M_outer32", "N_outer32", "n4", "m4", "m8", "n8"],
        )

    def test_add_mn_n_fp16_fp32_fp16_infers_fp16_output(self):
        spec = {
            "shape_bindings": {"sequence_length": 128, "hidden_size": 896},
            "params": {"slices_per_cluster": 4, "slices_num": 28},
            "tensors": {
                "a_fp16": {
                    "dtype": "fp16",
                    "shape": {"M": "sequence_length", "K": "hidden_size"},
                    "partition": {"K": {"by_scope": {"cluster": "slices_per_cluster", "global": "slices_num"}}},
                },
                "b_fp16": {
                    "dtype": "fp16",
                    "shape": {"K": "hidden_size", "N": "hidden_size"},
                    "partition": {"N": {"by_scope": {"cluster": "slices_num", "global": "slices_num"}}},
                },
                "bias_vec_fp32": {
                    "dtype": "fp32",
                    "shape": {"N": "hidden_size"},
                    "partition": {"N": {"follow": "inA:N"}},
                },
            },
            "model": [
                "gemm_out_fp16 = ring_gemm_fp16_fp16_fp16(a_fp16, b_fp16, ring_scope=cluster)",
                "v_out_fp16 = prefill_add_MN_N_fp16_fp32_fp16(gemm_out_fp16, bias_vec_fp32)",
            ],
        }
        expanded = expand_model_spec(spec)
        self.assertEqual(expanded["tensors"]["v_out_fp16"]["dtype"], "fp16")
        self.assertEqual(expanded["tensors"]["v_out_fp16"]["shape"], {"M": "M", "N": "N"})
        self.assertEqual(expanded["tensors"]["v_out_fp16"]["resolved_shape"], {"M": 128, "N": 32})

    def test_add_v_mn_n_fp16_fp32_fp16_infers_fp16_output(self):
        spec = {
            "shape_bindings": {"sequence_length": 128, "hidden_size": 896},
            "params": {"slices_per_cluster": 4, "slices_num": 28},
            "tensors": {
                "a_fp16": {
                    "dtype": "fp16",
                    "shape": {"M": "sequence_length", "K": "hidden_size"},
                    "partition": {"K": {"by_scope": {"cluster": "slices_per_cluster", "global": "slices_num"}}},
                },
                "b_fp16": {
                    "dtype": "fp16",
                    "shape": {"K": "hidden_size", "N": "hidden_size"},
                    "partition": {"N": {"by_scope": {"cluster": "slices_num", "global": "slices_num"}}},
                },
                "bias_vec_fp32": {
                    "dtype": "fp32",
                    "shape": {"N": "hidden_size"},
                    "partition": {"N": {"follow": "inA:N"}},
                },
            },
            "model": [
                "gemm_out_fp16 = ring_gemm_fp16_fp16_fp16(a_fp16, b_fp16, ring_scope=cluster)",
                "v_out_fp16 = prefill_add_V_MN_N_fp16_fp32_fp16(gemm_out_fp16, bias_vec_fp32)",
            ],
        }
        expanded = expand_model_spec(spec)
        self.assertEqual(expanded["tensors"]["v_out_fp16"]["dtype"], "fp16")
        self.assertEqual(expanded["tensors"]["v_out_fp16"]["shape"], {"M": "M", "N": "N"})
        self.assertEqual(expanded["tensors"]["v_out_fp16"]["resolved_shape"], {"M": 128, "N": 32})

    def test_remote_sum_fp16_fp32_preserves_mn_shape(self):
        spec = {
            "shape_bindings": {"M": 128, "N": 32},
            "tensors": {
                "qkt_local_fp16": {"dtype": "fp16", "shape": {"M": "M", "N": "N"}},
            },
            "model": [
                "qkt_sum_fp32 = prefill_remote_sum_fp16MN_fp32MN(qkt_local_fp16, reduce_scope=cluster)",
            ],
        }
        expanded = expand_model_spec(spec)
        self.assertEqual(expanded["tensors"]["qkt_sum_fp32"]["dtype"], "fp32")
        self.assertEqual(expanded["tensors"]["qkt_sum_fp32"]["shape"], {"M": "M", "N": "N"})
        self.assertEqual(expanded["tensors"]["qkt_sum_fp32"]["resolved_shape"], {"M": 128, "N": 32})

    def test_remote_sum_fp32_fp32_preserves_m_shape(self):
        spec = {
            "shape_bindings": {"M": 128},
            "tensors": {
                "row_partial_fp32": {"dtype": "fp32", "shape": {"M": "M"}},
            },
            "model": [
                "row_sum_fp32 = prefill_remote_sum_Mfp32_Mfp32(row_partial_fp32, reduce_scope=cluster)",
            ],
        }
        expanded = expand_model_spec(spec)
        self.assertEqual(expanded["tensors"]["row_sum_fp32"]["dtype"], "fp32")
        self.assertEqual(expanded["tensors"]["row_sum_fp32"]["shape"], {"M": "M"})
        self.assertEqual(expanded["tensors"]["row_sum_fp32"]["resolved_shape"], {"M": 128})

    def test_expand_model_spec_resolves_partition_by_scope_for_prefill_scope(self):
        spec = {
            "shape_bindings": {"sequence_length": 128, "hidden_size": 896},
            "params": {"slices_per_cluster": 4, "slices_num": 28},
            "tensors": {
                "hidden_in_fp32": {
                    "dtype": "fp32",
                    "shape": {"M": "sequence_length", "N": "hidden_size"},
                    "partition": {"N": {"by_scope": {"cluster": "slices_per_cluster", "global": "slices_num"}}},
                }
            },
            "model": [
                "row_scale_fp32 = prefill_summac(hidden_in_fp32, scope=cluster)",
            ],
        }
        expanded = expand_model_spec(spec)
        op_inputs = expanded["ops"]["prefill_summac_0"]["inputs"]
        self.assertEqual(op_inputs["inA"]["resolved_shape"], {"M": 128, "N": 224})

    def test_expand_model_spec_resolves_partition_by_scope_for_reduce_scope(self):
        spec = {
            "shape_bindings": {"M": 128, "N": 896},
            "params": {"slices_per_cluster": 4, "slices_num": 28},
            "tensors": {
                "qkt_local_fp16": {
                    "dtype": "fp16",
                    "shape": {"M": "M", "N": "N"},
                    "partition": {"N": {"by_scope": {"cluster": "slices_per_cluster", "global": "slices_num"}}},
                }
            },
            "model": [
                "qkt_sum_fp32 = prefill_remote_sum_fp16MN_fp32MN(qkt_local_fp16, reduce_scope=cluster)",
            ],
        }
        expanded = expand_model_spec(spec)
        op_inputs = expanded["ops"]["prefill_remote_sum_fp16MN_fp32MN_0"]["inputs"]
        self.assertEqual(op_inputs["inA"]["resolved_shape"], {"M": 128, "N": 224})

    def test_ring_gemm_bias_connection(self):
        graph = {
            "shape_bindings": {"sequence_length": 128, "hidden_size": 896},
            "params": {"slices_per_cluster": 4, "slices_num": 28},
            "tensors": {
                "a_fp16": {
                    "dtype": "fp16",
                    "shape": {"M": "sequence_length", "K": "hidden_size"},
                    "partition": {"K": {"by_scope": {"cluster": "slices_per_cluster", "global": "slices_num"}}},
                },
                "b_fp16": {
                    "dtype": "fp16",
                    "shape": {"K": "hidden_size", "N": "hidden_size"},
                    "partition": {"N": {"by_scope": {"cluster": "slices_num", "global": "slices_num"}}},
                },
                "bias_vec_fp32": {
                    "dtype": "fp32",
                    "shape": {"N": "hidden_size"},
                    "partition": {"N": {"follow": "inA:N"}},
                },
            },
            "model": [
                "gemm_out_fp16 = ring_gemm_fp16_fp16_fp16(a_fp16, b_fp16, ring_scope=cluster)",
                "bias_out_fp32 = prefill_add_MN_N_fp16_fp32_fp32(gemm_out_fp16, bias_vec_fp32)",
            ],
        }
        results = solve_graph(graph, self.hw)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].producer, "ring_gemm_fp16_fp16_fp16_0")
        self.assertEqual(results[0].consumer, "prefill_add_MN_N_fp16_fp32_fp32_1")
        self.assertEqual(results[0].tensor_name, "gemm_out_fp16")
        self.assertEqual(results[0].status, "ok")
        self.assertEqual(results[0].permutation[:11], [3, 4, 7, 8, 0, 1, 2, 5, 6, 9, 10])

    def test_single_axis_vector_layout_does_not_false_positive_write_reg(self):
        registry = build_default_registry()
        producer = registry["prefill_mac_SFU"].output_ports["out"].layout_builder("fp32")
        consumer = registry["prefill_mul_fp32MN_fp32M_fp32MN"].input_ports["inB"].layout_builder("fp32")
        result = solve_edge(
            producer,
            consumer,
            {"M": 32},
            memory_dtype="fp32",
            hw_cfg=self.hw,
            producer="op2",
            consumer="op3",
            tensor_name="op2__out",
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.permutation, list(range(self.hw.remap_bits)))
        self.assertFalse(result.write_reg_required)
        self.assertIsNone(result.write_reg_hint)

    def test_identical_mn_layout_does_not_trigger_write_reg(self):
        registry = build_default_registry()
        producer = registry["prefill_add_fp32MN_fp32MN_fp32MN"].output_ports["out"].layout_builder("fp32")
        consumer = registry["prefill_max"].input_ports["inA"].layout_builder("fp32")
        result = solve_edge(
            producer,
            consumer,
            {"M": 32, "N": 32},
            memory_dtype="fp32",
            hw_cfg=self.hw,
            producer="op0",
            consumer="op1",
            tensor_name="op0__out",
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.permutation, list(range(self.hw.remap_bits)))
        self.assertFalse(result.write_reg_required)
        self.assertIsNone(result.write_reg_hint)

    def test_build_expanded_graph_from_external_rmsnorm(self):
        payload = json.loads(Path("examples/graphs/rmsnorm.json").read_text(encoding="utf-8-sig"))
        expanded = build_expanded_graph_from_external_rmsnorm(payload)

        self.assertEqual(set(expanded["ops"]), {"op0", "op1", "op2", "op3"})
        self.assertEqual(len(expanded["edges"]), 3)
        self.assertEqual(expanded["ops"]["op0"]["op_type"], "prefill_summac")
        self.assertEqual(expanded["ops"]["op1"]["op_type"], "prefill_remote_sum_Mfp32_Mfp32")
        self.assertEqual(expanded["ops"]["op2"]["op_type"], "prefill_mac_SFU")
        self.assertEqual(expanded["ops"]["op3"]["op_type"], "prefill_mul_fp32MN_fp32M_fp32MN")
        self.assertEqual(expanded["tensors"]["op0__out"]["resolved_shape"], {"M": 32})
        self.assertEqual(expanded["tensors"]["op3__out"]["resolved_shape"], {"M": 32, "N": 32})
        self.assertEqual(
            [(edge["producer"], edge["consumer"], edge["consumer_port"]) for edge in expanded["edges"]],
            [("op0", "op1", "inA"), ("op1", "op2", "inA"), ("op2", "op3", "inB")],
        )

    def test_fill_external_rmsnorm_remapping_keeps_external_and_identity_edges_empty(self):
        payload = {
            "used_slices": 28,
            "operators": [
                {
                    "id": "op0",
                    "type": "prefill_summac_fp32MN_fp32MN",
                    "used_slices": "0b1111111111111111111111111111",
                    "inputs": {
                        "A": {
                            "shape": [1, 32, 32],
                            "remapping": None,
                            "source": {"type": "external"},
                        }
                    },
                    "output": {"shape": [1, 1, 32], "remapping": None},
                },
                {
                    "id": "op1",
                    "type": "prefill_remote_sum_fp32MN_fp32MN",
                    "used_slices": "0b1111111111111111111111111111",
                    "inputs": {"A": {"shape": [1, 28, 32], "source": "op0"}},
                    "output": {"shape": [1, 1, 32]},
                },
                {
                    "id": "op2",
                    "type": "prefill_mac_SFU_fp32MN_fp32MN",
                    "used_slices": "0b1111111111111111111111111111",
                    "inputs": {"A": {"shape": [1, 1, 32], "source": "op1"}},
                    "output": {"shape": [1, 1, 32]},
                },
            ],
        }
        expanded = build_expanded_graph_from_external_rmsnorm(payload)
        results = solve_graph(expanded, self.hw)
        by_key = {
            (edge["producer"], edge["consumer"], edge["consumer_port"]): result
            for edge, result in zip(expanded["edges"], results)
        }

        filled = fill_external_rmsnorm_remapping(payload, hw_cfg=self.hw)
        default_permutation = list(range(self.hw.remap_bits))

        op0_a = filled["operators"][0]["inputs"]["A"]
        self.assertIsNone(op0_a.get("remapping"))

        for operator in filled["operators"]:
            op_id = operator["id"]
            for external_port, input_data in operator.get("inputs", {}).items():
                source = input_data.get("source")
                if isinstance(source, dict) and source.get("type") == "external":
                    self.assertIsNone(input_data.get("remapping"))
                    continue
                internal_port = {"A": "inA", "B": "inB"}[external_port]
                result = by_key[(source, op_id, internal_port)]
                if result.permutation == default_permutation:
                    self.assertIsNone(input_data.get("remapping"))
                else:
                    self.assertEqual(input_data.get("remapping"), result.permutation)

    def test_fill_external_rmsnorm_remapping_real_rmsnorm_succeeds(self):
        payload = json.loads(Path("examples/graphs/rmsnorm.json").read_text(encoding="utf-8-sig"))
        filled = fill_external_rmsnorm_remapping(payload, hw_cfg=self.hw)
        self.assertIsNone(filled["operators"][0]["inputs"]["A"].get("remapping"))
        self.assertNotIn("writereg", filled["operators"][0]["inputs"]["A"])
        self.assertIsNone(filled["operators"][1]["inputs"]["A"].get("remapping"))
        self.assertNotIn("writereg", filled["operators"][1]["inputs"]["A"])
        self.assertIsNone(filled["operators"][2]["inputs"]["A"].get("remapping"))
        self.assertNotIn("writereg", filled["operators"][2]["inputs"]["A"])
        self.assertIsNone(filled["operators"][3]["inputs"]["A"].get("remapping"))
        self.assertNotIn("writereg", filled["operators"][3]["inputs"]["A"])
        self.assertIsNone(filled["operators"][3]["inputs"]["B"].get("remapping"))
        self.assertNotIn("writereg", filled["operators"][3]["inputs"]["B"])

    def test_normalize_graph_spec_preserves_rmsnorm_base_addrs_and_source_tensors(self):
        payload = json.loads(Path("examples/graphs/rmsnorm_withbaseaddr.json").read_text(encoding="utf-8-sig"))
        expanded = normalize_graph_spec(payload, require_base_addrs=True)

        self.assertEqual(expanded["tensors"]["op0__out"]["base_addr"], 0x800)
        self.assertEqual(expanded["tensors"]["op1__out"]["base_addr"], 0x840)
        self.assertEqual(expanded["tensors"]["op2__out"]["base_addr"], 0x880)
        self.assertEqual(expanded["tensors"]["op3__out"]["base_addr"], 0x10C0)
        self.assertEqual(expanded["tensors"]["op0__inA__external"]["base_addr"], 0x0)
        self.assertEqual(expanded["tensors"]["op3__inA__external"]["base_addr"], 0x8C0)
        self.assertEqual(expanded["ops"]["op0"]["inputs"]["inA"]["source_tensor"], "op0__inA__external")
        self.assertEqual(expanded["ops"]["op1"]["inputs"]["inA"]["source_tensor"], "op0__out")
        self.assertEqual(expanded["ops"]["op3"]["inputs"]["inB"]["source_tensor"], "op2__out")

    def test_normalize_graph_spec_rejects_missing_base_addr_for_performance(self):
        payload = json.loads(Path("examples/graphs/rmsnorm.json").read_text(encoding="utf-8-sig"))
        with self.assertRaisesRegex(RmsNormBridgeError, "must define 'base_addr'"):
            normalize_graph_spec(payload, require_base_addrs=True)

    def test_normalize_graph_spec_rejects_mismatched_connected_base_addr(self):
        payload = json.loads(Path("examples/graphs/rmsnorm_withbaseaddr.json").read_text(encoding="utf-8-sig"))
        payload["operators"][1]["inputs"]["A"]["base_addr"] = "0x00000820"

        with self.assertRaisesRegex(RmsNormBridgeError, "base_addr mismatch"):
            normalize_graph_spec(payload, require_base_addrs=True)

    def test_analyze_graph_performance_accepts_external_rmsnorm_with_base_addrs(self):
        payload = json.loads(Path("examples/graphs/rmsnorm_withbaseaddr.json").read_text(encoding="utf-8-sig"))
        hardware, perf_cfg = load_performance_config("examples/configs/performance_config.json")

        report = analyze_graph_performance(payload, hardware, perf_cfg)

        self.assertEqual(report["graph_summary"]["op_count"], 4)
        self.assertEqual(report["graph_summary"]["edge_count"], 3)
        self.assertEqual(set(report["modes"]), {"baseline", "remap", "remap_interleave"})
        self.assertIn(report["overview"]["best_mode_by_estimated_latency"], report["modes"])

    def test_fill_external_rmsnorm_remapping_writes_non_default_permutation(self):
        payload = json.loads(Path("examples/graphs/rmsnorm.json").read_text(encoding="utf-8-sig"))
        mocked_results = [
            mock.Mock(
                spec=EdgeSolveResult,
                producer="op0",
                consumer="op1",
                consumer_port="inA",
                tensor_name="op0__out",
                status="ok",
                permutation=list(range(self.hw.remap_bits)),
                write_reg_required=False,
                write_reg_hint=None,
                reason=None,
                reason_code=None,
            ),
            mock.Mock(
                spec=EdgeSolveResult,
                producer="op1",
                consumer="op2",
                consumer_port="inA",
                tensor_name="op1__out",
                status="ok",
                permutation=[1, 0] + list(range(2, self.hw.remap_bits)),
                write_reg_required=False,
                write_reg_hint=None,
                reason=None,
                reason_code=None,
            ),
            mock.Mock(
                spec=EdgeSolveResult,
                producer="op2",
                consumer="op3",
                consumer_port="inB",
                tensor_name="op2__out",
                status="ok",
                permutation=list(range(self.hw.remap_bits)),
                write_reg_required=False,
                write_reg_hint=None,
                reason=None,
                reason_code=None,
            ),
        ]

        with mock.patch("address_remapping.rmsnorm_bridge.solve_graph", return_value=mocked_results):
            filled = fill_external_rmsnorm_remapping(payload, hw_cfg=self.hw)

        self.assertIsNone(filled["operators"][0]["inputs"]["A"].get("remapping"))
        self.assertEqual(filled["operators"][2]["inputs"]["A"]["remapping"], [1, 0] + list(range(2, self.hw.remap_bits)))
        self.assertIsNone(filled["operators"][1]["inputs"]["A"].get("remapping"))
        self.assertIsNone(filled["operators"][3]["inputs"]["A"].get("remapping"))
        self.assertIsNone(filled["operators"][3]["inputs"]["B"].get("remapping"))
        self.assertNotIn("writereg", filled["operators"][2]["inputs"]["A"])

    def test_fill_external_rmsnorm_remapping_writes_writereg_object(self):
        payload = json.loads(Path("examples/graphs/rmsnorm.json").read_text(encoding="utf-8-sig"))
        mocked_results = [
            mock.Mock(
                spec=EdgeSolveResult,
                producer="op0",
                consumer="op1",
                consumer_port="inA",
                tensor_name="op0__out",
                status="ok",
                permutation=list(range(self.hw.remap_bits)),
                write_reg_required=True,
                write_reg_hint="reorder(x)->(y)",
                reason=None,
                reason_code=None,
            ),
            mock.Mock(
                spec=EdgeSolveResult,
                producer="op1",
                consumer="op2",
                consumer_port="inA",
                tensor_name="op1__out",
                status="ok",
                permutation=list(range(self.hw.remap_bits)),
                write_reg_required=False,
                write_reg_hint=None,
                reason=None,
                reason_code=None,
            ),
            mock.Mock(
                spec=EdgeSolveResult,
                producer="op2",
                consumer="op3",
                consumer_port="inB",
                tensor_name="op2__out",
                status="ok",
                permutation=list(range(self.hw.remap_bits)),
                write_reg_required=False,
                write_reg_hint=None,
                reason=None,
                reason_code=None,
            ),
        ]

        with mock.patch("address_remapping.rmsnorm_bridge.solve_graph", return_value=mocked_results):
            filled = fill_external_rmsnorm_remapping(payload, hw_cfg=self.hw)

        self.assertIsNone(filled["operators"][1]["inputs"]["A"].get("remapping"))
        self.assertEqual(
            filled["operators"][1]["inputs"]["A"]["writereg"],
            {"required": True, "hint": "reorder(x)->(y)"},
        )
        self.assertNotIn("writereg", filled["operators"][2]["inputs"]["A"])
        self.assertNotIn("writereg", filled["operators"][3]["inputs"]["B"])

    def test_cli_outputs_json(self):
        graph_path = Path("examples/graphs/ring_gemm_bias.json").resolve()
        output_path = Path("outputs/solver/ring_gemm_bias_result.json").resolve()
        if output_path.exists():
            output_path.unlink()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path("src").resolve())
        completed = subprocess.run(
            [sys.executable, "-m", "address_remapping.cli", "solve-graph", str(graph_path), "--format", "json"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        payload = json.loads(completed.stdout)
        self.assertIsInstance(payload, list)
        if payload:
            self.assertEqual(payload[0]["status"], "ok")
        self.assertTrue(output_path.exists())
        written_payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(written_payload, payload)

    def test_fill_rmsnorm_remapping_cli_outputs_json(self):
        source_payload = json.dumps(
            {
                "used_slices": 28,
                "operators": [
                    {
                        "id": "op0",
                        "type": "prefill_summac_fp32MN_fp32MN",
                        "used_slices": "0b1111111111111111111111111111",
                        "inputs": {
                            "A": {
                                "shape": [1, 32, 32],
                                "remapping": None,
                                "source": {"type": "external"},
                            }
                        },
                        "output": {"shape": [1, 1, 32], "remapping": None},
                    },
                    {
                        "id": "op1",
                        "type": "prefill_remote_sum_fp32MN_fp32MN",
                        "used_slices": "0b1111111111111111111111111111",
                        "inputs": {"A": {"shape": [1, 28, 32], "source": "op0"}},
                        "output": {"shape": [1, 1, 32]},
                    },
                    {
                        "id": "op2",
                        "type": "prefill_mac_SFU_fp32MN_fp32MN",
                        "used_slices": "0b1111111111111111111111111111",
                        "inputs": {"A": {"shape": [1, 1, 32], "source": "op1"}},
                        "output": {"shape": [1, 1, 32]},
                    },
                ],
            },
            indent=2,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path("src").resolve())
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            graph_path = temp_root / "rmsnorm.json"
            output_path = temp_root / "rmsnorm_remapped.json"
            graph_path.write_text(source_payload, encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "address_remapping.cli",
                    "fill-rmsnorm-remapping",
                    str(graph_path),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            payload = json.loads(completed.stdout)
            self.assertTrue(output_path.exists())
            written_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written_payload, payload)
            self.assertIsNone(payload["operators"][0]["inputs"]["A"].get("remapping"))
            self.assertIsNone(payload["operators"][1]["inputs"]["A"].get("remapping"))
            self.assertNotIn("writereg", payload["operators"][0]["inputs"]["A"])
            self.assertNotIn("writereg", payload["operators"][1]["inputs"]["A"])

    def test_performance_cli_outputs_json_and_markdown(self):
        graph_path = Path("examples/graphs/ring_gemm_bias.json").resolve()
        output_path = Path("outputs/performance/ring_gemm_bias/ring_gemm_bias_performance.json").resolve()
        markdown_path = output_path.with_suffix(".md")
        for path in (output_path, markdown_path):
            if path.exists():
                path.unlink()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path("src").resolve())
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "address_remapping.cli",
                "analyze-performance",
                str(graph_path),
                "--format",
                "json",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(set(payload["modes"]), {"baseline", "remap", "remap_interleave"})
        self.assertIn("summary_markdown", payload)
        self.assertIn("true_roofline", payload)
        self.assertIn("overview", payload)
        self.assertIn("mode_summaries", payload)
        self.assertEqual(payload["overview"]["cycle_domain"], "slice-cycle")
        self.assertEqual(payload["overview"]["memory_timing_domain"], "bank-cycle")
        self.assertEqual(
            payload["overview"]["best_mode_by_estimated_latency"],
            payload["overview"]["mode_order_by_estimated_latency"][0],
        )

    def test_performance_cli_accepts_external_rmsnorm_with_base_addrs(self):
        graph_path = Path("examples/graphs/rmsnorm_withbaseaddr.json").resolve()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path("src").resolve())
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output_path = temp_root / "rmsnorm_withbaseaddr_performance.json"
            markdown_path = output_path.with_suffix(".md")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "address_remapping.cli",
                    "analyze-performance",
                    str(graph_path),
                    "--format",
                    "json",
                    "--config",
                    str(Path("examples/configs/performance_config.json").resolve()),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["graph_summary"]["op_count"], 4)
            self.assertTrue(output_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertEqual(
                set(payload["overview"]["mode_order_by_estimated_latency"]),
                {"baseline", "remap", "remap_interleave"},
            )
            self.assertIn("top_ops_by_latency", payload["mode_summaries"][MODE_BASELINE])
            self.assertIn("analytical_model", payload["modes"][MODE_BASELINE])
            self.assertEqual(
                payload["modes"][MODE_BASELINE]["op_breakdown"][0]["true_roofline"]["peak_memory_bandwidth_bytes_per_cycle"],
                payload["hardware"]["derived"]["peak_memory_bandwidth_bytes_per_cycle"],
            )
            self.assertEqual(payload["hardware"]["dram"]["banks_per_slice"], 4)
            self.assertEqual(payload["hardware"]["compute"]["gemm_core"]["peak_ops_per_cycle"], 256)
            self.assertEqual(payload["hardware"]["derived"]["peak_memory_bandwidth_bytes_per_cycle"], 32.0)
            self.assertEqual(payload["hardware"]["derived"]["ring_a_buffer_bytes"], 128)
            self.assertEqual(payload["hardware"]["derived"]["ring_b_buffer_bytes"], 128)
            self.assertEqual(payload["performance_config"]["overlap"]["gemm_read"], 0.65)
            self.assertGreater(payload["true_roofline"]["roofline_cycles"], 0)
            written_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written_payload["modes"]["baseline"]["mode"], MODE_BASELINE)
            self.assertEqual(
                written_payload["true_roofline"]["roofline_cycles"],
                payload["true_roofline"]["roofline_cycles"],
            )
            self.assertIn("# Performance Summary", markdown_path.read_text(encoding="utf-8"))

    def test_build_roofline_summary_reports_limit_and_achieved(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/ring_gemm_bias.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=False,
        )
        summary = build_roofline_summary(payload, mode=MODE_REMAP, graph_name="ring_gemm_bias")
        gemm = next(op for op in summary["operators"] if op["op_type"] == "ring_gemm_fp16_fp16_fp16")
        self.assertEqual(summary["hardware"]["peak_memory_bandwidth_bytes_per_cycle"], 32.0)
        self.assertEqual(summary["hardware"]["gemm_peak_ops_per_cycle"], 256.0)
        self.assertAlmostEqual(gemm["roofline_limit_ops_per_cycle"], 256.0)
        self.assertLess(gemm["achieved_ops_per_cycle"], gemm["roofline_limit_ops_per_cycle"])
        if gemm["hardware_measured_cycles"] is None:
            self.assertIsNone(gemm["measured_ops_per_cycle"])
        else:
            self.assertAlmostEqual(
                gemm["measured_ops_per_cycle"],
                gemm["work_ops"] / float(gemm["hardware_measured_cycles"]),
            )
        self.assertAlmostEqual(
            gemm["analytical_bandwidth_utilization"],
            gemm["analytical_bandwidth_bytes_per_cycle"] / summary["hardware"]["peak_memory_bandwidth_bytes_per_cycle"],
        )
        self.assertEqual(gemm["utilization"]["analytical_bandwidth"], f"{gemm['analytical_bandwidth_utilization'] * 100.0:.2f}%")
        self.assertEqual(gemm["utilization"]["analytical_compute"], f"{gemm['analytical_efficiency'] * 100.0:.2f}%")

    def test_build_roofline_summary_includes_hardware_measured_point(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/rmsnorm_withbaseaddr.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=False,
        )
        summary = build_roofline_summary(payload, mode=MODE_BASELINE, graph_name="rmsnorm_withbaseaddr")
        summac = next(op for op in summary["operators"] if op["op_name"] == "op0")
        baseline_op0 = next(op for op in payload["modes"][MODE_BASELINE]["op_breakdown"] if op["op_name"] == "op0")
        measured_cycles = float(baseline_op0["hardware_measured_cycles"])
        self.assertEqual(summac["hardware_measured_cycles"], measured_cycles)
        self.assertAlmostEqual(summac["measured_ops_per_cycle"], summac["work_ops"] / measured_cycles)
        self.assertGreater(summac["measured_efficiency"], 0.0)
        self.assertAlmostEqual(summac["analytical_bandwidth_bytes_per_cycle"], summac["total_bytes"] / summac["latency_cycles"])
        self.assertAlmostEqual(summac["measured_bandwidth_bytes_per_cycle"], summac["total_bytes"] / measured_cycles)
        self.assertAlmostEqual(summac["analytical_bandwidth_utilization"], summac["analytical_bandwidth_bytes_per_cycle"] / 32.0)
        self.assertAlmostEqual(summac["measured_bandwidth_utilization"], summac["measured_bandwidth_bytes_per_cycle"] / 32.0)
        self.assertEqual(summac["utilization"]["analytical_compute"], f"{summac['analytical_efficiency'] * 100.0:.2f}%")
        self.assertEqual(summac["utilization"]["measured_compute"], f"{summac['measured_efficiency'] * 100.0:.2f}%")
        self.assertEqual(
            summac["utilization"]["analytical_bandwidth"],
            f"{summac['analytical_bandwidth_utilization'] * 100.0:.2f}%",
        )
        self.assertEqual(
            summac["utilization"]["measured_bandwidth"],
            f"{summac['measured_bandwidth_utilization'] * 100.0:.2f}%",
        )

    def test_plot_roofline_cli_outputs_svg_and_summary_json(self):
        graph_path = Path("examples/graphs/ring_gemm_bias.json").resolve()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path("src").resolve())
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output_path = temp_root / "ring_gemm_bias_roofline.svg"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "address_remapping.cli",
                    "plot-roofline",
                    str(graph_path),
                    "--config",
                    str(Path("examples/configs/performance_config.json").resolve()),
                    "--mode",
                    "remap",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            payload = json.loads(completed.stdout)
            summary_path = output_path.with_suffix(".json")
            self.assertTrue(output_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(payload["svg_path"], str(output_path))
            self.assertEqual(payload["summary_path"], str(summary_path))
            self.assertIn("<svg", output_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            gemm = next(op for op in summary["operators"] if op["op_type"] == "ring_gemm_fp16_fp16_fp16")
            self.assertIn("analytical_efficiency", gemm)

    def test_plot_roofline_svg_contains_hardware_measured_legend_and_summary(self):
        graph_path = Path("examples/graphs/rmsnorm_withbaseaddr.json").resolve()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path("src").resolve())
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output_path = temp_root / "rmsnorm_withbaseaddr_roofline.svg"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "address_remapping.cli",
                    "plot-roofline",
                    str(graph_path),
                    "--config",
                    str(Path("examples/configs/performance_config.json").resolve()),
                    "--mode",
                    "baseline",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            payload = json.loads(completed.stdout)
            summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
            svg_text = output_path.read_text(encoding="utf-8")
            op0 = next(op for op in summary["operators"] if op["op_name"] == "op0")
            self.assertIsNotNone(op0["hardware_measured_cycles"])
            self.assertGreater(float(op0["hardware_measured_cycles"]), 0.0)
            self.assertIn("hardware measured performance", svg_text)
            self.assertIn("analytical achieved performance", svg_text)

    def test_write_performance_outputs_adds_display_units(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/rmsnorm_withbaseaddr.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "perf.json"
            write_performance_outputs(str(output_path), payload, str(output_path))
            written = json.loads(output_path.read_text(encoding="utf-8"))
            baseline_cycles = float(payload["overview"]["baseline_latency_cycles"])
            self.assertEqual(
                written["overview"]["baseline_latency_cycles_display"],
                f"{baseline_cycles:.2f} cycles",
            )
            self.assertEqual(
                written["mode_summaries"]["baseline"]["top_ops_by_latency"][0]["latency_share_display"],
                f"{float(payload['mode_summaries']['baseline']['top_ops_by_latency'][0]['latency_share']) * 100.0:.2f}%",
            )

    def test_true_roofline_is_mode_independent(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/ring_gemm_bias.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=False,
        )
        baseline = payload["modes"][MODE_BASELINE]["op_breakdown"]
        remap = payload["modes"]["remap"]["op_breakdown"]
        interleave = payload["modes"][MODE_REMAP_INTERLEAVE]["op_breakdown"]
        self.assertEqual(
            [op["true_roofline"]["roofline_cycles"] for op in baseline],
            [op["true_roofline"]["roofline_cycles"] for op in remap],
        )
        self.assertEqual(
            [op["true_roofline"]["roofline_cycles"] for op in baseline],
            [op["true_roofline"]["roofline_cycles"] for op in interleave],
        )

    def test_baseline_inserts_software_relayout_stage(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/transformer_layer_single_slice.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=True,
        )
        baseline_breakdown = payload["modes"][MODE_BASELINE]["op_breakdown"]
        remap_breakdown = payload["modes"][MODE_REMAP]["op_breakdown"]
        self.assertTrue(any(op["kind"] == "relayout" for op in baseline_breakdown))
        self.assertFalse(any(op["kind"] == "relayout" for op in remap_breakdown))
        self.assertGreater(payload["mode_summaries"][MODE_BASELINE]["software_relayout_stage_count"], 0)

    def test_true_roofline_excludes_baseline_relayout_work(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/transformer_layer_single_slice.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=False,
        )
        baseline_total_work = sum(float(op["work_ops"]) for op in payload["modes"][MODE_BASELINE]["op_breakdown"])
        self.assertGreater(baseline_total_work, payload["true_roofline"]["work_ops"])

    def test_gemm_true_roofline_uses_tensor_core_peak(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/ring_gemm_bias.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=False,
        )
        gemm_op = next(op for op in payload["modes"][MODE_REMAP]["op_breakdown"] if op["op_type"] == "ring_gemm_fp16_fp16_fp16")
        self.assertEqual(gemm_op["true_roofline"]["peak_compute_ops_per_cycle"], 256)
        self.assertGreater(gemm_op["work_ops"], 0)
        self.assertEqual(gemm_op["work_ops"], 2 * 64 * 32 * 4 * 4)

    def test_reduction_work_uses_input_minus_output_elements(self):
        work_ops, compute_cycles, peak = _estimate_compute(
            "prefill_summac",
            {
                "inputs": {"inA": {"resolved_shape": {"M": 32, "N": 128}}},
                "outputs": {"out": {"resolved_shape": {"M": 32}}},
            },
            self.hw,
        )
        self.assertEqual(work_ops, 32 * (128 - 1))
        self.assertEqual(compute_cycles, (32 * (128 - 1) + peak - 1) // peak)
        self.assertEqual(peak, self.hw.general_peak_ops_per_cycle)

    def test_rmsnorm_summac_work_exceeds_output_elements(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/rmsnorm_withbaseaddr.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=False,
        )
        summac_op = next(op for op in payload["modes"][MODE_REMAP]["op_breakdown"] if op["op_type"] == "prefill_summac")
        self.assertEqual(summac_op["work_ops"], 32 * 31)
        self.assertGreater(summac_op["work_ops"], 32)

    def test_rmsnorm_summac_writeback_fits_buffer(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/rmsnorm_withbaseaddr.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=False,
        )
        summac_op = next(op for op in payload["modes"][MODE_BASELINE]["op_breakdown"] if op["op_name"] == "op0")
        self.assertEqual(summac_op["bank_timeline"]["write_buffer_bytes"], 128)
        self.assertEqual(summac_op["bank_timeline"]["forced_drain_count"], 0)
        self.assertEqual(summac_op["bank_timeline"]["phase_switch_penalty_cycles"], 28.0)
        self.assertEqual(summac_op["bank_timeline"]["memory_timeline_cycles"], 652.0)
        self.assertEqual(summac_op["latency_cycles"], 652.0)

    def test_rmsnorm_mul_writeback_forces_buffer_drains(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/rmsnorm_withbaseaddr.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=False,
        )
        mul_op = next(op for op in payload["modes"][MODE_BASELINE]["op_breakdown"] if op["op_name"] == "op3")
        self.assertGreater(mul_op["bank_timeline"]["forced_drain_count"], 0)
        self.assertGreater(mul_op["bank_timeline"]["phase_switch_penalty_cycles"], 0.0)
        self.assertGreater(mul_op["bank_timeline"]["memory_timeline_cycles"], 0.0)

    def test_ring_gemm_global_reports_ring_breakdown(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/ring_gemm_bias.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=True,
        )
        gemm_op = next(op for op in payload["modes"][MODE_REMAP]["op_breakdown"] if op["op_type"] == "ring_gemm_fp16_fp16_fp16")
        self.assertEqual(gemm_op["ring_participants"], 4)
        self.assertEqual(gemm_op["ring_a_total_bytes"], gemm_op["local_a_bytes"] * 3)
        self.assertEqual(gemm_op["ring_bandwidth_bytes_per_cycle"], 32.0)
        self.assertGreater(gemm_op["ring_a_transfer_cycles"], 0)
        self.assertEqual(gemm_op["microtile_bytes"], 128)
        self.assertEqual(gemm_op["microtile_k"], 2)
        self.assertEqual(gemm_op["a_buffer_bytes"], 128)
        self.assertEqual(gemm_op["b_buffer_bytes"], 128)
        self.assertEqual(gemm_op["a_reuse_factor"], 4)
        self.assertEqual(gemm_op["b_reuse_factor"], 4)
        self.assertEqual(gemm_op["output_tile_m"], 32)
        self.assertEqual(gemm_op["output_tile_n"], 32)
        self.assertEqual(gemm_op["pe_micro_ops_per_output_tile"], 16)
        self.assertIn("ring_microtile_timeline", gemm_op)
        self.assertIn("ring_tile_timeline", gemm_op)
        self.assertIn("ring_bank_timeline", gemm_op)
        self.assertIn("a_buffer_timeline", gemm_op)
        self.assertIn("b_buffer_timeline", gemm_op)
        self.assertIn("pe_compute_timeline", gemm_op)
        self.assertIn("psum_timeline", gemm_op)
        self.assertIn("output_writeback_timeline", gemm_op)
        self.assertEqual(gemm_op["ring_tile_timeline"]["tile_count"], gemm_op["ring_participants"])
        self.assertEqual(gemm_op["ring_tile_timeline"]["ping_pong_assignment"][:4], ["ping", "pong", "ping", "pong"])
        self.assertEqual(gemm_op["ring_microtile_timeline"]["microtile_count_per_participant"], 4)
        self.assertEqual(gemm_op["ring_microtile_timeline"]["total_compute_microtiles"], 16)
        self.assertEqual(gemm_op["a_buffer_timeline"]["tile_count"], 4)
        self.assertEqual(gemm_op["b_buffer_timeline"]["tile_count"], 8)
        self.assertEqual(gemm_op["pe_compute_timeline"]["micro_op_count"], 256)
        self.assertEqual(gemm_op["psum_timeline"]["tile_count"], 2)
        self.assertEqual(len(gemm_op["psum_timeline"]["tiles"]), 2)
        self.assertEqual(gemm_op["output_writeback_timeline"]["tile_count"], 2)
        self.assertGreaterEqual(
            gemm_op["analytical_model"]["ring_transfer_bound_cycles"],
            gemm_op["ring_a_transfer_cycles"],
        )
        self.assertGreater(gemm_op["per_tile_compute_cycles"], 0)
        self.assertGreater(gemm_op["per_tile_ring_a_transfer_cycles"], 0)
        self.assertEqual(
            gemm_op["ping_pong_pipeline_cycles"],
            gemm_op["ping_pong_startup_cycles"] + gemm_op["ping_pong_steady_cycles"],
        )
        self.assertLessEqual(
            gemm_op["psum_timeline"]["tiles"][0]["transfer_start_cycle"],
            gemm_op["pe_compute_timeline"]["output_tile_completion_cycles"][1],
        )
        self.assertGreater(
            gemm_op["ring_bank_timeline"]["memory_timeline_cycles"],
            gemm_op["ring_microtile_timeline"]["final_write_release_cycle"],
        )

    def test_ring_gemm_cluster_reports_ring_breakdown(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        perf_hw = replace(perf_hw, ring_a_buffer_bits=2048)
        graph = {
            "shape_bindings": {"sequence_length": 128, "hidden_size": 896},
            "params": {"slices_per_cluster": 4, "slices_num": 28},
            "tensors": {
                "a_fp16": {
                    "dtype": "fp16",
                    "shape": {"M": "sequence_length", "K": "hidden_size"},
                    "partition": {"K": {"by_scope": {"cluster": "slices_per_cluster", "global": "slices_num"}}},
                    "base_addr": 0x100000,
                },
                "b_fp16": {
                    "dtype": "fp16",
                    "shape": {"K": "hidden_size", "N": "hidden_size"},
                    "partition": {"N": {"by_scope": {"cluster": "slices_num", "global": "slices_num"}}},
                    "base_addr": 0x101000,
                },
                "gemm_out_fp16": {"base_addr": 0x102000},
            },
            "model": [
                "gemm_out_fp16 = ring_gemm_fp16_fp16_fp16(a_fp16, b_fp16, ring_scope=cluster)",
            ],
        }
        payload = analyze_graph_performance(graph, perf_hw, perf_cfg, include_request_traces=False)
        gemm_op = next(op for op in payload["modes"][MODE_REMAP]["op_breakdown"] if op["op_type"] == "ring_gemm_fp16_fp16_fp16")
        self.assertEqual(gemm_op["ring_participants"], 4)
        self.assertEqual(gemm_op["ring_a_total_bytes"], gemm_op["local_a_bytes"] * 3)
        self.assertEqual(gemm_op["work_ops"], 2 * 128 * 32 * 224 * 4)
        self.assertEqual(gemm_op["microtile_bytes"], 128)
        self.assertEqual(gemm_op["microtile_k"], 2)
        self.assertGreater(gemm_op["ping_pong_pipeline_cycles"], gemm_op["ping_pong_startup_cycles"])
        self.assertEqual(gemm_op["ring_tile_timeline"]["ping_pong_assignment"][:4], ["ping", "pong", "ping", "pong"])
        self.assertGreater(gemm_op["ring_bank_timeline"]["memory_timeline_cycles"], 0.0)
        self.assertEqual(gemm_op["ring_bank_timeline"]["write_buffer_bytes"], 128)
        self.assertEqual(gemm_op["ring_microtile_timeline"]["microtile_count_per_participant"], 448)
        self.assertEqual(len(gemm_op["ring_microtile_timeline"]["compute_end_cycles"]), 1792)
        self.assertEqual(len(gemm_op["ring_tile_timeline"]["tile_compute_end_cycles"]), 1792)
        self.assertEqual(gemm_op["a_buffer_timeline"]["tile_count"], 448)
        self.assertEqual(gemm_op["b_buffer_timeline"]["tile_count"], 448)
        self.assertEqual(gemm_op["pe_compute_timeline"]["micro_op_count"], 28672)
        self.assertEqual(gemm_op["psum_timeline"]["tile_count"], 4)

    def test_ring_gemm_rejects_non_integral_a_buffer_tiling(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        graph = {
            "shape_bindings": {"sequence_length": 48, "hidden_size": 16},
            "params": {"slices_per_cluster": 4, "slices_num": 28},
            "tensors": {
                "a_fp16": {
                    "dtype": "fp16",
                    "shape": {"M": "sequence_length", "K": "hidden_size"},
                    "partition": {"K": {"by_scope": {"cluster": "slices_per_cluster", "global": "slices_num"}}},
                    "base_addr": 0x100000,
                },
                "b_fp16": {
                    "dtype": "fp16",
                    "shape": {"K": "hidden_size", "N": "hidden_size"},
                    "partition": {"N": {"by_scope": {"cluster": "slices_num", "global": "slices_num"}}},
                    "base_addr": 0x101000,
                },
                "gemm_out_fp16": {"base_addr": 0x102000},
            },
            "model": [
                "gemm_out_fp16 = ring_gemm_fp16_fp16_fp16(a_fp16, b_fp16, ring_scope=cluster)",
            ],
        }
        with self.assertRaises(ValueError):
            analyze_graph_performance(graph, perf_hw, perf_cfg, include_request_traces=False)

    def test_request_order_aware_stream_model_distinguishes_round_robin_from_grouped(self):
        perf = PerformanceConfig()
        grouped = [
            make_physical_request(
                self.hw,
                i,
                "e",
                "ag0",
                "A",
                0,
                0 if i < 4 else 1 if i < 8 else 2 if i < 12 else 3,
                i // 2,
                i,
            )
            for i in range(16)
        ]
        round_robin = [
            make_physical_request(self.hw, i, "e", "ag0", "A", 0, i % 4, i // 2, i)
            for i in range(16)
        ]
        grouped_report = _analyze_request_stream(
            ag_id="ag0",
            tensor_name="x",
            edge_name="e",
            role="A",
            mode=MODE_BASELINE,
            requests=grouped,
            hw=self.hw,
            perf=perf,
        )
        round_robin_report = _analyze_request_stream(
            ag_id="ag0",
            tensor_name="x",
            edge_name="e",
            role="A",
            mode=MODE_REMAP_INTERLEAVE,
            requests=round_robin,
            hw=self.hw,
            perf=perf,
        )
        self.assertGreater(grouped_report["adjusted_stream_cycles"], round_robin_report["adjusted_stream_cycles"])
        self.assertGreater(round_robin_report["round_robin_score"], grouped_report["round_robin_score"])
        self.assertGreater(round_robin_report["row_switch_hiding_gain"], grouped_report["row_switch_hiding_gain"])

    def test_hardware_uses_interface_level_memory_timing_defaults(self):
        self.assertEqual(self.hw.request_latency_cycles, 14.0)
        self.assertEqual(self.hw.row_switch_penalty_cycles, 28.0)
        self.assertEqual(self.hw.bank_return_interval_cycles, 2.0)
        self.assertEqual(self.hw.row_hit_latency, 2.0)
        self.assertEqual(self.hw.row_miss_latency, 30.0)
        self.assertEqual(self.hw.row_empty_latency, 14.0)

    def test_request_stream_uses_request_latency_plus_row_switch_penalty(self):
        perf = PerformanceConfig()
        requests = [
            make_physical_request(self.hw, 0, "e", "ag0", "A", 0, 0, 0, 0),
            make_physical_request(self.hw, 1, "e", "ag0", "A", 0, 0, 0, 1),
            make_physical_request(self.hw, 2, "e", "ag0", "A", 0, 0, 1, 2),
            make_physical_request(self.hw, 3, "e", "ag0", "A", 0, 0, 1, 3),
            make_physical_request(self.hw, 4, "e", "ag0", "A", 0, 0, 2, 4),
        ]
        report = _analyze_request_stream(
            ag_id="ag0",
            tensor_name="x",
            edge_name="e",
            role="A",
            mode=MODE_BASELINE,
            requests=requests,
            hw=self.hw,
            perf=perf,
        )
        self.assertEqual(report["bank_stats"]["0"], {"hits": 2, "misses": 2, "empty": 1})
        self.assertEqual(report["raw_bank_max_cycles"], 22.0)
        self.assertEqual(report["row_switch_penalty_cycles"], 56.0)
        self.assertEqual(report["adjusted_stream_cycles"], 78.0)

    def test_request_trace_exposes_physical_address_chain(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/ring_gemm_bias.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=True,
        )
        request = payload["modes"][MODE_BASELINE]["request_trace"][0]
        self.assertIn("logical_addr", request)
        self.assertIn("base_addr", request)
        self.assertIn("address_transform", request)
        self.assertIn("physical_addr", request)
        self.assertEqual(
            request["physical_addr"],
            request["base_addr"] + (request["physical_addr"] - request["base_addr"]),
        )
        roles = {entry["role"] for entry in payload["modes"][MODE_BASELINE]["request_trace"]}
        self.assertNotIn("ring_a_transfer", roles)

    def test_mode_reports_expose_address_transforms(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/ring_gemm_bias.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=True,
        )
        transforms = payload["modes"][MODE_REMAP]["address_transforms"]
        self.assertTrue(transforms)
        self.assertIn("matrix", transforms[0])
        gemm_op = next(op for op in payload["modes"][MODE_REMAP]["op_breakdown"] if op["op_type"] == "ring_gemm_fp16_fp16_fp16")
        self.assertIn("ring_participants", gemm_op)

    def test_missing_base_addr_fails_performance_analysis(self):
        graph = load_graph_file("examples/graphs/ring_gemm_bias.json")
        del graph["tensors"]["a_fp16"]["base_addr"]
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        with self.assertRaises(ValueError):
            analyze_graph_performance(graph, perf_hw, perf_cfg, include_request_traces=False)

    def test_load_performance_config_rejects_legacy_flat_format(self):
        legacy_path = Path("outputs/tests/legacy_perf_config.json").resolve()
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(
            json.dumps(
                {
                    "hardware": {
                        "slave_bits": 5,
                        "bank_bits": 2,
                        "row_bits": 13,
                        "column_bits": 6,
                    },
                    "performance": {
                        "gemm_read_overlap": 0.65,
                    },
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            load_performance_config(str(legacy_path))

    def test_emit_trace_artifacts_generates_ramulator_trace_files(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/ring_gemm_bias.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=True,
        )
        output_path = Path("outputs/tests/test_perf_output.json").resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")

        artifacts = emit_trace_artifacts(payload, str(output_path), perf_hw)
        baseline_trace = Path(artifacts[MODE_BASELINE]["ramulator_trace"])
        self.assertTrue(baseline_trace.exists())
        lines = baseline_trace.read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines)
        self.assertRegex(lines[0], r"^(LD|ST) 0x[0-9a-f]+$")

    def test_run_validation_skips_reference_when_ramulator_missing(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/ring_gemm_bias.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=True,
        )
        output_path = Path("outputs/tests/test_validation_output.json").resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")

        validation = run_validation(
            payload,
            perf_hw,
            perf_cfg,
            str(output_path),
            ramulator_root="H:/definitely_missing_ramulator_root",
        )
        self.assertEqual(validation["reference_results"]["status"], "skipped")
        self.assertIn("validation_overview", validation)
        self.assertIn("reference_mode_summaries", validation)
        self.assertEqual(validation["validation_overview"]["reference_validation_status"], "skipped")
        self.assertTrue(validation["validation_overview"]["baseline_includes_software_relayout"])
        self.assertTrue(validation["validation_summary"]["internal_validation_passed"])

    def test_detect_ramulator_executable_prefers_wsl_build_linux(self):
        temp_root = Path("outputs/tests/test_ramulator_root").resolve()
        build_linux = temp_root / "build-linux"
        build_linux.mkdir(parents=True, exist_ok=True)
        linux_exe = build_linux / "ramulator2"
        linux_exe.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        os.chmod(linux_exe, 0o755)

        detected = _detect_ramulator_executable(str(temp_root))
        self.assertEqual(detected, linux_exe)

    def test_run_validation_uses_reference_results_when_ramulator_present(self):
        perf_hw, perf_cfg = load_performance_config("examples/configs/performance_config.json")
        payload = analyze_graph_performance(
            load_graph_file("examples/graphs/ring_gemm_bias.json"),
            perf_hw,
            perf_cfg,
            include_request_traces=True,
        )
        output_path = Path("outputs/tests/test_validation_reference_output.json").resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")

        fake_executable = Path("third_party/ramulator2/build-linux/ramulator2").resolve()
        completed = subprocess.CompletedProcess(
            args=[str(fake_executable)],
            returncode=0,
            stdout="memory_system_cycles: 4321\ncycles_recorded_core_0: 9999\n",
            stderr="",
        )
        with mock.patch("address_remapping.validation._detect_ramulator_executable", return_value=fake_executable):
            with mock.patch("address_remapping.validation.subprocess.run", return_value=completed) as run_mock:
                validation = run_validation(
                    payload,
                    perf_hw,
                    perf_cfg,
                    str(output_path),
                    ramulator_root="third_party/ramulator2",
                )

        self.assertEqual(validation["reference_results"]["status"], "ok")
        self.assertEqual(validation["validation_summary"]["reference_validation_status"], "ok")
        self.assertEqual(validation["validation_overview"]["reference_validation_status"], "ok")
        self.assertTrue(validation["validation_overview"]["baseline_includes_software_relayout"])
        self.assertEqual(
            set(validation["reference_mode_summaries"]),
            {"baseline", "remap", "remap_interleave"},
        )
        self.assertEqual(
            validation["reference_results"]["per_mode"][MODE_BASELINE]["memory_cycles_reference"],
            4321,
        )
        self.assertEqual(run_mock.call_count, 3)

    def test_parse_reference_cycles_prefers_memory_system_cycles(self):
        stdout = "\n".join(
            [
                "memory_access_cycles_recorded_core_0: 61",
                "cycles_recorded_core_0: 216815",
                "memory_system_cycles: 81306",
            ]
        )
        self.assertEqual(_parse_reference_cycles(stdout), 81306)


if __name__ == "__main__":
    unittest.main()
