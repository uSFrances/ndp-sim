from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


GP_ROOT = Path(__file__).resolve().parent
ROOT = GP_ROOT.parent
EXECPLAN_SRC = ROOT / "model_execplan" / "src"
for path in (GP_ROOT, ROOT, EXECPLAN_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from decode_ops import (  # noqa: E402
    SUPPORTED_DECODE_OPERATORS,
    build_decode_golden_cases_optimized,
    gemv_fp32_accumulate,
    load_decode_config,
    validate_decode_config,
)
from assemble_decode_package import (  # noqa: E402
    assemble_decode_package,
    resolve_manifest_path,
)
from generate_decode_execplan_inputs import build_execplan_operators  # noqa: E402
from single_op_data.decode_passthrough import (  # noqa: E402
    split_head_vectors,
    split_vector,
)
from single_op_data.relayout_gemv import _split_weight_streams  # noqa: E402
from tensor_io import (  # noqa: E402
    load_golden_tensor,
    save_golden_tensor,
    save_install_tensor,
)
from execution_plan_generator.register_mapping import (  # noqa: E402
    build_masked_register_writes,
    load_register_mapping,
)


class DecodeGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_decode_config(GP_ROOT / "config.json")
        cls.cases = build_decode_golden_cases_optimized(cls.config)
        cls.by_instance = {c.instance_id: c for c in cls.cases}

    def test_all_operator_types_covered(self) -> None:
        registry = {spec.name for spec in SUPPORTED_DECODE_OPERATORS}
        actual = {c.spec.name for c in self.cases}
        self.assertTrue(actual <= registry)
        for required in (
            "prefill_remote_summac_fp32MN_fp32MN",
            "prefill_remote_summac_4slice_fp32MN_fp32MN",
            "prefill_remote_sum_qkt_4slice_fp32MN_fp32MN",
            "decode_gemv_local_qkt",
            "decode_gemv_local",
        ):
            self.assertIn(required, actual)

    def test_layer_has_multiple_gemv_instances(self) -> None:
        gemv_ring_instances = [c for c in self.cases if c.spec.name == "decode_gemv_ring"]
        self.assertGreater(len(gemv_ring_instances), 3,
                           "decode layer should use GEMV for Q, K, V, O, gate, up, out")

    def test_full_dimensions_and_head_padding(self) -> None:
        self.assertEqual(self.config["hidden_size"], 1536)
        self.assertEqual(self.config["hidden_size_padded"], 1792)
        self.assertEqual(self.config["num_attention_heads"], 12)
        self.assertEqual(self.config["num_attention_heads_padded"], 14)
        self.assertEqual(self.config["slice_per_head"], 4)
        self.assertEqual(self.config["heads_per_group"], 1)
        self.assertEqual(self.config["attention_passes"], 2)
        self.assertEqual(self.config["heads_per_pass"], 7)
        self.assertEqual(self.config["used_slices"], 28)
        self.assertEqual(self.config["sequence_length"], 32)
        case = self.by_instance["attn_remote_summac"]
        self.assertEqual(case.inputs[0].shape, (8, 224, 1))
        self.assertEqual(case.output.shape, (8, 1, 1))

    def test_decode_shapes(self) -> None:
        def shapes(inst_id):
            c = self.by_instance[inst_id]
            return ([t.shape for t in c.inputs], c.output.shape)
        self.assertEqual(shapes("attn_max_A"), ([(32, 7, 1)], (1, 7, 1)))
        self.assertEqual(shapes("ffn_mul_cast"), ([(1792, 1, 1), (1792, 1, 1)], (1792, 1, 1)))
        self.assertEqual(shapes("q_gemv_A"), ([(1792, 896, 1), (1792, 1, 1)], (896, 1, 1)))
        self.assertEqual(shapes("q_gemv_B"), ([(1792, 896, 1), (1792, 1, 1)], (896, 1, 1)))
        self.assertEqual(shapes("attn_sv_A"), ([(128, 32, 7), (32, 7, 1)], (32, 4, 7)))
        self.assertEqual(shapes("attn_qkt_A"), ([(128, 32, 7), (128, 7, 1)], (32, 4, 7)))
        self.assertEqual(shapes("out_gemv_A"), ([(896, 1792, 1), (896, 1, 1)], (1792, 1, 1)))
        self.assertEqual(shapes("ffn_silu"), ([(8960, 1, 1)], (8960, 1, 1)))
        self.assertEqual(shapes("attn_mul_scale"), ([(1792, 1, 1), (1, 1, 1)], (1792, 1, 1)))

    def test_padded_heads_and_hidden_tail_are_zero(self) -> None:
        q_b = self.by_instance["q_gemv_B"].output.reshape(-1)
        np.testing.assert_array_equal(q_b[640:], np.zeros(256, dtype=q_b.dtype))
        final = self.by_instance["ffn_add_residual"].output.reshape(-1)
        np.testing.assert_array_equal(final[1536:], np.zeros(256, dtype=final.dtype))

    def test_attention_length_cannot_silently_become_33(self) -> None:
        invalid = dict(self.config)
        invalid["decode_attention_length"] = 33
        with self.assertRaisesRegex(ValueError, "decode_attention_length"):
            validate_decode_config(invalid)

    def test_small_gemv_uses_explicit_k_n_convention(self) -> None:
        weight_kn = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
        vector_k = np.asarray([7, 8], dtype=np.float16)
        actual = gemv_fp32_accumulate(weight_kn, vector_k)
        np.testing.assert_array_equal(actual, np.asarray([39, 54, 69], dtype=np.float32))


class DecodeSlicingTests(unittest.TestCase):
    def test_hidden_vector_is_contiguously_partitioned(self) -> None:
        vector = np.arange(896, dtype=np.float32).reshape(896, 1, 1, order="F")
        chunks = split_vector(vector, 28)
        self.assertEqual({chunk.size for chunk in chunks}, {32})
        np.testing.assert_array_equal(np.concatenate(chunks), np.arange(896, dtype=np.float32))

    def test_attention_is_partitioned_head_then_slice(self) -> None:
        values = np.arange(32 * 7, dtype=np.float32).reshape(32, 7, 1, order="F")
        chunks = split_head_vectors(values, heads=7, slices_per_head=4)
        self.assertEqual(len(chunks), 28)
        for head in range(7):
            reconstructed = np.concatenate(chunks[head * 4 : (head + 1) * 4])
            np.testing.assert_array_equal(reconstructed, values[:, head, 0])

    def test_gemv_b_and_bprime_are_equal_halves(self) -> None:
        linearized = np.arange(128, dtype=np.float16)
        b, bp = _split_weight_streams(linearized)
        self.assertEqual(b.size, 64)
        self.assertEqual(bp.size, 64)
        # Ping-pong split: even 64-value chunks -> B, odd -> B'.
        np.testing.assert_array_equal(b, np.concatenate([linearized[0:64]]))
        np.testing.assert_array_equal(bp, np.concatenate([linearized[64:128]]))


class TensorIoTests(unittest.TestCase):
    def test_golden_fortran_round_trip(self) -> None:
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4, order="F")
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = save_golden_tensor(tmp, "roundtrip", tensor)
            np.testing.assert_array_equal(load_golden_tensor(path), tensor)

    def test_128bit_text_reverses_lanes(self) -> None:
        values = np.asarray([1, 2, 3, 4], dtype=np.float32)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = save_install_tensor(tmp, "matrix_A_linearized_128bit.bin", values)
            line = path.with_suffix(".txt").read_text(encoding="ascii").strip()
        expected = "".join(
            f"{bits:032b}"
            for bits in values[::-1].view(np.uint32)
        )
        self.assertEqual(line, expected)
        self.assertEqual(len(line), 128)


class ExecplanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_decode_config(GP_ROOT / "config.json")

    def test_graphs_match_operator_registry(self) -> None:
        operators = build_execplan_operators(self.config)
        self.assertEqual(
            [(op["id"], op["type"]) for op in operators],
            [(spec.op_id, spec.name) for spec in SUPPORTED_DECODE_OPERATORS],
        )
        ring = next(op for op in operators if op["type"] == "decode_gemv_ring")
        self.assertEqual(
            ring["inputs"]["A"]["shape"],
            [1, 1, "hidden_size_padded//used_slices"],
        )
        self.assertEqual(
            ring["inputs"]["B"]["shape"],
            [1, "hidden_size_padded//used_slices", "hidden_size_padded"],
        )

    def test_standalone_constant_register_is_writable(self) -> None:
        db = load_register_mapping(
            ROOT / "model_execplan" / "config" / "register_map_with_groups1.csv",
            ROOT / "model_execplan" / "config" / "config_output.csv",
        )
        key = "ga_pe0.general_array.PE_array.PE.inport1.constant"
        binding = db.get_field(key)
        self.assertIsNotNone(binding)
        writes = build_masked_register_writes(binding, 0x3A124925)  # type: ignore[arg-type]
        self.assertEqual(len(writes), 1)
        write = next(iter(writes.values()))
        self.assertEqual(write.value, 0x3A124925)
        self.assertEqual(write.mask, 0xFFFFFFFF)


class PackageAssemblerTests(unittest.TestCase):
    def test_assemble_is_complete_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            temp_root = Path(tmp)
            package_root = temp_root / "package"
            data_root = temp_root / "data"
            execplan = package_root / "install" / "execplan.txt"
            config = package_root / "install" / "cfg_pkg" / "op0.txt"
            tensor = data_root / "op0" / "slice00" / "matrix_A_linearized_128bit.txt"
            for path, content in (
                (execplan, "0" * 128 + "\n"),
                (config, "1" * 128 + "\n"),
                (tensor, "01" * 64 + "\n"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="ascii")

            sca_cfg = package_root / "sca_cfg.json"
            sca_cfg.parent.mkdir(parents=True, exist_ok=True)
            sca_cfg.write_text(
                json.dumps(
                    {
                        "ExecutionPlan": {"path": "install/execplan.txt"},
                        "Config": {"path": "install/cfg_pkg/op0.txt"},
                        "Tensor": {
                            "path": "install/op0/slice00/matrix_A_linearized_128bit.txt"
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest_path = assemble_decode_package(sca_cfg, data_root, package_root)
            packaged_tensor = package_root / "install" / "op0" / "slice00" / tensor.name
            self.assertEqual(packaged_tensor.read_bytes(), tensor.read_bytes())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["referenced_files"], 3)
            self.assertEqual(manifest["tensor_files"], 1)
            self.assertEqual(manifest["copied_tensor_files"], 1)

            assemble_decode_package(sca_cfg, data_root, package_root)
            second = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(second["copied_tensor_files"], 0)
            self.assertEqual(second["unchanged_tensor_files"], 1)

    def test_manifest_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            with self.assertRaisesRegex(ValueError, "unsafe SCA path"):
                resolve_manifest_path(Path(tmp), "../outside.txt")


if __name__ == "__main__":
    unittest.main()
