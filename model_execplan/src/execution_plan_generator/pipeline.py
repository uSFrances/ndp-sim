from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from .address_planner import AddressPlanner
from .config_stream_decoder import (
    _load_template_from_bitstream_file,
    decode_initial_register_state,
)
from .control_registers import _MAPPING_REVIEW_CACHE, compute_control_register_updates
from .instruction_generator import InstructionGenerator
from .json_loader import load_execution_plan_json
from .models import AddressPlan, ExecutionPlanArtifact, ExecutionPlanInput, OperatorTemplate
from .output_writer import _patch_emulator_operator_json_payload
from .template_manager import OperatorTemplateManager


@dataclass(frozen=True)
class PipelineResult:
    execution_input: ExecutionPlanInput
    address_plan: AddressPlan
    templates: dict[str, OperatorTemplate]
    artifact: ExecutionPlanArtifact


class ExecutionPlanPipeline:
    """Top-level orchestration pipeline.

    Current status:
    - JSON parsing and validation: implemented
    - Address planning: implemented
    - Template adjustment: implemented for template loading, initial register decode and size check
    - Instruction generation: implemented for Write_Reg base address and gated control writes
    """

    def __init__(
        self,
        address_planner: AddressPlanner | None = None,
        template_manager: OperatorTemplateManager | None = None,
        instruction_generator: InstructionGenerator | None = None,
    ) -> None:
        self._address_planner = address_planner or AddressPlanner()
        self._template_manager = template_manager or OperatorTemplateManager()
        self._instruction_generator = instruction_generator or InstructionGenerator()

    def build_from_json(self, json_path: str | Path) -> PipelineResult:
        execution_input = load_execution_plan_json(json_path)
        templates = self._template_manager.adjust_for_operator(execution_input)

        # First pass: plan addresses with whatever config_lengths are available.
        config_lengths_by_op: dict[str, int] = {}
        sfu_config_lengths_by_op: dict[str, int] = {}
        for op in execution_input.operators:
            template = templates.get(op.op_id)
            config_lengths_by_op[op.op_id] = int((template.config_length if template else 0) or 0)
            sfu_config_lengths_by_op[op.op_id] = int((template.sfu_config_length if template else 0) or 0)
        address_plan = self._address_planner.plan(
            execution_input,
            config_lengths_by_op=config_lengths_by_op,
            sfu_config_lengths_by_op=sfu_config_lengths_by_op,
        )

        # Regenerate bitstreams into model_execplan/config/ so they are always
        # in sync with the current execution plan.  This bakes control-register
        # updates into the baseline and reduces Write_Reg instructions.
        templates = self._regenerate_bitstreams(execution_input, address_plan, templates)

        # After regeneration config_length may have changed (e.g. when the
        # config folder was deleted and the initial value was zero).  Re-plan
        # addresses so that config data gets a valid base address.
        for op in execution_input.operators:
            template = templates.get(op.op_id)
            config_lengths_by_op[op.op_id] = int((template.config_length if template else 0) or 0)
        address_plan = self._address_planner.plan(
            execution_input,
            config_lengths_by_op=config_lengths_by_op,
            sfu_config_lengths_by_op=sfu_config_lengths_by_op,
        )

        artifact = self._instruction_generator.generate(execution_input, address_plan, templates)

        return PipelineResult(
            execution_input=execution_input,
            address_plan=address_plan,
            templates=templates,
            artifact=artifact,
        )

    def _regenerate_bitstreams(
        self,
        execution_input: ExecutionPlanInput,
        address_plan: AddressPlan,
        templates: dict[str, OperatorTemplate],
    ) -> dict[str, OperatorTemplate]:
        """Patch each operator JSON with control-register updates, regenerate the
        bitstream directly into ``model_execplan/config/<op_type>/`` (overwriting
        the original files), and replace ``original_register_values`` with decoded
        values from the new bitstream.

        The regenerated 128-bit bitstream and patched JSON are saved side by side
        so that downstream consumers (``write_install_manifest``, manual inspection)
        always see the latest state.
        """
        project_root = Path(__file__).resolve().parents[2]
        repo_root = project_root.parent
        op_json_root = repo_root / "jsons"
        op_config_root = project_root / "config"
        bitstream_script = str(repo_root / "bitstream" / "main.py")

        updated_templates = dict(templates)
        patched_count = 0

        # Collect unique operator types so each config dir is regenerated once.
        seen_types: set[str] = set()
        for op in execution_input.operators:
            template = updated_templates.get(op.op_id)
            if template is None:
                continue
            if op.op_type in seen_types:
                continue
            seen_types.add(op.op_type)

            source_json = op_json_root / f"{op.op_type}.json"
            if not source_json.is_file():
                print(
                    f"[pipeline] JSON template not found for {op.op_type}: "
                    f"{source_json}"
                )
                continue

            print(f"[pipeline] regenerating bitstream for {op.op_type} ...")
            op_payload = _load_json_object(source_json)
            _patch_emulator_operator_json_payload(
                payload=op_payload,
                operator=op,
                address_plan=address_plan,
                use_global_addrs=True,
            )

            op_dir = op_config_root / op.op_type
            op_dir.mkdir(parents=True, exist_ok=True)

            # Write patched JSON alongside the regenerated outputs.
            patched_json = op_dir / f"{op.op_type}.json"
            patched_json.write_text(
                json.dumps(op_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            # Run bitstream tool directly into op_dir — no temp dir, no manual copy.
            cmd = [
                sys.executable,
                bitstream_script,
                "--visualize-placement",
                "-c", str(patched_json),
                "-o", str(op_dir),
                "-q",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=str(repo_root),
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            )
            if result.returncode != 0:
                print(
                    f"[pipeline] bitstream regeneration failed for "
                    f"{op.op_type} (rc={result.returncode}):\n"
                    f"  stdout: {result.stdout.strip()}\n"
                    f"  stderr: {result.stderr.strip()}"
                )
                continue

            # Clear the mapping cache for this op_type so the instruction
            # generator reloads from the regenerated mapping_review.json.
            _MAPPING_REVIEW_CACHE.pop(op.op_type, None)

            # Re-load the freshly written parsed bitstream so that every template
            # sharing this op_type uses the updated register baseline.
            parsed_path = op_dir / "parsed_bitstream.txt"
            if not parsed_path.is_file():
                print(
                    f"[pipeline] parsed bitstream missing for {op.op_type}"
                )
                continue

            raw: dict[str, object] = {"bitstream_file": str(parsed_path)}
            config_stream = _load_template_from_bitstream_file(
                raw,
                op_dir,
                self._template_manager._register_db,
            )
            decoded = decode_initial_register_state(
                config_stream,
                self._template_manager._register_db,
            )

            # Apply the updated register baseline to every operator of this type.
            # Also recompute control_register_values so that instance mappings
            # reflect the regenerated physical assignment.
            # config_length is measured in 64-bit words; the 128b file gives
            # a more accurate count because it avoids chunk-boundary padding
            # artefacts that can appear in the 64b representation.
            regen_config_length = _count_non_empty_lines(
                op_dir / f"{op.op_type}_bitstream_128b.bin"
            ) * 2
            for op2 in execution_input.operators:
                if op2.op_type != op.op_type:
                    continue
                tpl = updated_templates.get(op2.op_id)
                if tpl is None:
                    continue
                new_control_values = compute_control_register_updates(
                    operator=op2,
                    template=tpl,
                    address_plan=address_plan,
                    apply_instance_mapping=True,
                )
                updated_templates[op2.op_id] = replace(
                    tpl,
                    original_register_values=decoded.register_values,
                    enabled_register_addresses=frozenset(
                        decoded.enabled_register_addresses
                    ),
                    config_bitstream_path=str(
                        op_dir / f"{op.op_type}_bitstream_128b.bin"
                    ),
                    control_register_values=new_control_values,
                    config_length=regen_config_length,
                )

            patched_count += 1

        if patched_count:
            print(
                f"[pipeline] Regenerated bitstream + JSON for {patched_count} "
                f"operator type(s) under {op_config_root}."
            )

        return updated_templates


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _count_non_empty_lines(file_path: Path) -> int:
    if not file_path.is_file():
        return 0
    count = 0
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
