from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .address_planner import AddressPlanner
from .instruction_generator import InstructionGenerator
from .json_loader import load_execution_plan_json
from .models import AddressPlan, ExecutionPlanArtifact, ExecutionPlanInput, OperatorTemplate
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
        artifact = self._instruction_generator.generate(execution_input, address_plan, templates)

        return PipelineResult(
            execution_input=execution_input,
            address_plan=address_plan,
            templates=templates,
            artifact=artifact,
        )
