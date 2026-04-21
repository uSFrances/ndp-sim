from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class InputSourceType(str, Enum):
    EXTERNAL = "external"
    OPERATOR = "operator"


@dataclass(frozen=True)
class InputSource:
    source_type: InputSourceType
    operator_id: str | None = None


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, int, int]
    dtype: str = "fp32"
    source: InputSource | None = None
    remapping: tuple[int, ...] | None = None


@dataclass(frozen=True)
class OperatorSpec:
    op_id: str
    op_type: str
    used_slices: int
    inputs: dict[str, TensorSpec]
    output: TensorSpec

    def enabled_slice_ids(self) -> list[int]:
        return [slice_id for slice_id in range(28) if (self.used_slices >> slice_id) & 1]

    def used_slice_count(self) -> int:
        return len(self.enabled_slice_ids())


@dataclass(frozen=True)
class ExecutionPlanInput:
    used_slices: int
    operators: list[OperatorSpec]

    def enabled_slice_ids(self) -> list[int]:
        return [slice_id for slice_id in range(28) if (self.used_slices >> slice_id) & 1]

    def used_slice_count(self) -> int:
        return len(self.enabled_slice_ids())


@dataclass(frozen=True)
class AddressAssignment:
    tensor_name: str
    base_address: int
    per_slice_addresses: dict[int, int]
    size_bytes: int
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class AddressPlan:
    assignments: dict[str, AddressAssignment] = field(default_factory=dict)
    operator_io_to_tensor: dict[str, str] = field(default_factory=dict)
    operator_config_base_addresses: dict[str, int] = field(default_factory=dict)
    operator_config_lengths: dict[str, int] = field(default_factory=dict)
    operator_sfu_config_base_addresses: dict[str, int] = field(default_factory=dict)
    operator_sfu_config_lengths: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorTemplate:
    op_type: str
    config_length: int | None = None
    ddr_config_addr: int | None = None
    config_bitstream_addr: int | None = None
    config_bitstream_path: str | None = None
    initial_io_sizes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    target_io_sizes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    initial_size: tuple[int, int, int] | None = None
    target_size: tuple[int, int, int] | None = None
    should_update_control_registers: bool = False
    original_register_values: dict[int, int] = field(default_factory=dict)
    enabled_register_addresses: frozenset[int] = frozenset()
    control_register_values: dict[str, int] = field(default_factory=dict)
    config_sfu_type: str | None = None
    sfu_config_length: int | None = None


@dataclass(frozen=True)
class ExecutionPlanArtifact:
    commands: list[int] = field(default_factory=list)
    command_explanations: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
