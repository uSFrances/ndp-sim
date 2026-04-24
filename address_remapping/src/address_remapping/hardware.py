from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional


DTYPE_BITS = {
    "fp16": 16,
    "bf16": 16,
    "fp32": 32,
    "int8": 8,
    "uint8": 8,
    "int16": 16,
    "uint16": 16,
    "int32": 32,
    "uint32": 32,
}


@dataclass(frozen=True)
class AddressSpaceSpec:
    slave_bits: int = 5
    bank_bits: int = 2
    row_bits: int = 13
    column_bits: int = 6
    subword_bits: int = 4
    address_unit_bits: int = 8
    block_bits: int = 128

    def to_dict(self) -> Dict[str, object]:
        return {
            "slave_bits": self.slave_bits,
            "bank_bits": self.bank_bits,
            "row_bits": self.row_bits,
            "column_bits": self.column_bits,
            "subword_bits": self.subword_bits,
            "address_unit_bits": self.address_unit_bits,
            "block_bits": self.block_bits,
        }


@dataclass(frozen=True)
class ClockDomainSpec:
    slice_frequency_hz: float = 1_000_000_000.0
    memory_frequency_hz: float = 500_000_000.0

    @property
    def memory_to_slice_cycle_ratio(self) -> float:
        return self.slice_frequency_hz / self.memory_frequency_hz

    def to_dict(self) -> Dict[str, object]:
        return {
            "slice_frequency_hz": self.slice_frequency_hz,
            "memory_frequency_hz": self.memory_frequency_hz,
            "memory_to_slice_cycle_ratio": self.memory_to_slice_cycle_ratio,
        }


@dataclass(frozen=True)
class DramSpec:
    banks_per_slice: int = 4
    request_fifo_depth: int = 8
    bank_bandwidth_bits_per_cycle: int = 128
    request_latency_cycles: float = 7.0
    row_switch_penalty_cycles: float = 14.0
    bank_return_interval_cycles: float = 1.0
    tRCD: float = 14.0
    tRP: float = 14.0
    tCL: float = 14.0
    tBL: float = 4.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "banks_per_slice": self.banks_per_slice,
            "request_fifo_depth": self.request_fifo_depth,
            "bank_bandwidth_bits_per_cycle": self.bank_bandwidth_bits_per_cycle,
            "request_latency_cycles": self.request_latency_cycles,
            "row_switch_penalty_cycles": self.row_switch_penalty_cycles,
            "bank_return_interval_cycles": self.bank_return_interval_cycles,
            "tRCD": self.tRCD,
            "tRP": self.tRP,
            "tCL": self.tCL,
            "tBL": self.tBL,
        }


@dataclass(frozen=True)
class ArraySpec:
    rows: int
    cols: int
    k_per_cycle: int = 1
    mac_ops: int = 1

    @property
    def peak_ops_per_cycle(self) -> int:
        return self.rows * self.cols * self.k_per_cycle * self.mac_ops

    def to_dict(self) -> Dict[str, object]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "k_per_cycle": self.k_per_cycle,
            "mac_ops": self.mac_ops,
            "peak_ops_per_cycle": self.peak_ops_per_cycle,
        }


@dataclass(frozen=True)
class ComputeCoreSpec:
    gemm_core: ArraySpec = field(default_factory=lambda: ArraySpec(rows=8, cols=8, k_per_cycle=2, mac_ops=2))
    general_core: ArraySpec = field(default_factory=lambda: ArraySpec(rows=4, cols=4))

    def to_dict(self) -> Dict[str, object]:
        return {
            "gemm_core": self.gemm_core.to_dict(),
            "general_core": self.general_core.to_dict(),
        }


@dataclass(frozen=True)
class PerformanceConfig:
    gemm_read_overlap: float = 0.65
    general_read_overlap: float = 0.20
    writeback_overlap: float = 0.50
    controller_write_queue_depth: int = 16
    slice_write_buffer_depth: int = 32
    scheduler_epoch_cycles: int = 64

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "PerformanceConfig":
        if not data:
            return cls()
        if "overlap" not in data:
            raise ValueError("performance config must define an 'overlap' section.")
        if not isinstance(data.get("overlap"), Mapping):
            raise ValueError("performance.overlap must be a mapping.")
        overlap = dict(data["overlap"])
        controller = dict(data.get("controller", {}))
        return cls(
            gemm_read_overlap=float(overlap.get("gemm_read", 0.65)),
            general_read_overlap=float(overlap.get("general_read", 0.20)),
            writeback_overlap=float(overlap.get("writeback", 0.50)),
            controller_write_queue_depth=int(controller.get("write_queue_depth", 16)),
            slice_write_buffer_depth=int(controller.get("slice_write_buffer_depth", 32)),
            scheduler_epoch_cycles=int(controller.get("scheduler_epoch_cycles", 64)),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "overlap": {
                "gemm_read": self.gemm_read_overlap,
                "general_read": self.general_read_overlap,
                "writeback": self.writeback_overlap,
            },
            "controller": {
                "write_queue_depth": self.controller_write_queue_depth,
                "slice_write_buffer_depth": self.slice_write_buffer_depth,
                "scheduler_epoch_cycles": self.scheduler_epoch_cycles,
            },
        }


@dataclass(frozen=True)
class HardwareSpec:
    address_space: AddressSpaceSpec = field(default_factory=AddressSpaceSpec)
    clocks: ClockDomainSpec = field(default_factory=ClockDomainSpec)
    dram: DramSpec = field(default_factory=DramSpec)
    compute: ComputeCoreSpec = field(default_factory=ComputeCoreSpec)
    ag_issue_rate: int = 1
    write_buffer_bits: int = 1024
    ring_a_buffer_bits: int = 1024
    ring_b_buffer_bits: int = 1024

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "HardwareSpec":
        if not data:
            return cls()

        required_sections = ("address_space", "clocks", "dram", "compute")
        missing = [section for section in required_sections if section not in data]
        if missing:
            raise ValueError(
                "hardware config must define sections: address_space, clocks, dram, compute; "
                f"missing {', '.join(missing)}."
            )
        if not isinstance(data.get("address_space"), Mapping):
            raise ValueError("hardware.address_space must be a mapping.")
        if not isinstance(data.get("clocks"), Mapping):
            raise ValueError("hardware.clocks must be a mapping.")
        if not isinstance(data.get("dram"), Mapping):
            raise ValueError("hardware.dram must be a mapping.")
        if not isinstance(data.get("compute"), Mapping):
            raise ValueError("hardware.compute must be a mapping.")

        address_data = dict(data["address_space"])
        clocks_data = dict(data["clocks"])
        dram_data = dict(data["dram"])
        compute_data = dict(data["compute"])
        if "gemm_core" not in compute_data or "general_core" not in compute_data:
            raise ValueError("hardware.compute must define 'gemm_core' and 'general_core'.")
        if not isinstance(compute_data.get("gemm_core"), Mapping):
            raise ValueError("hardware.compute.gemm_core must be a mapping.")
        if not isinstance(compute_data.get("general_core"), Mapping):
            raise ValueError("hardware.compute.general_core must be a mapping.")
        gemm_data = dict(compute_data["gemm_core"])
        general_data = dict(compute_data["general_core"])

        address_space = AddressSpaceSpec(
            slave_bits=int(address_data.get("slave_bits", 5)),
            bank_bits=int(address_data.get("bank_bits", 2)),
            row_bits=int(address_data.get("row_bits", 13)),
            column_bits=int(address_data.get("column_bits", 6)),
            subword_bits=int(address_data.get("subword_bits", 4)),
            address_unit_bits=int(address_data.get("address_unit_bits", 8)),
            block_bits=int(address_data.get("block_bits", 128)),
        )
        clocks = ClockDomainSpec(
            slice_frequency_hz=float(clocks_data.get("slice_frequency_hz", 1_000_000_000.0)),
            memory_frequency_hz=float(clocks_data.get("memory_frequency_hz", 500_000_000.0)),
        )
        dram = DramSpec(
            banks_per_slice=int(dram_data.get("banks_per_slice", 4)),
            request_fifo_depth=int(dram_data.get("request_fifo_depth", 8)),
            bank_bandwidth_bits_per_cycle=int(dram_data.get("bank_bandwidth_bits_per_cycle", 128)),
            request_latency_cycles=float(dram_data.get("request_latency_cycles", 7.0)),
            row_switch_penalty_cycles=float(dram_data.get("row_switch_penalty_cycles", 14.0)),
            bank_return_interval_cycles=float(dram_data.get("bank_return_interval_cycles", 1.0)),
            tRCD=float(dram_data.get("tRCD", 14.0)),
            tRP=float(dram_data.get("tRP", 14.0)),
            tCL=float(dram_data.get("tCL", 14.0)),
            tBL=float(dram_data.get("tBL", 4.0)),
        )
        compute = ComputeCoreSpec(
            gemm_core=ArraySpec(
                rows=int(gemm_data.get("rows", 8)),
                cols=int(gemm_data.get("cols", 8)),
                k_per_cycle=int(gemm_data.get("k_per_cycle", 2)),
                mac_ops=int(gemm_data.get("mac_ops", 2)),
            ),
            general_core=ArraySpec(
                rows=int(general_data.get("rows", 4)),
                cols=int(general_data.get("cols", 4)),
                k_per_cycle=int(general_data.get("k_per_cycle", 1)),
                mac_ops=int(general_data.get("mac_ops", 1)),
            ),
        )
        return cls(
            address_space=address_space,
            clocks=clocks,
            dram=dram,
            compute=compute,
            ag_issue_rate=int(data.get("ag_issue_rate", 1)),
            write_buffer_bits=int(data.get("write_buffer_bits", 1024)),
            ring_a_buffer_bits=int(data.get("ring_a_buffer_bits", data.get("ping_pong_buffer_bits", 1024))),
            ring_b_buffer_bits=int(data.get("ring_b_buffer_bits", 1024)),
        )

    @property
    def slave_bits(self) -> int:
        return self.address_space.slave_bits

    @property
    def bank_bits(self) -> int:
        return self.address_space.bank_bits

    @property
    def row_bits(self) -> int:
        return self.address_space.row_bits

    @property
    def column_bits(self) -> int:
        return self.address_space.column_bits

    @property
    def subword_bits(self) -> int:
        return self.address_space.subword_bits

    @property
    def address_unit_bits(self) -> int:
        return self.address_space.address_unit_bits

    @property
    def block_bits(self) -> int:
        return self.address_space.block_bits

    @property
    def remap_bits(self) -> int:
        return self.slave_bits + self.bank_bits + self.row_bits + self.column_bits

    @property
    def slice_frequency_hz(self) -> float:
        return self.clocks.slice_frequency_hz

    @property
    def memory_frequency_hz(self) -> float:
        return self.clocks.memory_frequency_hz

    @property
    def memory_to_slice_cycle_ratio(self) -> float:
        return self.clocks.memory_to_slice_cycle_ratio

    @property
    def bank_count_per_slice(self) -> int:
        return self.dram.banks_per_slice

    @property
    def request_fifo_depth(self) -> int:
        return self.dram.request_fifo_depth

    @property
    def bank_bandwidth_bits_per_cycle(self) -> int:
        return self.dram.bank_bandwidth_bits_per_cycle

    @property
    def request_latency_cycles(self) -> float:
        return self.dram.request_latency_cycles * self.memory_to_slice_cycle_ratio

    @property
    def row_switch_penalty_cycles(self) -> float:
        return self.dram.row_switch_penalty_cycles * self.memory_to_slice_cycle_ratio

    @property
    def bank_return_interval_cycles(self) -> float:
        return self.dram.bank_return_interval_cycles * self.memory_to_slice_cycle_ratio

    @property
    def tRCD(self) -> float:
        return self.dram.tRCD

    @property
    def tRP(self) -> float:
        return self.dram.tRP

    @property
    def tCL(self) -> float:
        return self.dram.tCL

    @property
    def tBL(self) -> float:
        return self.dram.tBL

    @property
    def row_hit_latency(self) -> float:
        return self.bank_return_interval_cycles

    @property
    def row_miss_latency(self) -> float:
        return self.bank_return_interval_cycles + self.row_switch_penalty_cycles

    @property
    def row_empty_latency(self) -> float:
        return self.request_latency_cycles

    @property
    def gemm_peak_ops_per_cycle(self) -> int:
        return self.compute.gemm_core.peak_ops_per_cycle

    @property
    def general_peak_ops_per_cycle(self) -> int:
        return self.compute.general_core.peak_ops_per_cycle

    @property
    def peak_memory_bandwidth_bytes_per_cycle(self) -> float:
        bank_bytes = self.bank_bandwidth_bits_per_cycle / 8.0
        aggregate_bank_cycle_bytes = self.bank_count_per_slice * bank_bytes
        return aggregate_bank_cycle_bytes / self.memory_to_slice_cycle_ratio

    @property
    def write_buffer_bytes(self) -> int:
        return self.write_buffer_bits // 8

    @property
    def ring_a_buffer_bytes(self) -> int:
        return self.ring_a_buffer_bits // 8

    @property
    def ring_b_buffer_bytes(self) -> int:
        return self.ring_b_buffer_bits // 8

    def dtype_bits(self, dtype: str) -> int:
        if dtype not in DTYPE_BITS:
            raise ValueError(f"Unsupported dtype '{dtype}'.")
        return DTYPE_BITS[dtype]

    def block_elements(self, dtype: str) -> int:
        bits = self.dtype_bits(dtype)
        if self.block_bits % bits != 0:
            raise ValueError(
                f"Block size {self.block_bits} is not divisible by dtype {dtype} ({bits} bits)."
            )
        elems = self.block_bits // bits
        if elems <= 0 or elems & (elems - 1):
            raise ValueError(
                f"Block element count {elems} for dtype {dtype} is not a power of two."
            )
        return elems

    def to_dict(self) -> Dict[str, object]:
        return {
            "address_space": self.address_space.to_dict(),
            "clocks": self.clocks.to_dict(),
            "dram": self.dram.to_dict(),
            "compute": self.compute.to_dict(),
            "ag_issue_rate": self.ag_issue_rate,
            "write_buffer_bits": self.write_buffer_bits,
            "ring_a_buffer_bits": self.ring_a_buffer_bits,
            "ring_b_buffer_bits": self.ring_b_buffer_bits,
            "derived": {
                "remap_bits": self.remap_bits,
                "row_hit_latency_cycles": self.row_hit_latency,
                "row_miss_latency_cycles": self.row_miss_latency,
                "row_empty_latency_cycles": self.row_empty_latency,
                "request_latency_cycles": self.request_latency_cycles,
                "row_switch_penalty_cycles": self.row_switch_penalty_cycles,
                "bank_return_interval_cycles": self.bank_return_interval_cycles,
                "peak_memory_bandwidth_bytes_per_cycle": self.peak_memory_bandwidth_bytes_per_cycle,
                "gemm_peak_ops_per_cycle": self.gemm_peak_ops_per_cycle,
                "general_peak_ops_per_cycle": self.general_peak_ops_per_cycle,
                "write_buffer_bytes": self.write_buffer_bytes,
                "ring_a_buffer_bytes": self.ring_a_buffer_bytes,
                "ring_b_buffer_bytes": self.ring_b_buffer_bytes,
            },
        }
