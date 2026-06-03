from collections import defaultdict, deque
import unittest

from address_remapping.addressing import AddressTransform, encode_physical_address
from address_remapping.hardware import HardwareSpec, PerformanceConfig
from address_remapping.performance import (
    _BankTimelineState,
    _TimedPhysicalRequest,
    PhysicalRequest,
    _compute_write_readiness_requirements,
    _run_ring_bank_event_loop,
    _simulate_per_bank_timeline,
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


class ClosedLoopControllerTests(unittest.TestCase):
    def setUp(self):
        self.hw = HardwareSpec()
        self.perf = PerformanceConfig()

    def test_arbiter_prefers_same_row_and_type(self):
        requests = [
            make_physical_request(self.hw, 0, "e", "ag0", "A", 0, 0, 0, 0),
            make_physical_request(self.hw, 1, "e", "ag0", "A", 0, 0, 1, 1),
            make_physical_request(self.hw, 2, "e", "ag0", "A", 0, 0, 0, 2),
        ]
        timeline = _simulate_per_bank_timeline(
            read_requests=requests,
            write_requests=[],
            hw=self.hw,
            perf=self.perf,
        )
        self.assertGreater(timeline["arbiter1_wins"], 0)
        self.assertGreater(timeline["arbiter2_wins"], 0)

    def test_write_queue_backpressure_reports_full_cycles(self):
        writes = [
            make_physical_request(self.hw, idx, "e", "ag4", "writeback", 0, 0, idx % 4, idx)
            for idx in range(64)
        ]
        perf = PerformanceConfig(controller_write_queue_depth=2, slice_write_buffer_depth=4)
        timeline = _simulate_per_bank_timeline(
            read_requests=[],
            write_requests=writes,
            hw=self.hw,
            perf=perf,
        )
        self.assertGreater(timeline["q_w_full_cycles"], 0.0)
        self.assertGreater(timeline["forced_drain_count"], 0)

    def test_slice_blocking_can_be_triggered(self):
        reads = [
            make_physical_request(self.hw, idx, "e", "ag0", "A", 0, idx % 2, idx // 2, idx)
            for idx in range(32)
        ]
        writes = [
            make_physical_request(self.hw, idx, "e", "ag4", "writeback", 0, 0, idx % 8, idx)
            for idx in range(64)
        ]
        perf = PerformanceConfig(controller_write_queue_depth=2, slice_write_buffer_depth=2)
        timeline = _simulate_per_bank_timeline(
            read_requests=reads,
            write_requests=writes,
            hw=self.hw,
            perf=perf,
        )
        self.assertGreater(timeline["slice_blocked_cycles"], 0.0)

    def test_mul_frontier_waits_for_b_and_advances_with_a(self):
        reads = [
            make_physical_request(self.hw, idx, "e", "ag0", "A", 0, 0, idx // 4, idx)
            for idx in range(8)
        ] + [
            make_physical_request(self.hw, 100 + idx, "e", "ag1", "B", 0, 1, 0, idx)
            for idx in range(2)
        ]
        writes = [
            make_physical_request(self.hw, idx, "e", "ag4", "writeback", 0, 0, 10 + (idx // 2), idx)
            for idx in range(4)
        ]
        requirements = _compute_write_readiness_requirements(
            "prefill_mul_fp32MN_fp32M_fp32MN",
            None,
            reads,
            writes,
            self.hw,
        )
        self.assertEqual(requirements[0], (("B", 2), ("A", 4)))
        self.assertEqual(requirements[1], (("B", 2), ("A", 4)))
        self.assertEqual(requirements[2], (("B", 2), ("A", 8)))
        self.assertEqual(requirements[3], (("B", 2), ("A", 8)))

    def test_summac_frontier_unlocks_write_groups_progressively(self):
        class _Layout:
            dtype = "fp32"

        reads = [
            make_physical_request(self.hw, idx, "e", "ag0", "A", 0, 0, idx // 4, idx)
            for idx in range(16)
        ]
        writes = [
            make_physical_request(self.hw, idx, "e", "ag4", "writeback", 0, 0, 20, idx)
            for idx in range(16)
        ]
        requirements = _compute_write_readiness_requirements(
            "prefill_summac_fp32MN_fp32MN",
            {"outputs": {"out": {"layout": _Layout()}}},
            reads,
            writes,
            self.hw,
        )
        self.assertEqual(requirements[0], (("A", 2),))
        self.assertEqual(requirements[1], (("A", 2),))
        self.assertEqual(requirements[2], (("A", 4),))
        self.assertEqual(requirements[15], (("A", 16),))


class RingGemmGroupCompletionTests(unittest.TestCase):
    def setUp(self):
        self.hw = HardwareSpec()

    def test_group_ready_waits_for_all_requests(self):
        group_key = "a:0"
        requests = [
            make_physical_request(self.hw, idx, "e", "ag0", "A", 0, 0, 0, idx)
            for idx in range(8)
        ]
        bank_states = {0: _BankTimelineState()}
        future_reads = [
            _TimedPhysicalRequest(request=request, release_cycle=0.0, group_key=group_key)
            for request in requests
        ]
        future_writes = []
        completion_by_group = {}
        group_total_requests = {group_key: len(requests)}
        group_completed_requests = {group_key: 0}
        runtime_state = {
            "now": 0.0,
            "read_ready_by_bank": defaultdict(deque),
            "write_ready_by_bank": defaultdict(deque),
            "write_buffer_capacity_reqs": max(1, self.hw.write_buffer_bytes // max(1, self.hw.block_bits // 8)),
            "write_buffer_occupancy": 0,
            "forced_drain_count": 0,
            "in_forced_drain": False,
        }

        _run_ring_bank_event_loop(
            bank_states,
            future_reads,
            future_writes,
            completion_by_group,
            group_total_requests,
            group_completed_requests,
            [group_key],
            self.hw,
            runtime_state,
        )

        expected_ready = self.hw.request_latency_cycles + (len(requests) - 1) * self.hw.bank_return_interval_cycles
        self.assertEqual(completion_by_group[group_key], expected_ready)
        self.assertEqual(group_completed_requests[group_key], len(requests))

    def test_independent_banks_can_complete_in_parallel(self):
        requests = [
            make_physical_request(self.hw, 0, "e", "ag0", "A", 0, 0, 0, 0),
            make_physical_request(self.hw, 1, "e", "ag1", "B", 0, 1, 0, 0),
            make_physical_request(self.hw, 2, "e", "ag4", "writeback", 0, 3, 0, 0),
        ]
        bank_states = {0: _BankTimelineState(), 1: _BankTimelineState(), 3: _BankTimelineState()}
        future_reads = [
            _TimedPhysicalRequest(request=requests[0], release_cycle=0.0, group_key="a:0"),
            _TimedPhysicalRequest(request=requests[1], release_cycle=0.0, group_key="b:0"),
        ]
        future_writes = [
            _TimedPhysicalRequest(request=requests[2], release_cycle=0.0, group_key="w:0"),
        ]
        completion_by_group = {}
        group_total_requests = {"a:0": 1, "b:0": 1, "w:0": 1}
        group_completed_requests = {"a:0": 0, "b:0": 0, "w:0": 0}
        runtime_state = {
            "now": 0.0,
            "read_ready_by_bank": defaultdict(deque),
            "write_ready_by_bank": defaultdict(deque),
            "write_buffer_capacity_reqs": max(1, self.hw.write_buffer_bytes // max(1, self.hw.block_bits // 8)),
            "write_buffer_occupancy": 0,
            "forced_drain_count": 0,
            "in_forced_drain": False,
            "simultaneous_bank_commits": 0,
            "max_parallel_banks_active": 1,
        }

        _run_ring_bank_event_loop(
            bank_states,
            future_reads,
            future_writes,
            completion_by_group,
            group_total_requests,
            group_completed_requests,
            ["a:0", "b:0", "w:0"],
            self.hw,
            runtime_state,
        )

        self.assertEqual(completion_by_group["a:0"], self.hw.request_latency_cycles)
        self.assertEqual(completion_by_group["b:0"], self.hw.request_latency_cycles)
        self.assertEqual(completion_by_group["w:0"], self.hw.request_latency_cycles)
        self.assertEqual(bank_states[0].cycles, self.hw.request_latency_cycles)
        self.assertEqual(bank_states[1].cycles, self.hw.request_latency_cycles)
        self.assertEqual(bank_states[3].cycles, self.hw.request_latency_cycles)
        self.assertEqual(runtime_state["simultaneous_bank_commits"], 1)
        self.assertEqual(runtime_state["max_parallel_banks_active"], 3)


if __name__ == "__main__":
    unittest.main()
