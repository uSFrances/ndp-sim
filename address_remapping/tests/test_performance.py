import unittest

from address_remapping.addressing import AddressTransform, encode_physical_address
from address_remapping.hardware import HardwareSpec, PerformanceConfig
from address_remapping.performance import PhysicalRequest, _simulate_per_bank_timeline


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


if __name__ == "__main__":
    unittest.main()
