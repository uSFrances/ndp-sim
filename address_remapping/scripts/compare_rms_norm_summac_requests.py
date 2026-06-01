#!/usr/bin/env python3
"""Compare request timestamps between local_hub and bank frame logs.

This script:
1. Parses valid requests from local_hub_req_bank0.log.
2. Parses valid requests from bank0_frame.log after a target slice start marker.
3. Compares request counts and request timestamps in order.
4. Writes a merged CSV report for easy inspection.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


LOCAL_REQ_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*([A-Z_]+)\s*\|\s*(0b[01]+)\s*\|\s*(0b[01]+)\s*$"
)

BANK_REQ_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([01]\([RW]\))\s*\|"
    r"\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(0x[0-9a-fA-F]+)\s*\|\s*(0x[0-9a-fA-F]+)\s*$"
)

SLICE_START_RE = re.compile(r"^\[(\d+)\]\s+INFO:\s+slice start \(cycle=0\)\s*$")


@dataclass
class LocalRequest:
    time_ns: int
    req_ch: int
    bank_ch: int
    req_rw: int
    has_valid_read_req: int
    handshake_type: str
    arb_valid_mask: str
    req_rw_mask: str


@dataclass
class BankRequest:
    time_ns: int
    cycle_from_slice_start: int
    row_dec: int
    col_dec: int
    rw: str
    start_frame: int
    end_frame: int
    data_hex128: str
    addr_hex: str


def parse_local_requests(path: Path) -> list[LocalRequest]:
    requests: list[LocalRequest] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\n")
            match = LOCAL_REQ_RE.match(line)
            if not match:
                continue
            requests.append(
                LocalRequest(
                    time_ns=int(match.group(1)),
                    req_ch=int(match.group(2)),
                    bank_ch=int(match.group(3)),
                    req_rw=int(match.group(4)),
                    has_valid_read_req=int(match.group(5)),
                    handshake_type=match.group(6),
                    arb_valid_mask=match.group(7),
                    req_rw_mask=match.group(8),
                )
            )
    return requests


def parse_bank_requests(path: Path, slice_start_time: int) -> list[BankRequest]:
    requests: list[BankRequest] = []
    in_target_slice = False
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\n")

            slice_match = SLICE_START_RE.match(line)
            if slice_match:
                in_target_slice = int(slice_match.group(1)) == slice_start_time
                continue

            if not in_target_slice:
                continue

            # Stop when the next slice begins.
            if line.startswith("[") and "INFO: slice start" in line:
                break

            match = BANK_REQ_RE.match(line)
            if not match:
                continue

            requests.append(
                BankRequest(
                    time_ns=int(match.group(1)),
                    cycle_from_slice_start=int(match.group(2)),
                    row_dec=int(match.group(3)),
                    col_dec=int(match.group(4)),
                    rw=match.group(5),
                    start_frame=int(match.group(6)),
                    end_frame=int(match.group(7)),
                    data_hex128=match.group(8),
                    addr_hex=match.group(9),
                )
            )

    if not requests:
        raise ValueError(
            f"No bank requests found after slice start [${slice_start_time}] in {path}"
        )

    return requests


def write_report(
    output_path: Path,
    local_requests: list[LocalRequest],
    bank_requests: list[BankRequest],
) -> tuple[int, list[dict[str, str | int]]]:
    rows: list[dict[str, str | int]] = []
    mismatch_count = 0
    max_len = max(len(local_requests), len(bank_requests))

    for index in range(max_len):
        local_req = local_requests[index] if index < len(local_requests) else None
        bank_req = bank_requests[index] if index < len(bank_requests) else None

        local_time = local_req.time_ns if local_req else ""
        bank_time = bank_req.time_ns if bank_req else ""
        time_match = local_req is not None and bank_req is not None and local_req.time_ns == bank_req.time_ns
        status = "MATCH" if time_match else "MISMATCH"
        if status == "MISMATCH":
            mismatch_count += 1

        rows.append(
            {
                "index": index,
                "local_time_ns": local_time,
                "bank_time_ns": bank_time,
                "time_match": status,
                "local_req_ch": local_req.req_ch if local_req else "",
                "local_bank_ch": local_req.bank_ch if local_req else "",
                "local_req_rw": local_req.req_rw if local_req else "",
                "local_has_valid_read_req": local_req.has_valid_read_req if local_req else "",
                "local_handshake_type": local_req.handshake_type if local_req else "",
                "bank_cycle_from_slice_start": bank_req.cycle_from_slice_start if bank_req else "",
                "bank_row_dec": bank_req.row_dec if bank_req else "",
                "bank_col_dec": bank_req.col_dec if bank_req else "",
                "bank_rw": bank_req.rw if bank_req else "",
                "bank_start_frame": bank_req.start_frame if bank_req else "",
                "bank_end_frame": bank_req.end_frame if bank_req else "",
                "bank_addr_hex": bank_req.addr_hex if bank_req else "",
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return mismatch_count, rows


def count_matching_prefix(local_requests: list[LocalRequest], bank_requests: list[BankRequest]) -> int:
    count = 0
    for local_req, bank_req in zip(local_requests, bank_requests):
        if local_req.time_ns != bank_req.time_ns:
            break
        count += 1
    return count


def main() -> int:
    default_base = Path(r"H:\dev\projects\ndp-sim\address_remapping\golden\rms_norm\summac")

    parser = argparse.ArgumentParser(
        description="Compare rms_norm/summac request timestamps between local_hub and bank frame logs."
    )
    parser.add_argument(
        "--local-log",
        type=Path,
        default=default_base / "local_hub_req_bank0.log",
        help="Path to local_hub_req_bank0.log",
    )
    parser.add_argument(
        "--bank-log",
        type=Path,
        default=default_base / "bank0_frame.log",
        help="Path to bank0_frame.log",
    )
    parser.add_argument(
        "--slice-start-time",
        type=int,
        default=182214000,
        help="Use requests after '[time] INFO: slice start (cycle=0)' in bank log.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_base / "merged_request_compare_bank0.csv",
        help="CSV path for merged comparison report.",
    )

    args = parser.parse_args()

    local_requests = parse_local_requests(args.local_log)
    bank_requests = parse_bank_requests(args.bank_log, args.slice_start_time)

    mismatch_count, rows = write_report(args.output, local_requests, bank_requests)
    matching_prefix = count_matching_prefix(local_requests, bank_requests)
    local_times = [req.time_ns for req in local_requests]
    bank_times = [req.time_ns for req in bank_requests]
    shared_times = sorted(set(local_times) & set(bank_times))
    local_only_times = [time_ns for time_ns in local_times if time_ns not in set(bank_times)]
    bank_only_times = [time_ns for time_ns in bank_times if time_ns not in set(local_times)]

    print(f"Local request count: {len(local_requests)}")
    print(f"Bank request count:  {len(bank_requests)}")
    print(f"Count match:         {'YES' if len(local_requests) == len(bank_requests) else 'NO'}")
    print(f"Ordered time match:  {'YES' if mismatch_count == 0 else 'NO'}")
    print(f"Mismatch count:      {mismatch_count}")
    print(f"Matching prefix:     {matching_prefix}")
    print(f"Shared timestamps:   {len(shared_times)}")
    print(f"Local-only count:    {len(local_only_times)}")
    print(f"Bank-only count:     {len(bank_only_times)}")
    print(f"CSV report:          {args.output}")

    if shared_times:
        print(f"First shared times:  {shared_times[:10]}")
    if local_only_times:
        print(f"First local-only:    {local_only_times[:10]}")
    if bank_only_times:
        print(f"First bank-only:     {bank_only_times[:10]}")

    if mismatch_count:
        print("First mismatches:")
        shown = 0
        for row in rows:
            if row["time_match"] == "MATCH":
                continue
            print(
                f"  index={row['index']}, local_time_ns={row['local_time_ns']}, "
                f"bank_time_ns={row['bank_time_ns']}"
            )
            shown += 1
            if shown >= 10:
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
