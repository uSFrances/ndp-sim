#!/usr/bin/env python3
"""Merge local_hub handshake info into bank frame requests.

Rules implemented for the rms_norm/summac case:
1. Parse local_hub requests from local_hub_req_bank0(1).log.
2. Parse bank requests after a target slice start in bank0_frame(8).log.
3. Collapse consecutive bank requests to the same address and keep only the first one.
4. Align the remaining bank requests with local_hub requests by address sequence.
5. Emit a merged text file containing the bank-side fields plus local_hub HandshakeType.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import re


LOCAL_REQ_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(0x[0-9a-fA-F]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*([A-Z_]+)\s*\|\s*(0b[01]+)\s*\|\s*(0b[01]+)\s*$"
)

BANK_REQ_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([01]\([RW]\))\s*\|"
    r"\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(0x[0-9a-fA-F]+)\s*\|\s*(0x[0-9a-fA-F]+)\s*$"
)

SLICE_START_RE = re.compile(r"^\[(\d+)\]\s+INFO:\s+slice start \(cycle=0\)\s*$")


def normalize_hex(value: str) -> str:
    return hex(int(value, 16))


@dataclass
class LocalRequest:
    time_ns: int
    req_ch: int
    req_addr_full_hex: str
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
    source_line: str


@dataclass
class MergedRow:
    bank_request: BankRequest
    local_request: LocalRequest | None
    match_status: str


def parse_local_requests(path: Path) -> list[LocalRequest]:
    requests: list[LocalRequest] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            match = LOCAL_REQ_RE.match(line)
            if not match:
                continue
            requests.append(
                LocalRequest(
                    time_ns=int(match.group(1)),
                    req_ch=int(match.group(2)),
                    req_addr_full_hex=normalize_hex(match.group(3)),
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
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            slice_match = SLICE_START_RE.match(line)
            if slice_match:
                in_target_slice = int(slice_match.group(1)) == slice_start_time
                continue

            if not in_target_slice:
                continue

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
                    addr_hex=normalize_hex(match.group(9)),
                    source_line=line,
                )
            )

    return requests


def collapse_consecutive_bank_requests(requests: list[BankRequest]) -> tuple[list[BankRequest], int]:
    collapsed: list[BankRequest] = []
    skipped = 0
    previous_addr: str | None = None
    for request in requests:
        if request.addr_hex == previous_addr:
            skipped += 1
            continue
        collapsed.append(request)
        previous_addr = request.addr_hex
    return collapsed, skipped


def align_requests(
    local_requests: list[LocalRequest],
    bank_requests: list[BankRequest],
) -> tuple[list[MergedRow], int]:
    merged_rows: list[MergedRow] = []
    local_index = 0
    matched = 0

    for bank_request in bank_requests:
        local_request = local_requests[local_index] if local_index < len(local_requests) else None
        if local_request is not None and local_request.req_addr_full_hex == bank_request.addr_hex:
            merged_rows.append(
                MergedRow(
                    bank_request=bank_request,
                    local_request=local_request,
                    match_status="MATCHED",
                )
            )
            local_index += 1
            matched += 1
        else:
            merged_rows.append(
                MergedRow(
                    bank_request=bank_request,
                    local_request=None,
                    match_status="BANK_ONLY",
                )
            )

    return merged_rows, matched


def write_merged_report(
    output_path: Path,
    merged_rows: list[MergedRow],
    local_requests: list[LocalRequest],
    slice_start_time: int,
    raw_bank_count: int,
    duplicate_skip_count: int,
    matched_count: int,
) -> None:
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("# Merged bank/local_hub request view for rms_norm/summac\n")
        fh.write(f"# SliceStartTime(ns): {slice_start_time}\n")
        fh.write(f"# LocalRequestCount: {len(local_requests)}\n")
        fh.write(f"# RawBankRequestCount: {raw_bank_count}\n")
        fh.write(f"# DuplicateBankRequestsSkipped: {duplicate_skip_count}\n")
        fh.write(f"# CollapsedBankRequestCount: {len(merged_rows)}\n")
        fh.write(f"# MatchedCount: {matched_count}\n")
        fh.write(f"# UnmatchedBankCount: {len(merged_rows) - matched_count}\n")
        fh.write(f"# UnmatchedLocalCount: {len(local_requests) - matched_count}\n")
        fh.write(
            "# Format: BankTime(ns) | CycleFromSliceStart | Row(dec) | Col(dec) | "
            "RW(0:R,1:W) | StartFrame | EndFrame | Data(hex128) | Addr(hex) | "
            "LocalTime(ns) | LocalReqCh | LocalReqRW(1=Write,0=Read) | "
            "LocalHasValidReadReq | HandshakeType | ArbValidMask | ReqRwMask | MatchStatus\n"
        )

        for row in merged_rows:
            bank = row.bank_request
            local = row.local_request
            fh.write(
                f"{bank.time_ns:>12} | "
                f"{bank.cycle_from_slice_start:>16} | "
                f"{bank.row_dec:>8} | "
                f"{bank.col_dec:>8} | "
                f"{bank.rw:>9} | "
                f"{bank.start_frame:>10} | "
                f"{bank.end_frame:>8} | "
                f"{bank.data_hex128} | "
                f"{bank.addr_hex:>8} | "
                f"{local.time_ns if local else '':>13} | "
                f"{local.req_ch if local else '':>10} | "
                f"{local.req_rw if local else '':>22} | "
                f"{local.has_valid_read_req if local else '':>21} | "
                f"{local.handshake_type if local else '':>18} | "
                f"{local.arb_valid_mask if local else '':>12} | "
                f"{local.req_rw_mask if local else '':>10} | "
                f"{row.match_status}\n"
            )


def main() -> int:
    default_base = Path(r"H:\dev\projects\ndp-sim\address_remapping\golden\rms_norm\summac\64x64")

    parser = argparse.ArgumentParser(
        description="Merge local_hub HandshakeType into bank frame requests for rms_norm/summac."
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
        default=180033000,
        help="Use requests after '[time] INFO: slice start (cycle=0)' in bank log.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_base / "bank0_frame_with_handshake.log",
        help="Output merged log path.",
    )
    args = parser.parse_args()

    local_requests = parse_local_requests(args.local_log)
    raw_bank_requests = parse_bank_requests(args.bank_log, args.slice_start_time)
    collapsed_bank_requests, duplicate_skip_count = collapse_consecutive_bank_requests(raw_bank_requests)
    merged_rows, matched_count = align_requests(local_requests, collapsed_bank_requests)

    write_merged_report(
        output_path=args.output,
        merged_rows=merged_rows,
        local_requests=local_requests,
        slice_start_time=args.slice_start_time,
        raw_bank_count=len(raw_bank_requests),
        duplicate_skip_count=duplicate_skip_count,
        matched_count=matched_count,
    )

    print(f"Local request count:        {len(local_requests)}")
    print(f"Raw bank request count:     {len(raw_bank_requests)}")
    print(f"Collapsed bank count:       {len(collapsed_bank_requests)}")
    print(f"Duplicate bank skipped:     {duplicate_skip_count}")
    print(f"Matched count:              {matched_count}")
    print(f"Unmatched bank count:       {len(collapsed_bank_requests) - matched_count}")
    print(f"Unmatched local count:      {len(local_requests) - matched_count}")
    print(f"Merged report:              {args.output}")

    if merged_rows:
        for row in merged_rows[:10]:
            local_addr = row.local_request.req_addr_full_hex if row.local_request else ""
            print(
                f"sample: bank_time={row.bank_request.time_ns}, bank_addr={row.bank_request.addr_hex}, "
                f"local_addr={local_addr}, status={row.match_status}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
