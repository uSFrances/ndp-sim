#!/usr/bin/env python3
"""Analyze row-change causes in bank frame logs.

For the rms_norm/summac 64x64 bank0 trace, a row-change event is modeled as:
1. A request to some address.
2. The same address immediately appears again with EndFrame=1, which closes the row.
3. The following request opens the next row (usually StartFrame=1).

We classify each row change into:
- NORMAL_DATA: the next request keeps the same RW direction, so the row change is
  attributed to continuing normal sequential traffic.
- RW_SWITCH: the next request flips RW direction, so the row change is attributed
  to a read/write mode switch.
- SLICE_END_CLOSE: a close request with no following request inside the slice.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


BANK_REQ_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([01])\(([RW])\)\s*\|"
    r"\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(0x[0-9a-fA-F]+)\s*\|\s*(0x[0-9a-fA-F]+)\s*$"
)
SLICE_START_RE = re.compile(r"^\[(\d+)\]\s+INFO:\s+slice start \(cycle=0\)\s*$")
ROW_CHANGE_RE = re.compile(r"^\[(\d+)\]\s+INFO:\s+Row Change \(cycle=(\d+), total=(\d+)\)\s*$")


def normalize_hex(value: str) -> str:
    return hex(int(value, 16))


@dataclass
class BankRequest:
    time_ns: int
    cycle_from_slice_start: int
    row_dec: int
    col_dec: int
    rw_bit: int
    rw_kind: str
    start_frame: int
    end_frame: int
    data_hex128: str
    addr_hex: str


@dataclass
class RowChangeInfo:
    time_ns: int
    cycle_from_slice_start: int
    total: int


@dataclass
class RowChangeEvent:
    event_index: int
    event_time_ns: int
    event_cycle: int
    cause: str
    reason: str
    close_addr_hex: str
    close_rw: str
    close_row_dec: int
    close_col_dec: int
    previous_time_ns: int
    previous_cycle: int
    next_time_ns: int | None
    next_cycle: int | None
    next_rw: str | None
    next_row_dec: int | None
    next_col_dec: int | None
    gap_prev_to_close_ns: int
    gap_prev_to_close_cycles: int | None
    gap_close_to_next_ns: int | None
    gap_close_to_next_cycles: int | None
    row_jump: int | None
    rw_switched: bool


@dataclass
class BurstSegment:
    segment_index: int
    rw_kind: str
    start_time_ns: int
    end_time_ns: int
    start_cycle: int
    end_cycle: int
    unique_transfer_count: int
    close_repeat_count: int
    payload_bytes: int
    bank_rows: list[int]
    bank_cols: list[int]
    start_addr_hex: str
    end_addr_hex: str
    carried_in_close: bool


def parse_bank_slice(path: Path, slice_start_time: int) -> tuple[list[BankRequest], list[RowChangeInfo]]:
    requests: list[BankRequest] = []
    row_change_infos: list[RowChangeInfo] = []
    in_target_slice = False
    after_slice_completed = False

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

            if "INFO: slice completed" in line:
                after_slice_completed = True
                continue

            req_match = BANK_REQ_RE.match(line)
            if req_match:
                request = BankRequest(
                    time_ns=int(req_match.group(1)),
                    cycle_from_slice_start=int(req_match.group(2)),
                    row_dec=int(req_match.group(3)),
                    col_dec=int(req_match.group(4)),
                    rw_bit=int(req_match.group(5)),
                    rw_kind=req_match.group(6),
                    start_frame=int(req_match.group(7)),
                    end_frame=int(req_match.group(8)),
                    data_hex128=req_match.group(9),
                    addr_hex=normalize_hex(req_match.group(10)),
                )

                if after_slice_completed and requests:
                    previous = requests[-1]
                    trailing_close = (
                        previous.addr_hex == request.addr_hex
                        and previous.row_dec == request.row_dec
                        and previous.col_dec == request.col_dec
                        and previous.rw_kind == request.rw_kind
                        and previous.data_hex128 == request.data_hex128
                        and previous.end_frame == 0
                        and request.end_frame == 1
                    )
                    if not trailing_close:
                        break

                requests.append(request)
                continue

            if after_slice_completed and "INFO: Row Change Count (Start->Finish)" in line:
                continue

            if after_slice_completed:
                continue

            row_change_match = ROW_CHANGE_RE.match(line)
            if row_change_match:
                row_change_infos.append(
                    RowChangeInfo(
                        time_ns=int(row_change_match.group(1)),
                        cycle_from_slice_start=int(row_change_match.group(2)),
                        total=int(row_change_match.group(3)),
                    )
                )

    if not requests:
        raise ValueError(f"No bank requests found after slice start [{slice_start_time}] in {path}")

    return requests, row_change_infos


def is_close_repeat(previous: BankRequest, current: BankRequest) -> bool:
    return (
        previous.addr_hex == current.addr_hex
        and previous.row_dec == current.row_dec
        and previous.col_dec == current.col_dec
        and previous.rw_kind == current.rw_kind
        and previous.data_hex128 == current.data_hex128
        and previous.end_frame == 0
        and current.end_frame == 1
    )


def transition_label(event: RowChangeEvent) -> str:
    if event.cause == "SLICE_END_CLOSE":
        return f"{event.close_rw}->END"
    return f"{event.close_rw}->{event.next_rw}"


def summarize_burst_segments(requests: list[BankRequest]) -> list[BurstSegment]:
    segments: list[BurstSegment] = []
    if not requests:
        return segments

    index = 0
    while index < len(requests):
        start = index
        rw_kind = requests[index].rw_kind
        while index + 1 < len(requests) and requests[index + 1].rw_kind == rw_kind:
            index += 1
        end = index

        unique_transfer_count = 0
        close_repeat_count = 0
        bank_rows: list[int] = []
        bank_cols: list[int] = []
        previous_unique: BankRequest | None = None
        first = requests[start]
        carried_in_close = start == 0 and first.start_frame == 0 and first.end_frame == 1

        for position in range(start, end + 1):
            request = requests[position]
            if carried_in_close and position == start:
                previous_unique = request
                continue
            if previous_unique is not None and is_close_repeat(previous_unique, request):
                close_repeat_count += 1
                continue

            unique_transfer_count += 1
            previous_unique = request
            if not bank_rows or bank_rows[-1] != request.row_dec:
                bank_rows.append(request.row_dec)
            bank_cols.append(request.col_dec)

        last = requests[end]
        payload_bytes = unique_transfer_count * 16

        segments.append(
            BurstSegment(
                segment_index=len(segments) + 1,
                rw_kind=rw_kind,
                start_time_ns=first.time_ns,
                end_time_ns=last.time_ns,
                start_cycle=first.cycle_from_slice_start,
                end_cycle=last.cycle_from_slice_start,
                unique_transfer_count=unique_transfer_count,
                close_repeat_count=close_repeat_count,
                payload_bytes=payload_bytes,
                bank_rows=bank_rows,
                bank_cols=bank_cols,
                start_addr_hex=first.addr_hex,
                end_addr_hex=last.addr_hex,
                carried_in_close=carried_in_close,
            )
        )

        index += 1

    return segments


def classify_row_changes(requests: list[BankRequest]) -> list[RowChangeEvent]:
    events: list[RowChangeEvent] = []

    for index in range(1, len(requests)):
        previous = requests[index - 1]
        current = requests[index]
        if not is_close_repeat(previous, current):
            continue

        next_request = requests[index + 1] if index + 1 < len(requests) else None
        rw_switched = next_request is not None and next_request.rw_kind != current.rw_kind
        if next_request is None:
            cause = "SLICE_END_CLOSE"
            reason = "Close request is the last row-close that belongs to this slice."
        elif rw_switched:
            cause = "RW_SWITCH"
            reason = (
                f"Closed {current.rw_kind} row {current.row_dec} and next request switches to "
                f"{next_request.rw_kind} row {next_request.row_dec}."
            )
        else:
            cause = "NORMAL_DATA"
            reason = (
                f"Closed {current.rw_kind} row {current.row_dec} and next request continues with "
                f"the same {current.rw_kind} direction on row {next_request.row_dec}."
            )

        next_time_ns = next_request.time_ns if next_request is not None else None
        next_cycle = next_request.cycle_from_slice_start if next_request is not None else None
        next_rw = next_request.rw_kind if next_request is not None else None
        next_row_dec = next_request.row_dec if next_request is not None else None
        next_col_dec = next_request.col_dec if next_request is not None else None
        row_jump = next_row_dec - current.row_dec if next_row_dec is not None else None
        gap_prev_to_close_cycles = current.cycle_from_slice_start - previous.cycle_from_slice_start
        if gap_prev_to_close_cycles < 0:
            gap_prev_to_close_cycles = None

        gap_close_to_next_cycles = None
        if next_request is not None:
            gap_close_to_next_cycles = next_request.cycle_from_slice_start - current.cycle_from_slice_start
            if gap_close_to_next_cycles < 0:
                gap_close_to_next_cycles = None

        events.append(
            RowChangeEvent(
                event_index=len(events) + 1,
                event_time_ns=current.time_ns,
                event_cycle=current.cycle_from_slice_start,
                cause=cause,
                reason=reason,
                close_addr_hex=current.addr_hex,
                close_rw=current.rw_kind,
                close_row_dec=current.row_dec,
                close_col_dec=current.col_dec,
                previous_time_ns=previous.time_ns,
                previous_cycle=previous.cycle_from_slice_start,
                next_time_ns=next_time_ns,
                next_cycle=next_cycle,
                next_rw=next_rw,
                next_row_dec=next_row_dec,
                next_col_dec=next_col_dec,
                gap_prev_to_close_ns=current.time_ns - previous.time_ns,
                gap_prev_to_close_cycles=gap_prev_to_close_cycles,
                gap_close_to_next_ns=(
                    next_request.time_ns - current.time_ns if next_request is not None else None
                ),
                gap_close_to_next_cycles=gap_close_to_next_cycles,
                row_jump=row_jump,
                rw_switched=rw_switched,
            )
        )

    return events


def write_csv(output_path: Path, events: list[RowChangeEvent]) -> None:
    fieldnames = [
        "event_index",
        "event_time_ns",
        "event_cycle",
        "cause",
        "reason",
        "close_addr_hex",
        "close_rw",
        "close_row_dec",
        "close_col_dec",
        "previous_time_ns",
        "previous_cycle",
        "next_time_ns",
        "next_cycle",
        "next_rw",
        "next_row_dec",
        "next_col_dec",
        "gap_prev_to_close_ns",
        "gap_prev_to_close_cycles",
        "gap_close_to_next_ns",
        "gap_close_to_next_cycles",
        "row_jump",
        "rw_switched",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(event.__dict__)


def write_summary(
    output_path: Path,
    bank_log: Path,
    slice_start_time: int,
    total_requests: int,
    row_change_infos: list[RowChangeInfo],
    events: list[RowChangeEvent],
    segments: list[BurstSegment],
) -> None:
    total = len(events)
    normal_count = sum(1 for event in events if event.cause == "NORMAL_DATA")
    rw_switch_count = sum(1 for event in events if event.cause == "RW_SWITCH")
    slice_end_close_count = sum(1 for event in events if event.cause == "SLICE_END_CLOSE")

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("# Row change analysis for rms_norm/summac bank0_frame.log\n")
        fh.write(f"# BankLog: {bank_log}\n")
        fh.write(f"# SliceStartTime(ns): {slice_start_time}\n")
        fh.write(f"# RequestCountInSlice: {total_requests}\n")
        fh.write(f"# DerivedRowChangeCount: {total}\n")
        fh.write(f"# NormalDataRowChangeCount: {normal_count}\n")
        fh.write(f"# RwSwitchRowChangeCount: {rw_switch_count}\n")
        fh.write(f"# SliceEndCloseCount: {slice_end_close_count}\n")
        fh.write(f"# LoggedRowChangeInfoCount: {len(row_change_infos)}\n")
        if row_change_infos:
            fh.write(f"# LoggedRowChangeInfoLastTotal: {row_change_infos[-1].total}\n")
        fh.write("#\n")
        fh.write("# Cause legend:\n")
        fh.write("#   NORMAL_DATA = same RW direction before and after the close request.\n")
        fh.write("#   RW_SWITCH   = RW direction flips after the close request.\n")
        fh.write("#   SLICE_END_CLOSE = close request has no following request inside this slice.\n")
        fh.write("#\n")
        transition_sequence = [transition_label(event) for event in events]
        fh.write(f"# TransitionSequence: {' | '.join(transition_sequence)}\n")
        fh.write("#\n")
        fh.write("# Burst segments (contiguous same-RW requests):\n")
        fh.write("#   payload_bytes assumes each request carries 128-bit = 16 B payload.\n")
        fh.write("#   For fp32, 256 B = one 64-element tensor row and 32 B = eight fp32 outputs.\n")
        fh.write("#\n")
        for segment in segments:
            equivalent_tensor_rows = segment.payload_bytes / 256 if segment.payload_bytes else 0
            equivalent_fp32 = segment.payload_bytes / 4 if segment.payload_bytes else 0
            fh.write(
                f"#   SEG[{segment.segment_index:02d}] {segment.rw_kind}: "
                f"rows={segment.bank_rows}, cols={segment.bank_cols[0]}..{segment.bank_cols[-1]}, "
                f"payload={segment.payload_bytes}B, transfers={segment.unique_transfer_count}, "
                f"close_repeats={segment.close_repeat_count}, tensor_rows={equivalent_tensor_rows:.2f}, "
                f"fp32={equivalent_fp32:.0f}"
            )
            if segment.carried_in_close:
                fh.write(" (carried-in close from previous slice)")
            fh.write("\n")
        fh.write("#\n")

        for event in events:
            fh.write(
                f"[{event.event_index:02d}] time={event.event_time_ns} ns, cycle={event.event_cycle}, "
                f"cause={event.cause}, close={event.close_rw} row={event.close_row_dec} col={event.close_col_dec} "
                f"addr={event.close_addr_hex}, next_rw={event.next_rw or '-'} "
                f"next_row={event.next_row_dec if event.next_row_dec is not None else '-'} "
                f"gap(prev->close)={event.gap_prev_to_close_ns}ns/"
                f"{event.gap_prev_to_close_cycles if event.gap_prev_to_close_cycles is not None else '-'}cy, "
                f"gap(close->next)={event.gap_close_to_next_ns if event.gap_close_to_next_ns is not None else '-'}ns/"
                f"{event.gap_close_to_next_cycles if event.gap_close_to_next_cycles is not None else '-'}cy, "
                f"row_jump={event.row_jump if event.row_jump is not None else '-'}\n"
            )
            fh.write(f"      reason: {event.reason}\n")


def main() -> int:
    default_base = Path(r"H:\dev\projects\ndp-sim\address_remapping\golden\rms_norm\summac\64x64")

    parser = argparse.ArgumentParser(
        description="Analyze row-change causes after a target slice start in bank0_frame.log."
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
        default=209507000,
        help="Analyze requests after '[time] INFO: slice start (cycle=0)' in bank log.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=default_base / "bank0_row_change_summary.log",
        help="Path to the text summary output.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=default_base / "bank0_row_change_events.csv",
        help="Path to the CSV output.",
    )
    args = parser.parse_args()

    requests, row_change_infos = parse_bank_slice(args.bank_log, args.slice_start_time)
    segments = summarize_burst_segments(requests)
    events = classify_row_changes(requests)

    write_summary(
        output_path=args.summary_output,
        bank_log=args.bank_log,
        slice_start_time=args.slice_start_time,
        total_requests=len(requests),
        row_change_infos=row_change_infos,
        events=events,
        segments=segments,
    )
    write_csv(args.csv_output, events)

    normal_count = sum(1 for event in events if event.cause == "NORMAL_DATA")
    rw_switch_count = sum(1 for event in events if event.cause == "RW_SWITCH")
    slice_end_close_count = sum(1 for event in events if event.cause == "SLICE_END_CLOSE")

    print(f"Request count in slice:      {len(requests)}")
    print(f"Derived row change count:    {len(events)}")
    print(f"Normal-data row changes:     {normal_count}")
    print(f"RW-switch row changes:       {rw_switch_count}")
    print(f"Slice-end row changes:       {slice_end_close_count}")
    print(f"Logged row change count:     {len(row_change_infos)}")
    if row_change_infos:
        print(f"Logged final row change id:  {row_change_infos[-1].total}")
    print(f"Summary output:              {args.summary_output}")
    print(f"CSV output:                  {args.csv_output}")

    for event in events[:10]:
        print(
            f"sample[{event.event_index}]: time={event.event_time_ns}, cause={event.cause}, "
            f"close_rw={event.close_rw}, next_rw={event.next_rw}, "
            f"close_row={event.close_row_dec}, next_row={event.next_row_dec}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
