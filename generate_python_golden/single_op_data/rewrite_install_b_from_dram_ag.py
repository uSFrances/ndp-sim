from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


DEFAULT_SLICE_NUM = 28
DEFAULT_N = 896
DEFAULT_K = 896
INPUT_WEIGHT_NAME = "matrix_B_linearized_128bit"
AG_GROUP_LINES = 8
HEX_LINE_RE = re.compile(r"The data: ([0-9a-f ]+)\s+The transfer", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite per-slice install tensorB files from dram_ag slice*_ag1/ag2 results so the "
            "install B layout exactly matches the simulator's observed fetch order."
        ),
        epilog=(
            "Example:\n"
            "  python rewrite_install_b_from_dram_ag.py "
            "--install-dir ./_tmp_gemv_n2n_n32_install_test/install "
            "--dram-ag-dir ./_tmp_gemv_n2n_n32_run_test/dram_ag "
            "--n 896 --k 896 --slice-num 28\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--dram-ag-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--slice-num", type=int, default=DEFAULT_SLICE_NUM)
    parser.add_argument(
        "--group-lines",
        type=int,
        default=AG_GROUP_LINES,
        help="Number of consecutive ag1 lines followed by the same number of ag2 lines per group. Default: 8.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite files in place. If omitted, write into <install-dir>_from_dram_ag.",
    )
    return parser.parse_args()


def bytes_to_qgen_style_128bit_txt(byte_stream: bytes) -> str:
    if len(byte_stream) % 16 != 0:
        raise ValueError("Byte stream length must be a multiple of 16 for 128-bit text export")
    lines = []
    for offset in range(0, len(byte_stream), 16):
        chunk = byte_stream[offset : offset + 16]
        lines.append("".join(f"{byte:08b}" for byte in reversed(chunk)))
    return "\n".join(lines) + "\n"


def fp16_hex_lines(values: np.ndarray) -> list[str]:
    flat = np.asarray(values, dtype=np.float16).reshape(-1)
    return [f"{int(val.view(np.uint16)):04x}" for val in flat]


def parse_ag_line(line: str, *, path: Path, line_no: int) -> list[int]:
    match = HEX_LINE_RE.search(line)
    if match is None:
        raise ValueError(f"Could not parse data payload from {path}:{line_no}")
    words = match.group(1).split()
    if len(words) != 8:
        raise ValueError(f"Expected 8 fp16 hex words in {path}:{line_no}, got {len(words)}")
    # Dump format prints words from high address to low address, so reverse it back.
    return [int(word, 16) for word in reversed(words)]


def load_ag_words(path: Path) -> list[list[int]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dram_ag dump: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    payloads: list[list[int]] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payloads.append(parse_ag_line(line, path=path, line_no=line_no))
    return payloads


def interleave_ag_payloads(
    ag1_payloads: list[list[int]],
    ag2_payloads: list[list[int]],
    *,
    group_lines: int,
) -> np.ndarray:
    if len(ag1_payloads) != len(ag2_payloads):
        raise ValueError(
            f"ag1/ag2 line count mismatch: ag1={len(ag1_payloads)} ag2={len(ag2_payloads)}"
        )
    if len(ag1_payloads) % group_lines != 0:
        raise ValueError(
            f"ag payload line count {len(ag1_payloads)} is not divisible by group_lines={group_lines}"
        )

    merged_words: list[int] = []
    for group_start in range(0, len(ag1_payloads), group_lines):
        for payload in ag1_payloads[group_start : group_start + group_lines]:
            merged_words.extend(payload)
        for payload in ag2_payloads[group_start : group_start + group_lines]:
            merged_words.extend(payload)
    return np.asarray(merged_words, dtype=np.uint16).view(np.float16)


def copy_non_b_files(src_slice_dir: Path, dst_slice_dir: Path) -> None:
    for path in src_slice_dir.iterdir():
        if path.name.startswith(INPUT_WEIGHT_NAME):
            continue
        if path.is_file():
            dst_slice_dir.mkdir(parents=True, exist_ok=True)
            (dst_slice_dir / path.name).write_bytes(path.read_bytes())


def rewrite_one_slice(
    *,
    slice_id: int,
    src_slice_dir: Path,
    dst_slice_dir: Path,
    dram_ag_dir: Path,
    expected_elems: int,
    group_lines: int,
) -> None:
    ag1_path = dram_ag_dir / f"slice{slice_id}_ag1_results.txt"
    ag2_path = dram_ag_dir / f"slice{slice_id}_ag2_results.txt"
    ag1_payloads = load_ag_words(ag1_path)
    ag2_payloads = load_ag_words(ag2_path)
    merged_fp16 = interleave_ag_payloads(ag1_payloads, ag2_payloads, group_lines=group_lines)

    if merged_fp16.size != expected_elems:
        raise ValueError(
            f"slice{slice_id}: reconstructed B has {merged_fp16.size} fp16 values, expected {expected_elems}"
        )

    dst_slice_dir.mkdir(parents=True, exist_ok=True)
    bin_path = dst_slice_dir / f"{INPUT_WEIGHT_NAME}.bin"
    txt_path = dst_slice_dir / f"{INPUT_WEIGHT_NAME}.txt"
    hex_path = dst_slice_dir / f"{INPUT_WEIGHT_NAME}_{expected_elems}x1_hex.txt"

    payload_bytes = merged_fp16.astype(np.float16).reshape(-1).tobytes()
    bin_path.write_bytes(payload_bytes)
    txt_path.write_text(bytes_to_qgen_style_128bit_txt(payload_bytes), encoding="utf-8")
    hex_path.write_text("\n".join(fp16_hex_lines(merged_fp16)) + "\n", encoding="utf-8")
    print(f"[OK] slice{slice_id:02d}: reconstructed {INPUT_WEIGHT_NAME} from ag1/ag2 dumps")


def main() -> None:
    args = parse_args()
    install_dir = args.install_dir.resolve()
    dram_ag_dir = args.dram_ag_dir.resolve()
    if not install_dir.exists():
        raise FileNotFoundError(f"Install directory not found: {install_dir}")
    if not dram_ag_dir.exists():
        raise FileNotFoundError(f"dram_ag directory not found: {dram_ag_dir}")
    if args.n % args.slice_num != 0:
        raise ValueError(f"N={args.n} must be divisible by slice_num={args.slice_num}")

    local_n = args.n // args.slice_num
    expected_elems = args.k * local_n

    if args.in_place:
        output_install_dir = install_dir
    else:
        output_install_dir = install_dir.parent / f"{install_dir.name}_from_dram_ag"
        output_install_dir.mkdir(parents=True, exist_ok=True)

    for slice_id in range(args.slice_num):
        src_slice_dir = install_dir / f"slice{slice_id:02d}"
        dst_slice_dir = output_install_dir / f"slice{slice_id:02d}"
        if not src_slice_dir.exists():
            raise FileNotFoundError(f"Missing slice directory: {src_slice_dir}")
        if not args.in_place:
            copy_non_b_files(src_slice_dir, dst_slice_dir)
        rewrite_one_slice(
            slice_id=slice_id,
            src_slice_dir=src_slice_dir,
            dst_slice_dir=dst_slice_dir,
            dram_ag_dir=dram_ag_dir,
            expected_elems=expected_elems,
            group_lines=args.group_lines,
        )

    print(f"[OK] Rewritten install directory ready at {output_install_dir}")


if __name__ == "__main__":
    main()
