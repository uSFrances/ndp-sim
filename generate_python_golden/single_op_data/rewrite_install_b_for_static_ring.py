from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_SLICE_NUM = 28
DEFAULT_N = 896
DEFAULT_K = 896
REG_K_ROWS = 32  # reg_k=16 -> one logical K block spans 32 fp16 rows
RING_SLICE_ORDER = [
    0, 12, 13, 15, 17, 19, 21, 23, 25, 27, 26, 10, 11, 9,
    8, 24, 22, 20, 18, 16, 14, 2, 4, 6, 7, 5, 3, 1,
]

INPUT_WEIGHT_NAME = "matrix_B_linearized_128bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite per-slice install tensorB files so sequential K-block reads match "
            "the current ring-order simulator's dynamic B selection."
        ),
        epilog=(
            "Example:\n"
            "  python rewrite_install_b_for_static_ring.py "
            "--install-dir ./_tmp_gemv_n2n_n32_install_test/install "
            "--n 896 --k 896 --slice-num 28\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--slice-num", type=int, default=DEFAULT_SLICE_NUM)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite files in place. If omitted, write into <install-dir>_static_ring.",
    )
    return parser.parse_args()


def build_ring_maps(slice_num: int) -> tuple[dict[int, int], dict[int, int]]:
    if len(RING_SLICE_ORDER) != slice_num:
        raise ValueError(
            f"RING_SLICE_ORDER length {len(RING_SLICE_ORDER)} does not match slice_num {slice_num}"
        )
    expected = set(range(slice_num))
    actual = set(RING_SLICE_ORDER)
    if actual != expected:
        raise ValueError(f"RING_SLICE_ORDER must be a permutation of 0..{slice_num - 1}")
    ring_next_map = {
        src_slice: RING_SLICE_ORDER[(index + 1) % slice_num]
        for index, src_slice in enumerate(RING_SLICE_ORDER)
    }
    ring_prev_map = {dst_slice: src_slice for src_slice, dst_slice in ring_next_map.items()}
    return ring_next_map, ring_prev_map


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


def load_existing_tensor(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected_elems = int(np.prod(shape))
    file_size = path.stat().st_size
    fp16_size = expected_elems * np.dtype(np.float16).itemsize
    fp32_size = expected_elems * np.dtype(np.float32).itemsize

    if file_size == fp16_size:
        values = np.fromfile(path, dtype=np.float16)
    elif file_size == fp32_size:
        values = np.fromfile(path, dtype=np.float32).astype(np.float16)
    else:
        raise ValueError(
            f"Unexpected file size for {path}: got {file_size} bytes, "
            f"expected {fp16_size} (fp16) or {fp32_size} (fp32) for shape {shape}"
        )

    if values.size != expected_elems:
        raise ValueError(
            f"Unexpected element count for {path}: got {values.size}, expected {expected_elems}"
        )
    return values.reshape(shape)


def inverse_relayout_slice_matrix_b_n8_k_n4(
    relayout_flat: np.ndarray,
    *,
    total_k: int,
    local_n: int,
) -> np.ndarray:
    flat = np.asarray(relayout_flat, dtype=np.float16).reshape(-1)
    expected_elems = total_k * local_n
    if flat.size != expected_elems:
        raise ValueError(
            f"Unexpected relayouted B size: got {flat.size} elements, expected {expected_elems}"
        )
    if local_n % 8 != 0:
        raise ValueError(f"Per-slice N={local_n} must be divisible by 8 for N8K.. relayout")

    matrix = np.empty((total_k, local_n), dtype=np.float16)
    n_groups = local_n // 8
    cursor = 0
    for n_group in range(n_groups):
        col_start = n_group * 8
        col_end = col_start + 8
        for k_idx in range(total_k):
            matrix[k_idx, col_start:col_end] = flat[cursor : cursor + 8]
            cursor += 8
    return matrix


def relayout_slice_matrix_b_n8_k_n4(slice_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(slice_matrix, dtype=np.float16)
    k_dim, local_n = matrix.shape
    if local_n % 8 != 0:
        raise ValueError(f"Per-slice N={local_n} must be divisible by 8 for N8K.. relayout")

    n_groups = local_n // 8
    relayout = np.empty(k_dim * local_n, dtype=np.float16)
    cursor = 0
    for n_group in range(n_groups):
        col_start = n_group * 8
        col_end = col_start + 8
        for k_idx in range(k_dim):
            relayout[cursor : cursor + 8] = matrix[k_idx, col_start:col_end]
            cursor += 8
    return relayout


def compute_dynamic_source_sequence(
    *,
    slice_id: int,
    slice_num: int,
    ring_prev_map: dict[int, int],
) -> list[int]:
    source = slice_id
    order = []
    for _ in range(slice_num):
        order.append(source)
        source = ring_prev_map[source]
    return order


def rewrite_one_slice_b(
    *,
    src_slice_dir: Path,
    dst_slice_dir: Path,
    slice_id: int,
    total_k: int,
    local_n: int,
    source_sequence: list[int],
) -> None:
    src_bin = src_slice_dir / f"{INPUT_WEIGHT_NAME}.bin"
    if not src_bin.exists():
        raise FileNotFoundError(f"Missing source B file: {src_bin}")

    relayout_flat = load_existing_tensor(src_bin, (total_k * local_n,))
    raw_matrix = inverse_relayout_slice_matrix_b_n8_k_n4(
        relayout_flat,
        total_k=total_k,
        local_n=local_n,
    )

    if total_k % len(source_sequence) != 0:
        raise ValueError(
            f"Total K={total_k} must be divisible by slice_num={len(source_sequence)} to permute K blocks"
        )
    rows_per_block = total_k // len(source_sequence)
    if rows_per_block != REG_K_ROWS:
        raise ValueError(
            f"Expected rows_per_block={REG_K_ROWS}, got {rows_per_block}. "
            "Update REG_K_ROWS if reg_k changes."
        )

    remapped_matrix = np.empty_like(raw_matrix)
    for new_block_idx, old_block_idx in enumerate(source_sequence):
        src_start = old_block_idx * rows_per_block
        src_end = src_start + rows_per_block
        dst_start = new_block_idx * rows_per_block
        dst_end = dst_start + rows_per_block
        remapped_matrix[dst_start:dst_end, :] = raw_matrix[src_start:src_end, :]

    remapped_relayout = relayout_slice_matrix_b_n8_k_n4(remapped_matrix)

    dst_slice_dir.mkdir(parents=True, exist_ok=True)
    dst_bin = dst_slice_dir / f"{INPUT_WEIGHT_NAME}.bin"
    dst_txt = dst_slice_dir / f"{INPUT_WEIGHT_NAME}.txt"
    dst_hex = dst_slice_dir / f"{INPUT_WEIGHT_NAME}_{total_k * local_n}x1_hex.txt"

    dst_bin.write_bytes(remapped_relayout.astype(np.float16).reshape(-1).tobytes())
    dst_txt.write_text(
        bytes_to_qgen_style_128bit_txt(remapped_relayout.astype(np.float16).reshape(-1).tobytes()),
        encoding="utf-8",
    )
    dst_hex.write_text(
        "\n".join(fp16_hex_lines(remapped_relayout)) + "\n",
        encoding="utf-8",
    )

    print(
        f"[OK] slice{slice_id:02d}: remapped B blocks "
        f"{source_sequence} -> sequential 0..{len(source_sequence) - 1}"
    )


def copy_non_b_files(src_slice_dir: Path, dst_slice_dir: Path) -> None:
    for path in src_slice_dir.iterdir():
        if path.name.startswith(INPUT_WEIGHT_NAME):
            continue
        if path.is_file():
            dst_slice_dir.mkdir(parents=True, exist_ok=True)
            (dst_slice_dir / path.name).write_bytes(path.read_bytes())


def main() -> None:
    args = parse_args()
    install_dir = args.install_dir.resolve()
    if not install_dir.exists():
        raise FileNotFoundError(f"Install directory not found: {install_dir}")
    if args.n % args.slice_num != 0:
        raise ValueError(f"N={args.n} must be divisible by slice_num={args.slice_num}")
    if args.k % args.slice_num != 0:
        raise ValueError(f"K={args.k} must be divisible by slice_num={args.slice_num}")

    local_n = args.n // args.slice_num
    ring_next_map, ring_prev_map = build_ring_maps(args.slice_num)
    _ = ring_next_map

    if args.in_place:
        output_install_dir = install_dir
    else:
        output_install_dir = install_dir.parent / f"{install_dir.name}_static_ring"
        output_install_dir.mkdir(parents=True, exist_ok=True)

    for slice_id in range(args.slice_num):
        src_slice_dir = install_dir / f"slice{slice_id:02d}"
        dst_slice_dir = output_install_dir / f"slice{slice_id:02d}"
        if not src_slice_dir.exists():
            raise FileNotFoundError(f"Missing slice directory: {src_slice_dir}")

        if not args.in_place:
            copy_non_b_files(src_slice_dir, dst_slice_dir)

        source_sequence = compute_dynamic_source_sequence(
            slice_id=slice_id,
            slice_num=args.slice_num,
            ring_prev_map=ring_prev_map,
        )
        rewrite_one_slice_b(
            src_slice_dir=src_slice_dir,
            dst_slice_dir=dst_slice_dir,
            slice_id=slice_id,
            total_k=args.k,
            local_n=local_n,
            source_sequence=source_sequence,
        )

    print(f"[OK] Rewritten install directory ready at {output_install_dir}")


if __name__ == "__main__":
    main()
