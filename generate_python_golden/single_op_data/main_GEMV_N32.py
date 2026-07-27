import argparse
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from component.Buffer import Buffer
from component.DRAM import DRAM
from component.DataTransfer import (
    GEMV_LOCAL_N32_PROFILE,
    run_ag_to_buffer,
    run_buffer_to_pe,
    run_dram_to_ag,
    set_gemm_buffer_base_addrs,
)
from component.SpecialPEA import SpecialPEA
from utils.io_utils import reset_dynamic_dirs, reset_output_files


OUTPUT_FILES: list[str] = []

SLICE_NUM = 16
BANK_NUM = 4
ROW_NUM = 6144
COL_NUM = 64
SUBWORD_SIZE = 16
BYTES_PER_ROW = COL_NUM * SUBWORD_SIZE

BUFFER_ROW_LC_INDEX = 3
BUFFER_COL_LC_INDEX = 4

INPUT_LAYOUT_B = "K2N8"

GEMM_TENSOR_BASE_ADDRS = {
    "input": 0,
    "weight": 16384,
    "output": 229376,
}

LOCAL_TENSOR_FILENAMES = {
    "input": "matrix_A_linearized_128bit.bin",
    "weight": "matrix_B_linearized_128bit.bin",
}

DYNAMIC_DIR_NAMES = [
    "buffer_dump",
    "special_pea",
    "buffer_pe",
    "dram_ag",
    "ag_buffer",
    "special_pea_writeback",
    "hex_data",
]

B_SUBTILE_BYTES = 128
K2_ROWS_PER_BUFFER = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local GEMV with configurable register grouping.")
    parser.add_argument("--gemm-install-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True, help="Logical GEMV output width.")
    parser.add_argument("--k", type=int, required=True, help="Logical GEMV reduction size.")
    parser.add_argument("--reg-n", type=int, required=True, help="Number of output groups represented by 16 PE registers.")
    parser.add_argument("--reg-k", type=int, required=True, help="Number of register slots reduced into one output.")
    parser.add_argument(
        "--strict-shape-check",
        action="store_true",
        help="Fail if the install file size does not match the expected tensor shape exactly.",
    )
    return parser.parse_args()


def get_gemm_tensor_base_addr(role: str) -> int:
    try:
        return GEMM_TENSOR_BASE_ADDRS[role]
    except KeyError as exc:
        raise ValueError(f"Unknown GEMM tensor role: {role}") from exc


def sync_gemm_base_addrs_to_data_transfer() -> None:
    set_gemm_buffer_base_addrs(
        input_addr=get_gemm_tensor_base_addr("input"),
        weight_addr=get_gemm_tensor_base_addr("weight"),
        output_addr=get_gemm_tensor_base_addr("output"),
    )


def resolve_tensor_root(gemm_install_dir: Path) -> Path:
    direct_input = gemm_install_dir / LOCAL_TENSOR_FILENAMES["input"]
    direct_weight = gemm_install_dir / LOCAL_TENSOR_FILENAMES["weight"]
    if direct_input.exists() and direct_weight.exists():
        return gemm_install_dir

    slice_dir = gemm_install_dir / "slice00"
    slice_input = slice_dir / LOCAL_TENSOR_FILENAMES["input"]
    slice_weight = slice_dir / LOCAL_TENSOR_FILENAMES["weight"]
    if slice_input.exists() and slice_weight.exists():
        return slice_dir

    raise FileNotFoundError(
        "Could not find local GEMV tensors. Expected files either directly under "
        f"{gemm_install_dir} or under {slice_dir}."
    )


def load_fp16_tensor(path: Path, shape: tuple[int, ...], strict_shape_check: bool) -> np.ndarray:
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

    if strict_shape_check and values.size != expected_elems:
        raise ValueError(
            f"Unexpected element count for {path}: got {values.size}, expected {expected_elems}"
        )

    if values.size < expected_elems:
        raise ValueError(f"Tensor {path} is too small: got {values.size}, expected {expected_elems}")

    return values[:expected_elems].reshape(shape)


def pack_fp16_bytes(values_fp16: np.ndarray) -> np.ndarray:
    values_fp16 = np.asarray(values_fp16, dtype=np.float16).reshape(-1)
    values_u16 = values_fp16.view(np.uint16)
    packed = np.empty(values_u16.size * 2, dtype=np.uint8)
    packed[0::2] = values_u16 & 0xFF
    packed[1::2] = (values_u16 >> 8) & 0xFF
    return packed


def write_bytes_to_bank0_image(image: np.ndarray, dram: DRAM, base_addr: int, payload: np.ndarray) -> None:
    bytes_payload = np.asarray(payload, dtype=np.uint8).reshape(-1)
    for byte_offset, byte_val in enumerate(bytes_payload):
        address = base_addr + byte_offset
        mapped_slice, bank_id, row_id, col_id, subword_offset = dram.addrmap(address)
        if mapped_slice != 0 or bank_id != 0:
            raise ValueError(
                f"Address 0x{address:X} mapped outside local bank0: "
                f"slice={mapped_slice}, bank={bank_id}"
            )
        image[row_id, col_id * SUBWORD_SIZE + subword_offset] = byte_val


def write_bank0_image_to_hex(image: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fp:
        for row_id in range(image.shape[0]):
            row_bytes = image[row_id]
            fp.write(" ".join(f"{int(byte):02X}" for byte in row_bytes))
            if row_id != image.shape[0] - 1:
                fp.write("\n")


def load_bank0_image_into_dram(dram: DRAM, image: np.ndarray) -> None:
    for row_id in range(image.shape[0]):
        payload = np.asarray(image[row_id], dtype=np.uint8)
        dram.data[0, 0, row_id, :, :] = payload.view(np.int8).reshape(COL_NUM, SUBWORD_SIZE)


def build_a_bytes(vector_a: np.ndarray) -> np.ndarray:
    return pack_fp16_bytes(vector_a.reshape(-1))


def clone_buffer(src: Buffer) -> Buffer:
    dst = Buffer(col_num=src.col_num, row_num=src.row_num, bitwidth=src.bitwidth, dtype=src.data.dtype)
    np.copyto(dst.data, src.data)
    np.copyto(dst.tag_last, src.tag_last)
    np.copyto(dst.tag_last_index, src.tag_last_index)
    np.copyto(dst.tag_branch, src.tag_branch)
    return dst


def build_b_layout_bytes(matrix_b: np.ndarray, n: int, k: int, reg_n: int, reg_k: int) -> np.ndarray:
    outputs_per_pass = 8 * reg_n
    num_passes = n // outputs_per_pass
    if reg_k % K2_ROWS_PER_BUFFER != 0:
        raise ValueError("reg_k must be divisible by 4 because each ping/pong buffer stores K4K2N8")

    k_blocks = k // (reg_k * 2)
    ags_per_round = reg_k // K2_ROWS_PER_BUFFER
    pass_bytes_per_kblock = B_SUBTILE_BYTES * reg_n * ags_per_round
    total_bytes = num_passes * k_blocks * pass_bytes_per_kblock
    payload = np.zeros(total_bytes, dtype=np.uint8)

    for pass_idx in range(num_passes):
        pass_col_base = pass_idx * outputs_per_pass
        pass_base = pass_idx * k_blocks * pass_bytes_per_kblock
        for k_block in range(k_blocks):
            block_base = pass_base + k_block * pass_bytes_per_kblock
            for round_idx in range(reg_n):
                col_base = pass_col_base + round_idx * 8
                for ag_idx in range(ags_per_round):
                    subtile_base = block_base + (round_idx * ags_per_round + ag_idx) * B_SUBTILE_BYTES
                    for row_idx in range(K2_ROWS_PER_BUFFER):
                        local_k2 = ag_idx * K2_ROWS_PER_BUFFER + row_idx
                        logical_k0 = k_block * (reg_k * 2) + local_k2 * 2
                        logical_k1 = logical_k0 + 1
                        tile_values = np.concatenate(
                            [
                                matrix_b[logical_k0, col_base:col_base + 8],
                                matrix_b[logical_k1, col_base:col_base + 8],
                            ]
                        )
                        tile_bytes = pack_fp16_bytes(tile_values)
                        row_base = subtile_base + row_idx * 32
                        payload[row_base:row_base + tile_bytes.size] = tile_bytes

    return payload


def rebuild_hex_data_from_gemv(
    dram: DRAM,
    hex_data_dir: Path,
    tensor_root: Path,
    n: int,
    k: int,
    reg_n: int,
    reg_k: int,
    strict_shape_check: bool,
) -> tuple[np.ndarray, np.ndarray]:
    hex_data_dir.mkdir(parents=True, exist_ok=True)

    input_path = tensor_root / LOCAL_TENSOR_FILENAMES["input"]
    weight_path = tensor_root / LOCAL_TENSOR_FILENAMES["weight"]
    vector_a = load_fp16_tensor(input_path, (1, k), strict_shape_check)
    matrix_b = load_fp16_tensor(weight_path, (k, n), strict_shape_check)

    bank0_image = np.zeros((ROW_NUM, BYTES_PER_ROW), dtype=np.uint8)
    write_bytes_to_bank0_image(
        bank0_image,
        dram,
        get_gemm_tensor_base_addr("input"),
        build_a_bytes(vector_a),
    )
    write_bytes_to_bank0_image(
        bank0_image,
        dram,
        get_gemm_tensor_base_addr("weight"),
        build_b_layout_bytes(matrix_b, n=n, k=k, reg_n=reg_n, reg_k=reg_k),
    )

    load_bank0_image_into_dram(dram, bank0_image)
    write_bank0_image_to_hex(bank0_image, hex_data_dir / "dram_data_slice0_bank0.txt")
    return vector_a.reshape(-1), matrix_b


def _load_abuffer_m1k2(buffer_a: Buffer, tags, active_bytes: int = 4) -> None:
    buffer_a.clear()
    valid_tags = [t for t in tags if t.get("valid", 1) == 1 and t.get("padding", 0) == 0]
    if len(valid_tags) < active_bytes:
        raise ValueError(
            f"Insufficient A tags for M1K2: need {active_bytes}, got {len(valid_tags)}"
        )

    for c in range(active_bytes):
        tag = valid_tags[c]
        buffer_a.write(tag["data"], 0, c, branch=tag.get("branch", 0))

    for r in range(buffer_a.row_num):
        for c in range(buffer_a.col_num):
            if r == 0 and c < active_bytes:
                continue
            buffer_a.write(0, r, c, branch=1)


def collect_pe_registers(special_pea: SpecialPEA) -> np.ndarray:
    registers = np.zeros((special_pea.col_num, 16), dtype=np.float32)
    for pe_col in range(special_pea.col_num):
        pe = special_pea.pe_array[0][pe_col]
        for cycle in range(16):
            row_idx, col_idx = special_pea._cycle_to_psum_coords(cycle)
            registers[pe_col, cycle] = np.float32(pe.psum_array.read(row_idx, col_idx))
    return registers


def reduce_registers(registers: np.ndarray, reg_n: int, reg_k: int) -> np.ndarray:
    reduced = np.zeros((registers.shape[0], reg_n), dtype=np.float32)
    for pe_col in range(registers.shape[0]):
        reduced[pe_col, :] = registers[pe_col].reshape(reg_n, reg_k).sum(axis=1, dtype=np.float32)
    return reduced


def flatten_outputs_for_writeback(reduced: np.ndarray) -> np.ndarray:
    reg_n = reduced.shape[1]
    outputs = np.zeros(reduced.shape[0] * reg_n, dtype=np.float32)
    cursor = 0
    for group_idx in range(reg_n):
        for pe_col in range(reduced.shape[0] - 1, -1, -1):
            outputs[cursor] = reduced[pe_col, group_idx]
            cursor += 1
    return outputs


def fill_output_buffer(buffer: Buffer, outputs_fp16: np.ndarray) -> None:
    buffer.clear()
    byte_stream = pack_fp16_bytes(outputs_fp16)
    if byte_stream.size > buffer.row_num * buffer.col_num:
        raise ValueError("Output buffer is too small for this pass")

    cursor = 0
    remaining = byte_stream.size
    for row_id in range(buffer.row_num):
        if remaining <= 0:
            break
        row_bytes = min(buffer.col_num, remaining)
        # Buffer.write() reverses logical columns into physical storage.
        # For a partially used row (e.g. 8 fp16 = 16B), start from the higher
        # logical columns so the dumped physical row shows valid data first:
        # PE0 PE1 ... instead of leading zeros followed by payload.
        start_col = buffer.col_num - row_bytes
        for offset in range(row_bytes):
            buffer.write(int(byte_stream[cursor + offset]), row_id, start_col + offset, branch=0)
        cursor += row_bytes
        remaining -= row_bytes


def build_buffer5_row_outputs(
    outputs_fp16: np.ndarray,
    *,
    reg_n: int,
    pending_half_row: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    def _reverse_pe_word_order(row_fp16: np.ndarray) -> np.ndarray:
        pairs = np.asarray(row_fp16, dtype=np.float16).reshape(-1, 2)
        return pairs[::-1].reshape(-1).copy()

    current = np.asarray(outputs_fp16, dtype=np.float16).reshape(-1)

    if reg_n == 2:
        return _reverse_pe_word_order(current), None

    if reg_n == 1:
        if pending_half_row is None:
            return None, current.copy()

        if pending_half_row.size != current.size:
            raise ValueError("Mismatched pending/current half-row sizes for buffer5 packing")

        packed = np.empty(current.size * 2, dtype=np.float16)
        cursor = 0
        for lane in range(current.size):
            packed[cursor] = pending_half_row[lane]
            packed[cursor + 1] = current[lane]
            cursor += 2
        return _reverse_pe_word_order(packed), None

    return current.copy(), None


def write_output_vector_to_dram(dram: DRAM, outputs_fp16: np.ndarray) -> None:
    base_addr = get_gemm_tensor_base_addr("output")
    write_bytes = pack_fp16_bytes(outputs_fp16)
    for byte_offset, byte_val in enumerate(write_bytes):
        slice_id, bank_id, row_id, col_id, subword_offset = dram.addrmap(base_addr + byte_offset)
        dram.write(slice_id, bank_id, row_id, col_id, subword_offset, np.int8(byte_val))


def dump_output_vector(outputs_fp16: np.ndarray, output_root: Path) -> None:
    output_path = output_root / "gemv_output_vector_fp16.txt"
    with output_path.open("a", encoding="utf-8") as fp:
        fp.write(" ".join(f"0x{int(val.view(np.uint16)):04X}" for val in outputs_fp16.astype(np.float16)))
        fp.write("\n")


def dump_special_pea_cycle_state(
    special_pea: SpecialPEA,
    output_root: Path,
    *,
    n_outer_idx: int,
    k_outer_idx: int,
    n_reg_idx: int,
    k_reg_idx: int,
    cycle: int,
) -> None:
    outputs, branch_map = special_pea._collect_outputs_for_cycle(cycle)
    reg_idx = cycle
    output_path = output_root / "special_pea" / "special_pea_cycle_results.txt"
    with output_path.open("a", encoding="utf-8") as fp:
        fp.write(
            f"# N_outer(N//n_reg//8)={n_outer_idx} "
            f"K_outer(K//k_reg//2)={k_outer_idx} "
            f"n_reg_idx={n_reg_idx} "
            f"k_reg_idx={k_reg_idx} "
            f"reg_idx={reg_idx}\n"
        )
        for row_idx in range(outputs.shape[0]):
            row_hex = " ".join(f"0x{int(val):08X}" for val in outputs[row_idx].view(np.uint32))
            branch_bits = " ".join(str(int(bit)) for bit in branch_map[row_idx])
            fp.write(f"row{row_idx:02d} data=[{row_hex}] branch=[{branch_bits}]\n")


def prepare_general_pe_cycle_records(
    special_pea: SpecialPEA,
    buffer_a: Buffer,
    buffer_b: Buffer,
    *,
    cycle: int,
    gemv_paired_rows: bool = False,
) -> list[dict[str, object]]:
    def _row_all_branched(buf: Buffer, row_idx: int) -> bool:
        _, _, branch_row = buf.read_tags_row(row_idx)
        return bool(np.all(np.asarray(branch_row, dtype=np.uint8) == 1))

    gemv_mode = all(_row_all_branched(buffer_a, r) for r in range(1, buffer_a.row_num))
    if gemv_paired_rows:
        a_row_idx = cycle % buffer_a.row_num
        b_row_idx = cycle % buffer_b.row_num
    elif gemv_mode:
        a_row_idx = 0
        b_row_idx = cycle % buffer_b.row_num
    else:
        a_row_idx = cycle % buffer_a.row_num
        b_row_idx = cycle // buffer_a.row_num

    reversed_vec_a = buffer_a.read_row(a_row_idx)[::-1]
    reversed_vec_b = buffer_b.read_row(b_row_idx)[::-1]
    _, _, branch_row_a = buffer_a.read_tags_row(a_row_idx)
    _, _, branch_row_b = buffer_b.read_tags_row(b_row_idx)
    branch_elem_a = special_pea._branch_row_to_elements(branch_row_a[::-1])
    branch_elem_b = special_pea._branch_row_to_elements(branch_row_b[::-1])

    byte_data_a = reversed_vec_a.view(np.uint8)
    byte_data_b = reversed_vec_b.view(np.uint8)
    elem_a = np.frombuffer(byte_data_a.tobytes(), dtype="<f2")
    elem_b = np.frombuffer(byte_data_b.tobytes(), dtype="<f2")
    psum_row_idx, psum_col_idx = special_pea._cycle_to_psum_coords(cycle)

    records: list[dict[str, object]] = []
    for i in range(special_pea.col_num):
        start_a = 0
        end_a = start_a + special_pea.dot_size
        start_b = i * special_pea.dot_size
        end_b = start_b + special_pea.dot_size

        vec_a_pe = elem_a[start_a:end_a].astype(np.float16, copy=True)
        vec_b_pe = elem_b[start_b:end_b].astype(np.float16, copy=True)
        branch_a_pe = np.asarray(branch_elem_a[start_a:end_a], dtype=np.uint8)
        branch_b_pe = np.asarray(branch_elem_b[start_b:end_b], dtype=np.uint8)

        vec_a_calc = vec_a_pe.astype(np.float16, copy=True)
        vec_b_calc = vec_b_pe.astype(np.float16, copy=True)
        vec_a_calc[np.asarray(branch_a_pe, dtype=bool)] = np.float16(0)
        vec_b_calc[np.asarray(branch_b_pe, dtype=bool)] = np.float16(0)

        pe_col = special_pea.col_num - 1 - i
        pe = special_pea.pe_array[0][pe_col]
        psum_in = np.float32(pe.psum_array.read(psum_row_idx, psum_col_idx))
        dot_val = np.float32(np.dot(vec_a_calc.astype(np.float32), vec_b_calc.astype(np.float32)))

        records.append(
            {
                "logical_lane": pe_col,
                "storage_lane": i,
                "vec_a": vec_a_pe.copy(),
                "vec_b": vec_b_pe.copy(),
                "branch_a": branch_a_pe.copy(),
                "branch_b": branch_b_pe.copy(),
                "dot": dot_val,
                "psum_in": psum_in,
            }
        )
    return records


def dump_special_active_pe_cycle_results(
    special_pea: SpecialPEA,
    output_root: Path,
    *,
    n_outer_idx: int,
    k_outer_idx: int,
    n_reg_idx: int,
    k_reg_idx: int,
    cycle: int,
    records: list[dict[str, object]],
) -> None:
    def _fp16_hex(value: np.float16) -> str:
        return f"0x{int(np.float16(value).view(np.uint16)):04X}"

    def _fp32_hex(value: np.float32) -> str:
        return f"0x{int(np.float32(value).view(np.uint32)):08X}"

    reg_idx = cycle
    output_path = output_root / "general_pe" / "special_active_pe_cycle_results.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fp:
        fp.write(
            f"# derived_from_special_pe_execute "
            f"N_outer(N//n_reg//8)={n_outer_idx} "
            f"K_outer(K//k_reg//2)={k_outer_idx} "
            f"n_reg_idx={n_reg_idx} "
            f"k_reg_idx={k_reg_idx} "
            f"reg_idx={reg_idx}\n"
        )
        for record in records:
            pe_col = int(record["logical_lane"])
            pe = special_pea.pe_array[0][pe_col]
            psum_row_idx, psum_col_idx = special_pea._cycle_to_psum_coords(cycle)
            psum_out = np.float32(pe.psum_array.read(psum_row_idx, psum_col_idx))
            vec_a_hex = " ".join(_fp16_hex(val) for val in np.asarray(record["vec_a"], dtype=np.float16))
            vec_b_hex = " ".join(_fp16_hex(val) for val in np.asarray(record["vec_b"], dtype=np.float16))
            branch_a = " ".join(str(int(bit)) for bit in np.asarray(record["branch_a"], dtype=np.uint8))
            branch_b = " ".join(str(int(bit)) for bit in np.asarray(record["branch_b"], dtype=np.uint8))
            fp.write(
                f"pe_row=0 pe_col={pe_col} storage_lane={int(record['storage_lane'])} "
                f"A=[{vec_a_hex}] B=[{vec_b_hex}] "
                f"branch_a=[{branch_a}] branch_b=[{branch_b}] "
                f"dot={_fp32_hex(np.float32(record['dot']))} "
                f"psum_in={_fp32_hex(np.float32(record['psum_in']))} "
                f"psum_out={_fp32_hex(psum_out)}\n"
            )


def dump_general_array_accumulation_results(
    registers: np.ndarray,
    reduced: np.ndarray,
    output_root: Path,
    *,
    n_outer_idx: int,
    reg_n: int,
    reg_k: int,
) -> None:
    output_path = output_root / "general_pe" / "general_array_accumulation_results.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fp:
        fp.write(
            f"# reduction_summary N_outer(N//n_reg//8)={n_outer_idx} "
            f"reg_n={reg_n} reg_k={reg_k}\n"
        )
        for n_reg_idx in range(reg_n):
            start = n_reg_idx * reg_k
            end = start + reg_k
            fp.write(
                f"## output_group n_reg_idx={n_reg_idx} reg_range=[{start}:{end})\n"
            )
            for pe_col in range(registers.shape[0] - 1, -1, -1):
                lane_values = registers[pe_col, start:end].astype(np.float32)
                lane_hex = " ".join(
                    f"0x{int(np.float32(val).view(np.uint32)):08X}" for val in lane_values
                )
                lane_fp16_hex = " ".join(
                    f"0x{int(np.float16(val).view(np.uint16)):04X}" for val in lane_values
                )
                reduced_val = np.float32(reduced[pe_col, n_reg_idx])
                reduced_fp16 = np.float16(reduced_val)
                fp.write(
                    f"pe_col={pe_col} lane_values=[{lane_hex}] "
                    f"lane_values_fp16=[{lane_fp16_hex}] "
                    f"reduced_sum=0x{int(reduced_val.view(np.uint32)):08X} "
                    f"reduced_sum_fp16=0x{int(reduced_fp16.view(np.uint16)):04X}\n"
                )


def dump_output_writeback_trace(
    outputs_fp16: np.ndarray,
    output_root: Path,
    *,
    n_outer_idx: int,
    output_base_addr: int,
) -> None:
    payload = outputs_fp16.astype(np.float16)
    output_path = output_root / "dram_ag" / "ag5_results.txt"
    with output_path.open("a", encoding="utf-8") as fp:
        for chunk_idx in range(0, payload.size, 8):
            chunk = payload[chunk_idx:chunk_idx + 8]
            addr = output_base_addr + n_outer_idx * payload.size * 2 + chunk_idx * 2
            fp.write(
                f"The transfer addr address: {addr // 16:<10}    "
                f"The data: {' '.join(f'{int(val.view(np.uint16)):04x}' for val in chunk)}    "
                f"The transfer size: {chunk.size * 2:<5}    "
                f"The logic index(N_outer, N8_chunk): {(n_outer_idx, chunk_idx // 8)}\n"
            )


def reference_grouped_gemv(
    vector_a: np.ndarray,
    matrix_b: np.ndarray,
    n: int,
    reg_n: int,
    reg_k: int,
) -> np.ndarray:
    del n, reg_n, reg_k
    return (vector_a.astype(np.float32) @ matrix_b.astype(np.float32)).astype(np.float16)


def run_simulation(dram: DRAM, output_root: Path, n: int, k: int, reg_n: int, reg_k: int) -> np.ndarray:
    outputs_per_pass = 8 * reg_n
    num_passes = n // outputs_per_pass
    if reg_k % K2_ROWS_PER_BUFFER != 0:
        raise ValueError("reg_k must be divisible by 4 because each ping/pong buffer stores K4K2N8")

    k_blocks = k // (reg_k * 2)
    ags_per_round = reg_k // K2_ROWS_PER_BUFFER
    pass_bytes_per_kblock = B_SUBTILE_BYTES * reg_n * ags_per_round

    buffer_a = Buffer(col_num=32, row_num=4, bitwidth=8)
    output_buffer_rows = max(1, math.ceil(outputs_per_pass / 16))
    output_buffer = Buffer(col_num=32, row_num=output_buffer_rows, bitwidth=8)
    special_pea = SpecialPEA(col_num=8, row_num=8, dot_size=2, psum_bitwidth=32, datatype="fp16")

    aggregated_outputs = np.zeros(n, dtype=np.float16)
    pending_buffer5_half_row: Optional[np.ndarray] = None
    weight_base = get_gemm_tensor_base_addr("weight")

    for pass_idx in tqdm(range(num_passes), desc="GEMV pass", position=0):
        special_pea.clear_psums()
        pass_base = weight_base + pass_idx * k_blocks * pass_bytes_per_kblock

        for k_block in tqdm(range(k_blocks), desc="K-block loop", leave=False, position=1):
            block_base = pass_base + k_block * pass_bytes_per_kblock
            round_buffers: list[list[Buffer]] = []

            for round_idx in range(reg_n):
                round_buffer_group: list[Buffer] = []
                for ag_idx in range(ags_per_round):
                    work_buffer = Buffer(col_num=32, row_num=4, bitwidth=8)
                    buffer_id = 2 if (ag_idx % 2 == 0) else 3
                    base_subtile = block_base + (round_idx * ags_per_round + ag_idx) * B_SUBTILE_BYTES
                    full_unpacked = []
                    for row_idx in range(K2_ROWS_PER_BUFFER):
                        full_unpacked.extend(
                            run_dram_to_ag(
                                dram,
                                buffer_id,
                                0,
                                row_idx,
                                0,
                                0,
                                datatype="fp16",
                                address_profile=GEMV_LOCAL_N32_PROFILE,
                                base_addr_override=base_subtile,
                            )
                        )

                    run_ag_to_buffer(
                        work_buffer,
                        buffer_id,
                        full_unpacked,
                        k_block,
                        round_idx,
                        pass_idx,
                        [],
                        (BUFFER_ROW_LC_INDEX, BUFFER_COL_LC_INDEX),
                        0,
                        datatype="fp16",
                        input_layout=INPUT_LAYOUT_B,
                    )
                    run_buffer_to_pe(
                        work_buffer,
                        buffer_id,
                        k_block,
                        round_idx,
                        pass_idx,
                        0,
                        datatype="fp16",
                        trace_tag=f"pass={pass_idx}_round={round_idx}_Kblk={k_block}_AG={ag_idx}",
                    )
                    round_buffer_group.append(clone_buffer(work_buffer))
                round_buffers.append(round_buffer_group)

            for round_idx in range(reg_n):
                for local_k2 in range(reg_k):
                    global_k2 = k_block * reg_k + local_k2
                    unpacked_a = run_dram_to_ag(
                        dram,
                        0,
                        global_k2,
                        0,
                        0,
                        1,
                        datatype="fp16",
                        address_profile=GEMV_LOCAL_N32_PROFILE,
                    )
                    _load_abuffer_m1k2(buffer_a, unpacked_a, active_bytes=4)
                    run_buffer_to_pe(
                        buffer_a,
                        0,
                        global_k2,
                        0,
                        0,
                        1,
                        datatype="fp16",
                        trace_tag=f"pass={pass_idx}_K2={global_k2}_A",
                    )

                    active_buffer_idx = local_k2 // K2_ROWS_PER_BUFFER
                    active_buffer = round_buffers[round_idx][active_buffer_idx]
                    cycle = round_idx * reg_k + local_k2
                    general_pe_records = prepare_general_pe_cycle_records(
                        special_pea,
                        buffer_a,
                        active_buffer,
                        cycle=cycle,
                    )
                    special_pea.execute(buffer_a, active_buffer, cycle=cycle)
                    dump_special_pea_cycle_state(
                        special_pea,
                        output_root,
                        n_outer_idx=pass_idx,
                        k_outer_idx=k_block,
                        n_reg_idx=round_idx,
                        k_reg_idx=local_k2,
                        cycle=cycle,
                    )
                    dump_special_active_pe_cycle_results(
                        special_pea,
                        output_root,
                        n_outer_idx=pass_idx,
                        k_outer_idx=k_block,
                        n_reg_idx=round_idx,
                        k_reg_idx=local_k2,
                        cycle=cycle,
                        records=general_pe_records,
                    )

        registers = collect_pe_registers(special_pea)
        reduced = reduce_registers(registers, reg_n=reg_n, reg_k=reg_k)
        dump_general_array_accumulation_results(
            registers,
            reduced,
            output_root,
            n_outer_idx=pass_idx,
            reg_n=reg_n,
            reg_k=reg_k,
        )
        outputs_fp32 = flatten_outputs_for_writeback(reduced)
        outputs_fp16 = outputs_fp32.astype(np.float16)
        start = pass_idx * outputs_per_pass
        end = start + outputs_per_pass
        aggregated_outputs[start:end] = outputs_fp16
        buffer5_row_outputs, pending_buffer5_half_row = build_buffer5_row_outputs(
            outputs_fp16,
            reg_n=reg_n,
            pending_half_row=pending_buffer5_half_row,
        )
        if buffer5_row_outputs is not None:
            fill_output_buffer(output_buffer, buffer5_row_outputs)
            run_buffer_to_pe(
                output_buffer,
                5,
                pass_idx,
                0,
                0,
                0,
                datatype="fp16",
                trace_tag=f"pass={pass_idx}_buffer5",
            )
        dump_output_vector(outputs_fp16, output_root)
        dump_output_writeback_trace(
            outputs_fp16,
            output_root,
            n_outer_idx=pass_idx,
            output_base_addr=get_gemm_tensor_base_addr("output"),
        )

    write_output_vector_to_dram(dram, aggregated_outputs)
    return aggregated_outputs


def main() -> None:
    args = parse_args()

    if args.n <= 0 or args.k <= 0:
        raise ValueError("N and K must be positive")
    if args.n % 32 != 0:
        raise ValueError("N must be a multiple of 32")
    if args.k % 2 != 0:
        raise ValueError("K must be divisible by 2")
    if args.reg_n <= 0 or args.reg_k <= 0 or args.reg_n * args.reg_k != 16:
        raise ValueError("reg_n and reg_k must be positive and satisfy reg_n * reg_k == 16")
    if args.n % (8 * args.reg_n) != 0:
        raise ValueError("N must be divisible by outputs_per_pass = 8 * reg_n")

    gemm_install_dir = args.gemm_install_dir.resolve()
    if not gemm_install_dir.exists():
        raise FileNotFoundError(f"Install directory not found: {gemm_install_dir}")
    tensor_root = resolve_tensor_root(gemm_install_dir)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    try:
        os.chdir(output_root)
        reset_output_files(OUTPUT_FILES)
        reset_dynamic_dirs([Path(name) for name in DYNAMIC_DIR_NAMES])
        sync_gemm_base_addrs_to_data_transfer()

        dram = DRAM(
            slice_num=SLICE_NUM,
            bank_num=BANK_NUM,
            row_num=ROW_NUM,
            col_num=COL_NUM,
            subword_size=SUBWORD_SIZE,
        )

        vector_a, matrix_b = rebuild_hex_data_from_gemv(
            dram,
            Path("hex_data"),
            tensor_root,
            n=args.n,
            k=args.k,
            reg_n=args.reg_n,
            reg_k=args.reg_k,
            strict_shape_check=args.strict_shape_check,
        )

        simulated = run_simulation(
            dram,
            output_root=Path.cwd(),
            n=args.n,
            k=args.k,
            reg_n=args.reg_n,
            reg_k=args.reg_k,
        )

        golden = reference_grouped_gemv(
            vector_a,
            matrix_b,
            n=args.n,
            reg_n=args.reg_n,
            reg_k=args.reg_k,
        ).reshape(-1)
        np.savetxt("gemv_output_fp16_hex.txt", simulated.view(np.uint16).reshape(1, -1), fmt="0x%04X")
        np.savetxt("gemv_golden_fp16_hex.txt", golden.view(np.uint16).reshape(1, -1), fmt="0x%04X")
        if np.array_equal(simulated.view(np.uint16), golden.view(np.uint16)):
            print("[OK] Simulated GEMV output matches grouped fp16 reference.")
        else:
            mismatch = np.nonzero(simulated.view(np.uint16) != golden.view(np.uint16))[0]
            print(
                f"[WARN] Simulated GEMV differs from grouped fp16 reference at {mismatch.size} positions. "
                f"First mismatch index: {int(mismatch[0]) if mismatch.size else 'n/a'}"
            )
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
