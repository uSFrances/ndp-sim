import argparse
import math
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from component.Buffer import Buffer
from component.DRAM import DRAM
from component.DataTransfer import (
    GEMV_LOCAL_N32_PROFILE,
    run_ag_to_buffer,
    run_buffer_to_pe,
    run_buffer_writeback_to_dram,
    run_dram_to_ag,
    set_trace_dump_enabled,
)
from component.IGA import LoopState
from component.SpecialPEA import SpecialPEA
from main_GEMV_N32 import (
    BANK_NUM,
    BUFFER_COL_LC_INDEX,
    BUFFER_ROW_LC_INDEX,
    COL_NUM,
    DYNAMIC_DIR_NAMES,
    GEMM_TENSOR_BASE_ADDRS,
    K2_ROWS_PER_BUFFER,
    LOCAL_TENSOR_FILENAMES,
    OUTPUT_FILES,
    ROW_NUM,
    SUBWORD_SIZE,
    B_SUBTILE_BYTES,
    BUFFER_ROW_LC_INDEX as INPUT_BUFFER_ROW_LC_INDEX,
    BUFFER_COL_LC_INDEX as INPUT_BUFFER_COL_LC_INDEX,
    build_a_bytes,
    build_b_layout_bytes,
    build_buffer5_row_outputs,
    clone_buffer,
    collect_pe_registers,
    dump_general_array_accumulation_results,
    dump_output_vector,
    dump_output_writeback_trace,
    dump_special_active_pe_cycle_results,
    dump_special_pea_cycle_state,
    fill_output_buffer,
    flatten_outputs_for_writeback,
    get_gemm_tensor_base_addr,
    load_fp16_tensor,
    prepare_general_pe_cycle_records,
    reduce_registers,
    set_gemm_buffer_base_addrs,
    write_bank0_image_to_hex,
    write_bytes_to_bank0_image,
)
from utils.io_utils import reset_dynamic_dirs, reset_output_files


DEFAULT_SLICE_NUM = 28
BYTES_PER_ROW = COL_NUM * SUBWORD_SIZE
RING_SLICE_ORDER = [
    0, 12, 13, 15, 17, 19, 21, 23, 25, 27, 26, 10, 11, 9,
    8, 24, 22, 20, 18, 16, 14, 2, 4, 6, 7, 5, 3, 1,
]

RING_DYNAMIC_DIR_NAMES = list(DYNAMIC_DIR_NAMES) + ["general_pe"]
OUTBUFFER_ROW_LC_INDEX = 4
OUTBUFFER_COL_LC_INDEX = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 28-slice ring2ring GEMV_N32.")
    parser.add_argument("--gemm-install-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True, help="Global logical GEMV output width.")
    parser.add_argument("--k", type=int, required=True, help="Global logical GEMV reduction size.")
    parser.add_argument("--reg-n", type=int, required=True, help="Number of output groups represented by 16 PE registers.")
    parser.add_argument("--reg-k", type=int, required=True, help="Number of register slots reduced into one output.")
    parser.add_argument("--slice-num", type=int, default=DEFAULT_SLICE_NUM, help="Number of ring slices.")
    parser.add_argument(
        "--strict-shape-check",
        action="store_true",
        help="Fail if the install file size does not match the expected tensor shape exactly.",
    )
    parser.add_argument(
        "--lightweight-check",
        action="store_true",
        help="Disable heavy trace dumps for faster end-to-end self-check runs.",
    )
    parser.add_argument(
        "--debug-progress",
        action="store_true",
        help="Print coarse-grained progress logs for long runs.",
    )
    return parser.parse_args()


def sync_gemm_base_addrs_to_data_transfer() -> None:
    set_gemm_buffer_base_addrs(
        input_addr=GEMM_TENSOR_BASE_ADDRS["input"],
        weight_addr=GEMM_TENSOR_BASE_ADDRS["weight"],
        output_addr=GEMM_TENSOR_BASE_ADDRS["output"],
    )


def build_ring_maps(slice_num: int) -> tuple[dict[int, int], dict[int, int]]:
    if len(RING_SLICE_ORDER) != slice_num:
        raise ValueError(
            f"RING_SLICE_ORDER length {len(RING_SLICE_ORDER)} does not match slice_num {slice_num}"
        )

    expected_slices = set(range(slice_num))
    actual_slices = set(RING_SLICE_ORDER)
    if actual_slices != expected_slices:
        raise ValueError(
            f"RING_SLICE_ORDER must be a permutation of 0..{slice_num - 1}, got {RING_SLICE_ORDER}"
        )

    ring_next_map = {
        src_slice: RING_SLICE_ORDER[(index + 1) % slice_num]
        for index, src_slice in enumerate(RING_SLICE_ORDER)
    }
    ring_prev_map = {dst_slice: src_slice for src_slice, dst_slice in ring_next_map.items()}

    current_slice = RING_SLICE_ORDER[0]
    for _ in range(slice_num):
        current_slice = ring_next_map[current_slice]
    if current_slice != RING_SLICE_ORDER[0]:
        raise ValueError("RING_SLICE_ORDER does not form a closed ring")

    return ring_next_map, ring_prev_map


def resolve_slice_dirs(gemm_install_dir: Path, slice_num: int) -> list[Path]:
    direct_slice_dirs = [gemm_install_dir / f"slice{slice_idx:02d}" for slice_idx in range(slice_num)]
    if all((slice_dir / LOCAL_TENSOR_FILENAMES["input"]).exists() for slice_dir in direct_slice_dirs):
        return direct_slice_dirs

    install_dir = gemm_install_dir / "install"
    nested_slice_dirs = [install_dir / f"slice{slice_idx:02d}" for slice_idx in range(slice_num)]
    if all((slice_dir / LOCAL_TENSOR_FILENAMES["input"]).exists() for slice_dir in nested_slice_dirs):
        return nested_slice_dirs

    missing = [str(path) for path in direct_slice_dirs if not (path / LOCAL_TENSOR_FILENAMES["input"]).exists()]
    raise FileNotFoundError(
        "Could not resolve per-slice GEMV tensors. Expected slice directories under "
        f"{gemm_install_dir} or {install_dir}. Missing examples: {missing[:3]}"
    )


def load_slice_tensors(
    slice_dirs: list[Path],
    *,
    total_n: int,
    total_k: int,
    strict_shape_check: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    slice_num = len(slice_dirs)
    if total_n % slice_num != 0:
        raise ValueError(f"Global N={total_n} must be divisible by slice_num={slice_num}")
    if total_k % slice_num != 0:
        raise ValueError(f"Global K={total_k} must be divisible by slice_num={slice_num}")

    local_n = total_n // slice_num
    local_k = total_k // slice_num

    vectors: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    for slice_dir in slice_dirs:
        input_path = slice_dir / LOCAL_TENSOR_FILENAMES["input"]
        weight_path = slice_dir / LOCAL_TENSOR_FILENAMES["weight"]
        vectors.append(load_fp16_tensor(input_path, (1, local_k), strict_shape_check))
        matrices.append(load_fp16_tensor(weight_path, (total_k, local_n), strict_shape_check))
    return vectors, matrices


def build_slice_bank0_image(
    dram: DRAM,
    *,
    vector_a: np.ndarray,
    matrix_b: np.ndarray,
    local_n: int,
    total_k: int,
    reg_n: int,
    reg_k: int,
) -> np.ndarray:
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
        build_b_layout_bytes(matrix_b, n=local_n, k=total_k, reg_n=reg_n, reg_k=reg_k),
    )
    return bank0_image


def rebuild_hex_data_from_ring_slices(
    dram: DRAM,
    hex_data_dir: Path,
    slice_dirs: list[Path],
    *,
    total_n: int,
    total_k: int,
    reg_n: int,
    reg_k: int,
    strict_shape_check: bool,
    emit_hex_files: bool,
    debug_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if emit_hex_files:
        hex_data_dir.mkdir(parents=True, exist_ok=True)

    debug_print(
        debug_progress,
        f"[DEBUG] Loading {len(slice_dirs)} slices from install data: total_n={total_n}, total_k={total_k}",
    )

    vectors, matrices = load_slice_tensors(
        slice_dirs,
        total_n=total_n,
        total_k=total_k,
        strict_shape_check=strict_shape_check,
    )
    local_n = total_n // len(slice_dirs)

    for slice_id, (vector_a, matrix_b) in enumerate(zip(vectors, matrices)):
        if slice_id == 0 or slice_id == len(slice_dirs) - 1 or slice_id % 4 == 0:
            debug_print(
                debug_progress,
                f"[DEBUG] Preparing slice{slice_id:02d}: A_shape={tuple(vector_a.shape)}, B_shape={tuple(matrix_b.shape)}",
            )
        bank0_image = build_slice_bank0_image(
            dram,
            vector_a=vector_a,
            matrix_b=matrix_b,
            local_n=local_n,
            total_k=total_k,
            reg_n=reg_n,
            reg_k=reg_k,
        )
        dram.data[slice_id, 0, :, :, :] = bank0_image.view(np.int8).reshape(ROW_NUM, COL_NUM, SUBWORD_SIZE)
        if emit_hex_files:
            write_bank0_image_to_hex(bank0_image, hex_data_dir / f"dram_data_slice{slice_id}_bank0.txt")

    global_vector = np.concatenate([vec.reshape(-1) for vec in vectors], axis=0).astype(np.float16, copy=False)
    global_matrix = np.concatenate(matrices, axis=1).astype(np.float16, copy=False)
    debug_print(
        debug_progress,
        f"[DEBUG] Global tensors ready: vector_shape={tuple(global_vector.shape)}, matrix_shape={tuple(global_matrix.shape)}",
    )
    return global_vector, global_matrix


def _copy_buffer_state(src: Buffer, dst: Buffer) -> None:
    np.copyto(dst.data, src.data)
    np.copyto(dst.tag_last, src.tag_last)
    np.copyto(dst.tag_last_index, src.tag_last_index)
    np.copyto(dst.tag_branch, src.tag_branch)


def _load_abuffer_paired_k2_rows(
    buffer_a: Buffer,
    row_tags_per_k2: list[list[dict[str, int]]],
    *,
    active_bytes: int = 4,
) -> None:
    buffer_a.clear()
    if len(row_tags_per_k2) > buffer_a.row_num:
        raise ValueError(
            f"Too many K2 rows for buffer_a: got {len(row_tags_per_k2)}, capacity {buffer_a.row_num}"
        )

    for row_idx in range(buffer_a.row_num):
        if row_idx < len(row_tags_per_k2):
            valid_tags = [
                tag for tag in row_tags_per_k2[row_idx]
                if tag.get("valid", 1) == 1 and tag.get("padding", 0) == 0
            ]
            if len(valid_tags) < active_bytes:
                raise ValueError(
                    f"Insufficient A tags for paired GEMV row {row_idx}: "
                    f"need {active_bytes}, got {len(valid_tags)}"
                )
            for col_idx in range(active_bytes):
                tag = valid_tags[col_idx]
                buffer_a.write(tag["data"], row_idx, col_idx, branch=tag.get("branch", 0))

        for col_idx in range(buffer_a.col_num):
            if row_idx < len(row_tags_per_k2) and col_idx < active_bytes:
                continue
            buffer_a.write(0, row_idx, col_idx, branch=1)


def maybe_run_buffer_to_pe(*, enabled: bool, **kwargs) -> None:
    if enabled:
        run_buffer_to_pe(**kwargs)


def maybe_dump_special_cycle(
    *,
    enabled: bool,
    special_pea: SpecialPEA,
    output_root: Path,
    n_outer_idx: int,
    k_outer_idx: int,
    n_reg_idx: int,
    k_reg_idx: int,
    cycle: int,
    records: list[dict[str, object]],
    slice_id: int,
) -> None:
    if not enabled:
        return
    dump_special_pea_cycle_state(
        special_pea,
        output_root,
        n_outer_idx=n_outer_idx,
        k_outer_idx=k_outer_idx,
        n_reg_idx=n_reg_idx,
        k_reg_idx=k_reg_idx,
        cycle=cycle,
    )
    dump_special_active_pe_cycle_results_per_slice(
        special_pea,
        output_root,
        slice_id=slice_id,
        n_outer_idx=n_outer_idx,
        k_outer_idx=k_outer_idx,
        n_reg_idx=n_reg_idx,
        k_reg_idx=k_reg_idx,
        cycle=cycle,
        records=records,
    )


def debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def dump_special_active_pe_cycle_results_per_slice(
    special_pea: SpecialPEA,
    output_root: Path,
    *,
    slice_id: int,
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
    output_path = output_root / "general_pe" / f"slice{slice_id:02d}_special_active_pe_cycle_results.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fp:
        fp.write(
            f"# slice={slice_id} derived_from_special_pe_execute "
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


def dump_general_array_accumulation_results_per_slice(
    registers: np.ndarray,
    reduced: np.ndarray,
    output_root: Path,
    *,
    slice_id: int,
    n_outer_idx: int,
    reg_n: int,
    reg_k: int,
) -> None:
    output_path = output_root / "general_pe" / f"slice{slice_id:02d}_general_array_accumulation_results.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fp:
        fp.write(
            f"# slice={slice_id} reduction_summary N_outer(N//n_reg//8)={n_outer_idx} "
            f"reg_n={reg_n} reg_k={reg_k}\n"
        )
        for n_reg_idx in range(reg_n):
            start = n_reg_idx * reg_k
            end = start + reg_k
            fp.write(f"## output_group n_reg_idx={n_reg_idx} reg_range=[{start}:{end})\n")
            for pe_col in range(registers.shape[0] - 1, -1, -1):
                lane_values = registers[pe_col, start:end].astype(np.float32)
                lane_hex = " ".join(f"0x{int(np.float32(val).view(np.uint32)):08X}" for val in lane_values)
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


def dump_output_writeback_ag4_trace(
    output_buffer: Buffer,
    dram: DRAM,
    *,
    slice_id: int,
    linear_output_idx: int,
) -> None:
    loop_states = [
        LoopState(lc_index=0, current=0, step=1, end=1),
    ]
    run_buffer_writeback_to_dram(
        output_buffer,
        5,
        dram,
        linear_output_idx,
        0,
        0,
        loop_states,
        (OUTBUFFER_ROW_LC_INDEX, OUTBUFFER_COL_LC_INDEX),
        0,
        datatype="fp16",
        slice_id=slice_id,
    )


def _load_b_round_buffers_for_slice(
    dram: DRAM,
    *,
    slice_id: int,
    pass_idx: int,
    k_block_idx: int,
    reg_n: int,
    reg_k: int,
    total_k_blocks: int,
    enable_traces: bool,
    debug_progress: bool,
) -> list[list[Buffer]]:
    if reg_k % K2_ROWS_PER_BUFFER != 0:
        raise ValueError("reg_k must be divisible by 4 because each ping/pong buffer stores K4K2N8")

    ags_per_round = reg_k // K2_ROWS_PER_BUFFER
    pass_bytes_per_kblock = B_SUBTILE_BYTES * reg_n * ags_per_round
    weight_base = get_gemm_tensor_base_addr("weight")
    pass_base = weight_base + pass_idx * total_k_blocks * pass_bytes_per_kblock
    block_base = pass_base + k_block_idx * pass_bytes_per_kblock
    if debug_progress and slice_id == 0 and (k_block_idx == 0 or k_block_idx == total_k_blocks - 1):
        debug_print(
            True,
            f"[DEBUG] slice00 load B: pass={pass_idx}, k_block={k_block_idx}, block_base=0x{block_base:X}",
        )

    round_buffers: list[list[Buffer]] = []
    for round_idx in range(reg_n):
        round_group: list[Buffer] = []
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
                        slice_id=slice_id,
                        address_profile=GEMV_LOCAL_N32_PROFILE,
                        base_addr_override=base_subtile,
                    )
                )

            run_ag_to_buffer(
                work_buffer,
                buffer_id,
                full_unpacked,
                k_block_idx,
                round_idx,
                pass_idx,
                [],
                (BUFFER_ROW_LC_INDEX, BUFFER_COL_LC_INDEX),
                0,
                datatype="fp16",
                slice_id=slice_id,
                input_layout="K2N8",
            )
            maybe_run_buffer_to_pe(
                enabled=enable_traces,
                buffer=work_buffer,
                buffer_id=buffer_id,
                c_in=k_block_idx,
                w_k=round_idx,
                h_k=pass_idx,
                pingpong_id=0,
                datatype="fp16",
                slice_id=slice_id,
                trace_tag=f"pass={pass_idx}_ring_kblk={k_block_idx}_round={round_idx}_AG={ag_idx}",
            )
            round_group.append(clone_buffer(work_buffer))
        round_buffers.append(round_group)
    return round_buffers


def run_ring2ring_simulation(
    dram: DRAM,
    output_root: Path,
    *,
    total_n: int,
    total_k: int,
    reg_n: int,
    reg_k: int,
    slice_num: int,
    enable_traces: bool,
    debug_progress: bool,
) -> np.ndarray:
    outputs_per_pass = 8 * reg_n
    local_n = total_n // slice_num
    local_k = total_k // slice_num
    local_num_passes = local_n // outputs_per_pass
    local_k2 = local_k // 2
    if local_k2 % K2_ROWS_PER_BUFFER != 0:
        raise ValueError(
            f"Expected local_k2={local_k2} to be divisible by K2_ROWS_PER_BUFFER={K2_ROWS_PER_BUFFER}"
        )
    local_k2_groups = local_k2 // K2_ROWS_PER_BUFFER
    k_blocks = total_k // (reg_k * 2)

    if k_blocks != slice_num:
        raise ValueError(
            "This ring2ring N32 implementation expects one K-block per slice, "
            f"but got k_blocks={k_blocks} and slice_num={slice_num}. "
            "For your target test this should be reg_k=16 and K = slice_num * reg_k * 2."
        )
    if local_k2 != reg_k:
        raise ValueError(
            f"Expected each slice to hold exactly reg_k K2 tiles, but got local_k2={local_k2}, reg_k={reg_k}"
        )
    if local_k2_groups != reg_k // K2_ROWS_PER_BUFFER:
        raise ValueError(
            "Expected each paired-row A tile group to align with one B buffer group, "
            f"but got local_k2_groups={local_k2_groups} and reg_k={reg_k}"
        )

    buffers_a = [
        [Buffer(col_num=32, row_num=4, bitwidth=8), Buffer(col_num=32, row_num=4, bitwidth=8)]
        for _ in range(slice_num)
    ]
    output_buffer_rows = max(1, math.ceil(outputs_per_pass / 16))
    output_buffers = [Buffer(col_num=32, row_num=output_buffer_rows, bitwidth=8) for _ in range(slice_num)]
    special_peas = [
        SpecialPEA(col_num=8, row_num=8, dot_size=2, psum_bitwidth=32, datatype="fp16")
        for _ in range(slice_num)
    ]
    for slice_id, pea in enumerate(special_peas):
        pea.slice_id = slice_id

    aggregated_outputs = np.zeros(total_n, dtype=np.float16)
    pending_half_rows: list[np.ndarray | None] = [None for _ in range(slice_num)]
    ring_next_map, ring_prev_map = build_ring_maps(slice_num)
    debug_print(
        debug_progress,
        f"[DEBUG] Simulation config: local_n={local_n}, local_k={local_k}, "
        f"local_num_passes={local_num_passes}, local_k2={local_k2}, "
        f"local_k2_groups={local_k2_groups}, k_blocks={k_blocks}",
    )
    debug_print(debug_progress, f"[DEBUG] Using ring order: {RING_SLICE_ORDER}")

    for pass_idx in tqdm(range(local_num_passes), desc="Ring GEMV pass", position=0):
        debug_print(debug_progress, f"[DEBUG] Enter pass {pass_idx + 1}/{local_num_passes}")
        for pea in special_peas:
            pea.clear_psums()

        for local_k2_group_idx in tqdm(range(local_k2_groups), desc="Local K2 loop", leave=False, position=1):
            local_k2_base = local_k2_group_idx * K2_ROWS_PER_BUFFER
            local_k2_end = local_k2_base + K2_ROWS_PER_BUFFER - 1
            if debug_progress and (
                local_k2_group_idx == 0
                or local_k2_group_idx == local_k2_groups - 1
                or local_k2_group_idx % 2 == 0
            ):
                debug_print(
                    True,
                    f"[DEBUG] Pass {pass_idx + 1}/{local_num_passes}: "
                    f"local_k2_group={local_k2_group_idx + 1}/{local_k2_groups} "
                    f"(k2={local_k2_base}-{local_k2_end})",
                )
            current_slot = local_k2_group_idx % 2

            for slice_id in range(slice_num):
                unpacked_a_rows: list[list[dict[str, int]]] = []
                for row_offset in range(K2_ROWS_PER_BUFFER):
                    unpacked_a_rows.append(
                        run_dram_to_ag(
                            dram,
                            0,
                            local_k2_base + row_offset,
                            0,
                            0,
                            current_slot,
                            datatype="fp16",
                            slice_id=slice_id,
                            address_profile=GEMV_LOCAL_N32_PROFILE,
                        )
                    )
                _load_abuffer_paired_k2_rows(
                    buffers_a[slice_id][current_slot],
                    unpacked_a_rows,
                    active_bytes=4,
                )
                maybe_run_buffer_to_pe(
                    enabled=enable_traces,
                    buffer=buffers_a[slice_id][current_slot],
                    buffer_id=0,
                    c_in=local_k2_base,
                    w_k=0,
                    h_k=pass_idx,
                    pingpong_id=current_slot,
                    datatype="fp16",
                    slice_id=slice_id,
                    trace_tag=f"pass={pass_idx}_localK2grp={local_k2_group_idx}_ring=0_A",
                )

            active_slot = current_slot

            for ring_step in range(slice_num):
                if debug_progress and (ring_step == 0 or ring_step == slice_num - 1 or ring_step % 7 == 0):
                    debug_print(
                        True,
                        f"[DEBUG] Pass {pass_idx + 1}/{local_num_passes}, "
                        f"K2grp {local_k2_group_idx + 1}/{local_k2_groups}: "
                        f"ring_step={ring_step + 1}/{slice_num}",
                    )
                if ring_step > 0:
                    for slice_id in range(slice_num):
                        maybe_run_buffer_to_pe(
                            enabled=enable_traces,
                            buffer=buffers_a[slice_id][active_slot],
                            buffer_id=0,
                            c_in=local_k2_base,
                            w_k=0,
                            h_k=pass_idx,
                            pingpong_id=active_slot,
                            datatype="fp16",
                            slice_id=slice_id,
                            trace_tag=f"pass={pass_idx}_localK2grp={local_k2_group_idx}_ring={ring_step}_A",
                        )

                for slice_id in range(slice_num):
                    source_slice = slice_id
                    for _ in range(ring_step):
                        source_slice = ring_prev_map[source_slice]
                    round_buffers = _load_b_round_buffers_for_slice(
                        dram,
                        slice_id=slice_id,
                        pass_idx=pass_idx,
                        k_block_idx=source_slice,
                        reg_n=reg_n,
                        reg_k=reg_k,
                        total_k_blocks=k_blocks,
                        enable_traces=enable_traces,
                        debug_progress=debug_progress,
                    )

                    for round_idx in range(reg_n):
                        active_buffer = round_buffers[round_idx][local_k2_group_idx]
                        for row_offset in range(K2_ROWS_PER_BUFFER):
                            cycle = round_idx * reg_k + local_k2_base + row_offset
                            general_pe_records = prepare_general_pe_cycle_records(
                                special_peas[slice_id],
                                buffers_a[slice_id][active_slot],
                                active_buffer,
                                cycle=cycle,
                                gemv_paired_rows=True,
                            )
                            special_peas[slice_id].execute(
                                buffers_a[slice_id][active_slot],
                                active_buffer,
                                cycle=cycle,
                                gemv_paired_rows=True,
                            )
                            maybe_dump_special_cycle(
                                enabled=enable_traces,
                                special_pea=special_peas[slice_id],
                                output_root=output_root,
                                slice_id=slice_id,
                                n_outer_idx=pass_idx * slice_num + slice_id,
                                k_outer_idx=source_slice,
                                n_reg_idx=round_idx,
                                k_reg_idx=local_k2_base + row_offset,
                                cycle=cycle,
                                records=general_pe_records,
                            )

                if ring_step == slice_num - 1:
                    break

                next_slot = 1 - active_slot
                for slice_id in range(slice_num):
                    dst_slice = ring_next_map[slice_id]
                    _copy_buffer_state(buffers_a[slice_id][active_slot], buffers_a[dst_slice][next_slot])
                active_slot = next_slot

        for slice_id in range(slice_num):
            if debug_progress and (slice_id == 0 or slice_id == slice_num - 1 or slice_id % 7 == 0):
                debug_print(
                    True,
                    f"[DEBUG] Finalizing slice{slice_id:02d} outputs for pass {pass_idx + 1}/{local_num_passes}",
                )
            registers = collect_pe_registers(special_peas[slice_id])
            reduced = reduce_registers(registers, reg_n=reg_n, reg_k=reg_k)
            if enable_traces:
                dump_general_array_accumulation_results_per_slice(
                    registers,
                    reduced,
                    output_root,
                    slice_id=slice_id,
                    n_outer_idx=pass_idx * slice_num + slice_id,
                    reg_n=reg_n,
                    reg_k=reg_k,
                )
            outputs_fp16 = flatten_outputs_for_writeback(reduced).astype(np.float16)
            start = slice_id * local_n + pass_idx * outputs_per_pass
            end = start + outputs_per_pass
            aggregated_outputs[start:end] = outputs_fp16
            buffer5_row_outputs, pending_half_rows[slice_id] = build_buffer5_row_outputs(
                outputs_fp16,
                reg_n=reg_n,
                pending_half_row=pending_half_rows[slice_id],
            )
            if buffer5_row_outputs is not None:
                fill_output_buffer(output_buffers[slice_id], buffer5_row_outputs)
                maybe_run_buffer_to_pe(
                    enabled=enable_traces,
                    buffer=output_buffers[slice_id],
                    buffer_id=5,
                    c_in=pass_idx,
                    w_k=0,
                    h_k=0,
                    pingpong_id=0,
                    datatype="fp16",
                    slice_id=slice_id,
                    trace_tag=f"pass={pass_idx}_slice={slice_id}_buffer5",
                )
                if enable_traces:
                    dump_output_writeback_ag4_trace(
                        output_buffers[slice_id],
                        dram,
                        slice_id=slice_id,
                        linear_output_idx=pass_idx * slice_num + slice_id,
                    )
            slice_output_dir = output_root / f"slice{slice_id:02d}"
            if enable_traces:
                slice_output_dir.mkdir(parents=True, exist_ok=True)
                dump_output_vector(outputs_fp16, slice_output_dir)
                dump_output_writeback_trace(
                    outputs_fp16,
                    output_root,
                    n_outer_idx=pass_idx * slice_num + slice_id,
                    output_base_addr=get_gemm_tensor_base_addr("output"),
                )

    debug_print(debug_progress, "[DEBUG] Simulation loop finished")
    return aggregated_outputs


def reference_ring_grouped_gemv(global_vector: np.ndarray, global_matrix: np.ndarray) -> np.ndarray:
    return (global_vector.astype(np.float32) @ global_matrix.astype(np.float32)).astype(np.float16)


def main() -> None:
    args = parse_args()

    if args.n <= 0 or args.k <= 0:
        raise ValueError("N and K must be positive")
    if args.reg_n <= 0 or args.reg_k <= 0 or args.reg_n * args.reg_k != 16:
        raise ValueError("reg_n and reg_k must be positive and satisfy reg_n * reg_k == 16")
    if args.slice_num <= 0:
        raise ValueError("slice_num must be positive")
    if args.n % args.slice_num != 0:
        raise ValueError("Global N must be divisible by slice_num")
    if args.k % args.slice_num != 0 or args.k % 2 != 0:
        raise ValueError("Global K must be divisible by slice_num and by 2")

    local_n = args.n // args.slice_num
    if local_n % (8 * args.reg_n) != 0:
        raise ValueError("Per-slice N must be divisible by outputs_per_pass = 8 * reg_n")

    gemm_install_dir = args.gemm_install_dir.resolve()
    if not gemm_install_dir.exists():
        raise FileNotFoundError(f"Install directory not found: {gemm_install_dir}")
    slice_dirs = resolve_slice_dirs(gemm_install_dir, args.slice_num)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    try:
        os.chdir(output_root)
        reset_output_files(OUTPUT_FILES)
        reset_dynamic_dirs([Path(name) for name in RING_DYNAMIC_DIR_NAMES])
        sync_gemm_base_addrs_to_data_transfer()
        set_trace_dump_enabled(not args.lightweight_check)
        debug_print(args.debug_progress, "[DEBUG] Output directories reset")
        if not args.lightweight_check:
            Path("general_pe").mkdir(parents=True, exist_ok=True)
            Path("dram_ag").mkdir(parents=True, exist_ok=True)
            Path("buffer_dump").mkdir(parents=True, exist_ok=True)

        dram = DRAM(
            slice_num=args.slice_num,
            bank_num=BANK_NUM,
            row_num=ROW_NUM,
            col_num=COL_NUM,
            subword_size=SUBWORD_SIZE,
        )
        debug_print(args.debug_progress, "[DEBUG] DRAM created")

        global_vector, global_matrix = rebuild_hex_data_from_ring_slices(
            dram,
            Path("hex_data"),
            slice_dirs,
            total_n=args.n,
            total_k=args.k,
            reg_n=args.reg_n,
            reg_k=args.reg_k,
            strict_shape_check=args.strict_shape_check,
            emit_hex_files=not args.lightweight_check,
            debug_progress=args.debug_progress,
        )
        debug_print(args.debug_progress, "[DEBUG] Slice tensors loaded into DRAM")

        simulated = run_ring2ring_simulation(
            dram,
            output_root=Path.cwd(),
            total_n=args.n,
            total_k=args.k,
            reg_n=args.reg_n,
            reg_k=args.reg_k,
            slice_num=args.slice_num,
            enable_traces=not args.lightweight_check,
            debug_progress=args.debug_progress,
        )
        debug_print(args.debug_progress, "[DEBUG] Simulation complete, computing reference")

        golden = reference_ring_grouped_gemv(global_vector, global_matrix).reshape(-1)
        np.savetxt("gemv_n2n_n32_output_fp16_hex.txt", simulated.view(np.uint16).reshape(1, -1), fmt="0x%04X")
        np.savetxt("gemv_n2n_n32_golden_fp16_hex.txt", golden.view(np.uint16).reshape(1, -1), fmt="0x%04X")
        if np.array_equal(simulated.view(np.uint16), golden.view(np.uint16)):
            print("[OK] Ring2ring GEMV_N32 output matches fp16 reference.")
        else:
            mismatch = np.nonzero(simulated.view(np.uint16) != golden.view(np.uint16))[0]
            print(
                f"[WARN] Ring2ring GEMV_N32 differs from fp16 reference at {mismatch.size} positions. "
                f"First mismatch index: {int(mismatch[0]) if mismatch.size else 'n/a'}"
            )
    finally:
        set_trace_dump_enabled(True)
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
