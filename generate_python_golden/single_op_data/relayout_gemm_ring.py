import argparse
import os
import re
import struct

import numpy as np


MODEL_PARAMS = {
    "hidden_size": 896,
    "intermediate_size": 1792,
    "num_attention_heads": 7,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "sequence_length": 32,
}

BASE_HW_PARAMS = {
    "num_slices": 28,
    "file_order": "F",
    "ring_order": [0, 3, 2, 4, 7, 6, 5, 1, 13, 16, 19, 20, 23, 25, 8, 11, 10, 9, 24, 27, 26, 22, 21, 18, 17, 12, 15, 14],
    "physical_mapping": [0, 2, 3, 1, 5, 4, 6, 7, 8, 10, 11, 9, 15, 14, 12, 13, 16, 17, 19, 18, 20, 21, 23, 22, 26, 24, 25, 27],
}

KV_HW_PARAMS = {
    "num_slices": 4,
    "file_order": "F",
    "ring_order": [0, 3, 2, 1],
    "physical_mapping": [0, 2, 3, 1],
    "install_targets": [
        [0, 5, 8, 15, 16, 20, 26],
        [2, 4, 10, 14, 17, 21, 24],
        [3, 6, 11, 12, 19, 23, 25],
        [1, 7, 9, 13, 18, 22, 27],
    ],
}

FILENAME_PATTERN = re.compile(
    r"^blk\.(?P<layer>\d+)_(?P<op_name>.+?)_op-mul_mat_(?P<io_role>in0|in1|out)_shape(?P<shape>[\dx]+)_dtype_(?P<dtype>[a-z0-9]+)\.bin$"
)

OP_SPECS = {
    "ffn_gate": {
        "aliases": ["ffn_gate"],
        "dims_fn": lambda params: (
            params["hidden_size"],
            params["sequence_length"],
            params["intermediate_size"],
        ),
        "hw_params": BASE_HW_PARAMS,
        "pad_input_k_to_power_of_two": False,
    },
    "ffn_up": {
        "aliases": ["ffn_up"],
        "dims_fn": lambda params: (
            params["hidden_size"],
            params["sequence_length"],
            params["intermediate_size"],
        ),
        "hw_params": BASE_HW_PARAMS,
        "pad_input_k_to_power_of_two": False,
    },
    "ffn_out": {
        "aliases": ["ffn_out", "ffn_down"],
        "dims_fn": lambda params: (
            params["intermediate_size"],
            params["sequence_length"],
            params["hidden_size"],
        ),
        "hw_params": BASE_HW_PARAMS,
        "pad_input_k_to_power_of_two": False,
    },
    "atten_final": {
        "aliases": ["atten_final", "node_0_attn_final", "attn_final"],
        "dims_fn": lambda params: (
            params["hidden_size"],
            params["sequence_length"],
            params["hidden_size"],
        ),
        "hw_params": BASE_HW_PARAMS,
        "pad_input_k_to_power_of_two": False,
    },
    "q_gen": {
        "aliases": ["q_gen", "Q_gen", "Qcur", "node_q", "node_0_q", "attn_q"],
        "dims_fn": lambda params: (
            params["hidden_size"],
            params["sequence_length"],
            params["num_attention_heads"] * params["head_dim"],
        ),
        "hw_params": BASE_HW_PARAMS,
        "pad_input_k_to_power_of_two": False,
    },
    "k_gen": {
        "aliases": ["k_gen", "node_k", "node_0_k", "attn_k"],
        "dims_fn": lambda params: (
            params["hidden_size"],
            params["sequence_length"],
            params["num_key_value_heads"] * params["head_dim"],
        ),
        "hw_params": KV_HW_PARAMS,
        "pad_input_k_to_power_of_two": True,
    },
    "v_gen": {
        "aliases": ["v_gen", "node_v", "node_0_v", "attn_v"],
        "dims_fn": lambda params: (
            params["hidden_size"],
            params["sequence_length"],
            params["num_key_value_heads"] * params["head_dim"],
        ),
        "hw_params": KV_HW_PARAMS,
        "pad_input_k_to_power_of_two": True,
    },
}

TENSOR_SPECS = {
    "weight": {
        "source_io_role": "in0",
        "shape_fn": lambda dims: (dims["K"], dims["N"], 1, 1),
        "out_name": "matrix_in1_linearized_128bit.bin",
        "slice_axis": "N",
        "before_view": "identity",
        "use_ring": True,
        "relayout_fn": "in0",
        "save_after_ring": True,
    },
    "input": {
        "source_io_role": "in1",
        "shape_fn": lambda dims: (dims["K"], dims["M"], 1, 1),
        "out_name": "matrix_in0_linearized_128bit.bin",
        "slice_axis": "K",
        "before_view": "transpose",
        "use_ring": False,
        "relayout_fn": "in1",
        "save_after_ring": False,
    },
    "output": {
        "source_io_role": "out",
        "shape_fn": lambda dims: (dims["N"], dims["M"], 1, 1),
        "out_name": "matrix_out_linearized_128bit.bin",
        "slice_axis": "N",
        "before_view": "transpose",
        "use_ring": False,
        "relayout_fn": "out",
        "save_after_ring": False,
    },
}

RELAYOUT_LAYOUT_SPECS = {
    "weight": {
        "axis_order": ("K", "N"),
        "fixed_suffix_fastest_first": [("N", 8), ("K", 2), ("N", 4)],
    },
    "input": {
        "axis_order": ("K", "L"),
        "fixed_suffix_fastest_first": [("L", 8), ("K", 2), ("L", 4)],
    },
    "output": {
        "axis_order": ("N", "L"),
        "fixed_suffix_fastest_first": [("L", 8), ("N", 8), ("L", 4), ("N", 4)],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Split and relayout GEMM golden tensors.")
    parser.add_argument("--target-op", choices=sorted(OP_SPECS.keys()) + ["all"], default="all")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def float16_to_bin(f):
    return bin(np.float16(f).view(np.uint16))[2:].zfill(16)


def float_to_bin(f):
    return bin(struct.unpack("<I", struct.pack("<f", f))[0])[2:].zfill(32)


def convert_to_decimal_txt(bin_path, rows=None, cols=None, file_order="C"):
    data = np.fromfile(bin_path, dtype=np.float16)
    if rows is None or cols is None:
        rows, cols = data.size, 1
    if rows * cols != data.size:
        print(f"  WARNING: Decimal reshape mismatch: {bin_path}, fallback to Nx1")
        rows, cols = data.size, 1

    matrix = data.reshape((rows, cols), order=file_order)
    txt_path = bin_path.replace(".bin", f"_{rows}x{cols}_decimal.txt")
    with open(txt_path, "w", newline="\n") as f:
        for r in range(rows):
            f.write(",".join(f"{float(v):.10g}" for v in matrix[r]))
            f.write("\n")

    hex_txt_path = bin_path.replace(".bin", f"_{rows}x{cols}_hex.txt")
    matrix_uint = data.view(np.uint16).reshape((rows, cols), order=file_order)
    with open(hex_txt_path, "w", newline="\n") as f:
        for r in range(rows):
            f.write(",".join(f"{v:04x}" for v in matrix_uint[r]))
            f.write("\n")


def convert_to_128bit_txt(bin_path, rows=None, cols=None, file_order="C"):
    data = np.fromfile(bin_path, dtype=np.float16)

    remainder = len(data) % 8
    if remainder != 0:
        data = np.concatenate((data, np.zeros(8 - remainder, dtype=np.float16)))

    txt_path = bin_path.replace(".bin", ".txt")
    with open(txt_path, "w", newline="\n") as f:
        for i in range(0, len(data), 8):
            strs = [float16_to_bin(data[i + j]) for j in range(8)]
            f.write("".join(reversed(strs)) + "\n")

    convert_to_decimal_txt(bin_path, rows=rows, cols=cols, file_order=file_order)


def validate_positive_int(name, value):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def next_power_of_two(value):
    validate_positive_int("value", value)
    return 1 if value == 1 else 1 << (value - 1).bit_length()


def build_expected_shapes(target_op, model_params):
    K, M, N = OP_SPECS[target_op]["dims_fn"](model_params)
    for name, value in (("K", K), ("M", M), ("N", N)):
        validate_positive_int(name, value)
    return {"K": K, "M": M, "N": N}


def build_runtime_dims(target_op, model_params):
    raw_dims = build_expected_shapes(target_op, model_params)
    runtime_dims = dict(raw_dims)
    if OP_SPECS[target_op]["pad_input_k_to_power_of_two"]:
        runtime_dims["K"] = next_power_of_two(raw_dims["K"])
    runtime_dims["raw_K"] = raw_dims["K"]
    runtime_dims["raw_M"] = raw_dims["M"]
    runtime_dims["raw_N"] = raw_dims["N"]
    return runtime_dims


def derive_slice_sizes(K, N, num_slices):
    validate_positive_int("num_slices", num_slices)
    if K % num_slices != 0:
        raise ValueError(f"K={K} must be divisible by num_slices={num_slices}")
    if N % num_slices != 0:
        raise ValueError(f"N={N} must be divisible by num_slices={num_slices}")
    return {
        "slice_k": K // num_slices,
        "slice_n": N // num_slices,
    }


def validate_config_for_op(target_op, dims, hw_params):
    K = dims["K"]
    M = dims["M"]
    N = dims["N"]
    num_slices = hw_params["num_slices"]
    slice_k = dims["slice_k"]
    slice_n = dims["slice_n"]

    if K % num_slices != 0:
        raise ValueError(f"{target_op}: K={K} is not divisible by num_slices={num_slices}")
    if N % num_slices != 0:
        raise ValueError(f"{target_op}: N={N} is not divisible by num_slices={num_slices}")
    if M % 8 != 0:
        raise ValueError(f"{target_op}: M={M} must be divisible by 8 for the current layouts")
    if slice_k % 2 != 0:
        raise ValueError(f"{target_op}: slice_k={slice_k} must be divisible by 2")
    if slice_n % 8 != 0:
        raise ValueError(f"{target_op}: slice_n={slice_n} must be divisible by 8")


def parse_gemm_filename(filename):
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None

    shape = tuple(int(part) for part in match.group("shape").split("x"))
    return {
        "filename": filename,
        "layer_id": int(match.group("layer")),
        "op_name": match.group("op_name"),
        "io_role": match.group("io_role"),
        "shape": shape,
        "dtype": match.group("dtype"),
    }


def collect_parsed_entries(input_dir):
    parsed_entries = []
    for root, _, files in os.walk(input_dir):
        for entry in files:
            parsed = parse_gemm_filename(entry)
            if parsed is None:
                continue
            parsed["root_dir"] = root
            parsed["filepath"] = os.path.join(root, entry)
            parsed["relative_dir"] = os.path.relpath(root, input_dir)
            parsed_entries.append(parsed)
    return parsed_entries


def find_target_files(input_dir, target_op, layer_id, op_spec, dims):
    parsed_entries = collect_parsed_entries(input_dir)
    if not parsed_entries:
        raise FileNotFoundError(f"No GEMM .bin files found under {input_dir}")

    preferred_dirs = [target_op]
    preferred_dirs.extend(alias for alias in op_spec["aliases"] if alias not in preferred_dirs)

    alias_matches = {}
    for alias in op_spec["aliases"]:
        matching = [
            entry for entry in parsed_entries
            if entry["layer_id"] == layer_id and entry["op_name"] in {
                alias,
                f"{alias}-{layer_id}",
            }
        ]
        if matching:
            alias_matches[alias] = matching

    if not alias_matches:
        aliases = ", ".join(op_spec["aliases"])
        raise FileNotFoundError(
            f"Could not find files for target_op={target_op}, layer_id={layer_id}. Tried aliases: {aliases}"
        )

    if len(alias_matches) > 1:
        filtered_matches = {}
        for preferred_dir in preferred_dirs:
            dir_matches = {
                alias: entries
                for alias, entries in alias_matches.items()
                if any(
                    entry["relative_dir"] == preferred_dir
                    or entry["relative_dir"].startswith(preferred_dir + os.sep)
                    for entry in entries
                )
            }
            if dir_matches:
                filtered_matches = dir_matches
                break
        if filtered_matches:
            alias_matches = filtered_matches

    if len(alias_matches) > 1:
        raise ValueError(
            f"Multiple aliases matched target_op={target_op}: {', '.join(sorted(alias_matches.keys()))}"
        )

    matched_alias, matched_entries = next(iter(alias_matches.items()))
    preferred_entries = [
        entry for entry in matched_entries
        if entry["relative_dir"] == target_op or entry["relative_dir"].startswith(target_op + os.sep)
    ]
    if preferred_entries:
        matched_entries = preferred_entries

    matched_by_role = {}
    for tensor_role, tensor_spec in TENSOR_SPECS.items():
        expected_shape = tensor_spec["shape_fn"](dims)
        candidates = [
            entry for entry in matched_entries
            if entry["io_role"] == tensor_spec["source_io_role"] and entry["shape"] == expected_shape
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Missing {tensor_role} file for alias={matched_alias}, expected io_role={tensor_spec['source_io_role']}, "
                f"shape={expected_shape}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Found multiple {tensor_role} files for alias={matched_alias}, shape={expected_shape}"
            )
        matched_by_role[tensor_role] = candidates[0]

    return matched_alias, matched_by_role


def load_gemm_tensor(filepath, expected_shape, file_order="F"):
    data = np.fromfile(filepath, dtype=np.float16)
    expected_elems = int(np.prod(expected_shape))
    if data.size != expected_elems:
        raise ValueError(
            f"{filepath} contains {data.size} float16 values, expected {expected_elems} from shape {expected_shape}"
        )
    tensor = data.reshape(expected_shape, order=file_order)
    return squeeze_to_2d(tensor)


def pad_tensor_for_processing(role, data_2d, runtime_dims):
    data_2d = np.asarray(data_2d, dtype=np.float16)
    raw_k = runtime_dims["raw_K"]
    padded_k = runtime_dims["K"]
    if raw_k == padded_k or role == "output":
        return data_2d

    if role not in {"weight", "input"}:
        raise ValueError(f"Unsupported role for padding: {role}")
    if data_2d.shape[0] != raw_k:
        raise ValueError(
            f"{role} tensor expected first dimension raw_K={raw_k}, got shape {data_2d.shape}"
        )

    pad_rows = padded_k - raw_k
    if pad_rows < 0:
        raise ValueError(f"Padded K={padded_k} must be >= raw K={raw_k}")
    if pad_rows == 0:
        return data_2d

    padding = np.zeros((pad_rows, data_2d.shape[1]), dtype=np.float16)
    return np.concatenate((data_2d, padding), axis=0)


def squeeze_to_2d(data):
    data_2d = np.asarray(data).squeeze()
    if data_2d.ndim == 0:
        return data_2d.reshape(1, 1)
    if data_2d.ndim == 1:
        return data_2d.reshape(-1, 1)
    if data_2d.ndim != 2:
        raise ValueError(f"Expected squeezed tensor to be 2D, got shape {data_2d.shape}")
    return data_2d


def format_layout_token(token):
    kind = token[0]
    if kind == "outer":
        return f"{token[1]}_outer"
    return f"{token[1]}{token[2]}"


def factor_axis_with_suffix(axis_name, axis_size, suffix_factors):
    validate_positive_int(f"{axis_name} axis size", axis_size)
    required = 1
    for factor in suffix_factors:
        validate_positive_int(f"{axis_name} suffix factor", factor)
        required *= factor
    if axis_size % required != 0:
        suffix_name = "*".join(f"{axis_name}{factor}" for factor in reversed(suffix_factors))
        raise ValueError(
            f"{axis_name} axis requires divisibility by {suffix_name}={required}, got {axis_size}"
        )
    outer_factor = axis_size // required
    axis_dims = [outer_factor] + list(reversed(suffix_factors))
    return outer_factor, axis_dims


def build_layout_plan(axis_sizes, axis_order, fixed_suffix_fastest_first):
    fixed_entries = [(idx, axis_name, factor) for idx, (axis_name, factor) in enumerate(fixed_suffix_fastest_first)]

    reshape_dims = []
    reshape_tokens = []
    outer_factors = {}

    for axis_name in axis_order:
        axis_suffix_factors = [factor for _, suffix_axis, factor in fixed_entries if suffix_axis == axis_name]
        outer_factor, axis_dims = factor_axis_with_suffix(axis_name, axis_sizes[axis_name], axis_suffix_factors)
        outer_factors[axis_name] = outer_factor
        reshape_dims.append(axis_dims[0])
        reshape_tokens.append(("outer", axis_name))

        for idx, suffix_axis, factor in reversed(fixed_entries):
            if suffix_axis == axis_name:
                reshape_dims.append(factor)
                reshape_tokens.append(("fixed", suffix_axis, factor, idx))

    transpose_tokens = [("outer", axis_name) for axis_name in reversed(axis_order)]
    transpose_tokens.extend(
        [("fixed", axis_name, factor, idx) for idx, axis_name, factor in reversed(fixed_entries)]
    )

    index_by_token = {token: idx for idx, token in enumerate(reshape_tokens)}
    transpose_order = [index_by_token[token] for token in transpose_tokens]

    return {
        "reshape_dims": reshape_dims,
        "reshape_labels": [format_layout_token(token) for token in reshape_tokens],
        "transpose_order": transpose_order,
        "transpose_labels": [format_layout_token(token) for token in transpose_tokens],
        "outer_factors": outer_factors,
        "fixed_suffix_fastest_first": list(fixed_suffix_fastest_first),
    }


def apply_parametric_relayout(slice_data, axis_sizes, axis_order, fixed_suffix_fastest_first):
    array = np.asarray(slice_data)
    expected_shape = tuple(axis_sizes[axis_name] for axis_name in axis_order)
    if array.shape != expected_shape:
        raise ValueError(f"Expected slice_data shape {expected_shape}, got {array.shape}")

    plan = build_layout_plan(axis_sizes, axis_order, fixed_suffix_fastest_first)
    reshaped = array.reshape(plan["reshape_dims"])
    transposed = reshaped.transpose(plan["transpose_order"])
    return transposed.reshape(-1)


def relayout_in0_N8K2N4K(slice_data, K, slice_n, num_slices):
    if K % 2 != 0:
        raise ValueError(f"in0 relayout requires K to be divisible by K2=2, got K={K}")
    if slice_n % (8 * 4) != 0:
        raise ValueError(f"in0 relayout requires N to be divisible by N4*N8=32, got slice_n={slice_n}")
    if K % num_slices != 0:
        raise ValueError(f"in0 relayout requires K divisible by num_slices={num_slices}, got K={K}")
    return apply_parametric_relayout(
        slice_data,
        axis_sizes={"K": K, "N": slice_n},
        axis_order=RELAYOUT_LAYOUT_SPECS["weight"]["axis_order"],
        fixed_suffix_fastest_first=RELAYOUT_LAYOUT_SPECS["weight"]["fixed_suffix_fastest_first"],
    )


def reorder_in0_slice_by_ring(slice_data, slice_idx, num_slices, slice_k, ring_order):
    if len(ring_order) != num_slices:
        raise ValueError(f"ring_order length {len(ring_order)} does not match num_slices={num_slices}")
    if slice_data.shape[0] != num_slices * slice_k:
        raise ValueError(
            f"Expected weight slice K dimension {num_slices * slice_k}, got {slice_data.shape[0]}"
        )
    if slice_k % 2 != 0:
        raise ValueError(f"Ring reorder requires slice_k to be divisible by 2, got {slice_k}")

    blocks = np.split(slice_data, num_slices, axis=0)
    base_blocks = [blocks[idx] for idx in ring_order]
    stacked_blocks = np.stack(base_blocks, axis=0)

    reshaped_for_interleave = stacked_blocks.reshape(num_slices, slice_k // 2, 2, slice_data.shape[1])
    interleaved = reshaped_for_interleave.transpose(1, 0, 2, 3)

    start_pos = ring_order.index(slice_idx)
    interleaved = np.roll(interleaved, -start_pos, axis=1)

    tiles = interleaved.reshape((slice_data.shape[0] // 2), 2, slice_data.shape[1])
    return tiles.reshape(slice_data.shape[0], slice_data.shape[1])


def relayout_in1_L8K2L4K(slice_data, slice_k, M):
    if M % (8 * 4) != 0:
        raise ValueError(f"in1 relayout requires L to be divisible by L4*L8=32, got M={M}")
    if slice_k % 2 != 0:
        raise ValueError(f"in1 relayout requires K to be divisible by K2=2, got slice_k={slice_k}")
    return apply_parametric_relayout(
        slice_data,
        axis_sizes={"K": slice_k, "L": M},
        axis_order=RELAYOUT_LAYOUT_SPECS["input"]["axis_order"],
        fixed_suffix_fastest_first=RELAYOUT_LAYOUT_SPECS["input"]["fixed_suffix_fastest_first"],
    )


def relayout_out_L8N8L4N4N2L1(slice_data, slice_n, M):
    if slice_n % (8 * 4) != 0:
        raise ValueError(f"out relayout requires N to be divisible by N4*N8=32, got slice_n={slice_n}")
    if M % (8 * 4) != 0:
        raise ValueError(f"out relayout requires L to be divisible by L4*L8=32, got M={M}")
    return apply_parametric_relayout(
        slice_data,
        axis_sizes={"N": slice_n, "L": M},
        axis_order=RELAYOUT_LAYOUT_SPECS["output"]["axis_order"],
        fixed_suffix_fastest_first=RELAYOUT_LAYOUT_SPECS["output"]["fixed_suffix_fastest_first"],
    )


def relayout_slice_default(slice_data):
    return np.asarray(slice_data).reshape(-1)


def save_slice(output_dir, op_id, slice_idx, out_name, matrix_2d, file_order, generate_views=True):
    matrix_2d = np.asarray(matrix_2d, dtype=np.float16)
    if matrix_2d.ndim == 1:
        matrix_2d = matrix_2d.reshape(-1, 1)

    slice_dir = os.path.join(output_dir, op_id, f"slice{slice_idx:02d}")
    os.makedirs(slice_dir, exist_ok=True)

    out_path = os.path.join(slice_dir, out_name)
    matrix_2d.reshape(-1, order=file_order).tofile(out_path)
    if generate_views:
        convert_to_128bit_txt(out_path, rows=matrix_2d.shape[0], cols=matrix_2d.shape[1], file_order=file_order)


def save_slice_relayout(output_dir, op_id, slice_idx, out_name, data_1d, file_order):
    data_1d = np.asarray(data_1d, dtype=np.float16)

    slice_dir = os.path.join(output_dir, op_id, f"slice{slice_idx:02d}")
    os.makedirs(slice_dir, exist_ok=True)

    out_path = os.path.join(slice_dir, out_name)
    data_1d.tofile(out_path)
    convert_to_128bit_txt(out_path, rows=None, cols=None, file_order=file_order)


def install_target_slices(hw_params, slice_idx):
    install_targets = hw_params.get("install_targets")
    if install_targets is None:
        return [hw_params["physical_mapping"][slice_idx]]
    if slice_idx >= len(install_targets):
        raise ValueError(f"Missing install target mapping for logical slice {slice_idx}")
    return list(install_targets[slice_idx])


def create_decimal_hex_from_array(bin_path, array, rows, cols, file_order="C"):
    arr = np.asarray(array, dtype=np.float16)
    if arr.ndim != 2:
        arr = arr.reshape(rows, cols)

    decimal_path = bin_path.replace(".bin", f"_{rows}x{cols}_decimal.txt")
    with open(decimal_path, "w", newline="\n") as f:
        for r in range(rows):
            f.write(",".join(f"{float(v):.10g}" for v in arr[r]) + "\n")

    hex_path = bin_path.replace(".bin", f"_{rows}x{cols}_hex.txt")
    arr_uint = arr.view(np.uint16)
    with open(hex_path, "w", newline="\n") as f:
        for r in range(rows):
            f.write(",".join(f"{v:04x}" for v in arr_uint[r]) + "\n")


def create_linear_hex_from_array(bin_path, array, file_order="C"):
    arr = np.asarray(array, dtype=np.float16)
    flat = arr.reshape(-1, order=file_order)
    hex_path = bin_path.replace(".bin", "_linear_hex.txt")
    with open(hex_path, "w", newline="\n") as f:
        for v in flat:
            f.write(f"{np.uint16(np.array(v).view(np.uint16)):04x}\n")


def save_before_view(output_dir, op_id, slice_idx, out_name, slice_data, file_order, before_view):
    generate_views = before_view == "identity"
    save_slice(output_dir, op_id, slice_idx, out_name, slice_data, file_order, generate_views=generate_views)

    if before_view == "transpose":
        slice_dir = os.path.join(output_dir, op_id, f"slice{slice_idx:02d}")
        before_path = os.path.join(slice_dir, out_name)
        display = slice_data.T
        create_decimal_hex_from_array(before_path, display, rows=display.shape[0], cols=display.shape[1], file_order=file_order)
        create_linear_hex_from_array(before_path, slice_data, file_order=file_order)


def process_tensor_role(role, data_2d, dims, hw_params, output_paths, op_id):
    tensor_spec = TENSOR_SPECS[role]
    layout_spec = RELAYOUT_LAYOUT_SPECS[role]
    num_slices = hw_params["num_slices"]
    file_order = hw_params["file_order"]
    out_name = tensor_spec["out_name"]

    if role == "weight":
        axis_size = dims["slice_n"]
        relayout_axis_sizes = {"K": dims["K"], "N": dims["slice_n"]}
    elif role == "input":
        axis_size = dims["slice_k"]
        relayout_axis_sizes = {"K": dims["slice_k"], "L": dims["M"]}
    elif role == "output":
        axis_size = dims["slice_n"]
        relayout_axis_sizes = {"N": dims["slice_n"], "L": dims["M"]}
    else:
        raise ValueError(f"Unsupported tensor role: {role}")

    layout_plan = build_layout_plan(
        relayout_axis_sizes,
        layout_spec["axis_order"],
        layout_spec["fixed_suffix_fastest_first"],
    )
    slice_shape = tuple(relayout_axis_sizes[axis] for axis in layout_spec["axis_order"])
    fixed_suffix_desc = " -> ".join(f"{axis}{factor}" for axis, factor in layout_spec["fixed_suffix_fastest_first"])
    print(
        f"    role={role} slice_shape={slice_shape} outer_factors={layout_plan['outer_factors']} "
        f"fixed_suffix={fixed_suffix_desc}"
    )

    for slice_idx in range(num_slices):
        if tensor_spec["slice_axis"] == "N":
            n_start = slice_idx * axis_size
            slice_data = data_2d[:, n_start:n_start + axis_size] if role == "weight" else data_2d[n_start:n_start + axis_size, :]
        elif tensor_spec["slice_axis"] == "K":
            k_start = slice_idx * axis_size
            slice_data = data_2d[k_start:k_start + axis_size, :]
        else:
            raise ValueError(f"Unsupported slice axis: {tensor_spec['slice_axis']}")

        save_before_view(
            output_paths["before_install_dir"],
            op_id,
            slice_idx,
            out_name,
            slice_data,
            file_order,
            tensor_spec["before_view"],
        )

        if tensor_spec["use_ring"]:
            reordered_slice = reorder_in0_slice_by_ring(
                slice_data,
                slice_idx,
                num_slices,
                dims["slice_k"],
                hw_params["ring_order"],
            )
            save_slice(output_paths["after_ring_dir"], op_id, slice_idx, out_name, reordered_slice, file_order)
            relayout_data = relayout_in0_N8K2N4K(reordered_slice, dims["K"], dims["slice_n"], num_slices)
        else:
            if role == "input":
                relayout_data = relayout_in1_L8K2L4K(slice_data, dims["slice_k"], dims["M"])
            elif role == "output":
                relayout_data = relayout_out_L8N8L4N4N2L1(slice_data, dims["slice_n"], dims["M"])
            else:
                raise ValueError(f"Unsupported non-ring role: {role}")

        save_slice_relayout(output_paths["install_logic_dir"], op_id, slice_idx, out_name, relayout_data, file_order)
        for install_slice_idx in install_target_slices(hw_params, slice_idx):
            save_slice_relayout(
                output_paths["install_dir"],
                op_id,
                install_slice_idx,
                out_name,
                relayout_data,
                file_order,
            )


def process_gemm_tensors(input_dir, output_dir, target_op, layer_id):
    op_spec = OP_SPECS[target_op]
    hw_params = op_spec["hw_params"]
    expected_dims = build_expected_shapes(target_op, MODEL_PARAMS)
    dims = build_runtime_dims(target_op, MODEL_PARAMS)
    dims.update(derive_slice_sizes(dims["K"], dims["N"], hw_params["num_slices"]))
    validate_config_for_op(target_op, dims, hw_params)

    matched_alias, matched_files = find_target_files(input_dir, target_op, layer_id, op_spec, expected_dims)

    print(f"Starting GEMM tensor split and relayout in: {input_dir}")
    print(f"  target_op={target_op}, matched_alias={matched_alias}, layer_id={layer_id}")
    print(f"  K={dims['K']}, M={dims['M']}, N={dims['N']}")
    if dims["raw_K"] != dims["K"]:
        print(f"  raw_K={dims['raw_K']} padded_to={dims['K']}")
    print(f"  slice_k={dims['slice_k']}, slice_n={dims['slice_n']}")

    for role, entry in matched_files.items():
        print(f"  {role}: {entry['filename']} | shape={entry['shape']}")

    output_paths = {
        "install_dir": os.path.join(output_dir, "install"),
        "install_logic_dir": os.path.join(output_dir, "install_logic"),
        "before_install_dir": os.path.join(output_dir, "install_beforerelayout"),
        "after_ring_dir": os.path.join(output_dir, "install_after_ring"),
    }
    op_id = "gemm"

    for role, entry in matched_files.items():
        tensor_spec = TENSOR_SPECS[role]
        filepath = entry["filepath"]
        expected_shape = tensor_spec["shape_fn"](expected_dims)
        data_2d = load_gemm_tensor(filepath, expected_shape, file_order=hw_params["file_order"])
        data_2d = pad_tensor_for_processing(role, data_2d, dims)
        process_tensor_role(role, data_2d, dims, hw_params, output_paths, op_id)

    print(
        f"\nAll GEMM tensors split and saved under: {output_paths['install_dir']} and {output_paths['install_logic_dir']}"
    )


if __name__ == "__main__":
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(args.input_dir or os.path.join(current_dir, "golde_data"))
    output_dir = os.path.abspath(args.output_dir or os.path.join(current_dir, "outputs"))
    target_ops = sorted(OP_SPECS.keys()) if args.target_op == "all" else [args.target_op]
    for target_op in target_ops:
        process_gemm_tensors(
            input_dir=input_dir,
            output_dir=os.path.join(output_dir, target_op),
            target_op=target_op,
            layer_id=args.layer_id,
        )
