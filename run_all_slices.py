import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / 'jsons' / 'prefill_gemm_ring_4slice.json'
DEFAULT_TEMPLATE_JSON = DEFAULT_CONFIG_PATH
DEFAULT_GENERATED_JSON_DIR = BASE_DIR / 'generated_jsons'
DEFAULT_OUTPUT_BASE_DIR = BASE_DIR / 'slices_output'
DEFAULT_JSON_OUT_DIR = BASE_DIR / 'jsons'
BITSTREAM_CMD = [sys.executable, "bitstream/main.py", "--visualize-placement", "-c", "{json_path}", "-o", "{out_dir}"]
DEFAULT_SLICE_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Generate per-slice ring GEMM JSONs and bitstreams. '
            'Optionally generate a new base ring JSON from a template using M/N/K/slices.'
        )
    )
    parser.add_argument('--config-path', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--template-json', type=Path, default=DEFAULT_TEMPLATE_JSON)
    parser.add_argument('--json-out-dir', type=Path, default=DEFAULT_JSON_OUT_DIR)
    parser.add_argument('--generated-json-dir', type=Path, default=DEFAULT_GENERATED_JSON_DIR)
    parser.add_argument('--output-base-dir', type=Path, default=DEFAULT_OUTPUT_BASE_DIR)
    parser.add_argument('--M', type=int)
    parser.add_argument('--N', type=int)
    parser.add_argument('--K', type=int)
    parser.add_argument('--slices', type=int)
    return parser.parse_args()


def set_base_addr_high5(addr_hex, slice_id):
    if addr_hex.startswith('0b'):
        bin_str = addr_hex[2:].replace('_', '')
        new_bin = f'{slice_id:05b}' + bin_str[5:]
        new_bin_fmt = '_'.join([new_bin[:5], new_bin[5:7], new_bin[7:20], new_bin[20:26], new_bin[26:30]])
        return '0b' + new_bin_fmt
    val = int(addr_hex, 16)
    val = (val & ~(0b11111 << 25)) | (slice_id << 25)
    return hex(val)


def validate_ring_dims(M: int, N: int, K: int, slices: int) -> tuple[int, int, int]:
    if min(M, N, K, slices) <= 0:
        raise ValueError('M, N, K, and slices must be positive integers')
    if N % slices != 0:
        raise ValueError(f'N={N} must be divisible by slices={slices}')
    if K % slices != 0:
        raise ValueError(f'K={K} must be divisible by slices={slices}')

    nb = N // slices
    ka = K // slices
    kb = K

    if M % 32 != 0:
        raise ValueError(f'M={M} must be divisible by 32')
    if nb % 32 != 0:
        raise ValueError(f'NB={nb} must be divisible by 32')
    if ka % 2 != 0:
        raise ValueError(f'KA={ka} must be divisible by 2')
    if kb % 4 != 0:
        raise ValueError(f'KB={kb} must be divisible by 4')

    return nb, ka, kb


def generate_ring_gemm_json(template_json: Path, M: int, N: int, K: int, slices: int, json_out_dir: Path) -> Path:
    nb, ka, kb = validate_ring_dims(M, N, K, slices)

    with template_json.open('r', encoding='utf-8') as f:
        cfg = json.load(f)

    dram_lc = cfg['dram_loop_configs']
    dram_lc['LC0']['end'] = M // 32
    dram_lc['LC1']['end'] = nb // 32
    dram_lc['LC2']['end'] = ka // 2
    dram_lc['LC4']['end'] = kb // 4

    pe_cfg = cfg['lc_pe_configs']
    pe_cfg['PE0']['inport1']['constant'] = 2 * ka
    pe_cfg['PE1']['inport1']['constant'] = 2 * kb
    pe_cfg['PE3']['inport1']['constant'] = nb // 2

    cfg['gemm_shape'] = {
        'M': int(M),
        'N': int(nb),
        'K': int(kb),
    }
    cfg['ring_gemm_shape'] = {
        'M': int(M),
        'N_full': int(N),
        'NB': int(nb),
        'KA': int(ka),
        'KB': int(kb),
        'slice_count': int(slices),
    }

    json_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = json_out_dir / f'gemm_config_ring_M{M}N{nb}KA{ka}KB{kb}.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4)

    print(f'[OK] Generated base ring GEMM JSON: {out_path}')
    return out_path


def update_json_for_slice(json_path: Path, slice_id: int, generated_json_dir: Path) -> Path:
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    for i in range(4):
        key = f'stream{i}'
        if key in data['stream_engine']:
            addr = data['stream_engine'][key]['base_addr']
            data['stream_engine'][key]['base_addr'] = set_base_addr_high5(addr, slice_id)
    generated_json_dir.mkdir(parents=True, exist_ok=True)
    stem = json_path.stem
    tmp_path = generated_json_dir / f'{stem}_slice{slice_id}.json'
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    return tmp_path


def run_and_check(json_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'terminal.log'
    while True:
        cmd = [arg.format(json_path=json_path, out_dir=out_dir) for arg in BITSTREAM_CMD]
        print(f'Running: {" ".join(cmd)}')
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        combined_output = (result.stdout or '') + (result.stderr or '')
        # Print to terminal in real-time (already captured, print now)
        sys.stdout.write(combined_output)
        sys.stdout.flush()
        # Write to log file
        with log_path.open('w', encoding='utf-8') as f:
            f.write(combined_output)
        if '[✓] Mapping successful with zero violations' in combined_output:
            print(f'Slice {out_dir} mapping successful!')
            break
        print('Mapping failed, retrying in 2 seconds...')
        time.sleep(2)


def resolve_base_config(args: argparse.Namespace) -> tuple[Path, int]:
    dimension_args = [args.M, args.N, args.K, args.slices]
    provided_count = sum(value is not None for value in dimension_args)
    if provided_count == 0:
        config_path = args.config_path.resolve()
        if not config_path.exists():
            raise FileNotFoundError(f'Base config JSON not found: {config_path}')
        return config_path, DEFAULT_SLICE_COUNT
    if provided_count != 4:
        raise ValueError('When using generated ring JSON mode, --M --N --K --slices must all be provided')

    template_json = args.template_json.resolve()
    if not template_json.exists():
        raise FileNotFoundError(f'Template JSON not found: {template_json}')

    config_path = generate_ring_gemm_json(
        template_json=template_json,
        M=int(args.M),
        N=int(args.N),
        K=int(args.K),
        slices=int(args.slices),
        json_out_dir=args.json_out_dir.resolve(),
    )
    return config_path, int(args.slices)


def main() -> None:
    args = parse_args()
    base_config_path, slice_count = resolve_base_config(args)
    generated_json_dir = args.generated_json_dir.resolve()
    output_base_dir = args.output_base_dir.resolve()

    stem = base_config_path.stem
    for slice_id in range(slice_count):
        json_path = update_json_for_slice(base_config_path, slice_id, generated_json_dir)
        out_dir = output_base_dir / f'{stem}_slice{slice_id}'
        run_and_check(json_path, out_dir)


if __name__ == '__main__':
    main()
