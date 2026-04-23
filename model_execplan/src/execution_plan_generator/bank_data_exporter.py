from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


_SLICE_RE = re.compile(r"_slice(\d+)$")


@dataclass(frozen=True)
class MatrixPayload:
    slice_id: int
    bank_id: int
    bank_offset_bytes: int
    data: bytes
    source_key: str
    source_path: Path


def _parse_hex_addr(text: str) -> int:
    normalized = text.replace("_", "").strip()
    return int(normalized, 16)


def _decode_addr_fields(addr: int) -> tuple[int, int, int, int, int]:
    slave = (addr >> 25) & 0x1F
    bank = (addr >> 23) & 0x03
    row = (addr >> 10) & 0x1FFF
    col = (addr >> 4) & 0x3F
    subword = addr & 0xF
    return slave, bank, row, col, subword


def _bank_offset_bytes(addr: int) -> int:
    _, _, row, col, subword = _decode_addr_fields(addr)
    return (row << 10) | (col << 4) | subword


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest format: expected object, got {type(data).__name__}")
    return data


def _extract_slice_id(entry_key: str) -> int:
    match = _SLICE_RE.search(entry_key)
    if match is None:
        raise ValueError(f"Cannot parse slice id from key: {entry_key}")
    return int(match.group(1))


def _load_matrix_payload_bytes(data_path: Path) -> bytes | None:
    if data_path.is_file():
        return data_path.read_bytes()

    if data_path.suffix.lower() == ".bin":
        txt_path = data_path.with_suffix(".txt")
        if txt_path.is_file():
            words: list[bytes] = []
            with txt_path.open("r", encoding="utf-8") as f:
                for line in f:
                    bits = line.strip()
                    if not bits:
                        continue
                    if len(bits) != 128 or any(ch not in "01" for ch in bits):
                        raise ValueError(f"Invalid 128-bit binary line in {txt_path}: {bits[:64]}...")
                    words.append(int(bits, 2).to_bytes(16, byteorder="big", signed=False))
            return b"".join(words)

    return None


def _load_payloads(manifest_path: Path) -> list[MatrixPayload]:
    manifest = _load_manifest(manifest_path)
    root_dir = manifest_path.parent
    payloads: list[MatrixPayload] = []
    missing_count = 0

    for key, value in manifest.items():
        if "_matrix" not in key or "_slice" not in key:
            continue
        if not isinstance(value, dict):
            continue

        base_addr_raw = value.get("base_addr")
        rel_path = value.get("path")
        if not isinstance(base_addr_raw, str) or not isinstance(rel_path, str):
            continue

        slice_id = _extract_slice_id(key)
        abs_path = root_dir / rel_path
        file_bytes = _load_matrix_payload_bytes(abs_path)
        if file_bytes is None:
            missing_count += 1
            continue
        if not file_bytes:
            continue

        # Address planning is 128-bit granularity; pad tail to 16B if needed.
        if len(file_bytes) % 16 != 0:
            pad = 16 - (len(file_bytes) % 16)
            file_bytes = file_bytes + (b"\x00" * pad)

        base_addr = _parse_hex_addr(base_addr_raw)
        slave, bank, _, _, _ = _decode_addr_fields(base_addr)
        # if slave != slice_id:
        #     raise ValueError(
        #         f"Slice mismatch for {key}: key slice={slice_id}, addr slave={slave}, addr=0x{base_addr:08X}"
        #     )

        payloads.append(
            MatrixPayload(
                slice_id=slice_id,
                bank_id=bank,
                bank_offset_bytes=_bank_offset_bytes(base_addr),
                data=file_bytes,
                source_key=key,
                source_path=abs_path,
            )
        )

    if not payloads:
        raise ValueError("No matrix payload files found from manifest entries")
    if missing_count > 0:
        print(f"Skipped matrix entries due to missing files: {missing_count}")

    return payloads


def _write_bank_file(output_path: Path, bank_data: bytes) -> None:
    if len(bank_data) % 4 != 0:
        pad = 4 - (len(bank_data) % 4)
        bank_data = bank_data + (b"\x00" * pad)

    lines = []
    for i in range(0, len(bank_data), 4):
        word = int.from_bytes(bank_data[i : i + 4], byteorder="big", signed=False)
        lines.append(f"0x{word:08X}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_bank_data(manifest_path: Path, output_dir: Path) -> list[Path]:
    payloads = _load_payloads(manifest_path)
    grouped: dict[tuple[int, int], list[MatrixPayload]] = {}
    for payload in payloads:
        grouped.setdefault((payload.slice_id, payload.bank_id), []).append(payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    for (slice_id, bank_id), items in sorted(grouped.items()):
        max_end = 0
        for item in items:
            end = item.bank_offset_bytes + len(item.data)
            if end > max_end:
                max_end = end

        bank_image = bytearray(max_end)
        for item in sorted(items, key=lambda x: x.bank_offset_bytes):
            start = item.bank_offset_bytes
            end = start + len(item.data)
            existing = bank_image[start:end]
            if any(b != 0 for b in existing):
                # Reject overlapping writes to keep output deterministic.
                raise ValueError(
                    f"Overlapping payload detected: slice={slice_id}, bank={bank_id}, key={item.source_key}"
                )
            bank_image[start:end] = item.data

        if not bank_image:
            continue

        out_file = output_dir / f"slice{slice_id:02d}_Bank{bank_id:02d}_data.txt"
        _write_bank_file(out_file, bytes(bank_image))
        written_files.append(out_file)

    return written_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export per-slice per-bank data image from sca_cfg manifest and matrix binary files."
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        default="output/rmsnorm/sca_cfg.json",
        help="Path to sca_cfg.json (default: output/rmsnorm/sca_cfg.json)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for bank files (default: <manifest_dir>/Bank_data)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (manifest_path.parent / "Bank_data")
    written = export_bank_data(manifest_path=manifest_path, output_dir=out_dir)

    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {out_dir}")
    print(f"Generated files: {len(written)}")
    for path in written[:10]:
        print(f"  - {path}")
    if len(written) > 10:
        print(f"  ... and {len(written) - 10} more")


if __name__ == "__main__":
    main()
