from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


_SLICE_BANK_RE = re.compile(r"_slice(\d+)(?:_(\d+))?$")


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
    match = _SLICE_BANK_RE.search(entry_key)
    if match is None:
        raise ValueError(f"Cannot parse slice id from key: {entry_key}")
    return int(match.group(1))


def _extract_bank_id(entry_key: str) -> int | None:
    match = _SLICE_BANK_RE.search(entry_key)
    if match is None:
        return None
    bank_str = match.group(2)
    return int(bank_str) if bank_str is not None else None


def _resolve_slice_id(entry_key: str, base_addr: int) -> int:
    try:
        return _extract_slice_id(entry_key)
    except ValueError:
        slave, _, _, _, _ = _decode_addr_fields(base_addr)
        return slave


def _load_matrix_payload_bytes(data_path: Path) -> bytes | None:
    if data_path.is_file():
        if data_path.suffix.lower() == ".txt":
            return _parse_variable_width_txt(data_path)
        # Some bitstream files use .bin extension but contain text
        # (128-bit / 64-bit binary strings).  Auto-detect by peeking at
        # the first few bytes.
        raw = data_path.read_bytes()
        if _is_binary_text(raw):
            return _parse_bytes_as_variable_width(raw)
        return raw

    # Manifest may point to .txt while data exists as .bin, or vice versa.
    alt_path = data_path.with_suffix(".bin") if data_path.suffix.lower() != ".bin" else data_path.with_suffix(".txt")
    if alt_path.is_file():
        if alt_path.suffix.lower() == ".bin":
            raw = alt_path.read_bytes()
            if _is_binary_text(raw):
                return _parse_bytes_as_variable_width(raw)
            return raw
        return _parse_variable_width_txt(alt_path)
    return None


def _is_binary_text(data: bytes) -> bool:
    """Return True if *data* looks like a text file of 0/1 strings."""
    if not data:
        return False
    sample = data[:256].decode("ascii", errors="replace")
    allowed = set("01\r\n\t ")
    return all(ch in allowed for ch in sample)


def _parse_bytes_as_variable_width(raw: bytes) -> bytes:
    """Parse bytes containing 0/1 text lines into packed little-endian words."""
    text = raw.decode("ascii")
    words: list[bytes] = []
    detected_width: int | None = None
    for line in text.splitlines():
        bits = line.strip()
        if not bits:
            continue
        if any(ch not in "01" for ch in bits):
            raise ValueError(f"Invalid binary text: {bits[:64]}...")
        if detected_width is None:
            detected_width = len(bits)
            if detected_width not in (64, 128):
                raise ValueError(f"Unsupported line width {detected_width}")
        if len(bits) != detected_width:
            raise ValueError(
                f"Mixed line widths: expected {detected_width}, got {len(bits)}"
            )
        byte_count = detected_width // 8
        words.append(
            int(bits, 2).to_bytes(byte_count, byteorder="little", signed=False)
        )
    return b"".join(words)

    return None


def _parse_variable_width_txt(txt_path: Path) -> bytes:
    """Parse a binary text file that may contain 64-bit or 128-bit lines.

    SFU coefficient tables use 64-bit lines; regular matrix data uses
    128-bit lines.  The function auto-detects the width from the first
    non-empty line and treats all subsequent lines the same way.
    """
    words: list[bytes] = []
    detected_width: int | None = None
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            bits = line.strip()
            if not bits:
                continue
            if any(ch not in "01" for ch in bits):
                raise ValueError(
                    f"Invalid binary line in {txt_path}: {bits[:64]}..."
                )
            if detected_width is None:
                detected_width = len(bits)
                if detected_width not in (64, 128):
                    raise ValueError(
                        f"Unsupported line width {detected_width} in {txt_path}"
                    )
            if len(bits) != detected_width:
                raise ValueError(
                    f"Mixed line widths in {txt_path}: expected {detected_width}, got {len(bits)}"
                )
            byte_count = detected_width // 8
            words.append(
                int(bits, 2).to_bytes(byte_count, byteorder="little", signed=False)
            )
    return b"".join(words)


def _format_u32_word(word: int, output_format: str) -> str:
    if output_format == "hex":
        return f"0x{word:08X}"
    return f"{word:032b}"


def _resolve_manifest_path(manifest_arg: str) -> Path:
    manifest_path = Path(manifest_arg).expanduser()
    if manifest_path.is_file():
        return manifest_path.resolve()

    module_root = Path(__file__).resolve().parents[2]
    repo_root = module_root.parent
    search_roots = [Path.cwd(), module_root, repo_root]

    candidate_paths = [manifest_path]
    if manifest_path.parts and manifest_path.parts[0] == module_root.name:
        stripped = Path(*manifest_path.parts[1:])
        candidate_paths.append(stripped)

    for candidate in candidate_paths:
        if candidate.is_absolute() and candidate.is_file():
            return candidate.resolve()
        for root in search_roots:
            resolved = (root / candidate).resolve()
            if resolved.is_file():
                return resolved

    return manifest_path.resolve()


def _resolve_output_path(output_arg: str, *, prefer_existing_parent: bool = False) -> Path:
    output_path = Path(output_arg).expanduser()
    if output_path.is_absolute():
        return output_path.resolve()

    module_root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd().resolve()

    if cwd.name == module_root.name and output_path.parts and output_path.parts[0] == module_root.name:
        output_path = Path(*output_path.parts[1:])

    resolved = (cwd / output_path).resolve()
    if prefer_existing_parent and not resolved.parent.exists():
        return resolved
    return resolved


def _load_payloads(manifest_path: Path) -> list[MatrixPayload]:
    manifest = _load_manifest(manifest_path)
    root_dir = manifest_path.parent
    payloads: list[MatrixPayload] = []
    missing_count = 0

    for key, value in manifest.items():
        if not isinstance(value, dict):
            continue

        base_addr_raw = value.get("base_addr")
        rel_path = value.get("path")
        if not isinstance(base_addr_raw, str) or not isinstance(rel_path, str):
            continue

        base_addr = _parse_hex_addr(base_addr_raw)
        slice_id = _resolve_slice_id(key, base_addr)
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

        slave, bank, _, _, _ = _decode_addr_fields(base_addr)

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
        print("Warning: No payload files found from manifest entries — skipping bank data export.")
        return []

    if missing_count > 0:
        print(f"Skipped payload entries due to missing files: {missing_count}")

    return payloads


def _write_bank_file(
    output_path: Path,
    bank_data: bytes,
    *,
    line_width_bits: int = 32,
    output_format: str = "binary",
) -> None:
    """Write bank data as formatted words.

    Args:
        line_width_bits: 32 → one 32-bit word per line.
                        128 → four 32-bit words per line.
        output_format: "binary" or "hex".
    """
    if line_width_bits == 128:
        _write_bank_file_128bit(output_path, bank_data, output_format=output_format)
        return

    if len(bank_data) % 4 != 0:
        pad = 4 - (len(bank_data) % 4)
        bank_data = bank_data + (b"\x00" * pad)

    lines = []
    for i in range(0, len(bank_data), 4):
        word = int.from_bytes(bank_data[i : i + 4], byteorder="little", signed=False)
        lines.append(_format_u32_word(word, output_format))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bank_file_128bit(output_path: Path, bank_data: bytes, *, output_format: str = "binary") -> None:
    if len(bank_data) % 16 != 0:
        pad = 16 - (len(bank_data) % 16)
        bank_data = bank_data + (b"\x00" * pad)

    lines = []
    for i in range(0, len(bank_data), 16):
        chunk = bank_data[i : i + 16]
        if output_format == "hex":
            words = []
            for j in range(3, -1, -1):
                word = int.from_bytes(chunk[j * 4 : (j + 1) * 4], byteorder="little", signed=False)
                words.append(f"0x{word:08X}")
            lines.append(" ".join(words))
        else:
            line_value = int.from_bytes(chunk, byteorder="little", signed=False)
            lines.append(f"{line_value:0128b}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_bank_data(
    manifest_path: Path,
    output_dir: Path,
    *,
    line_width_bits: int = 32,
    output_format: str = "binary",
) -> list[Path]:
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
        written_starts: set[int] = set()
        overlapped: list[str] = []
        for item in sorted(items, key=lambda x: x.bank_offset_bytes):
            start = item.bank_offset_bytes
            end = start + len(item.data)

            # Skip entries that share the same physical start address
            # (e.g. operator output reused as downstream input — both
            #  manifest keys point to the same tensor data).
            if start in written_starts:
                continue

            existing = bank_image[start:end]
            if any(b != 0 for b in existing):
                free_start = start
                while free_start < end and bank_image[free_start] == 0:
                    free_start += 1
                avail = free_start - start
                if avail > 0:
                    bank_image[start:free_start] = item.data[:avail]
                    written_starts.add(start)
                    overlapped.append(
                        f"  {item.source_key} @ [{start}, {end}) "
                        f"(clipped to {avail} bytes, "
                        f"file={len(item.data)} bytes)"
                    )
            else:
                bank_image[start:end] = item.data
                written_starts.add(start)

        if overlapped:
            print(
                f"[bank_data] slice={slice_id:02d} bank={bank_id} — "
                f"{len(overlapped)} entry(s) clipped:\n"
                + "\n".join(overlapped)
            )

        if not bank_image:
            continue

        out_file = output_dir / f"slice{slice_id:02d}_Bank{bank_id:02d}_data.txt"
        _write_bank_file(
            out_file,
            bytes(bank_image),
            line_width_bits=line_width_bits,
            output_format=output_format,
        )
        written_files.append(out_file)

    return written_files


def export_combined_bank_data(
    manifest_path: Path,
    output_dir: Path,
    *,
    line_width_bits: int = 32,
    output_format: str = "binary",
) -> list[Path]:
    """Export per-slice combined bank data into a single file per slice.

    Within each slice file, bank N data is placed at offset
    ``N * max_bank_size``, where max_bank_size is the largest bank image
    size for that slice (rounded up to 16 bytes).
    """
    payloads = _load_payloads(manifest_path)
    grouped: dict[tuple[int, int], list[MatrixPayload]] = {}
    for payload in payloads:
        grouped.setdefault((payload.slice_id, payload.bank_id), []).append(payload)

    # Collect per-slice per-bank images.
    slice_bank_images: dict[int, dict[int, bytes]] = {}
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
                raise ValueError(
                    f"Overlapping payload detected: slice={slice_id}, bank={bank_id}, key={item.source_key}"
                )
            bank_image[start:end] = item.data

        slice_bank_images.setdefault(slice_id, {})[bank_id] = bytes(bank_image)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    BANK_COUNT = 4

    for slice_id in sorted(slice_bank_images.keys()):
        banks = slice_bank_images[slice_id]
        # Determine stride as max bank image size across all banks for this slice.
        max_bank_size = max((len(data) for data in banks.values()), default=0)
        if max_bank_size == 0:
            continue
        # Align stride to 16 bytes so each bank starts at a clean 128-bit boundary.
        stride = ((max_bank_size + 15) // 16) * 16

        combined_size = BANK_COUNT * stride
        combined = bytearray(combined_size)

        for bank_id in range(BANK_COUNT):
            data = banks.get(bank_id, b"")
            start = bank_id * stride
            combined[start : start + len(data)] = data

        out_file = output_dir / f"slice{slice_id:02d}_data.txt"
        _write_bank_file(
            out_file,
            bytes(combined),
            line_width_bits=line_width_bits,
            output_format=output_format,
        )
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
    parser.add_argument(
        "--combined",
        action="store_true",
        default=False,
        help="Export combined slice files (one file per slice) instead of separate per-bank files.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        choices=[32, 128],
        default=32,
        help="Output line width in bits: 32 (one word per line, default) or 128 (four words per line).",
    )
    parser.add_argument(
        "--output-format",
        choices=["binary", "hex"],
        default="hex",
        help="Output word format: binary (default) or hex.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    manifest_path = _resolve_manifest_path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    out_dir = _resolve_output_path(args.out_dir, prefer_existing_parent=True) if args.out_dir else (manifest_path.parent / "Bank_data")
    if args.combined:
        written = export_combined_bank_data(
            manifest_path=manifest_path,
            output_dir=out_dir,
            line_width_bits=args.line_width,
            output_format=args.output_format,
        )
    else:
        written = export_bank_data(
            manifest_path=manifest_path,
            output_dir=out_dir,
            line_width_bits=args.line_width,
            output_format=args.output_format,
        )

    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {out_dir}")
    print(f"Generated files: {len(written)}")
    for path in written[:10]:
        print(f"  - {path}")
    if len(written) > 10:
        print(f"  ... and {len(written) - 10} more")


if __name__ == "__main__":
    main()
