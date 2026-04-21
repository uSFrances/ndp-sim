from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .register_mapping import RegisterMappingDB


@dataclass(frozen=True)
class InstanceConfigStream:
    instance_name: str
    chunk_payload_bits: int
    words: tuple[int, ...]
    padding_bits: int = 0
    enabled_chunks: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class TemplateConfigStream:
    instances: tuple[InstanceConfigStream, ...]


@dataclass(frozen=True)
class DecodedRegisterState:
    register_values: dict[int, int]
    enabled_register_addresses: set[int]


@dataclass(frozen=True)
class ModuleChunkSpec:
    section_name: str
    instance_prefix: str
    chunk_payload_bits: int
    chunk_count: int
    padding_bits: int = 0


DEFAULT_MODULE_SPECS = {
    "iga_lc": ModuleChunkSpec("iga_lc", "iga_lc", 60, 1),
    "iga_row_lc": ModuleChunkSpec("iga_row_lc", "iga_row_lc", 17, 1),
    "iga_col_lc": ModuleChunkSpec("iga_col_lc", "iga_col_lc", 26, 1),
    "iga_pe": ModuleChunkSpec("iga_pe", "iga_pe", 48, 2),
    # Parsed chunk payloads are already extracted to exact payload width, so no
    # extra start padding should be applied when decoding packed bits.
    "se_rd_mse": ModuleChunkSpec("se_rd_mse", "rd_stream", 58, 10, 0),
    "se_wr_mse": ModuleChunkSpec("se_wr_mse", "wr_stream", 62, 8, 0),
    "se_nse": ModuleChunkSpec("se_nse", "se_nse", 8, 1),
    "buffer_manager_cluster": ModuleChunkSpec(
        "buffer_manager_cluster", "buffer_manager_cluster", 21, 1
    ),
    "special_array": ModuleChunkSpec("special_array", "special_array", 32, 1),
    "ga_inport_group": ModuleChunkSpec("ga_inport_group", "ga_inport_group", 20, 1),
    "ga_outport_group": ModuleChunkSpec("ga_outport_group", "ga_outport_group", 12, 1),
    "ga_pe": ModuleChunkSpec("ga_pe", "ga_pe", 36, 4),
}


def load_template_config_stream(
    raw: dict[str, Any],
    template_dir: Path,
    register_db: RegisterMappingDB,
) -> TemplateConfigStream:
    if "bitstream_file" in raw:
        return _load_template_from_bitstream_file(raw, template_dir, register_db)

    raw_instances = raw.get("instances", [])
    if not isinstance(raw_instances, list):
        raise ValueError("initial_config_stream.instances must be a list")

    instances: list[InstanceConfigStream] = []
    for item in raw_instances:
        if not isinstance(item, dict):
            raise ValueError("initial_config_stream.instances items must be objects")
        instance_name = _require_str(item, "instance_name")
        chunk_payload_bits = _require_int(item, "chunk_payload_bits")
        words = _load_words(item, template_dir)
        instances.append(
            InstanceConfigStream(
                instance_name=instance_name,
                chunk_payload_bits=chunk_payload_bits,
                padding_bits=0,
                words=tuple(words),
                enabled_chunks=None,
            )
        )

    return TemplateConfigStream(instances=tuple(instances))


def decode_initial_register_values(
    config_stream: TemplateConfigStream,
    register_db: RegisterMappingDB,
) -> dict[int, int]:
    return decode_initial_register_state(config_stream, register_db).register_values


def decode_initial_register_state(
    config_stream: TemplateConfigStream,
    register_db: RegisterMappingDB,
) -> DecodedRegisterState:
    register_values: dict[int, int] = {}
    enabled_register_addresses: set[int] = set()

    for instance_stream in config_stream.instances:
        segments = register_db.instance_segments.get(instance_stream.instance_name)
        if not segments:
            continue

        chunk_width = instance_stream.chunk_payload_bits
        packed_text = "".join(f"{word:0{chunk_width}b}" for word in instance_stream.words)
        packed_bits = int(packed_text, 2) if packed_text else 0

        for seg in segments:
            width = seg.high - seg.low + 1
            value_mask = (1 << width) - 1
            # In parsed bitstream mode, padding is inserted at the stream start,
            # so field bit positions are shifted by padding_bits.
            bit_low = seg.low + instance_stream.padding_bits
            bit_high = bit_low + width - 1
            if _is_bit_range_enabled(
                bit_low,
                bit_high,
                chunk_width,
                instance_stream.enabled_chunks,
            ):
                enabled_register_addresses.add(seg.address)
            register_values[seg.address] = (packed_bits >> bit_low) & value_mask

        # const0/1/2 rows are stored as standalone addresses in config_output.
        # When the instance is enabled in parsed bitstream, allow writes to them.
        const_addresses = register_db.instance_const_addresses.get(instance_stream.instance_name, {})
        if const_addresses and _is_instance_enabled(instance_stream.enabled_chunks):
            enabled_register_addresses.update(const_addresses.values())


    return DecodedRegisterState(
        register_values=register_values,
        enabled_register_addresses=enabled_register_addresses,
    )


def _load_template_from_bitstream_file(
    raw: dict[str, Any],
    template_dir: Path,
    register_db: RegisterMappingDB,
) -> TemplateConfigStream:
    bitstream_file = raw.get("bitstream_file")
    if not isinstance(bitstream_file, str) or not bitstream_file.strip():
        raise ValueError("bitstream_file must be a non-empty string")

    bitstream_path = (template_dir / bitstream_file).resolve()
    sections = _parse_bitstream_sections(bitstream_path)
    module_specs = _load_module_specs(raw)

    instances: list[InstanceConfigStream] = []
    for section_name, section_lines in sections.items():
        spec = module_specs.get(section_name)
        if spec is None:
            continue
        if len(section_lines) % spec.chunk_count != 0:
            raise ValueError(
                f"Section {section_name} line count {len(section_lines)} is not divisible by chunk count {spec.chunk_count}."
            )

        instance_names = _get_instance_names(register_db, spec.instance_prefix)
        expected_line_count = len(instance_names) * spec.chunk_count
        if len(section_lines) != expected_line_count:
            raise ValueError(
                "Section "
                f"{section_name} line count mismatch: got {len(section_lines)}, expected {expected_line_count} "
                f"(instances={len(instance_names)} * chunk_count={spec.chunk_count}). "
                "Disabled instances must still reserve all chunk lines with leading 0 markers "
                "(e.g. iga_pe with chunk_count=2 requires two 0 lines per disabled instance)."
            )

        required_instances = len(section_lines) // spec.chunk_count

        for instance_idx in range(required_instances):
            chunk_lines = section_lines[
                instance_idx * spec.chunk_count : (instance_idx + 1) * spec.chunk_count
            ]
            parsed_chunks = [
                _parse_chunk_line_with_enable(line, spec.chunk_payload_bits) for line in chunk_lines
            ]
            words = tuple(int(payload, 2) for _, payload in parsed_chunks)
            enabled_chunks = tuple(enabled for enabled, _ in parsed_chunks)
            instances.append(
                InstanceConfigStream(
                    instance_name=instance_names[instance_idx],
                    chunk_payload_bits=spec.chunk_payload_bits,
                    padding_bits=spec.padding_bits,
                    words=words,
                    enabled_chunks=enabled_chunks,
                )
            )

    return TemplateConfigStream(instances=tuple(instances))


def _load_module_specs(raw: dict[str, Any]) -> dict[str, ModuleChunkSpec]:
    specs = dict(DEFAULT_MODULE_SPECS)
    overrides = raw.get("module_layouts")
    if not overrides:
        return specs
    if not isinstance(overrides, dict):
        raise ValueError("module_layouts must be an object")

    for section_name, item in overrides.items():
        if not isinstance(item, dict):
            raise ValueError("module_layouts values must be objects")
        base = specs.get(section_name)
        specs[section_name] = ModuleChunkSpec(
            section_name=section_name,
            instance_prefix=_get_override_str(item, "instance_prefix", base.instance_prefix if base else section_name),
            chunk_payload_bits=_get_override_int(item, "chunk_payload_bits", base.chunk_payload_bits if base else 0),
            chunk_count=_get_override_int(item, "chunk_count", base.chunk_count if base else 0),
            padding_bits=_get_override_int(item, "padding_bits", base.padding_bits if base else 0),
        )

    return specs


def _parse_bitstream_sections(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]

    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":"):
            current_section = line[:-1]
            sections[current_section] = []
            continue
        if current_section is None:
            continue
        sections[current_section].append(line)

    return sections


def _get_instance_names(register_db: RegisterMappingDB, prefix: str) -> list[str]:
    names = [name for name in register_db.instance_segments.keys() if name.startswith(prefix)]
    names.sort(key=_instance_sort_key)
    return names


def _instance_sort_key(name: str) -> tuple[str, int]:
    suffix = ""
    for idx in range(len(name) - 1, -1, -1):
        if not name[idx].isdigit():
            suffix = name[idx + 1 :]
            prefix = name[: idx + 1]
            if suffix:
                return prefix, int(suffix)
            return name, -1
    return "", int(name) if name.isdigit() else -1


def _parse_chunk_line(line: str, chunk_payload_bits: int) -> str:
    _, payload = _parse_chunk_line_with_enable(line, chunk_payload_bits)
    return payload


def _parse_chunk_line_with_enable(line: str, chunk_payload_bits: int) -> tuple[bool, str]:
    if line == "0":
        return False, "0" * chunk_payload_bits
    if line == "1":
        return True, "0" * chunk_payload_bits

    parts = line.split(maxsplit=1)
    if len(parts) != 2 or parts[0] not in {"0", "1"}:
        raise ValueError(f"Invalid chunk line: {line}")

    enabled, payload = parts
    if enabled == "0":
        return False, "0" * chunk_payload_bits
    if len(payload) != chunk_payload_bits:
        raise ValueError(
            f"Chunk payload length mismatch. Expected {chunk_payload_bits}, got {len(payload)}: {line}"
        )
    if any(ch not in {"0", "1"} for ch in payload):
        raise ValueError(f"Chunk payload contains non-binary characters: {line}")
    return True, payload


def _is_bit_range_enabled(
    bit_low: int,
    bit_high: int,
    chunk_payload_bits: int,
    enabled_chunks: tuple[bool, ...] | None,
) -> bool:
    if enabled_chunks is None:
        return True
    if bit_low < 0 or bit_high < bit_low:
        return False

    first_chunk = bit_low // chunk_payload_bits
    last_chunk = bit_high // chunk_payload_bits
    if first_chunk >= len(enabled_chunks) or last_chunk >= len(enabled_chunks):
        return False

    for chunk_idx in range(first_chunk, last_chunk + 1):
        if not enabled_chunks[chunk_idx]:
            return False
    return True


def _is_instance_enabled(enabled_chunks: tuple[bool, ...] | None) -> bool:
    if enabled_chunks is None:
        return True
    return any(enabled_chunks)


def _load_words(item: dict[str, Any], template_dir: Path) -> list[int]:
    raw_words = item.get("words")
    words_file = item.get("words_file")

    if raw_words is not None:
        if not isinstance(raw_words, list):
            raise ValueError("words must be a list")
        return [_parse_word(v) for v in raw_words]

    if words_file is not None:
        if not isinstance(words_file, str) or not words_file.strip():
            raise ValueError("words_file must be a non-empty string")
        path = template_dir / words_file
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("words_file content must be a JSON list")
        return [_parse_word(v) for v in data]

    raise ValueError("Each initial_config_stream instance must define words or words_file")


def _parse_word(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("0x"):
            return int(text, 16)
        if text.startswith("0b"):
            return int(text, 2)
        return int(text)
    raise ValueError(f"Unsupported word value: {value!r}")


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _get_override_int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _get_override_str(data: dict[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value
