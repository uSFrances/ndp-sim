from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RegisterSegment:
    instance_name: str
    address: int
    high: int
    low: int


@dataclass(frozen=True)
class FieldBinding:
    instance_name: str
    config_name: str
    module_name: str
    field_high: int
    field_low: int
    segments: tuple[RegisterSegment, ...]


@dataclass(frozen=True)
class PartialRegisterWrite:
    address: int
    value: int
    mask: int


@dataclass
class RegisterMappingDB:
    field_bindings: dict[str, FieldBinding] = field(default_factory=dict)
    address_to_fields: dict[int, list[str]] = field(default_factory=dict)
    instance_segments: dict[str, tuple[RegisterSegment, ...]] = field(default_factory=dict)
    instance_const_addresses: dict[str, dict[str, int]] = field(default_factory=dict)

    def get_field(self, key: str) -> FieldBinding | None:
        return self.field_bindings.get(key)


MODULE_TO_INSTANCE_PREFIX = {
    # IGA
    ("IGA", "20 *DRAM LC"): "iga_lc",
    ("IGA", "5*BUFFER ROW LC"): "iga_row_lc",
    ("IGA", "5*BUFFER COL LC"): "iga_col_lc",
    ("IGA", "10*LC PE"): "iga_pe",
    # LSU
    ("LSU", "4*Read Memory Stream Engine"): "rd_stream",
    ("LSU", "1*Write Memory Stream Engine"): "wr_stream",
    ("LSU", "2*Neighbor Stream Engine"): "se_nse",
    ("LSU", "6*Buffer_Manager_Cluster"): "buffer_manager_cluster",
    # SA
    ("SA", "mode"): "special_array",
    ("SA", "3*Inport"): "special_array",
    ("SA", "1*PE"): "special_array",
    ("SA", "1*Outport"): "special_array",
    # GA
    ("GA", "16*PE"): "ga_pe",
    ("GA", "3*Inport"): "ga_inport_group",
    ("GA", "1*Outport"): "ga_outport_group",
}


def load_register_mapping(
    register_map_csv: str | Path,
    config_output_csv: str | Path,
    reorder_const_fields: bool = True,
    map_const_fields_to_const_addresses: bool = True,
) -> RegisterMappingDB:
    instance_segments, instance_const_addresses = _parse_config_output(config_output_csv)
    field_rows = _parse_register_map(
        register_map_csv,
        reorder_const_fields=reorder_const_fields,
    )

    db = RegisterMappingDB(
        instance_segments={
            instance_name: tuple(segments) for instance_name, segments in instance_segments.items()
        },
        instance_const_addresses=instance_const_addresses,
    )

    for group_name, module_name, config_name, field_high, field_low in field_rows:
        prefix = MODULE_TO_INSTANCE_PREFIX.get((group_name, module_name))
        if prefix is None:
            continue

        instances = sorted(name for name in instance_segments.keys() if name.startswith(prefix))
        for instance_name in instances:
            special_segments = None
            if map_const_fields_to_const_addresses:
                special_segments = _resolve_special_const_segments(
                    instance_name=instance_name,
                    module_name=module_name,
                    config_name=config_name,
                    instance_const_addresses=instance_const_addresses,
                )
            if special_segments is not None:
                segments = special_segments
            else:
                segments = _find_overlapping_segments(
                    segments=instance_segments[instance_name],
                    field_high=field_high,
                    field_low=field_low,
                )
            if not segments:
                continue

            key = f"{instance_name}.{config_name}"
            binding = FieldBinding(
                instance_name=instance_name,
                config_name=config_name,
                module_name=module_name,
                field_high=field_high,
                field_low=field_low,
                segments=tuple(segments),
            )
            db.field_bindings[key] = binding
            for seg in segments:
                db.address_to_fields.setdefault(seg.address, []).append(key)

    return db


def build_partial_register_writes(binding: FieldBinding, field_value: int) -> dict[int, int]:
    """Split one field value into affected register writes keyed by register address.

    The returned values only contain bits belonging to this field.
    Unrelated bits are left as 0 and should be merged with original register values later.
    """

    width = binding.field_high - binding.field_low + 1
    if field_value < 0 or field_value >= (1 << width):
        raise ValueError(
            f"field value out of range for {binding.instance_name}.{binding.config_name}: "
            f"value={field_value}, width={width}"
        )

    writes: dict[int, int] = {}
    for seg in binding.segments:
        overlap_low = max(seg.low, binding.field_low)
        overlap_high = min(seg.high, binding.field_high)
        if overlap_low > overlap_high:
            continue

        chunk_width = overlap_high - overlap_low + 1
        field_offset = overlap_low - binding.field_low
        segment_offset = overlap_low - seg.low

        chunk_value = (field_value >> field_offset) & ((1 << chunk_width) - 1)
        writes[seg.address] = writes.get(seg.address, 0) | (chunk_value << segment_offset)

    return writes


def build_masked_register_writes(binding: FieldBinding, field_value: int) -> dict[int, PartialRegisterWrite]:
    width = binding.field_high - binding.field_low + 1
    if field_value < 0 or field_value >= (1 << width):
        raise ValueError(
            f"field value out of range for {binding.instance_name}.{binding.config_name}: "
            f"value={field_value}, width={width}"
        )

    writes: dict[int, PartialRegisterWrite] = {}
    for seg in binding.segments:
        overlap_low = max(seg.low, binding.field_low)
        overlap_high = min(seg.high, binding.field_high)
        if overlap_low > overlap_high:
            continue

        chunk_width = overlap_high - overlap_low + 1
        field_offset = overlap_low - binding.field_low
        segment_offset = overlap_low - seg.low
        chunk_mask = ((1 << chunk_width) - 1) << segment_offset
        chunk_value = ((field_value >> field_offset) & ((1 << chunk_width) - 1)) << segment_offset

        existing = writes.get(seg.address)
        if existing is None:
            writes[seg.address] = PartialRegisterWrite(
                address=seg.address,
                value=chunk_value,
                mask=chunk_mask,
            )
        else:
            writes[seg.address] = PartialRegisterWrite(
                address=seg.address,
                value=existing.value | chunk_value,
                mask=existing.mask | chunk_mask,
            )

    return writes


def _parse_range_token(token: str) -> tuple[int, int] | None:
    t = token.strip()
    if not t.startswith("[") or not t.endswith("]"):
        return None
    body = t[1:-1]
    if ":" not in body:
        return None
    high_s, low_s = body.split(":", maxsplit=1)
    if not high_s.isdigit() or not low_s.isdigit():
        return None
    high = int(high_s)
    low = int(low_s)
    if high < low:
        high, low = low, high
    return high, low


def _parse_config_output(path: str | Path) -> tuple[dict[str, list[RegisterSegment]], dict[str, dict[str, int]]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    segments_by_instance: dict[str, list[RegisterSegment]] = {}
    const_addresses_by_instance: dict[str, dict[str, int]] = {}

    current_module = ""
    current_instance = ""
    for row in rows[1:]:
        if not row or all(not c.strip() for c in row):
            continue

        module = row[1].strip() if len(row) > 1 else ""
        instance = row[2].strip() if len(row) > 2 else ""
        address_s = row[3].strip() if len(row) > 3 else ""
        range_s = row[4].strip() if len(row) > 4 else ""

        if module:
            current_module = module
        if instance:
            current_instance = instance

        _ = current_module
        if not current_instance or not address_s:
            continue

        parsed = _parse_range_token(range_s)
        address = int(address_s, 2)
        if parsed is None:
            const_name = range_s.lower()
            if const_name in {"const0", "const1", "const2"}:
                const_addresses_by_instance.setdefault(current_instance, {})[const_name] = address
            continue
        high, low = parsed

        seg = RegisterSegment(
            instance_name=current_instance,
            address=address,
            high=high,
            low=low,
        )
        segments_by_instance.setdefault(current_instance, []).append(seg)

    for instance_name, segments in segments_by_instance.items():
        segments.sort(key=lambda s: (s.low, s.high))
        segments_by_instance[instance_name] = segments

    return segments_by_instance, const_addresses_by_instance


def _parse_register_map(
    path: str | Path,
    reorder_const_fields: bool,
) -> list[tuple[str, str, str, int, int]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    module_rows: dict[tuple[str, str], list[tuple[int, str]]] = {}
    special_rows: list[tuple[str, str, str, int]] = []
    current_group = ""
    current_module = ""

    for row in rows[1:]:
        if not row or all(not c.strip() for c in row):
            continue

        group_name = row[0].strip() if len(row) > 0 else ""
        module_name = row[1].strip() if len(row) > 1 else ""
        field_s = row[2].strip() if len(row) > 2 else ""
        config_name = row[3].strip() if len(row) > 3 else ""

        if group_name:
            current_group = group_name

        if module_name:
            current_module = module_name

        if not current_group or not current_module:
            continue

        field_width = _parse_field_width(field_s)
        if field_width is None:
            continue

        if reorder_const_fields and _is_special_constant_field(current_module, config_name):
            # Constant fields are routed to const0/1/2 addresses, and should not
            # consume the normal cumulative bit-space of regular fields.
            special_rows.append((current_group, current_module, config_name, field_width))
            continue

        # Use field width and row order to derive bit positions.
        # Some CSV ranges are known to be unreliable; unnamed rows still occupy bits
        # and must participate in accumulation to keep following fields aligned.
        module_rows.setdefault((current_group, current_module), []).append((field_width, config_name))

    out: list[tuple[str, str, str, int, int]] = []
    for (group_name, module_name), rows_for_module in module_rows.items():
        total_width = sum(width for width, _ in rows_for_module)
        next_high = total_width - 1
        for width, config_name in rows_for_module:
            high = next_high
            low = high - width + 1
            next_high = low - 1
            if config_name:
                out.append((group_name, module_name, config_name, high, low))

    if reorder_const_fields:
        for group_name, module_name, config_name, width in special_rows:
            out.append((group_name, module_name, config_name, width - 1, 0))

    return out
def _resolve_special_const_segments(
    instance_name: str,
    module_name: str,
    config_name: str,
    instance_const_addresses: dict[str, dict[str, int]],
) -> list[RegisterSegment] | None:
    const_addresses = instance_const_addresses.get(instance_name)
    if not const_addresses:
        return None

    if module_name == "10*LC PE" and ".constant" in config_name:
        inport_idx = _extract_inport_index(config_name)
        if inport_idx is None:
            return None
        const_addr = const_addresses.get(f"const{inport_idx}")
        if const_addr is None:
            return None
        return [
            RegisterSegment(
                instance_name=instance_name,
                address=const_addr,
                high=15,
                low=0,
            )
        ]

    if module_name == "16*PE" and ".constant" in config_name:
        inport_idx = _extract_inport_index(config_name)
        if inport_idx is None:
            return None
        const_addr = const_addresses.get(f"const{inport_idx}")
        if const_addr is None:
            return None
        return [
            RegisterSegment(
                instance_name=instance_name,
                address=const_addr,
                high=31,
                low=0,
            )
        ]

    return None


def _extract_inport_index(config_name: str) -> int | None:
    match = re.search(r"inport(\d)", config_name)
    if not match:
        return None
    return int(match.group(1))


def _is_special_constant_field(module_name: str, config_name: str) -> bool:
    if module_name == "10*LC PE" and ".constant" in config_name:
        return True
    if module_name == "16*PE" and ".constant" in config_name:
        return True
    return False


def _parse_field_width(field_s: str) -> int | None:
    # Prefer width prefix like "30bit[440:412]" or "1bit".
    match = re.search(r"(\d+)\s*bit", field_s)
    if not match:
        return None
    width = int(match.group(1))
    if width <= 0:
        return None
    return width


def _parse_field_range(field_s: str) -> tuple[int, int] | None:
    # format like 30bit[440:412]
    if "[" not in field_s or "]" not in field_s:
        return None
    range_token = field_s[field_s.find("[") : field_s.find("]") + 1]
    return _parse_range_token(range_token)


def _find_overlapping_segments(
    segments: list[RegisterSegment],
    field_high: int,
    field_low: int,
) -> list[RegisterSegment]:
    out: list[RegisterSegment] = []
    for seg in segments:
        if seg.high < field_low:
            continue
        if seg.low > field_high:
            continue
        out.append(seg)
    out.sort(key=lambda s: s.low)
    return out


