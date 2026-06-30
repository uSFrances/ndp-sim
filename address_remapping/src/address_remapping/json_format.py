from __future__ import annotations

import json
import re
from typing import Iterable


DEFAULT_COMPACT_LIST_KEYS = {
    "remapping",
    "permutation",
    "layout_permutation",
    "physical_permutation",
    "composed_permutation",
    "output_permutation",
    "input_permutation",
    "internal_permutation",
    "internal_layout_permutation",
    "internal_physical_permutation",
    "internal_composed_permutation",
    "internal_output_permutation",
    "internal_input_permutation",
}


def render_json(
    payload: object,
    *,
    compact_list_keys: Iterable[str] | None = None,
    ensure_ascii: bool = False,
) -> str:
    rendered = json.dumps(payload, indent=2, ensure_ascii=ensure_ascii)
    keys = set(compact_list_keys or DEFAULT_COMPACT_LIST_KEYS)
    for key in sorted(keys, key=len, reverse=True):
        rendered = _collapse_numeric_list_block(rendered, key)
    return rendered


def _collapse_numeric_list_block(rendered: str, key: str) -> str:
    pattern = re.compile(
        rf'"{re.escape(key)}": \[\n((?:\s+-?\d+,?\n)+)\s+\]'
    )
    return pattern.sub(_collapse_matched_numbers, rendered)


def _collapse_matched_numbers(match: re.Match[str]) -> str:
    numbers = re.findall(r"-?\d+", match.group(1))
    prefix = match.group(0).split("[", 1)[0]
    return f"{prefix}[{', '.join(numbers)}]"
