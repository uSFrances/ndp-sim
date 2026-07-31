from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional


DEFAULT_MODEL_CONFIG_PATH = Path("examples") / "configs" / "config.json"
DEFAULT_SEQUENCE_MULTIPLE = 32


@dataclass(frozen=True)
class ModelExecutionConfig:
    config_path: Path
    model_name: str
    logical_hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    requested_sequence_length: int
    sequence_length: int
    slice_per_head: int
    used_slices: int
    kv_padding: int
    kv_padding_a: int
    kv_padding_b: int
    clusters: int
    padded_attention_heads: int
    attention_waves: int
    q_heads_per_cluster: int
    q_heads_per_kv_head: int
    kv_heads_per_cluster: int
    execution_hidden_size: int
    sequence_multiple: int

    def values(self) -> Dict[str, int]:
        return {
            "logical_hidden_size": self.logical_hidden_size,
            "requested_sequence_length": self.requested_sequence_length,
            "hidden_size": self.execution_hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.padded_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "sequence_length": self.sequence_length,
            "slice_per_head": self.slice_per_head,
            "used_slices": self.used_slices,
            "kv_padding": self.kv_padding,
            "kv_padding_a": self.kv_padding_a,
            "kv_padding_b": self.kv_padding_b,
            "clusters": self.clusters,
            "padded_attention_heads": self.padded_attention_heads,
            "attention_waves": self.attention_waves,
            "q_heads_per_cluster": self.q_heads_per_cluster,
            "q_heads_per_kv_head": self.q_heads_per_kv_head,
            "kv_heads_per_cluster": self.kv_heads_per_cluster,
        }

    def summary(self) -> Dict[str, object]:
        return {
            "hidden_size": self.logical_hidden_size,
            "model_name": self.model_name,
            "execution_hidden_size": self.execution_hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "padded_attention_heads": self.padded_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "requested_sequence_length": self.requested_sequence_length,
            "sequence_length": self.sequence_length,
            "slice_per_head": self.slice_per_head,
            "used_slices": self.used_slices,
            "kv_padding": self.kv_padding,
            "kv_padding_a": self.kv_padding_a,
            "kv_padding_b": self.kv_padding_b,
            "clusters": self.clusters,
            "attention_waves": self.attention_waves,
            "q_heads_per_cluster": self.q_heads_per_cluster,
            "q_heads_per_kv_head": self.q_heads_per_kv_head,
            "kv_heads_per_cluster": self.kv_heads_per_cluster,
            "sequence_multiple": self.sequence_multiple,
        }


def load_model_config(
    path: Path,
    *,
    sequence_length: Optional[int] = None,
    sequence_multiple: int = DEFAULT_SEQUENCE_MULTIPLE,
) -> ModelExecutionConfig:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    return build_model_execution_config(
        config,
        config_path=path,
        sequence_length=sequence_length,
        sequence_multiple=sequence_multiple,
    )


def build_model_execution_config(
    config: Mapping[str, object],
    *,
    config_path: Path,
    sequence_length: Optional[int],
    sequence_multiple: int,
) -> ModelExecutionConfig:
    logical_hidden = int(config["hidden_size"])
    model_name = str(config.get("model_name", config.get("name", config_path.stem))).strip() or config_path.stem
    intermediate_size = int(config["intermediate_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    num_hidden_layers = int(config["num_hidden_layers"])
    slice_per_head = int(config["slice_per_head"])
    used_slices = int(config["used_slices"])
    kv_padding = int(config.get("kv_padding", 0))
    kv_padding_a = int(config.get("kv_padding_a", kv_padding // 2 if kv_padding else 0))
    kv_padding_b = int(config.get("kv_padding_b", kv_padding * 2 if kv_padding else 0))
    requested_seq_len = int(sequence_length if sequence_length is not None else config["sequence_length"])
    seq_len = _round_up_to_multiple(requested_seq_len, sequence_multiple)

    if used_slices % slice_per_head != 0:
        raise ValueError("used_slices must be divisible by slice_per_head.")
    if head_dim % slice_per_head != 0:
        raise ValueError("head_dim must be divisible by slice_per_head.")
    if intermediate_size % used_slices != 0:
        raise ValueError("intermediate_size must be divisible by used_slices.")
    if (num_key_value_heads * head_dim) % slice_per_head != 0:
        raise ValueError("num_key_value_heads * head_dim must be divisible by slice_per_head.")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads for grouped KV mapping.")

    clusters = used_slices // slice_per_head
    padded_attention_heads = int(math.ceil(num_attention_heads / float(clusters)) * clusters)
    attention_waves = padded_attention_heads // clusters
    execution_hidden_size = padded_attention_heads * head_dim
    if execution_hidden_size % used_slices != 0:
        raise ValueError("execution hidden size must be divisible by used_slices.")

    q_heads_per_cluster = padded_attention_heads // clusters
    q_heads_per_kv_head = num_attention_heads // num_key_value_heads
    kv_heads_per_cluster = int(math.ceil(q_heads_per_cluster / float(q_heads_per_kv_head)))

    return ModelExecutionConfig(
        config_path=config_path,
        model_name=model_name,
        logical_hidden_size=logical_hidden,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        num_hidden_layers=num_hidden_layers,
        requested_sequence_length=requested_seq_len,
        sequence_length=seq_len,
        slice_per_head=slice_per_head,
        used_slices=used_slices,
        kv_padding=kv_padding,
        kv_padding_a=kv_padding_a,
        kv_padding_b=kv_padding_b,
        clusters=clusters,
        padded_attention_heads=padded_attention_heads,
        attention_waves=attention_waves,
        q_heads_per_cluster=q_heads_per_cluster,
        q_heads_per_kv_head=q_heads_per_kv_head,
        kv_heads_per_cluster=kv_heads_per_cluster,
        execution_hidden_size=execution_hidden_size,
        sequence_multiple=sequence_multiple,
    )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        raise ValueError("sequence_length must be positive.")
    if multiple <= 0:
        raise ValueError("sequence_length rounding multiple must be positive.")
    return int(math.ceil(value / float(multiple)) * multiple)
