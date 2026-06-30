from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "presentations" / "ttft_llama8b_batch1_table.csv"
OUTPUT_PATH = ROOT / "outputs" / "presentations" / "ttft_llama8b_batch1_table_summary.csv"


def _build_execution_scenario(row: dict[str, str]) -> str:
    parts: list[str] = []

    workload = row.get("workload_name", "").strip()
    if workload:
        parts.append(workload)

    batch_size = row.get("batch_size", "").strip()
    if batch_size:
        parts.append(f"batch={batch_size}")

    input_tokens = row.get("input_tokens", "").strip()
    if input_tokens:
        parts.append(f"in={input_tokens}")

    output_tokens = row.get("output_tokens", "").strip()
    if output_tokens:
        parts.append(f"out={output_tokens}")

    num_prompts = row.get("num_prompts", "").strip()
    if num_prompts:
        parts.append(f"prompts={num_prompts}")

    max_req = row.get("max_request_concurrency", "").strip()
    if max_req:
        parts.append(f"max_conc={max_req}")

    peak_req = row.get("peak_concurrent_requests", "").strip()
    if peak_req:
        parts.append(f"peak_conc={peak_req}")

    engine = row.get("engine", "").strip()
    if engine:
        parts.append(engine)

    return ", ".join(parts)


def _build_ttft_text(row: dict[str, str]) -> str:
    mean_ttft = row.get("mean_ttft_ms", "").strip()
    if not mean_ttft:
        return ""

    median_ttft = row.get("median_ttft_ms", "").strip()
    p99_ttft = row.get("p99_ttft_ms", "").strip()

    parts = [f"mean={mean_ttft} ms"]
    if median_ttft:
        parts.append(f"median={median_ttft} ms")
    if p99_ttft:
        parts.append(f"p99={p99_ttft} ms")
    return ", ".join(parts)


def main() -> None:
    with INPUT_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    summary_rows: list[dict[str, str]] = []
    for row in rows:
        summary_rows.append(
            {
                "model": row.get("model_name", ""),
                "hardware": row.get("hardware", ""),
                "compute": row.get("compute_text", ""),
                "bandwidth": row.get("bandwidth_text", ""),
                "execution_scenario": _build_execution_scenario(row),
                "ttft": _build_ttft_text(row),
                "source_section": row.get("source_section", ""),
                "source_url": row.get("source_url", ""),
                "notes": row.get("notes", ""),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
