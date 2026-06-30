from __future__ import annotations

import csv
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "outputs" / "presentations" / "ttft_llama8b_batch1_table.csv"

GPUSTACK_PAGES = [
    {
        "url": "https://docs.gpustack.ai/2.0/performance-lab/qwen3-8b/910b/?utm_source=chatgpt.com",
        "model_name": "Qwen3-8B",
        "hardware": "Ascend 910B",
        "hardware_count": "1",
    },
    {
        "url": "https://docs.gpustack.ai/2.0/performance-lab/qwen3-8b/h100-latency/",
        "model_name": "Qwen3-8B",
        "hardware": "H100",
        "hardware_count": "1",
    },
    {
        "url": "https://docs.gpustack.ai/2.0/performance-lab/qwen3-14b/a100/",
        "model_name": "Qwen3-14B",
        "hardware": "A100",
        "hardware_count": "1",
    },
    {
        "url": "https://docs.gpustack.ai/2.0/performance-lab/qwen3-14b/h100/",
        "model_name": "Qwen3-14B",
        "hardware": "H100",
        "hardware_count": "1",
    },
    {
        "url": "https://docs.gpustack.ai/2.0/performance-lab/qwen3-32b/a100/",
        "model_name": "Qwen3-32B",
        "hardware": "A100",
        "hardware_count": "1",
    },
    {
        "url": "https://docs.gpustack.ai/2.0/performance-lab/qwen3-32b/h100/",
        "model_name": "Qwen3-32B",
        "hardware": "H100",
        "hardware_count": "1",
    },
    {
        "url": "https://docs.gpustack.ai/2.0/performance-lab/qwen3-30b-a3b/910b/",
        "model_name": "Qwen3-30B-A3B",
        "hardware": "Ascend 910B",
        "hardware_count": "1",
    },
    {
        "url": "https://docs.gpustack.ai/2.0/performance-lab/qwen3-235b-a22b/a100/",
        "model_name": "Qwen3-235B-A22B",
        "hardware": "A100",
        "hardware_count": "8",
    },
    {
        "url": "https://docs.gpustack.ai/2.0/performance-lab/qwen3-235b-a22b/h100/",
        "model_name": "Qwen3-235B-A22B",
        "hardware": "H100",
        "hardware_count": "8",
    },
]

HARDWARE_SPECS = {
    "A100": {"compute_text": "~312 TFLOPS FP16", "bandwidth_text": "600 GB/s"},
    "H100": {"compute_text": "800TFLOPS FP16", "bandwidth_text": "2TB/s"},
    "Ascend 910B": {"compute_text": "~313 TFLOPS FP16", "bandwidth_text": "392 GB/s"},
}


def _norm_latency_ms(text: str) -> str:
    value = text.strip().lower()
    if value.endswith("ms"):
        return value[:-2]
    if value.endswith("s"):
        return f"{float(value[:-1]) * 1000.0:.2f}"
    if value.endswith("min"):
        return f"{float(value[:-3]) * 60000.0:.2f}"
    return ""


def _normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\d)\s+\.\s+(\d)", r"\1.\2", text)
    text = re.sub(r"\(\s*ms\s*\)", "(ms)", text)
    text = re.sub(r"\(\s*req/s\s*\)", "(req/s)", text)
    text = re.sub(r"\(\s*tok/s\s*\)", "(tok/s)", text)
    return text.strip()


def _fetch_article_text(url: str) -> str:
    html = requests.get(url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup.find("main")
    if article is None:
        raise RuntimeError(f"Unable to locate article content for {url}")
    return _normalize_space(article.get_text(" ", strip=True))


def _extract_number(block: str, label: str) -> str:
    pattern = rf"{re.escape(label)}\s*:\s*([\d\.\s]+)"
    match = re.search(pattern, block)
    if not match:
        return ""
    return match.group(1).replace(" ", "")


def _extract_first_benchmark_block(section: str) -> str:
    marker = "============ Serving Benchmark Result ============"
    start = section.find(marker)
    if start == -1:
        return ""
    end = section.find("==================================================", start)
    if end == -1:
        return section[start:]
    return section[start : end + len("==================================================")]


def _metrics_from_block(block: str) -> dict[str, str]:
    return {
        "num_prompts": _extract_number(block, "Successful requests"),
        "max_request_concurrency": _extract_number(block, "Maximum request concurrency"),
        "peak_concurrent_requests": _extract_number(block, "Peak concurrent requests"),
        "request_throughput_req_s": _extract_number(block, "Request throughput (req/s)"),
        "output_token_throughput_tok_s": _extract_number(block, "Output token throughput (tok/s)"),
        "peak_output_token_throughput_tok_s": _extract_number(block, "Peak output token throughput (tok/s)"),
        "total_token_throughput_tok_s": _extract_number(block, "Total Token throughput (tok/s)"),
        "mean_ttft_ms": _extract_number(block, "Mean TTFT (ms)"),
        "median_ttft_ms": _extract_number(block, "Median TTFT (ms)"),
        "p99_ttft_ms": _extract_number(block, "P99 TTFT (ms)"),
        "mean_tpot_ms": _extract_number(block, "Mean TPOT (ms)"),
        "median_tpot_ms": _extract_number(block, "Median TPOT (ms)"),
        "p99_tpot_ms": _extract_number(block, "P99 TPOT (ms)"),
        "mean_itl_ms": _extract_number(block, "Mean ITL (ms)"),
        "median_itl_ms": _extract_number(block, "Median ITL (ms)"),
        "p99_itl_ms": _extract_number(block, "P99 ITL (ms)"),
    }


def _infer_workload_tokens(workload_name: str) -> tuple[str, str, str]:
    key = workload_name.strip().lower()
    if key == "random 32k input":
        return "random", "32768", "100"
    if key == "random 4k input":
        return "random", "4096", "200"
    if key == "random 2k input":
        return "random", "2048", "100"
    if key == "random 128 input":
        return "random", "128", "4"
    if "sharegpt" in key:
        return "ShareGPT", "", ""
    return "", "", ""


def _base_gpustack_row(meta: dict[str, str]) -> dict[str, str]:
    specs = HARDWARE_SPECS.get(meta["hardware"], {})
    return {
        "source_name": "GPUStack Performance Lab",
        "source_section": "",
        "source_url": meta["url"],
        "record_type": "online_serving_benchmark",
        "hardware": meta["hardware"],
        "hardware_count": meta["hardware_count"],
        "model_name": meta["model_name"],
        "precision_or_quantization": "BF16",
        "engine": "vLLM",
        "serving_script": f"vllm serve Qwen/{meta['model_name']}",
        "scenario_type": "online_serving_benchmark",
        "workload_name": "",
        "dataset_name": "",
        "batch_size": "",
        "input_tokens": "",
        "output_tokens": "",
        "num_prompts": "",
        "max_request_concurrency": "",
        "peak_concurrent_requests": "",
        "compute_text": specs.get("compute_text", ""),
        "bandwidth_text": specs.get("bandwidth_text", ""),
        "request_throughput_req_s": "",
        "output_token_throughput_tok_s": "",
        "peak_output_token_throughput_tok_s": "",
        "total_token_throughput_tok_s": "",
        "mean_ttft_ms": "",
        "median_ttft_ms": "",
        "p99_ttft_ms": "",
        "mean_tpot_ms": "",
        "median_tpot_ms": "",
        "p99_tpot_ms": "",
        "mean_itl_ms": "",
        "median_itl_ms": "",
        "p99_itl_ms": "",
        "comparison_value": "",
        "notes": "",
    }


def _ppt_rows() -> list[dict[str, str]]:
    specs = {
        "Groq LPU": {"compute_text": "1.2PFLOP/s FP8", "bandwidth_text": "150TB/s (SRAM)"},
        "H100 PCIe": {"compute_text": "800TFLOP/s FP16", "bandwidth_text": "2TB/s"},
        "ours@1GHz": {"compute_text": "7.168TFLOP/s FP16", "bandwidth_text": "896GB/s"},
    }
    prompts = [
        ("Short", "100", {"Groq LPU": "52ms", "H100 PCIe": "190ms", "ours@1GHz": "348.84ms"}, "3.7x"),
        ("Medium", "1000", {"Groq LPU": "80ms", "H100 PCIe": "280ms", "ours@1GHz": "3.3s"}, "3.5x"),
        ("Long", "8000", {"Groq LPU": "210ms", "H100 PCIe": "820ms", "ours@1GHz": "53.7s"}, "3.9x"),
        ("Very long", "32000", {"Groq LPU": "680ms", "H100 PCIe": "3400ms", "ours@1GHz": "10.1min"}, "5.0x"),
    ]

    rows: list[dict[str, str]] = []
    for workload_name, input_tokens, values, speed_gain in prompts:
        for hardware, latency_text in values.items():
            rows.append(
                {
                    "source_name": "Current PPT slide",
                    "source_section": "TTFT-llama 8B (batchsize=1)",
                    "source_url": "",
                    "record_type": "static_comparison",
                    "hardware": hardware,
                    "hardware_count": "",
                    "model_name": "Llama 8B",
                    "precision_or_quantization": "FP8" if hardware == "Groq LPU" else "FP16",
                    "engine": "",
                    "serving_script": "",
                    "scenario_type": "slide_summary",
                    "workload_name": workload_name,
                    "dataset_name": "",
                    "batch_size": "1",
                    "input_tokens": input_tokens,
                    "output_tokens": "",
                    "num_prompts": "",
                    "max_request_concurrency": "",
                    "peak_concurrent_requests": "",
                    "compute_text": specs[hardware]["compute_text"],
                    "bandwidth_text": specs[hardware]["bandwidth_text"],
                    "request_throughput_req_s": "",
                    "output_token_throughput_tok_s": "",
                    "peak_output_token_throughput_tok_s": "",
                    "total_token_throughput_tok_s": "",
                    "mean_ttft_ms": _norm_latency_ms(latency_text),
                    "median_ttft_ms": "",
                    "p99_ttft_ms": "",
                    "mean_tpot_ms": "",
                    "median_tpot_ms": "",
                    "p99_tpot_ms": "",
                    "mean_itl_ms": "",
                    "median_itl_ms": "",
                    "p99_itl_ms": "",
                    "comparison_value": speed_gain if hardware == "Groq LPU" else "",
                    "notes": "Extracted from current PPT slide. comparison_value is Groq speed gain vs H100 from the slide.",
                }
            )
    return rows


def _gpustack_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for meta in GPUSTACK_PAGES:
        text = _fetch_article_text(meta["url"])
        base = _base_gpustack_row(meta)

        choose_start = text.find("Choosing the Inference Engine")
        if choose_start != -1:
            choose_section = text[choose_start:]
            choose_block = _extract_first_benchmark_block(choose_section)
            if choose_block:
                choose_row = dict(base)
                choose_row.update(_metrics_from_block(choose_block))
                choose_row["source_section"] = "Experiment Results / Choosing the Inference Engine"
                choose_row["workload_name"] = "ShareGPT high concurrency"
                choose_row["dataset_name"] = "ShareGPT"
                choose_row["notes"] = (
                    "Choosing the Inference Engine result for vLLM on the GPUStack 2.0 page."
                )
                rows.append(choose_row)

        baseline_start = text.find("Baseline benchmark results")
        if baseline_start != -1:
            baseline_end = text.find("Optimized serving script", baseline_start)
            if baseline_end == -1:
                baseline_end = len(text)
            baseline_section = text[baseline_start:baseline_end]

            matches = list(
                re.finditer(
                    r"#\s*([^#=]+?)\s*=+\s*Serving Benchmark Result\s*=+",
                    baseline_section,
                )
            )
            for idx, match in enumerate(matches):
                block_start = match.start()
                block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(baseline_section)
                block = baseline_section[block_start:block_end]
                workload_name = match.group(1).strip()
                dataset_name, input_tokens, output_tokens = _infer_workload_tokens(workload_name)

                baseline_row = dict(base)
                baseline_row.update(_metrics_from_block(block))
                baseline_row["source_section"] = "Other Benchmark Cases / Baseline benchmark results"
                baseline_row["workload_name"] = workload_name
                baseline_row["dataset_name"] = dataset_name
                baseline_row["input_tokens"] = input_tokens
                baseline_row["output_tokens"] = output_tokens
                baseline_row["notes"] = (
                    f"Baseline vLLM result for '{workload_name}' on the GPUStack 2.0 page."
                )
                rows.append(baseline_row)

    return rows


def main() -> None:
    rows = _ppt_rows() + _gpustack_rows()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
