"""Recalculate HuggingFace (SLM) total_cost fields inside evaluation result JSON files.

Formula used (deterministic):
    total_cost = hourly_rate_usd * replicas * (execution_time_seconds / 3600)

This script updates:
- question_by_question[*].rag_results[rag][model].performance.total_cost
- summary[rag][model].performance.total_cost
- summary[rag][model].performance.average_cost_per_question

Only HuggingFace models are changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def _load_pricing(pricing_path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(pricing_path.read_text(encoding="utf-8"))
    hf = data.get("huggingface_endpoints", {})

    return {
        "microsoft/MediPhi-Instruct": {
            "hourly_rate_usd": float(hf.get("mediphi", {}).get("hourly_rate_usd", 0.0)),
            "replicas": float(hf.get("mediphi", {}).get("replicas", 1)),
        },
        "google/medgemma-1.5-4b-it": {
            "hourly_rate_usd": float(hf.get("medgemma", {}).get("hourly_rate_usd", 0.0)),
            "replicas": float(hf.get("medgemma", {}).get("replicas", 1)),
        },
    }


def _calc_cost(execution_time_seconds: float, hourly_rate_usd: float, replicas: float) -> float:
    return round(hourly_rate_usd * replicas * (execution_time_seconds / 3600.0), 10)


def _recalculate_question_level_costs(
    payload: dict,
    hf_model_rates: Dict[str, Dict[str, float]],
) -> Tuple[
    int,
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], int],
    Dict[str, float],
    Dict[str, int],
    Dict[str, float],
    Dict[str, int],
]:
    updated = 0
    cost_sums_by_rag_model: Dict[Tuple[str, str], float] = {}
    counts_by_rag_model: Dict[Tuple[str, str], int] = {}
    cost_sums_by_rag: Dict[str, float] = {}
    counts_by_rag: Dict[str, int] = {}
    cost_sums_by_model: Dict[str, float] = {}
    counts_by_model: Dict[str, int] = {}

    model_hint: Optional[str] = payload.get("metadata", {}).get("model_used")
    if model_hint not in hf_model_rates:
        model_hint = None

    def _update_perf(perf: dict, rag_name: Optional[str], model_name: str) -> None:
        nonlocal updated
        if "execution_time" not in perf:
            return

        rates = hf_model_rates[model_name]
        new_cost = _calc_cost(
            execution_time_seconds=float(perf.get("execution_time", 0.0)),
            hourly_rate_usd=float(rates["hourly_rate_usd"]),
            replicas=float(rates["replicas"]),
        )

        old_cost = float(perf.get("total_cost", 0.0))
        if abs(old_cost - new_cost) > 1e-12:
            perf["total_cost"] = new_cost
            updated += 1

        if rag_name is not None:
            rag_model_key = (rag_name, model_name)
            cost_sums_by_rag_model[rag_model_key] = cost_sums_by_rag_model.get(rag_model_key, 0.0) + new_cost
            counts_by_rag_model[rag_model_key] = counts_by_rag_model.get(rag_model_key, 0) + 1
            cost_sums_by_rag[rag_name] = cost_sums_by_rag.get(rag_name, 0.0) + new_cost
            counts_by_rag[rag_name] = counts_by_rag.get(rag_name, 0) + 1

        cost_sums_by_model[model_name] = cost_sums_by_model.get(model_name, 0.0) + new_cost
        counts_by_model[model_name] = counts_by_model.get(model_name, 0) + 1

    for row in payload.get("question_by_question", []):
        rag_results = row.get("rag_results", {})
        if not isinstance(rag_results, dict):
            continue

        for key, value in rag_results.items():
            if not isinstance(value, dict):
                continue

            # Schema A: rag_results[model_name] -> result (multi-model single-rag files)
            if key in hf_model_rates and "performance" in value:
                perf = value.get("performance") or {}
                _update_perf(perf, None, key)
                continue

            # Schema B: rag_results[rag_name] -> result (single-model single-rag files)
            if "performance" in value and model_hint is not None:
                perf = value.get("performance") or {}
                _update_perf(perf, key, model_hint)
                continue

            # Schema C: rag_results[rag_name][model_name] -> result (comprehensive files)
            rag_name = key
            for model_name, model_result in value.items():
                if model_name not in hf_model_rates:
                    continue
                if not isinstance(model_result, dict):
                    continue
                perf = model_result.get("performance") or {}
                _update_perf(perf, rag_name, model_name)

    return (
        updated,
        cost_sums_by_rag_model,
        counts_by_rag_model,
        cost_sums_by_rag,
        counts_by_rag,
        cost_sums_by_model,
        counts_by_model,
    )


def _recalculate_summary_costs(
    payload: dict,
    cost_sums_by_rag_model: Dict[Tuple[str, str], float],
    counts_by_rag_model: Dict[Tuple[str, str], int],
    cost_sums_by_rag: Dict[str, float],
    counts_by_rag: Dict[str, int],
    cost_sums_by_model: Dict[str, float],
    counts_by_model: Dict[str, int],
    hf_model_rates: Dict[str, Dict[str, float]],
) -> int:
    updated = 0

    model_hint: Optional[str] = payload.get("metadata", {}).get("model_used")
    if model_hint not in hf_model_rates:
        model_hint = None

    summary = payload.get("summary", {})
    for key, value in summary.items():
        if not isinstance(value, dict):
            continue

        # Schema A/B summary[key] has direct performance (single-rag or multi-model files)
        if "performance" in value:
            perf = value.get("performance") or {}

            aggregate_sum: Optional[float] = None
            aggregate_count: Optional[int] = None

            # Multi-model summary: key is model name
            if key in cost_sums_by_model:
                aggregate_sum = cost_sums_by_model[key]
                aggregate_count = counts_by_model[key]
            # Single-model summary: key is rag name
            elif key in cost_sums_by_rag and model_hint is not None:
                aggregate_sum = cost_sums_by_rag[key]
                aggregate_count = counts_by_rag[key]

            if aggregate_sum is None or aggregate_count is None:
                continue

            total_cost = round(aggregate_sum, 6)
            avg_cost = round(aggregate_sum / max(aggregate_count, 1), 6)

            old_total = float(perf.get("total_cost", 0.0))
            old_avg = float(perf.get("average_cost_per_question", 0.0))

            if abs(old_total - total_cost) > 1e-12:
                perf["total_cost"] = total_cost
                updated += 1
            if abs(old_avg - avg_cost) > 1e-12:
                perf["average_cost_per_question"] = avg_cost
                updated += 1

            continue

        # Schema C summary[rag][model]
        rag_name = key
        for model_name, model_data in value.items():
            if not isinstance(model_data, dict):
                continue

            rag_model_key = (rag_name, model_name)
            if rag_model_key not in cost_sums_by_rag_model:
                continue

            perf = model_data.get("performance") or {}
            total_cost = round(cost_sums_by_rag_model[rag_model_key], 6)
            avg_cost = round(
                cost_sums_by_rag_model[rag_model_key] / max(counts_by_rag_model[rag_model_key], 1),
                6,
            )

            old_total = float(perf.get("total_cost", 0.0))
            old_avg = float(perf.get("average_cost_per_question", 0.0))

            if abs(old_total - total_cost) > 1e-12:
                perf["total_cost"] = total_cost
                updated += 1
            if abs(old_avg - avg_cost) > 1e-12:
                perf["average_cost_per_question"] = avg_cost
                updated += 1

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalculate HF total_cost fields in a results JSON file")
    parser.add_argument("results_file", type=Path, help="Path to a results JSON file")
    parser.add_argument(
        "--pricing-file",
        type=Path,
        default=Path("config/pricing.json"),
        help="Path to pricing config JSON",
    )
    args = parser.parse_args()

    payload = json.loads(args.results_file.read_text(encoding="utf-8"))
    hf_model_rates = _load_pricing(args.pricing_file)

    (
        q_updated,
        cost_sums_by_rag_model,
        counts_by_rag_model,
        cost_sums_by_rag,
        counts_by_rag,
        cost_sums_by_model,
        counts_by_model,
    ) = _recalculate_question_level_costs(payload, hf_model_rates)
    s_updated = _recalculate_summary_costs(
        payload,
        cost_sums_by_rag_model,
        counts_by_rag_model,
        cost_sums_by_rag,
        counts_by_rag,
        cost_sums_by_model,
        counts_by_model,
        hf_model_rates,
    )

    args.results_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Updated question-level entries: {q_updated}")
    print(f"Updated summary fields: {s_updated}")
    print(f"File updated: {args.results_file}")


if __name__ == "__main__":
    main()
