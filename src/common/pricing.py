"""Centralized pricing resolution for LLM and SLM calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRICING_CONFIG_PATH = PROJECT_ROOT / "config" / "pricing.json"


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        return parsed
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def load_pricing_config() -> Dict[str, Any]:
    """Load pricing config from JSON file with safe defaults."""
    custom_path = os.getenv("PRICING_CONFIG_PATH", "").strip()
    pricing_path = Path(custom_path) if custom_path else DEFAULT_PRICING_CONFIG_PATH

    if not pricing_path.exists():
        return {}

    try:
        with open(pricing_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_openai_estimated_cost(
    *,
    model_name: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    pricing_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    rates_by_model = pricing_config.get("openai_token_pricing_per_1m", {})
    if not isinstance(rates_by_model, dict):
        return None

    model_key_candidates = [model_name, model_id]
    rates = None
    for candidate in model_key_candidates:
        if candidate in rates_by_model:
            rates = rates_by_model[candidate]
            break

    if not isinstance(rates, dict):
        return None

    input_rate = _to_float(rates.get("input"))
    output_rate = _to_float(rates.get("output"))

    if input_rate is None or output_rate is None:
        return None

    input_cost = (max(input_tokens, 0) / 1_000_000.0) * input_rate
    output_cost = (max(output_tokens, 0) / 1_000_000.0) * output_rate
    total_cost = input_cost + output_cost
    return {
        "total_cost": total_cost,
        "pricing_context": {
            "pricing_method": "token_based",
            "input_rate_per_1m": input_rate,
            "output_rate_per_1m": output_rate,
            "input_tokens": int(max(input_tokens, 0)),
            "output_tokens": int(max(output_tokens, 0)),
            "input_cost": round(input_cost, 10),
            "output_cost": round(output_cost, 10),
        },
    }


def _resolve_hf_estimated_cost(
    *,
    model_name: str,
    model_id: str,
    execution_time_seconds: Optional[float],
    pricing_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    hf_config = pricing_config.get("huggingface_endpoints", {})
    if not isinstance(hf_config, dict):
        return None

    endpoint_cfg = None
    for candidate in (model_name, model_id):
        if candidate in hf_config:
            endpoint_cfg = hf_config[candidate]
            break

    if not isinstance(endpoint_cfg, dict):
        return None

    hourly_rate = _to_float(endpoint_cfg.get("hourly_rate_usd"))
    replicas = _to_int(endpoint_cfg.get("replicas", 1))
    allocation_mode = str(endpoint_cfg.get("allocation_mode", "runtime_proportional")).strip().lower()

    if hourly_rate is None or hourly_rate < 0 or replicas <= 0:
        return None

    base_context = {
        "pricing_method": "endpoint_hourly",
        "cloud_provider": endpoint_cfg.get("cloud_provider"),
        "instance_family": endpoint_cfg.get("instance_family"),
        "instance_size": endpoint_cfg.get("instance_size"),
        "accelerator": endpoint_cfg.get("accelerator"),
        "gpu_count": _to_int(endpoint_cfg.get("gpu_count", 0)),
        "vram_gb": _to_float(endpoint_cfg.get("vram_gb")),
        "hourly_rate_usd_per_replica": hourly_rate,
        "replicas": replicas,
        "allocation_mode": allocation_mode,
        "pricing_source_url": endpoint_cfg.get("pricing_source_url"),
        "pricing_updated_at": endpoint_cfg.get("pricing_updated_at"),
    }

    if allocation_mode == "amortized_window":
        active_hours = _to_float(endpoint_cfg.get("active_hours_window"))
        processed_queries = _to_int(endpoint_cfg.get("processed_queries_window"))
        if active_hours is None or active_hours <= 0 or processed_queries <= 0:
            return None
        total_cost = (hourly_rate * replicas * active_hours) / processed_queries
        base_context.update(
            {
                "active_hours_window": active_hours,
                "processed_queries_window": processed_queries,
            }
        )
        return {
            "total_cost": total_cost,
            "pricing_context": base_context,
        }

    run_seconds = _to_float(execution_time_seconds)
    if run_seconds is None or run_seconds <= 0:
        return None

    # Hourly-based pricing: cost = hourly_rate × replicas × (execution_time / 3600)
    # No rounding or granularity applied; billing is continuous by the hour.
    total_cost = hourly_rate * replicas * (run_seconds / 3600.0)
    base_context.update(
        {
            "execution_time_seconds": run_seconds,
        }
    )
    return {
        "total_cost": total_cost,
        "pricing_context": base_context,
    }


def resolve_total_cost(
    *,
    provider: str,
    model_name: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    provider_reported_cost: Optional[float],
    provider_cost_source: str,
    execution_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve total cost with callback-first and provider-specific fallbacks."""
    parsed_provider_cost = _to_float(provider_reported_cost)

    if parsed_provider_cost is not None and parsed_provider_cost > 0:
        return {
            "total_cost": round(parsed_provider_cost, 10),
            "cost_source": provider_cost_source or "provider_reported",
            "pricing_context": {
                "pricing_method": "provider_reported",
            },
        }

    pricing_config = load_pricing_config()

    provider_key = str(provider or "").strip().lower()
    if provider_key == "openai":
        estimated = _resolve_openai_estimated_cost(
            model_name=model_name,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            pricing_config=pricing_config,
        )
        if estimated is not None:
            return {
                "total_cost": round(max(float(estimated["total_cost"]), 0.0), 10),
                "cost_source": "estimated_openai_token_pricing",
                "pricing_context": estimated.get("pricing_context", {}),
            }

    if provider_key == "huggingface":
        estimated = _resolve_hf_estimated_cost(
            model_name=model_name,
            model_id=model_id,
            execution_time_seconds=execution_time_seconds,
            pricing_config=pricing_config,
        )
        if estimated is not None:
            return {
                "total_cost": round(max(float(estimated["total_cost"]), 0.0), 10),
                "cost_source": "estimated_hf_endpoint_pricing",
                "pricing_context": estimated.get("pricing_context", {}),
            }

    return {
        "total_cost": 0.0,
        "cost_source": "missing",
        "pricing_context": {
            "pricing_method": "missing",
        },
    }


def get_pricing_config_summary() -> Dict[str, Any]:
    """
    Generate a summary of pricing configuration for documentation/traceability.
    
    Returns a dict with OpenAI and HuggingFace endpoint configurations,
    referenced once at the start of evaluation results for auditability.
    """
    pricing_config = load_pricing_config()
    
    summary = {
        "openai_models": {},
        "huggingface_endpoints": {},
    }
    
    # OpenAI models
    openai_rates = pricing_config.get("openai_token_pricing_per_1m", {})
    if isinstance(openai_rates, dict):
        for model_key, rates in openai_rates.items():
            if isinstance(rates, dict):
                summary["openai_models"][model_key] = {
                    "input_rate_per_1m": _to_float(rates.get("input")),
                    "output_rate_per_1m": _to_float(rates.get("output")),
                    "pricing_source_url": rates.get("pricing_source_url"),
                    "pricing_updated_at": rates.get("pricing_updated_at"),
                }
    
    # HuggingFace endpoints
    hf_endpoints = pricing_config.get("huggingface_endpoints", {})
    if isinstance(hf_endpoints, dict):
        for endpoint_key, cfg in hf_endpoints.items():
            if isinstance(cfg, dict):
                summary["huggingface_endpoints"][endpoint_key] = {
                    "cloud_provider": cfg.get("cloud_provider"),
                    "instance_family": cfg.get("instance_family"),
                    "instance_size": cfg.get("instance_size"),
                    "accelerator": cfg.get("accelerator"),
                    "gpu_count": _to_int(cfg.get("gpu_count", 0)),
                    "vram_gb": _to_float(cfg.get("vram_gb")),
                    "hourly_rate_usd": _to_float(cfg.get("hourly_rate_usd")),
                    "replicas": _to_int(cfg.get("replicas", 1)),
                    "allocation_mode": cfg.get("allocation_mode", "runtime_proportional"),
                    "pricing_source_url": cfg.get("pricing_source_url"),
                    "pricing_updated_at": cfg.get("pricing_updated_at"),
                }
    
    return summary
