"""Provider-agnostic token usage extraction helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _to_int(value: Any) -> int:
    """Safely coerce values to non-negative integers."""
    try:
        if value is None:
            return 0
        parsed = int(value)
        return max(parsed, 0)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> Optional[float]:
    """Safely coerce values to non-negative floats."""
    try:
        if value is None:
            return None
        parsed = float(value)
        return max(parsed, 0.0)
    except (TypeError, ValueError):
        return None


def _extract_from_mapping(data: Any) -> Dict[str, int]:
    """Read usage fields from provider metadata mappings."""
    if not isinstance(data, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = _to_int(
        data.get("input_tokens")
        or data.get("prompt_tokens")
        or data.get("input")
    )
    output_tokens = _to_int(
        data.get("output_tokens")
        or data.get("completion_tokens")
        or data.get("output")
    )
    total_tokens = _to_int(data.get("total_tokens") or data.get("total"))

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def extract_usage_from_ai_message(message: Any) -> Dict[str, int | str]:
    """
    Extract token usage in a provider-agnostic way.

    Priority:
    1) `message.usage_metadata` (LangChain standard)
    2) `message.response_metadata["token_usage"]`
    3) `message.response_metadata["usage"]`
    4) missing -> zeros
    """
    usage_metadata = getattr(message, "usage_metadata", None)
    extracted = _extract_from_mapping(usage_metadata)
    if extracted["total_tokens"] > 0:
        extracted["usage_source"] = "usage_metadata"
        return extracted

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("token_usage", "usage"):
            extracted = _extract_from_mapping(response_metadata.get(key))
            if extracted["total_tokens"] > 0:
                extracted["usage_source"] = "response_metadata"
                return extracted

    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "usage_source": "missing",
    }


def extract_cost_from_ai_message(message: Any) -> Dict[str, Optional[float] | str]:
    """
    Extract provider-reported cost when available in response metadata.

    The function intentionally does not estimate cost from a local price table.
    If the provider does not return billing metadata, cost is reported as missing.
    """
    response_metadata = getattr(message, "response_metadata", None)
    if not isinstance(response_metadata, dict):
        return {"total_cost": None, "cost_source": "missing"}

    direct_candidates = (
        response_metadata.get("total_cost"),
        response_metadata.get("cost"),
        response_metadata.get("usd_cost"),
    )
    for candidate in direct_candidates:
        parsed = _to_float(candidate)
        if parsed is not None:
            return {"total_cost": parsed, "cost_source": "response_metadata"}

    usage = response_metadata.get("usage")
    if isinstance(usage, dict):
        for key in ("total_cost", "cost", "usd_cost"):
            parsed = _to_float(usage.get(key))
            if parsed is not None:
                return {"total_cost": parsed, "cost_source": "response_metadata.usage"}

    billing = response_metadata.get("billing")
    if isinstance(billing, dict):
        for key in ("total_cost", "cost", "usd_cost"):
            parsed = _to_float(billing.get(key))
            if parsed is not None:
                return {"total_cost": parsed, "cost_source": "response_metadata.billing"}

    return {"total_cost": None, "cost_source": "missing"}
