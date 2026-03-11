# Pricing Methodology for Benchmark Experiments

This document defines how `total_cost` is computed when provider callbacks do not include USD billing metadata.

## 1. LLM APIs (OpenAI, Gemini-style pricing)

For hosted API models billed by token, use:

`request_cost_usd = (input_tokens / 1_000_000) * input_price_per_1m + (output_tokens / 1_000_000) * output_price_per_1m`

Why this is defensible:
- OpenAI and Gemini pricing pages publish per-token (or per-million-token) input/output rates.
- This is the same billing basis used by model APIs.
- It is reproducible and model-specific.

Implementation notes:
- Use provider-reported callback cost first (if available and > 0).
- If missing, estimate from token counts + configured per-model rates.
- Keep rates versioned in `config/pricing.json` and cite source date in the paper.

## 2. SLM on Hugging Face Inference Endpoints

Dedicated endpoints are billed by infrastructure runtime (hourly rate per running replica), not by token.
Hugging Face displays this as `USD/hour per running replica`, so total runtime cost scales linearly with `replicas`.

### Calculation for per-request experiment accounting

`request_cost_usd = endpoint_hourly_rate_usd * replicas * (execution_time_seconds / 3600)`

This allocates the continuous hourly cost of running replicas to each request based on its actual execution time.

In this project, each query now stores `metadata.pricing_context` with the exact assumptions used for reproducibility:
- cloud provider
- instance family and size
- accelerator type
- GPU count and VRAM
- hourly rate per replica
- replicas currently allocated
- allocation mode
- pricing source URL and update date

### Alternative for accounting-period amortization

`request_cost_usd = (endpoint_hourly_rate_usd * replicas * active_hours_window) / processed_queries_window`

Use this mode when you want to allocate both active and idle endpoint time across all requests in a defined period.

## 3. Parameters to report in paper

For each model/endpoint, report:
- Provider and model id.
- Pricing source URL and retrieval date.
- Input/output token rates (LLMs) or endpoint hourly rate (SLMs).
- Endpoint hardware profile (instance type, GPU type, VRAM, replicas).
- Allocation mode (`runtime_proportional` or `amortized_window`).

## 4. Sources used

- OpenAI pricing (token-based API billing): `https://openai.com/api/pricing/`
- Google Gemini pricing (token-based API billing): `https://ai.google.dev/gemini-api/docs/pricing`
- Hugging Face Inference Endpoints pricing (hourly runtime billing, billed per minute): `https://huggingface.co/docs/inference-endpoints/pricing`
