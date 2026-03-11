from src.common.pricing import resolve_total_cost, get_pricing_config_summary


def test_resolve_total_cost_prefers_provider_reported_value():
    resolved = resolve_total_cost(
        provider="openai",
        model_name="gpt-5",
        model_id="gpt-5",
        input_tokens=1000,
        output_tokens=500,
        provider_reported_cost=0.123456,
        provider_cost_source="response_metadata",
        execution_time_seconds=2.0,
    )

    assert resolved["total_cost"] == 0.123456
    assert resolved["cost_source"] == "response_metadata"


def test_resolve_total_cost_estimates_openai_from_tokens():
    resolved = resolve_total_cost(
        provider="openai",
        model_name="gpt-5",
        model_id="gpt-5",
        input_tokens=1000,
        output_tokens=2000,
        provider_reported_cost=0.0,
        provider_cost_source="missing",
        execution_time_seconds=3.0,
    )

    assert resolved["total_cost"] > 0
    assert resolved["cost_source"] == "estimated_openai_token_pricing"


def test_resolve_total_cost_estimates_hf_from_runtime():
    resolved = resolve_total_cost(
        provider="huggingface",
        model_name="mediphi",
        model_id="microsoft/MediPhi",
        input_tokens=0,
        output_tokens=0,
        provider_reported_cost=0.0,
        provider_cost_source="missing",
        execution_time_seconds=61.0,
    )

    assert resolved["total_cost"] > 0
    assert resolved["cost_source"] == "estimated_hf_endpoint_pricing"
    assert resolved["pricing_context"]["pricing_method"] == "endpoint_hourly"
    assert resolved["pricing_context"]["cloud_provider"] == "aws"
    assert resolved["pricing_context"]["instance_family"] == "nvidia-l40s"
    assert resolved["pricing_context"]["gpu_count"] == 1


def test_get_pricing_config_summary():
    summary = get_pricing_config_summary()
    
    assert "openai_models" in summary
    assert "huggingface_endpoints" in summary
    
    # Check OpenAI models
    assert "gpt-5" in summary["openai_models"]
    assert summary["openai_models"]["gpt-5"]["input_rate_per_1m"] == 2.5
    assert summary["openai_models"]["gpt-5"]["output_rate_per_1m"] == 15.0
    assert summary["openai_models"]["gpt-5"]["pricing_source_url"] == "https://developers.openai.com/api/pricing/"
    assert summary["openai_models"]["gpt-5"]["pricing_updated_at"] == "2026-03-10"
    
    # Check HuggingFace endpoints
    assert "mediphi" in summary["huggingface_endpoints"]
    assert summary["huggingface_endpoints"]["mediphi"]["cloud_provider"] == "aws"
    assert summary["huggingface_endpoints"]["mediphi"]["hourly_rate_usd"] == 1.8
