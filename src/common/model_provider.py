"""
Model provider abstraction for LLMs and SLMs.

Supports both OpenAI models (gpt-5, gpt-4.1) and HuggingFace models (mediphi, medgemma)
deployed via TGI or HF Inference Endpoints (which expose OpenAI-compatible APIs).
"""

import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel


def load_dotenv_if_needed():
    """Load .env if not already loaded."""
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        ENV_PATH = PROJECT_ROOT / ".env"
        
        if not os.getenv("OPENAI_API_KEY"):
            load_dotenv(dotenv_path=ENV_PATH)
    except ImportError:
        pass


@dataclass
class ModelConfig:
    """Configuration for a language model."""
    
    name: str
    """Short name/key for the model (e.g., 'gpt-4.1', 'mediphi')."""
    
    model_id: str
    """Full model identifier (e.g., 'gpt-4.1', 'microsoft/MediPhi')."""
    
    provider: Literal["openai", "huggingface"]
    """Model provider type."""
    
    endpoint_url_env: Optional[str] = None
    """Environment variable name for HuggingFace endpoint URL (e.g., 'MEDIPHI_ENDPOINT_URL').
    Only used for HuggingFace models. Should point to base endpoint URL without /v1."""
    
    temperature: float = 0.0
    """Sampling temperature for the model."""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.provider == "huggingface" and not self.endpoint_url_env:
            raise ValueError(
                f"HuggingFace model '{self.name}' requires 'endpoint_url_env' to be set."
            )


def get_model_identity(model_name: Optional[str] = None, llm: Optional[BaseChatModel] = None) -> Dict[str, str]:
    """
    Resolve provider/model identity from registry hints and runtime LLM instance.

    This keeps metadata and pricing lookups consistent across OpenAI and
    OpenAI-compatible providers like HuggingFace inference endpoints.
    """
    runtime_model_name = ""
    if llm is not None:
        runtime_model_name = str(getattr(llm, "model_name", "") or "").strip()

    requested_name = str(model_name or "").strip()
    candidates = [c for c in (requested_name, runtime_model_name) if c]

    for candidate in candidates:
        if candidate in MODELS_REGISTRY:
            config = MODELS_REGISTRY[candidate]
            return {
                "provider": config.provider,
                "model_id": config.model_id,
                "model_name": config.name,
            }

    for config in MODELS_REGISTRY.values():
        if runtime_model_name and runtime_model_name == config.model_id:
            return {
                "provider": config.provider,
                "model_id": config.model_id,
                "model_name": config.name,
            }

    inferred_provider = "unknown"
    if runtime_model_name and "/" in runtime_model_name:
        inferred_provider = "huggingface"
    elif runtime_model_name:
        inferred_provider = "openai"

    final_model_id = runtime_model_name or requested_name or "unknown"
    return {
        "provider": inferred_provider,
        "model_id": final_model_id,
        "model_name": requested_name or runtime_model_name or "unknown",
    }


def create_llm(config: ModelConfig) -> BaseChatModel:
    """
    Factory function to create a language model based on configuration.
    
    OpenAI models return standard ChatOpenAI instances.
    HuggingFace models return ChatOpenAI instances pointing to a TGI/HF Inference Endpoint
    that exposes the OpenAI-compatible API.
    
    Args:
        config: ModelConfig instance.
        
    Returns:
        BaseChatModel: Configured language model.
        
    Raises:
        ValueError: If endpoint URL environment variable is not set for HF models.
    """
    load_dotenv_if_needed()
    
    if config.provider == "openai":
        return ChatOpenAI(
            model_name=config.model_id,
            temperature=config.temperature
        )
    
    elif config.provider == "huggingface":
        endpoint_url = os.getenv(config.endpoint_url_env)
        if not endpoint_url:
            raise ValueError(
                f"Environment variable '{config.endpoint_url_env}' not set. "
                f"Required to serve HuggingFace model '{config.name}'."
            )
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable not set. "
                "Required for authentication with HuggingFace endpoints."
            )
        
        return ChatOpenAI(
            base_url=f"{endpoint_url}/v1",
            api_key=hf_token,
            model_name=config.model_id,
            temperature=config.temperature,
            max_tokens=512,  # Required: HF TGI models loop without a stop limit
        )
    
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


# Model registry with canonical configurations
MODELS_REGISTRY: Dict[str, ModelConfig] = {
    "gpt-5": ModelConfig(
        name="gpt-5",
        model_id="gpt-5",
        provider="openai",
        temperature=0.0
    ),
    "gpt-4.1": ModelConfig(
        name="gpt-4.1",
        model_id="gpt-4.1",
        provider="openai",
        temperature=0.0
    ),
    "mediphi": ModelConfig(
        name="mediphi",
        model_id="microsoft/MediPhi",
        provider="huggingface",
        endpoint_url_env="MEDIPHI_ENDPOINT_URL",
        temperature=0.0
    ),
    "medgemma": ModelConfig(
        name="medgemma",
        model_id="google/medgemma-4b-it",
        provider="huggingface",
        endpoint_url_env="MEDGEMMA_ENDPOINT_URL",
        temperature=0.0
    ),
}


__all__ = [
    "ModelConfig",
    "create_llm",
    "get_model_identity",
    "MODELS_REGISTRY",
    "load_dotenv_if_needed",
]
