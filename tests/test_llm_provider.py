import pytest
from langchain_core.messages import AIMessage

from agent_freeform import extract_llm_telemetry, get_llm, get_llm_config


def test_get_llm_config_defaults(monkeypatch):
    """When no environment variables are set, api_key is None and defaults apply."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    config = get_llm_config()
    assert config["api_key"] is None
    assert config["base_url"] == "https://openrouter.ai/api/v1"
    assert config["model"] == "google/gemini-3.1-flash-lite"
    assert config["fallback_model"] == "google/gemini-2.5-flash"
    assert config["temperature"] == 0.0


def test_get_llm_config_openrouter_compat(monkeypatch):
    """Backward compatibility: setting OPENROUTER_API_KEY and OPENROUTER_MODEL resolves correctly."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b")

    config = get_llm_config()
    assert config["api_key"] == "sk-or-test-key"
    assert config["base_url"] == "https://openrouter.ai/api/v1"
    assert config["model"] == "meta-llama/llama-3.3-70b"


def test_get_llm_config_gemini_direct(monkeypatch):
    """Setting GEMINI_API_KEY automatically configures Google AI Studio OpenAI endpoint and Gemini defaults."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestGeminiKey")

    config = get_llm_config()
    assert config["api_key"] == "AIzaSyTestGeminiKey"
    assert config["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert config["model"] == "gemini-2.5-flash"
    assert config["fallback_model"] == "gemini-2.5-pro"


def test_get_llm_config_custom_provider(monkeypatch):
    """Full custom provider configuration via LLM_* variables."""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_API_KEY", "local-token")
    monkeypatch.setenv("LLM_MODEL", "custom-model-v1")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "custom-model-backup")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")

    config = get_llm_config()
    assert config["api_key"] == "local-token"
    assert config["base_url"] == "http://localhost:8000/v1"
    assert config["model"] == "custom-model-v1"
    assert config["fallback_model"] == "custom-model-backup"
    assert config["temperature"] == 0.7


def test_get_llm_factory_missing_key(monkeypatch):
    """Calling get_llm without any API key raises a clear ValueError."""
    with pytest.raises(ValueError, match="No LLM API key configured"):
        get_llm()


def test_get_llm_factory_instantiation(monkeypatch):
    """get_llm successfully creates a ChatOpenAI client with proper endpoint, model, and fallback."""
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "gpt-4o")

    # Primary model
    llm = get_llm()
    assert llm.model_name == "gpt-4o-mini"
    assert llm.openai_api_base == "https://api.openai.com/v1"
    assert llm.openai_api_key.get_secret_value() == "test-key-123"

    # Fallback model
    fallback_llm = get_llm(fallback=True)
    assert fallback_llm.model_name == "gpt-4o"

    # Explicit model override
    override_llm = get_llm(model="custom-override-model")
    assert override_llm.model_name == "custom-override-model"


def test_extract_llm_telemetry():
    """extract_llm_telemetry parses usage_metadata and response_metadata correctly."""
    msg = AIMessage(
        content="Response",
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 45,
            "total_tokens": 165,
            "input_token_details": {"cached_tokens": 20},
        },
        response_metadata={
            "token_usage": {
                "cost": 0.00042,
                "cost_details": {"prompt_cost": 0.0002, "completion_cost": 0.00022},
            }
        },
    )

    telem = extract_llm_telemetry(msg)
    assert telem["prompt_tokens"] == 120
    assert telem["completion_tokens"] == 45
    assert telem["total_tokens"] == 165
    assert telem["prompt_tokens_details"] == {"cached_tokens": 20}
    assert telem["cost"] == 0.00042
    assert telem["cost_details"]["prompt_cost"] == 0.0002


def test_extract_llm_telemetry_empty_message():
    """extract_llm_telemetry handles messages with no usage metadata gracefully."""
    msg = AIMessage(content="Empty usage")
    telem = extract_llm_telemetry(msg)
    assert telem["prompt_tokens"] == 0
    assert telem["completion_tokens"] == 0
    assert telem["total_tokens"] == 0
    assert telem["cost"] == 0.0
