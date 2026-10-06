import os
import pytest


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    """
    Ensure offline unit and integration tests run deterministically without
    unexpectedly invoking live remote LLM endpoints due to ambient developer environment keys.
    Tests testing live or configured keys can set them explicitly via monkeypatch.setenv.
    """
    for key in ["LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
