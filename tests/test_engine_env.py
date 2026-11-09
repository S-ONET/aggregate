"""The default router: generic env names, no endpoint the environment does not name, 0.1.x names as fallbacks."""
import pytest

from aggregate import engine

VARS = ["LLM_API_KEY", "LLM_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "CLAUDE_API_KEY", "CLAUDE_BASE_URL",
        "ONET_CLAUDE_API_KEY", "ONET_CLAUDE_BASE_URL", "DIYI_API_KEY", "LITELLM_OPENAI_API_KEY", "LITELLM_OPENAI_BASE_URL"]


@pytest.fixture
def env(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    engine._warned.clear()
    return monkeypatch


def _url(c):
    return str(c.base_url).rstrip("/")


def test_openai_route_defaults_to_openai(env):
    env.setenv("OPENAI_API_KEY", "k")
    assert _url(engine._default_router("gpt-5.4")) == "https://api.openai.com/v1"
    env.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    assert _url(engine._default_router("gpt-5.4")) == "https://proxy.example/v1"


def test_claude_needs_a_named_gateway(env):
    env.setenv("CLAUDE_API_KEY", "k")
    with pytest.raises(RuntimeError, match="CLAUDE_BASE_URL"):
        engine._default_router("claude-sonnet-4-6")
    env.setenv("CLAUDE_BASE_URL", "https://gw.example/v1")
    c = engine._default_router("claude-sonnet-4-6")
    assert _url(c) == "https://gw.example/v1" and c.api_key == "k"


def test_one_endpoint_for_everything(env):
    env.setenv("LLM_API_KEY", "one")
    env.setenv("LLM_BASE_URL", "https://litellm.example/v1")
    for m in ("gpt-5.4", "claude-sonnet-4-6", "text-embedding-3-large"):
        c = engine._default_router(m)
        assert _url(c) == "https://litellm.example/v1" and c.api_key == "one"


def test_deprecated_names_still_work_with_a_warning(env, capsys):
    env.setenv("ONET_CLAUDE_API_KEY", "old")
    env.setenv("ONET_CLAUDE_BASE_URL", "https://old-gw.example/v1")
    c = engine._default_router("claude-sonnet-4-6")
    assert _url(c) == "https://old-gw.example/v1" and c.api_key == "old"
    assert "ONET_CLAUDE_API_KEY is deprecated" in capsys.readouterr().err
    env.setenv("LITELLM_OPENAI_API_KEY", "lk")
    env.setenv("LITELLM_OPENAI_BASE_URL", "https://lite.example/v1")
    c = engine._default_router("gpt-5.4")
    assert _url(c) == "https://lite.example/v1" and c.api_key == "lk"


def test_the_old_claude_key_alone_no_longer_reaches_a_built_in_gateway(env):
    env.setenv("ONET_CLAUDE_API_KEY", "old")
    with pytest.raises(RuntimeError, match="CLAUDE_BASE_URL"):
        engine._default_router("claude-sonnet-4-6")
