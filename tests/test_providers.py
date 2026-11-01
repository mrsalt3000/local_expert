"""Тесты слоя провайдеров: Ollama и OpenAI-compatible на httpx.MockTransport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from book_expert.config import Settings
from book_expert.providers import (
    ChatMessage,
    ProviderError,
    ProviderUnavailableError,
    get_provider,
)
from book_expert.providers.ollama import OllamaProvider
from book_expert.providers.deepseek import DeepSeekProvider
from book_expert.providers.openai_compat import OpenAICompatibleProvider

Handler = Any


def make_client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def ollama_settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        provider="ollama",
        ollama_base_url="http://ollama.test",
        **overrides,
    )


def openai_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "provider": "openai",
        "openai_base_url": "http://openai.test/v1",
        "openai_api_key": "sk-test",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def deepseek_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "provider": "deepseek",
        "deepseek_base_url": "http://deepseek.test/v1",
        "deepseek_api_key": "sk-deepseek",
        "ollama_base_url": "http://ollama.test",
        "embedding_model": "bge-m3",
        "embedding_dim": 2,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_deepseek_chat_and_local_embed_delegate() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "model": "deepseek-chat",
                    "choices": [{"message": {"content": "облако"}}],
                    "usage": {"total_tokens": 3},
                },
            )
        if request.url.path.endswith("/api/embed"):
            return httpx.Response(
                200,
                json={"embeddings": [[0.1, 0.2]], "model": "bge-m3"},
            )
        return httpx.Response(500, json={})

    provider = DeepSeekProvider(deepseek_settings(), client=make_client(handler))
    chat = provider.chat([ChatMessage(role="user", content="hi")])
    embed = provider.embed(["текст"])

    assert chat.text == "облако"
    assert chat.model == "deepseek-chat"
    assert embed.dim == 2
    assert any("/chat/completions" in url for url in calls)
    assert any("/api/embed" in url for url in calls)


def test_ollama_chat_payload_and_parsing() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen3.5:4b",
                "message": {"role": "assistant", "content": "привет"},
                "prompt_eval_count": 7,
                "eval_count": 3,
            },
        )

    provider = OllamaProvider(ollama_settings(), client=make_client(handler))
    result = provider.chat(
        [
            ChatMessage(role="system", content="s"),
            ChatMessage(role="user", content="u"),
        ],
        temperature=0.2,
        max_tokens=64,
    )

    assert captured["method"] == "POST"
    assert captured["url"] == "http://ollama.test/api/chat"
    assert captured["body"] == {
        "model": "qwen3.5:4b",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 64},
    }
    assert result.text == "привет"
    assert result.model == "qwen3.5:4b"
    assert result.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


def test_ollama_chat_omits_num_predict_without_max_tokens() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"model": "qwen3.5:4b", "message": {"content": "ok"}},
        )

    provider = OllamaProvider(ollama_settings(), client=make_client(handler))
    result = provider.chat([ChatMessage(role="user", content="hi")])

    assert captured["body"]["options"] == {"temperature": 0.0, "num_ctx": 8192}
    assert result.usage == {}
    assert result.text == "ok"


def test_ollama_chat_uses_configured_num_ctx() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"model": "qwen3.5:4b", "message": {"content": "ok"}},
        )

    provider = OllamaProvider(
        ollama_settings(ollama_num_ctx=12288),
        client=make_client(handler),
    )
    provider.chat([ChatMessage(role="user", content="hi")])

    assert captured["body"]["options"]["num_ctx"] == 12288


def test_ollama_embed_multiple_texts() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "bge-m3",
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            },
        )

    provider = OllamaProvider(ollama_settings(), client=make_client(handler))
    result = provider.embed(["первый", "второй"])

    assert captured["url"] == "http://ollama.test/api/embed"
    assert captured["body"] == {"model": "bge-m3", "input": ["первый", "второй"]}
    assert result.vectors == ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    assert result.dim == 3
    assert result.model == "bge-m3"


def test_ollama_embed_count_mismatch_is_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "bge-m3", "embeddings": [[0.1, 0.2]]},
        )

    provider = OllamaProvider(ollama_settings(), client=make_client(handler))
    with pytest.raises(ProviderError, match="ожидалось 2 векторов"):
        provider.embed(["a", "b"])


def test_ollama_health_parses_tags() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "http://ollama.test/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "bge-m3"},
                    {"name": "qwen3.5:4b"},
                    {"size": 1},
                ]
            },
        )

    provider = OllamaProvider(ollama_settings(), client=make_client(handler))
    health = provider.health()

    assert health.ok is True
    assert health.models == ("bge-m3", "qwen3.5:4b")
    assert "ollama.test" in health.detail


def test_ollama_health_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = OllamaProvider(ollama_settings(), client=make_client(handler))
    with pytest.raises(ProviderUnavailableError, match="ollama serve"):
        provider.health()


def test_ollama_http_error_has_status_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom " + "x" * 500)

    provider = OllamaProvider(ollama_settings(), client=make_client(handler))
    with pytest.raises(ProviderError, match="HTTP 500") as excinfo:
        provider.chat([ChatMessage(role="user", content="hi")])

    assert "x" * 500 not in str(excinfo.value)
    assert len(str(excinfo.value)) < 500


def test_openai_chat_payload_and_parsing() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ответ"}}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            },
        )

    provider = OpenAICompatibleProvider(openai_settings(), client=make_client(handler))
    result = provider.chat(
        [ChatMessage(role="user", content="hi")],
        temperature=0.0,
        max_tokens=32,
    )

    assert captured["url"] == "http://openai.test/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"] == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 32,
    }
    assert result.text == "ответ"
    assert result.usage["total_tokens"] == 7


def test_openai_uses_default_base_url() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    settings = openai_settings(openai_base_url=None)
    provider = OpenAICompatibleProvider(settings, client=make_client(handler))
    provider.chat([ChatMessage(role="user", content="hi")])

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"


def test_openai_requires_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be used")

    settings = openai_settings(openai_api_key=None)
    provider = OpenAICompatibleProvider(settings, client=make_client(handler))

    with pytest.raises(ProviderUnavailableError, match="OPENAI_API_KEY"):
        provider.chat([ChatMessage(role="user", content="hi")])


def test_openai_embed_keeps_order() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "text-embedding-3-small",
                "data": [
                    {"index": 1, "embedding": [4.0, 5.0]},
                    {"index": 0, "embedding": [1.0, 2.0]},
                ],
            },
        )

    provider = OpenAICompatibleProvider(openai_settings(), client=make_client(handler))
    result = provider.embed(["a", "b"])

    assert captured["url"] == "http://openai.test/v1/embeddings"
    assert captured["body"] == {
        "model": "text-embedding-3-small",
        "input": ["a", "b"],
    }
    assert result.vectors == ((1.0, 2.0), (4.0, 5.0))
    assert result.dim == 2


def test_openai_health_uses_models_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "http://openai.test/v1/models"
        assert request.headers.get("authorization") == "Bearer sk-test"
        return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})

    provider = OpenAICompatibleProvider(openai_settings(), client=make_client(handler))
    health = provider.health()

    assert health.ok is True
    assert health.models == ("gpt-4o-mini",)


def test_openai_http_error_has_status_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal")

    provider = OpenAICompatibleProvider(openai_settings(), client=make_client(handler))
    with pytest.raises(ProviderError, match="HTTP 500"):
        provider.chat([ChatMessage(role="user", content="hi")])


def test_get_provider_selects_adapter() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    with make_client(handler) as client:
        ollama = get_provider(ollama_settings(), client=client)
        openai = get_provider(openai_settings(), client=client)
        deepseek = get_provider(deepseek_settings(), client=client)

    assert isinstance(ollama, OllamaProvider)
    assert isinstance(openai, OpenAICompatibleProvider)
    assert isinstance(deepseek, DeepSeekProvider)
    assert ollama.name == "ollama"
    assert openai.name == "openai"
    assert deepseek.name == "deepseek"


def test_get_provider_rejects_unknown_provider() -> None:
    settings = ollama_settings()
    settings.provider = "anthropic"

    with pytest.raises(ValueError, match="anthropic"):
        get_provider(settings)
