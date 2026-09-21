"""Tests for the on-demand web search tool.

The live lookup isn't tested — what matters here is that the `:online` suffix
actually gets applied (without it you silently get a model with no web access
and confident stale answers), and that a failed search degrades into a message
rather than taking down the turn.
"""

import asyncio
from types import SimpleNamespace

from brain.tools.web_search import ONLINE_SUFFIX, build_web_search_dispatch, web_search_schemas


class FakeClient:
    """Stands in for AsyncOpenAI, recording what it was asked for."""

    def __init__(self, reply: str = "an answer", raises: Exception | None = None):
        self.reply = reply
        self.raises = raises
        self.model_used: str | None = None
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, *, model, messages, max_tokens):  # noqa: ANN001
        self.model_used = model
        if self.raises:
            raise self.raises
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    async def close(self):
        self.closed = True


def dispatch_with(monkeypatch, client: FakeClient):
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_kwargs: client)
    return build_web_search_dispatch("key", "z-ai/glm-5.3-flash", "https://example.test/v1")


class TestSchemas:
    def test_exposes_web_search(self):
        assert [s["function"]["name"] for s in web_search_schemas()] == ["web_search"]

    def test_requires_a_query(self):
        params = web_search_schemas()[0]["function"]["parameters"]
        assert params["required"] == ["query"]

    def test_description_steers_away_from_banter(self):
        # The whole cost argument rests on it not firing on every turn.
        description = web_search_schemas()[0]["function"]["description"].lower()
        assert "don't use it" in description or "not" in description


class TestDispatch:
    def test_schemas_and_dispatch_agree(self, monkeypatch):
        dispatch = dispatch_with(monkeypatch, FakeClient())
        advertised = {s["function"]["name"] for s in web_search_schemas()}
        assert advertised == set(dispatch)

    def test_appends_the_online_suffix(self, monkeypatch):
        # Without this the search silently doesn't happen and the model
        # answers from stale training data as if it had looked it up.
        client = FakeClient()
        dispatch = dispatch_with(monkeypatch, client)
        asyncio.run(dispatch["web_search"]({"query": "who won last night"}))
        assert client.model_used == f"z-ai/glm-5.3-flash{ONLINE_SUFFIX}"

    def test_returns_the_answer(self, monkeypatch):
        dispatch = dispatch_with(monkeypatch, FakeClient(reply="Knicks won in five."))
        got = asyncio.run(dispatch["web_search"]({"query": "nba finals"}))
        assert got == "Knicks won in five."

    def test_empty_query_short_circuits(self, monkeypatch):
        client = FakeClient()
        dispatch = dispatch_with(monkeypatch, client)
        got = asyncio.run(dispatch["web_search"]({"query": "   "}))
        assert "No query" in got
        assert client.model_used is None

    def test_empty_answer_says_so(self, monkeypatch):
        dispatch = dispatch_with(monkeypatch, FakeClient(reply=""))
        assert "empty" in asyncio.run(dispatch["web_search"]({"query": "x"})).lower()

    def test_a_failed_search_does_not_raise(self, monkeypatch):
        # A dropped lookup should let the model say it couldn't check, not
        # error out the whole conversational turn.
        dispatch = dispatch_with(monkeypatch, FakeClient(raises=RuntimeError("upstream 503")))
        got = asyncio.run(dispatch["web_search"]({"query": "x"}))
        assert "Search failed" in got
        assert "upstream 503" in got

    def test_closes_the_client_even_on_failure(self, monkeypatch):
        client = FakeClient(raises=RuntimeError("boom"))
        dispatch = dispatch_with(monkeypatch, client)
        asyncio.run(dispatch["web_search"]({"query": "x"}))
        assert client.closed
