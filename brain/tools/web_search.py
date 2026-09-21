"""Web search, on demand, via OpenRouter's `:online` plugin.

Deliberately a *tool* rather than always-on augmentation. Flipping the main
model to `...:online` would search the web on every single turn — including
"what's on my calendar" and pure banter — which costs ~$0.007 a turn (an order
of magnitude more than the model itself) and drags irrelevant web text into
replies that never needed it. That's the same mistake auto-injected Canon made;
see `Jarvis.retrieve`.

So: the model calls this when it decides it needs something current, and
otherwise never pays for it.

Implementation is a second, small OpenRouter call with `:online` appended to
the model slug. That reuses the key you already have rather than adding an
account with Exa or Brave directly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

__all__ = ["build_web_search_dispatch", "web_search_schemas", "ONLINE_SUFFIX"]

ToolFn = Callable[[dict[str, Any]], Awaitable[str]]

ONLINE_SUFFIX = ":online"

# A spoken reply can't carry a research report. Keeping the searched answer
# short also keeps it from swamping the register when it lands in context.
MAX_ANSWER_TOKENS = 350

_SEARCH_SYSTEM = (
    "You are a search assistant. Answer the query factually and concisely from "
    "current web results, in at most four sentences. Include specific numbers, "
    "dates and names where relevant. If the results don't answer it, say so "
    "plainly rather than guessing."
)


def web_search_schemas() -> list[dict[str, Any]]:
    """The OpenAI `tools` entry for web search."""
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Look something up on the live web. Use this for anything that "
                    "changed recently or that you can't know: today's scores, current "
                    "standings, injuries, trades, news, who won last night. Don't use "
                    "it for opinions, banter, or things already in your context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to look up, as a search query.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]


def build_web_search_dispatch(api_key: str, model: str, base_url: str) -> dict[str, ToolFn]:
    """Map `web_search` to a live lookup, closing over the OpenRouter config."""

    async def web_search(args: dict[str, Any]) -> str:
        from openai import AsyncOpenAI

        query = str(args["query"]).strip()
        if not query:
            return "No query given."

        # A fresh client per search: these are infrequent by design, and it
        # avoids threading another lifecycle through the agent.
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        try:
            response = await client.chat.completions.create(
                model=f"{model}{ONLINE_SUFFIX}",
                messages=[
                    {"role": "system", "content": _SEARCH_SYSTEM},
                    {"role": "user", "content": query},
                ],
                max_tokens=MAX_ANSWER_TOKENS,
            )
            answer = (response.choices[0].message.content or "").strip()
            return answer or "Search came back empty."
        except Exception as exc:
            # A failed lookup shouldn't kill the turn — the model can say it
            # couldn't check rather than the whole reply erroring out.
            return f"Search failed: {type(exc).__name__}: {exc}"
        finally:
            await client.close()

    return {"web_search": web_search}
