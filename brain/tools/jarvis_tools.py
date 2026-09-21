"""The bot's tools, as OpenAI-style function-calling schemas.

Deliberately narrow. This agent gets **no filesystem access, no shell, no
editing** — none of Claude Code's built-in tools are enabled. A desk robot that
can rm -rf your home directory because it misheard you is not a desk robot.

Everything here is read-mostly: search the show, search what it remembers, write
a memory. That's the whole surface area in v1.

GLM 5.3 Flash speaks OpenAI-compatible tool calling (function name + JSON
arguments), not the Agent SDK's MCP protocol — so tools are plain JSON schemas
plus a dispatch table, and the tool-call loop lives in `brain.agent`.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from brain.embeddings import embed_one
from brain.store.db import Store

__all__ = ["build_tool_schemas", "build_tool_dispatch"]

ToolFn = Callable[[dict[str, Any]], Awaitable[str]]


def build_tool_schemas(skill_names: Sequence[str] = ()) -> list[dict[str, Any]]:
    """OpenAI `tools` array describing the tools below.

    `load_skill` only appears when there are skills to load — advertising a
    tool whose every call would fail is worse than not having it.
    """
    schemas: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "search_show",
                "description": (
                    "Search Numbers on the Board transcripts for what the guys actually "
                    "said about a topic, player, or team. Use this only when you're about "
                    "to claim something specific was said on the show, or need a real "
                    "basketball take to ground an answer — not for ordinary requests like "
                    "the calendar, memory, or small talk, where your own voice is enough."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": (
                    "Save something about Neil worth recalling in future conversations "
                    "— a preference, a goal, a decision, a fact about his life. Not for "
                    "trivia and not for things he only said in passing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"fact": {"type": "string"}},
                    "required": ["fact"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": (
                    "Search what you've been told about Neil previously. Use it when a "
                    "question depends on something he mentioned in an earlier conversation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]

    if skill_names:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "load_skill",
                    "description": (
                        "Read the full instructions for one of the skills listed in your "
                        "system prompt. Call this before relying on a skill — the summary "
                        "in the menu is not the procedure itself."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "enum": sorted(skill_names)}
                        },
                        "required": ["name"],
                    },
                },
            }
        )

    return schemas


def build_tool_dispatch(store: Store, skills: Mapping[str, str] | None = None) -> dict[str, ToolFn]:
    """Map tool name to its implementation, closing over `store` and `skills`.

    A factory rather than module-level globals so tests can point the tools at
    a temporary database. `skills` maps skill name to its full body.
    """

    async def search_show(args: dict[str, Any]) -> str:
        hits = store.search_canon(embed_one(str(args["query"])), k=6)
        if not hits:
            return "Nothing in the corpus on that. Say so; don't invent."
        return "\n\n".join(
            f"[{h.episode} @ {h.start:.0f}s, score {h.score:.2f}]\n{h.text}" for h in hits
        )

    async def remember(args: dict[str, Any]) -> str:
        fact = str(args["fact"]).strip()
        if not fact:
            return "Nothing to remember."
        store.add_memory(fact, embed_one(fact))
        return f"Remembered: {fact}"

    async def recall(args: dict[str, Any]) -> str:
        hits = store.search_memories(embed_one(str(args["query"])), k=5)
        if not hits:
            return "Nothing remembered about that."
        return "\n".join(f"- {h.text}" for h in hits)

    dispatch: dict[str, ToolFn] = {
        "search_show": search_show,
        "remember": remember,
        "recall": recall,
    }

    if skills:
        bodies = dict(skills)

        async def load_skill(args: dict[str, Any]) -> str:
            name = str(args["name"]).strip()
            body = bodies.get(name)
            if body is None:
                return f"No skill named {name!r}. Available: {', '.join(sorted(bodies))}."
            return body

        dispatch["load_skill"] = load_skill

    return dispatch


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """Parse a tool call's JSON argument string.

    Models occasionally emit malformed JSON for tool arguments; surfacing that
    as a clear error beats a raw `JSONDecodeError` three frames from here.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed tool arguments: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"tool arguments must be a JSON object, got {raw!r}")
    return parsed
