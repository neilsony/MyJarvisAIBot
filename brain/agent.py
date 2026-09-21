"""Wire the persona, the store, and the tools into a running agent.

Talks to GLM 5.3 Flash through OpenRouter's OpenAI-compatible endpoint, via the
`openai` SDK pointed at OpenRouter's base URL. There is no Agent SDK here and
no server-side tool loop — OpenRouter just returns `tool_calls`, so this module
runs that loop itself: send messages, execute any requested tools, feed the
results back, repeat until the model answers in plain text or `max_turns` is
hit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brain.config import Settings
from brain.embeddings import embed_one
from brain.persona.prompt import build_system_prompt, format_context_block
from brain.skills import discover_skills
from brain.store.db import Store
from brain.tools.calendar import build_calendar_dispatch, calendar_schemas, connect_calendar
from brain.tools.jarvis_tools import build_tool_dispatch, build_tool_schemas, parse_tool_arguments
from brain.tools.web_search import build_web_search_dispatch, web_search_schemas

__all__ = ["Jarvis", "load_persona"]

REGISTER_PATH = Path(__file__).parent / "persona" / "register.md"
SKILLS_PATH = Path(__file__).parent / "skills"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# How much retrieved material to put in front of the model each turn. More is
# not better: past a point it dilutes the register with other hosts' voices.
MEMORY_RESULTS = 3

# Voice replies are short; a long tool-call chain means a long silence.
MAX_TOOL_ROUNDS = 6


@dataclass(frozen=True)
class Persona:
    register: str
    profile: dict[str, str]


def load_persona(settings: Settings) -> Persona:
    """Read the register and every profile document from disk.

    Profile files are optional — a fresh checkout has no `profile/` at all,
    since it's gitignored, and the bot should still run.

    `<!-- -->` blocks are stripped from the register before it reaches the
    model. They're editor's notes about *how to tune this file* — genuinely
    useful for a human, actively confusing as an instruction the model is
    expected to follow, and previously going out verbatim on every turn.
    """
    register = REGISTER_PATH.read_text(encoding="utf-8") if REGISTER_PATH.is_file() else ""
    register = _HTML_COMMENT.sub("", register).strip()
    profile: dict[str, str] = {}
    if settings.profile_dir.is_dir():
        for path in sorted(settings.profile_dir.glob("*.md")):
            profile[path.name] = path.read_text(encoding="utf-8")
    return Persona(register=register, profile=profile)


class Jarvis:
    """One conversation with DmillsGPT.

    Retrieval happens here, per turn, and the results go into the *user
    message* — never the system prompt. See `brain.persona.prompt` for why
    that ordering is load-bearing (it keeps the cached prefix stable).
    """

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.persona = load_persona(settings)
        self.skills = discover_skills(SKILLS_PATH)
        self._tool_dispatch = build_tool_dispatch(
            store, skills={skill.name: skill.content for skill in self.skills}
        )
        self._tool_schemas = build_tool_schemas([skill.name for skill in self.skills])

        # Calendar is optional: no GOOGLE_CLIENT_ID/SECRET in .env means no
        # calendar tools, and the bot runs exactly as it did before.
        calendar = connect_calendar(
            settings.google_client_id,
            settings.google_client_secret,
            settings.google_token_json,
            settings.env_file,
        )
        if calendar is not None:
            self._tool_schemas += calendar_schemas()
            self._tool_dispatch |= build_calendar_dispatch(calendar)

        # Web search rides on the same OpenRouter key, so it's available
        # whenever the agent itself is. On-demand only — see web_search.py.
        if settings.openrouter_api_key:
            self._tool_schemas += web_search_schemas()
            self._tool_dispatch |= build_web_search_dispatch(
                settings.openrouter_api_key, settings.model, OPENROUTER_BASE_URL
            )

        self._system_prompt = build_system_prompt(
            register=self.persona.register,
            profile=self.persona.profile,
            skills=[(skill.name, skill.description) for skill in self.skills],
        )
        self._client: Any = None
        self._history: list[dict[str, Any]] = []

    def retrieve(self, message: str) -> str:
        """Build this turn's reference block from memory.

        Canon is deliberately *not* auto-injected here. It used to be, on
        every turn, regardless of relevance — so "add lunch to my calendar"
        got basketball transcript snippets shoved into it as "reference
        material" just as often as an actual basketball question did, which
        pulled the model toward citing the show even on requests that had
        nothing to do with it. Register carries the voice; Canon is for when
        the model itself decides a claim needs grounding, via `search_show`.
        """
        query = embed_one(message)
        memories = [h.text for h in self.store.search_memories(query, k=MEMORY_RESULTS)]
        return format_context_block(canon=[], memories=memories)

    async def __aenter__(self) -> Jarvis:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.settings.require("openrouter_api_key"),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def ask(self, message: str) -> str:
        """Send one turn and return the reply text."""
        if self._client is None:
            raise RuntimeError("use Jarvis as an async context manager")

        context = self.retrieve(message)
        prompt = f"{context}\n\n{message}" if context else message
        self._history.append({"role": "user", "content": prompt})

        for _ in range(MAX_TOOL_ROUNDS):
            response = await self._client.chat.completions.create(
                model=self.settings.model,
                messages=[{"role": "system", "content": self._system_prompt}, *self._history],
                tools=self._tool_schemas,
            )
            choice = response.choices[0].message
            self._history.append(choice.model_dump(exclude_none=True))

            if not choice.tool_calls:
                return (choice.content or "").strip()

            for call in choice.tool_calls:
                result = await self._run_tool(call.function.name, call.function.arguments)
                self._history.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )

        return "Sorry, that took too many steps to work out — try asking it differently."

    async def _run_tool(self, name: str, raw_arguments: str) -> str:
        fn = self._tool_dispatch.get(name)
        if fn is None:
            return f"Unknown tool: {name}"
        try:
            args = parse_tool_arguments(raw_arguments)
            return await fn(args)
        except ValueError as exc:
            return f"Tool error: {exc}"
