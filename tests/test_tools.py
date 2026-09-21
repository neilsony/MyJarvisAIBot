"""Tests for tool schema assembly and dispatch.

The tools the model can see and the tools it can actually call must stay in
lockstep. Advertising one the dispatch can't serve produces a confident tool
call that fails at runtime — a bug the model itself will then try to talk its
way around.
"""

import asyncio

import pytest

from brain.store.db import Store
from brain.tools.jarvis_tools import build_tool_dispatch, build_tool_schemas, parse_tool_arguments

SKILLS = {"citing": "Always call search_show first.", "banter": "Keep it short."}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


def names(schemas) -> list[str]:
    return [s["function"]["name"] for s in schemas]


class TestSchemas:
    def test_the_three_core_tools_are_always_present(self):
        assert names(build_tool_schemas()) == ["search_show", "remember", "recall"]

    def test_load_skill_is_absent_when_there_are_no_skills(self):
        # A tool whose every call would fail is worse than no tool.
        assert "load_skill" not in names(build_tool_schemas())

    def test_load_skill_appears_when_skills_exist(self):
        assert "load_skill" in names(build_tool_schemas(["citing"]))

    def test_load_skill_constrains_name_to_known_skills(self):
        schema = build_tool_schemas(["banter", "citing"])[-1]
        assert schema["function"]["parameters"]["properties"]["name"]["enum"] == [
            "banter",
            "citing",
        ]

    def test_every_schema_is_a_function_tool(self):
        assert all(s["type"] == "function" for s in build_tool_schemas(["citing"]))


class TestDispatch:
    def test_core_tools_are_always_dispatchable(self, store):
        assert set(build_tool_dispatch(store)) == {"search_show", "remember", "recall"}

    def test_no_load_skill_without_skills(self, store):
        assert "load_skill" not in build_tool_dispatch(store)

    def test_schemas_and_dispatch_agree(self, store):
        # The invariant that matters: anything advertised can be called.
        advertised = set(names(build_tool_schemas(list(SKILLS))))
        callable_ = set(build_tool_dispatch(store, skills=SKILLS))
        assert advertised == callable_

    def test_load_skill_returns_the_body(self, store):
        dispatch = build_tool_dispatch(store, skills=SKILLS)
        got = asyncio.run(dispatch["load_skill"]({"name": "citing"}))
        assert got == SKILLS["citing"]

    def test_load_skill_tolerates_surrounding_whitespace(self, store):
        dispatch = build_tool_dispatch(store, skills=SKILLS)
        assert asyncio.run(dispatch["load_skill"]({"name": " citing "})) == SKILLS["citing"]

    def test_unknown_skill_lists_what_is_available(self, store):
        dispatch = build_tool_dispatch(store, skills=SKILLS)
        got = asyncio.run(dispatch["load_skill"]({"name": "nope"}))
        assert "banter" in got and "citing" in got


class TestParseToolArguments:
    def test_parses_an_object(self):
        assert parse_tool_arguments('{"query": "wemby"}') == {"query": "wemby"}

    def test_malformed_json_raises_with_the_payload(self):
        with pytest.raises(ValueError, match="malformed"):
            parse_tool_arguments("{not json")

    def test_non_object_json_raises(self):
        with pytest.raises(ValueError, match="JSON object"):
            parse_tool_arguments('["a list"]')
