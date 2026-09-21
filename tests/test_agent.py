"""Tests for persona loading and per-turn retrieval.

Covers two related bugs found in the same review: Canon used to be force-fed
into every turn regardless of relevance (fighting the "Register carries the
voice" design), and the register's own editor's-note comment was leaking into
the model's actual system prompt.
"""

from pathlib import Path

import numpy as np

from brain.agent import _HTML_COMMENT, Jarvis, load_persona
from brain.config import Settings
from brain.store.db import Store
from pipeline.transcript import Chunk


def vec(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


def chunk(text: str, episode: str = "ep1", start: float = 0.0) -> Chunk:
    return Chunk(text=text, speakers=("DARRICK",), episode=episode, start=start, end=start + 10)


class TestStripHtmlComments:
    def test_removes_a_comment_block(self):
        assert _HTML_COMMENT.sub("", "<!-- note -->\nbody") == "\nbody"

    def test_removes_a_multiline_comment_block(self):
        text = "<!--\nline one\nline two\n-->\nbody"
        assert _HTML_COMMENT.sub("", text).strip() == "body"

    def test_leaves_normal_text_alone(self):
        assert _HTML_COMMENT.sub("", "just text, no comments") == "just text, no comments"


class TestLoadPersona:
    def test_register_excludes_editor_comment(self, tmp_path, monkeypatch):
        from brain import agent as agent_module

        register_path = tmp_path / "register.md"
        register_path.write_text("<!-- tuning notes for a human -->\nActual voice text.")
        monkeypatch.setattr(agent_module, "REGISTER_PATH", register_path)

        settings = Settings.load(env_file=tmp_path / "none")
        persona = load_persona(settings)

        assert "tuning notes" not in persona.register
        assert "Actual voice text." in persona.register

    def test_missing_register_file_is_not_an_error(self, tmp_path, monkeypatch):
        from brain import agent as agent_module

        monkeypatch.setattr(agent_module, "REGISTER_PATH", tmp_path / "nope.md")
        settings = Settings.load(env_file=tmp_path / "none")
        assert load_persona(settings).register == ""


class TestRetrieve:
    def _jarvis(self, tmp_path: Path, monkeypatch) -> Jarvis:
        from brain import agent as agent_module

        monkeypatch.setattr(agent_module, "REGISTER_PATH", tmp_path / "nope.md")
        monkeypatch.setattr(agent_module, "SKILLS_PATH", tmp_path / "no-skills")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        settings = Settings.load(env_file=tmp_path / "none")
        store = Store(tmp_path / "test.db")
        return Jarvis(settings, store)

    def test_does_not_auto_inject_canon(self, tmp_path, monkeypatch):
        # The bug: Canon used to be searched and injected on every turn
        # regardless of relevance, fighting the "Register carries the voice,
        # search_show is opt-in" design. This is the regression test for it.
        jarvis = self._jarvis(tmp_path, monkeypatch)
        jarvis.store.add_chunks([chunk("DARRICK: wemby is a top-5 player")], [vec(1, 0)])

        monkeypatch.setattr("brain.agent.embed_one", lambda _text: vec(1, 0))
        got = jarvis.retrieve("what did the guys say about wemby")

        assert "wemby" not in got.lower()

    def test_still_surfaces_relevant_memories(self, tmp_path, monkeypatch):
        jarvis = self._jarvis(tmp_path, monkeypatch)
        jarvis.store.add_memory("Neil roots for the Thunder", vec(1, 0))

        monkeypatch.setattr("brain.agent.embed_one", lambda _text: vec(1, 0))
        got = jarvis.retrieve("what team do I like")

        assert "Thunder" in got

    def test_no_memories_and_no_canon_yields_empty_block(self, tmp_path, monkeypatch):
        jarvis = self._jarvis(tmp_path, monkeypatch)
        monkeypatch.setattr("brain.agent.embed_one", lambda _text: vec(1, 0))
        assert jarvis.retrieve("add lunch to my calendar") == ""
