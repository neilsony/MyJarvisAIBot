"""Tests for skill file discovery and parsing.

Skills are instructions the bot pulls in on demand. A skill that fails to parse
must say so loudly: a silently-skipped skill looks exactly like one the model
chose not to use, which is a miserable thing to debug at 11pm.
"""

import pytest

from brain.skills import discover_skills, parse_skill_file

VALID = """\
---
name: citing-the-show
description: How to quote the show with episode and timestamp.
---

Always call `search_show` before claiming the guys said something.
"""


def write(tmp_path, filename: str, text: str):
    path = tmp_path / filename
    path.write_text(text, encoding="utf-8")
    return path


class TestParseSkillFile:
    def test_reads_name_and_description(self, tmp_path):
        skill = parse_skill_file(write(tmp_path, "a.md", VALID))
        assert skill.name == "citing-the-show"
        assert skill.description == "How to quote the show with episode and timestamp."

    def test_body_excludes_frontmatter(self, tmp_path):
        skill = parse_skill_file(write(tmp_path, "a.md", VALID))
        assert skill.content.startswith("Always call")
        assert "---" not in skill.content
        assert "description:" not in skill.content

    def test_body_keeps_internal_markdown(self, tmp_path):
        text = "---\nname: n\ndescription: d\n---\n\n## Heading\n\n- bullet\n- another\n"
        assert "## Heading" in parse_skill_file(write(tmp_path, "a.md", text)).content

    def test_description_may_contain_colons(self, tmp_path):
        text = "---\nname: n\ndescription: Do this: then that.\n---\n\nbody\n"
        got = parse_skill_file(write(tmp_path, "a.md", text))
        assert got.description == "Do this: then that."

    def test_extra_frontmatter_fields_are_ignored(self, tmp_path):
        text = "---\nname: n\ndescription: d\nauthor: neil\n---\n\nbody\n"
        assert parse_skill_file(write(tmp_path, "a.md", text)).name == "n"

    def test_blank_frontmatter_lines_are_fine(self, tmp_path):
        text = "---\nname: n\n\ndescription: d\n---\n\nbody\n"
        assert parse_skill_file(write(tmp_path, "a.md", text)).description == "d"

    def test_missing_opening_fence_raises(self, tmp_path):
        with pytest.raises(ValueError, match="frontmatter fence"):
            parse_skill_file(write(tmp_path, "a.md", "name: n\n\nbody\n"))

    def test_unterminated_frontmatter_raises(self, tmp_path):
        with pytest.raises(ValueError, match="never closed"):
            parse_skill_file(write(tmp_path, "a.md", "---\nname: n\ndescription: d\n\nbody\n"))

    def test_missing_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="name"):
            parse_skill_file(write(tmp_path, "a.md", "---\ndescription: d\n---\n\nbody\n"))

    def test_missing_description_raises(self, tmp_path):
        with pytest.raises(ValueError, match="description"):
            parse_skill_file(write(tmp_path, "a.md", "---\nname: n\n---\n\nbody\n"))

    def test_malformed_frontmatter_line_reports_line_number(self, tmp_path):
        text = "---\nname: n\nthis line has no colon\ndescription: d\n---\n\nbody\n"
        with pytest.raises(ValueError, match="line 3"):
            parse_skill_file(write(tmp_path, "a.md", text))

    def test_empty_body_raises(self, tmp_path):
        # A skill with nothing to load is a menu entry that wastes a tool call.
        with pytest.raises(ValueError, match="no body"):
            parse_skill_file(write(tmp_path, "a.md", "---\nname: n\ndescription: d\n---\n\n"))


class TestDiscoverSkills:
    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert discover_skills(tmp_path / "nope") == []

    def test_empty_directory_yields_nothing(self, tmp_path):
        assert discover_skills(tmp_path) == []

    def test_finds_every_markdown_file(self, tmp_path):
        write(tmp_path, "a.md", "---\nname: alpha\ndescription: d\n---\n\nbody\n")
        write(tmp_path, "b.md", "---\nname: beta\ndescription: d\n---\n\nbody\n")
        assert [s.name for s in discover_skills(tmp_path)] == ["alpha", "beta"]

    def test_sorted_by_name_not_filename(self, tmp_path):
        write(tmp_path, "z-first.md", "---\nname: alpha\ndescription: d\n---\n\nbody\n")
        write(tmp_path, "a-second.md", "---\nname: zulu\ndescription: d\n---\n\nbody\n")
        assert [s.name for s in discover_skills(tmp_path)] == ["alpha", "zulu"]

    def test_ignores_non_markdown_files(self, tmp_path):
        write(tmp_path, "a.md", "---\nname: alpha\ndescription: d\n---\n\nbody\n")
        write(tmp_path, "notes.txt", "not a skill")
        assert [s.name for s in discover_skills(tmp_path)] == ["alpha"]

    def test_duplicate_names_raise(self, tmp_path):
        # Names address the skill in load_skill, so a collision makes one of
        # them unreachable — and which one would depend on filename order.
        write(tmp_path, "a.md", "---\nname: same\ndescription: d\n---\n\nbody\n")
        write(tmp_path, "b.md", "---\nname: same\ndescription: d\n---\n\nbody\n")
        with pytest.raises(ValueError, match="duplicate"):
            discover_skills(tmp_path)

    def test_a_broken_skill_fails_the_whole_discovery(self, tmp_path):
        write(tmp_path, "good.md", "---\nname: good\ndescription: d\n---\n\nbody\n")
        write(tmp_path, "bad.md", "no frontmatter at all\n")
        with pytest.raises(ValueError):
            discover_skills(tmp_path)
