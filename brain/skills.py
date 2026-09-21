"""Skill files: instructions the bot loads on demand rather than always carrying.

A skill is a markdown file with two-field frontmatter. Its *description* is
cheap and always visible; its *body* costs real tokens and is fetched only when
the model asks for it via the `load_skill` tool.

That split is the point. Everything the bot might ever need to know cannot live
in the system prompt — a spoken-reply bot with a 3-sentence budget does not want
four pages of procedure diluting the register on every turn. A one-line menu
plus on-demand loading keeps the stable prefix small and still lets the model
reach for detail when a turn actually calls for it.

Format:

    ---
    name: citing-the-show
    description: How to quote the show accurately, with episode and timestamp.
    ---

    ...markdown body, loaded only on demand...

Parsing is deliberately strict — a malformed skill file raises rather than being
skipped, because a silently-ignored skill is indistinguishable from one the
model simply chose not to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["Skill", "discover_skills", "parse_skill_file"]

_REQUIRED_FIELDS = frozenset({"name", "description"})


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str


def parse_skill_file(path: Path) -> Skill:
    """Parse one skill file into a Skill, or raise explaining what's wrong."""
    lines = path.read_text(encoding="utf-8").splitlines()

    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path}: a skill file must open with a '---' frontmatter fence")

    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"{path}: frontmatter is never closed (missing a second '---')") from exc

    fields: dict[str, str] = {}
    for lineno, raw in enumerate(lines[1:closing], start=2):
        line = raw.strip()
        if not line:
            continue
        key, sep, value = line.partition(":")
        if not sep or not key.strip():
            raise ValueError(f"{path}: malformed frontmatter on line {lineno}: {raw!r}")
        fields[key.strip()] = value.strip()

    missing = _REQUIRED_FIELDS - fields.keys()
    if missing:
        raise ValueError(f"{path}: frontmatter is missing {', '.join(sorted(missing))}")

    body = "\n".join(lines[closing + 1 :]).strip()
    if not body:
        raise ValueError(f"{path}: skill has frontmatter but no body")

    return Skill(name=fields["name"], description=fields["description"], content=body)


def discover_skills(skills_dir: Path) -> list[Skill]:
    """Every skill in `skills_dir`, name-sorted.

    A missing directory is not an error — the bot runs fine with no skills at
    all. Sorted so the generated prompt is byte-identical across runs.
    """
    if not skills_dir.is_dir():
        return []

    skills = [parse_skill_file(path) for path in sorted(skills_dir.glob("*.md"))]

    names = [skill.name for skill in skills]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"{skills_dir}: duplicate skill name(s): {', '.join(sorted(duplicates))}. "
            f"Names address the skill in `load_skill`, so they must be unique."
        )

    return sorted(skills, key=lambda skill: skill.name)
