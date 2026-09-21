"""Tests for .env parsing and settings resolution.

A secrets file that silently drops a key produces a confusing 401 three layers
away, so malformed lines are loud and the real environment always wins.
"""

import pytest

from brain.config import (
    Settings,
    is_encrypted_env_file,
    load_env_file,
    parse_env,
    set_env_value,
)


class TestParseEnv:
    def test_simple_assignment(self):
        assert parse_env("FOO=bar") == {"FOO": "bar"}

    def test_multiple_lines(self):
        assert parse_env("A=1\nB=2") == {"A": "1", "B": "2"}

    def test_ignores_blank_lines_and_comments(self):
        assert parse_env("\n# a comment\n\nFOO=bar\n") == {"FOO": "bar"}

    def test_strips_whitespace_around_key_and_value(self):
        assert parse_env("  FOO  =  bar  ") == {"FOO": "bar"}

    def test_allows_export_prefix(self):
        assert parse_env("export FOO=bar") == {"FOO": "bar"}

    def test_empty_value(self):
        assert parse_env("FOO=") == {"FOO": ""}

    def test_value_may_contain_equals(self):
        assert parse_env("TOKEN=abc=def==") == {"TOKEN": "abc=def=="}

    def test_double_quoted_value_preserves_spaces(self):
        assert parse_env('FOO="  bar baz  "') == {"FOO": "  bar baz  "}

    def test_single_quoted_value(self):
        assert parse_env("FOO='bar baz'") == {"FOO": "bar baz"}

    def test_quoted_value_keeps_hash(self):
        assert parse_env('KEY="sk-ant-#-not-a-comment"') == {"KEY": "sk-ant-#-not-a-comment"}

    def test_strips_inline_comment_from_unquoted_value(self):
        assert parse_env("FOO=bar  # trailing note") == {"FOO": "bar"}

    def test_hash_without_leading_space_is_part_of_the_value(self):
        # API keys legitimately contain '#'; only ' #' starts a comment.
        assert parse_env("FOO=bar#baz") == {"FOO": "bar#baz"}

    def test_last_duplicate_wins(self):
        assert parse_env("FOO=1\nFOO=2") == {"FOO": "2"}

    def test_malformed_line_raises_with_line_number(self):
        with pytest.raises(ValueError, match="line 2"):
            parse_env("FOO=bar\nthis is not valid\n")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="line 1"):
            parse_env("=value")


class TestLoadEnvFile:
    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_env_file(tmp_path / "nope.env") == {}

    def test_reads_a_real_file(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("DEEPGRAM_API_KEY=dg-test\n")
        assert load_env_file(p) == {"DEEPGRAM_API_KEY": "dg-test"}


class TestSetEnvValue:
    def test_creates_a_new_file_when_none_exists(self, tmp_path):
        p = tmp_path / ".env"
        set_env_value(p, "FOO", "bar")
        assert load_env_file(p) == {"FOO": "bar"}

    def test_appends_a_new_key_to_an_existing_file(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("EXISTING=1\n")
        set_env_value(p, "NEW", "2")
        assert load_env_file(p) == {"EXISTING": "1", "NEW": "2"}

    def test_replaces_an_existing_key_in_place(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("A=1\nTARGET=old\nB=2\n")
        set_env_value(p, "TARGET", "new")
        assert load_env_file(p) == {"A": "1", "TARGET": "new", "B": "2"}

    def test_does_not_disturb_comments_or_other_lines(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("# a comment\nA=1\nTARGET=old\n")
        set_env_value(p, "TARGET", "new")
        assert "# a comment" in p.read_text()
        assert "A=1" in p.read_text()

    def test_value_with_embedded_double_quotes_survives_a_round_trip(self, tmp_path):
        # This is the actual use case: a JSON blob (the Google OAuth token).
        p = tmp_path / ".env"
        token = '{"access_token": "abc", "refresh_token": "xyz"}'
        set_env_value(p, "GOOGLE_TOKEN_JSON", token)
        assert load_env_file(p)["GOOGLE_TOKEN_JSON"] == token

    def test_only_the_matching_key_is_replaced_not_a_prefix_match(self, tmp_path):
        # FOO_BAR=1 must not be mistaken for a match on FOO.
        p = tmp_path / ".env"
        p.write_text("FOO_BAR=1\n")
        set_env_value(p, "FOO", "2")
        got = load_env_file(p)
        assert got["FOO_BAR"] == "1"
        assert got["FOO"] == "2"

    def test_rejects_a_value_containing_an_apostrophe(self, tmp_path):
        # Single-quoting is how embedded double quotes survive; a value with
        # its own apostrophe would close the quote early and corrupt the file.
        with pytest.raises(ValueError, match="apostrophe"):
            set_env_value(tmp_path / ".env", "KEY", "it's broken")


class TestEncryptedEnv:
    """dotenvx encrypts values in place; `dotenvx run --` injects the real ones
    into the environment, where they outrank the file. The danger is running
    *without* dotenvx, where ciphertext would be read as the secret itself."""

    def test_detects_an_encrypted_file(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text('#DOTENV_PUBLIC_KEY="03abc"\nFOO="encrypted:BX9s"\n')
        assert is_encrypted_env_file(p)

    def test_plain_file_is_not_encrypted(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("FOO=bar\n")
        assert not is_encrypted_env_file(p)

    def test_missing_file_is_not_encrypted(self, tmp_path):
        assert not is_encrypted_env_file(tmp_path / "nope")

    def test_ciphertext_from_file_raises_instead_of_being_used(self, tmp_path, monkeypatch):
        # The whole point: a nonsense key must fail loudly here, not as a
        # confusing 401 from OpenRouter three layers down.
        p = tmp_path / ".env"
        p.write_text('#DOTENV_PUBLIC_KEY="03abc"\nOPENROUTER_API_KEY="encrypted:BX9sQ2"\n')
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="dotenvx run"):
            Settings.load(env_file=p)

    def test_real_environment_still_wins_over_ciphertext(self, tmp_path, monkeypatch):
        # This is how dotenvx actually works — it injects the decrypted value
        # into the environment, which takes precedence over the file.
        p = tmp_path / ".env"
        p.write_text('#DOTENV_PUBLIC_KEY="03abc"\nOPENROUTER_API_KEY="encrypted:BX9sQ2"\n')
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-decrypted-real-value")
        assert Settings.load(env_file=p).openrouter_api_key == "sk-decrypted-real-value"


class TestSettings:
    def test_real_environment_beats_the_env_file(self, tmp_path, monkeypatch):
        p = tmp_path / ".env"
        p.write_text("DEEPGRAM_API_KEY=from-file\n")
        monkeypatch.setenv("DEEPGRAM_API_KEY", "from-shell")
        assert Settings.load(env_file=p).deepgram_api_key == "from-shell"

    def test_falls_back_to_the_env_file(self, tmp_path, monkeypatch):
        p = tmp_path / ".env"
        p.write_text("DEEPGRAM_API_KEY=from-file\n")
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        assert Settings.load(env_file=p).deepgram_api_key == "from-file"

    def test_model_defaults_to_glm_5_3_flash(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        assert Settings.load(env_file=tmp_path / "none").model == "z-ai/glm-5.3-flash"

    def test_model_is_overridable_for_ab_testing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_MODEL", "z-ai/glm-5.3")
        assert Settings.load(env_file=tmp_path / "none").model == "z-ai/glm-5.3"

    def test_missing_secret_is_none_not_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert Settings.load(env_file=tmp_path / "none").hf_token is None

    def test_require_raises_a_actionable_message(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        s = Settings.load(env_file=tmp_path / "none")
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            s.require("hf_token")

    def test_require_returns_the_value_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_abc")
        assert Settings.load(env_file=tmp_path / "none").require("hf_token") == "hf_abc"
