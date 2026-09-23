"""Settings and secrets for the whole project.

Secrets live in a gitignored `.env` at the repo root, encrypted at rest with
dotenvx. Real environment variables always win over the file, which is exactly
what makes the encryption work: `dotenvx run -- <cmd>` decrypts and injects the
real values into the process environment before Python starts, so they take
precedence over the ciphertext still sitting in the file.

The consequence is that **commands must be run through `dotenvx run --`**.
Without it, the file's encrypted values would be read as if they were the
secrets themselves — a garbage API key producing a puzzling 401 three layers
down. `_reject_ciphertext` turns that into a loud, obvious error instead.

The model id lives here, alone, on purpose: swapping models while tuning the
register should be a one-line change, not a search-and-replace.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Settings", "load_env_file", "parse_env", "set_env_value", "REPO_ROOT"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

# How dotenvx marks an encrypted value, and the header it writes at the top of
# an encrypted file.
CIPHERTEXT_PREFIX = "encrypted:"
ENCRYPTED_FILE_MARKER = "DOTENV_PUBLIC_KEY"

# Human-readable names for the error message when a secret is missing.
_ENV_NAMES = {
    "deepgram_api_key": "DEEPGRAM_API_KEY",
    "hf_token": "HF_TOKEN",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "google_client_id": "GOOGLE_CLIENT_ID",
    "google_client_secret": "GOOGLE_CLIENT_SECRET",
    "spotify_client_id": "SPOTIFY_CLIENT_ID",
    "spotify_client_secret": "SPOTIFY_CLIENT_SECRET",
}


def parse_env(text: str) -> dict[str, str]:
    """Parse dotenv-format `text`.

    Deliberately strict: a line that isn't blank, a comment, or a `KEY=value`
    raises rather than being skipped. In a secrets file a typo'd line is a bug
    you want at parse time, not as a 401 from a service three layers down.

    Quoting rules follow the usual dotenv convention — quoted values are taken
    literally (so a `#` inside an API key survives), unquoted values have a
    ` #` trailing comment stripped.
    """
    values: dict[str, str] = {}

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        line = line.removeprefix("export ").lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"malformed .env line {lineno}: expected KEY=value, got {raw!r}")

        key = key.strip()
        if not key:
            raise ValueError(f"malformed .env line {lineno}: empty key in {raw!r}")

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            # Only ' #' opens a comment — bare '#' is legal inside a token.
            comment = value.find(" #")
            if comment != -1:
                value = value[:comment].rstrip()

        values[key] = value

    return values


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a dotenv file, or return {} when it doesn't exist.

    A missing `.env` is normal — everything in it may be set in the shell instead.
    """
    if not path.is_file():
        return {}
    return parse_env(path.read_text(encoding="utf-8"))


def is_encrypted_env_file(path: Path) -> bool:
    """True when this dotenv file has been encrypted by dotenvx."""
    if not path.is_file():
        return False
    return ENCRYPTED_FILE_MARKER in path.read_text(encoding="utf-8")


def _reject_ciphertext(name: str, value: str) -> str:
    """Refuse a value that is still encrypted, with an actionable message.

    Reading ciphertext as if it were the secret is the one failure mode
    encryption introduces, and it's a quiet one: the request goes out with a
    nonsense key and comes back a 401 from somewhere unrelated.
    """
    if value.startswith(CIPHERTEXT_PREFIX):
        raise RuntimeError(
            f"{name} is still encrypted — this process wasn't started through dotenvx.\n"
            f"Run it as:  dotenvx run -- <your command>\n"
            f"e.g.        dotenvx run -- python -m brain.cli"
        )
    return value


def set_env_value(path: Path, key: str, value: str) -> None:
    """Write one `KEY=value` line into a dotenv file, touching nothing else.

    For values that legitimately change at runtime and must persist — right
    now, only the Google and Spotify OAuth tokens after they refresh. Everything else in
    the file (comments, blank lines, other keys) is preserved byte-for-byte;
    only the matching `KEY=` line is replaced, or appended if absent.

    Always single-quotes the value. The token is a JSON blob full of double
    quotes; wrapping it in single quotes is the one form `parse_env` reads back
    literally without those inner quotes closing the value early.

    If the file is encrypted, the freshly-written plaintext is re-encrypted
    immediately. Otherwise refreshing an OAuth token would silently punch a
    plaintext secret back into an otherwise-encrypted file — and nothing would
    ever tell you it happened.
    """
    if "'" in value:
        raise ValueError("set_env_value cannot safely quote a value containing an apostrophe")

    was_encrypted = is_encrypted_env_file(path)

    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    new_line = f"{key}='{value}'"
    prefix = f"{key}="

    for i, raw in enumerate(lines):
        stripped = raw.strip().removeprefix("export ").lstrip()
        if stripped.startswith(prefix):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if was_encrypted:
        _encrypt_key(path, key)


def _encrypt_key(path: Path, key: str) -> None:
    """Re-encrypt a single key in place via the dotenvx CLI."""
    try:
        subprocess.run(
            ["npx", "--yes", "@dotenvx/dotenvx@latest", "encrypt", "-f", str(path), "-k", key],
            check=True,
            capture_output=True,
            cwd=str(path.parent),
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"Wrote {key} to {path} but could not re-encrypt it — that value is "
            f"now sitting in plaintext in an otherwise-encrypted file. "
            f"Re-encrypt by hand with: dotenvx encrypt -k {key}"
        ) from exc


@dataclass(frozen=True)
class Settings:
    """Resolved configuration. Build with `Settings.load()`."""

    model: str
    deepgram_api_key: str | None
    hf_token: str | None
    openrouter_api_key: str | None
    data_dir: Path
    voices_dir: Path
    profile_dir: Path
    # Google OAuth — all three live in .env as plain environment variables,
    # never as a client-secrets JSON file or a separate token file. `env_file`
    # is kept so the token can be rewritten in place after it refreshes; see
    # brain/tools/calendar.py.
    google_client_id: str | None
    google_client_secret: str | None
    google_token_json: str | None
    # Spotify OAuth — same arrangement as Google; see brain/tools/spotify.py.
    # `spotify_device_name` is an optional tiebreaker when several devices
    # have Spotify open and none is active (a substring of its name).
    spotify_client_id: str | None
    spotify_client_secret: str | None
    spotify_token_json: str | None
    spotify_device_name: str | None
    env_file: Path

    @classmethod
    def load(cls, env_file: Path | None = None) -> Settings:
        resolved_env_file = env_file if env_file is not None else DEFAULT_ENV_FILE
        file_values = load_env_file(resolved_env_file)

        def get(name: str, default: str | None = None) -> str | None:
            # Real environment wins; the file is the fallback. Under dotenvx
            # that ordering *is* the decryption: `dotenvx run --` puts the
            # real values in the environment, so they beat the file's
            # ciphertext. Reaching the file for a value that's still
            # encrypted means dotenvx wasn't used — say so rather than
            # handing a nonsense key to an API.
            value = os.environ.get(name)
            if value:
                return value
            from_file = file_values.get(name)
            if from_file:
                return _reject_ciphertext(name, from_file)
            return default

        data_dir = Path(get("DATA_DIR") or REPO_ROOT / "data")

        return cls(
            # GLM 5.3 Flash via OpenRouter — see MASTER-PLAN.md § Running cost.
            # Override with OPENROUTER_MODEL to A/B during register tuning.
            model=get("OPENROUTER_MODEL", "z-ai/glm-5.3-flash") or "z-ai/glm-5.3-flash",
            deepgram_api_key=get("DEEPGRAM_API_KEY"),
            hf_token=get("HF_TOKEN"),
            openrouter_api_key=get("OPENROUTER_API_KEY"),
            data_dir=data_dir,
            voices_dir=Path(get("VOICES_DIR") or REPO_ROOT / "voices"),
            profile_dir=Path(get("PROFILE_DIR") or REPO_ROOT / "profile"),
            google_client_id=get("GOOGLE_CLIENT_ID"),
            google_client_secret=get("GOOGLE_CLIENT_SECRET"),
            google_token_json=get("GOOGLE_TOKEN_JSON"),
            spotify_client_id=get("SPOTIFY_CLIENT_ID"),
            spotify_client_secret=get("SPOTIFY_CLIENT_SECRET"),
            spotify_token_json=get("SPOTIFY_TOKEN_JSON"),
            spotify_device_name=get("SPOTIFY_DEVICE_NAME"),
            env_file=resolved_env_file,
        )

    def require(self, field: str) -> str:
        """Return a secret, or explain exactly how to set it.

        Used at the point of use rather than at load time, so running the corpus
        pipeline doesn't demand a Deepgram key it will never touch.
        """
        value = getattr(self, field)
        if not value:
            name = _ENV_NAMES.get(field, field.upper())
            raise RuntimeError(
                f"{name} is not set. Add `{name}=...` to {self.env_file} "
                f"or export it in your shell. See SETUP.md."
            )
        return str(value)
