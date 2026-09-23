"""The bot's own Spotify speaker: a headless librespot process it owns.

Without this, playback has to land on some device that happens to have Spotify
open — which means opening it by hand, and guessing between a browser tab, a
laptop and a phone. With it, the Brain starts a Spotify Connect device named
`PLAYER_NAME` itself, and every "play X" goes there and nowhere else. On the
Pi the same binary (packaged as raspotify) makes the robot the speaker.

**The one credential outside `.env`.** librespot signs in with its own OAuth
flow and caches the result as `credentials.json` in its own format; there is
no supported way to hand it a token from the environment. It lives in
`data/librespot/` (gitignored, directory mode 0700). Everything else Spotify
still goes through `.env`. Mint it once with:

    python -m brain.authorize_spotify --player

Zeroconf discovery is disabled on purpose: with it on, anyone on the LAN could
see "DmillsGPT" in their Spotify app and play through it on their own account.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

__all__ = [
    "CREDENTIALS_FILE",
    "PLAYER_NAME",
    "LibrespotPlayer",
    "connect_player",
    "player_cache_dir",
    "player_command",
    "secure_cache_dir",
]

PLAYER_NAME = "DmillsGPT"
CREDENTIALS_FILE = "credentials.json"

Spawn = Callable[..., Any]


def player_cache_dir(data_dir: Path) -> Path:
    return data_dir / "librespot"


def player_command(binary: str, cache_dir: Path, name: str = PLAYER_NAME) -> list[str]:
    """The librespot invocation for the bot's player."""
    return [
        binary,
        "--name", name,
        "--system-cache", str(cache_dir),
        # No audio cache: nothing to gain for a speaker that streams, and it
        # would quietly grow a pile of encrypted audio under data/.
        "--disable-audio-cache",
        "--disable-discovery",
        "--device-type", "speaker",
        "--bitrate", "320",
        "--initial-volume", "60",
    ]


class LibrespotPlayer:
    """One librespot process, started on demand and restarted if it dies."""

    def __init__(
        self,
        binary: str,
        cache_dir: Path,
        name: str = PLAYER_NAME,
        *,
        spawn: Spawn = subprocess.Popen,
    ) -> None:
        self.binary = binary
        self.cache_dir = cache_dir
        self.name = name
        self._spawn = spawn
        self._process: Any = None
        self._log: IO[str] | None = None

    @property
    def log_path(self) -> Path:
        return self.cache_dir / "librespot.log"

    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_running(self) -> None:
        """Start the player if it isn't up. Cheap to call before every command."""
        if self.running():
            return
        if self._log is None:
            # Its chatter goes to a file, not the terminal the CLI is using.
            self._log = self.log_path.open("a", encoding="utf-8")
        self._process = self._spawn(
            [*player_command(self.binary, self.cache_dir, self.name), "--quiet"],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        if self.running():
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        if self._log is not None:
            self._log.close()
            self._log = None


def connect_player(data_dir: Path) -> LibrespotPlayer | None:
    """The bot's player, or None when it isn't set up.

    None (not an error) when librespot isn't installed or hasn't been signed
    in yet: playback then falls back to whatever device has Spotify open.
    """
    binary = shutil.which("librespot")
    cache_dir = player_cache_dir(data_dir)
    if binary is None or not (cache_dir / CREDENTIALS_FILE).is_file():
        return None
    return LibrespotPlayer(binary, cache_dir)


def secure_cache_dir(cache_dir: Path) -> None:
    """Create the credentials directory readable by this user only."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_dir, 0o700)
