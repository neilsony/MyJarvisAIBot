"""Tests for the bot's own librespot player.

librespot itself is never launched — a fake spawn stands in for Popen, so
these cover when the process is (re)started and how, not whether librespot
works.
"""

import stat

from brain.spotify_player import (
    CREDENTIALS_FILE,
    PLAYER_NAME,
    LibrespotPlayer,
    connect_player,
    player_command,
    secure_cache_dir,
)


class FakeProcess:
    def __init__(self):
        self.exit_code = None
        self.terminated = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True
        self.exit_code = 0

    def wait(self, timeout=None):
        return self.exit_code

    def kill(self):
        self.exit_code = -9


class FakeSpawn:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        self.processes.append(FakeProcess())
        return self.processes[-1]


def player(tmp_path, spawn):
    return LibrespotPlayer("/bin/librespot", tmp_path, spawn=spawn)


class TestCommand:
    def test_names_the_device_and_uses_the_private_cache(self, tmp_path):
        cmd = player_command("librespot", tmp_path)
        assert cmd[cmd.index("--name") + 1] == PLAYER_NAME
        assert cmd[cmd.index("--system-cache") + 1] == str(tmp_path)

    def test_discovery_is_off(self, tmp_path):
        # Zeroconf on would let anyone on the LAN play through the bot.
        assert "--disable-discovery" in player_command("librespot", tmp_path)


class TestLifecycle:
    def test_starts_once_while_running(self, tmp_path):
        spawn = FakeSpawn()
        p = player(tmp_path, spawn)
        p.ensure_running()
        p.ensure_running()
        assert len(spawn.calls) == 1
        assert p.running()

    def test_restarts_after_the_process_dies(self, tmp_path):
        spawn = FakeSpawn()
        p = player(tmp_path, spawn)
        p.ensure_running()
        spawn.processes[0].exit_code = 1
        p.ensure_running()
        assert len(spawn.calls) == 2

    def test_stop_terminates(self, tmp_path):
        spawn = FakeSpawn()
        p = player(tmp_path, spawn)
        p.ensure_running()
        p.stop()
        assert spawn.processes[0].terminated
        assert not p.running()

    def test_stop_without_start_is_harmless(self, tmp_path):
        player(tmp_path, FakeSpawn()).stop()

    def test_runs_quiet_and_logs_to_file(self, tmp_path):
        spawn = FakeSpawn()
        p = player(tmp_path, spawn)
        p.ensure_running()
        assert "--quiet" in spawn.calls[0]
        assert p.log_path.is_file()
        p.stop()


class TestConnect:
    def test_none_without_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/bin/librespot")
        assert connect_player(tmp_path) is None

    def test_none_without_binary(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        (tmp_path / "librespot").mkdir()
        (tmp_path / "librespot" / CREDENTIALS_FILE).write_text("{}")
        assert connect_player(tmp_path) is None

    def test_player_when_installed_and_signed_in(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/bin/librespot")
        (tmp_path / "librespot").mkdir()
        (tmp_path / "librespot" / CREDENTIALS_FILE).write_text("{}")
        got = connect_player(tmp_path)
        assert got is not None and got.name == PLAYER_NAME


def test_cache_dir_is_owner_only(tmp_path):
    cache = tmp_path / "librespot"
    secure_cache_dir(cache)
    assert stat.S_IMODE(cache.stat().st_mode) == 0o700
