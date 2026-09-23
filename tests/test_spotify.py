"""Tests for the Spotify playback tools.

The Web API itself isn't exercised — these cover which device gets targeted,
what gets played, and what the model hears back, which is where the bugs
that reach a real speaker would come from.
"""

import asyncio
import json

import pytest

from brain.tools.spotify import (
    EnvTokenStore,
    build_spotify_dispatch,
    connect_spotify,
    spotify_schemas,
)

MAC = {"id": "mac-id", "name": "Neil's MacBook Pro", "is_active": False}
PHONE = {"id": "phone-id", "name": "iPhone", "is_active": True}


class FakeSpotifyError(Exception):
    """Same attributes spotipy's SpotifyException carries."""

    def __init__(self, http_status: int, msg: str):
        super().__init__(msg)
        self.http_status = http_status
        self.msg = msg


class FakeSpotifyClient:
    """Stands in for spotipy.Spotify — records playback calls, serves seeded
    devices, search results and playlists."""

    def __init__(self, devices=(), results=None, playlists=(), error=None):
        self._devices = [dict(d) for d in devices]
        self._results = results or {}
        self._playlists = list(playlists)
        self._error = error
        self.played: list[dict] = []
        self.paused = 0
        self.skipped = 0

    def devices(self):
        return {"devices": self._devices}

    def search(self, q, limit=10, offset=0, type="track", market=None):  # noqa: A002 - spotipy's spelling
        return {f"{type}s": {"items": self._results.get(type, [])}}

    def current_user_playlists(self, limit=50, offset=0):
        return {"items": self._playlists}

    def start_playback(self, device_id=None, context_uri=None, uris=None, offset=None,
                       position_ms=None):
        if self._error:
            raise self._error
        self.played.append({"device_id": device_id, "context_uri": context_uri, "uris": uris})

    def pause_playback(self, device_id=None):
        if self._error:
            raise self._error
        self.paused += 1

    def next_track(self, device_id=None):
        if self._error:
            raise self._error
        self.skipped += 1


GNX = {"name": "GNX", "uri": "spotify:track:gnx", "artists": [{"name": "Kendrick Lamar"}]}
KENDRICK = {"name": "Kendrick Lamar", "uri": "spotify:artist:kdot"}


def call(dispatch, name, args=None):
    return asyncio.run(dispatch[name](args or {}))


def dispatch_for(client, default_device=None):
    return build_spotify_dispatch(client, default_device)


class TestSchemas:
    def test_schemas_and_dispatch_agree(self):
        advertised = {s["function"]["name"] for s in spotify_schemas()}
        assert advertised == set(dispatch_for(FakeSpotifyClient()))


class TestPlay:
    def test_track_plays_uri_on_active_device(self):
        client = FakeSpotifyClient(devices=[MAC, PHONE], results={"track": [GNX]})
        got = call(dispatch_for(client), "play_music", {"query": "GNX"})
        assert client.played == [{"device_id": "phone-id", "context_uri": None,
                                  "uris": ["spotify:track:gnx"]}]
        assert "GNX by Kendrick Lamar" in got and "iPhone" in got

    def test_artist_plays_as_context(self):
        client = FakeSpotifyClient(devices=[PHONE], results={"artist": [KENDRICK]})
        call(dispatch_for(client), "play_music", {"query": "Kendrick", "kind": "artist"})
        assert client.played[0]["context_uri"] == "spotify:artist:kdot"
        assert client.played[0]["uris"] is None

    def test_own_playlist_wins_over_search(self):
        mine = {"name": "Gym", "uri": "spotify:playlist:mine", "owner": {"display_name": "Neil"}}
        theirs = {"name": "Gym Hits", "uri": "spotify:playlist:theirs"}
        client = FakeSpotifyClient(devices=[PHONE], playlists=[mine],
                                   results={"playlist": [theirs]})
        call(dispatch_for(client), "play_music", {"query": "gym", "kind": "playlist"})
        assert client.played[0]["context_uri"] == "spotify:playlist:mine"

    def test_null_playlist_search_entries_are_skipped(self):
        found = {"name": "Chill", "uri": "spotify:playlist:chill"}
        client = FakeSpotifyClient(devices=[PHONE], results={"playlist": [None, found]})
        call(dispatch_for(client), "play_music", {"query": "chill", "kind": "playlist"})
        assert client.played[0]["context_uri"] == "spotify:playlist:chill"

    def test_falls_back_to_default_device_when_none_active(self):
        client = FakeSpotifyClient(devices=[MAC], results={"track": [GNX]})
        got = call(dispatch_for(client, default_device="macbook"), "play_music", {"query": "GNX"})
        assert client.played[0]["device_id"] == "mac-id"
        assert "MacBook" in got

    def test_no_device_at_all_plays_nothing(self):
        client = FakeSpotifyClient(devices=[], results={"track": [GNX]})
        got = call(dispatch_for(client), "play_music", {"query": "GNX"})
        assert client.played == []
        assert "Nothing's running Spotify" in got

    def test_uses_any_open_device_without_a_default(self):
        web = {"id": "web-id", "name": "Web Player (Chrome)", "is_active": False}
        client = FakeSpotifyClient(devices=[web], results={"track": [GNX]})
        got = call(dispatch_for(client), "play_music", {"query": "GNX"})
        assert client.played[0]["device_id"] == "web-id"
        assert "Web Player (Chrome)" in got

    def test_unmatched_default_still_falls_through_to_an_open_device(self):
        web = {"id": "web-id", "name": "Web Player (Chrome)", "is_active": False}
        client = FakeSpotifyClient(devices=[web], results={"track": [GNX]})
        call(dispatch_for(client, default_device="MacBook"), "play_music", {"query": "GNX"})
        assert client.played[0]["device_id"] == "web-id"

    def test_default_breaks_ties_between_open_devices(self):
        web = {"id": "web-id", "name": "Web Player (Chrome)", "is_active": False}
        client = FakeSpotifyClient(devices=[web, MAC], results={"track": [GNX]})
        call(dispatch_for(client, default_device="macbook"), "play_music", {"query": "GNX"})
        assert client.played[0]["device_id"] == "mac-id"

    def test_restricted_devices_are_never_targeted(self):
        tv = {"id": "tv-id", "name": "Living Room TV", "is_active": False, "is_restricted": True}
        client = FakeSpotifyClient(devices=[tv], results={"track": [GNX]})
        got = call(dispatch_for(client), "play_music", {"query": "GNX"})
        assert client.played == [] and "Nothing's running Spotify" in got

    def test_nothing_found(self):
        client = FakeSpotifyClient(devices=[PHONE])
        got = call(dispatch_for(client), "play_music", {"query": "zzzz"})
        assert "Couldn't find" in got and client.played == []

    def test_empty_query(self):
        assert "No song" in call(dispatch_for(FakeSpotifyClient()), "play_music", {"query": " "})

    def test_unknown_kind(self):
        got = call(dispatch_for(FakeSpotifyClient()), "play_music",
                   {"query": "x", "kind": "podcast"})
        assert "Unknown kind" in got


class FakePlayer:
    """Stands in for LibrespotPlayer: a name, and a count of start requests."""

    def __init__(self, name="DmillsGPT"):
        self.name = name
        self.starts = 0

    def ensure_running(self):
        self.starts += 1


class SlowDevices(FakeSpotifyClient):
    """The bot's player only appears after a few device-list polls."""

    def __init__(self, appears_after, **kwargs):
        super().__init__(**kwargs)
        self._appears_after = appears_after
        self._polls = 0

    def devices(self):
        self._polls += 1
        found = self._polls > self._appears_after
        extra = [{"id": "bot-id", "name": "DmillsGPT", "is_active": False}] if found else []
        return {"devices": self._devices + extra}


class TestOwnPlayer:
    def test_plays_on_own_player_even_when_another_device_is_active(self):
        bot = {"id": "bot-id", "name": "DmillsGPT", "is_active": False}
        client = FakeSpotifyClient(devices=[PHONE, bot], results={"track": [GNX]})
        p = FakePlayer()
        got = call(build_spotify_dispatch(client, None, p), "play_music", {"query": "GNX"})
        assert client.played[0]["device_id"] == "bot-id"
        assert p.starts == 1 and "DmillsGPT" in got

    def test_waits_for_the_player_to_register(self):
        client = SlowDevices(appears_after=3, devices=[PHONE], results={"track": [GNX]})
        naps: list[float] = []
        dispatch = build_spotify_dispatch(client, None, FakePlayer(), sleep=naps.append)
        call(dispatch, "play_music", {"query": "GNX"})
        assert client.played[0]["device_id"] == "bot-id"
        assert len(naps) == 3

    def test_gives_up_with_a_spoken_reason(self):
        client = SlowDevices(appears_after=10_000, devices=[PHONE], results={"track": [GNX]})
        dispatch = build_spotify_dispatch(
            client, None, FakePlayer(), sleep=lambda _s: None, wait_seconds=2.0
        )
        got = call(dispatch, "play_music", {"query": "GNX"})
        assert client.played == []
        assert "didn't come up" in got

    def test_resume_goes_to_own_player(self):
        bot = {"id": "bot-id", "name": "DmillsGPT", "is_active": False}
        client = FakeSpotifyClient(devices=[PHONE, bot])
        call(build_spotify_dispatch(client, None, FakePlayer()), "resume_music")
        assert client.played[0]["device_id"] == "bot-id"


class TestControls:
    def test_pause(self):
        client = FakeSpotifyClient()
        assert call(dispatch_for(client), "pause_music") == "Paused."
        assert client.paused == 1

    def test_skip(self):
        client = FakeSpotifyClient()
        assert call(dispatch_for(client), "skip_track") == "Skipped."
        assert client.skipped == 1

    def test_resume_targets_device(self):
        client = FakeSpotifyClient(devices=[PHONE])
        got = call(dispatch_for(client), "resume_music")
        assert client.played == [{"device_id": "phone-id", "context_uri": None, "uris": None}]
        assert "iPhone" in got


class TestErrors:
    def test_premium_required_is_spoken_not_raised(self):
        client = FakeSpotifyClient(devices=[PHONE], results={"track": [GNX]},
                                   error=FakeSpotifyError(403, "PREMIUM_REQUIRED"))
        got = call(dispatch_for(client), "play_music", {"query": "GNX"})
        assert "Premium" in got

    def test_no_active_device_on_pause(self):
        client = FakeSpotifyClient(error=FakeSpotifyError(404, "No active device found"))
        assert "Nothing's playing" in call(dispatch_for(client), "pause_music")

    def test_other_errors_pass_the_message_through(self):
        client = FakeSpotifyClient(error=FakeSpotifyError(429, "Too many requests"))
        assert "Too many requests" in call(dispatch_for(client), "skip_track")


class TestConnect:
    def test_none_without_client_credentials(self, tmp_path):
        assert connect_spotify(None, None, None, tmp_path / ".env") is None
        assert connect_spotify("id", None, None, tmp_path / ".env") is None

    def test_raises_without_token(self, tmp_path):
        with pytest.raises(RuntimeError, match="authorize_spotify"):
            connect_spotify("id", "secret", None, tmp_path / ".env")


class TestEnvTokenStore:
    def test_seeds_from_json(self, tmp_path):
        store = EnvTokenStore(tmp_path / ".env", '{"access_token": "a"}')
        assert store.get() == {"access_token": "a"}

    def test_empty_without_token(self, tmp_path):
        assert EnvTokenStore(tmp_path / ".env", None).get() is None

    def test_save_persists_to_env_and_keeps_other_lines(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("# keep me\nHF_TOKEN=hf_x\nSPOTIFY_TOKEN_JSON='{}'\n", encoding="utf-8")
        store = EnvTokenStore(env, "{}")
        token = {"access_token": "new", "refresh_token": "rotated", "expires_at": 1}

        store.save(token)

        assert store.get() == token
        text = env.read_text(encoding="utf-8")
        assert "# keep me" in text and "HF_TOKEN=hf_x" in text
        line = next(ln for ln in text.splitlines() if ln.startswith("SPOTIFY_TOKEN_JSON="))
        assert json.loads(line.split("=", 1)[1].strip("'")) == token
