"""Tests for the voice loop's web page: images, the state feed, the talk button."""

import time

from fastapi.testclient import TestClient

from body.session import BodyState, Controls, Press
from body.ui.server import create_app, find_image


def client(tmp_path, controls=None):
    return TestClient(create_app(controls or Controls(), tmp_path))


class TestImages:
    def test_serves_a_present_image(self, tmp_path):
        (tmp_path / "darrick.jpg").write_bytes(b"jpeg")
        response = client(tmp_path).get("/img/darrick")
        assert response.status_code == 200
        assert response.content == b"jpeg"

    def test_missing_image_is_404_so_the_page_falls_back(self, tmp_path):
        assert client(tmp_path).get("/img/notb").status_code == 404

    def test_only_the_two_named_images_are_served(self, tmp_path):
        (tmp_path / "secret.png").write_bytes(b"x")
        assert find_image(tmp_path, "secret") is None
        assert find_image(tmp_path, "../ui/secret") is None


def test_index_page(tmp_path):
    response = client(tmp_path).get("/")
    assert response.status_code == 200
    assert "MillsGPT" in response.text


class TestSocket:
    def test_sends_the_current_state_on_connect(self, tmp_path):
        controls = Controls()
        controls.set(BodyState.THINKING)
        with client(tmp_path, controls).websocket_connect("/ws") as ws:
            assert ws.receive_json() == {"state": "thinking", "note": ""}

    def test_pushes_state_changes(self, tmp_path):
        controls = Controls()
        with client(tmp_path, controls).websocket_connect("/ws") as ws:
            ws.receive_json()
            controls.set(BodyState.IDLE, "didn't catch that")
            assert ws.receive_json() == {"state": "idle", "note": "didn't catch that"}

    def test_toggle_presses_the_button(self, tmp_path):
        controls = Controls()
        pressed = []
        controls.press = lambda press: pressed.append(press) or True
        with client(tmp_path, controls).websocket_connect("/ws") as ws:
            ws.receive_json()
            # Messages are handled in order, so once the toggle lands the
            # junk before it has already been seen and ignored.
            ws.send_text("anything else")
            ws.send_text("toggle")
            deadline = time.monotonic() + 2
            while not pressed and time.monotonic() < deadline:
                time.sleep(0.01)
        assert pressed == [Press.TOGGLE]
