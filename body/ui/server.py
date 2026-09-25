"""A local web page for the voice loop: Darrick, the Body state, a talk button.

    http://127.0.0.1:8765   (started by `python -m body.voice_loop`)

Served from inside the voice loop's own process, on its event loop, because
the Body is what knows when the mic is open. The page only speaks one small
WebSocket protocol — state out as `{"state": ..., "note": ...}`, the literal
text "toggle" in — so it can later point at the Brain service unchanged.

Bound to 127.0.0.1 only: the button opens a mic, and nothing else on the
network gets to press it.

The two pictures live in `data/ui/` (gitignored — they're the show's images,
not ours to commit): `darrick.*` and `notb.*`, any of png/jpg/jpeg/webp. The
page falls back to a plain placeholder when either is missing.
"""

from __future__ import annotations

import asyncio
import webbrowser
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path

# Module level, not lazy: FastAPI resolves the handlers' type hints against
# this module's globals to tell a WebSocket parameter from a query string.
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from body.session import Controls, Press, Status

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
IMAGES = ("darrick", "notb")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
INDEX = Path(__file__).parent / "static" / "index.html"


def find_image(images_dir: Path, name: str) -> Path | None:
    if name not in IMAGES:
        return None
    for suffix in IMAGE_SUFFIXES:
        path = images_dir / f"{name}{suffix}"
        if path.is_file():
            return path
    return None


def create_app(controls: Controls, images_dir: Path) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX.read_text(encoding="utf-8"))

    @app.get("/img/{name}")
    async def image(name: str) -> FileResponse:
        path = find_image(images_dir, name)
        if path is None:
            raise HTTPException(status_code=404)
        return FileResponse(path)

    @app.websocket("/ws")
    async def socket(ws: WebSocket) -> None:
        await ws.accept()
        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue[Status] = asyncio.Queue()

        # State changes can come from another thread (tests do this), so hop
        # onto this connection's loop before touching its queue.
        def on_change(status: Status) -> None:
            loop.call_soon_threadsafe(outbox.put_nowait, status)

        unsubscribe = controls.subscribe(on_change)
        outbox.put_nowait(controls.status)

        async def send() -> None:
            while True:
                await ws.send_json((await outbox.get()).to_dict())

        sender = asyncio.create_task(send())
        try:
            while True:
                if await ws.receive_text() == Press.TOGGLE.value:
                    controls.press(Press.TOGGLE)
        except WebSocketDisconnect:
            pass
        finally:
            unsubscribe()
            sender.cancel()

    return app


@asynccontextmanager
async def running_ui(
    controls: Controls,
    images_dir: Path,
    *,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> AsyncIterator[str]:
    """Serve the page for the duration of the block. Yields its URL."""
    import uvicorn

    class _Server(uvicorn.Server):
        # uvicorn grabs Ctrl-C for itself by default, which would stop the
        # page and leave the conversation running. Leave it to the voice loop.
        @contextmanager
        def capture_signals(self) -> Iterator[None]:
            yield

    config = uvicorn.Config(
        create_app(controls, images_dir),
        host=HOST,
        port=port,
        log_level="warning",
        lifespan="off",
        timeout_graceful_shutdown=1,
    )
    server = _Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            # uvicorn exits via SystemExit when it can't bind the port.
            with suppress(SystemExit):
                task.result()
            raise RuntimeError(
                f"The UI couldn't start on port {port} (already in use?). "
                f"Run with --no-ui to skip it."
            )
        await asyncio.sleep(0.05)

    url = f"http://{HOST}:{port}"
    if open_browser:
        webbrowser.open(url)
    try:
        yield url
    finally:
        server.should_exit = True
        await task
