import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import brain, datasets, health, playground, runs

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="crusoe-hack",
    description="LLM improvement loop: Crusoe inference, Modal sandboxes, Neon Postgres + storage",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(runs.router)
app.include_router(playground.router)
app.include_router(brain.router)

# Demo frontend: serve the built web/ app (cd web && npm run build) with an SPA fallback.
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> FileResponse:
        file = WEB_DIST / path
        if path and file.is_file() and WEB_DIST in file.resolve().parents:
            return FileResponse(file)
        return FileResponse(WEB_DIST / "index.html")
