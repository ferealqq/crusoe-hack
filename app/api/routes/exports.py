"""Create, inspect and download trained models for completed research setups."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from fastapi.responses import FileResponse

from app.config import Setup, get_settings
from app.schemas import ModelExportOut
from app.services import model_exports

router = APIRouter(prefix="/research-runs/{run_id}/setups/{setup}/export", tags=["model exports"])


async def source(run_id: str, setup: Setup) -> model_exports.Source:
    try:
        return await asyncio.to_thread(
            model_exports.load_source, get_settings().data_dir, run_id, setup
        )
    except model_exports.ExportError as e:
        raise HTTPException(e.status_code, str(e)) from None


SourceDep = Annotated[model_exports.Source, Depends(source)]


@router.post(
    "",
    response_model=ModelExportOut,
    status_code=202,
    responses={200: {"model": ModelExportOut, "description": "Cached export is ready"}},
)
async def create_export(
    source: SourceDep, background: BackgroundTasks, response: Response
) -> ModelExportOut:
    try:
        info, job = await asyncio.to_thread(model_exports.prepare, source)
    except model_exports.ExportError as e:
        raise HTTPException(e.status_code, str(e)) from None
    if job is not None:
        background.add_task(model_exports.build, job)
    if info.status == "ready":
        response.status_code = 200
    response.headers["Location"] = f"/research-runs/{info.run_id}/setups/{info.setup}/export"
    response.headers["Cache-Control"] = "no-store"
    return info


@router.get("", response_model=ModelExportOut)
async def export_status(source: SourceDep, response: Response) -> ModelExportOut:
    response.headers["Cache-Control"] = "no-store"
    return await asyncio.to_thread(model_exports.status, source)


@router.get(
    "/download",
    response_class=FileResponse,
    responses={
        200: {"content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}}
    },
)
async def download_export(source: SourceDep) -> FileResponse:
    try:
        path, info = await asyncio.to_thread(model_exports.download, source)
    except model_exports.ExportError as e:
        raise HTTPException(e.status_code, str(e)) from None
    return FileResponse(
        path,
        filename=info.filename,
        media_type="application/zip",
        headers={"ETag": f'"{info.sha256}"', "Cache-Control": "private, no-cache"},
    )
