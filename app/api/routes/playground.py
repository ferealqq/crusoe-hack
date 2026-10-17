"""Setup checks: does the Crusoe key work, and can Modal run code?"""

from fastapi import APIRouter

from app.schemas import SandboxRequest, SandboxResult
from app.services import inference, sandbox

router = APIRouter(prefix="/playground", tags=["playground"])


@router.get("/models")
async def models() -> list[str]:
    return await inference.list_models()


@router.post("/sandbox", response_model=SandboxResult)
async def run_sandbox(body: SandboxRequest):
    return await sandbox.run_python(body.code, body.timeout)
