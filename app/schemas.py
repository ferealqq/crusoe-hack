from pydantic import BaseModel


class SandboxRequest(BaseModel):
    code: str
    timeout: int | None = None


class SandboxResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
