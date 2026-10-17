from typing import Literal

from pydantic import BaseModel


class SandboxRequest(BaseModel):
    code: str
    timeout: int | None = None


class SandboxResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int


class TrainResult(BaseModel):
    """One train.py run in the sandbox. `preds` is set only when status is "ok"."""

    status: Literal["ok", "error", "timeout"]
    preds: list[float] | None = None
    output_tail: str = ""
    runtime_s: float = 0.0


class Experiment(BaseModel):
    """One sandbox run, as stored in experiments.jsonl."""

    run_id: str
    generation: int
    agent: str
    seed: int
    split: Literal["val", "test"] = "val"
    status: Literal["ok", "error", "timeout"]
    loss: float | None = None
    runtime_s: float
    note: str = ""
    output_tail: str = ""
    code: str
