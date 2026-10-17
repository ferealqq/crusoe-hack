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


class Usage(BaseModel):
    prompt: int = 0
    cached: int = 0
    output: int = 0
    calls: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt=self.prompt + other.prompt,
            cached=self.cached + other.cached,
            output=self.output + other.output,
            calls=self.calls + other.calls,
        )


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # JSON text, as the model wrote it


class Reply(BaseModel):
    """One chat completion, reduced to what the agent loop needs."""

    content: str
    tool_calls: list[ToolCall] = []
    usage: Usage = Usage()
    finish_reason: str | None = None
