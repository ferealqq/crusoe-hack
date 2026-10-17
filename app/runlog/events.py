"""Event contract between the orchestrator and the demo frontend.

Mirrored in web/src/lib/events.ts - change both together. Every event carries `seq`
(0, 1, 2, ... per run) and `ts` (seconds since the run started); `RunLogger` fills them in.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

Setup = Literal["shared_brain", "raw_log", "no_memory", "one_shot"]
Role = Literal["researcher", "curator"]
Tool = Literal["get_memory", "upsert_memory", "remove_memory", "run_code", "web_search"]
Section = Literal["works", "failed", "try_next"]


class _Event(BaseModel):
    seq: int = 0
    ts: float = 0.0


class ResearcherSpec(BaseModel):
    focus: str  # optimizer | architecture | regularization | data
    model: str


class RunConfig(BaseModel):
    generations: int
    researchers: list[ResearcherSpec]
    curator_model: str
    budget: str  # human-readable, e.g. "300 steps per run_code"
    seeds: int
    memory_cap_tokens: int
    dataset: str


class RunStarted(_Event):
    t: Literal["run.started"] = "run.started"
    run_id: str
    setup: Setup
    config: RunConfig
    baseline_val_loss: float  # L0 on val


class GenStarted(_Event):
    t: Literal["gen.started"] = "gen.started"
    gen: int
    start_best_val_loss: float


class AgentSpawned(_Event):
    t: Literal["agent.spawned"] = "agent.spawned"
    gen: int
    agent_id: str
    role: Role
    focus: str | None = None
    model: str


class ToolCalled(_Event):
    t: Literal["tool.called"] = "tool.called"
    agent_id: str
    call_id: str
    tool: Tool
    summary: str  # one line for the feed, e.g. 'web_search("lion optimizer small LMs")'


class ToolResult(_Event):
    t: Literal["tool.result"] = "tool.result"
    agent_id: str
    call_id: str
    tool: Tool
    ok: bool = True
    summary: str
    val_loss: float | None = None  # run_code only
    duration_ms: int | None = None
    detail: str | None = None  # full output / train.py diff, shown in the trace drawer


class LlmUsage(_Event):
    t: Literal["llm.usage"] = "llm.usage"
    agent_id: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    ttft_ms: int


class AgentMessage(_Event):
    t: Literal["agent.message"] = "agent.message"
    agent_id: str
    text: str  # assistant reasoning/prose, shown in the trace drawer


class AgentReported(_Event):
    t: Literal["agent.reported"] = "agent.reported"
    gen: int
    agent_id: str
    best_val_loss: float
    report: str


class RankEntry(BaseModel):
    agent_id: str
    best_val_loss: float
    gain: float  # vs the generation's starting best
    bucket: Literal["top", "middle", "bottom"]
    tie: bool = False  # within seed noise of the next rank


class CuratorRanked(_Event):
    t: Literal["curator.ranked"] = "curator.ranked"
    gen: int
    ranking: list[RankEntry]


class CuratorVerified(_Event):
    t: Literal["curator.verified"] = "curator.verified"
    gen: int
    agent_id: str
    claim: str
    claimed_val_loss: float
    rerun_val_loss: float
    held: bool


class MemoryOp(BaseModel):
    op: Literal["add", "edit", "remove"]
    section: Section
    entry_id: str
    text: str
    reason: str
    proof: str | None = None  # e.g. "run gen2/optimizer#2: val 1.94"


class BestSoFar(BaseModel):
    val_loss: float
    summary: str


class MemoryUpdated(_Event):
    t: Literal["memory.updated"] = "memory.updated"
    gen: int
    version: int
    ops: list[MemoryOp]
    best: BestSoFar
    size_tokens: int
    cap_tokens: int


class GenFinished(_Event):
    t: Literal["gen.finished"] = "gen.finished"
    gen: int
    best_val_loss: float
    waste_runs: int  # runs repeating a known dead end
    total_runs: int
    memory_tokens: int
    raw_log_tokens: int  # what an uncurated log would be by now
    prompt_tokens: int
    cached_tokens: int
    mean_ttft_ms: int


class SeedScore(BaseModel):
    seed: int
    test_l0: float
    test_lbest: float
    gain: float


class EvalFinished(_Event):
    t: Literal["eval.finished"] = "eval.finished"
    setup: Setup
    seeds: list[SeedScore]
    fitness: float  # mean gain across seeds, on the test split


class RunFinished(_Event):
    t: Literal["run.finished"] = "run.finished"
    status: Literal["completed", "failed"] = "completed"
    error: str | None = None


Event = Annotated[
    RunStarted
    | GenStarted
    | AgentSpawned
    | ToolCalled
    | ToolResult
    | LlmUsage
    | AgentMessage
    | AgentReported
    | CuratorRanked
    | CuratorVerified
    | MemoryUpdated
    | GenFinished
    | EvalFinished
    | RunFinished,
    Field(discriminator="t"),
]

event_adapter: TypeAdapter[Event] = TypeAdapter(Event)
