"""Rebuild the demo's event stream (web/src/lib/events.ts) from the research run files in data/.

Research runs write JSONL files, not events (README "Storage"). The UI replays events, so they
are derived here:
- Order comes from the files: each agent's transcript (exact), experiments, generations.
- Times come from the git trail (`lab.checkpoint` commits: baseline, every memory change, best
  changes, generation ends) plus sandbox runtimes in between. Times inside a generation are
  estimates; the order of steps inside one agent is exact.
- Memory states come from `memory.jsonl` at each commit. Without git, the final file is used.

Setups started together (within GROUP_WINDOW_S) form one demo run, e.g. `churn-20261017-223053`.
"""

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from app import scoring
from app.memory import Entry, Memory, count_tokens
from app.store import DATA_DIR, read_jsonl

GROUP_WINDOW_S = 120
STALL_AFTER_S = 600  # no file changes for this long: a run in progress is "stalled"
THINK_S = 6.0  # nominal LLM turn, before scaling to the generation's real length
TOOL_S = 0.5
DEFAULT_SCALE = 1.5  # for a generation still in progress (no end commit yet)
SETUP_ORDER = ("shared_brain", "raw_log", "no_memory", "one_shot")
RUN_ID = re.compile(
    r"^(?P<dataset>.+)-(?P<setup>shared_brain|raw_log|no_memory|one_shot)-(?P<ts>\d{8}-\d{6})$"
)
SMALL_RESULT_KEYS = 8


# ---- loading ----


@dataclass
class Commit:
    sha: str
    at: datetime
    message: str  # without the "chore(brain): " prefix and the " [run_id]" suffix


@dataclass
class SetupRun:
    run_id: str
    dataset: str
    setup: str
    dir: Path
    started: datetime
    row: dict[str, Any] | None  # research_runs.jsonl row, once finished
    experiments: list[dict[str, Any]]
    agents: list[dict[str, Any]]
    generations: list[dict[str, Any]]
    memory: list[dict[str, Any]]
    commits: list[Commit]
    mtime: float
    _memory_at: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def config(self) -> dict[str, Any]:
        from app.config import ResearchConfig

        if self.row:
            saved = self.row["config"]
        elif (self.dir / "config.json").exists():  # written at start (Lab.save_config)
            saved = json.loads((self.dir / "config.json").read_text())
        else:
            saved = {}
        return ResearchConfig(**saved).model_dump()  # older runs lack newer fields, e.g. goal

    def memory_at(self, sha: str | None) -> list[dict[str, Any]]:
        """memory.jsonl as of a commit (the final file if there's no commit or no git)."""
        if sha is None:
            return self.memory
        if sha not in self._memory_at:
            out = _git(self.dir, "show", f"{sha}:./memory.jsonl")
            rows = (
                [json.loads(line) for line in out.splitlines() if line.strip()]
                if out is not None
                else self.memory
            )
            self._memory_at[sha] = rows
        return self._memory_at[sha]


def _git(cwd: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False, timeout=20
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def _commits(run_dir: Path, run_id: str) -> list[Commit]:
    out = _git(run_dir, "log", "--reverse", "--fixed-strings", f"--grep=[{run_id}]",
               "--format=%h%x1f%aI%x1f%s")  # fmt: skip
    commits = []
    for line in (out or "").splitlines():
        sha, at, subject = line.split("\x1f", 2)
        message = subject.removeprefix("chore(brain): ").removesuffix(f" [{run_id}]")
        commits.append(Commit(sha, datetime.fromisoformat(at).astimezone(UTC), message))
    return commits


def load_setup_runs(root: Path | None = None) -> list[SetupRun]:
    root = root or DATA_DIR
    rows = {r["run_id"]: r for r in read_jsonl(root / "research_runs.jsonl")}
    runs = []
    for d in sorted((root / "research_runs").glob("*")):
        m = RUN_ID.match(d.name)
        if not d.is_dir() or not m:
            continue
        files = [p for p in d.iterdir() if p.is_file()]
        runs.append(
            SetupRun(
                run_id=d.name,
                dataset=m["dataset"],
                setup=m["setup"],
                dir=d,
                started=datetime.strptime(m["ts"], "%Y%m%d-%H%M%S").replace(tzinfo=UTC),
                row=rows.get(d.name),
                experiments=read_jsonl(d / "experiments.jsonl"),
                agents=read_jsonl(d / "agents.jsonl"),
                generations=read_jsonl(d / "generations.jsonl"),
                memory=read_jsonl(d / "memory.jsonl"),
                commits=_commits(d, d.name),
                mtime=max((p.stat().st_mtime for p in files), default=0.0),
            )
        )
    return runs


def group_runs(runs: list[SetupRun]) -> dict[str, list[SetupRun]]:
    """Setups of one dataset started within GROUP_WINDOW_S of each other form one demo run."""
    groups: dict[str, list[SetupRun]] = {}
    for run in sorted(runs, key=lambda r: r.started):
        for gid, members in groups.items():
            first = members[0]
            if (
                first.dataset == run.dataset
                and run.setup not in {m.setup for m in members}
                and (run.started - first.started).total_seconds() <= GROUP_WINDOW_S
            ):
                members.append(run)
                break
        else:
            groups[f"{run.dataset}-{run.started:%Y%m%d-%H%M%S}"] = [run]
    for members in groups.values():
        members.sort(key=lambda r: SETUP_ORDER.index(r.setup))
    return groups


# ---- event building ----


class Timeline:
    """Collects events with absolute times; `finish` sorts them and numbers them from 1."""

    def __init__(self, run_id: str, t0: datetime) -> None:
        self.run_id = run_id
        self.t0 = t0
        self.items: list[tuple[datetime, int, dict[str, Any]]] = []

    def emit(self, at: datetime, type_: str, data: dict[str, Any] | None = None, **env: Any):
        self.items.append((at, len(self.items), {"type": type_, "data": data or {}, **env}))

    def finish(self) -> list[dict[str, Any]]:
        out = []
        for seq, (at, _, e) in enumerate(sorted(self.items, key=lambda x: (x[0], x[1])), 1):
            out.append(
                {
                    "version": 1,
                    "seq": seq,
                    "run_id": self.run_id,
                    "timestamp": at.isoformat(),
                    "elapsed_ms": max(0, round((at - self.t0).total_seconds() * 1000)),
                    "type": e["type"],
                    "setup": e.get("setup"),
                    "gen": e.get("gen"),
                    "agent": e.get("agent"),
                    "role": e.get("role"),
                    "phase": e.get("phase"),
                    "experiment_id": e.get("experiment_id"),
                    "seed": e.get("seed"),
                    "turn": e.get("turn"),
                    "model": e.get("model"),
                    "data": e["data"],
                }
            )
        return out


def _secs(s: float) -> timedelta:
    return timedelta(seconds=s)


def _cap(t: datetime, limit: datetime | None) -> datetime:
    return min(t, limit) if limit else t


_majority: dict[Path, float | None] = {}


def _majority_accuracy(dataset: str) -> float | None:
    """Accuracy of always predicting the most common test label (None without the data)."""
    folder = DATA_DIR / dataset
    if folder not in _majority:
        try:
            target = json.loads((folder / "meta.json").read_text())["target"]
            y = pd.read_parquet(folder / "test_y.parquet")[target].to_numpy()
            _majority[folder] = scoring.majority_accuracy(y)
        except (OSError, KeyError, ValueError):
            _majority[folder] = None
    return _majority[folder]


def _memory_payload(run: SetupRun, rows: list[dict[str, Any]]) -> tuple[str, int]:
    mem = Memory(run.dir / "__no_file__", run.config["memory_cap_tokens"])
    mem.entries = [Entry(**r) for r in rows]
    text = mem.render()
    return text, count_tokens(text)


def _entry(row: dict[str, Any] | None) -> dict[str, Any] | None:
    return {**row, "active": True} if row else None


def _small(result: Any) -> dict[str, Any]:
    """Tool results without code, output or long text, for the trace."""
    if not isinstance(result, dict):
        return {"chars": len(str(result))}
    out: dict[str, Any] = {}
    for k, v in result.items():
        if k in ("code", "output", "output_tail", "errors") or len(out) >= SMALL_RESULT_KEYS:
            continue
        if isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, str) and len(v) <= 120:
            out[k] = v
        elif (
            isinstance(v, list) and len(v) <= 6 and all(isinstance(x, (int, float, str)) for x in v)
        ):
            out[k] = v
    return out


def _parse(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


@dataclass
class Step:
    """One thing an agent did, in transcript order, with a nominal duration."""

    kind: str  # turn | tool
    duration: float
    turn: int
    text: str = ""
    finish_reason: str | None = None
    tool: str = ""
    call_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None


def _steps(agent: dict[str, Any], experiments: dict[str, dict[str, Any]]) -> list[Step]:
    results = {
        m.get("tool_call_id"): _parse(m.get("content"))
        for m in agent.get("transcript", [])
        if m.get("role") == "tool"
    }
    steps: list[Step] = []
    turn = 0
    for m in agent.get("transcript", []):
        if m.get("role") != "assistant":
            continue
        turn += 1
        calls = m.get("tool_calls") or []
        steps.append(
            Step(
                "turn",
                THINK_S,
                turn,
                text=str(m.get("content") or "").strip(),
                finish_reason="tool_calls" if calls else "stop",
            )  # fmt: skip
        )
        for tc in calls:
            fn = tc.get("function", {})
            args = _parse(fn.get("arguments") or "{}")
            result = results.get(tc.get("id"))
            duration = TOOL_S
            if fn.get("name") == "run_code" and isinstance(result, dict):
                ids = result.get("run_ids") or [result.get("run_id")]
                runtimes = [experiments[i]["runtime_s"] for i in ids if i in experiments]
                duration += max(runtimes, default=0.0)  # verification seeds run in parallel
            steps.append(
                Step(
                    "tool",
                    duration,
                    turn,
                    tool=fn.get("name", ""),
                    call_id=tc.get("id", ""),
                    args=args if isinstance(args, dict) else {},
                    result=result,
                )  # fmt: skip
            )
    return steps


class SetupBuilder:
    def __init__(self, tl: Timeline, run: SetupRun) -> None:
        self.tl = tl
        self.run = run
        self.cfg = run.config
        self.exps = {e["run_id"]: e for e in run.experiments}
        self.gens = {g["generation"]: g for g in run.generations}
        self.scale = DEFAULT_SCALE
        self.best_version = 0
        self.l0 = run.row["l0"] if run.row else None
        self.threshold = run.row["noise_threshold"] if run.row else None
        self.models = {a["role"]: a.get("model") for a in run.agents}

    def commit(self, pattern: str) -> Commit | None:
        rx = re.compile(pattern)
        return next((c for c in self.run.commits if rx.match(c.message)), None)

    def emit(self, at: datetime, type_: str, data: dict[str, Any] | None = None, **env: Any):
        self.tl.emit(at, type_, data, setup=self.run.setup, **env)

    # -- baseline --

    def baseline(self) -> tuple[datetime, dict[str, Any] | None]:
        base = [e for e in self.run.experiments if e["generation"] == 0]
        losses = [e["loss"] for e in base if e["loss"] is not None]
        start = self.commit(r"^start:")
        at = (
            start.at
            if start
            else self.run.started + _secs(max((e["runtime_s"] for e in base), default=0) + 5)
        )
        if self.l0 is None and losses:
            self.l0 = sum(losses) / len(losses)
        if (
            self.threshold is None
            and start
            and (m := re.search(r"noise threshold ([\d.]+)", start.message))
        ):
            self.threshold = float(m[1])
        if not losses:
            return at, None
        return at, {
            "l0": self.l0,
            "threshold": self.threshold,
            "val_losses": losses,
            "test_losses": (self.run.row or {}).get("test_start") or [],
        }

    # -- generations --

    def build(self, t_baseline: datetime) -> None:
        run = self.run
        self.emit(run.started, "setup.started", {"generations": self.cfg["generations"]})
        gen_start = t_baseline
        prev_best = self.l0
        last_gen = max([e["generation"] for e in run.experiments if e["split"] == "val"] + [0])
        for g in range(1, self.cfg["generations"] + 1):
            if g > last_gen and g not in self.gens:
                break
            end = self.commit(rf"^g{g} done:")
            gen_end = end.at if end else self.estimated_end(g, gen_start)
            self.generation(g, gen_start, gen_end, prev_best)
            row = self.gens.get(g)
            if row:
                prev_best = row["end_best"]
                self.best_version = row["best_version"]
            if gen_end is None:
                return
            gen_start = gen_end
        self.final(gen_start)

    def estimated_end(self, g: int, gs: datetime) -> datetime | None:
        """No git trail (e.g. a copy of data/): a logged generation ends after its agents' work."""
        if g not in self.gens:
            return None
        agents = [a for a in self.run.agents if a["generation"] == g]
        work = [
            sum(s.duration for s in _steps(a, self.exps))
            for a in agents
            if a["role"] == "researcher"
        ]
        curator = sum(
            sum(s.duration for s in _steps(a, self.exps)) for a in agents if a["role"] == "curator"
        )
        return gs + _secs((max(work, default=THINK_S) + curator) * DEFAULT_SCALE + 2)

    def generation(self, g: int, gs: datetime, ge: datetime | None, start_best: float | None):
        run, row = self.run, self.gens.get(g)
        self.emit(gs, "generation.started", {"best_val": start_best, "best_version": self.best_version,
                                              "researchers": self.cfg["researchers"]}, gen=g)  # fmt: skip
        shared = run.setup == "shared_brain"
        if shared:
            snap = [c for c in run.commits if c.at <= gs]
            text, tokens = _memory_payload(run, run.memory_at(snap[-1].sha if snap else None))
            self.emit(gs, "memory.updated", {"action": "snapshot", "text": text, "tokens": tokens,
                                              "best_version": self.best_version}, gen=g)  # fmt: skip

        researchers = [a for a in run.agents if a["generation"] == g and a["role"] == "researcher"]
        curator = next(
            (a for a in run.agents if a["generation"] == g and a["role"] == "curator"), None
        )
        curator_commits = [c for c in run.commits if re.match(rf"^g{g} curator ", c.message)]

        # Researchers fill [gs, research_end] and the curator the rest of the generation, split by
        # their nominal work; the curator's memory edits are pinned to their commit times.
        plans = {a["agent"]: _steps(a, self.exps) for a in researchers}
        longest = max((sum(s.duration for s in p) for p in plans.values()), default=0.0)
        cur_steps = _steps(curator, self.exps) if shared and curator else []
        cur_raw = sum(s.duration for s in cur_steps)
        research_end = cur_scale = None
        if ge is not None:
            window = (ge - gs).total_seconds()
            research_end = (
                gs + _secs(window * longest / (longest + cur_raw)) if longest + cur_raw else ge
            )
            if curator_commits:
                research_end = min(research_end, curator_commits[0].at - _secs(1))
            research_end = max(research_end, gs)
            if longest:
                self.scale = (research_end - gs).total_seconds() / longest
            cur_scale = (ge - research_end).total_seconds() / cur_raw if cur_raw else self.scale
        for i, a in enumerate(sorted(researchers, key=lambda a: a["agent"])):
            self.agent(a, plans[a["agent"]], gs + _secs(0.3 * i), self.scale, limit=research_end)
        self.unfinished_researchers(g, gs, {a["agent"] for a in researchers})

        if shared:
            cs = research_end or gs
            if curator:
                self.agent(curator, cur_steps, cs, cur_scale or self.scale,
                                     commits=curator_commits, limit=ge)  # fmt: skip
            else:
                self.unfinished_curator(g, cs, curator_commits)
        if row is None or ge is None:
            return

        best = self.commit(rf"^g{g} best -> v")
        if row["best_changed"]:
            at = best.at if best else ge - _secs(1)
            note = self.exps.get(row["best_run_id"], {}).get("note", "")
            self.emit(at, "best.updated", {"val_loss": row["end_best"], "version": row["best_version"],
                                            "run_id": row["best_run_id"], "note": note, "verified": shared,
                                            "val_gain": row["val_gain"]}, gen=g)  # fmt: skip
            if shared and best:
                text, tokens = _memory_payload(run, run.memory_at(best.sha))
                self.emit(best.at, "memory.updated", {"action": "snapshot", "text": text, "tokens": tokens,
                                                       "best_version": row["best_version"]}, gen=g)  # fmt: skip
        if shared:
            end = self.commit(rf"^g{g} done:")
            text, tokens = _memory_payload(run, run.memory_at(end.sha if end else None))
            self.emit(ge, "memory.updated", {"action": "curation_completed", "text": text,
                      "tokens": tokens, "best_version": row["best_version"], "dropped_ids": row["dropped"]},
                      gen=g, agent="curator", role="curator", phase="curation")  # fmt: skip
        self.emit(
            ge, "generation.completed", self.metrics(row, gs, ge, researchers, curator), gen=g
        )

    def metrics(self, row, gs, ge, researchers, curator) -> dict[str, Any]:
        g = row["generation"]
        exps = [e for e in self.run.experiments if e["generation"] == g]
        focus = {a["agent"]: a.get("focus") for a in researchers}
        best_by_agent = []
        for rank, r in enumerate(row.get("ranking", []), 1):
            e = self.exps.get(r.get("best_run_id") or "")
            best_by_agent.append({
                "name": r["agent"],
                "focus": focus.get(r["agent"], ""),
                "best": {"run_id": e["run_id"], "note": e["note"], "status": e["status"], "val_loss": e["loss"]} if e else None,
                "rank": rank,
            })  # fmt: skip

        def usage(agents):
            keys = ("prompt", "cached", "output", "calls")
            return {k: sum(a.get("usage", {}).get(k, 0) for a in agents) for k in keys}

        return {
            "gen": g,
            "start_best_val": row["start_best"],
            "end_best_val": row["end_best"],
            "best_version": row["best_version"],
            "new_best": row["best_changed"],
            "best_run_id": row["best_run_id"],
            "best_note": self.exps.get(row["best_run_id"], {}).get("note", ""),
            "val_gain": row["val_gain"],
            "test_losses": [],
            "test_gain": None,
            "memory_tokens": row.get("memory_tokens_after", 0),
            "runs": row["runs"],
            "errors": sum(e["status"] == "error" for e in exps),
            "timeouts": sum(e["status"] == "timeout" for e in exps),
            "researcher_best": best_by_agent,
            "usage": {
                "researchers": usage(researchers),
                "curator": usage([curator]) if curator else {},
            },
            "minutes": round((ge - gs).total_seconds() / 60, 2),
        }

    # -- agents --

    def agent(
        self, a, steps: list[Step], t: datetime, scale: float, commits=None, limit=None
    ) -> datetime:
        """Emit one finished agent's steps from `t`; returns when it ended."""
        role, name, g = a["role"], a["agent"], a["generation"]
        phase = "curation" if role == "curator" else "candidate"
        env = {"gen": g, "agent": name, "role": role, "phase": phase, "model": a.get("model")}
        pending = list(commits or [])
        budget = (self.cfg["curator_runs"] if role == "curator"
                  else self.cfg["researcher_runs"] - (-(-self.cfg["curator_runs"] // self.cfg["researchers"])
                                                     if self.run.setup == "shared_brain" else 0))  # fmt: skip
        self.emit(t, "agent.started", {"max_turns": self.cfg["curator_turns" if role == "curator" else "researcher_turns"],
                                       "run_budget": budget, "focus": a.get("focus"),
                                       "best_version": self.best_version}, **env)  # fmt: skip
        turns = max(1, sum(s.kind == "turn" for s in steps))
        total = a.get("usage", {})
        share = {k: total.get(k, 0) // turns for k in ("prompt", "cached", "output")}
        prev_turn: Step | None = None
        for s in steps:
            if s.kind == "turn":
                if prev_turn:
                    self.emit(t, "agent.turn.completed", {"usage": share, "finish_reason": prev_turn.finish_reason},
                              turn=prev_turn.turn, **env)  # fmt: skip
                self.emit(t, "agent.turn.started", {}, turn=s.turn, **env)
                t = _cap(t + _secs(s.duration * scale), limit)
                prev_turn = s
                if s.text and len(steps) > 1:
                    self.emit(t, "agent.highlight", {"source": "tool_intent", "tool": "run_code", "text": s.text[:240]},
                              turn=s.turn, **env)  # fmt: skip
                continue
            t = _cap(self.tool(s, t, scale, env, pending, limit), limit)
        if prev_turn:
            self.emit(t, "agent.turn.completed", {"usage": share, "finish_reason": prev_turn.finish_reason},
                      turn=prev_turn.turn, **env)  # fmt: skip
        self.emit(t, "agent.completed", {"report": a.get("report", ""), "usage": {**share, **total},
                                         "turns": a.get("turns", turns)}, **env)  # fmt: skip
        return t

    def tool(self, s: Step, t: datetime, scale: float, env: dict[str, Any], pending: list[Commit],
             limit: datetime | None = None) -> datetime:  # fmt: skip
        call = {"tool": s.tool, "call_id": s.call_id}
        shown = {k: str(v)[:200] for k, v in s.args.items() if k != "code"}
        if s.tool == "run_code" and "code" in s.args and "note" not in shown:
            shown["note"] = "new code"
        self.emit(t, "tool.started", {**call, "arguments": shown}, turn=s.turn, **env)
        res = s.result if isinstance(s.result, dict) else {}
        has_error = bool(res.get("error"))
        end = _cap(t + _secs(s.duration * scale), limit)
        g = env["gen"]

        if s.tool == "run_code" and not has_error:
            ids = res.get("run_ids") or ([res["run_id"]] if res.get("run_id") else [])
            verify = env["role"] == "curator"
            vphase = {**env, "phase": "verification"} if verify else env
            if verify:
                source = str(s.args.get("run_id") or "new code")
                seeds = [self.exps[i]["seed"] for i in ids if i in self.exps]
                self.emit(
                    t,
                    "verification.started",
                    {"source": source, "seeds": seeds},
                    turn=s.turn,
                    **vphase,
                )
            for i in ids:
                e = self.exps.get(i)
                if not e:
                    continue
                exp_env = {**vphase, "experiment_id": i, "seed": e["seed"], "turn": s.turn}
                self.emit(t, "experiment.started", {"sandbox_mode": "research"}, **exp_env)
                self.emit(_cap(t + _secs(e["runtime_s"] * min(scale, 1.0) + 0.5), end), "experiment.completed",
                          {"status": e["status"], "val_loss": e["loss"], "runtime_s": e["runtime_s"],
                           "note": e["note"] or shown.get("note", ""), "started_from": self.best_version},
                          **exp_env)  # fmt: skip
            if verify:
                losses = res.get("losses") or [res.get("val_loss")]
                self.emit(end, "verification.completed", {
                    "source": str(s.args.get("run_id") or "new code"),
                    "run_ids": ids,
                    "seeds": [self.exps[i]["seed"] for i in ids if i in self.exps],
                    "losses": losses,
                    "mean": res.get("mean", losses[0] if len(losses) == 1 else None),
                    "spread": res.get("spread"),
                    "beats_best": bool(res.get("beats_best")),
                    "by": "curator",
                }, turn=s.turn, **vphase)  # fmt: skip

        if s.tool in ("upsert_memory", "remove_memory") and not has_error:
            entry_id = str(res.get("id") or s.args.get("id") or "")
            verb = (
                "remove" if s.tool == "remove_memory" else ("edit" if s.args.get("id") else "add")
            )
            c = next(
                (
                    c
                    for c in pending
                    if re.match(rf"^g{g} curator {verb} {re.escape(entry_id)} ", c.message)
                ),
                None,
            )
            if c:
                pending.remove(c)
                end = max(end, c.at)
            rows = self.run.memory_at(c.sha if c else None)
            text, tokens = _memory_payload(self.run, rows)
            data: dict[str, Any] = {
                "text": text,
                "tokens": tokens,
                "best_version": self.best_version,
            }
            if s.tool == "upsert_memory":
                data |= {
                    "action": "upsert",
                    "entry": _entry(next((r for r in rows if r["id"] == entry_id), None)),
                }
            else:
                data |= {
                    "action": "remove",
                    "id": entry_id,
                    "reason": str(s.args.get("reason", "")),
                }
            self.emit(end, "memory.updated", data, turn=s.turn, **env)

        self.emit(end, "tool.completed", {**call, "result": _small(s.result), "has_error": has_error},
                  turn=s.turn, **env)  # fmt: skip
        return end

    def unfinished_researchers(self, g: int, gs: datetime, done: set[str]) -> None:
        """Agents still working (no agents.jsonl row yet): their experiments so far."""
        by_agent: dict[str, list[dict[str, Any]]] = {}
        for e in self.run.experiments:
            if e["generation"] == g and e["agent"].startswith("r") and e["agent"] not in done:
                by_agent.setdefault(e["agent"], []).append(e)
        focuses = self.cfg["focuses"]
        for name, exps in sorted(by_agent.items()):
            idx = int(name.removeprefix("r")) - 1
            env = {"gen": g, "agent": name, "role": "researcher", "phase": "candidate",
                   "model": self.models.get("researcher")}  # fmt: skip
            t = gs + _secs(0.3 * idx)
            self.emit(t, "agent.started", {"max_turns": self.cfg["researcher_turns"], "run_budget": self.cfg["researcher_runs"],
                                           "focus": focuses[(g - 1 + idx) % len(focuses)],
                                           "best_version": self.best_version}, **env)  # fmt: skip
            for e in exps:
                t += _secs((THINK_S + e["runtime_s"]) * self.scale)
                self.emit(t, "experiment.completed", {"status": e["status"], "val_loss": e["loss"], "runtime_s": e["runtime_s"],
                          "note": e["note"], "started_from": self.best_version},
                          experiment_id=e["run_id"], seed=e["seed"], **env)  # fmt: skip

    def unfinished_curator(self, g: int, cs: datetime, commits: list[Commit]) -> None:
        exps = [e for e in self.run.experiments if e["generation"] == g and e["agent"] == "curator"]
        if not exps and not commits:
            return
        env = {
            "gen": g,
            "agent": "curator",
            "role": "curator",
            "phase": "curation",
            "model": self.models.get("curator"),
        }
        self.emit(cs, "agent.started", {"max_turns": self.cfg["curator_turns"], "run_budget": self.cfg["curator_runs"],
                                        "best_version": self.best_version}, **env)  # fmt: skip
        t = cs
        for e in exps:
            t += _secs((THINK_S + e["runtime_s"]) * self.scale)
            self.emit(t, "experiment.completed", {"status": e["status"], "val_loss": e["loss"], "runtime_s": e["runtime_s"],
                      "note": e["note"], "started_from": self.best_version},
                      **{**env, "phase": "verification"}, experiment_id=e["run_id"], seed=e["seed"])  # fmt: skip
        for c in commits:
            rows = self.run.memory_at(c.sha)
            text, tokens = _memory_payload(self.run, rows)
            m = re.match(rf"^g{g} curator (add|edit|remove) (m\d+)", c.message)
            data: dict[str, Any] = {
                "text": text,
                "tokens": tokens,
                "best_version": self.best_version,
            }
            if m and m[1] == "remove":
                data |= {"action": "remove", "id": m[2], "reason": ""}
            else:
                data |= {
                    "action": "upsert",
                    "entry": _entry(next((r for r in rows if m and r["id"] == m[2]), None)),
                }
            self.emit(max(c.at, t), "memory.updated", data, **env)

    # -- final test --

    def test_accuracies(self, key: str, version: int) -> list[float | None]:
        """Per-seed test accuracy of train.py v{version}: from the row, else from test runs
        that recorded it (older runs, re-scored by scripts/backfill_accuracy.py)."""
        row = self.run.row or {}
        if row.get(key):
            return row[key]
        by_seed: dict[int, float] = {}
        for e in self.run.experiments:
            tag = f"test: train.py v{version}"
            if (e["split"] == "test" and e.get("accuracy") is not None
                    and (e["note"] == tag or e["note"].startswith(tag + " "))):  # fmt: skip
                by_seed[e["seed"]] = e["accuracy"]
        return [by_seed.get(s) for s in self.cfg["test_seeds"]] if by_seed else []

    def final(self, last_gen_end: datetime) -> None:
        row = self.run.row
        if not row:
            return
        done = self.commit(r"^finished:")
        at = done.at if done else datetime.fromisoformat(row["finished_at"])
        seeds = self.cfg["test_seeds"]
        env = {"phase": "test", "role": "orchestrator"}
        self.emit(last_gen_end + _secs(1), "test.started", {"seeds": seeds}, **env)
        self.emit(at, "test.completed", {"seeds": seeds, "losses": row["test_best"]}, **env)
        self.emit(at, "setup.completed", {
            "setup": self.run.setup,
            "dataset": self.run.dataset,
            "l0": row["l0"],
            "noise_threshold": row["noise_threshold"],
            "best_version": row["best_version"],
            "best_val": row["best_val_loss"],
            "val_gain": row["val_gain"],
            "baseline_test_losses": row["test_start"],
            "best_test_losses": row["test_best"],
            "baseline_test_accuracies": self.test_accuracies("test_start_accuracy", 0),
            "best_test_accuracies": self.test_accuracies("test_best_accuracy", row["best_version"]),
            "majority_test_accuracy": row.get("test_majority_accuracy")
            or _majority_accuracy(self.run.dataset),
            "fitness": row["fitness"],
            "runs": row["runs"],
            "minutes": round((at - self.run.started).total_seconds() / 60, 2),
        })  # fmt: skip


def research_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The run's budget, in the shape of events.ts ResearchConfig."""
    curator_share = -(-cfg["curator_runs"] // cfg["researchers"])
    return {
        "generations": cfg["generations"],
        "researchers": cfg["researchers"],
        "researcher_runs": cfg["researcher_runs"],
        "researcher_runs_shared_brain": cfg["researcher_runs"] - curator_share,
        "curator_runs": cfg["curator_runs"],
        "curator_seeds": cfg["verify_seeds"],
        "test_seeds": cfg["test_seeds"],
        "baseline_seeds": cfg["baseline_seeds"],
        "max_turns": cfg["researcher_turns"],
        "curator_max_turns": cfg["curator_turns"],
        "memory_cap_tokens": cfg["memory_cap_tokens"],
        "raw_log_cap_tokens": 0,
    }


def build_events(run_id: str, members: list[SetupRun]) -> list[dict[str, Any]]:
    t0 = min(m.started for m in members)
    tl = Timeline(run_id, t0)
    models = {a["role"]: a.get("model") for m in members for a in m.agents}
    first = members[0]
    tl.emit(t0, "run.started", {
        "dataset": first.dataset,
        "setups": [m.setup for m in members],
        "research": research_config(first.config),
        "researcher_model": models.get("researcher") or "",
        "curator_model": models.get("curator") or "",
        "prompt": first.config["goal"],
    })  # fmt: skip
    tl.emit(t0, "baseline.started", {"seeds": first.config["baseline_seeds"]}, phase="baseline")
    baseline_done = None
    for m in members:
        b = SetupBuilder(tl, m)
        at, baseline = b.baseline()
        if baseline and baseline_done is None:
            baseline_done = at
            tl.emit(at, "baseline.completed", baseline, phase="baseline")
        b.build(at)
    if all(m.row for m in members):
        finished = max(datetime.fromisoformat(m.row["finished_at"]) for m in members if m.row)
        commits = [c.at for m in members for c in m.commits]
        tl.emit(max([finished, *commits]), "run.completed")
    return tl.finish()


# ---- runs as the API and the static export see them ----


def _status(members: list[SetupRun], now: float) -> str:
    if all(m.row for m in members):
        return "completed"
    return "running" if now - max(m.mtime for m in members) < STALL_AFTER_S else "stalled"


def run_info(run_id: str, members: list[SetupRun], events: list[dict[str, Any]]) -> dict[str, Any]:
    now = time.time()
    last = events[-1] if events else None
    summary = {}
    for m in members:
        gens = m.generations
        summary[m.setup] = {
            "status": "completed" if m.row else _status([m], now),
            "generations_done": len(gens),
            "l0": (m.row or {}).get("l0") or next((e["data"]["l0"] for e in events if e["type"] == "baseline.completed"), None),
            "best_val": (m.row or {}).get("best_val_loss") or (gens[-1]["end_best"] if gens else None),
            "fitness": (m.row or {}).get("fitness"),
        }  # fmt: skip
    status = _status(members, now)
    return {
        "run_id": run_id,
        "dataset": members[0].dataset,
        "setups": [m.setup for m in members],
        "status": status,
        "events_available": bool(events),
        "last_seq": last["seq"] if last else 0,
        "duration_ms": last["elapsed_ms"] if last else 0,
        "last_event_at": last["timestamp"] if last else None,
        "replay_available": status == "completed",
        "generations": members[0].config["generations"],
        "started_at": min(m.started for m in members).isoformat(),
        "summary": summary,
    }


@dataclass
class Snapshot:
    info: dict[str, Any]
    events: list[dict[str, Any]]


_cache: dict[str, tuple[str, Snapshot]] = {}


def _fingerprint(members: list[SetupRun]) -> str:
    parts = [f"{m.run_id}:{m.mtime}:{len(m.commits)}:{bool(m.row)}" for m in members]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def snapshots(root: Path | None = None) -> dict[str, Snapshot]:
    """Every demo run under `root`, newest first. Rebuilt only when its files change."""
    out = {}
    groups = group_runs(load_setup_runs(root))
    for run_id in sorted(groups, key=lambda gid: groups[gid][0].started, reverse=True):
        members = groups[run_id]
        key = _fingerprint(members)
        cached = _cache.get(run_id)
        if cached and cached[0] == key:
            snap = cached[1]
            snap.info["status"] = _status(members, time.time())
        else:
            events = build_events(run_id, members)
            snap = Snapshot(run_info(run_id, members, events), events)
            _cache[run_id] = (key, snap)
        out[run_id] = snap
    return out
