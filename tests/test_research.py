"""Offline tests: no Crusoe, no Modal (README "Done when")."""

import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app import loop, scoring
from app.agents import prompts, tools
from app.config import ResearchConfig
from app.data import Dataset
from app.lab import Best, Lab, runs_per_generation
from app.memory import Memory, MemoryFull
from app.schemas import Experiment, Reply, ToolCall, TrainResult, Usage
from app.services import inference, sandbox

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.anyio


def _frame(data: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(data))


# --- labels never reach the research sandbox ----------------------------------------------


def test_research_files_hold_no_val_labels_or_test_data(dataset: Dataset, tmp_path: Path):
    files = dataset.research_files
    assert set(files) == {"train.parquet", "X.parquet"}
    X = _frame(files["X.parquet"])
    assert "y" not in X.columns
    pd.testing.assert_frame_equal(X, pd.read_parquet(tmp_path / "toy" / "val_X.parquet"))
    test_X = pd.read_parquet(tmp_path / "toy" / "test_X.parquet")
    train = _frame(files["train.parquet"])
    assert len(train) == 140  # train only: no val or test rows
    assert not X.merge(test_X).shape[0] and not train.drop(columns="y").merge(test_X).shape[0]


def test_test_files_give_train_plus_val_labels_and_test_inputs(dataset: Dataset):
    train = _frame(dataset.test_files["train.parquet"])
    X = _frame(dataset.test_files["X.parquet"])
    assert len(train) == 170 and train["y"].notna().all()
    assert "y" not in X.columns and len(X) == 30


async def test_research_runs_send_only_research_files(lab: Lab, monkeypatch):
    seen: list[dict[str, bytes]] = []

    async def run_train(code, files, target, seed, loss="log_loss"):
        seen.append(files)
        return TrainResult(status="ok", preds=[0.2] * 30)

    monkeypatch.setattr(sandbox, "run_train", run_train)
    exp = await lab.run_experiment("x", prefix="p", agent="r1", generation=1, seed=0)
    assert seen == [lab.dataset.research_files]
    assert exp.loss == pytest.approx(scoring.log_loss(lab.dataset.val_y, np.full(30, 0.2)))


def test_harness_rejects_bad_predictions(tmp_path: Path, dataset: Dataset):
    for name, data in dataset.research_files.items():
        (tmp_path / name).write_bytes(data)
    (tmp_path / "harness.py").write_text((ROOT / "sandbox" / "harness.py").read_text())

    def run(train_py: str) -> subprocess.CompletedProcess[str]:
        (tmp_path / "train.py").write_text(train_py)
        cmd = [sys.executable, "harness.py", "--target", "y", "--seed", "0"]
        return subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)

    good = (
        "def fit(t, target, seed):\n    return None\n"
        "def predict(m, X):\n    return [0.2] * len(X)\n"
    )
    assert run(good).returncode == 0
    assert json.loads((tmp_path / "preds.json").read_text()) == [0.2] * 30
    assert run(good.replace("[0.2] * len(X)", "[0.2]")).returncode != 0
    assert run(good.replace("0.2", "1.5")).returncode != 0  # log loss needs probabilities
    regression = good.replace("0.2", "41.5")
    (tmp_path / "train.py").write_text(regression)
    cmd = [sys.executable, "harness.py", "--target", "y", "--seed", "0", "--loss", "rmse"]
    assert subprocess.run(cmd, cwd=tmp_path, capture_output=True).returncode == 0


# --- run limits ----------------------------------------------------------------------------


async def test_researcher_run_limit(lab: Lab, monkeypatch):
    calls = 0

    async def run_train(code, files, target, seed, loss="log_loss"):
        nonlocal calls
        calls += 1
        return TrainResult(status="error", output_tail="boom")

    monkeypatch.setattr(sandbox, "run_train", run_train)
    lab.best = Best(0, "x", 0.5, "g0-base-1", "start", 0.5)
    budget, runs = tools.Budget(2), []
    run_code = tools.researcher_run_code(lab, agent="r1", generation=1, budget=budget, runs=runs)
    results = [await run_code.fn({"code": "x", "note": "n"}) for _ in range(3)]
    assert [r.get("runs_left") for r in results] == [1, 0, 0]
    assert "no runs left" in results[2]["error"]
    assert calls == 2  # crashes use up a run; the refused call never reaches the sandbox


async def test_curator_check_needs_enough_runs(lab: Lab, monkeypatch):
    async def run_train(code, files, target, seed, loss="log_loss"):
        return TrainResult(status="ok", preds=[0.2] * 30)

    monkeypatch.setattr(sandbox, "run_train", run_train)
    lab.best = Best(0, "x", 0.6, "g0-base-1", "start", 0.6)
    budget, checks = tools.Budget(4), []
    run_code = tools.curator_run_code(lab, generation=1, budget=budget, verifications=checks)
    first = await run_code.fn({"code": "x", "seeds": 3})
    assert first["beats_best"] is True and first["runs_left"] == 1
    assert "needs 3 runs" in (await run_code.fn({"code": "x"}))["error"]
    assert (await run_code.fn({"code": "x", "seeds": 1}))["runs_left"] == 0
    assert [c.full for c in checks] == [True, False]


def test_equal_total_runs_across_setups():
    config = ResearchConfig()
    assert runs_per_generation(config, "shared_brain") == runs_per_generation(config, "no_memory")
    assert runs_per_generation(config, "no_memory") == 28


# --- memory --------------------------------------------------------------------------------


def _add(memory: Memory, kind: str, claim: str, gen: int = 1, tested_on: int = 0) -> str:
    entry = memory.upsert(
        kind=kind,  # type: ignore[arg-type]
        claim=claim,
        run_ids=["g1-r1-1"],
        val_change=-0.01,
        seeds=[0],
        tested_on=tested_on,
        generation=gen,
    )
    return entry.id


def test_memory_cap_refuses_and_drops_oldest_try_next_then_failed(tmp_path: Path):
    memory = Memory(tmp_path / "memory.jsonl", cap_tokens=10_000)
    memory.set_best(claim="best " + "b" * 150, run_ids=["x"], val_change=None, version=0, gen=0)
    ids = {
        "works": _add(memory, "works", "w" * 150),
        "failed_old": _add(memory, "failed", "f" * 150, gen=1),
        "failed_new": _add(memory, "failed", "g" * 150, gen=2),
        "try_old": _add(memory, "try_next", "t" * 150, gen=1),
        "try_new": _add(memory, "try_next", "u" * 150, gen=2),
    }
    memory.cap = memory.tokens() - 1
    with pytest.raises(MemoryFull):
        _add(memory, "works", "too much")
    memory.cap = memory.tokens() - 100  # needs about three drops
    dropped = [e.id for e in memory.enforce_cap()]
    assert dropped == [ids["try_old"], ids["try_new"], ids["failed_old"]]
    kinds = {e.kind for e in memory.entries}
    assert kinds == {"best", "works", "failed"}  # best and works are never dropped
    assert Memory(tmp_path / "memory.jsonl", 0).entries == memory.entries  # saved to disk


def test_memory_render_is_stable_and_best_is_protected(tmp_path: Path):
    memory = Memory(tmp_path / "memory.jsonl", cap_tokens=2000)
    t = _add(memory, "try_next", "try this")
    w = _add(memory, "works", "this works")
    memory.set_best(claim="best so far", run_ids=["x"], val_change=None, version=1, gen=1)
    text = memory.render()
    assert text.index("## best") < text.index("## works") < text.index("## try_next")
    assert text.index(f"[{w}]") < text.index(f"[{t}]")
    best_id = next(e.id for e in memory.entries if e.kind == "best")
    with pytest.raises(ValueError):
        memory.remove(best_id)
    with pytest.raises(ValueError):
        _add(memory, "best", "nope")
    with pytest.raises(ValueError):
        _add(memory, "works", "x" * 201)


def test_staleness(tmp_path: Path):
    memory = Memory(tmp_path / "memory.jsonl", cap_tokens=2000)
    old = _add(memory, "works", "old", tested_on=1)
    _add(memory, "works", "recent", tested_on=2)
    memory.set_best(claim="best", run_ids=["x"], val_change=None, version=0, gen=0)
    assert [e.id for e in memory.stale(best_version=3)] == [old]
    assert memory.stale(best_version=2) == []


# --- ranking and scoring -------------------------------------------------------------------


def _exp(run_id: str, loss: float | None) -> Experiment:
    return Experiment(
        run_id=run_id,
        generation=1,
        agent=run_id.split("-")[1],
        seed=0,
        status="ok" if loss is not None else "error",
        loss=loss,
        runtime_s=1.0,
        code="",
    )


def test_ranking_with_ties():
    logs = {
        "r1": [_exp("g1-r1-1", 0.40), _exp("g1-r1-2", 0.35)],
        "r2": [_exp("g1-r2-1", 0.352)],
        "r3": [_exp("g1-r3-1", 0.30)],
        "r4": [_exp("g1-r4-1", None)],
    }
    ranked = scoring.rank_researchers(logs, start_best=0.40, threshold=0.005)
    assert [r.agent for r in ranked] == ["r3", "r1", "r2", "r4"]
    assert [r.tied for r in ranked] == [False, False, True, False]
    assert ranked[0].best_run_id == "g1-r3-1" and ranked[0].gain == pytest.approx(0.25)
    assert ranked[3].gain is None


def test_curator_brief_flags_gains_within_noise(lab: Lab):
    lab.threshold = 0.002
    lab.best = Best(1, "x", 0.3244, "g1-r2-3", "CatBoost", seed_loss=0.3236)
    logs = {"r1": [_exp("g2-r1-1", 0.3232)], "r2": [_exp("g2-r2-1", 0.3100)]}
    ranked = scoring.rank_researchers(logs, lab.best.seed_loss, lab.threshold)
    reports = [(r.agent, "f", "report", logs[r.agent]) for r in ranked]
    brief = prompts.curator_brief(lab, 2, reports, ranked)
    assert "0.32360" in brief  # gains are against the best's seed-0 score, not 0.3244
    r2, r1 = brief.split("\n2. ")[0], brief.split("\n2. ")[1]
    assert "clears the noise threshold" in r2 and "within noise" in r1.splitlines()[0]


def test_scoring_math():
    assert scoring.log_loss(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(np.log(2))
    assert scoring.log_loss(np.array([1]), np.array([1.0])) == pytest.approx(0, abs=1e-12)
    assert scoring.noise_threshold([0.4, 0.4]) == pytest.approx(0.002)  # 0.5% of L0
    assert scoring.noise_threshold([0.38, 0.42]) == pytest.approx(0.04)  # 2 x spread
    assert scoring.gain(0.4, 0.3) == pytest.approx(0.25)
    assert scoring.score("rmse", np.array([1.0, 3.0]), np.array([2.0, 2.0])) == pytest.approx(1.0)
    assert scoring.fitness([0.4, 0.5], [0.3, 0.5]) == pytest.approx(0.125)
    assert scoring.beats(0.39, 0.40, 0.005) and not scoring.beats(0.398, 0.40, 0.005)


# --- a whole research run, with fake models and a fake sandbox -----------------------------


def _fake_sandbox(monkeypatch) -> None:
    """Predicts the constant P=<x> found in the code; the starting train.py predicts 0.5."""

    async def run_train(code, files, target, seed, loss="log_loss"):
        m = re.search(r"P=([0-9.]+)", code)
        n = len(_frame(files["X.parquet"]))
        return TrainResult(status="ok", preds=[float(m.group(1)) if m else 0.5] * n)

    monkeypatch.setattr(sandbox, "run_train", run_train)


def _fake_models(monkeypatch) -> list[str]:
    roles: list[str] = []

    def call(name: str, **args: Any) -> Reply:
        return Reply(
            content="",
            tool_calls=[ToolCall(id=f"c{len(roles)}", name=name, arguments=json.dumps(args))],
            usage=Usage(prompt=100, cached=50, output=10, calls=1),
        )

    async def complete(messages, model, tools=None, **params) -> Reply:
        names = {t["function"]["name"] for t in tools or []}
        turn = sum(m["role"] == "assistant" for m in messages)
        roles.append("curator" if "upsert_memory" in names else "researcher")
        done = Reply(content="Tried: P=0.2\nBest run: x\nLearned: ok\nNext: more")
        if "upsert_memory" not in names:  # researcher: one run, then report
            if turn == 0:
                return call("run_code", code="P=0.2", note="predict the base rate")
            return done
        top = re.search(r"best run (\S+),", messages[1]["content"]).group(1)
        if turn == 0:
            return call("run_code", run_id=top, seeds=3)
        if turn == 1:
            return call("upsert_memory", kind="works", claim="P=0.2 beats 0.5", run_ids=[top])
        return Reply(content="Added one works entry.")

    monkeypatch.setattr(inference, "complete", complete)
    return roles


@pytest.mark.parametrize("setup", ["shared_brain", "no_memory", "one_shot"])
async def test_research_run_end_to_end(setup, dataset: Dataset, tmp_path: Path, monkeypatch):
    _fake_sandbox(monkeypatch)
    roles = _fake_models(monkeypatch)
    (tmp_path / "baselines" / "toy").mkdir(parents=True)
    (tmp_path / "baselines" / "toy" / "train.py").write_text("starting train.py")
    monkeypatch.setattr(loop, "BASELINES", tmp_path / "baselines")
    config = ResearchConfig(generations=1, researchers=2, researcher_runs=3, curator_runs=4)
    row = await loop.research_run(dataset, setup, config, root=tmp_path)

    run_dir = tmp_path / "research_runs" / row["run_id"]
    assert row["best_version"] == 1 and row["val_gain"] > 0 and row["fitness"] > 0
    assert (run_dir / "best_train.py").read_text() == "P=0.2"
    gens = [json.loads(line) for line in (run_dir / "generations.jsonl").read_text().splitlines()]
    assert len(gens) == 1 and gens[0]["best_changed"]
    agents = (run_dir / "agents.jsonl").read_text().splitlines()
    memory = Memory(run_dir / "memory.jsonl", 2000)
    if setup == "shared_brain":
        assert "curator" in roles and len(agents) == 3
        assert {e.kind for e in memory.entries} == {"best", "works"}
        assert next(e for e in memory.entries if e.kind == "best").tested_on == 1
    elif setup == "no_memory":
        assert "curator" not in roles and len(agents) == 2 and memory.entries == []
    else:  # one_shot: one researcher, one run, whatever the config says
        assert "curator" not in roles and len(agents) == 1 and memory.entries == []
        assert row["config"]["researchers"] == 1 and row["runs"] == 5 + 1
    tests = [
        json.loads(line)
        for line in (run_dir / "experiments.jsonl").read_text().splitlines()
        if json.loads(line)["split"] == "test"
    ]
    assert len(tests) == 6  # starting and best train.py, seeds 1-3
