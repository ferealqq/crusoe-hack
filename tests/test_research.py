"""Shared Brain tests. No network: inference and sandboxes are monkeypatched."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app import report, scoring
from app.agents import base, researcher
from app.config import SETUPS, ResearchConfig, get_settings
from app.lab import Dataset, Lab
from app.loop import SetupRun, measure_baseline, rank
from app.memory import RawLog, SharedMemory, count_tokens
from app.services import inference, sandbox
from app.store import Store

# --- data isolation ---------------------------------------------------------------


def test_research_sandbox_never_gets_labels_or_test_data(tmp_path: Path):
    remote = set(sandbox.sandbox_files(tmp_path, "research").values())
    assert remote == {
        "/data/train.parquet",
        "/data/val_X.parquet",
        "/data/meta.json",
        "/harness/harness.py",
    }
    for name in ("val_y", "test_X", "test_y", "trainval", "test.py"):
        assert not any(name in r for r in remote)


def test_test_sandbox_gets_test_inputs_but_no_test_labels(tmp_path: Path):
    remote = set(sandbox.sandbox_files(tmp_path, "test").values())
    assert "/data/test_X.parquet" in remote and "/data/trainval.parquet" in remote
    assert not any("test_y" in r or "val_y" in r for r in remote)


# --- scoring ----------------------------------------------------------------------


def test_scoring_math():
    y = np.array([1.0, 0.0])
    assert scoring.log_loss(y, np.array([0.5, 0.5])) == pytest.approx(np.log(2))
    assert scoring.rmse(np.array([1.0, 3.0]), np.array([1.0, 1.0])) == pytest.approx(np.sqrt(2))
    assert scoring.pinball_90(np.array([10.0]), np.array([0.0])) == pytest.approx(9.0)
    assert scoring.gain(0.4, 0.3) == pytest.approx(0.25)
    assert scoring.fitness([0.4, 0.5], [0.3, 0.25]) == pytest.approx((0.25 + 0.5) / 2)
    # threshold: max(2 x spread, 0.5% of L0)
    assert scoring.noise_threshold([1.0, 1.0, 1.0]) == pytest.approx(0.005)
    assert scoring.noise_threshold([1.0, 1.2]) == pytest.approx(2 * np.std([1.0, 1.2], ddof=1))
    assert scoring.beats(0.9, 1.0, 0.05) and not scoring.beats(0.96, 1.0, 0.05)


def test_score_rejects_bad_predictions():
    y = np.array([1.0, 0.0])
    for bad in (np.array([0.5]), np.array([np.nan, 0.5]), np.array([1.5, 0.5])):
        with pytest.raises(ValueError):
            scoring.score("log_loss", y, bad)


# --- ranking ------------------------------------------------------------------------


def test_ranking_with_ties():
    # 0.300 and 0.302 tie within 0.005; 0.350 is clearly worse; None ranks last
    ranked = rank([0.350, 0.302, None, 0.300], threshold=0.005)
    assert [i for i, _ in ranked] == [3, 1, 0, 2]
    assert dict(ranked) == {3: 1, 1: 1, 0: 3, 2: 4}


# --- budget -------------------------------------------------------------------------


def test_equal_total_runs_across_setups():
    cfg = ResearchConfig()
    totals = {s: cfg.runs_per_generation(s) for s in SETUPS if s != "one_shot"}
    assert set(totals.values()) == {28}
    assert cfg.runs_per_generation("one_shot") == 1


async def test_researcher_run_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = []

    async def fake_run(code: str, seed: int, agent: str, note: str, gen: int, started: int):
        calls.append(code)
        return SimpleNamespace(id=f"e{len(calls)}", status="ok", val_loss=0.3, output_tail="")

    lab = SimpleNamespace(run=fake_run)
    log: list[dict[str, Any]] = []
    tools = researcher.researcher_tools(
        lab=lab,  # type: ignore[arg-type]
        name="r1",
        gen=1,
        best_version=0,
        runs=2,
        memory=None,
        cfg=ResearchConfig(),
        run_log=log,
    )
    run_code = next(t for t in tools if t.name == "run_code").fn
    assert (await run_code(code="a", note="x"))["runs_left"] == 1
    assert (await run_code(code="b", note="y"))["runs_left"] == 0
    assert "error" in await run_code(code="c", note="z")
    assert len(calls) == 2 and len(log) == 2


# --- memory -------------------------------------------------------------------------


def _mem(cap: int = 2000) -> SharedMemory:
    return SharedMemory(cap_tokens=cap)


def test_memory_cap_refuses_and_keeps_state():
    m = _mem(cap=60)
    ok = m.upsert(kind="works", claim="a" * 150, run_ids=["e1"], gen=1, best_version=0)
    assert "id" in ok
    refused = m.upsert(kind="failed", claim="b" * 150, run_ids=["e2"], gen=1, best_version=0)
    assert "error" in refused and len(m.active()) == 1
    assert "error" in m.upsert(kind="works", claim="x" * 201, run_ids=[], gen=1, best_version=0)
    assert "error" in m.upsert(kind="best", claim="x", run_ids=[], gen=1, best_version=0)


def test_memory_drop_order_try_next_then_failed_never_best():
    m = _mem()
    for kind in ("failed", "try_next", "try_next", "works"):
        m.upsert(kind=kind, claim=kind * 10, run_ids=["e1"], gen=1, best_version=0)
    m.set_best("v1: " + "z" * 200, ["e1"], 1, 1, 0.3)
    m.cap_tokens = count_tokens(m.render(1)) - 1
    dropped = m.enforce_cap(gen=2, best_version=1)
    assert dropped == ["m2"]  # the oldest try_next
    m.cap_tokens = 10
    dropped = m.enforce_cap(gen=2, best_version=1)
    assert dropped == ["m3", "m1"]  # remaining try_next, then failed
    kinds = {e.kind for e in m.active()}
    assert kinds == {"best", "works"}


def test_memory_render_is_stable_and_ordered():
    m = _mem()
    m.upsert(kind="try_next", claim="t", run_ids=[], gen=1, best_version=0)
    m.upsert(kind="works", claim="w", run_ids=["e1"], gen=1, best_version=0)
    m.set_best("v1: best", ["e1"], 1, 1, 0.3)
    text = m.render(1)
    assert text.index("## Best") < text.index("## Works") < text.index("## Try next")
    assert text == m.render(1)


def test_staleness():
    m = _mem()
    m.upsert(kind="works", claim="old", run_ids=["e1"], gen=1, best_version=0)
    m.upsert(kind="works", claim="new", run_ids=["e2"], gen=2, best_version=1)
    assert [e.id for e in m.stale(best_version=1)] == []
    assert [e.id for e in m.stale(best_version=2)] == ["m1"]
    assert "STALE" in m.render(2)


def test_raw_log_trims_oldest():
    r = RawLog(cap_tokens=50)
    for i in range(5):
        r.append(f"report {i} " + "x" * 80)
    assert r.reports[-1].startswith("report 4")
    assert r.tokens() <= 50 or len(r.reports) == 1
    assert not any(rep.startswith("report 0") for rep in r.reports)


# --- end to end with fakes ------------------------------------------------------------


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Dataset:
    rng = np.random.default_rng(0)
    d = tmp_path / "churn"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"name": "churn", "target": "y", "loss": "log_loss"}))
    pd.DataFrame({"y": rng.integers(0, 2, 200)}).to_parquet(d / "val_y.parquet")
    pd.DataFrame({"y": rng.integers(0, 2, 200)}).to_parquet(d / "test_y.parquet")
    return Dataset(d)


def _fake_sandbox(dataset: Dataset):
    async def run_train(code: str, data_dir: Path, seed: int, mode: str = "research"):
        y = dataset.val_y if mode == "research" else dataset.test_y
        noise = 0.001 * seed
        preds = y * 0.6 + 0.2 + noise if "GOOD" in code else np.full_like(y, 0.45 + noise)
        return sandbox.TrainResult("ok", preds, "fake output", 0.1)

    return run_train


def _msg(content: str = "", calls: list[tuple[str, dict[str, Any]]] | None = None):
    tool_calls = [
        SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(calls or [])
    ] or None
    return SimpleNamespace(content=content, tool_calls=tool_calls)


async def fake_chat_turn(messages: list[dict[str, Any]], model: str, tools, **params):
    usage = {"prompt": 100, "cached": 50, "output": 10}
    tool_results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    if model == get_settings().researcher_model:
        if not tool_results:
            return _msg(calls=[("run_code", {"code": "GOOD", "note": "better model"})]), None, usage
        return _msg("Tried: GOOD\nBest run: x\nLearned: works\nNext: more"), "stop", usage
    # curator: re-check, record the finding, finish
    if not tool_results:
        return _msg(calls=[("run_code", {"code": "GOOD"})]), None, usage
    if len(tool_results) == 1:
        ids = tool_results[0]["run_ids"]
        claim = {"kind": "works", "claim": "GOOD model lowers log loss", "run_ids": ids}
        return _msg(calls=[("upsert_memory", claim)]), None, usage
    return _msg("Added one works entry."), "stop", usage


async def test_shared_brain_generation_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_dataset: Dataset
):
    monkeypatch.setattr(sandbox, "run_train", _fake_sandbox(tiny_dataset))
    monkeypatch.setattr(inference, "chat_turn", fake_chat_turn)
    monkeypatch.setattr(get_settings(), "ydc_api_key", "")
    cfg = ResearchConfig(generations=2, researchers=2)
    root = tmp_path / "run"
    store = Store(root)
    baseline = await measure_baseline(tiny_dataset, "BASE", cfg, store)
    assert baseline.threshold > 0

    run = SetupRun("shared_brain", tiny_dataset, baseline, cfg, root)
    result = await run.run()

    assert result["best_version"] == 1 and result["fitness"] > 0
    gens = Store(root / "shared_brain").read("generations")
    assert [g["new_best"] for g in gens] == [True, False]
    assert gens[0]["runs"] == 2 * 1 + 3  # each fake researcher runs once; curator 3 seeds
    kinds = {e.kind for e in run.memory.active()} if run.memory else set()
    assert {"best", "works"} <= kinds

    store.write_json("config", {
        "setups": ["shared_brain"], "research": cfg.model_dump(), "started": "test",
        "researcher_model": "r", "curator_model": "c", "pricing": {},
    })  # fmt: skip
    store.write_json("baseline", baseline.__dict__)
    assert "Shared Brain" in report.write_report(root).read_text()


async def test_agent_survives_api_failure(monkeypatch: pytest.MonkeyPatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("429 forever")

    monkeypatch.setattr(inference, "chat_turn", boom)
    result = await base.run_agent(
        model="m", messages=[{"role": "user", "content": "hi"}], tools=[], params={}, max_turns=3
    )
    assert result.final.startswith("(stopped early")


def test_lab_records_bad_predictions_as_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_dataset: Dataset
):
    async def run_train(code: str, data_dir: Path, seed: int, mode: str = "research"):
        return sandbox.TrainResult("ok", np.array([2.0]), "", 0.1)

    monkeypatch.setattr(sandbox, "run_train", run_train)
    import asyncio

    exp = asyncio.run(Lab(tiny_dataset, Store(tmp_path)).run("x", 0, "r1", "n", 1, 0))
    assert exp.status == "error" and exp.val_loss is None
