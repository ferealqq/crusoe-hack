"""app/replay.py: events rebuilt from research run files (no git here: the fallback path)."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import replay, scoring
from app.config import ResearchConfig
from app.main import app
from app.store import read_jsonl, write_jsonl

RUN = "churn-shared_brain-20261017-100000"
CFG = ResearchConfig(generations=1, researchers=1).model_dump()


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _exp(run_id, gen, agent, seed, loss, runtime=10.0):
    return {"run_id": run_id, "generation": gen, "agent": agent, "seed": seed, "split": "val",
            "status": "ok", "loss": loss, "runtime_s": runtime, "note": f"note {run_id}",
            "output_tail": "", "code": "..."}  # fmt: skip


def _call(cid, name, args):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "DATA_DIR", tmp_path)
    replay._cache.clear()
    d = tmp_path / "research_runs" / RUN
    exps = [_exp(f"g0-base-{i + 1}", 0, "orchestrator", i, 0.40) for i in range(5)]
    exps += [_exp("g1-r1-1", 1, "r1", 0, 0.33)]
    exps += [_exp(f"g1-curator-{i}", 1, "curator", i, 0.331) for i in (1, 2, 3)]
    _jsonl(d / "experiments.jsonl", exps)
    researcher = {
        "agent": "r1", "role": "researcher", "generation": 1, "focus": "features", "model": "m",
        "report": "Tried CatBoost.", "run_ids": ["g1-r1-1"], "turns": 2,
        "usage": {"prompt": 100, "cached": 50, "output": 10, "calls": 2},
        "transcript": [
            {"role": "system", "content": "rules"},
            {"role": "assistant", "content": "Try CatBoost.", "tool_calls": [_call("c1", "run_code", {"code": "x", "note": "CatBoost"})]},
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"run_id": "g1-r1-1", "status": "ok", "val_loss": 0.33, "runs_left": 4})},
            {"role": "assistant", "content": "Report."},
        ],
    }  # fmt: skip
    curator = {
        "agent": "curator", "role": "curator", "generation": 1, "focus": None, "model": "c",
        "report": "Done.", "run_ids": [], "turns": 2,
        "usage": {"prompt": 80, "cached": 0, "output": 8, "calls": 2},
        "transcript": [
            {"role": "assistant", "content": "", "tool_calls": [_call("v1", "run_code", {"run_id": "g1-r1-1", "seeds": 3})]},
            {"role": "tool", "tool_call_id": "v1", "content": json.dumps({"run_ids": ["g1-curator-1", "g1-curator-2", "g1-curator-3"], "losses": [0.331] * 3, "mean": 0.331, "spread": 0.0, "beats_best": True})},
            {"role": "assistant", "content": "", "tool_calls": [_call("u1", "upsert_memory", {"kind": "works", "claim": "CatBoost works", "run_ids": ["g1-r1-1"]})]},
            {"role": "tool", "tool_call_id": "u1", "content": json.dumps({"id": "m2", "tokens": 40, "cap": 2000})},
        ],
    }  # fmt: skip
    _jsonl(d / "agents.jsonl", [researcher, curator])
    _jsonl(d / "generations.jsonl", [{
        "generation": 1, "start_best": 0.40, "end_best": 0.331, "best_version": 1, "best_run_id": "g1-r1-1",
        "best_changed": True, "val_gain": 0.17, "memory_tokens": 20, "memory_tokens_after": 60, "dropped": [],
        "runs": 4, "usage": {}, "ranking": [{"agent": "r1", "best_run_id": "g1-r1-1", "best_loss": 0.33, "gain": 0.17, "tied": False}],
    }])  # fmt: skip
    _jsonl(d / "memory.jsonl", [
        {"id": "m1", "kind": "best", "claim": "train.py v1", "run_ids": [], "val_change": None, "seeds": [], "tested_on": 1, "created_gen": 0, "updated_gen": 1},
        {"id": "m2", "kind": "works", "claim": "CatBoost works", "run_ids": ["g1-r1-1"], "val_change": -0.07, "seeds": [0], "tested_on": 0, "created_gen": 1, "updated_gen": 1},
    ])  # fmt: skip
    _jsonl(tmp_path / "research_runs.jsonl", [{
        "run_id": RUN, "dataset": "churn", "setup": "shared_brain", "config": CFG, "l0": 0.40,
        "noise_threshold": 0.002, "best_version": 1, "best_run_id": "g1-r1-1", "best_val_loss": 0.331,
        "val_gain": 0.17, "test_start": [0.41] * 3, "test_best": [0.34] * 3, "fitness": 0.17, "runs": 9,
        "memory_tokens": 60, "usage": {}, "started_at": "2026-10-17T10:00:00+00:00",
        "finished_at": "2026-10-17T10:20:00+00:00",
    }])  # fmt: skip
    return tmp_path


def test_events_follow_the_files(data):
    snap = replay.snapshots()["churn-20261017-100000"]
    events = snap.events
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert [e["elapsed_ms"] for e in events] == sorted(e["elapsed_ms"] for e in events)
    types = [e["type"] for e in events]
    assert types[0] == "run.started" and types[-1] == "run.completed"
    for t in ("baseline.completed", "experiment.completed", "verification.completed",
              "best.updated", "generation.completed", "setup.completed"):  # fmt: skip
        assert t in types

    exp = next(
        e for e in events if e["type"] == "experiment.completed" and e["phase"] == "candidate"
    )
    assert exp["experiment_id"] == "g1-r1-1" and exp["data"]["val_loss"] == 0.33
    upsert = next(
        e for e in events if e["type"] == "memory.updated" and e["data"]["action"] == "upsert"
    )
    assert upsert["data"]["entry"]["claim"] == "CatBoost works"
    assert "[m2] CatBoost works" in upsert["data"]["text"]
    done = next(e for e in events if e["type"] == "setup.completed")["data"]
    assert done["fitness"] == 0.17 and done["best_test_losses"] == [0.34] * 3
    # tool arguments never carry the code
    assert all("code" not in e["data"].get("arguments", {}) for e in events)

    assert snap.info["status"] == "completed"
    assert snap.info["summary"]["shared_brain"]["fitness"] == 0.17


def test_api_serves_runs_and_pages(data):
    client = TestClient(app)
    runs = client.get("/research-runs").json()
    assert [r["run_id"] for r in runs] == ["churn-20261017-100000"]
    page = client.get("/research-runs/churn-20261017-100000/events?after=0&limit=5").json()
    assert [e["seq"] for e in page["events"]] == [1, 2, 3, 4, 5] and page["has_more"]
    assert client.get("/research-runs/nope").status_code == 404

    with client.stream("GET", "/research-runs/churn-20261017-100000/stream?after=3") as resp:
        lines = [line for line in resp.iter_lines() if line.startswith("event: ")]
    assert lines[0] == "event: research" and lines[-1] == "event: stream.end"


def _done(tmp):
    replay._cache.clear()
    events = replay.snapshots()["churn-20261017-100000"].events
    return next(e for e in events if e["type"] == "setup.completed")["data"]


def test_test_accuracy_from_backfilled_runs(data):
    """Older rows have no accuracy: it comes from re-scored test runs, majority from test labels."""
    d = data / "research_runs" / RUN
    test = [{**_exp(f"test-v{v}-backfill-{s}", 2, "orchestrator", s, 0.3), "split": "test",
             "accuracy": acc, "note": f"test: train.py v{v} (accuracy backfill)"}
            for v, acc in ((0, 0.82), (1, 0.86), (10, 0.5)) for s in (1, 2, 3)]  # fmt: skip
    with (d / "experiments.jsonl").open("a") as f:
        f.writelines(json.dumps(e) + "\n" for e in test)
    (data / "churn").mkdir()
    (data / "churn" / "meta.json").write_text(json.dumps({"target": "exited"}))
    pd.DataFrame({"exited": [0, 0, 0, 1]}).to_parquet(data / "churn" / "test_y.parquet")
    replay._majority.clear()

    done = _done(data)
    assert done["baseline_test_accuracies"] == [0.82] * 3
    assert done["best_test_accuracies"] == [0.86] * 3  # v1, not v10
    assert done["majority_test_accuracy"] == 0.75


def test_test_accuracy_from_row(data):
    rows = read_jsonl(data / "research_runs.jsonl")
    rows[0] |= {"test_start_accuracy": [0.8] * 3, "test_best_accuracy": [0.9] * 3,
                "test_majority_accuracy": 0.7}  # fmt: skip
    write_jsonl(data / "research_runs.jsonl", rows)
    done = _done(data)
    assert done["baseline_test_accuracies"] == [0.8] * 3
    assert done["best_test_accuracies"] == [0.9] * 3 and done["majority_test_accuracy"] == 0.7


def test_accuracy_scores():
    y = np.array([0, 1, 1, 0])
    assert scoring.accuracy(y, np.array([0.1, 0.9, 0.4, 0.5])) == 0.5
    assert scoring.majority_accuracy(np.array([0, 0, 0, 1])) == 0.75
