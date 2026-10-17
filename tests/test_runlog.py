import json

from fastapi.testclient import TestClient

from app.main import app
from app.runlog import store
from app.runlog.events import GenStarted, ResearcherSpec, RunConfig, event_adapter
from app.runlog.logger import RunLogger

CONFIG = RunConfig(
    generations=1,
    researchers=[ResearcherSpec(focus="optimizer", model="m")],
    curator_model="c",
    budget="10 steps",
    seeds=1,
    memory_cap_tokens=100,
    dataset="d",
)


def test_samples_match_contract():
    for path in store.SAMPLES_DIR.glob("*.jsonl"):
        events = [event_adapter.validate_json(line) for line in path.read_text().splitlines()]
        assert [e.seq for e in events] == list(range(len(events)))
        assert events[0].t == "run.started" and events[-1].t == "run.finished"


def test_api_lists_and_serves_sample_runs():
    client = TestClient(app)
    runs = {r["run_id"]: r for r in client.get("/api/runs").json()}
    assert runs["sample-shared-brain"]["status"] == "completed"
    assert runs["sample-shared-brain"]["fitness"] > runs["sample-no-memory"]["fitness"]

    body = client.get("/api/runs/sample-shared-brain/events").text
    assert json.loads(body.splitlines()[0])["t"] == "run.started"
    assert client.get("/api/runs/nope/events").status_code == 404


def test_run_logger_writes_and_streams(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "LOCAL_DIR", tmp_path)
    with RunLogger.start("t1", setup="shared_brain", config=CONFIG, baseline_val_loss=2.0) as rl:
        rl.emit(GenStarted(gen=1, start_best_val_loss=2.0))

    lines = (tmp_path / "t1.jsonl").read_text().splitlines()
    assert [json.loads(line)["t"] for line in lines] == [
        "run.started",
        "gen.started",
        "run.finished",
    ]
    assert [json.loads(line)["seq"] for line in lines] == [0, 1, 2]

    with TestClient(app).stream("GET", "/api/runs/t1/stream?after=0") as resp:
        data = [line for line in resp.iter_lines() if line.startswith("data: ")]
    assert [json.loads(d[6:])["t"] for d in data] == ["gen.started", "run.finished"]
