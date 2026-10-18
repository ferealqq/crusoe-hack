"""Runs started from the UI: the prompt becomes the researchers' brief (app/launch.py)."""

import json

import pytest
from fastapi.testclient import TestClient

from app import launch, replay
from app.agents import prompts
from app.config import DEFAULT_GOAL, ResearchConfig, get_settings
from app.lab import Best, Lab
from app.main import app


def test_prompt_becomes_the_researchers_brief(dataset, tmp_path):
    goal = "You are an ML researcher. Lower the log loss; try gradient boosting first."
    lab = Lab(dataset, "shared_brain", ResearchConfig(goal=goal), root=tmp_path, run_id="r")
    lab.best = Best(0, "x", 0.5, "g0-base-1", "start", 0.5)
    assert prompts.researcher_system(lab).startswith(goal + "\n\n## How it works")
    assert goal in prompts.curator_system(lab)


def test_default_brief_keeps_the_prompts_unchanged(lab):
    lab.best = Best(0, "x", 0.5, "g0-base-1", "start", 0.5)
    assert prompts.researcher_system(lab).startswith(DEFAULT_GOAL + "\n\n## How it works")
    assert "The team's brief" not in prompts.curator_system(lab)


def test_saved_config_shows_up_while_the_run_is_going(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "DATA_DIR", tmp_path)
    for setup in launch.SETUPS:
        cfg = ResearchConfig(goal="Find a better model.", generations=2)
        Lab(dataset, setup, cfg, root=tmp_path, run_id=f"toy-{setup}-20261017-100000").save_config()
    snap = replay.snapshots(tmp_path)["toy-20261017-100000"]
    assert snap.info["status"] == "running"
    assert snap.info["generations"] == 2
    assert snap.events[0]["data"]["prompt"] == "Find a better model."


class FakeProc:
    def __init__(self, cmd, cwd, **_):
        self.cmd, self.returncode = cmd, None
        args = dict(zip(cmd[3::2], cmd[4::2], strict=False))
        d = (
            cwd
            / "data"
            / "research_runs"
            / f"{args['--dataset']}-{args['--setup']}-{args['--stamp']}"
        )
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"goal": args["--goal"]}))

    def poll(self):
        return self.returncode


@pytest.fixture
def fake_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    monkeypatch.setattr(launch, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(launch, "_launches", [])
    monkeypatch.setattr(launch.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(replay, "snapshots", lambda: {})
    return tmp_path


def test_post_starts_both_setups_with_the_prompt(fake_launch):
    client = TestClient(app)
    res = client.post("/research-runs", json={"prompt": "  Beat the baseline.  ", "generations": 2})
    assert res.status_code == 201
    run_id = res.json()["run_id"]
    (started,) = launch._launches
    assert started.run_id == run_id
    for proc, setup in zip(started.procs, launch.SETUPS, strict=True):
        assert proc.cmd[proc.cmd.index("--setup") + 1] == setup
        assert proc.cmd[proc.cmd.index("--goal") + 1] == "Beat the baseline."
        assert proc.cmd[proc.cmd.index("--generations") + 1] == "2"
    # one run at a time
    again = client.post("/research-runs", json={"prompt": "again"})
    assert again.status_code == 409
    assert again.json()["detail"]["run_id"] == run_id


def test_post_refuses_when_launch_is_off(fake_launch, monkeypatch):
    monkeypatch.setattr(get_settings(), "research_launch_enabled", False)
    assert TestClient(app).post("/research-runs", json={"prompt": "x"}).status_code == 403
    assert launch._launches == []
