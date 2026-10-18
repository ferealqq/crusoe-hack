"""The demo home page shows the real starting prompt: web/src/lib/prompt.ts mirrors the code."""

import json
from pathlib import Path

from app.config import DEFAULT_GOAL

ROOT = Path(__file__).resolve().parents[1]


def test_web_prompt_matches_researcher_rules_and_dataset_card():
    ts = (ROOT / "web/src/lib/prompt.ts").read_text()
    meta = json.loads((ROOT / "data/churn/meta.json").read_text())
    assert DEFAULT_GOAL in ts
    assert meta["task"] in ts
    assert str(meta["rows"]["train"]) in ts
