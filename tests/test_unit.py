from fastapi.testclient import TestClient

from app.main import app
from app.services import inference
from app.store import append_jsonl, read_jsonl, write_jsonl


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_playground_models(monkeypatch):
    async def list_models():
        return ["openai/gpt-oss-120b"]

    monkeypatch.setattr(inference, "list_models", list_models)
    assert TestClient(app).get("/playground/models").json() == ["openai/gpt-oss-120b"]


def test_jsonl_store(tmp_path):
    path = tmp_path / "run" / "experiments.jsonl"
    assert read_jsonl(path) == []
    append_jsonl(path, {"id": 1, "note": "baseline"})
    append_jsonl(path, {"id": 2, "note": "añadir árboles"})
    assert [r["id"] for r in read_jsonl(path)] == [1, 2]
    write_jsonl(path, [{"id": 3}])
    assert read_jsonl(path) == [{"id": 3}]
