"""Research results as files under data/runs/<run>/ (git-ignored). JSONL files are append-only."""

import json
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def append(self, name: str, record: dict[str, Any]) -> None:
        with (self.root / f"{name}.jsonl").open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read(self, name: str) -> list[dict[str, Any]]:
        path = self.root / f"{name}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def write_json(self, name: str, obj: Any) -> None:
        (self.root / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))

    def read_json(self, name: str) -> Any:
        return json.loads((self.root / f"{name}.json").read_text())
