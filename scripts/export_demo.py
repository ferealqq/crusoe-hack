"""Export the finished research runs in data/ as the static demo (web/public/samples).

    uv run python scripts/export_demo.py            # every finished run
    uv run python scripts/export_demo.py --all      # also runs still in progress

Writes index.json (run metadata, newest first) and <run_id>.jsonl (events, app/replay.py) for
each run, replacing what was there. The demo area of the UI reads only these files.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import replay  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "web" / "public" / "samples"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="include runs still in progress")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    snaps = {
        run_id: s
        for run_id, s in replay.snapshots().items()
        if args.all or s.info["status"] == "completed"
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.jsonl"):
        old.unlink()
    for run_id, s in snaps.items():
        lines = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in s.events)
        (args.out / f"{run_id}.jsonl").write_text(lines)
        print(f"{run_id}: {s.info['status']}, {len(s.events)} events, setups {s.info['setups']}")
    (args.out / "index.json").write_text(
        json.dumps([s.info for s in snaps.values()], indent=1) + "\n"
    )
    print(f"wrote {len(snaps)} runs to {args.out}")


if __name__ == "__main__":
    main()
