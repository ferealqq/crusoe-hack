"""Build the report's data files (data/report/*.json) from the research runs and git trail.

uv run python scripts/report_data.py
"""

import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

DATA = Path("data")
OUT = DATA / "report"
LABEL = {"shared_brain": "Shared Brain", "no_memory": "No memory", "one_shot": "One-shot"}
KIND_ORDER = {"best": 0, "works": 1, "try_next": 2, "failed": 3}


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def memory_trail(run_id: str, dataset: str) -> list[dict[str, Any]]:
    """Every memory change of one run, parsed from its chore(brain) commit messages."""
    log = subprocess.run(
        [
            "git",
            "log",
            "--reverse",
            "--format=%h|%ad|%s",
            "--date=format:%H:%M:%S",
            f"--grep=\\[{run_id}\\]",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    rows: list[dict[str, Any]] = []
    for line in log:
        sha, time, subject = line.split("|", 2)
        msg = re.sub(r" \[[^\]]+\]$", "", subject.removeprefix("chore(brain): "))
        row = {"dataset": dataset, "commit": sha, "time": time}
        if m := re.match(r"g(\d+) curator (add|edit|remove) (m\d+) \((\w+)\): (.*)", msg):
            row |= {
                "generation": int(m[1]),
                "change": m[2],
                "entry": m[3],
                "kind": m[4],
                "claim": m[5][:200],
            }
        elif m := re.match(r"g(\d+) best -> (v\d+): (.*)", msg):
            row |= {
                "generation": int(m[1]),
                "change": "new best",
                "entry": m[2],
                "kind": "best",
                "claim": m[3],
            }
        elif m := re.match(r"g(\d+) cap: (.*)", msg):
            row |= {
                "generation": int(m[1]),
                "change": "dropped (cap)",
                "entry": "",
                "kind": "",
                "claim": m[2],
            }
        else:
            continue
        rows.append({"step": f"{dataset}-{len(rows) + 1}", **row})
    return rows


def main() -> None:
    runs, gens, exps, tests, trail, memory = [], [], [], [], [], []
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in jsonl(DATA / "research_runs.jsonl"):  # the latest run of each dataset and setup
        latest[(r["dataset"], r["setup"])] = r
    for (dataset, setup), r in sorted(latest.items()):
        d = DATA / "research_runs" / r["run_id"]
        research = [
            e for e in jsonl(d / "experiments.jsonl") if e["split"] == "val" and e["generation"] > 0
        ]
        wasted = sum(e["status"] != "ok" for e in research)
        s = LABEL[setup]
        runs.append(
            {
                "dataset": dataset,
                "setup": s,
                "run_id": r["run_id"],
                "loss": r.get("loss", "log_loss"),
                "fitness": round(r["fitness"], 5),
                "val_gain": round(r["val_gain"], 5),
                "start_val_loss": round(r["l0"], 5),
                "best_val_loss": round(r["best_val_loss"], 5),
                "test_start_loss": round(statistics.mean(r["test_start"]), 5),
                "test_best_loss": round(statistics.mean(r["test_best"]), 5),
                "sandbox_runs": len(research),
                "wasted_runs": wasted,
                "tokens_prompt": r["usage"]["prompt"],
                "tokens_cached": r["usage"]["cached"],
                "tokens_output": r["usage"]["output"],
                "memory_tokens": r["memory_tokens"],
                "noise_threshold": round(r["noise_threshold"], 5),
                "best_run_id": r["best_run_id"],
            }
        )
        for seed, a, b in zip(
            r["config"]["test_seeds"], r["test_start"], r["test_best"], strict=True
        ):
            tests.append(
                {
                    "key": f"{dataset}/{s}/{seed}",
                    "dataset": dataset,
                    "setup": s,
                    "seed": seed,
                    "start_loss": round(a, 5),
                    "best_loss": round(b, 5),
                    "gain": round((a - b) / a, 5),
                }
            )
        gens.append(
            {
                "dataset": dataset,
                "setup": s,
                "generation": 0,
                "best_val_loss": round(r["l0"], 5),
                "val_gain": 0.0,
                "sandbox_runs": 0,
                "wasted_runs": 0,
                "memory_tokens": 0,
            }
        )
        for g in jsonl(d / "generations.jsonl"):
            n = g["generation"]
            gens.append(
                {
                    "dataset": dataset,
                    "setup": s,
                    "generation": n,
                    "best_val_loss": round(g["end_best"], 5),
                    "val_gain": round(g["val_gain"], 5),
                    "sandbox_runs": g["runs"],
                    "wasted_runs": sum(
                        e["status"] != "ok" for e in research if e["generation"] == n
                    ),
                    "memory_tokens": g.get("memory_tokens_after", g["memory_tokens"]),
                }
            )
        for e in research:
            exps.append(
                {
                    "dataset": dataset,
                    "setup": s,
                    "generation": e["generation"],
                    "run_id": e["run_id"],
                    "agent": e["agent"],
                    "role": "curator" if e["agent"] == "curator" else "researcher",
                    "status": e["status"],
                    "val_loss": round(e["loss"], 5) if e["loss"] is not None else None,
                    "note": e["note"][:140],
                }
            )
        if setup == "shared_brain":
            trail += memory_trail(r["run_id"], dataset)
            final = sorted(
                jsonl(d / "memory.jsonl"), key=lambda e: (KIND_ORDER[e["kind"]], int(e["id"][1:]))
            )
            memory += [
                {
                    "dataset": dataset,
                    "entry": f"{dataset}-{e['id']}",
                    "id": e["id"],
                    "kind": e["kind"],
                    "claim": e["claim"],
                    "tested_on": f"v{e['tested_on']}",
                }
                for e in final
            ]

    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "runs": runs,
        "generations": gens,
        "experiments": exps,
        "memory_trail": trail,
        "memory_final": memory,
        "test_scores": tests,
    }
    for name, rows in files.items():
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1) + "\n")
        print(f"{OUT / name}.json: {len(rows)} rows")


if __name__ == "__main__":
    main()
