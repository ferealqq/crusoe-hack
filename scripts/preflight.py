"""Check that every key and service a research run needs actually works.

    uv run python scripts/preflight.py [--dataset churn]

Crusoe: both model IDs are listed and answer a chat. You.com: one search (skipped when
YDC_API_KEY is unset). Modal: the baseline train.py runs in a research sandbox and a test
sandbox, and the research sandbox holds no val labels or test data.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import scoring  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.lab import Dataset  # noqa: E402
from app.services import inference, sandbox, search  # noqa: E402

LIST_FILES = """import os
def fit(train, target, seed):
    print("FILES", sorted(os.listdir("/data")), sorted(os.listdir("/harness")))
    raise SystemExit(3)
def predict(model, X):
    return None
"""


async def check_crusoe() -> list[str]:
    s = get_settings()
    if not s.crusoe_api_key:
        return ["CRUSOE_API_KEY is not set"]
    ids = await inference.list_models()
    problems = [f"model {m} is not on Crusoe" for m in (s.researcher_model, s.curator_model)
                if m not in ids]  # fmt: skip
    for model, extra in (
        (s.researcher_model, {"chat_template_kwargs": {"enable_thinking": False}}),
        (s.curator_model, {"chat_template_kwargs": {"thinking": False}}),
    ):
        if model not in ids:
            continue
        msg, _, _ = await inference.chat_turn(
            [{"role": "user", "content": "Reply with the word ok."}],
            model,
            None,
            max_tokens=64,
            extra_body=extra,
        )
        print(f"  {model}: {(msg.content or '').strip()[:40]!r}")
    return problems


async def check_search() -> list[str]:
    if not get_settings().ydc_api_key:
        print("  YDC_API_KEY not set: researchers run without web_search")
        return []
    results = await search.web_search("lightgbm early stopping", 2)
    print(f"  {len(results)} results, first: {results[0]['title'] if results else None!r}")
    return [] if results else ["You.com returned no results"]


async def check_modal(dataset: Dataset) -> list[str]:
    code = (Path("baselines") / dataset.meta["name"] / "train.py").read_text()
    research, files, test = await asyncio.gather(
        sandbox.run_train(code, dataset.dir, 0, "research"),
        sandbox.run_train(LIST_FILES, dataset.dir, 0, "research"),
        dataset.test_losses(code, [1]),
    )
    problems = []
    if research.status != "ok" or research.preds is None:
        problems.append(f"research sandbox: {research.status}: {research.output[-500:]}")
    else:
        val = scoring.score(dataset.loss, dataset.val_y, research.preds)
        print(f"  baseline val {dataset.loss}: {val:.5f} ({research.runtime_s:.1f}s)")
    print(f"  baseline test {dataset.loss} (seed 1): {test[0]}")
    if test[0] is None:
        problems.append("test sandbox failed")
    visible = next((ln for ln in files.output.splitlines() if ln.startswith("FILES")), "")
    print(f"  research sandbox sees: {visible[6:]}")
    for secret in ("val_y", "test", "trainval"):
        if secret in visible:
            problems.append(f"research sandbox can see {secret}")
    if not visible:
        problems.append(f"could not list sandbox files: {files.output[-300:]}")
    return problems


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="churn")
    args = ap.parse_args()
    data_dir = Path(get_settings().data_dir) / args.dataset
    if not (data_dir / "meta.json").exists():
        print(f"missing {data_dir}: run scripts/prepare_data.py {args.dataset}")
        return 1
    dataset = Dataset(data_dir)
    failed = False
    for name, check in (
        ("Crusoe", check_crusoe()),
        ("You.com", check_search()),
        ("Modal", check_modal(dataset)),
    ):
        print(f"{name}:")
        try:
            problems = await check
        except Exception as e:
            problems = [f"{type(e).__name__}: {e}"]
        for p in problems:
            print(f"  FAIL {p}")
        print(f"  {'FAIL' if problems else 'ok'}")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
