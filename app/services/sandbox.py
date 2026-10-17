"""Run untrusted (model-generated) code in Modal Sandboxes."""

import json
import os
import time
from pathlib import Path

import modal

from app.config import get_settings
from app.schemas import SandboxResult, TrainResult

# Everything train.py may import (README "The train.py contract"); no installs at run time.
SANDBOX_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")  # OpenMP, for lightgbm/xgboost
    .pip_install("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "xgboost", "catboost")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
)
APP_NAME = "crusoe-hack-sandboxes"
HARNESS = Path(__file__).resolve().parents[2] / "sandbox" / "harness.py"
WORKDIR = "/sandbox"
OUTPUT_TAIL_CHARS = 2000

_app: modal.App | None = None


async def _get_app() -> modal.App:
    global _app
    if _app is None:
        s = get_settings()
        # Modal reads os.environ / ~/.modal.toml, not our .env. Tokens in .env win over the
        # active ~/.modal.toml profile, so sandboxes bill to this project's workspace.
        if s.modal_token_id and s.modal_token_secret:
            os.environ.setdefault("MODAL_TOKEN_ID", s.modal_token_id)
            os.environ.setdefault("MODAL_TOKEN_SECRET", s.modal_token_secret)
            # The active profile's `environment` belongs to another workspace; "" = default.
            os.environ.setdefault("MODAL_ENVIRONMENT", "")
        _app = await modal.App.lookup.aio(APP_NAME, create_if_missing=True)
    return _app


async def _create(timeout: int) -> modal.Sandbox:
    s = get_settings()
    return await modal.Sandbox.create.aio(
        app=await _get_app(),
        image=SANDBOX_IMAGE,
        timeout=timeout + 60,
        block_network=True,
        cpu=(s.sandbox_cpu, s.sandbox_cpu),
        memory=(s.sandbox_memory_mb, s.sandbox_memory_mb),
        workdir=WORKDIR,
    )


async def run_python(code: str, timeout: int | None = None) -> SandboxResult:
    timeout = timeout or get_settings().sandbox_timeout_seconds
    sb = await _create(timeout)
    try:
        p = await sb.exec.aio("python", "-c", code, timeout=timeout)
        stdout = await p.stdout.read.aio()
        stderr = await p.stderr.read.aio()
        await p.wait.aio()
        return SandboxResult(stdout=stdout, stderr=stderr, exit_code=p.returncode)
    finally:
        await sb.terminate.aio()


async def run_train(
    code: str, files: dict[str, bytes], target: str, seed: int, loss: str = "log_loss"
) -> TrainResult:
    """Run train.py through the locked harness: fit on train.parquet, predict X.parquet.

    Exec timeout -> exit code -1 ("timeout"); anything else non-zero is "error".
    """
    timeout = get_settings().sandbox_timeout_seconds
    started = time.perf_counter()
    try:
        sb = await _create(timeout)
    except Exception as e:  # Modal failed to start the sandbox; still uses the run
        return TrainResult(status="error", output_tail=f"sandbox failed to start: {e}"[-2000:])
    try:
        fs = sb.filesystem
        await fs.write_text.aio(HARNESS.read_text(), f"{WORKDIR}/harness.py")
        await fs.write_text.aio(code, f"{WORKDIR}/train.py")
        for name, data in files.items():
            await fs.write_bytes.aio(data, f"{WORKDIR}/{name}")
        p = await sb.exec.aio(
            "python",
            "harness.py",
            "--target",
            target,
            "--seed",
            str(seed),
            "--loss",
            loss,
            timeout=timeout,
        )
        stdout = await p.stdout.read.aio()
        stderr = await p.stderr.read.aio()
        exit_code = await p.wait.aio()
        output_tail = (stdout + stderr)[-OUTPUT_TAIL_CHARS:]
        runtime_s = round(time.perf_counter() - started, 1)
        if exit_code == -1:
            return TrainResult(status="timeout", output_tail=output_tail, runtime_s=runtime_s)
        if exit_code != 0:
            return TrainResult(status="error", output_tail=output_tail, runtime_s=runtime_s)
        preds = json.loads(await fs.read_text.aio(f"{WORKDIR}/preds.json"))
        return TrainResult(status="ok", preds=preds, output_tail=output_tail, runtime_s=runtime_s)
    except Exception as e:
        return TrainResult(
            status="error",
            output_tail=f"{type(e).__name__}: {e}"[-OUTPUT_TAIL_CHARS:],
            runtime_s=round(time.perf_counter() - started, 1),
        )
    finally:
        await sb.terminate.aio()
