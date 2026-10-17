"""Run untrusted (model-generated) code in Modal Sandboxes."""

import asyncio
import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import modal
import numpy as np
from modal.types import FileType

from app import events
from app.config import get_settings
from app.schemas import SandboxResult

log = logging.getLogger(__name__)

# Extend with whatever the generated code is allowed to import.
SANDBOX_IMAGE = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "pytest")

# The libraries train.py may use (README: "Allowed libraries"). No installs at run time.
TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")
    .pip_install("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "xgboost", "catboost")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
)
EXPORT_IMAGE = TRAIN_IMAGE.pip_install("cloudpickle>=3.1")

SANDBOX_DIR = Path(__file__).resolve().parents[2] / "sandbox"

Mode = Literal["research", "test"]

# What each kind of sandbox gets. Research never gets val labels or any test data;
# test gets train + val with labels and only the test inputs.
MODE_FILES: dict[Mode, tuple[str, ...]] = {
    "research": ("train.parquet", "val_X.parquet", "meta.json"),
    "test": ("trainval.parquet", "test_X.parquet", "meta.json"),
}
MODE_SCRIPT: dict[Mode, str] = {"research": "harness.py", "test": "test.py"}

_app: modal.App | None = None
_images: dict[tuple[str, Mode], modal.Image] = {}
_semaphore: asyncio.Semaphore | None = None

# Fixed output names: never copy arbitrary sandbox paths or deserialize a model on the host.
EXPORT_FILES = {
    "model.pkl": 256 * 1024 * 1024,
    "runtime.json": 1024 * 1024,
    "requirements.txt": 128 * 1024,
}


class ModelExportError(Exception):
    """An actionable export failure, safe to include in API status."""


def _capacity() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(get_settings().sandbox_max_concurrency)
    return _semaphore


async def _get_app() -> modal.App:
    global _app
    if _app is None:
        s = get_settings()
        # Modal reads os.environ / ~/.modal.toml, not our .env; bridge tokens loaded by Settings.
        if s.modal_token_id and s.modal_token_secret:
            os.environ.setdefault("MODAL_TOKEN_ID", s.modal_token_id)
            os.environ.setdefault("MODAL_TOKEN_SECRET", s.modal_token_secret)
        if s.modal_environment:
            os.environ.setdefault("MODAL_ENVIRONMENT", s.modal_environment)
        _app = await modal.App.lookup.aio(s.modal_app_name, create_if_missing=True)
    return _app


async def run_python(code: str, timeout: int | None = None) -> SandboxResult:
    timeout = timeout or get_settings().sandbox_timeout_seconds
    sb = await modal.Sandbox.create.aio(
        app=await _get_app(),
        image=SANDBOX_IMAGE,
        timeout=timeout + 30,
        block_network=True,
    )
    try:
        p = await sb.exec.aio("python", "-c", code, timeout=timeout)
        stdout = await p.stdout.read.aio()
        stderr = await p.stderr.read.aio()
        await p.wait.aio()
        return SandboxResult(stdout=stdout, stderr=stderr, exit_code=p.returncode)
    finally:
        await sb.terminate.aio()


def sandbox_files(data_dir: Path, mode: Mode) -> dict[Path, str]:
    """Local file -> path inside the sandbox, for one kind of run."""
    files = {data_dir / name: f"/data/{name}" for name in MODE_FILES[mode]}
    files[SANDBOX_DIR / MODE_SCRIPT[mode]] = f"/harness/{MODE_SCRIPT[mode]}"
    return files


def _image(data_dir: Path, mode: Mode) -> modal.Image:
    key = (str(data_dir.resolve()), mode)
    if key not in _images:
        image = TRAIN_IMAGE
        for local, remote in sandbox_files(data_dir, mode).items():
            image = image.add_local_file(local, remote)
        _images[key] = image
    return _images[key]


@dataclass
class TrainResult:
    status: Literal["ok", "error", "timeout"]
    preds: np.ndarray | None
    output: str
    runtime_s: float


async def run_train(code: str, data_dir: Path, seed: int, mode: Mode = "research") -> TrainResult:
    """Run one train.py in a fresh sandbox and return its predictions.

    Sandbox infrastructure errors are retried; errors in `code` are returned as status
    "error" (they use up the run).
    """
    async with _capacity():
        for attempt in range(3):
            try:
                return await _run_train_once(code, data_dir, seed, mode)
            except modal.exception.Error as e:
                if attempt == 2:
                    raise
                log.warning("sandbox infrastructure error, retrying: %s", e)
                events.emit("sandbox.retry", error_type=type(e).__name__, attempt=attempt + 1)
                await asyncio.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


async def _run_train_once(code: str, data_dir: Path, seed: int, mode: Mode) -> TrainResult:
    s = get_settings()
    t = s.sandbox_timeout_seconds
    sb = await modal.Sandbox.create.aio(
        app=await _get_app(),
        image=_image(data_dir, mode),
        timeout=t + 120,
        block_network=True,
        cpu=(s.sandbox_cpu, s.sandbox_cpu),
        memory=(s.sandbox_memory_mb, s.sandbox_memory_mb),
        workdir="/work",
    )
    try:
        await sb.filesystem.write_text.aio(code, "/work/train.py")
        if mode == "research":
            events.emit("experiment.started", sandbox_mode=mode)
        started = time.perf_counter()
        p = await sb.exec.aio(
            "python", f"/harness/{MODE_SCRIPT[mode]}", "--seed", str(seed), timeout=t
        )
        out, err = await asyncio.gather(p.stdout.read.aio(), p.stderr.read.aio())
        exit_code = await p.wait.aio()
        runtime = time.perf_counter() - started
        output = out + err
        if exit_code == -1:
            return TrainResult("timeout", None, output + f"\n[timed out after {t}s]", runtime)
        if exit_code != 0:
            if exit_code == 137:
                output += "\n[killed: out of memory (4 GB) or terminated]"
            return TrainResult("error", None, output, runtime)
        raw = await sb.filesystem.read_bytes.aio("/out/preds.npy")
        preds = np.load(io.BytesIO(raw), allow_pickle=False)
        return TrainResult("ok", preds, output, runtime)
    finally:
        await sb.terminate.aio()


def export_files(data_dir: Path) -> dict[Path, str]:
    """Final fitting has train + val labels, and no test inputs or labels."""
    return {
        data_dir / "trainval.parquet": "/data/trainval.parquet",
        data_dir / "meta.json": "/data/meta.json",
        SANDBOX_DIR / "export.py": "/harness/export.py",
        SANDBOX_DIR / "predict.py": "/harness/predict.py",
    }


async def export_model(code: str, data_dir: Path, seed: int, output_dir: Path) -> None:
    """Refit and copy opaque artifacts to output_dir. The caller publishes them atomically."""
    async with _capacity():
        s = get_settings()
        t = s.sandbox_timeout_seconds
        sb = await modal.Sandbox.create.aio(
            app=await _get_app(),
            image=EXPORT_IMAGE,
            timeout=t + 120,
            block_network=True,
            cpu=(s.sandbox_cpu, s.sandbox_cpu),
            memory=(s.sandbox_memory_mb, s.sandbox_memory_mb),
            workdir="/work",
        )
        try:
            # Copy current files rather than caching a dataset image across export requests.
            for local, remote in export_files(data_dir).items():
                await sb.filesystem.copy_from_local.aio(local, remote)
            await sb.filesystem.write_text.aio(code, "/work/train.py")
            p = await sb.exec.aio("python", "/harness/export.py", "--seed", str(seed), timeout=t)
            out, err = await asyncio.gather(p.stdout.read.aio(), p.stderr.read.aio())
            exit_code = await p.wait.aio()
            if exit_code != 0:
                log.warning("model export failed (exit %s): %s", exit_code, (out + err)[-2000:])
                if exit_code == -1:
                    raise ModelExportError(f"Model fitting or serialization timed out after {t}s")
                raise ModelExportError(
                    f"Model fitting or serialization failed in sandbox (exit code {exit_code})"
                )
            for name, limit in EXPORT_FILES.items():
                remote = f"/out/{name}"
                info = await sb.filesystem.stat.aio(remote)
                if info.type != FileType.FILE or not 0 < info.size <= limit:
                    raise ModelExportError(f"Invalid or oversized export file: {name}")
                await sb.filesystem.copy_to_local.aio(remote, output_dir / name)
                if not 0 < (output_dir / name).stat().st_size <= limit:
                    raise ModelExportError(f"Invalid or oversized export file: {name}")
        finally:
            await sb.terminate.aio()
