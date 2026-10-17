"""On-demand trained model bundles, persisted beside the CLI's completed results."""

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import SETUPS, Setup
from app.schemas import ModelExportOut
from app.services import research_events, sandbox

log = logging.getLogger(__name__)
EXPORT_SEED = 1
BUNDLE_VERSION = 1


class ExportError(Exception):
    def __init__(self, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


class FinalResult(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    setup: Setup
    dataset: str
    best_code: str = Field(min_length=1)
    best_version: int = Field(ge=0)
    best_val: float
    l0: float
    noise_threshold: float | None = None
    val_gain: float | None = None
    baseline_test_losses: list[float | None] = Field(default_factory=list)
    best_test_losses: list[float | None] = Field(default_factory=list)
    fitness: float | None = None


@dataclass
class Source:
    run_id: str
    directory: Path
    data_root: Path
    result: FinalResult

    @property
    def export_dir(self) -> Path:
        return self.directory / "export"

    def initial_status(self) -> ModelExportOut:
        return ModelExportOut(
            run_id=self.run_id,
            setup=self.result.setup,
            status="not_created",
            best_version=self.result.best_version,
            seed=EXPORT_SEED,
            code_sha256=hashlib.sha256(self.result.best_code.encode()).hexdigest(),
        )


@dataclass
class ExportJob:
    source: Source
    data_dir: Path
    info: ModelExportOut
    lock_file: BinaryIO


def _safe_child(parent: Path, name: str) -> Path:
    path = parent / name
    if path.is_symlink() or path.resolve().parent != parent.resolve():
        raise ExportError("Unsafe export path", 404)
    return path


def load_source(data_dir: str, run_id: str, setup: Setup) -> Source:
    try:
        root = research_events.resolve_run(data_dir, run_id)
    except FileNotFoundError:
        raise ExportError("research run not found", 404) from None
    if setup not in SETUPS:
        raise ExportError("research setup not found", 404)
    directory = _safe_child(root, setup)
    if not directory.is_dir():
        config_path = _safe_child(root, "config.json")
        try:
            config = json.loads(config_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            config = {}
        if isinstance(config, dict) and setup in config.get("setups", []):
            raise ExportError("This setup has not completed yet")
        raise ExportError("research setup not found", 404)
    result_path = _safe_child(directory, "result.json")
    try:
        result = FinalResult.model_validate_json(result_path.read_text())
    except FileNotFoundError:
        raise ExportError("This setup has no completed result to export") from None
    except (ValidationError, UnicodeDecodeError, IsADirectoryError):
        raise ExportError("The saved final result is incomplete or invalid") from None
    if result.setup != setup or not result.best_code.strip():
        raise ExportError("The saved final result does not match this setup")
    source = Source(root.name, directory, Path(data_dir).resolve(), result)
    export_dir = _safe_child(directory, "export")
    for name in ("status.json", "model.zip", ".lock"):
        _safe_child(export_dir, name)
    return source


def _dataset(source: Source) -> Path:
    name = source.result.dataset
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}", name):
        raise ExportError("The saved dataset name is invalid")
    path = _safe_child(source.data_root, name)
    for name in ("trainval.parquet", "meta.json"):
        if not _safe_child(path, name).is_file():
            raise ExportError("Export requires the original trainval.parquet and meta.json")
    try:
        meta = json.loads((path / "meta.json").read_text())
        if meta["name"] != source.result.dataset or not isinstance(meta["target"], str):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        raise ExportError(
            "The prepared dataset metadata is invalid or does not match the run"
        ) from None
    return path


def _lock(directory: Path) -> BinaryIO | None:
    # A held OS lock prevents duplicates across API workers and is released on process death.
    fd = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    # File ownership also closes the descriptor if a request is cancelled before
    # its BackgroundTask is attached to a response.
    return os.fdopen(fd, "r+b")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    # The caller holds the export lock. Readers always see a complete JSON document.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def _save(source: Source, info: ModelExportOut) -> None:
    _write_json(source.export_dir / "status.json", info.model_dump(mode="json"))


def _read_status(source: Source) -> ModelExportOut:
    initial = source.initial_status()
    try:
        info = ModelExportOut.model_validate_json((source.export_dir / "status.json").read_text())
    except FileNotFoundError:
        return initial
    except (ValidationError, UnicodeDecodeError, IsADirectoryError):
        return initial.model_copy(
            update={"status": "failed", "error": "Invalid export status; retry export"}
        )
    if (info.run_id, info.setup, info.code_sha256, info.best_version, info.seed) != (
        initial.run_id,
        initial.setup,
        initial.code_sha256,
        initial.best_version,
        EXPORT_SEED,
    ):
        return initial
    if info.status == "ready":
        bundle = source.export_dir / "model.zip"
        if not bundle.is_file() or bundle.stat().st_size != info.size_bytes:
            return _failed(info, "Saved bundle is missing or incomplete; retry export")
    return info


def _failed(info: ModelExportOut, error: str) -> ModelExportOut:
    return info.model_copy(
        update={
            "status": "failed",
            "error": error,
            "finished_at": datetime.now(UTC),
            "download_url": None,
            "filename": None,
            "sha256": None,
            "size_bytes": None,
        }
    )


def status(source: Source) -> ModelExportOut:
    if not source.export_dir.exists():
        return source.initial_status()
    fd = _lock(source.export_dir)
    try:
        info = _read_status(source)
        if fd is not None and info.status == "building":
            info = _failed(info, "Export was interrupted; POST again to retry")
            _save(source, info)
        return info
    finally:
        if fd is not None:
            fd.close()


def prepare(source: Source) -> tuple[ModelExportOut, ExportJob | None]:
    source.export_dir.mkdir(exist_ok=True)
    fd = _lock(source.export_dir)
    if fd is None:
        # The first request may still be preparing its status file.
        info = _read_status(source)
        if info.status == "ready":
            return info, None
        return source.initial_status().model_copy(
            update={"status": "building", "created_at": info.created_at}
        ), None
    try:
        info = _read_status(source)
        if info.status == "ready":
            fd.close()
            return info, None
        data_dir = _dataset(source)
        info = source.initial_status().model_copy(
            update={
                "status": "building",
                "created_at": datetime.now(UTC),
            }
        )
        _save(source, info)
        return info, ExportJob(source, data_dir, info, fd)
    except BaseException:
        fd.close()
        raise


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


BUNDLE_README = """# Exported model

This bundle refits the accepted training code on train + validation with seed 1.
Use the Python version and Linux runtime recorded in runtime.json. Install libgomp1
for the boosting libraries, then install the pinned Python dependencies:

    python -m pip install -r requirements.txt
    python predict.py inputs.parquet predictions.csv

CSV inputs are also supported. Supply the feature columns and types in runtime.json.
The output has one `prediction` column. From Python:

    from predict import Predictor
    model = Predictor()
    predictions = model.predict(features_dataframe)

model.pkl contains the fitted model and its prediction function, serialized with
cloudpickle to preserve custom preprocessing and classes. train.py is the exact
accepted source. runtime.json records the export environment and feature schema.
metadata.json identifies the run, training data hashes, code hash, and original
evaluation metrics. Those metrics describe the research evaluation across its
recorded seeds; they are not a fresh test score for this refitted single-seed model.
"""


def _package(job: ExportJob, temp: Path, data_hashes: dict[str, str]) -> ModelExportOut:
    result, info = job.source.result, job.info
    for name, limit in sandbox.EXPORT_FILES.items():
        path = temp / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
            raise sandbox.ModelExportError(f"Invalid or oversized export file: {name}")
    # These are inert metadata, never a pickle/model load on the API host.
    runtime = json.loads((temp / "runtime.json").read_text())
    if runtime["seed"] != EXPORT_SEED or runtime["training_split"] != "train+val":
        raise sandbox.ModelExportError("Export runtime does not match the requested fit")
    metrics = result.model_dump(exclude={"best_code", "setup", "dataset", "best_version"})
    metadata = {
        "bundle_version": BUNDLE_VERSION,
        "run_id": info.run_id,
        "setup": info.setup,
        "dataset": result.dataset,
        "best_version": result.best_version,
        "is_improved": result.best_version > 0,
        "code_sha256": info.code_sha256,
        "seed": EXPORT_SEED,
        "training_split": "train+val",
        "training_files_sha256": data_hashes,
        "created_at": info.created_at.isoformat() if info.created_at else None,
        "evaluation_metrics": metrics,
        "evaluation_note": "Original research metrics, not a new score for this refitted model",
    }
    bundle = temp / "model.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in sandbox.EXPORT_FILES:
            z.write(temp / name, name)
        z.writestr("train.py", result.best_code)
        z.write(sandbox.SANDBOX_DIR / "predict.py", "predict.py")
        z.writestr("metadata.json", json.dumps(metadata, indent=2, allow_nan=False))
        z.writestr("README.md", BUNDLE_README)
    filename = f"{info.run_id}-{info.setup}-v{info.best_version}-model.zip"
    ready = info.model_copy(
        update={
            "status": "ready",
            "finished_at": datetime.now(UTC),
            "filename": filename,
            "size_bytes": bundle.stat().st_size,
            "sha256": _sha256(bundle),
            "download_url": f"/research-runs/{info.run_id}/setups/{info.setup}/export/download",
        }
    )
    bundle.replace(job.source.export_dir / "model.zip")
    _save(job.source, ready)
    return ready


async def build(job: ExportJob) -> None:
    try:
        with tempfile.TemporaryDirectory(prefix=".build-", dir=job.source.export_dir) as directory:
            temp = Path(directory)
            data_dir = temp / "data"
            data_hashes = await _thread(_snapshot_data, job.data_dir, data_dir)
            await sandbox.export_model(job.source.result.best_code, data_dir, EXPORT_SEED, temp)
            await _thread(_package, job, temp, data_hashes)
    except asyncio.CancelledError:
        _save(job.source, _failed(job.info, "Export was interrupted; POST again to retry"))
        raise
    except Exception as e:
        log.exception("export failed for %s/%s", job.source.run_id, job.info.setup)
        message = (
            str(e)
            if isinstance(e, sandbox.ModelExportError)
            else f"Export failed ({type(e).__name__}); retry export"
        )
        await _thread(_save, job.source, _failed(job.info, message))
    finally:
        job.lock_file.close()


def download(source: Source) -> tuple[Path, ModelExportOut]:
    info = status(source)
    if info.status != "ready":
        raise ExportError(info.error or "The model export is not ready for download")
    return source.export_dir / "model.zip", info


def _snapshot_data(source: Path, destination: Path) -> dict[str, str]:
    destination.mkdir()
    hashes = {}
    for name in ("trainval.parquet", "meta.json"):
        shutil.copyfile(_safe_child(source, name), destination / name)
        hashes[name] = _sha256(destination / name)
    return hashes


async def _thread(fn, *args):
    # Do not remove a temp directory or release the job lock while a cancelled
    # to_thread operation is still copying/publishing files in the background.
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
