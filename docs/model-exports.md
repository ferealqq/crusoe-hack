# Trained model export API

Export the accepted final model for one research setup as a downloadable ZIP.
The API refits the saved `result.json` → `best_code` on **train + validation** with
**seed 1** in a Modal sandbox, then preserves the fitted object and prediction
function with cloudpickle. Test inputs and labels are never sent to this sandbox.

## Endpoints

The base path is `/research-runs/{run_id}/setups/{setup}/export`.
`setup` is `one_shot`, `no_memory`, `raw_log`, or `shared_brain`.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | base path | Build an export, or return the existing export |
| `GET` | base path | Read export status |
| `GET` | base path + `/download` | Download the ready ZIP attachment |

POST takes no body. It returns **202** for a new or already-building export and
**200** for a cached ready export. Its `Location` header points to the status URL.
The build runs in the background; poll GET until `ready` or `failed`.
Repeated/concurrent POSTs reuse the same build, including across API workers.
POST again after `failed` to retry.

Example ready status (before creation, `status` is `not_created` and the artifact
fields and timestamps are null):

```json
{
  "run_id": "20261018-120000-000000-churn",
  "setup": "shared_brain",
  "status": "ready",
  "best_version": 2,
  "seed": 1,
  "code_sha256": "<SHA-256 of the winning train.py>",
  "created_at": "2026-10-18T12:30:00Z",
  "finished_at": "2026-10-18T12:31:00Z",
  "error": null,
  "filename": "20261018-120000-000000-churn-shared_brain-v2-model.zip",
  "size_bytes": 123456,
  "sha256": "<SHA-256 of the ZIP>",
  "download_url": "/research-runs/20261018-120000-000000-churn/setups/shared_brain/export/download"
}
```

The download responds with `Content-Type: application/zip`,
`Content-Disposition: attachment`, and the ZIP hash as its `ETag`.

```sh
base=http://localhost:8000/research-runs/REPLACE_RUN_ID/setups/shared_brain/export
curl -X POST "$base"
curl "$base"
# Once status is ready:
curl --fail "$base/download" -o model.zip
```

### Availability and errors

- **404:** unknown run/setup, or unsafe/symlinked paths.
- **422:** invalid setup name.
- **409:** setup has no valid final result, required training data is missing, or
  download was requested before the bundle is ready.
- Failures after POST was accepted appear as `status: failed` with an `error`
  message. A failed build never exposes a partial ZIP.

A setup is final when its valid `result.json` exists. This supports older runs
without telemetry and successful setups even if another setup failed. A run that
kept the baseline (`best_version: 0`) can also be exported; its bundle explicitly
records `is_improved: false`.

## Bundle contents and usage

| File | Contents |
| --- | --- |
| `model.pkl` | Fitted model and prediction function, including custom preprocessing/classes |
| `train.py` | Exact accepted source code |
| `predict.py` | Python `Predictor` API and CSV/Parquet prediction CLI |
| `requirements.txt` | Dependency versions from the fitting environment, including the CPU PyTorch index |
| `runtime.json` | Python/platform versions, packages, feature names/types, target, training row count and seed |
| `metadata.json` | Bundle version, run/setup/dataset, best version, source/training-file hashes and research metrics |
| `README.md` | Usage instructions |

Use the recorded Python version on Linux, install `libgomp1` and the pinned Python
dependencies, then:

```sh
python -m pip install -r requirements.txt
python predict.py inputs.parquet predictions.csv
```

Or use `from predict import Predictor; model = Predictor(); model.predict(frame)`.
Inputs must contain the feature columns with the types recorded in `runtime.json`.
The helper orders columns and returns one finite prediction per row. The bundle
is checked with this helper in a fresh process inside the export sandbox before
publication. The backend only copies opaque model bytes; it never unpickles them.

Evaluation metrics come from the original research result, which can aggregate
several seeds. They are labelled as research metrics, not a new measurement of
the exported single-seed fit. Model selection uses the saved accepted winner.
Dataset files, agent transcripts, and credentials are not included in the ZIP.

## Storage and runtime

- Uses the same `DATA_DIR` and Modal configuration as the research CLI. Creating
  an export requires `DATA_DIR/<dataset>/trainval.parquet` and `meta.json`; retain
  the dataset used by the run. Existing runs did not store historical dataset
  hashes, so exports record the hashes of the files actually used for refitting.
- Status and the cached bundle live under
  `DATA_DIR/runs/<run_id>/<setup>/export/`. Downloads of a ready bundle do not
  require Modal or the training data to remain present.
- Builds snapshot the two input files, use the configured sandbox timeout,
  CPU/memory limits and concurrency cap, and terminate the sandbox after copying
  artifacts. Model artifacts are limited to 256 MiB before ZIP compression.
- A filesystem lock prevents duplicate builds. The API workers must share the
  same filesystem with POSIX locking, as they already share the research files.
- These are in-process background jobs. On process death, the OS releases the
  lock; status reports the abandoned build as `failed`, and POST retries it.
  Interrupted jobs are not automatically resumed. Temporary `.build-*` directories
  left by a hard kill can be removed while no export is running.
- Status and ZIP publication use atomic replacement; failures and cancellation
  clean up temporary files during normal shutdown.

For a future UI: enable export for completed setups, POST on the user's action,
poll this status resource, and download `download_url` once ready. During recorded
replay, gate the action on the displayed setup's completion rather than exposing
the final export before that point in the replay.
