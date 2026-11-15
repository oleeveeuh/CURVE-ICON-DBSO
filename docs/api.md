# Inference service (Phase 2): synthetic model, API, and container

> **Research-only demonstration.** Everything on this page runs on
> deterministic **synthetic** data. The served model is trained on the
> synthetic cohort generator, makes no claim about real outcomes, and is
> never a clinical tool. There is **no** `/explain` endpoint in Phase 2
> ([ADR-002](decisions/ADR-002-explanation-endpoint.md)). The
> institutional real-data track remains disabled
> ([docs/approvals.md](approvals.md)).

This document covers: locked local setup, deterministic training, the
FastAPI service, the Docker image, request examples, the artifact
manifest, and troubleshooting. Requirement IDs refer to
[docs/mlops_extension_plan.md](mlops_extension_plan.md).

---

## 1. Locked local setup

The committed `uv.lock` pins one environment for local runs, CI, and the
container (REP-003):

```bash
git clone https://github.com/oleeveeuh/CURVE-ICON-DBSO.git
cd CURVE-ICON-DBSO

uv sync --extra dev --extra api     # tests/lint + FastAPI service
source .venv/bin/activate
# plain pip works too: pip install -e ".[dev,api]"
```

Verify before working:

```bash
pytest -q                            # offline suite; no data needed
ruff check src tests scripts
```

## 2. Train the synthetic artifact (deterministic)

```bash
python scripts/train_synthetic_model.py
# equivalent module form (this is what the container build runs):
python -m curve_icon_dbso.training --output-dir artifacts/synthetic-demo
```

What it does:

1. Generates the deterministic synthetic cohort (seed 42, 8 subjects,
   9 rows) — the same generator the offline demo uses.
2. Runs the corrected leave-one-subject-out evaluation over the fixed
   candidates **and** the naive baselines; the manifest metrics are those
   honest out-of-fold numbers.
3. Applies the DD-006 promotion rule: lowest-MAE candidate; it is labeled
   `promotion_label: "performance"` only if it beats BOTH baselines on
   BOTH MAE and RMSE (`beats_baseline: true`), otherwise it is served as
   a clearly-labeled `demonstration_only` model (SCI-014).
4. Packages `model.pkl` + `schema.json` + `manifest.json` with verified
   hashes into `artifacts/synthetic-demo/` (gitignored, disposable —
   retrain any time).

**Determinism:** with the default offline path, two runs on the same
environment produce byte-identical artifacts (`model.pkl`,
`schema.json`, `manifest.json` — including `run_id`, a deterministic
fingerprint of source SHA / lock hash / schema hash / model hash / seed).
The optional `--mlflow` flag logs lineage to the local synthetic store
and stamps the real MLflow run ID instead — that path is intentionally
outside the byte-determinism promise.

The artifact refuses to overwrite an existing directory; build a new
versioned directory instead (artifacts are immutable once written).

## 3. Run the API locally

```bash
# artifact dir resolution: create_app() argument > $CURVE_ARTIFACT_DIR
# > ./artifacts/synthetic-demo
uvicorn curve_icon_dbso.api.main:app --host 127.0.0.1 --port 8000 \
    --workers 1 --no-access-log
```

`--workers 1` is the required default (API-010); `--no-access-log` keeps
the log stream to the service's structured metadata-only lines.

Interactive OpenAPI docs are served at `/docs` (the schema itself is
public-safe synthetic-track material).

## 4. API contract

| Endpoint | Purpose | Success | Failure |
|---|---|---|---|
| `GET /health` | Liveness + readiness state (API-002) | 200 always while the process is alive; body carries `ready`, versions, `error_class` | — (never fails while alive) |
| `GET /ready` | Readiness probe for orchestrators (CON-003) | 200 once the artifact is loaded and hash-verified | 503 + `error_class` |
| `GET /model-info` | The approved manifest fields (API-003) | 200, manifest JSON | 503 when not ready |
| `POST /predict` | Ordered batch prediction (API-004) | 200, predictions in input order | 422/400/413/503 (see error classes) |
| `GET /explain` | **Intentionally absent** (ADR-002) | — | 404 |

### `/predict` request

```json
{
  "rows": [
    {
      "hfo_power_mean": 1.55,
      "hfo_peak_freq_mean": 206.2,
      "Brady_Asymmetry": 0.04,
      "Rigidity_Asymmetry": 0.33,
      "Tremor_Asymmetry": 0.56,
      "years_since_diagnosis": 18.0,
      "age": 62.0
    }
  ]
}
```

Rules (schema v1.0.0, enforced by Pydantic v2 from the canonical
definition in `src/curve_icon_dbso/schema.py`):

* exactly the 7 prespecified features, complete rows — the service does
  not impute missing features;
* finite JSON numbers only: strings, booleans, `NaN`/`Infinity`, and
  `null` are rejected; integers are accepted;
* per-feature limits (`feature_limits` in the schema document);
* batch of 1–256 rows; predictions are returned in input order;
* `extra="forbid"` everywhere — unknown fields and subject identifiers
  are rejected, never silently dropped (API-006).

### `/predict` response

```json
{
  "predictions": [29.53, 23.98],
  "n_rows": 2,
  "model_version": "synthetic-demo-1.0.0",
  "schema_version": "1.0.0",
  "schema_hash": "598fdccb…"
}
```

### Error responses (API-005)

Errors carry a stable `error_class`, a fixed message, and per-field
`loc` paths — **never payload values**:

```json
{
  "error_class": "out_of_range",
  "message": "request failed schema validation; see errors for field paths",
  "errors": [
    {"loc": ["body", "rows", 0, "age"], "error_class": "out_of_range",
     "message": "value is outside the schema limit"}
  ]
}
```

| Status | `error_class` | Trigger |
|---|---|---|
| 400 | `malformed_json` | body is not valid JSON |
| 413 | `payload_too_large` | body > 1 MB (largest valid batch ≈ 60 KB) |
| 422 | `missing_field` | a required feature / `rows` missing |
| 422 | `unknown_field` | unknown field or identifier (row or body level) |
| 422 | `invalid_type` | string/bool/null where a number is required |
| 422 | `non_finite_value` | `NaN`/`Infinity` |
| 422 | `out_of_range` | value outside the feature's schema limit |
| 422 | `empty_batch` | `rows: []` |
| 422 | `batch_too_large` | more than 256 rows |
| 422 | `invalid_row` | a row that is not a JSON object |
| 422 | `invalid_body` | body that is not a JSON object |
| 422 | `invalid_type` | `rows` not a JSON array |
| 503 | `artifact_missing` / `artifact_corrupt` / `artifact_incompatible` / `not_started` | artifact not loaded (MOD-003: fail closed) |
| 500 | `internal_error` | unexpected error (logged as class + exception type only) |

### Logging (API-008/009)

The service logs one JSON line per request to stdout containing exactly:
`request_id`, `route`, `status`, `duration_ms`, `row_count`,
`schema_hash`, `model_version`, `error_class` (the framework prefix adds
a timestamp). Request/response bodies, feature values, predictions, and
infrastructure details are never logged. Startup lines carry only
`schema_hash`/`schema_version`/`model_version` on success or the
`error_class` on failure.

## 5. curl walkthrough

```bash
BASE=http://127.0.0.1:8000

curl -s "$BASE/health" | jq
curl -s "$BASE/ready" | jq
curl -s "$BASE/model-info" | jq '.model_version, .beats_baseline, .intended_use'

ROWS=$(python -c "import json; from curve_icon_dbso.api.payloads import \
    synthetic_payload_rows; print(json.dumps({'rows': synthetic_payload_rows(4)}))")
curl -s -X POST "$BASE/predict" -H 'content-type: application/json' -d "$ROWS" | jq

# rejected: unknown field, no values echoed
curl -s -X POST "$BASE/predict" -H 'content-type: application/json' \
     -d '{"rows":[{"hfo_power_mean":1.0}]}' | jq
```

The synthetic payload helpers (`api.payloads`) also provide
`boundary_payload_rows()` (rows exactly at the schema limits) and
`shifted_payload_rows()` (distribution-shift fixtures for the Phase 5
drift demonstration — not valid requests).

## 6. Artifact package and manifest

```
artifacts/synthetic-demo/
├── model.pkl      # pickled sklearn Pipeline (imputer → scaler → estimator)
├── schema.json    # the canonical schema document (schema.py)
└── manifest.json  # the model manifest (below)
```

`artifact.py` computes three hashes: `model_sha256` and `schema_sha256`
(file bytes), and `artifact_sha256` — one digest binding the two, defined
as `sha256("model.pkl=<model_sha>\nschema.json=<schema_sha>\n")`.

The loader (`artifact.load_artifact`) verifies, in order: files exist →
manifest is complete JSON with all required keys → file hashes match →
binding hash matches → schema document equals the deployed schema
(version, features, limits) → library versions are compatible → and only
then is the pickle deserialized. Any failure raises `ArtifactError` with
a stable `error_class`, and the API fails readiness (MOD-003) — it never
serves an artifact it cannot vouch for.

Manifest fields:

| Field | Meaning |
|---|---|
| `manifest_version` | manifest format version (`1`) |
| `model_version` | artifact identity (`synthetic-demo-1.0.0`) |
| `model_class` / `estimator_class` | served object and final estimator FQNs |
| `library_versions` | python/numpy/pandas/scikit-learn/scipy at training; exact-match checked for the numerics stack at load |
| `schema_version` / `schema_hash` / `schema_sha256` | which schema the artifact speaks (v1.0.0 / canonical document hash / file hash) |
| `training_seed`, `training_rows`, `training_subjects` | seed 42, 9 rows, 8 subjects |
| `data_provenance` | `synthetic — pipeline verification only, NOT study results` |
| `intended_use` | `research only; not for clinical use` |
| `prohibited_uses` | explicit list (clinical use, real data, biomarker claims, …) |
| `selection_rule` / `primary_metric` / `promotion_label` | DD-006 rule, MAE, `performance` or `demonstration_only` |
| `fold_strategy` | splitter used for the manifest metrics (LOSO, 8 folds) |
| `metrics` | out-of-fold MAE/RMSE/R²/Pearson r for the selected model |
| `baselines` | the same metrics for the mean/median `DummyRegressor`s |
| `candidates` | all fixed candidates' metrics (comparison transparency) |
| `beats_baseline` | strict rule: beat both baselines on MAE **and** RMSE |
| `run_id` | deterministic fingerprint (`synthetic-train-…`) or MLflow run ID |
| `source_sha` / `lock_hash` | git commit and `uv.lock` fingerprints |
| `analysis_stage` | `exploratory` / `confirmatory` (SCI-011) |
| `research_only` / `not_for_clinical_use` | always `true` |

## 7. Docker

```bash
SHA=$(git rev-parse HEAD)
docker build \
  --build-arg SOURCE_SHA="$SHA" \
  -t curve-icon-dbso-api:"$(git rev-parse --short=12 HEAD)" .
docker image ls curve-icon-dbso-api          # size check (<600 MB target)
docker run --rm -p 8000:8000 \
  curve-icon-dbso-api:"$(git rev-parse --short=12 HEAD)"
```

Properties:

* **Multi-stage**: a `uv` builder syncs the committed `uv.lock`
  (`uv sync --locked --no-dev --extra api`) and trains the artifact
  INSIDE the build; the runtime stage copies only `.venv` + `artifact`
  onto `python:3.11-slim-bookworm` — no build tooling, no uv.
* **No model bytes from the context**: `.dockerignore` excludes
  `artifacts/` and every `*.pkl`, so the image cannot contain a
  real-data-derived artifact by construction; the baked-in model is the
  deterministic synthetic one.
* **Non-root**: runtime user `10001:10001`, no home, `nologin` shell.
* **One worker** (`--workers 1`), access log off.
* **HEALTHCHECK** polls `/ready` (readiness-gated, CON-003).
* **Immutable tagging (CON-006)**: tag by git SHA as above; deploy by
  digest. `latest`-style tags are convenience aliases only.
* **Build-context exclusions**: data (`data/`, `outputs/`, `*.csv`,
  `*.mat`, `*.edf`, `*.xlsx`, …), experiment stores (`mlruns/`,
  `mlartifacts/`), state (`.git/`, `.terraform/`, `*.tfstate`),
  credentials (`.env`, `*.pem`, `*.key`), caches, virtualenvs, and
  runtime-irrelevant content (`tests/`, `docs/`, `scripts/`, `legacy/`).
  `tests/test_container.py` pins these categories and simulates the
  filtered context; the Docker-gated integration tests build/run/scan
  the image when a daemon is available.

Targets (CON-005): image < 600 MB; local cold start < 2 s. Both are
**measurements on a specific machine, not universal guarantees** —
re-measure on your hardware (see §9).

## 8. Privacy boundary

* The served artifact is trained on synthetic data only; a real-data
  artifact is restricted derived data and must never be built here or
  served by this code ([docs/data_access.md](data_access.md)).
* `/predict` rejects identifier-bearing payloads structurally (unknown
  fields are forbidden).
* Logs are metadata-only (§4); nothing that contains a payload value is
  ever written to stdout.
* The public synthetic track (repository, personal AWS in Phase 4,
  images) may contain only synthetic artifacts
  ([docs/approvals.md](approvals.md) gates everything else).

## 9. Measured on the development machine (2026-11-14)

* Cold start (launch `uvicorn` → first 200 from `/ready`), 5 runs,
  Python 3.11.17 venv, Apple Silicon macOS: **median 1.15 s**
  (min 1.04 s, max 1.82 s) — under the 2 s target with visible variance.
* Container image size / containerized cold start: **not measured** —
  no container runtime was available on the machine where this phase was
  implemented. The Docker-gated integration tests
  (`tests/test_container.py`) measure both automatically wherever a
  Docker daemon exists; an estimate from the locked dependency set
  (python-slim + numpy/pandas/scipy/scikit-learn/matplotlib + FastAPI
  stack, compiled bytecode) is ≈ 450–550 MB, comfortably under the
  600 MB budget — but treat it as an estimate until measured.

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/ready` → 503 `artifact_missing` | artifact dir absent or files missing | train it: `python scripts/train_synthetic_model.py`; check `$CURVE_ARTIFACT_DIR` |
| `/ready` → 503 `artifact_corrupt` | file bytes changed after training (hash mismatch) | retrain into a fresh directory; never edit an artifact |
| `/ready` → 503 `artifact_incompatible` | schema or library drift between training and serving | retrain with the current code/lock; keep the serving env locked |
| 422 `unknown_field` | extra key or identifier in the payload | send exactly the 7 schema features (see `/docs` or schema.json) |
| 422 `non_finite_value` | NaN/Infinity in the JSON | send finite JSON numbers only |
| 413 `payload_too_large` | body > 1 MB | batch ≤ 256 rows per request |
| `FileNotFoundError: uv.lock` / version drift at build | building outside the repo root | build from the repository root so `uv sync --locked` sees the lock |
| `Refusing to overwrite an existing artifact` | artifact dirs are immutable | use a new `--output-dir` (or delete the old one deliberately) |
| No log lines from the service | another handler owns stdout logging | the service logger is `curve_icon_dbso.api` at INFO with its own stdout handler |

---

*Scope reminder: this service is an engineering demonstration of a
validated inference path. It is not a validated model, and nothing on
this page is a performance claim about real patients or outcomes.*
