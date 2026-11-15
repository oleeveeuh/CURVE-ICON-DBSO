# CURVE-ICON-DBSO Modernization Requirements and Roadmap

**Document status:** Draft for design review; planning only  
**Prepared:** 2026-11-13  
**Implementation status:** Not started  
**Source:** Verified repository audit and user-provided extension proposal  
**Audited repository:** `/Users/olivialiau/CURVE-ICON-DBSO` at commit `eb45b11`, including the five preserved working-tree modifications listed below  
**Design approval:** DD-001 through DD-003 approved by the project owner on 2026-11-13

## 1. Purpose

This document defines the requirements, controls, delivery roadmap, and approval gates for modernizing the CURVE-ICON-DBSO research project. The goal is to demonstrate current scientific-computing and MLOps skills without reviving invalidated findings, overstating a small clinical dataset, or moving restricted data into an unapproved environment.

The modernization has two equally important outcomes:

1. A reproducible, leakage-resistant research evaluation workflow.
2. A production-style inference and deployment demonstration using synthetic data.

The work is an engineering and research prototype. It is not a clinical tool, diagnostic device, or evidence of clinical utility.

## 2. Guiding principles

1. **Scientific validity comes before deployment.** No cloud or API work may be presented as validating the underlying model.
2. **Historical errors remain visible.** The retracted result and its causes must be documented rather than silently replaced.
3. **Privacy is enforced structurally.** Real data and real-data-derived artifacts must be unable to enter a personal cloud account, public repository, container image, or public log.
4. **Synthetic and institutional tracks share code, not artifacts.** The same pipeline may operate in both tracks, but their data, models, logs, state, and accounts remain separate.
5. **Small-sample uncertainty is explicit.** Model comparisons, confidence intervals, and explanations must not be framed as biomarker or clinical claims.
6. **Infrastructure must be proportional to the workload.** Complexity is added only when it answers a concrete engineering need.
7. **Every phase is independently verifiable and reversible.** Each phase has entry criteria, deliverables, tests, an exit gate, and a rollback or cleanup procedure.

## 3. Corrected project baseline

The following corrections govern all requirements and public descriptions. They were reconciled against the repository at commit `eb45b11` and its current working tree on 2026-11-13.

| Historical or prompt premise | Corrected baseline | Required treatment |
|---|---|---|
| Approximately 10 subjects with EEG and clinical data | ECoG; reportedly 32 local recordings across 32 subjects. The historical outcome cohort was 8 subjects and 9 recordings. | State the cohort and unit of observation for each analysis. Do not use “10 subjects” without evidence. |
| 108 engineered features | The 108-feature claim was invalidated. The corrected outcome design reportedly uses 7 prespecified features; the proposed kinematic design uses 5. | Serving schemas and synthetic data must use the real 5–7 feature contract. |
| XGBoost treatment-outcome classifier | Existing models were reportedly regression models using scikit-learn RF, gradient boosting, and linear regression. No saved model artifacts exist. | If included, add XGBoost only as a fixed-parameter regressor in the same comparison. |
| 71% cross-validation accuracy | The value was regression R² = 0.713 and was retracted because feature selection, imputation, and scaling occurred before cross-validation. | Never call the value accuracy or report it as a result. |
| Corrected figures showed R² = -0.672 | The supplied audit attributes this value to fold-order predictions being paired with original-row-order targets. | Preserve an aligned out-of-fold regression test. |
| Existing SHAP analysis | No corrected-pipeline SHAP implementation reportedly exists; historical rankings included merge-suffix artifacts. | Explanations may describe model behavior only and may not support biomarker claims. |

### 3.1 Verified repository state

- Commit: `eb45b11`.
- Preserved modified files: `README.md`, `docs/methodology.md`, `src/curve_icon_dbso/hfo.py`, `tests/test_hfo.py`, and `tests/test_validation.py`.
- Preserved untracked planning files: `docs/mlops_extension_plan.md` and `docs/implementation_prompts.md` (the phased implementation prompt pack; added after the original audit and re-verified on 2026-11-13).
- Test result: **43 passed** using the repository virtual environment on 2026-11-13.
- CI: GitHub Actions tests Python 3.11 and 3.12 with Ruff, pytest, and the deterministic synthetic demonstration.
- Corrected evaluation: subject-grouped splitting, fold-local median imputation and scaling, fixed models, naive baselines, and index-aligned out-of-fold predictions are implemented.
- Outcome schema: seven prespecified features are implemented in `features.PRESPECIFIED_FEATURES`.
- Kinematic pivot: five predictors and the tapping-rate target are prespecified in `docs/methodology.md`, but the proposal states that the real-data run has not occurred.
- Missing extension surface (updated 2026-11-13, after Phase 1): environment lock (`uv.lock`), subject-cluster bootstrap, label-permutation null, and local MLflow lineage are now implemented. Still absent: API, Dockerfile, model artifact, Terraform, deployment workflow, Locust profile, and drift monitoring (Phases 2–5).

The README still says “40 offline tests”; the current suite contains 43 passing tests. This documentation mismatch should be corrected in the next approved repository edit.

### 3.2 Unit of observation

The subject is the independence unit. Recordings, REST/MOVE blocks, channels, epochs, and trials are repeated measures within a subject and must not be split across training and test partitions as independent observations.

If a subject contributes more than one recording, all of that subject’s recordings must remain in the same fold. Subject-level resampling must resample subjects, not rows, epochs, or channels.

### 3.3 Historical-result policy

The project documentation must retain a concise correction stating that:

- R² = 0.713 was a retracted regression result, not classification accuracy.
- Full-dataset feature selection and preprocessing leaked information into cross-validation.
- A separate prediction/row alignment error corrupted saved figures.
- Historical placeholder behavior may have attached synthetic HFO values to real subject identifiers.
- No modernized result is optimized to recover or approximate the retracted value.

Invalidated scripts and figures may remain for provenance only if clearly marked “historical,” “invalidated,” and “do not cite.”

## 4. Scope

### 4.1 In scope

- Restore and verify the repository baseline.
- Harden grouped evaluation and statistical uncertainty reporting.
- Lock the software environment.
- Add narrowly scoped experiment lineage tracking.
- Train and package a synthetic-only demonstration model.
- Build a validated FastAPI inference service and container image.
- Provision a synthetic-only AWS deployment through Terraform.
- Add OIDC-based continuous delivery, monitoring, drift demonstration, load testing, and cleanup controls.
- Document the authorization path for a future institutional deployment.
- Optionally demonstrate Kubernetes concepts locally if justified after measuring the ECS implementation.

### 4.2 Out of scope

- Clinical decision support, diagnosis, treatment recommendations, or patient-facing use.
- Claims of HIPAA compliance or regulatory readiness.
- Uploading real data or real-data-derived artifacts to a personal AWS account.
- Public release of subject-level outputs, subject identifiers, restricted MLflow runs, or real model artifacts.
- Hyperparameter searches intended to maximize performance on the small cohort.
- Cloud EKS unless separately justified and approved.
- A model registry unless multiple models require governed promotion.
- Restoring the retracted R² result or presenting historical SHAP rankings as evidence.

## 5. Track architecture and trust boundary

### 5.1 Public synthetic track

The public track may run locally and in a personal AWS account. It uses synthetic data, a synthetic-trained model, synthetic requests, and synthetic reference distributions. It demonstrates the complete engineering path without real data or restricted derived artifacts.

Permitted public artifacts:

- Source code, tests, documentation, schemas, Dockerfile, and parameterized Terraform.
- Synthetic generator, fixtures, payloads, model, manifest, reference distributions, and load-test data.
- CI/CD definitions with no account IDs, secrets, or persistent credentials.
- Performance and reliability measurements generated only from the synthetic deployment.

### 5.2 Institutional real-data track

The institutional track is disabled until all Phase 0 approvals are documented. If approved, it uses the same versioned source and infrastructure modules in an institution-approved account. Real data, per-subject results, real model artifacts, reference distributions, experiment tracking, Terraform state, and logs remain within that boundary.

### 5.3 Prohibited crossings

The following must never cross from the institutional track into the public track without explicit written approval:

- Raw or transformed patient data.
- Subject codes, including values matching `DBS_bG*`.
- Per-subject predictions or errors.
- Real model weights, explainers, training fingerprints, or feature distributions.
- Real-data MLflow runs and artifacts.
- Request or response bodies.
- Small-cell aggregates or metrics not approved for publication.

## 6. Stakeholders and approvals

| Decision or approval | Accountable party | Required before |
|---|---|---|
| Engineering prototype scope, kinematic target, and synthetic public track | PI and project owner | Real-data Phase 1 work |
| Human-subjects/IRB determination for new analysis | PI and IRB office | Any new real-data model run |
| DUA permission for cloud processing, copies, training, and artifact retention | PI and data provider | Institutional cloud use |
| Institutional AWS and security architecture | Institutional security/ISO | Institutional deployment |
| Public release of cohort size and aggregate MAE/RMSE/R² | PI | Public documentation |
| Real-data model explanations | PI and, if applicable, IRB/DUA owner | Computing or publishing explanations |
| Final public/private repository strategy | Project owner | Phase 6 publication |

Approval documents must be stored privately. The repository may contain only a redacted checklist with status, date, decision owner, and any operational restriction.

## 7. Functional requirements

Requirement IDs are stable and should be referenced in issues, pull requests, tests, and release notes.

### 7.1 Repository restoration and reproducibility

- **REP-001:** Restore the project repository and verify the supplied audit against the intended commit and working tree.
- **REP-002:** Preserve unrelated and uncommitted user changes; commit them only with owner approval.
- **REP-003:** Use `uv` to generate a committed deterministic `uv.lock` consumed by local setup, CI, and Docker.
- **REP-004:** Provide a fresh-clone workflow that runs tests, the synthetic demonstration, and the local API without private data.
- **REP-005:** Record the git SHA, lock-file hash, configuration hash, random seed, feature-schema version, and data provenance for each evaluation.
- **REP-006:** Keep generated data, model artifacts, experiment stores, Terraform state, credentials, and private inputs out of version control.

### 7.2 Data validation and feature schema

- **DAT-001:** Maintain one versioned feature-schema definition used by training, synthetic generation, API validation, and model manifests.
- **DAT-002:** The active schema must contain only the verified prespecified feature set, expected to be 5 or 7 numeric features.
- **DAT-003:** Reject unknown fields, subject identifiers, non-finite values, values outside documented physical or synthetic limits, and batches larger than 256 rows.
- **DAT-004:** Validate subject-to-recording and target alignment before model evaluation.
- **DAT-005:** Distinguish extraction failure, missingness, quality-control exclusion, and a valid zero measurement.
- **DAT-006:** Create synthetic examples that exercise valid, boundary, missing, malformed, and shifted-distribution cases.

### 7.3 Evaluation and statistical reporting

- **SCI-001:** Use leave-one-subject-out cross-validation by default when there are 12 or fewer subject groups; otherwise use an explicitly documented group-aware strategy.
- **SCI-002:** Assert that training and test subject sets are disjoint for every fold.
- **SCI-003:** Fit imputation, scaling, feature selection, and any other learned transformation inside each training fold only.
- **SCI-004:** Preallocate out-of-fold predictions and assign them by original test indices.
- **SCI-005:** Compare every candidate model with mean and median dummy regressors under identical folds.
- **SCI-006:** Use fixed, documented model parameters. Any future tuning requires nested group-aware cross-validation and a protocol amendment.
- **SCI-007:** Produce aligned per-subject out-of-fold results and keep real-data tables private.
- **SCI-008:** Compute subject-level cluster-bootstrap confidence intervals for MAE and RMSE using 2,000 resamples and a fixed seed.
- **SCI-009:** Run at least 200 subject-level label permutations through the complete LOSO workflow and compare observed metrics with the null distribution.
- **SCI-010:** Treat permutation results as descriptive at n=8 and qualify their interpretation at approximately n=31.
- **SCI-011:** Label every run `exploratory` or `confirmatory`. Changes made after viewing real results reset the analysis to exploratory unless a new held-out evaluation is available.
- **SCI-012:** Report cohort size, target, unit of observation, fold strategy, baselines, MAE, RMSE, R², uncertainty intervals, missing-data handling, and provenance together.
- **SCI-013:** Do not claim generalizable performance, clinical utility, biomarker validity, subgroup effects, or stable model ranking from these cohorts.
- **SCI-014:** A model that does not beat the baseline may still be served as an engineering demonstration only when `beats_baseline: false` is visible in model metadata.

### 7.4 Model packaging and lineage

- **MOD-001:** Serialize a model only from a reproducible, logged run.
- **MOD-002:** Package a model manifest containing model version, class, library versions, schema version/hash, training seed, data provenance, intended use, prohibited uses, performance summary, baseline comparison, `beats_baseline`, and experiment run ID.
- **MOD-003:** Load the model once at application startup and fail readiness if the model, manifest, schema, or hashes are inconsistent.
- **MOD-004:** Use local MLflow tracking for parameters, fold metrics, aggregate metrics, bootstrap results, permutation results, artifacts, git SHA, and analysis stage.
- **MOD-005:** Do not introduce an MLflow model registry until multiple models require governed promotion.
- **MOD-006:** Treat real model artifacts and real experiment stores as restricted derived data.

### 7.5 Inference API

- **API-001:** Implement the service with FastAPI and Pydantic v2.
- **API-002:** `GET /health` must expose separate liveness and readiness state and include the schema and model versions without leaking infrastructure or data details.
- **API-003:** `GET /model-info` must return the approved manifest fields, including `intended_use: research only; not for clinical use` and `beats_baseline`.
- **API-004:** `POST /predict` must accept one or more schema-valid rows, enforce a batch maximum of 256, and return predictions in input order.
- **API-005:** Invalid inputs must return actionable validation errors without echoing sensitive payload values.
- **API-006:** The service must not accept or require subject identifiers.
- **API-007:** Model artifacts must be loaded at startup with no per-request disk or object-store read.
- **API-008:** Request and response bodies and feature values must never be logged.
- **API-009:** The service must emit structured logs containing only request ID, route, status, duration, row count, schema hash, model version, and error class.
- **API-010:** The application must run as a non-root user and use one worker by default unless load testing demonstrates a need for more.

### 7.6 Model explanation

- **EXP-001:** Explanations must be labeled as model-behavior descriptions, not biomarker evidence, causal effects, or clinical interpretation.
- **EXP-002:** Public explanations may use only the synthetic model and synthetic background/reference data.
- **EXP-003:** Real-data explanations require explicit approval and remain inside the institutional boundary.
- **EXP-004:** Any explanation output must identify method, model version, schema version, and known limitations.
- **EXP-005:** If fold-level explanation stability is not measured, the public API must not imply stable feature ranking.
- **EXP-006:** The `/explain` endpoint is conditional on design decision **DD-002** and may be deferred without blocking the core `/predict` service.

### 7.7 Containerization

- **CON-001:** Use a minimal supported Python base image, deterministic dependencies, multi-stage build where useful, and a non-root runtime user.
- **CON-002:** Exclude raw data, CSV/MAT/XLSX files, experiment stores, credentials, Terraform state, caches, tests not needed at runtime, and real artifacts from the build context.
- **CON-003:** Add an image health check compatible with the deployment health endpoint.
- **CON-004:** Generate an SBOM and scan the image; unresolved HIGH or CRITICAL findings block promotion unless a time-limited exception is documented.
- **CON-005:** Target an image smaller than 600 MB and a local cold start under 2 seconds.
- **CON-006:** Tag images with immutable git-SHA tags; mutable tags may be convenience aliases only.

### 7.8 CI/CD and supply-chain controls

- **CIC-001:** Pull requests must run formatting/linting, unit tests, contract tests, synthetic end-to-end tests, lock-file consistency, coverage reporting, secret/identifier scanning, container build, vulnerability scan, and Terraform validation.
- **CIC-002:** CI must run without private data, real artifacts, AWS access keys, or network access except where dependency/image retrieval is explicitly required.
- **CIC-003:** GitHub Actions must authenticate to AWS through OIDC with repository, branch/environment, and audience conditions.
- **CIC-004:** Terraform plans must run on pull requests; applies must require a protected environment and human approval.
- **CIC-005:** Deployment must promote an immutable image digest and wait for service stability.
- **CIC-006:** Failed deployments must leave the prior task revision available for rollback.
- **CIC-007:** CI must detect prohibited identifier patterns and verify that no private file types are present in the image or release artifacts.

### 7.9 Infrastructure as code

- **INF-001:** Manage public-track AWS resources with Terraform; no production resource may depend on a manual console-only configuration.
- **INF-002:** Provision ECR, a versioned and private S3 artifact bucket, ECS Fargate, ALB, CloudWatch logs/alarms/dashboard, IAM roles, security groups, and GitHub OIDC federation.
- **INF-003:** Use least-privilege task and execution roles. Application tasks may read only the required versioned artifact path and write approved telemetry.
- **INF-004:** Block S3 public access, enforce encryption at rest and TLS in transit, enable versioning, and apply lifecycle policies.
- **INF-005:** Keep the service reachable only through the ALB; security groups must not expose the task directly.
- **INF-006:** Keep Terraform state in a private remote backend with locking and encryption. State files must never be committed.
- **INF-007:** Make desired task count configurable and default it to zero when the demonstration is idle.
- **INF-008:** Configure AWS Budgets notifications at $5 and $20 monthly thresholds.
- **INF-009:** Tag all resources with project, environment, track, owner, and managed-by values.
- **INF-010:** A second `terraform plan` after apply must report no changes.
- **INF-011:** The public deployment may contain only synthetic models and artifacts.
- **INF-012:** Public ingress and TLS behavior follow design decision **DD-004**.

### 7.10 Observability, performance, and drift demonstration

- **OBS-001:** Monitor request count, HTTP status classes, p50/p95 response time, target health, task restarts, memory/CPU, deployment failures, invalid-input rate, and deployed model version.
- **OBS-002:** Alarm when 5xx rate exceeds 1% for 5 minutes, p95 exceeds 200 ms for 5 minutes, any target is unhealthy, or an ECS task unexpectedly restarts.
- **OBS-003:** Use synthetic Locust traffic to ramp from 1 to 50 requests/second over 5 minutes and sustain the target for 10 minutes.
- **OBS-004:** The target acceptance criteria are p95 under 200 ms, error rate under 0.1%, zero task restarts, and stable memory.
- **OBS-005:** Archive the load-test report with model, image, infrastructure, and test-profile versions.
- **OBS-006:** Demonstrate feature drift with synthetic distributions only. Use versioned reference aggregates and PSI as an educational signal, not a clinical monitoring claim.
- **OBS-007:** Treat PSI above 0.2 as a warning for the demonstration; document sensitivity and limitations.
- **OBS-008:** Emit aggregate distribution telemetry only if it cannot expose raw values or sparse small-cell information.

### 7.11 Documentation and operations

- **DOC-001:** The README must explain the corrected history, two-track boundary, research-only status, setup, tests, local run, deployment, cost, and teardown.
- **DOC-002:** `docs/methodology.md` must contain the prespecified target, features, exclusions, evaluation protocol, and an immutable run log.
- **DOC-003:** `docs/data_access.md` must describe authorization requirements, prohibited locations, data retention, artifact sensitivity, and approval status.
- **DOC-004:** `docs/runbook.md` must cover local start, deploy, verify, rollback, scale-to-zero, destroy, cost verification, credential incident response, and alarm response.
- **DOC-005:** Public documentation must not disclose unapproved real-data metrics or imply that deployment quality establishes scientific validity.
- **DOC-006:** A new reviewer must be able to run the synthetic local demonstration from a fresh clone in under 10 minutes.

## 8. Non-functional requirements

| Area | Requirement |
|---|---|
| Security | No static AWS credentials; least privilege; non-root container; no payload logging; encrypted private storage; immutable releases. |
| Privacy | No real data, identifiers, real artifacts, or unapproved aggregates in public systems. |
| Reliability | Health checks, deployment rollback, idempotent infrastructure, monitored service state. |
| Reproducibility | Locked dependencies, fixed seeds, versioned schema/config/artifacts, logged provenance. |
| Performance | Cold start under 2 seconds locally; p95 under 200 ms at 50 RPS for the defined synthetic profile. |
| Maintainability | One canonical schema; modular infrastructure; stable requirement IDs; documented decisions. |
| Portability | Local execution without AWS; deployment parameters separated from application logic. |
| Cost | Scale to zero when idle; destroy between milestones; budget alerts; no always-on Kubernetes cluster. |
| Accessibility | Clear error messages and documentation suitable for a reviewer unfamiliar with the original project. |

## 9. Proposed architecture

### 9.1 Local research layer

The restored research package remains the source of feature extraction and grouped evaluation. New modules provide statistical resampling, experiment lineage, synthetic model training, schema packaging, and API serving. The lock file is shared by local development, CI, and container builds.

### 9.2 Public synthetic deployment

```text
GitHub push or pull request
        |
        v
GitHub Actions: tests, scans, image build, Terraform plan
        |
        | OIDC, protected apply
        v
AWS public synthetic track
  ECR immutable image
  S3 versioned synthetic artifact + manifest
  ALB -> ECS Fargate FastAPI task
  CloudWatch logs, metrics, alarms, dashboard
        ^
        |
Local/manual Locust synthetic load test
```

### 9.3 Why ECS Fargate is the default

The workload is a stateless CPU service with a small model, no GPU, no multi-model routing, and modest traffic. ECS Fargate demonstrates containers, networking, IAM, infrastructure as code, deployment, and observability without the fixed operational burden of an EKS cluster. SageMaker and cloud Kubernetes remain rejected unless a measured requirement emerges.

Local `kind` plus Helm may be used as an optional learning exercise after the ECS implementation. It must not be presented as necessary production architecture.

## 10. Experiment and artifact contract

Each reproducible run must produce:

1. A configuration record with target, feature list, cohort filter, grouping key, preprocessing, model parameters, baselines, seeds, and analysis stage.
2. Fold assignments and aligned out-of-fold predictions.
3. Aggregate metrics and baseline comparison.
4. Subject-cluster bootstrap distributions or summaries.
5. Permutation-null distributions or summaries.
6. Environment, schema, source, and data fingerprints.
7. A provenance label stating whether the data are synthetic or restricted real data.
8. An MLflow run containing the preceding metadata and approved artifacts.
9. For promoted models, a signed-off model manifest tied to the run.

Real-data versions of items 2–8 remain private unless the PI approves specific aggregate fields.

## 11. Test and verification strategy

### 11.1 Unit tests

- Group integrity and no subject overlap.
- Fold-local preprocessing under test-only outliers.
- Out-of-fold alignment under nonsequential and reversed fold order.
- Hand-calculated metric fixtures.
- Bootstrap determinism and cluster-level resampling.
- Permutation determinism and null construction.
- Schema limits and synthetic generator conformance.
- Manifest and artifact hash validation.

### 11.2 Contract and security tests

- Valid single and batch inference.
- Missing, unknown, mistyped, non-finite, out-of-range, identifier-bearing, and oversized requests.
- Readiness failure for missing or incompatible artifacts.
- Logs contain approved metadata and omit request/response values.
- Container filesystem contains no private file types, credentials, experiment store, or Terraform state.
- Secret and subject-identifier scans fail on controlled planted fixtures.

### 11.3 Integration and deployment tests

- Train synthetic model, package manifest, start container, check health, predict, and compare with direct model output.
- Confirm `/model-info` matches the loaded artifact.
- Build by immutable digest and deploy to a test environment.
- Verify ALB health, security-group path, alarm delivery, rollback, and post-apply no-op plan.

### 11.4 Performance and resilience tests

- Run the defined Locust profile and archive the report.
- Induce a controlled failing deployment in a test service and confirm alarm and rollback behavior.
- Restart a task and confirm recovery and telemetry.
- Generate synthetic feature shift and confirm the drift demonstration reacts without logging raw values.

## 12. Delivery roadmap

Effort estimates are focused working time, not elapsed approval time. The synthetic track may proceed while Phase 0 institutional approvals are pending, except where a phase explicitly depends on the restored repository or account decision.

### Phase R — Restore and reconcile the repository — completed

**Completed:** 2026-11-13  
**Verified baseline:** commit `eb45b11` plus the five preserved working-tree modifications

Deliverables:

- Located the intended repository and preserved its working tree.
- Compared the supplied audit with the actual tree, commit, dependencies, tests, data inventory, and pending changes.
- Corrected this plan where evidence differed.
- Recorded the verified baseline in Section 3.1.

Exit gate:

- Repository facts in Sections 3 and 16 are verified, subject to the real-data facts that remain unavailable for independent inspection.
- All 43 existing tests pass.
- Pending user changes are preserved.

Rollback: No destructive changes; reconciliation occurs on a new branch.

### Phase 0 — Authorization and scope freeze

**Estimated effort:** Calendar-bound, commonly 1–4 weeks  
**Dependencies:** Phase R for exact scope; may run alongside technical work on synthetic data

Deliverables:

- PI scope decision.
- IRB/human-subjects determination.
- DUA answers for analysis, cloud processing, copies, training, and derived artifacts.
- Institutional account/security decision.
- Public aggregate-metric decision.
- Redacted approval checklist.

Exit gate:

- Written answers exist for all institutional-track gates.
- Allowed data locations, artifacts, and publication fields are unambiguous.

Rollback: If approval is denied or limited, park the institutional track and continue only with the synthetic track.

### Phase 1 — Scientific and reproducibility hardening

**Status:** Synthetic-path implementation completed 2026-11-13 (uncommitted working tree; commits pending owner approval). The kinematic evaluation script exists but has NOT been executed; real-data use remains approval-gated. MLflow lineage verified only against a local synthetic store.  
**Estimated effort:** 2–4 days  
**Dependencies:** Phase R; approvals required before new real-data analysis

Deliverables:

- Deterministic environment lock.
- Subject-cluster bootstrap and label-permutation evaluation.
- Optional fixed-parameter XGBoost parity exercise, isolated in the optional analysis dependency group as approved in **DD-003**.
- Local MLflow tracking and experiment/artifact contract.
- Updated methodology and historical correction.
- Kinematic evaluation script; execution remains approval-gated.

Exit gate:

- All existing and new tests pass.
- Synthetic runs are byte-stable where promised.
- Every reported metric traces to a run, configuration, source version, and schema.
- No outcome is tuned toward the retracted result.

Rollback: Revert the phase branch. Local synthetic MLflow stores are disposable; restricted stores follow institutional retention policy.

### Phase 2 — Synthetic model, API, and container

**Estimated effort:** 2–3 days  
**Dependencies:** Phase 1 artifact and schema contract

Deliverables:

- Synthetic payload generator and training entry point.
- Versioned synthetic model and manifest.
- FastAPI endpoints for health, prediction, and model information.
- No explanation endpoint in the Phase 2 minimum viable API, as approved in **DD-002**.
- Non-root deterministic container and build exclusions.
- API, security, integration, and image-content tests.

Exit gate:

- Local container serves only the synthetic model.
- Invalid payloads fail safely.
- Image contains no prohibited files.
- Cold start is under 2 seconds and image size is under 600 MB.

Rollback: Remove the container/API branch; no external state exists.

### Phase 3 — Continuous integration, delivery, and security gates

**Status:** Repository implementation complete 2026-11-14 (uncommitted working tree). CI gates (locked install, Ruff, tests, coverage, secret/identifier scan, container build + SBOM + exceptions-aware vulnerability gate, image-privacy check, Terraform fmt/validate/lint/security, PR plan) and the OIDC-only deploy workflow are implemented and offline-validated; they remain UNEXERCISED against real AWS until the bootstrap, environment setup, and first apply are approved (Prompt 6).  
**Estimated effort:** 2 days  
**Dependencies:** Phase 2; public AWS account selection for delivery setup

Deliverables:

- Expanded pull-request checks.
- Immutable image build and ECR push.
- GitHub-to-AWS OIDC roles and protected environments.
- Terraform plan on pull request and approved apply on the deployment branch.
- Identifier/secret scan and image vulnerability scan.

Exit gate:

- No static AWS credentials are stored in GitHub or the repository.
- Negative security tests demonstrate that planted identifiers and secrets are caught.
- Deployment requires explicit human approval.

Rollback: Disable the deployment workflow and OIDC trust; retain CI-only checks.

### Phase 4 — Public synthetic AWS environment

**Status:** Terraform and operations documentation complete 2026-11-14 (uncommitted working tree) per **ADR-004** (CloudFront default-domain ingress, ephemeral; region us-east-2 — the new-AWS-experience project Region, switched from the earlier us-west-1 choice on 2026-11-14 before first apply), **ADR-005** (single root module, `dev` environment, Phase-4 bootstrap, GitHub environment `aws-synthetic`), and strict **ADR-006**. Offline acceptance criteria verified (format/validate/lint/security checks, least-privilege policy fixture tests, no account IDs committed). NO AWS resource exists yet; the bootstrap and first apply remain owner-gated (Prompt 6).  
**Estimated effort:** 2–4 days  
**Dependencies:** Phase 3 and decisions **DD-004** and **DD-005**

Deliverables:

- Terraform modules/resources for networking, ALB, ECS, ECR, artifacts, observability, OIDC, budgets, and remote state.
- Least-privilege IAM review.
- Deployment, verification, rollback, scale-to-zero, and destroy procedures.

Exit gate:

- The service returns the synthetic manifest through the approved ingress path.
- Task access is available only through the ALB.
- Terraform is idempotent.
- The environment can scale to zero and be fully destroyed without orphaned billable resources.

Rollback: Deploy the previous task revision or destroy the environment while preserving approved state and audit records.

### Phase 5 — Observability, load, drift, and recovery

**Estimated effort:** 1–2 days  
**Dependencies:** Phase 4

Deliverables:

- Dashboard and alarms.
- Structured, payload-free application telemetry.
- Locust test profile and versioned report.
- Controlled failure and rollback exercise.
- Synthetic drift demonstration.

Exit gate:

- Defined alarms fire during controlled tests and return to healthy state.
- Load acceptance criteria are met or bottlenecks and revised targets are documented.
- No test logs contain payload or feature values.

Rollback: Restore the prior task revision; disable custom telemetry if it creates privacy or stability risk.

### Phase 6 — Publication and reviewer experience

**Estimated effort:** 1–2 days  
**Dependencies:** Phases 2–5 and decision **DD-001**

Deliverables:

- Approved public repository structure.
- End-to-end README and architecture documentation.
- Sanitized performance report, cost notes, and teardown evidence.
- Resume/interview narrative grounded in completed work.

Exit gate:

- A reviewer with no project context completes the local synthetic demonstration in under 10 minutes.
- Publication scan finds no restricted artifacts or unapproved metrics.
- Claims match measured results and completed phases.

Rollback: Unpublish or make the companion repository private until the release checklist passes.

### Phase 7 — Optional local Kubernetes demonstration

**Estimated effort:** 1–2 days  
**Dependencies:** Phase 5 measurements and explicit owner approval

Deliverables, if approved:

- Local `kind` cluster and Helm chart.
- Probes, resource requests/limits, rolling update, and HPA demonstration.
- Local metrics collection.
- Written ECS-versus-Kubernetes conclusion using Phase 5 evidence.

Exit gate:

- The local demonstration works reproducibly.
- Documentation states whether Kubernetes solves a measured problem.
- No cloud EKS resources are created.

Rollback: Delete the local cluster and optional chart branch.

## 13. Milestones and dependency map

```text
Restore repository (R)
        |
        +---------------------> Authorization (0) ------------------+
        |                                                           |
        v                                                           v
Scientific hardening (1) -> API/container (2) -> CI/CD (3) -> AWS synthetic (4)
                                                        -> Observability/load (5)
                                                        -> Publication (6)
                                                        -> Optional local K8s (7)

Institutional real-data execution requires both Phase 1 readiness and Phase 0 approval.
The synthetic path does not require permission to use real data.
```

## 14. Cost and cleanup requirements

The working estimate from the supplied proposal is approximately $9/month for one small Fargate task plus approximately $16/month for an ALB, excluding transfer and ancillary charges. These figures are planning estimates and must be refreshed from current AWS pricing before deployment.

Cost controls:

- Default to zero running tasks outside demonstrations.
- Destroy the ALB and environment between milestones when continued availability is unnecessary.
- Create $5 and $20 budget notifications before the first deployment.
- Use short log retention in the public demonstration unless longer retention is justified.
- Apply ECR and S3 lifecycle rules.
- Tag resources and verify that no tagged resources remain after teardown.
- Do not create a cloud Kubernetes cluster for portfolio value alone.

The runbook must include an after-destroy check covering ECS, load balancers, NAT gateways if any, public IPs, ECR, S3, CloudWatch logs/alarms, KMS resources, and the remote-state exceptions intentionally retained.

## 15. Risk register

| Risk | Likelihood/impact | Mitigation | Trigger or owner action |
|---|---|---|---|
| Repository changes make the verified baseline stale | Medium / Medium | Re-run the Phase R checks before implementation and record the new commit | Refresh the baseline when implementation begins |
| PI, IRB, DUA, or security review blocks real-data work | Medium / High | Synthetic track remains independent | Park institutional track |
| Kinematic evaluation is null or negative | High / Medium | Show baselines and `beats_baseline`; do not tune after viewing results | Retain as honest feasibility result |
| Tiny cohort produces unstable estimates | High / High | LOSO, cluster bootstrap, permutation null, per-subject review, strong limitations | Avoid generalization claims |
| Data or artifacts cross the trust boundary | Low / Critical | Separate accounts/stores, build exclusions, scans, least privilege, no payload logs | Stop deployment and follow incident runbook |
| Explanation is mistaken for biomarker evidence | Medium / High | Conditional endpoint, explicit labels, synthetic-only public reference, stability analysis | Defer `/explain` if framing is weak |
| Dependency or image vulnerabilities | Medium / Medium | Locking, scanning, SBOM, regular refresh | Block promotion on unexcepted HIGH/CRITICAL |
| ALB and idle infrastructure create surprise cost | Medium / Medium | Budgets, scale-to-zero, destroy procedure | Destroy after demonstrations |
| Infrastructure complexity distracts from research correction | Medium / Medium | Phase gates and ECS-first architecture | Do not begin Phase 4 before Phase 1 gate |
| Resume claims get ahead of completed work | Medium / Medium | Phase-linked claims and measured evidence | Publish only unlocked bullets |

## 16. Expected repository changes

This planned change inventory was reconciled with the repository during Phase R. Exact module boundaries may still be refined through review before implementation.

### 16.1 Modify

- `pyproject.toml`: dependency groups, packaging, and lock-tool configuration.
- `.gitignore`: experiment stores, artifacts, Terraform state/cache, private inputs, and generated reports.
- `.github/workflows/ci.yml`: locked install, coverage, synthetic end-to-end test, scans, container build, and Terraform checks.
- `README.md`: corrected history, track boundary, setup, cost, research-only notice, and publication claims.
- `docs/methodology.md`: prespecification and run log.
- `docs/data_access.md`: authorization checklist, artifact policy, retention, and prohibited locations.
- `src/curve_icon_dbso/synthetic.py`: serving-schema-compatible synthetic payloads.
- `src/curve_icon_dbso/evaluation.py`: bootstrap/permutation integration and optional model comparison.
- `src/curve_icon_dbso/reporting.py`: provenance and manifest-ready outputs.

### 16.2 Add

- `uv.lock`.
- `Dockerfile` and `.dockerignore`.
- `src/curve_icon_dbso/api/` with application, schemas, model loading, and approved explanation implementation.
- `src/curve_icon_dbso/tracking.py`.
- `src/curve_icon_dbso/bootstrap.py` or equivalent statistical module.
- `scripts/train_synthetic_model.py`.
- `scripts/evaluate_real_kinematics.py`, execution-gated by approval.
- `infra/` Terraform configuration for providers, variables, networking, ECR, S3, ECS, ALB, IAM/OIDC, observability, budgets, and outputs.
- `.github/workflows/deploy.yml`.
- API, security, bootstrap, permutation, manifest, artifact, and payload-generator tests.
- `locust/locustfile.py`.
- `docs/runbook.md`.
- Architecture decision records under `docs/decisions/`.

## 17. Design decisions requiring owner consultation

No item in this section is final until the owner approves it. Recommended defaults are provided to make review concrete.

### DD-001 — Public repository layout — approved

- **Decision:** Keep scientific correction history and the MLOps extension in the existing repository.
- **Alternative:** Use a separate public companion repository containing only the synthetic engineering path, linking back to a private or archival research repository.
- **Decision effect:** Determines Phase 6 structure, release scanning, documentation links, and how much historical context is visible.
- **Status:** Approved by the owner on 2026-11-13. Reassess only if the publication scan finds restricted history that cannot be safely retained.

### DD-002 — Explanation endpoint — approved

- **Decision:** Defer `/explain` from the minimum viable API. Add it only after the prediction service is stable, using a synthetic-only explainer and an explicit limitations response.
- **Alternative:** Include `/explain` in Phase 2 using SHAP or permutation-based local attribution bundled with the synthetic artifact.
- **Decision effect:** Changes API scope, dependencies, image size, latency, and the risk of overstating scientific meaning.
- **Status:** Approved by the owner on 2026-11-13. `/explain` is not an acceptance criterion for Phase 2.

### DD-003 — Dependency and XGBoost choices — approved

- **Decision:** Use `uv` to generate a committed cross-platform lock and keep XGBoost in an optional analysis dependency group. Include the fixed-parameter XGBoost comparison only when explicitly enabled for the learning exercise; it is not required for the core corrected analysis.
- **Alternative:** Use `pip-tools` and a platform-specific `requirements.lock`; omit XGBoost to minimize native dependencies and keep the scientific comparison unchanged.
- **Decision effect:** Affects contributor workflow, container build time/size, CI, and scientific scope.
- **Status:** Approved by the owner on 2026-11-13.

### DD-004 — Public ingress and TLS

- **Recommended:** Use TLS with an approved domain for any environment left online; otherwise use a short-lived HTTP ALB demonstration restricted to approved source CIDRs and destroy it immediately afterward.
- **Alternative:** Keep the service private and exercise it through a controlled tunnel or CI runner.
- **Decision effect:** Changes DNS/ACM work, cost, reviewer accessibility, and security posture.
- **Status:** Owner decision required before Phase 4.

### DD-005 — Terraform organization

- **Recommended:** Begin with a single small root module split into readable resource files and separate environment variable files. Extract reusable modules only after a second environment proves the need.
- **Alternative:** Build reusable network, service, artifact, and observability modules from the outset.
- **Decision effect:** Trades early simplicity against institutional-track reuse.
- **Status:** Owner decision required before Phase 4.

### DD-006 — Model promotion rule

- **Recommended:** Promote the best fixed candidate only if it beats both naive baselines under the prespecified primary metric; otherwise promote a clearly labeled demonstration model with `beats_baseline: false` and no performance claim.
- **Alternative:** Always serve one predetermined model family so deployment behavior is independent of the observed comparison.
- **Decision effect:** Determines the artifact selection rule and confirmatory integrity.
- **Status:** Owner and PI decision required before any real model promotion.

## 18. Definition of done

The modernization is complete when:

1. Repository facts and historical corrections are verified and documented.
2. The grouped evaluation, uncertainty estimates, permutation null, and provenance contract pass automated tests.
3. A synthetic model can be trained, packaged, served locally, and reproduced from a clean clone.
4. The container and release artifacts pass privacy, secret, identifier, and vulnerability checks.
5. GitHub Actions deploys an immutable synthetic release to Terraform-managed ECS using OIDC and human approval.
6. Monitoring, alarms, rollback, load criteria, drift demonstration, scale-to-zero, and full teardown have been exercised.
7. Public documentation contains no restricted material and makes no unsupported scientific or clinical claim.
8. Institutional-track work remains disabled unless all written approvals are complete.
9. Every published portfolio or resume statement is backed by a completed phase and archived evidence.

## 19. Phase-linked portfolio claims

Claims should be written only after the relevant phase is complete and should use measured values rather than targets.

- **After Phase 1:** Rebuilt an invalidated evaluation as a leakage-resistant, subject-grouped protocol with fold-local preprocessing, aligned out-of-fold predictions, naive baselines, cluster-bootstrap uncertainty, and a permutation null.
- **After Phases 2–3:** Packaged a reproducible synthetic research model behind a strictly validated FastAPI service in a non-root container, with automated contract, privacy, and supply-chain checks.
- **After Phases 4–5:** Deployed the synthetic service to ECS Fargate using Terraform and GitHub OIDC, then verified alarms, rollback, cost controls, and measured load-test performance.
- **After Phase 6:** Enforced and documented a two-track privacy architecture that demonstrates the complete MLOps workflow without placing clinical data or real model artifacts in a personal cloud account or public repository.

Never publish the retracted R² = 0.713 as an achievement, call it accuracy, claim biomarker discovery from feature attribution, claim HIPAA compliance, or state target performance figures before they are measured.

## 20. Immediate next actions

1. Preserve and commit the five existing working-tree changes only when the owner is ready; do not mix them with MLOps implementation commits.
2. Start the Phase 0 approval checklist with the PI.
3. Convert Phase 1 requirements into tracked issues and architecture decision records. *(ADR-001–006 recorded 2026-11-13; Phase 1 synthetic-path implementation complete.)*
4. ~~Resolve DD-004 and DD-005 before Phase 4; resolve DD-006 with the PI before any real-model promotion.~~ — DD-004/005/006 approved by the owner 2026-11-14 (ADR-004…006); the PI gate for real-model promotion is unchanged.
5. ~~Correct the README test count~~ — done 2026-11-13 (hard-coded counts removed; the count is printed by pytest).
6. Do not execute new real-data analysis until the applicable approval gate is satisfied.
7. Resolve the kinematic prominence-floor binning ambiguity (see methodology) with the PI before any real-data QC decision.
