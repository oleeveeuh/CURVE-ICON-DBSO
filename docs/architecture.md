# Architecture

Current through the v1.0.1 corrective follow-on run (2026-11-18). Complements: [methodology.md](methodology.md)
(scientific design), [data_access.md](data_access.md) (privacy rules),
[runbook.md](runbook.md) (operations), [deployment_report.md](deployment_report.md)
(measured results from the one deployment), [decisions/](decisions/README.md)
(ADRs).

> **Research prototype — not for clinical use.** Everything shown here is
> deployment and serving measurements use synthetic data. Sanitized aggregate
> results from an exploratory, authorized local kinematic analysis are included,
> but they are not clinical results and do not evaluate treatment outcome.

## 1. Local research flow (offline, no data required)

The corrected evaluation loop is fully runnable on the deterministic synthetic
cohort. The external-data path was also run locally on an authorized 31-subject
kinematic cohort; only sanitized aggregate results are tracked here
([results_followon_real.md](results_followon_real.md)).

```mermaid
flowchart LR
    subgraph synthetic["Synthetic track (this repository)"]
        GEN["synthetic.py<br/>deterministic cohort<br/>seed 42"] --> EVAL
        subgraph EVAL["evaluation loop"]
            LOSO["LeaveOneGroupOut<br/>subjects = folds"] --> PIPE["sklearn Pipeline<br/>fold-local impute+scale"]
            PIPE --> OOF["aligned OOF predictions<br/>oof[test_idx] = ..."]
            OOF --> MET["MAE / RMSE / R² / r<br/>+ subject & recording counts"]
        end
        EVAL --> BOOT["bootstrap.py<br/>subject-cluster CIs (2000)"]
        EVAL --> PERM["label-permutation null<br/>(>= 200, block-stratified)"]
        EVAL --> TRACK["tracking.py<br/>local MLflow lineage<br/>(synthetic only, mlruns/ gitignored)"]
        BOOT --> REP["reporting / figures"]
        PERM --> REP
    end
    subgraph real["External-data track (local, authorized)"]
        REAL["DABI pilot data — owner's distributor<br/>authorization; local paths only;<br/>never enters the public track"] -.->|"non-blocking data-use reminder<br/>(docs/data_access.md guidance)"| GATE["evaluate_real_kinematics.py<br/>always runs; optional<br/>--acknowledge-data-use provenance note"]
    end
    real -.->|"never crosses"| synthetic
```

Two research questions are kept separate ([methodology.md](methodology.md)):

- **DBS-outcome prediction** (`updrs_improvement`; 7 prespecified
  features, n = 8) — **paused**: the clinical table was unavailable at
  analysis time and could not be recovered.
- **Kinematic neural–motor analysis** (`tap_rate_mean`; 5-feature reference
  plus a frozen follow-on matrix, n = 31) — **executed, negative, exploratory**:
  the v1.0.1 global-null verdict is definitive for the frozen matrix, not DBS
  outcome prediction.

## 2. Synthetic artifact flow (training → serving)

One versioned artifact is produced deterministically and verified
fail-closed everywhere it is loaded (MOD-003).

```mermaid
flowchart TB
    T["training.py<br/>python -m curve_icon_dbso.training<br/>seed 42, fixed candidates"]
    T --> A["artifact/ directory<br/>model.pkl + schema.json + manifest.json"]
    A --> REF["manifest adds<br/>versioned reference_aggregates<br/>(bin edges + shares, aggregate-only)"]
    A --> HASH["manifest records<br/>schema_sha256 / model_sha256 / artifact_sha256"]
    subgraph serve["Serving (choose one)"]
        LOCAL["uvicorn locally<br/>CURVE_ARTIFACT_DIR"]
        IMG["Docker image build<br/>trains INSIDE the build<br/>(same code, same seed)"]
    end
    A --> LOCAL
    A --> IMG
    LOCAL --> V["artifact loader:<br/>hash + schema-version +<br/>library-version checks"]
    IMG --> V
    V -->|"any mismatch"| FAIL["readiness fails closed<br/>/ready 503 forever<br/>(the Phase 5 rollback-exercise hook<br/>deliberately triggers this)"]
    V --> OK["/predict serves<br/>schema v1.0.0, 1–256 rows,<br/>order-preserving"]
    subgraph drift["Drift demonstration (OBS-006/007)"]
        AGG["reference_aggregates"] --> PSI["drift.py PSI per feature<br/>clean ≈ 0 / shifted > 0.2<br/>educational only"]
    end
    REF --> AGG
    A -. "lineage copy" .-> S3["private versioned S3 bucket<br/>(synthetic artifacts only)"]
    OK --> LOG["metadata-only JSON logs<br/>never payloads/predictions"]
```

The image is the serving source of truth (`CURVE_ARTIFACT_DIR=/app/artifact`,
trained during the build); the S3 copy is a lineage archive. Both are
synthetic-only by policy.

## 2b. Follow-on study flow (frozen contract v1.0.1)

`protocol.py` (frozen machine-readable contract) → `neural_features`/`glove_phenotyping`/`signal_qc` (frozen feature extraction) → `subject_modeling` (nested subject-grouped REST/DELTA arms) → `move_modeling` + `dmd_representation` (window-level MOVE arm) → `scripts/run_followon_subject_modeling.py` and `scripts/run_followon_move_modeling.py` (local-only versioned runs). The original v1.0.0 run was corrected under v1.0.1 on identical manifest-verified inputs; the definitive aggregate verdict is in `docs/results_followon_real.md`.

## 3. CI/CD

```mermaid
flowchart TB
    PR["pull request"] --> CI["ci.yml gates:<br/>locked install · ruff · pytest+coverage<br/>secret/identifier scan · container build<br/>trivy SBOM + exceptions-aware vuln gate<br/>image-privacy check<br/>terraform fmt/validate/lint/trivy-config<br/>read-only plan on same-repo PRs"]
    MAIN["push to main / manual dispatch"] --> DEP["deploy.yml:<br/>build once by git SHA → ECR<br/>resolve immutable digest<br/>terraform plan"]
    DEP --> ENVCK{"protected GitHub<br/>environment `aws-synthetic`<br/>owner cost safeguard (CIC-004)"}
    ENVCK -->|"standard accounts<br/>(OIDC only, no static keys)"| APPLY["terraform apply →<br/>wait ECS services-stable →<br/>CloudFront smoke test"]
    DEP -.->|"this project account:<br/>SCP denies OIDC providers<br/>(observed 2026-11-14)"| LOCAL["local apply instead<br/>(runbook §3–§4)"]
    APPLY --> BREAK["ECS deployment circuit breaker:<br/>failing deployment rolls back<br/>to the prior revision (CIC-006)"]
    LOCAL --> BREAK
```

On this project's account the deploy **workflow is intentionally
unusable** (`enable_github_oidc = false`); the workflows are kept for
standard accounts, and the same gates run locally via the runbook.

## 4. AWS runtime (the ephemeral `dev` environment)

```mermaid
flowchart LR
    U["reviewer / load test"] -->|"HTTPS only"| CF["CloudFront<br/>default *.cloudfront.net TLS<br/>PriceClass_100"]
    CF -->|"adds secret X-Origin-Verify<br/>origin-facing prefix list"| ALB["Application Load Balancer<br/>(internet-facing)"]
    ALB --> TG["target group<br/>health check /ready"]
    TG --> ECS["ECS Fargate task<br/>0.25 vCPU / 0.5 GB, non-root<br/>desired 0 when idle"]
    subgraph obs["Observability (OBS-001/002)"]
        LOGS["log group, 14-day retention<br/>metadata-only JSON lines"]
        MF["2 log metric filters<br/>(error_class + restarts)"]
        DASH["CloudWatch dashboard"]
        AL["4 alarms: 5xx ratio · p95 ·<br/>unhealthy targets · restarts"]
        SNS["SNS email topic<br/>(one-time owner confirmation)"]
    end
    ECS --> LOGS --> MF --> AL
    AL --> SNS
    LOGS --> DASH
    AL --> DASH
    ART["private S3 artifact bucket<br/>versioned, synthetic-only"] -.-> ECS
    BUD["AWS Budgets: $5 actual ·<br/>$20 forecast/actual"] -.-> EMAIL["owner email"]
    SNS --> EMAIL
```

Direct task access is impossible (the task security group admits only
the ALB security group — INF-005). The environment is destroyed between
milestones; teardown left **zero** project resources
([deployment_report.md](deployment_report.md) §8).

## 5. Privacy boundary

```mermaid
flowchart TB
    subgraph pub["PUBLIC track (this repository + personal AWS)"]
        SRC["source, tests, docs, schemas"]
        SYN["synthetic generator / fixtures / payloads"]
        SMOD["synthetic-trained model + manifest<br/>+ reference aggregates"]
        PERF["synthetic deployment performance<br/>and reliability results"]
    end
    subgraph inst["EXTERNAL-data track (local, authorized — docs/data_access.md)"]
        RDATA["DABI pilot data — owner's distributor authorization<br/>(local copy; the data itself is not redistributable)"]
        ROUT["per-subject outputs / external artifacts<br/>/ external MLflow stores"]
        RMOD["externally trained model / explainers"]
    end
    RDATA --> ROUT
    ROUT --> RMOD
    inst ==>|"NEVER crosses:<br/>raw or transformed participant data · subject codes (DBS_bG*)<br/>per-subject predictions/errors · restricted weights,<br/>explainers, fingerprints · restricted MLflow runs<br/>real request/response bodies ·<br/>small-cell aggregates beyond the recorded terms"| pub
```

Enforcement in code, not prose alone: `.gitignore` blocks data file
types; `scripts/security_scan.py` (CI) rejects credential patterns,
subject-identifier patterns, and prohibited file types — with the single
sanctioned `legacy/` identifier archive exempted, as documented in
[data_access.md](data_access.md). Serving telemetry is metadata-only by
construction and by test.
