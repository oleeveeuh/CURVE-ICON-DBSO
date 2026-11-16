# CURVE-ICON-DBSO Implementation Prompt Pack

> **Historical status note (amended 2026-11-16):** the institutional
> approval framework several prompts reference (PI scope, IRB, DUA,
> institutional AWS/security) was retired on 2026-11-15 — this is an
> independent personal project — and the follow-on "personal public-data
> compliance gate" (checklist with `OPEN`/`VERIFIED` statuses in
> `docs/approvals.md`, enforced by script) was removed on 2026-11-16.
> **No approval, verification, or gate precondition in any prompt below
> is operative.** Repository work is never blocked by an approval
> workflow; data, licensing, privacy, and attribution considerations are
> documented as the owner's non-blocking responsibilities in
> [`docs/approvals.md`](approvals.md) and
> [`docs/data_access.md`](data_access.md). Prompts are retained verbatim
> as build history.

**Purpose:** Sequenced prompts for implementing the modernization defined in `docs/mlops_extension_plan.md`.  
**Repository:** `/Users/olivialiau/CURVE-ICON-DBSO`  
**Use:** Run one prompt at a time, in order. Review its handoff before starting the next prompt.  
**Important:** The approval/authorization preconditions written inside these historical prompts are no longer operative (see the status note above). Repository work — writing, implementation, experimentation, testing, analysis, and documentation changes — is never blocked pending any approval.

## How to use this pack

1. Start with Prompt 0 to establish the current baseline and protect the existing working tree.
2. Run Prompts 1–3 before any cloud implementation.
3. Use Prompt 4 to make the remaining owner decisions before running Prompt 5.
4. Run Prompts 5–8 sequentially.
5. Run Prompt 9 only if the optional local Kubernetes exercise is still wanted.
6. Prompt 10 is RETIRED (2026-11-15) — the institutional framework it assumed no longer applies.
7. Finish with Prompt 11 for the release-readiness audit.

Every implementation prompt requires the agent to:

- Read `docs/mlops_extension_plan.md` completely before changing files.
- Inspect the repository and current diff instead of assuming the plan is still current.
- Preserve the five pre-existing working-tree modifications and all unrelated user changes.
- Avoid committing or pushing unless separately requested.
- Keep real data, subject identifiers, real artifacts, and real-data metrics out of the public track.
- Run tests appropriate to the phase and report exact results.
- Record decisions transparently; never invent facts about datasets, licenses, approvals, or affiliations.
- Update documentation when behavior or interfaces change.
- End with a concise handoff listing changed files, verification, remaining risks, and the next suggested prompt.

---

## Prompt 0 — Baseline, worktree protection, and execution map

```text
Work in `/Users/olivialiau/CURVE-ICON-DBSO`.

This is a planning and repository-baseline task. Do not implement features yet, do not commit, do not push, do not run real-data analysis, and do not touch AWS.

Read these files completely:
- `docs/mlops_extension_plan.md`
- `README.md`
- `docs/methodology.md`
- `docs/data_access.md`
- `docs/limitations.md`
- `pyproject.toml`
- `.github/workflows/ci.yml`
- relevant source and tests for evaluation, feature schemas, HFO extraction, and reporting

Then:
1. Inspect `git status`, the current commit, and every existing diff. Treat all existing changes as user-owned.
2. Confirm whether the baseline recorded in Section 3.1 of the extension plan is still accurate.
3. Run the existing offline test suite using the repository environment without changing dependencies. Disable cache/bytecode writes if needed.
4. Map every numbered requirement in the extension plan to its implementation phase and expected files.
5. Identify conflicts, hidden dependencies, stale assumptions, or acceptance criteria that cannot currently be tested.
6. Propose a branch and commit sequence, but do not create branches or commits unless I approve it.
7. Produce a short execution map that separates:
   - synthetic/public work that can begin immediately;
   - real-data work (historically described as "blocked on PI/IRB/DUA/security approval" — no longer applicable; data guidance is non-blocking, see `docs/approvals.md`);
   - owner decisions still open;
   - optional work.

Do not modify repository files unless the recorded baseline is factually stale. If it is stale, explain the exact discrepancy and ask before editing the plan.

Return:
- baseline commit and working-tree summary;
- exact test result;
- requirement-to-phase map;
- blocking decisions and approvals;
- proposed implementation/commit sequence;
- confirmation that no implementation, commit, push, real-data operation, or AWS action occurred.
```

---

## Prompt 1 — Governance artifacts and approval gates

> *(Historical; superseded 2026-11-15/16 — the approval checklist this
> prompt created is retired and its successor gate removed. The current
> policy is non-blocking notices only: `docs/approvals.md`,
> `docs/data_access.md`.)*

```text
Work in `/Users/olivialiau/CURVE-ICON-DBSO` and implement only the governance/documentation work allowed by Phase 0 of `docs/mlops_extension_plan.md`.

Read the extension plan completely and inspect the current working tree before editing. Preserve all existing changes. Do not commit or push. Do not run real-data analysis. Do not create AWS resources. Never claim that an approval exists unless I provide the actual decision.

Create or update repository documentation so Phase 0 is operational rather than aspirational:

1. Add a redacted approval checklist template covering:
   - PI scope approval;
   - IRB/human-subjects determination;
   - DUA permission for analysis, cloud processing, copies, training, and derived-artifact retention;
   - institutional AWS/security approval;
   - public aggregate-metric permission;
   - real-data explanation permission.
2. Give each item an owner, status, decision date, evidence location outside the repository, restrictions, and expiration/review date.
3. Mark every unknown item `PENDING`; do not infer approval.
4. Expand `docs/data_access.md` with the two-track boundary, prohibited crossings, artifact classification, approved storage locations, retention/destruction expectations, and incident-response trigger.
5. Add a concise decision log or ADR index for DD-001 through DD-006. Record DD-001 through DD-003 as approved exactly as stated in the extension plan. Leave DD-004 through DD-006 unresolved.
6. Add a repository-facing rule that real-data scripts must require an explicit acknowledgement flag and must default outputs outside the repository. Do not implement the scripts yet unless a minimal guard is necessary to enforce documentation tests.
7. Add automated documentation/configuration checks only where they are deterministic and do not encode fake approval state.

Acceptance criteria:
- Unknown approvals remain visibly pending.
- The public synthetic track can proceed without real data.
- The institutional track is explicitly disabled until all required approvals are recorded.
- No private evidence, names beyond existing public project attribution, account IDs, subject codes, or real metrics are added.
- Existing tests and lint remain green.

End with:
- files changed;
- approval items still pending;
- test/lint results;
- any question that must be answered before Phase 1;
- confirmation that no real-data or AWS action occurred.
```

---

## Prompt 2 — Scientific hardening, reproducibility, and experiment lineage

```text
Work in `/Users/olivialiau/CURVE-ICON-DBSO`. Implement Phase 1 of `docs/mlops_extension_plan.md` for the synthetic/public path. Do not run any real-data evaluation. Do not commit or push. Preserve all pre-existing user changes.

Before editing:
- Read the extension plan, methodology, limitations, evaluation code, reporting code, synthetic pipeline, tests, and current diff.
- Confirm the approved choices: one repository; `/explain` deferred; `uv` lock; XGBoost optional.
- Run the existing suite and record the baseline.

Implement the following:

1. Reproducible environment
   - Adopt `uv` without deleting the existing `pyproject.toml` workflow.
   - Create and commit-ready `uv.lock` content.
   - Define clear dependency groups for core, development, API, tracking, cloud/load testing, and optional XGBoost analysis.
   - Update CI/local documentation to use locked installs.
   - Do not require XGBoost for the default test suite or core corrected analysis.

2. Subject-cluster bootstrap
   - Implement subject-level resampling for MAE and RMSE confidence intervals.
   - Default to 2,000 resamples with an explicit fixed seed.
   - Resample subjects, carrying all rows belonging to a sampled subject together.
   - Define behavior for duplicated sampled clusters, missing predictions, too few subjects, and non-finite inputs.
   - Add hand-checkable and deterministic tests.

3. Label-permutation null
   - Implement at least 200 subject-level target permutations through the complete grouped-CV workflow.
   - Keep every subject’s repeated rows together.
   - Define and document the empirical comparison statistic and p-value convention.
   - Label n=8 inference descriptive and qualify interpretation around n≈31.
   - Add deterministic tests that are fast enough for CI; allow a smaller test-only permutation count while production defaults remain compliant.

4. Experiment and artifact contract
   - Add local MLflow tracking for configuration, model parameters, fold metrics, aggregate metrics, bootstrap summaries, permutation summaries, git SHA, lock hash, schema hash, data provenance, seed, and `exploratory|confirmatory` stage.
   - Keep MLflow optional for callers that only use the core package.
   - Default stores and artifacts to ignored local paths.
   - Do not add a model registry.
   - Treat real-run stores as restricted in documentation and configuration.

5. Optional XGBoost exercise
   - Put XGBoost behind an optional dependency and explicit configuration.
   - Use one fixed, documented regressor configuration with the same folds, preprocessing, and baselines.
   - Do not tune it and do not make it the default.
   - Skip its tests cleanly when the optional dependency is absent.

6. Reporting and provenance
   - Produce machine-readable run metadata and aligned per-subject outputs.
   - Add the required provenance and research-only labels.
   - Ensure no report calls R² “accuracy” or revives the retracted result.
   - Keep real per-subject outputs outside the repository.

7. Documentation
   - Update methodology, limitations, setup, and the extension-plan status where implementation genuinely changed.
   - Correct the README’s test count dynamically or avoid hard-coded counts that immediately go stale.

Required verification:
- locked clean-environment install;
- Ruff;
- complete default test suite;
- optional XGBoost tests if the dependency is available;
- deterministic bootstrap and permutation checks;
- deterministic synthetic end-to-end run;
- proof that MLflow and generated artifacts are ignored;
- no private data required.

Do not execute `scripts/evaluate_real_kinematics.py` or any equivalent real-data path. If the planned kinematic implementation cannot be separated from real execution, implement and test it using synthetic fixtures only.

End with changed files, exact commands/results, requirement IDs satisfied, unresolved scientific choices, and the next permitted phase.
```

---

## Prompt 3 — Synthetic model artifact, FastAPI service, and Docker image

```text
Work in `/Users/olivialiau/CURVE-ICON-DBSO`. Implement Phase 2 of `docs/mlops_extension_plan.md` using synthetic data only. Do not commit, push, run real-data analysis, or access AWS. Preserve the existing working tree.

Read the plan and all Phase 1 implementation before editing. Verify the locked environment and tests first.

Implement:

1. Canonical schema and synthetic artifact
   - Create one versioned feature-schema definition used by synthetic generation, training, API validation, and the model manifest.
   - Use only the approved 5- or 7-feature schema represented by the current demonstrator; do not create 108 placeholder features.
   - Add a deterministic synthetic model-training entry point.
   - Package the model, schema, and manifest together with hashes.
   - Include model/library versions, schema version/hash, seed, provenance, intended/prohibited use, metrics, baselines, `beats_baseline`, run ID, source SHA, and artifact hash.

2. API
   - Add FastAPI/Pydantic v2 application modules.
   - Implement `GET /health`, `GET /model-info`, and `POST /predict`.
   - Keep `/explain` out of the Phase 2 API, per DD-002.
   - Load and validate the artifact once at startup.
   - Separate liveness from readiness.
   - Accept ordered batches of 1–256 rows and return predictions in order.
   - Forbid unknown fields, identifiers, non-finite values, invalid types, out-of-range values, and oversized batches.
   - Return useful errors without echoing feature values.
   - Never log payloads, predictions, or feature values.
   - Emit structured metadata-only logs.

3. Security and packaging
   - Add a non-root Docker image using the locked environment and an appropriate slim supported Python base.
   - Use a multi-stage build if it materially reduces the runtime image.
   - Add `.dockerignore` exclusions for data, outputs, experiment stores, state, credentials, caches, local environments, and private file types.
   - Use one application worker by default.
   - Add a container health check.
   - Tag/document immutable images by source SHA.

4. Tests
   - Schema round trips and manifest/hash consistency.
   - Valid single and batch prediction.
   - Input-order preservation.
   - Every malformed-input class in the plan.
   - Startup/readiness failures for missing, corrupt, or incompatible artifacts.
   - Direct-model versus API prediction equivalence.
   - Logging redaction.
   - Synthetic training determinism.
   - Container starts and answers health/predict requests.
   - Image contains no prohibited data/private files.

5. Documentation
   - Add local locked setup, training, API, Docker build/run, curl examples, manifest explanation, and troubleshooting.
   - Keep the research-only and synthetic-provenance notices visible.

Acceptance targets:
- all tests/lint green;
- image under 600 MB;
- local cold start under 2 seconds on the test machine, reported as a measurement rather than a universal guarantee;
- no real data, subject identifiers, or real artifacts in the image or logs;
- no `/explain` endpoint yet.

End with changed files, API contract, exact verification results, measured image size/cold start, satisfied requirements, and remaining issues.
```

---

## Prompt 4 — Owner design checkpoint for ingress, Terraform, and promotion

```text
Work in `/Users/olivialiau/CURVE-ICON-DBSO`. This is a design consultation only. Do not modify code or infrastructure, commit, push, contact external services, or create AWS resources.

Read `docs/mlops_extension_plan.md`, especially DD-004 through DD-006, plus the implemented API/container and current repository diff.

Prepare a concise decision memo for me covering:

1. DD-004 public ingress and TLS
   - Compare short-lived CIDR-restricted HTTP ALB, HTTPS with domain/ACM, and private/tunneled access.
   - State security, cost, reviewer accessibility, and setup tradeoffs.
   - Recommend the smallest option consistent with how I intend to demonstrate the project.

2. DD-005 Terraform organization
   - Compare a single small root split into readable files with reusable modules from the outset.
   - Account for one public synthetic environment now and a possible institutional environment later.
   - Recommend a structure and show the proposed directory tree.

3. DD-006 model promotion rule
   - Compare conditional promotion only when a model beats both baselines with predetermined-family promotion regardless of score.
   - Protect the exploratory/confirmatory boundary and avoid selecting a winner after seeing real results.
   - Recommend an exact machine-testable rule for synthetic artifacts and a separately gated rule for future real artifacts.

Also ask me for:
- AWS region;
- GitHub owner/repository slug;
- allowed ingress CIDRs or domain strategy;
- whether the demonstration should be routinely online or ephemeral;
- monthly budget ceiling;
- whether remote state bootstrap already exists;
- desired Terraform environment names.

Do not infer missing account IDs, domains, region, or approvals. End with a compact list of decisions I must approve before Prompt 5.
```

> *(Historical; the "decisions I must approve" framing was the design-consultation flow. DD-004…006 were decided by the owner on 2026-11-14 — recorded in `docs/decisions/`. No approval precondition is operative.)*

---

## Prompt 5 — CI/CD security gates and AWS Terraform implementation

> *(Historical; prerequisites below were satisfied on 2026-11-14. Note: "do not run Terraform apply or create AWS resources without a separate explicit instruction" is a spend-control instruction for the agent, not a compliance gate — repository work itself is never gated.)*

```text
Work in `/Users/olivialiau/CURVE-ICON-DBSO`. Implement Phases 3 and 4 of `docs/mlops_extension_plan.md` for the public synthetic track only.

Prerequisites:
- DD-004 through DD-006 have been explicitly answered by me.
- I have supplied the AWS region, GitHub repository identity, ingress choice, budget, environment names, and remote-state decision.
- If any prerequisite is absent, stop and ask for it before changing infrastructure code.
 
Read the plan, decision records, API/container implementation, current diff, and all existing workflows first. Preserve user changes. Do not commit or push unless separately requested. Do not run Terraform apply or create AWS resources without a separate explicit instruction after I review a plan. Building local files and running offline validation are authorized.

Implement repository code for:

1. CI gates
   - Locked `uv` install.
   - Ruff, unit/contract/integration tests, synthetic end-to-end test, and coverage report.
   - Secret scanning and prohibited subject-identifier scanning.
   - Container build, SBOM generation, and vulnerability scan that blocks unresolved HIGH/CRITICAL findings unless a documented temporary exception exists.
   - Image-content privacy check.
   - Terraform format, validate, lint/security checks, and plan where credentials are available.

2. Delivery workflow
   - GitHub Actions OIDC only; no long-lived AWS keys.
   - Restrict trust to the exact repository and protected environment/branch.
   - Build once, tag by git SHA, resolve the immutable digest, push to ECR, and deploy that digest.
   - Run Terraform plan for pull requests when safe.
   - Require a protected environment and human approval before apply/deploy.
   - Wait for ECS stability and preserve the prior task revision for rollback.

3. Terraform
   - Follow the approved DD-005 structure.
   - Pin Terraform/provider versions.
   - Implement remote state/bootstrap documentation according to the approved decision.
   - Provision the selected network layout, ALB/ingress, ECS Fargate service, ECR, versioned private S3 artifact bucket, CloudWatch log group, IAM execution/task roles, security groups, OIDC federation, budgets, lifecycle rules, and required outputs.
   - Block public S3 access; require encryption and TLS.
   - Give the application access only to the required versioned synthetic artifact path and telemetry.
   - Expose the task only through the approved ingress path.
   - Make desired task count configurable and zero by default when idle.
   - Tag every resource with project, environment, track, owner, and managed-by.
   - Do not add Secrets Manager when there is no application secret.
   - Do not add NAT gateways unless the design proves they are required and I approve their cost.

4. Operations documentation
   - Document bootstrap, plan, approved apply, verification, rollback, scale-to-zero, destroy, state retention, budget alerts, and orphan-resource checks.
   - Clearly separate commands that are safe/read-only from commands that create or destroy resources.
   - Include estimated recurring cost and note that pricing must be checked before apply.

Offline/local acceptance criteria:
- workflows parse and least-privilege assumptions are documented;
- Terraform format/validate/security checks pass;
- a fixture or policy test proves task and OIDC permissions are scoped;
- no account IDs, credentials, subject identifiers, private data, or real artifacts are committed;
- no AWS resource is created during this prompt.

At the end, show the exact Terraform plan/apply command that would be used, but do not run apply. Report changed files, validations, estimated resources/cost, unresolved findings, and the explicit approval needed for the first deployment.
```

---

## Prompt 6 — First synthetic deployment and verification

> *(Historical; the deployment it describes was executed, verified, and destroyed on 2026-11-14/15 — see `docs/deployment_report.md`. The "approval" framing was the owner's spend-confirmation flow; the environment deployment already happened.)*

```text
Work in `/Users/olivialiau/CURVE-ICON-DBSO` and deploy the public synthetic track defined in Phase 4 only after confirming that I explicitly approved the Terraform plan/apply and expected cost in this conversation.

If that approval is not present, do not deploy; produce or refresh the read-only plan instead.

Before applying:
- inspect the current diff and source SHA;
- run the full local quality/security suite;
- confirm the artifact and image are synthetic-only;
- confirm AWS identity, account type, region, expected resources, ingress restriction, budget alerts, and state backend;
- review the plan for unexpected replacements, public storage, broad IAM, NAT gateways, or unbounded cost;
- stop if anything differs from the approved design.

After approval, deploy the smallest public synthetic environment through the documented workflow or Terraform path. Then verify:
- ECR image uses the expected immutable digest;
- S3 contains only the versioned synthetic artifact and manifest;
- ALB/approved ingress is healthy;
- ECS runs the expected task revision and model version;
- `/health`, `/model-info`, and `/predict` behave correctly with synthetic requests;
- task security group is not directly reachable;
- IAM access is least privilege;
- logs contain metadata only;
- budget notifications and baseline alarms exist;
- a second Terraform plan reports no changes.

Do not send real features, real identifiers, or real artifacts. Do not enable `/explain`.

End with deployed resource summary, endpoint exposure (redact sensitive identifiers if appropriate), verification evidence, actual/estimated cost, no-op plan result, rollback revision, and exact scale-to-zero/destroy commands. Do not destroy unless I ask.
```

---

## Prompt 7 — Monitoring, load testing, drift demonstration, and recovery

> *(Historical; the observability/load/drift/rollback work it describes was completed and measured on 2026-11-15 — see `docs/deployment_report.md`. The failure-exercise "explicit approval" line was the owner's spend-confirmation flow — the exercise is complete.)*

```text
Work in `/Users/olivialiau/CURVE-ICON-DBSO`. Implement and verify Phase 5 of `docs/mlops_extension_plan.md` against the public synthetic environment. Preserve the working tree and do not commit/push unless separately requested.

Confirm the deployed environment is synthetic-only before sending traffic. If it is absent, implement all repository-side work and provide a local verification path without creating infrastructure.

Implement:

1. Observability
   - Structured metadata-only application logs.
   - Request count, HTTP status classes, p50/p95 latency, invalid-input count by error class, target health, task restarts, CPU/memory, deployment failure, and model-version visibility.
   - Dashboard and alarms required by OBS-001/OBS-002.
   - Avoid high-cardinality labels and payload-derived dimensions.

2. Load test
   - Add a Locust profile using only generated schema-valid synthetic payloads.
   - Ramp 1 to 50 RPS over 5 minutes and sustain the target for 10 minutes.
   - Capture model version, image digest, infrastructure revision, timestamps, response-time percentiles, error rate, restarts, and memory behavior.
   - Archive an HTML/report artifact without payloads.
   - Treat p95 < 200 ms, error < 0.1%, zero restarts, and stable memory as acceptance targets, not fabricated outcomes.

3. Drift demonstration
   - Add versioned synthetic reference aggregates to the synthetic model artifact.
   - Demonstrate PSI per feature with intentionally shifted synthetic input.
   - Use PSI > 0.2 as a documented educational warning threshold.
   - Never emit raw feature values or sparse buckets.
   - Explicitly state that this is not validated clinical drift monitoring.

4. Failure and rollback exercise
   - Use a safe test-only mechanism to induce an unhealthy deployment or controlled application failure.
   - Get explicit approval before deploying a deliberately failing image.
   - Verify alarm delivery and rollback to the previous known-good task revision.
   - Do not end the task while the environment is unhealthy.

5. Tests and documentation
   - Add unit tests for telemetry redaction, drift calculations, and Locust payload generation.
   - Update the runbook with alarm triage, rollback, and load-test procedures.

End with measured results, whether each target passed, alarm/failure evidence, privacy checks, changes made, rollback status, and current running-cost state. If the environment is still running, say so prominently and provide the approved cleanup options.
```

---

## Prompt 8 — Documentation, publication readiness, and one-repository reviewer flow

```text
Work in `/Users/olivialiau/CURVE-ICON-DBSO`. Implement Phase 6 of `docs/mlops_extension_plan.md` using the approved one-repository decision. Do not publish, commit, push, make the repository public, or disclose real-data metrics unless I separately approve those actions.

Read the entire repository, current diff, runbook, decision records, manifests, workflow files, and measured Phase 5 results.

Create a reviewer-ready documentation experience:

1. README
   - Correct historical premise and retraction story.
   - Distinguish outcome and kinematic questions.
   - Explain the synthetic/public versus institutional/real boundary.
   - Provide locked setup, tests, synthetic training, local API, Docker, and optional cloud demonstration.
   - Include measured—not target—performance/load values, or mark them pending.
   - Include research-only/not-clinical-use notices.
   - Avoid hard-coded test counts unless automatically maintained.

2. Architecture and operations
   - Add current diagrams for local research flow, synthetic artifact flow, CI/CD, AWS runtime, and privacy boundary.
   - Complete deploy, verify, rollback, scale-to-zero, destroy, incident, and alarm runbooks.
   - Document current monthly cost and teardown evidence.

3. Scientific and governance documentation
   - Ensure methodology, limitations, data access, experiment lineage, approval checklist, and decision records agree.
   - Keep unknown dataset identifiers and permissions visibly unresolved.
   - Do not publish unapproved aggregate real metrics or per-subject outputs.

4. Reviewer test
   - Perform a fresh-clone or clean-worktree simulation in a temporary directory.
   - Follow only the README.
   - Verify locked installation, tests, synthetic model creation, container start, health, model info, and prediction in under 10 minutes if the machine permits.
   - Record actual timing and every manual correction required.
   - Fix documentation defects and repeat until the flow is clean.

5. Publication scan
   - Scan history and working tree for secrets, account identifiers, subject codes, private file types, real outputs, MLflow stores, Terraform state, and prohibited claims.
   - Pay special attention to archived legacy files. Do not rewrite history or delete evidence without asking me.
   - Provide a release checklist and identify anything that prevents public release.

6. Portfolio claims
   - Draft claims only for phases with evidence.
   - Use actual test/load/deployment measurements.
   - Never present the retracted R², “accuracy,” SHAP biomarkers, HIPAA compliance, or target SLOs as completed results.

End with reviewer-test timing, publication-scan findings, documentation changed, claims currently supportable, blockers to public release, and a proposed commit sequence. Do not commit or publish.
```

---

## Prompt 9 — Optional local Kubernetes and Helm exercise

> *(Historical; the optional Kubernetes exercise it describes was completed on 2026-11-15 — see `docs/kubernetes.md`. Its "explicit confirmation" precondition is satisfied; no approval precondition is operative.)*

```text
Work in `/Users/olivialiau/CURVE-ICON-DBSO`. This prompt is optional and requires my explicit confirmation that I still want Phase 7 after reviewing the ECS load-test evidence. If that confirmation is absent, stop with an ECS-versus-Kubernetes recommendation and make no changes.

This work is local only. Do not create EKS or any cloud Kubernetes resources.

If approved:
1. Read Phase 5 measurements and state what learning goal Kubernetes serves.
2. Add a reproducible local `kind` setup and Helm chart for the synthetic API.
3. Configure liveness/readiness probes, non-root security context, resource requests/limits, rolling updates, and an HPA CPU target of 70%.
4. Add local metrics scraping and a minimal dashboard only if it can be reproduced without bloating the core project.
5. Run synthetic load against the local cluster and demonstrate probes, rollout, rollback, and scaling behavior.
6. Keep all model/data inputs synthetic and keep `/explain` deferred unless separately approved.
7. Document setup, teardown, resource use, and the measured ECS-versus-Kubernetes comparison.

Acceptance criteria:
- reproducible local cluster setup and teardown;
- chart lint/tests pass;
- synthetic service passes health and prediction tests;
- rollout and rollback demonstrated;
- written conclusion says candidly whether Kubernetes solves a measured problem.

End with local resource usage, test results, comparison conclusion, changed files, and confirmation that no EKS/cloud Kubernetes resource exists.
```

---

## Prompt 10 — Institutional real-data track, approval-gated (RETIRED 2026-11-15 — the institutional framework it assumed is retired; the follow-on compliance gate was removed 2026-11-16; NOT_APPLICABLE — independent personal project, see docs/approvals.md)

```text
Work in `/Users/olivialiau/CURVE-ICON-DBSO`. This prompt must not proceed unless I provide evidence that every required Phase 0 approval is complete: PI scope, IRB determination, DUA permissions, institutional security approval, institutional AWS/account decision, publication rules, and artifact-retention rules.

First inspect the redacted approval checklist. If any required item is pending, expired, ambiguous, or inconsistent, stop. List the missing decision and do not access real data, run real analysis, create institutional resources, or modify the public-track boundary.

If and only if all approvals are complete:
1. Restate the approved scope, account, storage, retention, logging, publication, and explanation restrictions before acting.
2. Verify that the kinematic target and five predictors were frozen before any real-data result was viewed.
3. Run the approved real-data evaluation locally or in the approved institutional environment only.
4. Use the same grouped evaluation, fold-local preprocessing, baselines, aligned OOF predictions, subject-cluster bootstrap, permutation null, and experiment contract.
5. Mark the first real run exploratory unless the approved protocol explicitly establishes confirmatory status.
6. Keep raw data, per-subject outputs, real model artifacts, real reference distributions, MLflow runs, logs, and state inside the institutional boundary.
7. Apply the approved DD-006 promotion rule without changing it after seeing results.
8. If the model does not beat baselines, preserve that outcome and mark `beats_baseline: false`; do not tune toward a positive result.
9. Do not compute real-data explanations unless the approval checklist explicitly permits them.
10. Publish only the exact aggregate fields approved by the PI.

Verification must include group integrity, provenance, artifact location, access controls, log redaction, and confirmation that nothing real entered the public repository or personal AWS account.

End with a redacted summary suitable for the repository and a separate private handoff location for restricted results. Do not copy restricted values into chat if the approved workflow forbids it.
```

---

## Prompt 11 — Final requirements and release audit

```text
Work in `/Users/olivialiau/CURVE-ICON-DBSO`. Perform a final read-only audit against every numbered requirement and definition-of-done item in `docs/mlops_extension_plan.md`. Do not fix issues until after presenting the audit unless I explicitly ask you to implement the fixes.

Inspect source, tests, workflows, dependency lock, model/artifact manifests, Docker configuration, Terraform, runbooks, decision records, the data & compliance notices, monitoring/load evidence, current cloud state if read-only access is approved, and the entire git diff/history relevant to release.

For every requirement ID:
- mark PASS, PARTIAL, FAIL, BLOCKED, or NOT APPLICABLE;
- cite the exact file/test/evidence;
- explain any gap in one sentence;
- identify the phase and smallest fix.

Also audit:
- the nine definition-of-done items;
- public/private boundary enforcement;
- unsupported scientific or clinical claims;
- reproducibility from a clean clone;
- security scan results and exceptions;
- actual running cloud resources and cost exposure;
- rollback and destroy readiness;
- documentation consistency;
- optional Kubernetes status;
- whether portfolio claims match completed evidence.

Run non-destructive verification where practical. Do not run Terraform apply/destroy, publish, commit, push, or execute real-data analysis.

Return:
1. executive release verdict;
2. requirement traceability table;
3. critical blockers;
4. non-critical debt;
5. privacy/security findings;
6. scientific-validity findings;
7. current cloud/cost state;
8. exact ordered remediation prompts;
9. confirmation of all actions not taken.
```

## Recommended stopping points

- Stop after Prompt 2 if the scientific pipeline is not sound.
- Stop after Prompt 3 if the synthetic artifact cannot be reproduced and validated locally.
- Stop after Prompt 5 to review Terraform plans before authorizing any deployment (a spend-caution review, not an approval gate).
- Stop after Prompt 7 to decide whether the environment should be scaled to zero or destroyed.
- Treat Prompt 9 as optional even if every other prompt is completed.
- Prompt 10 is retired — the institutional track it described is NOT_APPLICABLE for this independent personal project (`docs/approvals.md`).
