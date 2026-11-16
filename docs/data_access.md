# Data access, privacy, and governance

## The dataset is NOT in this repository

No participant data of any kind is distributed with this repository:
no raw ECoG, no participant tables, no subject identifiers, no dates
of service, no imaging, no per-subject outputs. The pipeline works on
a deterministic synthetic cohort (`curve_icon_dbso.synthetic`) so the
code can be verified without any data access.

## Historical data source (retired) and the active public-data rule

The restricted data used in the historical pilot came from the USC
**DABI** initiative (Data Archive for the BRAIN Initiative) and was
accessed through the Informatics and Computing in Neuroscience
(**ICON**) Lab at the University of Southern California.

That former-lab, restricted dataset is **no longer part of this
project — it is prohibited, not pending approval**. This is an
independent personal project with no institutional affiliation: it
must not use private lab data, restricted datasets, institutional
credentials, accounts, systems, funding, or unpublished lab
materials. No DUA exists or is sought; the institutional approval
gate was retired on 2026-11-15
([docs/approvals.md](approvals.md), items recorded `NOT_APPLICABLE`).

Any non-synthetic data this project ever uses must be legitimately
public and must pass the compliance register below **before** any use.

## Dataset compliance register

Every third-party dataset must be recorded here — one entry per
dataset, every field taken from the source itself — before any part of
it may be downloaded, processed, or committed. No field may be
guessed: until a dataset is verified, its facts stay explicit
`[REQUIRED]` placeholders (a visible verification item, not a
completed record). Per-dataset verification status is `OPEN` (default),
`VERIFIED`, or `REJECTED`; the gate semantics live in
[`docs/approvals.md`](approvals.md).

### Entry — (no public dataset identified yet)

- Dataset name: **[REQUIRED — no public dataset has been identified]**
- Canonical URL: **[REQUIRED]**
- Version / accession: **[REQUIRED if the source provides one]**
- Access date: **[REQUIRED]**
- License or terms: **[REQUIRED]**
- Permitted uses: **[REQUIRED]**
- Attribution requirements: **[REQUIRED]**
- Redistribution / training restrictions: **[REQUIRED]**
- Verification status: OPEN

## If you adapt this pipeline to a public dataset

1. Verify the dataset is genuinely public and lawfully usable for this
   purpose, and complete its register entry above **first**. A code
   repository license — even where one exists — grants **no** rights
   to third-party data, and nothing here permits restricted, private,
   identifiable, or former-lab data.
2. Store the data locally, outside this repository (the `.gitignore`
   blocks common data file types, but do not rely on that alone).
3. Run the extraction step against your local copy, for example:

   ```bash
   python scripts/run_hfo_extraction.py \
     --input-dir /your/local/path/all_subs_preprocessed_data \
     --output-csv /your/local/outputs/hfo_channels.csv \
     --aggregated-csv /your/local/outputs/hfo_subjects.csv \
     --band-low 80 --band-high 500
   ```

4. Keep any derived, participant-level outputs outside the repository.
   Only synthetic fixtures and aggregates whose publication the
   recorded terms clearly permit may be committed — with provenance
   stated.

## Privacy rules observed in this repository

- Subject codes must not appear in committed files. (Subject-code
  strings from the pre-refactor scripts exist **in git history only** —
  early October 2026 commits at the repository root — retained as
  evidence of the random placeholder-data fallback; no clinical values
  are attached to them. The current `legacy/` scripts use a `SUBJ_CODE`
  column placeholder instead, and history is deliberately not rewritten:
  see [data_access.md](data_access.md) "Incident-response trigger" and
  the Phase 6 publication scan before any public release.)
- No dates, records, imaging metadata, or intermediate datasets are
  tracked.
- `data/` and `outputs/` are gitignored; `*.csv`, `*.mat`, and raw
  imaging formats are blocked repo-wide.

## Two-track boundary

The project operates two separated tracks
([`docs/mlops_extension_plan.md`](mlops_extension_plan.md) §5, as
amended 2026-11-15). The same versioned code may run in both, but their
data, models, logs, state, and accounts never mix.

| | Public synthetic track | External-data track (closed) |
|---|---|---|
| Data | Deterministic synthetic data only | Compliance-verified public third-party data only; restricted, private, identifiable, and former-lab data are prohibited |
| Models | Synthetic-trained model, manifest, reference aggregates | Artifacts trained on register-verified public data, under its recorded terms |
| Storage | This repository; personal AWS account (synthetic only) | Personal resources only, and only after the register entry is `VERIFIED`; institutional accounts are out of scope entirely |
| Status | Active | **Closed** until a dataset passes the compliance register ([`docs/approvals.md`](approvals.md)) |

## Prohibited crossings

The following must never enter the public track (repository, personal
cloud account, container image, public log, or CI artifact). For the
former-lab dataset and any restricted material there is no approval
path — they are prohibited outright. Aggregates from a
compliance-verified public dataset may be published only as its
recorded terms and attribution requirements allow:

- raw or transformed patient data;
- subject codes, including values matching `DBS_bG*`;
- per-subject predictions or errors;
- former-lab model weights, explainers, training fingerprints, or
  feature distributions;
- former-lab MLflow runs and artifacts;
- real request or response bodies;
- small-cell aggregates or metrics whose publication the recorded
  terms do not clearly permit.

## Artifact classification

| Class | Examples | Where allowed |
|---|---|---|
| Public-safe | source code, tests, documentation, schemas, synthetic generator/fixtures/payloads, synthetic-trained model + manifest + reference aggregates, Dockerfile, parameterized Terraform, CI/CD definitions, performance and reliability results from the synthetic deployment only | repository, personal AWS, public CI |
| Restricted derived | former-lab or restricted per-subject metrics tables, per-subject predictions/errors, model artifacts and explainers, experiment (MLflow) stores, reference distributions | nowhere in this project — prohibited outright |
| Compliance records | dataset register entries and verification notes (no credentials, account IDs, or per-subject values) | this repository, per [`docs/approvals.md`](approvals.md) |

## Approved storage locations

- **External (non-synthetic) data:** local, user-supplied paths outside
  this repository checkout — and only after the register entry is
  `VERIFIED` (`data/` is gitignored, but do not rely on gitignore alone).
- **External per-subject outputs:** outside the repository by default;
  in-repo paths must not be used for real-derived outputs (`outputs/` is
  gitignored; prefer a path outside the checkout regardless).
- **Synthetic-track cloud:** a personal AWS account may host synthetic
  artifacts only. This happened once (deployed 2026-11-14, verified and
  destroyed 2026-11-15; only the Terraform remote-state store is
  retained — see [deployment_report.md](deployment_report.md)) and is
  re-created per-deployment from the runbook.
- **External-data cloud:** personal resources only, and only for
  compliance-verified public data used under its recorded terms.
  Institutional accounts are out of scope entirely; a former-lab or
  restricted dataset is never acceptable anywhere, even temporarily.
- **Terraform state:** private remote backend with locking and encryption;
  never committed.
- **Experiment lineage:** the default local MLflow store (`mlruns/`,
  gitignored) is for **synthetic** runs only. A store that logs
  former-lab or restricted-data runs must never exist anywhere in this
  project (MOD-006); a store for compliance-verified public data stays
  local, gitignored, and unpublished.

## Retention and destruction

- Synthetic MLflow stores and generated synthetic artifacts are disposable and
  may be deleted freely; they are gitignored and never published.
- Former-lab and restricted artifacts must not exist or be retained
  anywhere in this project. Artifacts derived from a compliance-verified
  public dataset are retained only as its recorded terms allow and are
  deleted when no longer needed; until a dataset is verified, no
  real-derived artifact may be retained in any cloud location.
- Cloud environments default to zero running tasks when idle and are destroyed
  between milestones; the runbook (`docs/runbook.md`, Phases 3–5) includes an
  after-destroy orphan-resource check (ECS, load balancers, NAT gateways,
  public IPs, ECR, S3, CloudWatch, KMS) and documents the retained
  remote-state exceptions. The first full teardown (2026-11-15) completed
  with the orphan check returning empty.

## Incident-response trigger

If real data, subject identifiers, real artifacts, or unapproved aggregates
are suspected to have entered the public track (repository history, personal
AWS account, container image, logs, CI artifacts):

1. **Stop** — halt deployments and any running jobs that could extend the
   exposure; do not push or publish anything else.
2. **Preserve** — do not delete or rewrite evidence, including git history;
   record what crossed the boundary, where, and when.
3. **Report** — notify the project owner immediately; if the data came
   from an external distributor whose terms impose incident-notification
   duties, follow those terms. (Institutional reporting channels no
   longer exist for this independent project.)
4. **Remediate** only after the exposure is documented, using the
   rollback/destroy procedures in the runbook (`docs/runbook.md` §7/§9,
   incident summary §12).

## Real-data script acknowledgement rule

Any repository script that reads non-synthetic (external) data must:

1. **refuse to run** without an explicit acknowledgement flag
   (`--acknowledge-data-use`, carrying the exact statement that the
   dataset is compliance-verified public data — lawfully licensed for
   this use — and not restricted, private, identifiable, or former-lab
   data);
2. **default all outputs outside the repository checkout**, rejecting
   in-repository output paths unless deliberately overridden;
3. **record provenance** on its outputs
   (`public third-party data — compliance-verified local copy;
   exploratory kinematic target; NOT DBS outcome`, per the
   methodology's labeling convention).

The `scripts/evaluate_real_kinematics.py` script (Phase 1;
execution-gated by the compliance register, per
[`docs/approvals.md`](approvals.md)) implements this rule and has never
been run: it refuses to execute without the explicit acknowledgement,
and it refuses while no dataset in the register is `VERIFIED`. The
`scripts/run_hfo_extraction.py` script predates this rule and must gain
the same guard before it is used against any external dataset.
