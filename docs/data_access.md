# Data access, privacy, and governance

## The dataset is NOT in this repository

No participant data of any kind is distributed with this repository:
no raw ECoG, no participant tables, no subject identifiers, no dates
of service, no imaging, no per-subject outputs. The pipeline works on
a deterministic synthetic cohort (`curve_icon_dbso.synthetic`) so the
code can be verified without any data access.

## Historical data source (retired) and the public-data rule

The restricted data used in the historical pilot came from the USC
**DABI** initiative (Data Archive for the BRAIN Initiative) and was
accessed through the Informatics and Computing in Neuroscience
(**ICON**) Lab at the University of Southern California.

That former-lab, restricted dataset is **no longer part of this
project — it is out of scope**. This is an independent personal
project with no institutional affiliation: it does not use private lab
data, restricted datasets, institutional credentials, accounts,
systems, funding, or unpublished lab materials. No DUA exists or is
sought; the institutional items are recorded `NOT_APPLICABLE`
([docs/approvals.md](approvals.md)).

Any non-synthetic data this project uses should be legitimately
public. Recording its facts in the register below is recommended
guidance — it is never a precondition, and nothing in the repository
blocks work pending it.

## Dataset register (guidance, not a gate)

Every third-party dataset should be recorded here — one entry per
dataset, every field taken from the source itself. No field may be
guessed: until the facts are looked up, they stay explicit
`[REQUIRED]` placeholders. A complete entry is a courtesy to future
readers and a record of the owner's due diligence; missing or
incomplete entries never block any repository work.

### Entry — (no public dataset recorded yet)

- Dataset name: **[REQUIRED — no public dataset has been recorded]**
- Canonical URL: **[REQUIRED]**
- Version / accession: **[REQUIRED if the source provides one]**
- Access date: **[REQUIRED]**
- License or terms: **[REQUIRED]**
- Permitted uses: **[REQUIRED]**
- Attribution requirements: **[REQUIRED]**
- Redistribution / training restrictions: **[REQUIRED]**

## If you adapt this pipeline to a public dataset

1. Record the dataset's facts in the register above (recommended).
   A code repository license — even where one exists — grants **no**
   rights to third-party data, and this project does not use
   restricted, private, identifiable, or former-lab data.
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
  the publication scan before any public release.)
- No dates, records, imaging metadata, or intermediate datasets are
  tracked.
- `data/` and `outputs/` are gitignored; `*.csv`, `*.mat`, and raw
  imaging formats are blocked repo-wide.

## Two-track boundary

The project operates two separated tracks
([`docs/mlops_extension_plan.md`](mlops_extension_plan.md) §5). The same
versioned code may run in both, but their data, models, logs, state, and
accounts never mix.

| | Public synthetic track | External-data track (not currently in use) |
|---|---|---|
| Data | Deterministic synthetic data only | Public third-party data; restricted, private, identifiable, and former-lab data are out of scope |
| Models | Synthetic-trained model, manifest, reference aggregates | Artifacts trained on recorded public data, under its terms |
| Storage | This repository; personal AWS account (synthetic only) | Personal resources only; institutional accounts are out of scope entirely |
| Status | Active | Not in use (the guidance in this file applies if it is ever opened) |

## Out-of-scope material

The following never enter the public track (repository, personal
cloud account, container image, public log, or CI artifact). For the
former-lab dataset and any restricted material there is no approval
path — they are out of scope outright. Aggregates from a public
dataset may be published only as its recorded terms and attribution
requirements allow:

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
| Restricted derived | former-lab or restricted per-subject metrics tables, per-subject predictions/errors, model artifacts and explainers, experiment (MLflow) stores, reference distributions | nowhere in this project — out of scope outright |
| Dataset register entries | register facts and notes (no credentials, account IDs, or per-subject values) | this repository, per [`docs/approvals.md`](approvals.md) |

## Recommended storage locations

- **External (non-synthetic) data:** local, user-supplied paths outside
  this repository checkout (`data/` is gitignored, but do not rely on
  gitignore alone).
- **External per-subject outputs:** outside the repository by default;
  in-repo paths are discouraged for real-derived outputs (`outputs/` is
  gitignored; prefer a path outside the checkout regardless). Scripts
  print a reminder if an in-repo output path is used.
- **Synthetic-track cloud:** a personal AWS account may host synthetic
  artifacts only. This happened once (deployed 2026-11-14, verified and
  destroyed 2026-11-15; only the Terraform remote-state store is
  retained — see [deployment_report.md](deployment_report.md)) and is
  re-created per-deployment from the runbook.
- **External-data cloud:** personal resources only, for lawfully usable
  public data used under its recorded terms. Institutional accounts are
  out of scope entirely; a former-lab or restricted dataset is never
  acceptable anywhere, even temporarily.
- **Terraform state:** private remote backend with locking and encryption;
  never committed.
- **Experiment lineage:** the default local MLflow store (`mlruns/`,
  gitignored) is for **synthetic** runs only. A store that logs
  former-lab or restricted-data runs must never exist anywhere in this
  project (MOD-006); a store for recorded public data stays local,
  gitignored, and unpublished.

## Retention and destruction

- Synthetic MLflow stores and generated synthetic artifacts are disposable and
  may be deleted freely; they are gitignored and never published.
- Former-lab and restricted artifacts must not exist or be retained
  anywhere in this project. Artifacts derived from a public dataset are
  retained only as its recorded terms allow and are deleted when no
  longer needed; while the terms are unrecorded, prefer not retaining
  real-derived artifacts in any cloud location.
- Cloud environments default to zero running tasks when idle and are destroyed
  between milestones; the runbook (`docs/runbook.md`, Phases 3–5) includes an
  after-destroy orphan-resource check (ECS, load balancers, NAT gateways,
  public IPs, ECR, S3, CloudWatch, KMS) and documents the retained
  remote-state exceptions. The first full teardown (2026-11-15) completed
  with the orphan check returning empty.

## Incident-response trigger

If real data, subject identifiers, real artifacts, or aggregates whose
publication the recorded terms do not permit are suspected to have
entered the public track (repository history, personal AWS account,
container image, logs, CI artifacts):

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

## Real-data script reminders (non-blocking)

Any repository script that reads non-synthetic (external) data:

1. **prints a reminder** about the owner's data responsibilities
   (lawful public source, license/terms, no restricted, private,
   identifiable, or former-lab material) — purely informational; the
   reminder never prevents a run. `scripts/evaluate_real_kinematics.py`
   accepts an optional `--acknowledge-data-use` note that is recorded in
   the run summary as provenance when given, and prints the reminder
   when it is not;
2. **defaults outputs outside the repository checkout** and prints a
   warning (never an error) if an in-repository output path is used;
3. **records provenance** on its outputs
   (`public third-party data — locally supplied copy; exploratory
   kinematic target; NOT DBS outcome`, per the methodology's labeling
   convention).

The `scripts/evaluate_real_kinematics.py` and
`scripts/build_kinematic_table.py` scripts implement these reminders
(the builder is the source-agnostic extraction half: per-recording
`.npz` files in, the per-recording feature table out). The
`scripts/run_hfo_extraction.py` script predates the convention and
prints the same reminder.
