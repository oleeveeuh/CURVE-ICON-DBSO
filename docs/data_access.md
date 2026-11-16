# Data access, privacy, and governance

## The dataset is NOT in this repository

No participant data of any kind is distributed with this repository:
no raw ECoG, no participant tables, no subject identifiers, no dates
of service, no imaging, no per-subject outputs. The pipeline works on
a deterministic synthetic cohort (`curve_icon_dbso.synthetic`) so the
code can be verified without any data access.

## Historical data source and the current authorization basis

The data used in the historical pilot came from the USC **DABI**
initiative (Data Archive for the BRAIN Initiative) and was originally
accessed through the Informatics and Computing in Neuroscience
(**ICON**) Lab at the University of Southern California.

**Current basis (owner attestation, 2026-11-16):** the project owner
holds an **independent authorization from the distributor** — not via
the former lab — authorizing use of the data in **analysis and
publication of research findings**. **The data itself may not be
published or redistributed** by this project. The former-lab items are
recorded `NOT_APPLICABLE` ([docs/approvals.md](approvals.md)); the
analysis runs on a local copy held by the owner, outside this
repository.

Standing rules that do not change with the authorization: no raw data,
subject codes, per-subject outputs, or identifiers ever enter the
repository or the public track; findings are published as findings
(aggregates), never as redistributed data. Recording the dataset's
facts in the register below is recommended guidance — nothing in the
repository blocks work pending it.

## Dataset register (guidance, not a gate)

Every third-party dataset should be recorded here — one entry per
dataset, every field taken from the source itself. No field may be
guessed: until the facts are looked up, they stay explicit
`UNKNOWN` / `TO BE VERIFIED` values. A complete entry is a courtesy to
future readers and a record of the owner's due diligence; missing or
incomplete entries never block any repository work.

### Entry — USC DABI kinematic/ECoG pilot collection

- Dataset name: USC DABI (Data Archive for the BRAIN Initiative) — the
  pilot intracranial/ECoG collection used for the kinematic analysis
- Canonical URL: UNKNOWN (the distributor's public collection page /
  DOI is TO BE VERIFIED from the official public page when consulted)
- Version / accession: UNKNOWN
- Access date: UNKNOWN
- Authorization basis: the owner's independent agreement with the
  distributor (owner attestation 2026-11-16; not via the former lab)
- License or terms: UNKNOWN (verbatim terms TO BE VERIFIED from the
  agreement)
- Permitted uses: analysis and publication of research findings
  (owner attestation 2026-11-16)
- Attribution requirements: UNKNOWN
- Redistribution / training restrictions: the data itself may not be
  published or redistributed; it stays local; findings are published
  as findings, never as data

**Register fields are informational.** `UNKNOWN` / `TO BE VERIFIED`
fields may be filled in whenever the public page or agreement text is
at hand; missing or incomplete fields generate at most a non-blocking
warning and never stop implementation, testing, analysis, or output
generation. Facts are never guessed — an unfilled field stays
`UNKNOWN`.

## If you adapt this pipeline to the authorized dataset

1. Keep raw data and per-subject derived files in the designated
   gitignored directories (`data/`, `outputs/`, `artifacts/`) or other
   local paths — they are never committed. Completing the register
   facts above is recommended whenever convenient; it never gates any
   work. A code repository license — even this one — grants **no**
   rights to third-party data.
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

| | Public synthetic track | External-data track (local, authorized) |
|---|---|---|
| Data | Deterministic synthetic data only | DABI pilot data under the owner's distributor authorization — local copy only, never redistributed |
| Models | Synthetic-trained model, manifest, reference aggregates | Artifacts trained on the authorized data — local only |
| Storage | This repository; personal AWS account (synthetic only) | The owner's local machine; never the repository or the public track |
| Status | Active | In local use (authorization attested 2026-11-16) |

## Out-of-scope material (public track)

The following never enter the public track (repository, personal
cloud account, container image, public log, or CI artifact). The
authorized DABI data may be processed **locally** — it never crosses;
the former lab's own artifacts remain prohibited outright. Findings
are published as findings (aggregates), only as the recorded terms and
attribution requirements allow:

- raw or transformed patient data (the local copy never leaves the
  owner's machine);
- subject codes, including values matching `DBS_bG*`;
- per-subject predictions or errors;
- the dataset itself, in any redistributable form;
- the former lab's model weights, explainers, training fingerprints,
  feature distributions, MLflow runs, and artifacts;
- real request or response bodies;
- small-cell aggregates or metrics whose publication the recorded
  terms do not clearly permit.

## Artifact classification

| Class | Examples | Where allowed |
|---|---|---|
| Public-safe | source code, tests, documentation, schemas, synthetic generator/fixtures/payloads, synthetic-trained model + manifest + reference aggregates, Dockerfile, parameterized Terraform, CI/CD definitions, performance and reliability results from the synthetic deployment only | repository, personal AWS, public CI |
| External-track derived | per-subject metrics tables, per-subject predictions/errors, models and explainers trained on the authorized data, external MLflow stores, reference distributions | **local machine only, outside the repository** — never committed, never published; the former lab's own artifacts are prohibited outright |
| Dataset register entries | register facts and notes (no credentials, account IDs, or per-subject values) | this repository, per [`docs/approvals.md`](approvals.md) |

## Recommended storage locations

- **External (non-synthetic) data:** local paths; the designated
  in-repository locations are the gitignored `data/` and `outputs/`
  directories (gitignore keeps them out of version control; anything
  not matching the ignore rules will be flagged by the scanner as a
  non-blocking notice).
- **External per-subject outputs:** the same designated gitignored
  directories (`outputs/`, `artifacts/`) or any local path; they are
  simply never committed.
- **Synthetic-track cloud:** a personal AWS account may host synthetic
  artifacts only. This happened once (deployed 2026-11-14, verified and
  destroyed 2026-11-15; only the Terraform remote-state store is
  retained — see [deployment_report.md](deployment_report.md)) and is
  re-created per-deployment from the runbook.
- **External-data cloud:** none — the authorized data and everything
  derived from it stays on the owner's local machine; institutional
  accounts are out of scope entirely.
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
- The former lab's own artifacts (weights, explainers, experiment
  stores) must not exist anywhere in this project. Artifacts derived
  from the authorized data stay on the owner's machine and are retained
  only as the agreement's terms allow; delete them when no longer
  needed.
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
   (raw data, subject codes, and per-subject outputs stay out of
   version control; findings are published as findings, never as data)
   — purely informational; the reminder never prevents a run.
   `scripts/evaluate_real_kinematics.py` accepts an optional
   `--acknowledge-data-use` note that is recorded in the run summary as
   provenance when given, and prints the reminder when it is not;
2. **writes outputs to designated gitignored local directories**
   (`data/`, `outputs/`, `artifacts/`) or other local paths — an in-repo
   path outside those directories prints a warning (never an error);
3. **records provenance** on its outputs
   (`DABI — authorized local copy; exploratory kinematic target; NOT
   DBS outcome`, per the methodology's labeling convention).

The `scripts/evaluate_real_kinematics.py` and
`scripts/build_kinematic_table.py` scripts implement these reminders
(the builder is the source-agnostic extraction half: per-recording
`.npz` files in, the per-recording feature table out). The
`scripts/run_hfo_extraction.py` script predates the convention and
prints the same reminder.
