# Data access, privacy, and governance

## The dataset is NOT in this repository

No participant data of any kind is distributed with this repository:
no raw ECoG, no participant tables, no subject identifiers, no dates
of service, no imaging, no per-subject outputs. The pipeline works on
a deterministic synthetic cohort (`curve_icon_dbso.synthetic`) so the
code can be verified without any data access.

## Source of the real data

The real (restricted) data used in the historical pilot came from the
USC **DABI** initiative (Data Archive for the BRAIN Initiative) and
was accessed through the Informatics and Computing in Neuroscience
(**ICON**) Lab at the University of Southern California.

| Item | Status |
|---|---|
| Exact DABI dataset name | **[TO BE SUPPLIED by the repository owner]** |
| DABI accession / project ID | **[TO BE SUPPLIED]** |
| Contributing investigators | **[TO BE SUPPLIED]** |
| Required dataset citation | **[TO BE SUPPLIED]** |
| Data Use Agreement (DUA) reference | **[TO BE SUPPLIED]** |

The repository owner must supply these details and the required
citation before this project is shared beyond a private audience;
until then the dataset must be treated as fully restricted.

## If you are an authorized user

1. Obtain your own authorization for the dataset through DABI/USC and
   comply with the applicable Data Use Agreement. A code repository
   license — even where one exists — grants **no** rights to the
   underlying clinical data.
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
   Only synthetic fixtures and sufficiently aggregated, approved
   summary numbers may be committed — with provenance stated.

## Privacy rules observed in this repository

- Subject codes must not appear in committed files. (Historical
  subject codes appear only inside the archived legacy script as
  evidence of the placeholder-data fallback; no clinical values are
  attached to them.)
- No dates, records, imaging metadata, or intermediate datasets are
  tracked.
- `data/` and `outputs/` are gitignored; `*.csv`, `*.mat`, and raw
  imaging formats are blocked repo-wide.

## Two-track boundary

The project operates two separated tracks
([`docs/mlops_extension_plan.md`](mlops_extension_plan.md) §5). The same
versioned code may run in both, but their data, models, logs, state, and
accounts never mix.

| | Public synthetic track | Institutional real-data track |
|---|---|---|
| Data | Deterministic synthetic data only | Restricted DABI/USC data |
| Models | Synthetic-trained model, manifest, reference aggregates | Real-trained artifacts |
| Storage | This repository; personal AWS account (synthetic only) | Institution-approved systems and accounts only |
| Status | Active | **Disabled** until every item in [`docs/approvals.md`](approvals.md) is recorded as approved |

## Prohibited crossings

The following must never move from the institutional track into the public
track (repository, personal cloud account, container image, public log, or CI
artifact) without explicit written approval recorded in
[`docs/approvals.md`](approvals.md):

- raw or transformed patient data;
- subject codes, including values matching `DBS_bG*`;
- per-subject predictions or errors;
- real model weights, explainers, training fingerprints, or feature
  distributions;
- real-data MLflow runs and artifacts;
- real request or response bodies;
- small-cell aggregates or metrics not approved for publication.

## Artifact classification

| Class | Examples | Where allowed |
|---|---|---|
| Public-safe | source code, tests, documentation, schemas, synthetic generator/fixtures/payloads, synthetic-trained model + manifest + reference aggregates, Dockerfile, parameterized Terraform, CI/CD definitions, performance and reliability results from the synthetic deployment only | repository, personal AWS, public CI |
| Restricted derived | real per-subject metrics tables, per-subject predictions/errors, real model artifacts and explainers, real experiment (MLflow) stores, real reference distributions, institutional Terraform state | institutional boundary only |
| Private evidence | approval documents, DUA, IRB determination, security review records | private storage outside the repository, referenced redacted in [`docs/approvals.md`](approvals.md) |

## Approved storage locations

- **Real data:** local, user-supplied paths outside this repository checkout
  (`data/` is gitignored, but do not rely on gitignore alone).
- **Real per-subject outputs:** outside the repository by default; in-repo
  paths must not be used for real-derived outputs (`outputs/` is gitignored;
  prefer a path outside the checkout regardless).
- **Synthetic-track cloud:** a personal AWS account may host synthetic
  artifacts only (Phase 4+, not yet provisioned).
- **Real-track cloud:** an institution-approved account only. A personal
  account is never acceptable for real data or real-data-derived artifacts,
  even temporarily.
- **Terraform state:** private remote backend with locking and encryption;
  never committed.
- **Experiment lineage:** the default local MLflow store (`mlruns/`,
  gitignored) is for **synthetic** runs only. A store that logs real-data
  runs is restricted derived data (MOD-006): it lives inside the
  institutional boundary, never in this repository or a personal cloud
  account.

## Retention and destruction

- Synthetic MLflow stores and generated synthetic artifacts are disposable and
  may be deleted freely; they are gitignored and never published.
- Restricted stores, real artifacts, and per-subject outputs follow the DUA
  and institutional retention policy once recorded (approvals item 3). Until
  that approval exists, no real-derived artifact may be retained in any cloud
  location.
- Cloud environments default to zero running tasks when idle and are destroyed
  between milestones; the runbook (`docs/runbook.md`, Phase 4) includes an
  after-destroy orphan-resource check (ECS, load balancers, NAT gateways,
  public IPs, ECR, S3, CloudWatch, KMS) and documents the retained
  remote-state exceptions.

## Incident-response trigger

If real data, subject identifiers, real artifacts, or unapproved aggregates
are suspected to have entered the public track (repository history, personal
AWS account, container image, logs, CI artifacts):

1. **Stop** — halt deployments and any running jobs that could extend the
   exposure; do not push or publish anything else.
2. **Preserve** — do not delete or rewrite evidence, including git history;
   record what crossed the boundary, where, and when.
3. **Report** — notify the project owner and PI immediately; follow DUA and
   institutional security reporting obligations once those contacts are on
   file (approvals items 3–4).
4. **Remediate** only after the exposure is documented, using the
   rollback/destroy procedures in the runbook once it exists.

## Real-data script acknowledgement rule

Any repository script that reads restricted real data must:

1. **refuse to run** without an explicit acknowledgement flag (planned form:
   `--acknowledge-data-use`, taking a statement that the operator is
   authorized under the applicable DUA);
2. **default all outputs outside the repository checkout**, rejecting
   in-repository output paths unless deliberately overridden;
3. **record provenance** on its outputs
   (`real — authorized local access; exploratory kinematic target; NOT DBS
   outcome`, per the methodology's labeling convention).

The planned `scripts/evaluate_real_kinematics.py` (Phase 1; execution-gated by
[`docs/approvals.md`](approvals.md)) must implement this rule from the start.
The existing `scripts/run_hfo_extraction.py` predates this rule and must gain
the same guard in Phase 1 before it is used against real data again.
