# Compliance checklist (personal public-data gate)

**Status: the external-data path is CLOSED — no third-party dataset has
passed the compliance register yet. All runnable results are synthetic.**
**Last reviewed:** 2026-11-15 ·
**Governing documents:** `docs/data_access.md` (Dataset compliance
register and boundary rules) · `docs/mlops_extension_plan.md` (§6, as
amended 2026-11-15)

This is the only compliance record kept in the repository. Verification
evidence for any dataset lives in the public register itself (canonical
URL, license, terms); nothing here records private correspondence, and
the presence of an item never grants or implies permission by itself.

## Framework change (2026-11-15)

This file previously carried an institutional approval checklist (PI
scope, IRB determination, DUA permissions, institutional AWS/security)
that gated a former-lab real-data track. This is an **independent
personal project**: it is not sponsored by, conducted for, or
represented as affiliated with any former lab or institution, and it
must use only legitimately public data and personal resources — never
private lab data, restricted datasets, institutional credentials,
accounts, systems, funding, or unpublished lab materials.

The institutional approval gate is therefore **retired**. The historical
restricted dataset (the former-lab USC DABI / ICON Lab pilot data) is
**out of scope — prohibited, not pending**: it may not be used,
requested, stored, or processed here under any circumstances. The
former institutional items are recorded `NOT_APPLICABLE` in the table
below. `NOT_APPLICABLE` means the requirement no longer applies because
the activity is out of scope; it does **not** mean approved, and no
institutional approval is claimed or implied.

## Active gate: personal public-data compliance

The only gate that can unlock non-synthetic data. It is satisfied per
dataset, only through the **Dataset compliance register** in
[`docs/data_access.md`](data_access.md); synthetic work is never
blocked by it.

### 1. Public-source verification

- Requires: the dataset is genuinely public and comes from a
  legitimate distributor; it is not leaked, restricted, scraped from a
  private source, or former-lab material; and it is lawfully usable for
  this project's exploratory research and model-training purpose.
- Owner: project owner
- Status: OPEN
- Evidence: a complete register entry in `docs/data_access.md`
- Decision date: —
- Re-check date: —

### 2. License and terms compliance

- Requires: the license or terms are identified from the source itself;
  permitted uses, attribution requirements, and
  redistribution/training restrictions are recorded in the register;
  and the planned analysis, derived artifacts, and any redistribution
  conform to them.
- Owner: project owner
- Status: OPEN
- Evidence: a complete register entry in `docs/data_access.md`
- Decision date: —
- Re-check date: —

### 3. Provenance boundary (no restricted, private, identifiable, or former-lab data)

- Requires: the dataset is not restricted, private, or identifiable
  beyond what its public terms permit; it is not the historical
  former-lab dataset and contains no former-lab or institutional data;
  and no institutional credentials, accounts, systems, or funding are
  used to obtain or process it.
- Owner: project owner
- Status: OPEN
- Evidence: a complete register entry in `docs/data_access.md`
- Decision date: —
- Re-check date: —

### 4. No endorsement or affiliation

- Requires: documentation, metadata, and derived artifacts must not
  state or imply institutional or former-lab endorsement, affiliation,
  sponsorship, or approval.
- Owner: project owner
- Status: OPEN
- Evidence: a complete register entry in `docs/data_access.md`
- Decision date: —
- Re-check date: —

### 5. Re-trigger review on scope change

- Requires: if restricted data, human-subject interaction,
  institutional resources, or institutional affiliation are ever
  introduced, the relevant external review requirements reactivate,
  this checklist gains the corresponding items with real owners,
  decision dates, and evidence, and the external-data path closes until
  they are recorded.
- Owner: project owner
- Status: OPEN
- Evidence: standing rule in this checklist and `docs/data_access.md`
- Decision date: —
- Re-check date: each scope change

## Rules

- Every item starts `OPEN`. An item may become `VERIFIED` only when a
  decision date **and** a complete register entry in
  `docs/data_access.md` are both recorded.
- Allowed statuses: `OPEN`, `VERIFIED`, `REJECTED`, `RETIRED`,
  `SUPERSEDED`, `NOT_APPLICABLE`.
- The external-data path stays **closed** until items 1–4 are
  `VERIFIED` for the dataset in question. No synthetic/public work is
  blocked by these items.
- `scripts/evaluate_real_kinematics.py` enforces this gate at execution
  time: it refuses to run unless the register records a `VERIFIED`
  dataset and the operator passes the exact compliance acknowledgement
  (`--acknowledge-data-use`).
- No account IDs, subject codes, per-subject values, credentials, or
  private evidence documents may be added to this file.
- Dataset facts (name, URL, version, license, terms) are recorded only
  from the source itself. Unverified fields stay explicit `[REQUIRED]`
  placeholders in the register — they are never guessed or invented,
  and the retired institutional gate is never restored: institutional
  items are `NOT_APPLICABLE`, never `APPROVED`.

## Retired institutional items (NOT_APPLICABLE — activity out of scope)

These record that former institutional requirements no longer apply to
this independent personal project. None is approved; none is awaited.

| # | Former item | Why it no longer applies |
|---|---|---|
| 1 | PI scope approval (gated real-data Phase 1 work) | No PI and no institutional affiliation; scope is set by the project owner |
| 2 | IRB / human-subjects determination | No human-subjects research is conducted; only legitimately public, non-identifying data may be used; item 5 re-triggers review if that ever changes |
| 3 | DUA permissions (former-lab USC DABI / ICON Lab data) | No DUA exists or is sought; the former-lab restricted dataset is prohibited outright, not pending approval |
| 4 | Institutional AWS and security architecture | Only personal resources are used; no institutional account, system, or deployment exists in scope |
| 5 | Public release of cohort size and aggregate metrics (formerly PI-owned) | Publication of results from a compliance-verified public dataset is a project-owner decision under items 1–4; synthetic results are already public |
| 6 | Real-data model explanations (formerly PI/IRB/DUA-owned) | Explanations may be computed only from a compliance-verified public dataset; the former restricted dataset is prohibited (see ADR-002 for the synthetic-only public rule) |

## Owner design decisions are not compliance verifications

Owner-level design decisions DD-001 through DD-006 were approved by the
project owner (ADR-001…003 on 2026-11-13, ADR-004…006 on 2026-11-14) and
are recorded in [`docs/decisions/`](decisions/README.md). These settle
engineering scope only; they do **not** verify any dataset, and none of
them authorizes use of any third-party data or implies any institutional
endorsement.
