# Data & compliance notices (non-blocking)

**Nothing in this file blocks, gates, or delays repository work.**
Writing code, running analyses, experimenting, testing, and editing
documentation never require any approval or verification status. The
notes below are the project owner's documented responsibilities —
reminders, not gates.

**Last reviewed:** 2026-11-16 ·
**Related:** [`docs/data_access.md`](data_access.md) (data-handling
guidance and the dataset register) ·
[`docs/mlops_extension_plan.md`](mlops_extension_plan.md)

## History of this file

- **Before 2026-11-15** it carried an institutional approval checklist
  (PI scope, IRB determination, DUA permissions, institutional
  AWS/security) gating a former-lab real-data track. That framework is
  retired: this is an independent personal project, and the former
  institutional items are recorded `NOT_APPLICABLE` in the table below.
- **2026-11-15 → 2026-11-16** it carried a "personal public-data
  compliance gate": five checklist items with `OPEN`/`VERIFIED`
  statuses, enforced at execution time by
  `scripts/evaluate_real_kinematics.py`, which refused to run until a
  dataset was recorded `VERIFIED` in the register and an exact
  acknowledgement string was passed.
- **From 2026-11-16** all approval/verification enforcement is removed.
  Repository work is never blocked by an approval workflow. Data,
  licensing, privacy, and attribution considerations live on as
  non-blocking guidance here and in `docs/data_access.md`.

## Project status (disclaimer)

This is an independent personal project of the repository owner. It is
not sponsored by, conducted for, or represented as affiliated with any
former lab or institution, and nothing in this repository implies
endorsement, sponsorship, or approval by them. The owner is solely
responsible for ensuring that any data and resources used are lawful
and appropriate.

## Institutional requirements: NOT_APPLICABLE — independent personal project

These record that former institutional requirements no longer apply to
this independent personal project. None is approved; none is awaited;
`NOT_APPLICABLE` means the activity is out of scope — it does **not**
mean approved, and no institutional approval is claimed or implied.

| # | Former item | Why it no longer applies |
|---|---|---|
| 1 | PI scope approval (gated real-data work) | No PI and no institutional affiliation; scope is set by the project owner |
| 2 | IRB / human-subjects determination | No human-subjects research is conducted; only legitimately public, non-identifying data may be used |
| 3 | DUA permissions (former-lab USC DABI / ICON Lab data) | The former lab's DUA is not used and is not sought; the owner holds **independent authorization from the distributor** (attested 2026-11-16 — analysis and publication of findings authorized; the data itself is not redistributable) |
| 4 | Institutional AWS and security architecture | Only personal resources are used; no institutional account, system, or deployment exists |
| 5 | Public release of cohort size and aggregate metrics (formerly PI-owned) | Publication of results is a project-owner decision |
| 6 | Real-data model explanations (formerly PI/IRB/DUA-owned) | Explanations may be computed from lawful data at the owner's discretion (see ADR-002 for the synthetic-only public rule) |

## Owner responsibilities (non-blocking guidance)

When using any non-synthetic (external) data, the owner is responsible
for the following. These are reminders that may be repeated by scripts
or documentation — they never prevent any work:

1. **Authorization basis (recorded).** The DABI pilot data is used
   under the owner's independent authorization from the distributor —
   analysis and publication of research findings; the data itself is
   not redistributable. Facts live in the register
   ([`docs/data_access.md`](data_access.md)) when known; missing
   metadata is recorded `UNKNOWN` / `TO BE VERIFIED` and never blocks
   any work.
2. **Terms conformance.** Keep analysis, derived artifacts, attribution,
   and any publication conformant with the agreement's terms; publish
   findings, never the data.
3. **Provenance boundaries.** Stay within the authorization terms; no
   raw data, subject codes, or per-subject outputs in the repository or
   public track; no institutional credentials, accounts, systems, or
   funding; the former lab's own artifacts are prohibited outright.
4. **No endorsement or affiliation.** Keep documentation, metadata, and
   derived artifacts free of statements implying institutional or
   former-lab endorsement, affiliation, sponsorship, or approval.
5. **Scope changes.** If restricted data, human-subject interaction,
   institutional resources, or institutional affiliation are ever
   introduced, revisit this page and the applicable external
   requirements at that time — as a documentation update, not a gate.

Additional standing rules for this file:

- No account IDs, subject codes, per-subject values, credentials, or
  private evidence documents may be added to this file.
- Dataset facts (name, URL, version, license, terms) are recorded only
  from the source itself. Unknown fields stay explicit `UNKNOWN` /
  `TO BE VERIFIED` values in the register — they are never guessed or
  invented.
- Institutional items above are `NOT_APPLICABLE`, never `APPROVED`.

## Owner design decisions are not compliance verifications

Owner-level design decisions DD-001 through DD-006 were made by the
project owner (ADR-001…003 on 2026-11-13, ADR-004…006 on 2026-11-14)
and are recorded in [`docs/decisions/`](decisions/README.md). These
settle engineering scope only; they do not verify any dataset, and none
of them authorizes use of any third-party data or implies any
institutional endorsement.
