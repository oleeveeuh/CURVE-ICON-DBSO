# ADR-001 — Public repository layout

**Status:** Approved by the owner on 2026-11-13
**Source:** `docs/mlops_extension_plan.md` §17, DD-001
**Gates affected:** Phase 6 structure, release scanning, documentation links,
visibility of historical context

## Decision

Keep scientific correction history and the MLOps extension in the existing
repository.

## Alternative considered

Use a separate public companion repository containing only the synthetic
engineering path, linking back to a private or archival research repository.

## Decision effect

Determines Phase 6 structure, release scanning, documentation links, and how
much historical context is visible.

## Notes

Reassess only if the publication scan finds restricted history that cannot be
safely retained. This decision covers repository layout only; it does not
authorize release of third-party data by itself (see `docs/approvals.md`
for the compliance gate).
