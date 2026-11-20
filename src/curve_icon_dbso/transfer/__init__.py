"""Adaptive-transfer v2 study (exploratory; separately versioned contract).

This subpackage implements the v2 adaptive-transfer protocol
(``docs/adaptive_transfer_protocol.md``; owner decisions recorded
2026-11-19). It is deliberately separate from the frozen v1.0.1
contract (``curve_icon_dbso.protocol``, hash-pinned, immutable): v2
never re-fits, re-scores, or reinterprets any v1.0.1 result, and no
module here imports or modifies v1 frozen quantities except through
explicitly reused, tested machinery (signal QC, window math, bootstrap
nulls, glove phenotyping).
"""
