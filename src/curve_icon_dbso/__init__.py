"""CURVE-ICON-DBSO: exploratory DBS outcome analysis prototype.

Exploratory, hypothesis-generating analysis of whether high-frequency
oscillation (HFO) features and clinical measures relate to Deep Brain
Stimulation (DBS) outcome in a very small retrospective pilot cohort
(eight subjects, nine recordings). This package is a research prototype
and is NOT intended for clinical use.

Importing this package has no side effects: no data is loaded, no
directories are created, and no analysis runs unless you call an entry
point explicitly (see ``curve_icon_dbso.demo`` and ``scripts/``).
"""

__version__ = "0.2.0"

__all__ = [
    "config",
    "errors",
    "evaluation",
    "features",
    "hfo",
    "reporting",
    "synthetic",
]
