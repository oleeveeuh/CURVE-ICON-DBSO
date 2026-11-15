"""Model artifact packaging and verified loading (MOD-002, MOD-003).

A Phase 2 **artifact** is a directory holding exactly three files:

``model.pkl``
    The pickled fitted sklearn ``Pipeline`` (protocol 5). Trained on
    deterministic synthetic data only (``curve_icon_dbso.training``).
``schema.json``
    The canonical schema document (``curve_icon_dbso.schema``).
``manifest.json``
    The model manifest: versions, hashes, seed, provenance, intended
    and prohibited use, LOSO metrics and baselines, ``beats_baseline``,
    run ID and source SHA. Human-readable (indent 2), but every hash in
    it is verified on load.

Hash rules
----------

* ``model_sha256`` / ``schema_sha256`` — SHA-256 of the raw file bytes.
* ``artifact_sha256`` — a single digest binding the two file hashes,
  computed by :func:`artifact_binding_hash`; it lets a deployment
  pipeline refer to ONE immutable value for the whole directory.

Loading is **verification, not deserialization**: every hash is
recomputed and compared, the schema file must match BOTH its recorded
hash and the schema compiled into this code (so an artifact trained
against an older schema cannot be served), the library versions are
checked, and only then is the pickle deserialized. Any mismatch raises
:class:`~curve_icon_dbso.errors.ArtifactError` and the API fails
readiness (MOD-003) — it never serves a model it cannot vouch for.

Privacy: only synthetic-trained artifacts may ever exist in the public
track. A real-data-trained artifact is restricted derived data
(docs/data_access.md) and must never be placed here or served by this
code.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
from pathlib import Path
from typing import Any

from .errors import ArtifactError
from .schema import canonical_json, schema_document, schema_hash

MANIFEST_VERSION = 1

MODEL_FILE = "model.pkl"
SCHEMA_FILE = "schema.json"
MANIFEST_FILE = "manifest.json"

#: Manifest keys that must be present for the manifest to be loadable.
#: Kept as data so the consistency tests can pin the contract.
REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "manifest_version",
    "model_version",
    "model_class",
    "estimator_class",
    "library_versions",
    "schema_version",
    "schema_hash",
    "schema_sha256",
    "model_sha256",
    "artifact_sha256",
    "training_seed",
    "training_rows",
    "training_subjects",
    "data_provenance",
    "intended_use",
    "prohibited_uses",
    "selection_rule",
    "primary_metric",
    "fold_strategy",
    "metrics",
    "baselines",
    "candidates",
    "beats_baseline",
    "run_id",
    "source_sha",
    "lock_hash",
    "analysis_stage",
    "research_only",
    "not_for_clinical_use",
)

#: Libraries whose exact version must match the loading environment
#: (numerics-bearing; a patch difference can change predictions).
_EXACT_MATCH_LIBS = ("numpy", "scikit-learn", "scipy", "pandas")
#: Libraries matched on major.minor only (python interpreter drift
#: across patch releases is tolerated; micro version is not load-bearing).
_MINOR_MATCH_LIBS = ("python",)


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def artifact_binding_hash(model_sha256: str, schema_sha256: str) -> str:
    """One digest binding the model and schema file hashes.

    Canonical string form is stable by construction (documented here and
    in docs/api.md); changing it invalidates all artifacts — intended.
    """
    payload = f"{MODEL_FILE}={model_sha256}\n{SCHEMA_FILE}={schema_sha256}\n"
    return sha256_bytes(payload.encode("utf-8"))


def library_versions() -> dict[str, str]:
    """The versions this process would record in a new manifest."""
    import importlib.metadata

    versions = {"python": platform.python_version()}
    for dist, key in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "scikit-learn"),
        ("scipy", "scipy"),
    ):
        versions[key] = importlib.metadata.version(dist)
    return versions


def build_artifact(
    output_dir: str | Path,
    *,
    model: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Write ``model.pkl`` + ``schema.json`` + verified ``manifest.json``.

    The caller passes a manifest WITHOUT hash fields; this function
    serializes the model and schema, computes all hashes, and returns
    the completed manifest (also written to disk). The written directory
    is exactly what :func:`load_artifact` accepts.

    Raises:
        ArtifactError: If the output directory already exists (never
            overwrite a published artifact in place — build a new
            versioned directory instead).
    """
    out = Path(output_dir)
    if out.exists():
        raise ArtifactError(
            f"Refusing to overwrite an existing artifact directory: {out}. "
            "Build a new versioned directory instead — artifacts are immutable.",
            error_class="artifact_invalid",
        )
    out.mkdir(parents=True)

    model_bytes = pickle.dumps(model, protocol=5)
    model_sha = sha256_bytes(model_bytes)

    schema_bytes = (canonical_json() + "\n").encode("utf-8")
    schema_sha = sha256_bytes(schema_bytes)

    completed = dict(manifest)
    completed.update({
        "schema_version": schema_document()["schema_version"],
        "schema_hash": schema_hash(),
        "schema_sha256": schema_sha,
        "model_sha256": model_sha,
        "artifact_sha256": artifact_binding_hash(model_sha, schema_sha),
    })
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in completed]
    if missing:
        raise ArtifactError(
            f"Manifest is missing required keys: {sorted(missing)}",
            error_class="artifact_invalid",
        )

    (out / MODEL_FILE).write_bytes(model_bytes)
    (out / SCHEMA_FILE).write_bytes(schema_bytes)
    (out / MANIFEST_FILE).write_text(json.dumps(completed, indent=2) + "\n")
    return completed


def load_artifact(artifact_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load and fully verify an artifact package; return ``(model, manifest)``.

    Order matters: hashes and schema/library compatibility are checked
    BEFORE the pickle is deserialized, so corrupt or tampered files are
    rejected without executing their contents.

    Raises:
        ArtifactError: On a missing/incomplete directory, hash mismatch,
            schema drift (artifact schema != code schema), incompatible
            library versions, or a manifest missing required keys.
    """
    root = Path(artifact_dir)
    paths = {name: root / name for name in (MODEL_FILE, SCHEMA_FILE, MANIFEST_FILE)}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ArtifactError(
            f"missing artifact file(s) in {root}: {missing}",
            error_class="artifact_missing",
        )

    try:
        manifest = json.loads(paths[MANIFEST_FILE].read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"manifest.json is not valid JSON ({exc})",
            error_class="artifact_corrupt",
        ) from exc
    if not isinstance(manifest, dict):
        raise ArtifactError("manifest.json must be a JSON object", error_class="artifact_corrupt")

    missing_keys = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing_keys:
        raise ArtifactError(
            f"manifest is missing keys: {sorted(missing_keys)}",
            error_class="artifact_corrupt",
        )

    # File-integrity verification happens before any deserialization.
    model_bytes = paths[MODEL_FILE].read_bytes()
    schema_bytes = paths[SCHEMA_FILE].read_bytes()
    checks = (
        ("model_sha256", manifest["model_sha256"], sha256_bytes(model_bytes), "artifact_corrupt"),
        ("schema_sha256", manifest["schema_sha256"], sha256_bytes(schema_bytes), "artifact_corrupt"),
        (
            "artifact_sha256",
            manifest["artifact_sha256"],
            artifact_binding_hash(manifest["model_sha256"], manifest["schema_sha256"]),
            "artifact_corrupt",
        ),
    )
    for name, recorded, actual, error_class in checks:
        if recorded != actual:
            raise ArtifactError(
                f"{name} mismatch — recorded {recorded!r}, file is {actual!r}",
                error_class=error_class,
            )

    try:
        artifact_schema = json.loads(schema_bytes)
    except json.JSONDecodeError as exc:  # defensive; the hash above already matched
        raise ArtifactError(
            f"schema.json is not valid JSON ({exc})", error_class="artifact_corrupt"
        ) from exc

    # Schema drift: the artifact's schema must equal the schema compiled
    # into this code, feature for feature, limit for limit.
    if manifest["schema_version"] != schema_document()["schema_version"]:
        raise ArtifactError(
            f"artifact schema version {manifest['schema_version']!r} != deployed "
            f"schema version {schema_document()['schema_version']!r}",
            error_class="artifact_incompatible",
        )
    if artifact_schema != schema_document():
        raise ArtifactError(
            "artifact schema document differs from the deployed schema "
            "(features, limits or bounds were changed after training)",
            error_class="artifact_incompatible",
        )
    if manifest["schema_hash"] != schema_hash():
        raise ArtifactError(
            f"manifest schema hash {manifest['schema_hash']!r} != deployed schema "
            f"hash {schema_hash()!r}",
            error_class="artifact_incompatible",
        )

    _check_library_versions(manifest.get("library_versions"))

    # Only now, with every check green, deserialize the model.
    try:
        model = pickle.loads(model_bytes)
    except Exception as exc:  # noqa: BLE001 - any unpickle failure is a corrupt artifact
        raise ArtifactError(
            f"model.pkl could not be deserialized ({exc})", error_class="artifact_corrupt"
        ) from exc
    return model, manifest


def _check_library_versions(recorded: Any) -> None:
    """Fail readiness when the artifact was trained on an incompatible stack."""
    if not isinstance(recorded, dict):
        raise ArtifactError(
        "manifest.library_versions must be an object", error_class="artifact_corrupt"
    )
    current = library_versions()
    for key in _EXACT_MATCH_LIBS:
        if recorded.get(key) != current[key]:
            raise ArtifactError(
                f"{key} version mismatch — artifact trained on {recorded.get(key)!r}, "
                f"serving environment has {current[key]!r}",
                error_class="artifact_incompatible",
            )
    for key in _MINOR_MATCH_LIBS:
        recorded_minor = _major_minor(recorded.get(key))
        current_minor = _major_minor(current[key])
        if recorded_minor != current_minor:
            raise ArtifactError(
                f"{key} version mismatch — artifact trained on {recorded.get(key)!r}, "
                f"serving environment has {current[key]!r}",
                error_class="artifact_incompatible",
            )


def _major_minor(version: Any) -> str | None:
    """``"3.11.17"`` -> ``"3.11"``; unparseable input -> ``None`` (mismatch)."""
    if not isinstance(version, str):
        return None
    parts = version.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        return None
    return ".".join(parts[:2])
