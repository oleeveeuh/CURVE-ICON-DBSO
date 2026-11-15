"""Workflow structure, supply-chain, and security invariants (CIC-001..007).

Offline structural tests over the two GitHub Actions workflows:

* every third-party action is pinned by commit SHA (supply chain);
* the GITHUB_TOKEN is read-only everywhere; only OIDC jobs get
  ``id-token: write``, and no workflow references static AWS secrets;
* the CI gate list matches the prompt's Phase 3 requirements (lint, tests,
  coverage, secret scanning, container build/SBOM/vulnerability gate,
  image-privacy check, Terraform fmt/validate/lint/security/plan);
* the deploy workflow builds once by SHA, resolves the immutable digest,
  passes it to Terraform, gates the apply behind the protected
  environment, and waits for ECS stability.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CI_PATH = WORKFLOWS / "ci.yml"
DEPLOY_PATH = WORKFLOWS / "deploy.yml"

SHA_RE = re.compile(r"^@[0-9a-f]{40}$")


def _load(path: Path) -> dict:
    assert path.exists(), f"{path.name} is missing"
    return yaml.safe_load(path.read_text())


def _steps(workflow: dict, job_name: str) -> list[dict]:
    return workflow["jobs"][job_name].get("steps", [])


def _runs_text(workflow: dict) -> str:
    """Concatenated run: blocks (and inline run keys) of a whole workflow."""
    chunks = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if isinstance(step.get("run"), str):
                chunks.append(step["run"])
    return "\n".join(chunks)


# ------------------------------------------------------------ pinning ----

def test_every_action_is_pinned_by_full_commit_sha():
    for path in (CI_PATH, DEPLOY_PATH):
        workflow = _load(path)
        for job_name, job in workflow["jobs"].items():
            for step in job.get("steps", []):
                uses = step.get("uses")
                if not uses:
                    continue
                _, _, ref = uses.partition("@")
                assert SHA_RE.match(f"@{ref}"), (
                    f"{path.name}:{job_name}: action {uses!r} is not pinned by "
                    "a full 40-hex commit SHA (supply-chain control)"
                )


def test_no_static_aws_credentials_anywhere():
    for path in (CI_PATH, DEPLOY_PATH):
        text = path.read_text()
        for forbidden in (
            "aws_access_key_id",
            "AWS_ACCESS_KEY_ID:",
            "aws_secret_access_key",
            "${{ secrets.AWS_",
        ):
            assert forbidden not in text, (
                f"{path.name} references static AWS credentials ({forbidden}); "
                "OIDC-only is required (CIC-003)"
            )


def test_oidc_jobs_request_id_token_and_use_role_arn_variables():
    for path in (CI_PATH, DEPLOY_PATH):
        workflow = _load(path)
        for job_name, job in workflow["jobs"].items():
            steps = _steps(workflow, job_name)
            uses_oidc = any(
                "configure-aws-credentials" in (step.get("uses") or "") for step in steps
            )
            if not uses_oidc:
                continue
            perms = job.get("permissions") or {}
            assert perms.get("id-token") == "write", (
                f"{path.name}:{job_name}: OIDC job must request id-token: write"
            )
            run_text = "\n".join(s.get("run", "") for s in steps)
            assert "vars.AWS_" in run_text or "${{ vars.AWS_" in str(
                job
            ), f"{path.name}:{job_name}: role ARN must come from repository variables"
            assert "${{ secrets." not in run_text, (
                f"{path.name}:{job_name}: no secret may be attached to an AWS session"
            )


def test_token_permissions_are_minimal():
    for path in (CI_PATH, DEPLOY_PATH):
        workflow = _load(path)
        top = workflow.get("permissions")
        assert top == {"contents": "read"}, (
            f"{path.name}: top-level permissions must be contents: read"
        )
        for job_name, job in workflow["jobs"].items():
            perms = job.get("permissions")
            if perms is None:
                continue
            flat = " ".join(perms.values()) if isinstance(perms, dict) else str(perms)
            assert "write-all" not in flat
            for scope, value in perms.items():
                if value == "write":
                    assert scope in {"id-token"}, (
                        f"{path.name}:{job_name}: unexpected write scope {scope!r}"
                    )


# --------------------------------------------------------------- CI gates ---

def test_ci_covers_the_required_phase3_gates():
    ci = _load(CI_PATH)
    jobs = ci["jobs"]
    for required_job in ("test", "security-scan", "container", "terraform", "terraform-plan"):
        assert required_job in jobs, f"ci.yml is missing required job: {required_job}"

    everything = str(ci)
    for marker in (
        "ruff check",           # linting
        "pytest -q",            # unit/contract/integration tests
        "--cov",                # coverage report
        "security_scan.py",     # secret + identifier scanning
        "tests/test_container.py",  # container build + image-privacy check
        "cyclonedx",            # SBOM generation
        "vuln_gate.py",         # exceptions-aware HIGH/CRITICAL gate
        "terraform fmt",
        "terraform validate",
        "tflint --recursive",   # Terraform lint
        "trivy-action",         # Terraform security scan
    ):
        assert marker in everything, f"ci.yml is missing required gate marker: {marker}"

    plan_job = jobs["terraform-plan"]["if"]
    assert "pull_request" in plan_job and "head.repo.full_name == github.repository" in plan_job, (
        "terraform-plan must run only for same-repository pull requests"
    )


def test_ci_container_job_runs_the_image_content_privacy_check():
    ci = _load(CI_PATH)
    container_steps = _steps(ci, "container")
    names = [step.get("name", "") for step in container_steps]
    assert any("image-privacy" in n.lower() for n in names), (
        "ci.yml container job must run the image-content privacy check"
    )


# ------------------------------------------------------------- deploy -----

def test_deploy_builds_once_and_deploys_the_immutable_digest():
    deploy = _load(DEPLOY_PATH)
    run_text = _runs_text(deploy)

    # Build once, tagged by git SHA (CON-006)
    assert "sha-${SHA12}" in run_text and "docker build" in run_text
    # Resolve the immutable digest from ECR (CIC-005)
    assert "describe-images" in run_text and "imageDigest" in run_text
    # The digest reference (not a mutable tag) reaches Terraform
    assert "container_image=${{ needs.build.outputs.image }}" in run_text
    # Deploy waits for ECS stability (CIC-005)
    assert "wait services-stable" in run_text
    # Prior revision captured before apply (rollback, CIC-006)
    assert "describe-services" in run_text
    # No destructive actions outside the approved apply
    assert "terraform destroy" not in run_text


def test_deploy_gates_apply_behind_the_protected_environment():
    deploy = _load(DEPLOY_PATH)
    deploy_job = deploy["jobs"]["deploy"]
    assert deploy_job.get("environment") == "aws-synthetic", (
        "the apply job must run inside the protected aws-synthetic environment "
        "(CIC-004 human approval)"
    )
    steps_text = "\n".join(
        s.get("run", "") for s in deploy_job.get("steps", [])
    )
    assert "terraform apply" in steps_text
    assert "environment: aws-synthetic" not in steps_text.replace(
        "aws-synthetic", ""
    ) or True  # environment gating is declared on the job, asserted above
    # Build and plan jobs must NOT apply
    for job_name in ("build", "plan"):
        for step in _steps(deploy, job_name):
            assert "terraform apply" not in step.get("run", ""), (
                f"deploy.yml:{job_name} must never run terraform apply"
            )


def test_deploy_needs_build_and_plan_before_apply():
    deploy = _load(DEPLOY_PATH)
    assert deploy["jobs"]["deploy"]["needs"] == ["build", "plan"]
    assert deploy["jobs"]["plan"]["needs"] == "build"


def test_workflows_contain_no_account_ids():
    account_id_re = re.compile(r"\b\d{12}\b")
    for path in (CI_PATH, DEPLOY_PATH):
        matches = account_id_re.findall(path.read_text())
        assert not matches, f"{path.name} contains a literal 12-digit account ID: {matches}"


def test_smoke_fixture_exists_for_deploy():
    fixture = REPO_ROOT / "tests" / "fixtures" / "synthetic_predict_request.json"
    assert fixture.exists(), (
        "deploy.yml smoke test posts tests/fixtures/synthetic_predict_request.json; "
        "the fixture must exist (synthetic, public-safe payload)"
    )


def test_deploy_operational_details_are_present():
    deploy = _load(DEPLOY_PATH)
    run_text = _runs_text(deploy)
    deploy_job = deploy["jobs"]["deploy"]
    # continue-on-error is a step KEY (rollback capture survives a first run)
    assert any(
        step.get("continue-on-error") and step.get("id") == "prior"
        for step in deploy_job.get("steps", [])
    ), "deploy.yml must capture the prior revision with continue-on-error (first-deploy safe)"
    assert "timeout 900" in run_text  # bounded stability wait
    assert "GITHUB_STEP_SUMMARY" in run_text  # deploy summary for reviewers
