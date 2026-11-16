"""Least-privilege policy assertions (INF-003, CIC-003) — offline fixtures.

The JSON policy documents under ``infra/policies/`` are the single source
for the ECS task role and the three GitHub Actions OIDC roles. These tests
parse them (placeholders substituted with obviously-fake values) and prove:

* the application task role can read ONLY the versioned synthetic artifact
  path, can emit ONLY namespaced metrics, and carries explicit Deny
  guardrails;
* the Terraform PLAN role has no write/create/delete/tag actions at all;
* the Terraform APPLY role's wildcards are constrained (PassRole only to
  project-named roles; the state bucket cannot be deleted or repolicied;
  no Secrets Manager / KMS / SSM anywhere);
* the OIDC trust policies pin the exact repository slug plus the exact
  ref / protected environment, with the sts.amazonaws.com audience;
* no account ID, AKIA key, or subject identifier appears anywhere under
  infra/ (the no-account-IDs acceptance criterion).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICIES = REPO_ROOT / "infra" / "policies"

# Placeholder form expected in the raw policy documents. Substituting with
# fake values adds nothing: every scope assertion below targets the
# placeholder-carrying string itself, so the policies are parsed RAW.
PLACEHOLDER_RE = re.compile(r"\$\{[a-z_]+\}")

WRITE_VERB_RE = re.compile(
    r"^(Create|Put|Post|Update|Modify|Delete|Tag|Untag|Attach|Detach|Register|"
    r"Deregister|Set|Run|Start|Stop|Terminate|Import|Move|Enable|Disable)"
)


def _policy(name: str, substitutions: dict[str, str] | None = None) -> dict:
    text = (POLICIES / name).read_text()
    for old, new in (substitutions or {}).items():
        text = text.replace(old, new)
    return json.loads(text)


@pytest.mark.parametrize("name", ["ci-apply-policy.json", "ci-plan-policy.json", "task-role.json"])
def test_policies_carry_no_legacy_comment_element(name):
    """The new-AWS-experience IAM endpoint REJECTS identity policies that
    combine the legacy top-level 'Comment' element with multiple statements
    (MalformedPolicyDocument). Explanatory text lives in iam.tf comments and
    in this test file instead — never in the policy documents."""
    doc = _policy(name)
    assert "Comment" not in doc, (
        f"{name} contains a top-level Comment element; inline role policies "
        "with Comment + multiple statements are rejected by the account's "
        "strict IAM validation"
    )


def _statements(policy: dict) -> list[dict]:
    return policy["Statement"]


def _allow_actions(policy: dict) -> set[str]:
    return {
        action
        for stmt in _statements(policy)
        if stmt.get("Effect") == "Allow"
        for action in _as_list(stmt.get("Action", []))
    }


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


# ------------------------------------------------------------ task role ---


def test_task_role_reads_only_the_synthetic_artifact_prefix():
    policy = _policy("task-role.json")
    for stmt in _statements(policy):
        if stmt.get("Effect") != "Allow" or "s3:GetObject" not in _as_list(
            stmt.get("Action", [])
        ):
            continue
        for resource in _as_list(stmt.get("Resource", [])):
            assert resource.endswith("/${artifact_prefix}/*"), (
                f"task-role s3 read is not scoped to the synthetic artifact prefix: {resource}"
            )


def test_task_role_metric_writes_are_namespace_bound():
    policy = _policy("task-role.json")
    metric_stmts = [
        s for s in _statements(policy) if "cloudwatch:PutMetricData" in _as_list(s.get("Action", []))
    ]
    assert len(metric_stmts) == 1, "exactly one PutMetricData statement expected"
    stmt = metric_stmts[0]
    assert stmt.get("Condition", {}).get("StringEquals", {}).get(
        "cloudwatch:namespace"
    ) == "${metrics_namespace}", "PutMetricData must be bound to the project namespace"
    assert stmt.get("Resource") == "*", "PutMetricData is account-global by AWS design"


def test_task_role_denies_artifact_writes_and_secret_stores():
    policy = _policy("task-role.json")
    denied = {
        action
        for stmt in _statements(policy)
        if stmt.get("Effect") == "Deny"
        for action in _as_list(stmt.get("Action", []))
    }
    for expected in (
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:DeleteBucket",
        "secretsmanager:GetSecretValue",
        "ssm:GetParameter",
    ):
        assert expected in denied, f"task role must explicitly deny {expected}"


def test_task_role_has_no_broad_allow_wildcards():
    policy = _policy("task-role.json")
    for stmt in _statements(policy):
        if stmt.get("Effect") != "Allow":
            continue
        for action in _as_list(stmt.get("Action", [])):
            assert action != "*", "task role must not allow *"
        if "cloudwatch:PutMetricData" in _as_list(stmt.get("Action", [])):
            continue  # account-global Resource is covered by the namespace condition test
        for resource in _as_list(stmt.get("Resource", [])):
            assert resource != "*", (
                f"task-role Allow statement {stmt.get('Sid')} must not use Resource *"
            )


# ------------------------------------------------------------ plan role ---


def test_plan_role_contains_no_write_actions():
    policy = _policy("ci-plan-policy.json")
    write_actions = [
        action for action in _allow_actions(policy) if WRITE_VERB_RE.match(action)
    ]
    assert not write_actions, f"plan role has write-capable actions: {write_actions}"
    assert "s3:PutObject" not in _allow_actions(policy)
    assert "dynamodb:PutItem" not in _allow_actions(policy)


def test_plan_role_reads_state_and_managed_services_only():
    policy = _policy("ci-plan-policy.json")
    actions = _allow_actions(policy)
    # Everything must be a read/list/describe-style action on the expected services
    allowed_services = {
        "s3", "dynamodb", "ec2", "ecs", "elasticloadbalancing", "cloudfront",
        "ecr", "logs", "cloudwatch", "iam", "budgets", "sts", "sns",
    }
    for action in actions:
        service, _, _ = action.partition(":")
        assert service in allowed_services, f"unexpected service in plan role: {action}"


# ----------------------------------------------------------- apply role ---


def test_apply_role_passrole_is_constrained_to_project_roles():
    policy = _policy("ci-apply-policy.json")
    passrole_stmts = [
        s for s in _statements(policy) if "iam:PassRole" in _as_list(s.get("Action", []))
    ]
    assert len(passrole_stmts) == 1, "exactly one PassRole statement expected"
    resource = _as_list(passrole_stmts[0].get("Resource", []))[0]
    assert re.fullmatch(r"arn:aws:iam::\*:role/\$\{name_prefix\}-\*", resource), (
        f"PassRole must be constrained to project-named roles, got: {resource}"
    )


def test_apply_role_cannot_delete_or_repolicy_the_state_store():
    policy = _policy("ci-apply-policy.json")
    denied = {
        action
        for stmt in _statements(policy)
        if stmt.get("Effect") == "Deny"
        for action in _as_list(stmt.get("Action", []))
    }
    for expected in ("s3:DeleteBucket", "s3:PutBucketPolicy", "dynamodb:DeleteTable"):
        assert expected in denied, (
            f"apply role must explicitly deny state-store destruction: {expected}"
        )


def test_apply_role_has_no_secret_store_or_kms_actions():
    policy = _policy("ci-apply-policy.json")
    actions = _allow_actions(policy)
    for forbidden_service in ("secretsmanager", "kms", "ssm"):
        leaks = [a for a in actions if a.startswith(f"{forbidden_service}:")]
        assert not leaks, f"apply role must not touch {forbidden_service}: {leaks}"


def test_apply_role_artifact_bucket_writes_are_scoped_to_that_bucket():
    policy = _policy("ci-apply-policy.json")
    bucket_write = next(
        s for s in _statements(policy) if s.get("Sid") == "ManageProjectArtifactBucket"
    )
    for resource in _as_list(bucket_write.get("Resource", [])):
        assert "${artifact_bucket_arn}" in resource, (
            f"artifact-bucket management is not bucket-scoped: {resource}"
        )


# ------------------------------------------------------------- OIDC trust ---


def _locals_value(name: str) -> str:
    text = (REPO_ROOT / "infra" / "locals.tf").read_text()
    match = re.search(rf'{name}\s*=\s*"([^"]+)"', text)
    assert match, f"locals.tf is missing {name}"
    return match.group(1)


def test_oidc_trust_pins_audience_and_repository():
    # The sub value is filled by iam.tf through jsonencode (string OR list of
    # allowed subs), so the placeholder is UNQUOTED in the raw template:
    raw = (POLICIES / "oidc-trust.json").read_text()
    assert re.search(
        r'"token\.actions\.githubusercontent\.com:sub":\s*\$\{sub_pattern_json\}', raw
    ), "the sub condition must be a jsonencode placeholder (not a quoted literal)"

    trust = _policy("oidc-trust.json", {"${sub_pattern_json}": '"__SUB__"'})
    stmt = _statements(trust)[0]
    condition = stmt["Condition"]["StringEquals"]
    assert condition["token.actions.githubusercontent.com:aud"] == "sts.amazonaws.com"
    assert condition["token.actions.githubusercontent.com:sub"] == "__SUB__"
    assert stmt["Principal"]["Federated"] == (
        "arn:aws:iam::${account_id}:oidc-provider/token.actions.githubusercontent.com"
    )

    # iam.tf fills it with jsonencode(...): the list form must also parse.
    as_list = _policy("oidc-trust.json", {"${sub_pattern_json}": '["a", "b"]'})
    assert _statements(as_list)[0]["Condition"]["StringEquals"][
        "token.actions.githubusercontent.com:sub"
    ] == ["a", "b"]


def test_oidc_role_scope_assignments_match_the_workflow_structure():
    """iam.tf must wire: pull_request + main -> plan, protected env -> apply,
    main ref -> push. This is the human-readable control matrix."""
    iam_text = (REPO_ROOT / "infra" / "iam.tf").read_text()

    def sub_literal(tail: str) -> str:
        return rf'"repo:\$\{{local\.github_repo\}}:{re.escape(tail)}"'

    # plan: a LIST trusting both the PR plan job and the read-only main plan
    plan_pattern = (
        r"\bplan\s*=\s*\[" + sub_literal("pull_request") + r",\s*" + sub_literal("ref:refs/heads/main") + r"\]"
    )
    assert re.search(plan_pattern, iam_text), (
        "iam.tf: plan role must trust exactly repo:<slug>:pull_request AND "
        "repo:<slug>:ref:refs/heads/main"
    )
    apply_pattern = rf'\bapply\s*=\s*{sub_literal("environment:${local.github_environment}")}'
    assert re.search(apply_pattern, iam_text), (
        "iam.tf: apply role must trust ONLY repo:<slug>:environment:aws-synthetic"
    )
    push_pattern = rf'\bpush\s*=\s*{sub_literal("ref:refs/heads/main")}'
    assert re.search(push_pattern, iam_text), (
        "iam.tf: push role must trust ONLY repo:<slug>:ref:refs/heads/main"
    )


def test_github_environment_name_is_consistent_everywhere():
    env_name = _locals_value("github_environment")
    assert env_name == "aws-synthetic"  # ADR-005
    deploy_text = (REPO_ROOT / ".github" / "workflows" / "deploy.yml").read_text()
    assert re.search(rf"environment:\s*{re.escape(env_name)}\b", deploy_text), (
        "deploy.yml apply job must run in the protected environment named in locals.tf"
    )


def test_environment_tfvars_match_the_approved_decisions():
    tfvars = (REPO_ROOT / "infra" / "environments" / "dev.tfvars").read_text()
    assert re.search(r'environment\s*=\s*"dev"', tfvars)
    assert re.search(r'aws_region\s*=\s*"us-east-2"', tfvars)
    assert re.search(r"desired_count\s*=\s*0\b", tfvars)


# ------------------------------------------------- no-account-ID criterion ---


def test_no_account_ids_or_keys_anywhere_under_infra():
    account_id_re = re.compile(r"\b\d{12}\b")
    key_re = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
    identifier_re = re.compile(r"DBS_bG\d")
    for path in sorted((REPO_ROOT / "infra").rglob("*")):
        if not path.is_file():
            continue
        if ".terraform" in path.parts:
            continue
        # Terraform STATE files legitimately contain account IDs and ARNs;
        # they are gitignored (INF-006: state must never be committed). The
        # no-account-IDs rule governs COMMITTED content — .tf files, tfvars,
        # policies, lock files, docs.
        if path.name.startswith("terraform.tfstate") or ".tfstate" in path.name:
            continue
        text = path.read_text(errors="replace")
        assert not account_id_re.search(text), (
            f"{path.relative_to(REPO_ROOT)} contains a literal 12-digit account ID"
        )
        assert not key_re.search(text), f"{path.relative_to(REPO_ROOT)} contains an AKIA key"
        assert not identifier_re.search(text), (
            f"{path.relative_to(REPO_ROOT)} contains a subject identifier"
        )


def test_terraform_state_files_are_gitignored():
    """Belt and braces around the exclusion above: if any state file exists
    on disk it MUST also be ignored by git (INF-006)."""
    import subprocess

    for path in (REPO_ROOT / "infra").rglob("*tfstate*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        result = subprocess.run(
            ["git", "check-ignore", rel], cwd=REPO_ROOT, capture_output=True, text=True
        )
        assert result.returncode == 0, (
            f"{rel} exists on disk but is NOT gitignored — state must never "
            "be committable (INF-006)"
        )


def test_exceptions_register_is_valid_and_dated():
    register = yaml.safe_load(
        (REPO_ROOT / "docs" / "vulnerability-exceptions.yml").read_text()
    )
    entries = register.get("exceptions") or []
    assert isinstance(entries, list)
    for entry in entries:
        for field in ("id", "scope", "reason", "approver", "expires"):
            assert field in entry, f"exception entry missing field {field}: {entry}"
        assert "component" in entry or "component_source" in entry, (
            f"exception {entry['id']} must target a component or a Debian "
            "source package (component_source)"
        )
        assert entry["scope"] in {"vuln", "config"}
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(entry["expires"])), (
            f"exception {entry['id']} must carry a concrete expiry date"
        )
