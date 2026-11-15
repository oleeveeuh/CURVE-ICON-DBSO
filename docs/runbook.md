# Runbook — public synthetic AWS track (Phases 3–4)

**Scope:** the single ephemeral `dev` environment described by
[ADR-004](decisions/ADR-004-public-ingress-and-tls.md) and
[ADR-005](decisions/ADR-005-terraform-organization.md) — CloudFront
(default `*.cloudfront.net` TLS) → ALB → ECS Fargate, scale-to-zero when
idle, destroyed between milestones. Synthetic artifacts only
([data_access.md](data_access.md)).

**Research only; not for clinical use.** Nothing in this environment
carries real data, real artifacts, or real-derived material.

---

## Safety legend

| Marker | Meaning |
|---|---|
| ✅ SAFE | read-only, or touches nothing billable; run freely |
| ⚠️ MUTATING | creates or changes billable AWS resources |
| 🛑 APPROVAL-GATED | requires an explicit owner decision recorded in the conversation (Prompt 6) before running |
| ❌ DESTRUCTIVE | removes resources or state; double-check before running |

**Rules that always apply**

- No static AWS keys anywhere: humans use their own console/SSO/profile
  credentials; CI uses OIDC roles (CIC-003). If a credential ever leaks,
  follow the incident procedure in [data_access.md](data_access.md).
- Terraform state is never committed, copied, or e-mailed (INF-006).
- Never run a ❌ DESTRUCTIVE command against the wrong region — every
  command below assumes `AWS_REGION=us-east-2` (or passes it explicitly).

---

## 0. Prerequisites

✅ SAFE — install and check tooling:

```bash
terraform -version        # >= 1.9, < 2.0 (infra/versions.tf pins providers)
tflint --version
trivy --version
docker --version          # optional locally; CI always has it
aws --version; aws sts get-caller-identity   # confirms WHO you are acting as
```

Run everything in a **personal account dedicated to this demo** if
possible; the apply role's wildcards (documented in
[infra/README.md](../infra/README.md)) are safest in an account with
nothing else in it.

**Credentials on this machine (2026-11-14):** the AWS CLI `aws login`
browser session under profile **`main`** (region us-east-2; 12-hour
sessions, 90-day renewal). The login cache is NOT readable by Terraform's
SDK directly — bridge it for every mutating command:

```bash
eval "$(aws configure export-credentials --profile main --format env)"
```

**Account constraint (observed 2026-11-14):** this account is a
*new-AWS-experience project account*. Its managed service control policy
DENIES `iam:CreateOpenIDConnectProvider`, so `enable_github_oidc = false`
here: there are no GitHub CI roles, and the two workflows cannot run
against this account. Deploys are LOCAL (the commands in this runbook);
on a standard account set `enable_github_oidc = true` and §2 applies as
written. The strict IAM validation on these accounts also rejects inline
role policies carrying the legacy top-level `Comment` element — the
policy JSONs under `infra/policies/` deliberately omit it (guarded by
tests).

## 1. One-time remote-state bootstrap

See [infra/bootstrap/README.md](../infra/bootstrap/README.md). Summary:

```bash
cd infra/bootstrap
terraform init                                                # ✅ SAFE
terraform plan -var owner=olivialiau                          # ✅ SAFE (preview)
terraform apply -var owner=olivialiau                         # ⚠️ MUTATING (~$0.5/mo total)
terraform output state_bucket_name                            # ✅ SAFE — record it
terraform output lock_table_name                              # ✅ SAFE — record it
```

Then edit `infra/environments/dev.tfvars`: set the real
`budget_alert_email` (the deploy workflow refuses to apply while the
`REPLACE` placeholder remains) and uncomment/fill
`state_bucket_name` / `state_lock_table`.

## 2. One-time GitHub repository setup

> **Only possible on standard accounts** — the new-AWS-experience project
> account's SCP denies OIDC providers, so with `enable_github_oidc = false`
> there are no CI roles to wire up. Skip this section until the project
> moves to a standard account.

✅ SAFE (settings only):

1. **Environments → create `aws-synthetic`** (exact name — the IAM trust
   policy pins it):
   - Required reviewers: the owner (this IS the human-approval gate, CIC-004).
   - Deployment branches policy: `main` only.
2. **Variables** (Settings → Secrets and variables → Actions → Variables;
   no secrets — OIDC only):

   | Variable | Value |
   |---|---|
   | `AWS_REGION` | `us-east-2` |
   | `TF_STATE_BUCKET` | bootstrap output `state_bucket_name` |
   | `TF_LOCK_TABLE` | bootstrap output `lock_table_name` |
   | `TF_STATE_KEY` | `public-synthetic/dev/terraform.tfstate` |
   | `AWS_PLAN_ROLE_ARN` / `AWS_APPLY_ROLE_ARN` / `AWS_PUSH_ROLE_ARN` | outputs of the FIRST apply (step 3) |

3. **Branch protection on `main`** (owner-side control that keeps the
   `ref:refs/heads/main` trust meaningful).

## 3. First plan / first apply

```bash
cd infra
terraform init \
  -backend-config="bucket=<state_bucket_name>" \
  -backend-config="dynamodb_table=<lock_table_name>" \
  -backend-config="key=public-synthetic/dev/terraform.tfstate" \
  -backend-config="region=us-east-2"                       # ✅ SAFE (after this, init is cached)

terraform plan \
  -var-file=environments/dev.tfvars                        # ✅ SAFE (read-only preview)
```

🛑 **APPROVAL-GATED — the first apply creates every billable resource
(ALB, CloudFront, ECS, ECR, S3, budgets, ~$25/month if left running).**
It requires the owner's explicit approval in the conversation (Prompt 6
review of the plan) before running:

```bash
terraform apply \
  -var-file=environments/dev.tfvars \
  -var "container_image=$(aws ecr describe-images \
        --repository-name curve-icon-dbso-dev-api \
        --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true)" || \
terraform apply -var-file=environments/dev.tfvars           # ⚠️ MUTATING, 🛑 APPROVAL-GATED
```

(The first apply may simply omit `container_image` — the service starts
with `desired_count = 0`, so the placeholder image reference is never run;
INF-007.) Record the outputs:

```bash
terraform output                                 # ✅ SAFE
terraform output github_oidc_role_arns           # ✅ SAFE — set the three AWS_*_ROLE_ARN variables in GitHub
```

## 4. Routine deploys (after setup)

⚠️ MUTATING + 🛑 the `aws-synthetic` environment approval: push to `main`
(app/app/infra paths) or trigger **Deploy → workflow_dispatch**. The
workflow builds once by SHA, resolves the immutable digest, plans, waits
for your approval in the `aws-synthetic` environment, applies, waits for
ECS stability, and smoke-tests through CloudFront. Nothing to run by hand.

## 5. Verify a deployment

✅ SAFE:

```bash
DOMAIN=$(cd infra && terraform output -raw cloudfront_domain)

curl -fsS "https://$DOMAIN/ready"       # 200 once the artifact is verified
curl -fsS "https://$DOMAIN/health" | python3 -m json.tool
curl -fsS "https://$DOMAIN/model-info" | python3 -m json.tool   # intended_use + beats_baseline visible

curl -fsS -X POST -H 'content-type: application/json' \
  --data @tests/fixtures/synthetic_predict_request.json \
  "https://$DOMAIN/predict"             # synthetic payload ONLY — never real features

# Task is NOT directly reachable (INF-005): the task SG admits only the ALB SG
aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=curve-icon-dbso-dev-task-sg" \
  --query 'SecurityGroups[].IpPermissions'

# Logs are metadata-only (API-008/009)
aws logs tail "$(cd infra && terraform output -raw log_group_name)" --since 10m

# Idempotency (INF-010): a second plan with the SAME var set reports no changes
cd infra && terraform plan \
  -var-file=environments/dev.tfvars \
  -var "container_image=<the deployed registry/repo@sha256:...>" \
  -var "desired_count=1"                # ✅ SAFE — expect: "No changes."
```

## 6. Scale to zero (idle between demonstrations)

⚠️ MUTATING (removes running-task cost; keeps ALB/CloudFront):

```bash
cd infra
terraform apply -var-file=environments/dev.tfvars -var "desired_count=0"
aws ecs describe-services \
  --cluster "$(terraform output -raw ecs_cluster_name)" \
  --services "$(terraform output -raw ecs_service_name)" \
  --query 'services[0].runningCount'    # ✅ SAFE — expect 0
```

## 7. Rollback

Automatic: the service's **deployment circuit breaker rolls back** a
failing deployment by itself (CIC-006), and every prior task revision
stays registered in ECS.

Manual rollback to the revision captured before the last deploy
(the deploy summary prints it):

```bash
aws ecs update-service \
  --cluster curve-icon-dbso-dev-cluster \
  --service api \
  --task-definition <PRIOR_TASK_DEFINITION_ARN>     # ⚠️ MUTATING
aws ecs wait services-stable \
  --cluster curve-icon-dbso-dev-cluster --services api   # ✅ SAFE

# Roll back the IMAGE only, re-using CI: re-run Deploy (workflow_dispatch)
# on the prior commit SHA instead of hand-editing task definitions, so
# Terraform state and reality stay aligned.
```

Note: a manual `update-service --task-definition` intentionally drifts
from Terraform's recorded task definition; the next workflow deploy
re-converges it (Terraform re-registers from `var.container_image`).

## 8. Budget alerts (INF-008)

You will receive email at `budget_alert_email`:

- **$5 ACTUAL** — something is spending; expect the ALB (~$16/mo is the
  dominant fixed cost once created). If tasks are at zero and you still
  burn: check for orphaned resources (§10) and consider destroying (§9).
- **$20 FORECASTED / $20 ACTUAL** — the monthly ceiling; destroy the
  environment unless a demonstration explicitly needs it.

⚠️ Alert emails do not stop spend — only you do (§6/§9).

## 9. Destroy the environment

🛑 APPROVAL-GATED (Prompt 6/7 owner decision) + ❌ DESTRUCTIVE:

```bash
cd infra
terraform plan -destroy -var-file=environments/dev.tfvars   # ✅ SAFE preview first
terraform destroy \
  -var-file=environments/dev.tfvars \
  -var "state_bucket_name=<state_bucket_name>" \
  -var "state_lock_table=<lock_table_name>"                # ❌ DESTRUCTIVE
```

The artifact bucket is `force_destroy` (synthetic-only content by policy),
so destroy completes in one pass. The **remote-state stack is NOT
destroyed** — see §10/§11.

### State retention (INF-006)

After destroying, the following are intentionally RETAINED:

- `infra/bootstrap` state bucket + lock table (the main module's history
  lives there; deleting it would orphan any future environment).
- `infra/bootstrap/terraform.tfstate` (local, gitignored) — keep this file
  private and safe; without it, the bootstrap resources can only be
  removed by manual console work.
- AWS Budgets (created by the main module — destroyed with it; the
  bootstrap stack has no budget). Recreate on next apply.

## 10. After-destroy orphan check (plan §14)

✅ SAFE — run all; every command should return empty (or only the
retained remote-state items). Tag-based filtering uses INF-009 tags.

```bash
aws ecs list-clusters                     --query 'clusterArns[?contains(@,`curve-icon-dbso`)]'
aws elbv2 describe-load-balancers         --query 'LoadBalancers[?contains(LoadBalancerName,`curve-icon-dbso`)]'
aws elbv2 describe-target-groups          --query 'TargetGroups[?contains(TargetGroupName,`curve-icon-dbso`)]'
aws ec2 describe-nat-gateways             --filter "Name=tag:Project,Values=curve-icon-dbso" --query 'NatGateways[?State!=`deleted`]'
aws ec2 describe-addresses                --query 'Addresses[?tags[?Key==`Project`&&Value==`curve-icon-dbso`]]'   # stray public IPs
aws ecr describe-repositories             --query 'repositories[?contains(repositoryName,`curve-icon-dbso`)]'
aws s3api list-buckets                    --query "Buckets[?contains(Name,'curve-icon-dbso')][]" \
  --output json | grep -v tfstate || true   # retained: the tfstate bucket
aws logs describe-log-groups              --query 'logGroups[?contains(logGroupName,`curve-icon-dbso`)]'
aws cloudwatch describe-alarms            --query 'MetricAlarms[?contains(AlarmName,`curve-icon-dbso`)]'
aws cloudfront list-distributions         --query "DistributionList.Items[?contains(Comment,'curve-icon-dbso')]"
aws budgets describe-budgets --account-id "$(aws sts get-caller-identity --query Account --output text)" \
  --query 'Budgets[?contains(BudgetName,`curve-icon-dbso`)]'
# KMS: no customer-managed keys are created by design (SSE-S3); any KMS hit is an orphan.
aws kms list-aliases --query 'Aliases[?contains(AliasName,`curve-icon-dbso`)]'
```

**Expected leftovers (the documented remote-state exceptions):** the
`curve-icon-dbso-tfstate-*` bucket, the `*-tfstate-locks` table, and the
GitHub OIDC provider + `*-gha-*` IAM roles only if you did not also
destroy them with the environment (they are part of the main module, so a
full destroy removes them; re-created on next apply).

## 11. Cost (check before apply — figures are planning estimates, not quotes)

Pricing varies by region and changes over time. **Verify against the
AWS pricing pages (Fargate, ELB, CloudFront, S3, ECR, CloudWatch) in
us-east-2 before the first apply.** Rough order of magnitude:

| Item | Est. monthly (always-on) | Notes |
|---|---|---|
| ALB | ~$16–18 | the dominant cost; exists even with zero tasks |
| Fargate task (0.25 vCPU / 0.5 GB) | ~$8–9 | $0 when scaled to zero |
| CloudFront | ~$0 | permanent free tier covers demo traffic; PriceClass_100 |
| ECR + S3 + logs | ~$0.5–1 | tiny synthetic artifact + 14-day log retention |
| State bucket + lock table | ~$0.5 | bootstrap stack, retained |
| **Total always-on** | **~$25–28** | |
| **Scaled to zero** | **~$17–19** | ALB only |
| **Destroyed** | **~$0.5–1** | state store only |

Budget notifications arrive at $5 and $20 (INF-008). The intended pattern
per the plan (§14): **destroy between milestones**; the environment is
ephemeral by decision (ADR-004).

---

## Requirement map

| Runbook section | Requirements |
|---|---|
| §1 bootstrap | INF-006 |
| §2 GitHub setup | CIC-003, CIC-004 |
| §3 first plan/apply | CIC-004, INF-010 |
| §4 deploys | CIC-005, CIC-006, CON-006 |
| §5 verify | INF-005, INF-010, API-008 |
| §6 scale-to-zero | INF-007 |
| §7 rollback | CIC-006 |
| §8 budgets | INF-008 |
| §9 destroy + state retention | INF-006, plan §14 |
| §10 orphan check | plan §14 |
