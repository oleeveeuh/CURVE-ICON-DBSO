# Runbook — public synthetic AWS track (Phases 3–5)

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
(In this checkout both were filled on 2026-11-14 — the bucket is
`curve-icon-dbso-tfstate-fdf9b4aa`, the table `curve-icon-dbso-tfstate-locks`
— and the values survive a destroy, so a redeploy does not need to redo
this step unless the bootstrap stack is rebuilt.)

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

## Apply warning — ALWAYS pass the full variable set

Every `terraform apply` on this root must carry `-var-file=environments/dev.tfvars`
**plus** `-var "container_image=<repo-url>@sha256:..."` and
`-var "desired_count=1"` (or the intended count). An apply that omits them
silently reverts to the tfvars idle defaults — **desired_count = 0 and the
bootstrap placeholder image** — taking a running deployment down. This bit
once during the Phase 5 observability apply (2026-11-15) and was restored
within minutes; the alarm/watch pattern is: after ANY apply, re-check
`desired_count` via §5 verify.

## 8b. Alarm triage (OBS-002) {#alarm-triage}

Four alarms notify `budget_alert_email` via the `${prefix}-alarms` SNS
topic (one-time email confirmation required after first apply). Dashboard:
CloudWatch → Dashboards → `curve-icon-dbso-dev-dashboard`.

| Alarm | Fires when | First actions (✅ SAFE unless noted) |
|---|---|---|
| `*-http-5xx-ratio-high` | >1 % of requests 5xx for 5 min | `aws logs tail /ecs/curve-icon-dbso-dev --since 15m` — look for `internal_error` classes; check `RunningCount` (§5); recent deploy? roll back (§7) |
| `*-target-p95-latency-high` | p95 > 200 ms for 5 min | confirm with the dashboard p50/p95 widget; single task at 0.25 vCPU — check ECS CPU/Mem widget; scale-out is NOT available by design (demo) — reduce load or roll back a recent image change |
| `*-unhealthy-targets` | any target unhealthy 5 min | `aws ecs describe-services ... --query 'services[0].[runningCount,deployments]'`; failing health checks = failing image → §7 rollback |
| `*-unexpected-restarts` | ≥2 server starts in 5 min (crash loop) | `aws logs tail` for `error_class` on the startup lines; a rolling deploy contributes exactly 1 — bursts mean crash loop → §7 rollback |

Alarm-email subscription is pending until the owner clicks the SNS
confirmation link (sent on first apply).

## 8c. Load test (OBS-003/OBS-004) {#load-test}

✅ SAFE (read-only against the service; generates load):

```bash
# 15 minutes total: ramp 1→50 users over 5 min, sustain 50 for 10 min.
# Payloads are generated schema-valid synthetic rows only.
uv run --extra cloud locust -f locust/locustfile.py --headless \
  -H https://$(cd infra && terraform output -raw cloudfront_domain) \
  --csv locust/results --html locust/report.html
```

Acceptance (measured per run, archived with versions): p95 < 200 ms,
error rate < 0.1 %, zero task restarts, stable ECS memory. Report files
(`locust/results_*.csv`, `report.html`) contain aggregate statistics
only — never payloads. Archive them with the model version (from
`/model-info`), image digest (`aws ecr describe-images`), and the
task-definition revision.

## 8d. Drift demonstration (OBS-006/OBS-007) {#drift-demo}

✅ SAFE (synthetic data, local execution):

```bash
.venv/bin/python scripts/train_synthetic_model.py        # rebuild with reference aggregates
.venv/bin/python scripts/drift_demo.py                   # clean vs shifted samples
.venv/bin/python scripts/drift_demo.py --output reports/drift-$(date +%F).json
```

The clean sample (same cohort seed as the artifact reference) must show
PSI ≈ 0 on every feature; the shifted sample crosses the 0.2 warning
threshold. PSI > 0.2 here is a documented educational warning, NOT
validated clinical drift monitoring and NOT a model-decay claim. The
report contains aggregate PSI floats only — never raw feature values.

## 8e. Failure and rollback exercise (Phase 5, owner-gated)

🛑 **APPROVAL-GATED**: deploying a deliberately failing image requires the
owner's explicit go in the conversation. The mechanism: the image builds
with `--build-arg FAILURE_DEMO=1` (Dockerfile), which removes the trained
artifact directory — the fail-closed readiness path (MOD-003) then returns
503 forever, and the deployment fails ALB health checks. Expected chain:
unhealthy-target alarm → deployment circuit breaker rolls back
automatically → service returns to the prior known-good revision.

```bash
SHA12=$(git rev-parse --short=12 HEAD)
docker build --build-arg FAILURE_DEMO=1 --build-arg SOURCE_SHA=$(git rev-parse HEAD) \
  -t curve-icon-dbso-api:sha-$SHA12-failure-demo .
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-2.amazonaws.com
docker tag curve-icon-dbso-api:sha-$SHA12-failure-demo \
  <account>.dkr.ecr.us-east-2.amazonaws.com/curve-icon-dbso-dev-api:sha-$SHA12-failure-demo
docker push ...:sha-$SHA12-failure-demo        # ⚠️ MUTATING (ECR)
DIGEST=$(aws ecr describe-images --repository-name curve-icon-dbso-dev-api \
  --image-ids imageTag=sha-$SHA12-failure-demo --query 'imageDetails[0].imageDigest' --output text)
# 🛑 owner-approved only:
terraform apply -var-file=environments/dev.tfvars -auto-approve \
  -var "container_image=<repo-url>@$DIGEST" -var "desired_count=1"
# then: watch the alarm fire, confirm the circuit breaker rollback, and
# CONVERGE back to the good digest (terraform apply -var container_image=<good-digest> ...).
# Do not end the exercise with the environment unhealthy.
```



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

The artifact bucket is `force_destroy` (synthetic-only content by policy).
The ECR repository is NOT `force_delete`: if it still holds pushed images,
destroy fails on `RepositoryNotEmptyException` after everything else is
gone (hit 2026-11-15). Delete the images first, then re-run the destroy —
it will remove only the empty repository:

```bash
aws ecr batch-delete-image --repository-name ${prefix}-api \
  --image-ids "$(aws ecr describe-images --repository-name ${prefix}-api \
    --query 'imageDetails[].{imageDigest:imageDigest}' --output json)"  # ❌ DESTRUCTIVE
terraform destroy -var-file=environments/dev.tfvars          # then: 1 resource
```

The **remote-state stack is NOT destroyed** — see §10/§11.

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

**Measured operating costs (2026-11-15, us-east-2, one deployment):**
with one task running ≈ **$0.84/day** (task ≈ $0.30 + ALB ≈ $0.54;
CloudFront/ECR/S3/logs negligible) — this matches the ~$25–28 always-on
planning estimate above and would breach the $20 budget inside a month.
Scaled to zero: ≈ $0.54/day. Destroyed: ≈ $0 (state store only). See
[deployment_report.md](deployment_report.md) §7–§8 for the burn-down
and teardown evidence.

**Teardown evidence (2026-11-15, first full destroy):** preview 50 to
destroy → 49 removed + 1 failure (ECR `RepositoryNotEmptyException`,
§9 two-step) + 1 destroyed on re-run; §10 orphan check returned empty
for every resource class; `terraform state list` empty; retained items
exactly the documented remote-state bucket + lock table. No orphaned
billable resource remained.

---

## 12. Incident response

Safety events this runbook covers: suspected credential leak, real
data/identifier exposure in the public track, and a service that is
unhealthy or behaving unexpectedly in the cloud track.

1. **Suspected credential or key leak** (any `AKIA…`, password, token,
   or private key): rotate/revoke the credential immediately in the AWS
   console or provider, then follow **Report**/**Preserve** in
   [data_access.md](data_access.md § "Incident-response trigger"). This
   repository uses no static keys by construction (CIC-003: humans use
   profile sessions; CI uses OIDC only, and OIDC is disabled on this
   account) — so a leak here most likely means a machine-level profile
   cache: revoke the `aws login` session and renew.
2. **Suspected real data, subject codes, or real-derived artifacts in
   the public track** (repo, image, logs, CI artifact, cloud account):
   **Stop → Preserve → Report → Remediate** exactly as written in
   [data_access.md](data_access.md). Do NOT rewrite git history or
   delete evidence before the exposure is documented (owner decision). Then: destroy the cloud environment (§9), purge the
   affected ECR images (§9 two-step), and re-scan
   (`uv run python scripts/security_scan.py`, plus a history audit)
   before any redeploy.
3. **Unhealthy or misbehaving deployment** (alarms from §8b, or any
   5xx/latency/restart burst): triage per §8b; roll back per §7
   (automatic circuit breaker or manual prior revision); if the service
   cannot be made healthy quickly, scale to zero (§6) or destroy (§9) —
   never leave the environment unhealthy (plan §7.12).
4. **Budget breach** ($5/$20 notifications, §8): verify desired count
   (§5), check for orphaned resources (§10), scale to zero (§6) or
   destroy (§9). Alert emails do not stop spend — only you do.

After any incident: record what happened, what crossed the boundary
(if anything), and the remediation, in the conversation log and — for
governance-relevant events — in [approvals.md](approvals.md) as redacted
status only; verification evidence lives in the compliance register
([data_access.md](data_access.md)) and outside the repository.

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
| §12 incident response | data_access.md incident trigger, CIC-003, plan §7.12 |
