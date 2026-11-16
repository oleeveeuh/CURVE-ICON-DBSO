# Public synthetic track — Terraform (Phases 4–5)

Single small root module per **ADR-005** (DD-005, approved 2026-11-14).
Nothing here runs automatically; applies go through the protected
GitHub environment cost safeguard documented in
[docs/runbook.md](../docs/runbook.md). **No AWS resource is
created by reading or writing these files.**

## Layout

| File | Contents |
|---|---|
| `versions.tf` | Pinned Terraform (`>= 1.9, < 2.0`) and providers (`aws ~> 5.60`, `random ~> 3.6`); empty S3 backend block (values supplied at init) |
| `providers.tf` | AWS provider with account-wide default tags (INF-009) |
| `locals.tf` | Name prefix, tags, GitHub repo/environment, artifact prefix, origin-header name |
| `variables.tf` | All inputs (region `us-east-2` — the project Region, env `dev`, `desired_count` default 0, budget, state-backend names) |
| `network.tf` | VPC, 2 public subnets, IGW, ALB/task security groups — **no NAT gateways** (prompt constraint; header comment documents the tradeoff) |
| `alb.tf` | Internet-facing ALB, target group (`/ready` health), listener with fixed-403 default + secret-header forward rule (ADR-004) |
| `cloudfront.tf` | CloudFront distribution, default `*.cloudfront.net` TLS, HTTPS-only viewers, disabled caching, secret origin header (ADR-004, INF-012) |
| `ecs.tf` | Fargate cluster, task definition (digest-deployed image), service with `desired_count` default 0 (INF-007) and deployment circuit breaker with rollback (CIC-006) |
| `ecr.tf` | IMMUTABLE-tag repository + lifecycle (keep 10 sha-tagged, expire untagged) (CON-006) |
| `s3.tf` | Versioned private artifact bucket: public access blocked, AES256, TLS-only policy, lifecycle (INF-004) |
| `iam.tf` | ECS execution/task roles; GitHub OIDC provider + plan/apply/push roles trusted for the exact repo + ref/environment (CIC-003, INF-003) |
| `policies/*.json` | The policy documents (template placeholders filled by `iam.tf`) — parsed and scope-asserted offline by `tests/test_iam_policy_scope.py` |
| `observability.tf` | CloudWatch log group, 14-day retention (API-009) |
| `alarms.tf` | OBS-002 alarms (5xx ratio, p95, unhealthy targets, restarts) → SNS email topic + the 2 low-cardinality log metric filters |
| `dashboard.tf` | OBS-001 CloudWatch dashboard incl. model-version visibility |
| `budgets.tf` | Monthly cost budget with $5/$20 notifications (INF-008) |
| `outputs.tf` | Bucket/repo/cluster/service/domain names + GitHub role ARNs |
| `environments/dev.tfvars` | The single environment's values (no secrets; region/env/budget/email) |
| `bootstrap/` | One-time remote-state stack — see [bootstrap/README.md](bootstrap/README.md) |

## Safety

- ✅ **SAFE:** `terraform fmt`, `terraform validate` (`init -backend=false`),
  `tflint`, `trivy config` — all offline, run in CI on every PR.
- ⚠️ **MUTATING:** `terraform plan` (reads live state), and every `apply`.
- ❌ **DESTRUCTIVE:** `terraform destroy` and `update-service` rollbacks.

The exact plan/apply commands live in `docs/runbook.md` — that document,
not this README, is the operational source of truth.

## Account constraint (2026-11-14)

The owner's account is a **new-AWS-experience project account** whose
managed SCP denies `iam:CreateOpenIDConnectProvider`. `enable_github_oidc`
defaults to `false`: no GitHub CI roles exist, and the delivery workflows
stay unused until the project runs on a standard account. Deploys are run
locally per `docs/runbook.md` with the owner's `aws login` session (12-hour
credentials — no long-lived keys anywhere). Also note: inline role policies
on these accounts reject the legacy top-level `Comment` element; the policy
JSONs omit it by design.

## One-time setup checklist (owner)

After the bootstrap stack ([bootstrap/README.md](bootstrap/README.md)):

1. **GitHub protected environment** `aws-synthetic` (repo → Settings →
   Environments):
   - Required reviewer: the owner, if you want the confirmation click
     before billable applies (recommended cost safeguard). The apply/deploy
     job in `.github/workflows/deploy.yml` runs inside this environment
     (CIC-004); the IAM apply role's trust pins this environment name on
     standard accounts.
   - Deployment branches policy: **`main` only**.
2. **GitHub repository variables** (Settings → Secrets and variables →
   Actions → Variables; no secrets needed — OIDC only, CIC-003):

   | Variable | Value source |
   |---|---|
   | `AWS_REGION` | `us-east-2` |
   | `TF_STATE_BUCKET` | `terraform output state_bucket_name` (bootstrap) |
   | `TF_LOCK_TABLE` | `terraform output lock_table_name` (bootstrap) |
   | `TF_STATE_KEY` | `public-synthetic/dev/terraform.tfstate` |
   | `AWS_PLAN_ROLE_ARN` | `terraform output github_oidc_role_arns` (first main apply) |
   | `AWS_APPLY_ROLE_ARN` | same |
   | `AWS_PUSH_ROLE_ARN` | same |

   The first apply is run **locally** by the owner (runbook §"First
   plan/apply"); it creates the roles, whose ARNs then enable the CI paths.
3. **Edit `environments/dev.tfvars`**: replace the budget email placeholder
   and (optionally) fill the state bucket/table names.

## Least-privilege summary (CIC-003 / INF-003)

- `*-gha-plan` — trusted for `repo:oleeveeuh/CURVE-ICON-DBSO:pull_request`
  and `:ref:refs/heads/main`; read-only on AWS + state. No write actions.
- `*-gha-apply` — trusted ONLY for
  `repo:oleeveeuh/CURVE-ICON-DBSO:environment:aws-synthetic` (the protected
  environment); scoped writes; `iam:PassRole` limited to
  `role/curve-icon-dbso-dev-*`; explicit Deny on deleting/repolicing the
  state bucket and lock table; no Secrets Manager/KMS/SSM anywhere.
- `*-gha-push` — trusted for `repo:...:ref:refs/heads/main`; ECR push only.
- Application task role — `s3:Get/GetObjectVersion` on
  `artifacts/synthetic-demo/*` only, `cloudwatch:PutMetricData` bound to
  the project namespace by condition; explicit denies on artifact writes
  and secret stores.

Branch protection of `main` is the owner-side control that keeps
`ref:refs/heads/main` trust meaningful; the GitHub environment's
deployment-branch policy keeps the apply role to `main` runs.
