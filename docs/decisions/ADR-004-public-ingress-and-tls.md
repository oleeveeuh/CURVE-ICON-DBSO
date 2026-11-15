# ADR-004 — Public ingress and TLS

**Status:** Approved by the owner on 2026-11-14 (DD-004)
**Source:** `docs/mlops_extension_plan.md` §17, DD-004
**Decided by:** owner

## Decision

CloudFront in front of the ALB, using the **default `*.cloudfront.net`
distribution domain**, for the ephemeral public synthetic environment:

- TLS terminates at CloudFront on the default distribution domain — no
  domain purchase, no custom ACM certificate, no DNS work.
- Viewer traffic is HTTPS-only (`redirect-to-https`).
- The ALB is internet-facing but accepts traffic **only from CloudFront**:
  its security group allows port 80 from the CloudFront origin-facing
  AWS managed prefix list (`com.amazonaws.global.cloudfront.origin-facing`),
  and the distribution authenticates to the ALB with a secret origin
  header (random value generated in Terraform; infrastructure-level
  credential — no Secrets Manager is introduced for it).
- The ECS task is never directly reachable: it accepts traffic only from
  the ALB security group.
- The environment is **ephemeral**: desired task count zero when idle and
  the environment is destroyed between milestones. Between CloudFront and
  the ALB, traffic is HTTP (origin protocol policy `http-only`); without a
  custom domain there is no ACM certificate for the ALB to terminate TLS
  with, so the secret header plus the prefix-list security group are the
  origin-access controls. The service payload is public synthetic data by
  construction, and no private track ever uses this ingress shape.

## Alternatives considered

- **Custom domain + ACM certificate:** strongest reviewer-facing posture;
  rejected for now because it requires a purchased/approved domain and DNS
  control. Can supersede this ADR later without changing the application.
- **Short-lived CIDR-restricted HTTP ALB only:** cheapest; rejected because
  it is HTTP-only and needs per-demonstration CIDR maintenance.
- **Private service via tunnel or CI runner:** no public cost; rejected
  because reviewers cannot exercise the API and it hides the real ingress
  path this project is meant to demonstrate.

## Decision effect

- Adds a CloudFront distribution (~$0 on the permanent free tier at demo
  traffic) in front of the ALB; ALB cost is unchanged.
- Removes all DNS/ACM setup work from Phase 4.
- Security posture: public HTTPS entry point with CloudFront-only origin
  access; the task security group and the default listener fixed response
  (403 without the origin header) are the structural controls.
- Reviewers access the API at `https://<distribution>.cloudfront.net`.

## Operational restrictions carried by this decision

- The distribution is part of the ephemeral environment and is destroyed
  with it (no always-on distribution).
- Any future routinely-online environment must re-open this decision and
  move to a custom domain + ACM (plan §17 DD-004 recommended option).
