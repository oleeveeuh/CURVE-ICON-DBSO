# Phase 7 — optional local Kubernetes demonstration (measured)

**Scope: local `kind` only. No EKS, GKE, or any cloud Kubernetes resource
exists or was created (verified in §7).** All model/data inputs are the
same deterministic synthetic generator used everywhere else
([data_access.md](data_access.md)); `/explain` remains deferred per
[ADR-002](decisions/ADR-002-explanation-endpoint.md). Research only; not
for clinical use.

## 1. Why this phase exists (and why it almost didn't)

Phase 5's ECS measurements (see [deployment_report.md](deployment_report.md))
showed no technical problem for Kubernetes to solve: p95 110 ms against a
200 ms target, zero failures, zero restarts, flat memory, automated
circuit-breaker rollback, scale-to-zero cost control. The honest learning
goal of this phase is therefore **not** "fix something" — it is to compare
Kubernetes' declarative primitives (probes, rolling updates, Helm
rollbacks, HPA on resource metrics) against the ECS equivalents on
identical synthetic traffic, and to say candidly what each buys.

## 2. What was built

| Artifact | Purpose |
|---|---|
| [deploy/kind/kind-config.yaml](deploy/kind/kind-config.yaml) | single-node local cluster; maps `localhost:8000` → NodePort 30080 so the unchanged Locust profile drives it |
| [deploy/helm/curve-icon-dbso/](deploy/helm/curve-icon-dbso/) | chart: Deployment (non-root 10001, read-only root FS, `drop ALL` capabilities, seccomp RuntimeDefault), NodePort Service, HPA (CPU 70 %, 1→3 replicas), `helm test` readiness check |
| metrics-server (cluster-scoped, not in the repo) | resource metrics for HPA + `kubectl top` |

**Metrics/dashboards decision (OBS note):** the application intentionally
exposes no `/metrics` endpoint, and a Prometheus/Grafana stack would add
a third-party subsystem to a repository whose telemetry contract is
metadata-only ECS logs. Resource metrics via metrics-server cover the
HPA and capacity evidence; anything richer would bloat the core project,
so it was **not** added (the prompt's explicit escape hatch).

## 3. Reproducible setup (tools: kind 0.33.0, helm 4.3.2, kubectl 1.37.2, colima/`docker` backend)

```bash
kind create cluster --config deploy/kind/kind-config.yaml
docker build --build-arg SOURCE_SHA=$(git rev-parse HEAD) -t curve-icon-dbso-api:phase7 .
docker build --build-arg FAILURE_DEMO=1 --build-arg SOURCE_SHA=$(git rev-parse HEAD) \
  -t curve-icon-dbso-api:phase7-failure .          # rollback candidate only
kind load docker-image curve-icon-dbso-api:phase7 curve-icon-dbso-api:phase7-failure \
  --name curve-icon-dbso
helm install phase7 deploy/helm/curve-icon-dbso
kubectl rollout status deployment/phase7-api --timeout=180s
helm test phase7                                   # readiness hook: Succeeded
curl -fsS http://127.0.0.1:8000/ready              # via the NodePort mapping
# metrics for HPA/top:
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.7.2/components.yaml
kubectl patch deployment metrics-server -n kube-system --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

Teardown:

```bash
helm uninstall phase7
kind delete cluster --name curve-icon-dbso
docker rmi curve-icon-dbso-api:phase7 curve-icon-dbso-api:phase7-failure
```

## 4. Demonstrated behavior (2026-11-15, all local)

1. **Probes**: readiness (`/ready`, 2 s period) gates every rollout;
   liveness (`/health`) held. During the failure rollout the probe log
   shows the fail-closed path verbatim: connection refused during
   startup, then `Readiness probe failed: HTTP probe failed with
   statuscode: 503` on every cycle.
2. **Rolling update** (benign spec change): surge pod created → Ready →
   old pod terminated; total ≈ 10 s with `maxUnavailable: 0`.
3. **Rollback**: deploying the `FAILURE_DEMO=1` image (same mechanism as
   the ECS exercise — artifact removed, `/ready` 503 forever) stalled the
   rollout with the old pod **still serving 200 throughout** (zero
   dropped requests, like the ECS circuit breaker). `helm rollback` to
   the prior revision recovered immediately; the good pod was never
   interrupted.
4. **HPA scaling**: CPU 70 % target over the 100m request, 1→3 replicas,
   demonstrated under the load in §5.
5. **Load**: the unchanged canonical Locust shape (1→50 users over 5 min,
   50 sustained 10 min; synthetic payloads only) — measured values in §5.

## 5. Measured results (2026-11-15, single-node kind on colima 2 CPU / 4 GiB)

Canonical Locust shape unchanged (1→50 users / 5 min ramp, 50 sustained
10 min; synthetic payloads only). Reports: `deploy/kind/results_kind_*.csv`,
`deploy/kind/report_kind.html` (aggregate statistics only — no payloads).

| Metric (aggregated) | kind (localhost → NodePort) | ECS via CloudFront/ALB (same shape, [deployment_report.md](deployment_report.md)) |
|---|---|---|
| Requests | **36,900** | 34,177 |
| Failures | **25 (0.068 %)** — all `RemoteDisconnected`, **all inside the 14:16:40–41 UTC-5 window** | **0 (0 %)** |
| Median | 3 ms | 88 ms |
| **p95** | **19 ms** | 110 ms |
| p99 / max | 140 ms / 855 ms | 140 ms / 350 ms |
| Peak throughput | ~41 req/s | ~38 req/s |
| Restarts | 0 | 0 |

**The 25 failures are a demo-procedure artifact, not a service defect:**
they are exactly the in-flight keep-alive connections broken when the
operator ran a mid-run `helm upgrade` (the fix that made the HPA target
reachable) through the single-pod NodePort mapping at ~50 req/s. Zero
failures before that instant, zero after it, and ECS — which had no
mid-run deployment — showed none. On ECS the same shape ran with an
ALB/CloudFront path and no interrupted rollout.

**HPA evidence** (`kubectl describe hpa` / sampled log):
`ScalingActive True / ValidMetricFound`; `SuccessfulRescale — New size:
2; reason: cpu resource utilization above target` then `New size: 3`;
after load, scale-down back to 1 replica (idle utilization 16 %/70 %).
Pod memory flat ≈ 125 Mi per replica throughout. Two chart defects were
found *by* the demonstration and fixed: (1) the `helm test` pod carried
the Deployment's selector labels, so the HPA tried to compute utilization
over a completed pod with no CPU request (`FailedGetResourceMetric`);
(2) the measured compute was so light (~60 m CPU at full protocol load)
that the 70 % threshold over the original 100 m request was unreachable —
the CPU request is now 50 m with the measurement documented in
[values.yaml](deploy/helm/curve-icon-dbso/values.yaml).

**Local resource usage:** kind node container ≈ 26 % of 2 CPUs /
**1.16 GiB** (of colima's 4 GiB); control-plane node inside the cluster
≈ 209 m / 1.19 GiB; three app replicas ≈ 125 Mi each; images ≈ 1 GB
(kind node) + 2 × app image, all reclaimed at teardown
(`docker rmi` + `kind delete cluster`; ~6 s total).

## 6. ECS-versus-Kubernetes conclusion (candid)

**Kubernetes solved no measured problem for this project.** The Phase 5
bottleneck analysis stands: the service is compute-trivial (a replica
needs ≈ 0.06–0.08 core under the full protocol load — memory flat at
125 Mi), and the kind p95 of 19 ms versus ECS's 110 ms measures a
*shorter network path* (localhost NodePort versus CloudFront→ALB→Fargate),
not a better service. The failures Kubernetes introduced in this run
(25, via the mid-run upgrade) were operator-procedure artifacts, but they
illustrate the honest asymmetry: ECS gave this demo a zero-failure
15-minute run *including* its deployment exercises, because the
circuit-breaker path never dropped a serving connection.

What Kubernetes demonstrably adds: automatic CPU-target autoscaling
(HPA 1→3→1, ECS `desired_count` in this demo was manual by design),
declarative probes (ECS equivalent lives in the task definition), Helm
revision rollback (parity with the ECS circuit breaker, both
zero-downtime), and portability of the deployment contract. What it
costs: a cluster lifecycle, metrics-server, Helm, node resource overhead
(≈ 1.16 GiB resident for an idle-capable node), and a failure mode
(HPA silently `FailedGetResourceMetric` over a label-collision) that
took instrumentation to catch.

**Recommendation: unchanged — the public synthetic track stays on ECS
Fargate.** The kind setup is retained under [deploy/](deploy/) as a
reproducible learning artifact for reviewers who specifically want the
Kubernetes rendering of the same service; it is not needed to run,
verify, or cost the demonstration.

## 7. Cloud-Kubernetes confirmation

No EKS/GKE/AKS or any cloud Kubernetes resource was created: this phase
ran entirely against a local `kind` cluster on the existing local Docker
engine (colima). During the run `kubectl config current-context` read
`kind-curve-icon-dbso`; that cluster and its context were **removed at
teardown** (§3), so no `kind-*` context exists now. The AWS account
contains no Kubernetes resource (the EKS API was never called; the
retained AWS resources remain the Terraform remote-state bucket + lock
table from the Phase 5 teardown, plus — since 2026-11-16 — the
persistent budget in the retained bootstrap stack).
