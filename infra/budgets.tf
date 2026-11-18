# AWS Budgets notifications (INF-008) MOVED on 2026-11-16 (pre-publication
# review): the persistent budget now lives in the RETAINED bootstrap stack
# (infra/bootstrap/main.tf) so that destroying the application stack can
# never remove the cost guardrail while the remote-state store survives.
# This file is intentionally empty of resources.
