# tflint configuration for infra/ (CI lint gate).
#
# The bundled `terraform` ruleset only (syntax, naming, deprecated syntax,
# type checks). The deep `aws` plugin is deliberately NOT enabled: it
# requires a plugin download and its value overlaps with the trivy IaC
# security scan that also runs in CI (see .github/workflows/ci.yml,
# "Terraform lint & security").

plugin "terraform" {
  enabled = true
  preset  = "recommended"
}

rule "terraform_deprecated_index" {
  enabled = true
}
