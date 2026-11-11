# GitHub protections: owner setup and verification

No remote settings are changed by Project Zero. After explicitly authorizing
publication, first publish the feature branch and let the Guardrails workflow run.
Then configure Settings → Rules → Rulesets for `main`: Active, require a pull
request, zero required approving reviews (single maintainer), require status check
`guardrails` from GitHub Actions, block force pushes and restrict branch deletion.
Require the branch up to date if compatible with the maintainer's workflow.
Do not grant an ordinary admin/agent bypass. Require conversation resolution.
Verify both main creation/update and deletion scopes; test a disposable feature
branch under an equivalent temporary rule if needed. Do not test destructive
operations on main. Never enable a required check before its exact name exists.

Alternatively use classic branch protection: require PRs with zero required
approvals, require `guardrails`, include administrators, disable force pushes and
deletion. No rule can replace the maintainer's deliberate review. Protecting main
prevents merge/direct changes; CI after a public branch push cannot undo disclosure.
Native push protection covers recognized secrets, not arbitrary personal data.

Settings → Code security: keep secret scanning and push protection enabled;
enable Dependabot alerts and consider security-update PRs once dependency manifests
exist. Enable private vulnerability reporting. Review non-provider pattern options
for availability/noise. Do not bypass unknown secret findings.

Settings → Actions → General: retain read repository contents permission, disallow
Actions approving PRs, require approval for outside contributors, restrict allowed
actions to approved GitHub actions and require immutable pins if available. No
repository secrets or writable token are needed by this workflow. Review changes
to hooks, workflows, allowlists and policies as consequential changes. Avoid
`pull_request_target` execution of contributor code and privileged artifact reuse.

Read-only verification (requires owner credentials; do not print authentication):

```sh
gh api repos/Rowanjgc/Rosevine --jq '.security_and_analysis'
gh api repos/Rowanjgc/Rosevine/branches/main/protection
gh api repos/Rowanjgc/Rosevine/rulesets
gh api repos/Rowanjgc/Rosevine/vulnerability-alerts --silent
gh api repos/Rowanjgc/Rosevine/actions/permissions
gh api repos/Rowanjgc/Rosevine/actions/permissions/workflow
```

An empty ruleset list or “Branch not protected” is not enforcement. Inspect each
active ruleset's conditions, rules and bypass actors; a successful alerts request
indicates enabled alerts, while a not-found response requires checking permissions
and settings. Verify a successful workflow run and the required status context.
Review the Security tab for private reporting availability. Record actual enabled
state; do not infer protection from checked-in YAML.

Official references: [protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches),
[rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets),
[push protection](https://docs.github.com/en/code-security/how-tos/secure-your-secrets/work-with-leak-prevention/push-protection-in-the-github-ui),
[Gitleaks usage](https://github.com/gitleaks/gitleaks),
[pinned release](https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1).
