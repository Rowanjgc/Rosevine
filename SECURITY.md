# Security and disclosure

Do not post credentials, private logs, personal configurations or exploit details
in public issues. Use the repository Security tab's **Report a vulnerability**
(private vulnerability reporting) if available. Its availability must be checked;
this repository does not claim that a private reporting channel is enabled.
If unavailable, open a minimal public issue asking the maintainer to arrange a
private channel, without sensitive details. Do not invent a contact address.

Reports should describe affected revision, impact and a minimal synthetic
reproduction. Coordinate a fix and disclosure with the maintainer. There is no
promised response SLA or supported release series during this research stage.

If a real credential is exposed, stop publication, notify the owner privately,
and rotate/revoke through its provider with authorization. Deleting a file does
not remove history, forks or caches. History rewrites and force pushes require
separate explicit authorization. Do not repeat the value in reports.

See the [privacy policy](docs/development/privacy-policy.md) for safe research,
allowlist review and scanner limitations, and the
[hardening report](docs/development/hardening-report.md) for verified control status.
