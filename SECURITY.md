# Security Policy

## Reporting a vulnerability
Please **do not open a public issue** for security problems. Instead, use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
(Security tab → "Report a vulnerability"). We aim to acknowledge within 72 hours.

## Supported versions
This is an actively developed project; security fixes target the `main` branch.

## How RAGGym protects you

**Secrets**
- No secrets in code. Configuration comes from environment / `.env` (gitignored)
  or, in cloud mode, **Azure Key Vault** accessed via **managed identity** (no
  stored credentials).
- `gitleaks` runs as a pre-commit hook and in CI to catch committed secrets.
- The `raggym config` command redacts API keys and connection strings.

**Supply chain**
- `pip-audit` (dependency CVEs), `bandit` (SAST), and `Trivy` (filesystem/image)
  run in CI; `CodeQL` provides static analysis; `Dependabot` keeps deps current.

**Data & access (cloud mode)**
- **Supabase Row-Level Security (RLS)** scopes every user-owned row to its owner.
- **Supabase Auth** gates the app; HTTPS-only ingress (Azure Container Apps).
- Least-privilege RBAC for the deploy service principal and the app identity.

**LLM / RAG-specific**
- Answers are grounded **only** in retrieved context and cite their sources,
  reducing hallucination and making outputs auditable.
- Retrieved/user content is treated as untrusted input (prompt-injection aware);
  generated code in *practice mode* runs only in the user's own workspace via the
  local test runner — never executed server-side from untrusted input.

## Hardening checklist for self-hosting
- [ ] Set all secrets via Key Vault / environment, never in the image.
- [ ] Restrict the deploy SP to the single resource group.
- [ ] Enable Supabase RLS on every table before going multi-user.
- [ ] Keep `min_replicas` low and set spend alerts on the subscription.
- [ ] Rotate API keys and the service principal periodically.
