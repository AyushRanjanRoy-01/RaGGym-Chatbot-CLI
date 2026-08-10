# Azure setup — deploy RAGGym (step by step)

This is the one-time setup to take RAGGym from local to a live Azure URL, with
**auto-deploy on every merge to `main`**. Cost stays ~$0 (Container App scales to
zero; free monthly grant covers light demo traffic). **Tear down when done.**

> You only do this once. Until you finish it, the deploy workflow stays dormant
> (it's gated on the `AZURE_DEPLOY_ENABLED` variable), so your merges stay green.

---

## 0. Prerequisites
- An Azure account — [free tier](https://azure.microsoft.com/free/) is enough.
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) and
  [Terraform](https://developer.hashicorp.com/terraform/install) ≥ 1.6 installed.
- This repo cloned locally.

```bash
az login
az account set --subscription "<YOUR_SUBSCRIPTION_ID>"
```

---

## 1. Provision infrastructure (Terraform)
```bash
cd terraform
terraform init
terraform plan      # review
terraform apply     # type 'yes'
```
Note the outputs:
```bash
terraform output app_url            # your live URL (once an image is deployed)
terraform output resource_group     # e.g. raggym-rg
terraform output container_app_name # e.g. raggym-app
```

---

## 2. Create a service principal for CI
GitHub Actions needs credentials to deploy. Create a scoped SP:
```bash
az ad sp create-for-rbac \
  --name "raggym-github" \
  --role "Contributor" \
  --scopes "/subscriptions/<SUB_ID>/resourceGroups/raggym-rg" \
  --sdk-auth
```
Copy the **entire JSON** it prints (starts with `{ "clientId": ... }`).

---

## 3. Configure GitHub (Settings → Secrets and variables → Actions)
**Secret:**
| Name | Value |
|---|---|
| `AZURE_CREDENTIALS` | the SP JSON from step 2 |

**Variables:**
| Name | Value |
|---|---|
| `AZURE_DEPLOY_ENABLED` | `true` |
| `AZURE_RESOURCE_GROUP` | `raggym-rg` (your `resource_group`) |
| `AZURE_CONTAINERAPP_NAME` | `raggym-app` (your `container_app_name`) |

---

## 4. Add app secrets (LLM provider, optional Supabase)
The app reads these from Key Vault / Container App env. Quickest path — set them
as Container App secrets/env:
```bash
az containerapp secret set -n raggym-app -g raggym-rg \
  --secrets openai-api-key=sk-...

az containerapp update -n raggym-app -g raggym-rg \
  --set-env-vars LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini \
                 OPENAI_API_KEY=secretref:openai-api-key
```
> For zero-cost retrieval-only demos you can skip this — embeddings use FastEmbed
> (no key). But chat answers need an LLM provider.

---

## 5. First deploy
The very first image must exist before `az containerapp update` can roll it out.
Either:
- **Merge to `main`** — the [deploy workflow](../.github/workflows/deploy.yml)
  builds, pushes to GHCR, and rolls out; **or**
- build/push once manually:
  ```bash
  docker build -t ghcr.io/<owner>/<repo>:latest .
  docker push ghcr.io/<owner>/<repo>:latest
  az containerapp update -n raggym-app -g raggym-rg \
    --image ghcr.io/<owner>/<repo>:latest
  ```
Then open `terraform output app_url`.

---

## 6. Verify
- Visit the app URL → the RAGGym chat UI loads.
- Telemetry flows to Application Insights (Azure Portal → your `*-appi`).
- A subsequent merge to `main` triggers an automatic redeploy.

---

## 7. Tear down (avoid any charges)
```bash
cd terraform
terraform destroy
```
Also set `AZURE_DEPLOY_ENABLED` back to `false` (or delete it) to disarm CI.

---

## Cost cheatsheet
| Resource | Idle cost |
|---|---|
| Container App (`min_replicas=0`) | ~$0 (scales to zero) |
| Log Analytics / App Insights | free-tier daily cap |
| Key Vault | a few cents/month |
| GHCR image storage | free |
| LLM calls | pay-per-use (gpt-4o-mini ≈ pennies) |
