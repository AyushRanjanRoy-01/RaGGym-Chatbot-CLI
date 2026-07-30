# Terraform — RAGGym on Azure

Provisions a **near-$0** footprint: a resource group, Log Analytics + Application
Insights, a Key Vault (RBAC), and a **Container App that scales to zero**.

## Prerequisites
- [Terraform](https://developer.hashicorp.com/terraform/install) ≥ 1.6
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) (`az login`)
- An Azure subscription (the free tier is enough)

## Usage
```bash
cd terraform
terraform init
terraform plan      # review — no resources created yet
terraform apply     # creates everything; prints app_url
```

Override defaults via `-var` or a `terraform.tfvars` file:
```hcl
location           = "eastus"
container_app_name = "raggym-app"
image              = "ghcr.io/<owner>/<repo>:latest"
```

## After apply
- `terraform output app_url` → your live URL.
- Set the CI variables/secrets (see [docs/AZURE_SETUP.md](../docs/AZURE_SETUP.md))
  so merges to `main` auto-deploy.

## Tear down (avoid any charges)
```bash
terraform destroy
```

> Cost note: idle cost is ~zero (Container App `min_replicas = 0`). The free
> monthly grant (vCPU-seconds / GiB-seconds / requests) covers light demo use.
> Always `terraform destroy` when you're done demoing.
