variable "prefix" {
  description = "Name prefix for all resources."
  type        = string
  default     = "raggym"
}

variable "resource_group_name" {
  description = "Resource group to create."
  type        = string
  default     = "raggym-rg"
}

variable "location" {
  description = "Azure region."
  type        = string
  default     = "eastus"
}

variable "container_app_name" {
  description = "Container App name (must match AZURE_CONTAINERAPP_NAME in CI)."
  type        = string
  default     = "raggym-app"
}

variable "image" {
  description = "Container image to deploy (GHCR). CI updates this on each merge."
  type        = string
  default     = "ghcr.io/ayushranjanroy-01/rag-project:latest"
}

variable "tags" {
  description = "Tags applied to all resources."
  type        = map(string)
  default     = { project = "raggym", managed_by = "terraform" }
}
