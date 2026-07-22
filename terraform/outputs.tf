output "app_url" {
  description = "Public HTTPS URL of the Streamlit app."
  value       = "https://${azurerm_container_app.app.ingress[0].fqdn}"
}

output "resource_group" {
  value = azurerm_resource_group.rg.name
}

output "container_app_name" {
  value = azurerm_container_app.app.name
}

output "key_vault_name" {
  value = azurerm_key_vault.kv.name
}

output "appinsights_connection_string" {
  value     = azurerm_application_insights.appi.connection_string
  sensitive = true
}
