resource "google_artifact_registry_repository" "mlflow" {
  location      = var.region
  repository_id = "mlflow"
  description   = "Docker repository for MlFlow"
  format        = "DOCKER"
}