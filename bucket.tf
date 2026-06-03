resource "google_storage_bucket" "mlflow" {
  name                        = "mlflow-${var.project}"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  force_destroy = true # Allows cleanup later
}