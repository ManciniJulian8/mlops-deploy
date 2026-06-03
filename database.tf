resource "google_sql_database_instance" "mlflow" {
  name             = "mlflow"
  database_version = "POSTGRES_15"
  region           = var.region

  deletion_protection = false # Allows cleanup later

  settings {
    tier = "db-f1-micro"
  }
}

resource "google_sql_database" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.mlflow.name
}

resource "random_password" "mlflow" {
  length  = 16
  special = false
  upper   = true
  lower   = true
  numeric = true
}

resource "google_sql_user" "mlflow" {
  name        = "mlflow"
  instance    = google_sql_database_instance.mlflow.name
  password_wo = random_password.mlflow.result
  password_wo_version = 1
}