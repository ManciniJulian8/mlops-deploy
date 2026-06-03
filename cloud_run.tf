resource "null_resource" "docker_build_push" {
  provisioner "local-exec" {
    command = "gcloud builds submit --project ${var.project} --tag ${var.region}-docker.pkg.dev/${var.project}/mlflow/mlflow ./assets/"
  }

  depends_on = [google_artifact_registry_repository.mlflow]
}

resource "google_cloud_run_v2_service" "mlflow" {
  name     = "mlflow"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  deletion_protection = false # Allows cleanup later

  depends_on = [null_resource.docker_build_push]

  template {
    max_instance_request_concurrency = 10

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [
          google_sql_database_instance.mlflow.connection_name,
        ]
      }
    }
    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project}/mlflow/mlflow:latest"

      ports {
        container_port = 5000
      }

      command = ["mlflow"]
      args = [
        "server",
        "--port", "5000",
        "--host", "0.0.0.0",
        "--allowed-hosts", "*.run.app,localhost",
        "--default-artifact-root", google_storage_bucket.mlflow.url,
        "--backend-store-uri", "postgresql+psycopg2://mlflow:${random_password.mlflow.result}@/mlflow?host=/cloudsql/${var.project}:${var.region}:${google_sql_database_instance.mlflow.name}",
      ]

      volume_mounts {
        mount_path = "/cloudsql"
        name       = "cloudsql"
      }

      resources {
        cpu_idle = false
        limits = {
          cpu    = "2000m" # 2 vCPU
          memory = "4096Mi"
        }
      }
    }
  }
}

resource "google_cloud_run_service_iam_binding" "access_to_mlflow" {
  location = google_cloud_run_v2_service.mlflow.location
  service  = google_cloud_run_v2_service.mlflow.name
  role     = "roles/run.invoker"
  members  = ["allUsers"]
}