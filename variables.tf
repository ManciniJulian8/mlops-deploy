variable "project" {
  description = "The project in which the resources will be created"
  type        = string
}

variable "region" {
  description = "The region in which the resources will be created"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The zone in which the resources will be created"
  type        = string
  default     = "us-central1-a"
}