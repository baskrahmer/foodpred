resource "aws_ecr_repository" "repo" {
  name = var.ecr_registry
}