resource "aws_ecr_repository" "repo" {
  name = var.ecr_registry
}

#resource null_resource ecr_image {
#  triggers = {
#    python_file = md5(file("${path.module}/lambdas/git_client/index.py"))
#    docker_file = md5(file("${path.module}/lambdas/git_client/Dockerfile"))
#  }
#}

#data "aws_ecr_image" "lambda_image" {
#  #  depends_on = [
#  #    null_resource.ecr_image
#  #  ]
#  #  repository_name = var.ecr_registry
#  repository_name = "hello-world"
#  image_tag       = "latest"
#}
