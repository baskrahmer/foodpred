terraform {
  required_version = ">= 1.2.0"

  backend "s3" {
    region = "eu-central-1"
    key    = "oidc/terraform.tfstate"
  }
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = "eu-central-1"
}

provider "aws" {
  alias  = "cert_provider"
  region = "us-east-1"
}

provider "cloudflare" {
}
