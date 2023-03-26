resource "aws_s3_bucket" "site" {
  bucket = var.example_bucket_name
}

resource "aws_s3_bucket_website_configuration" "site" {
  bucket = aws_s3_bucket.site.id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "error.html"
  }
}

# For completeness this map can be used:
# https://github.com/hashicorp/terraform-template-dir/blob/17b81de441645a94f4db1449fc8269cd32f26fde/variables.tf#L18
resource "aws_s3_object" "index" {
  for_each     = fileset("../website/", "**")
  bucket       = aws_s3_bucket.site.id
  key          = each.value
  source       = "../website/${each.value}"
  etag         = each.value
  acl          = "public-read"
  content_type = lookup({
    ".html" = "text/html"
    ".css"  = "text/css"
    ".js"   = "application/javascript"
  }, regex("\\.[^.]+$", each.key), "")
  content_disposition = "inline"
}

resource "aws_s3_bucket_acl" "site" {
  bucket = aws_s3_bucket.site.id
  acl    = "public-read"
}
