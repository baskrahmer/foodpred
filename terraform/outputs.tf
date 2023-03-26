output "bucket_name" {
  value = aws_s3_bucket.site.bucket
}

output "bucket_url" {
  value = aws_s3_bucket.site.bucket_domain_name
}