import boto3;

boto3


def lambda_handler(event, context):
    result = "Hello World"
    return {
        'statusCode': 200,
        'body': result
    }
