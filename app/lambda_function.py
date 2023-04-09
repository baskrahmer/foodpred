import json
import random


def lambda_handler(event, context):
    input = event.get("queryStringParameters", {}).get("product", None)
    if input is None:
        return {
            'statusCode': 400,
            'body': json.dumps({"message": 'Please provide named parameter "product"'})
        }
    if random.random() < 0.5:
        return {
            'statusCode': 200,
            'body': json.dumps({
                "pred": "Pastis (anise-flavoured spirit)",
                "ef_score": 0.12303632999999999,
                "confidence": 0.4,
            })
        }
    else:
        return {
            'statusCode': 200,
            'body': json.dumps({
                "pred": "Cheese and ham, breaded",
                "ef_score": 0.52999348,
                "confidence": 0.6,
            })
        }
