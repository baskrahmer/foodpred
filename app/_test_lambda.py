from app import lambda_handler

out = lambda_handler(
    event={
        "queryStringParameters": {
            "query": "pizza"
        }
    },
    context=None
)

print(out)
