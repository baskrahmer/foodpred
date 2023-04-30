from app import lambda_handler

out = lambda_handler(
    event={
        "queryStringParameters": {
            "query": "persil"
        }
    },
    context=None
)

print(out)
