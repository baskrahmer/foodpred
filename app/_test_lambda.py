from app import lambda_handler

out = lambda_handler(
    event={
        "queryStringParameters": {
            "query": "Falafels"
        }
    },
    context=None
)

print(out)
