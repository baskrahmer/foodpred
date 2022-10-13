from app.app import lambda_handler
import json

def test_lambda_handler_sample():

    expected = [
        "Pizza, kebab",
        "Pizza, tuna",
        "Pizza, four cheeses",
        "Pizza, seafood",
        "Pizza, chicken"
    ]

    out = lambda_handler({"queryStringParameters": {"query": "Pizza"}}, None)
    str_list = json.loads(out['body'])

    assert out['statusCode'] == 200

    for str in str_list:
        assert str in expected
