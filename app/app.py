import json
import logging

import numpy as np
import yaml

from app_helpers import preprocess, get_model_function

if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

config = yaml.safe_load(open("config.yaml"))
data = np.load(config["raw_names"])
lci_data = yaml.safe_load(open(config["data_file"]))
model, tokenizer = get_model_function(config)


def lambda_handler(event, context):
    query = event["queryStringParameters"].get("query")

    tokens = tokenizer(query, return_tensors="pt")
    input_ids = tokens["input_ids"].numpy()

    # Run the ONNX model
    ort_inputs = {model.get_inputs()[0].name: input_ids}
    ort_outputs = model.run(None, ort_inputs)

    # Process the output as needed
    probs = ort_outputs[0].flatten()

    pred = data[int(np.argmax(probs))]
    prob = float(np.max(probs))
    ef_score = lci_data[preprocess(pred)]['synthese']

    return {
        'statusCode': 200,
        'body': json.dumps({
            'pred': pred,
            'ef_score': ef_score,
            'prob': prob,
        }),
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
    }
