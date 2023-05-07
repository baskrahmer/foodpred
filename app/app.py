import json
import logging

import numpy as np
import yaml

from app_helpers import get_model_function
from static import IDX_TO_CIQUAL

if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

config = yaml.safe_load(open("config.yaml"))
lci_data = yaml.safe_load(open(config["data_file"]))
model, tokenizer = get_model_function(config)


def lambda_handler(event, context):
    if event.get('path', '').lower() == '/warmup':
        return {
            'statusCode': 200,
            'body': json.dumps('Warm-up successful')
        }

    query = event["queryStringParameters"].get("query")

    logging.info("tokenizing")
    tokens = tokenizer(query, return_tensors="np")
    input_ids = tokens["input_ids"]

    # Run the ONNX model
    logging.info("running ONNX model")
    ort_inputs = {model.get_inputs()[0].name: input_ids}
    ort_outputs = model.run(None, ort_inputs)
    probs = ort_outputs[0].flatten()
    logging.info("running finished")

    prob = float(np.max(probs))
    argmax_idx = int(np.argmax(probs))
    ciqual_id = IDX_TO_CIQUAL[argmax_idx]
    data = lci_data[ciqual_id]
    logging.info(f"argmax_idx: {argmax_idx}")

    return {
        'statusCode': 200,
        'body': json.dumps({
            'pred': data['name'],
            'prob': prob,
            'ef_score': data['ef_scaled'],
            'ef_phases': data['ef_phases'],
            'co2': data['co2'],
            'co2_phases': data['co2_phases'],
        }),
        'headers': {
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
    }
