import json
import logging

import numpy as np
import yaml

from app_helpers import preprocess, get_model_function
from static import classes

if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

config = yaml.safe_load(open("config.yaml"))
lci_data = yaml.safe_load(open(config["data_file"]))
model, tokenizer = get_model_function(config)


def lambda_handler(event, context):
    if 'warmup' in event:
        return {
            'statusCode': 200,
            'body': json.dumps('Warm-up successful')
        }

    query = event["queryStringParameters"].get("query")

    logging.info("tokenizing")
    tokens = tokenizer(query, return_tensors="pt")
    input_ids = tokens["input_ids"].numpy()

    # Run the ONNX model
    logging.info("running ONNX model")
    ort_inputs = {model.get_inputs()[0].name: input_ids}
    ort_outputs = model.run(None, ort_inputs)

    # Process the output as needed
    probs = ort_outputs[0].flatten()

    argmax_idx = int(np.argmax(probs))
    logging.info(f"argmax_idx: {argmax_idx}")
    pred = classes[argmax_idx]
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
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
    }
