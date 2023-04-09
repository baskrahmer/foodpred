import json
import logging

import numpy as np
import torch
import yaml

from app_helpers import preprocess, get_model_function

if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

config = yaml.safe_load(open("config.yaml"))
data = np.load(config["raw_names"])
lci_data = yaml.safe_load(open(config["data_file"]))
model, tokenizer = get_model_function(config["model_path"])


def lambda_handler(event, context):
    query = event["queryStringParameters"].get("query")

    with torch.no_grad():
        tokens = tokenizer(preprocess(query), return_tensors="pt").data
        probs = np.array(model(**tokens).flatten())

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
    }
