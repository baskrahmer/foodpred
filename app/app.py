import json
import logging
from datetime import datetime

import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_distances

from .app_helpers import preprocess, get_model_function

if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

config = yaml.safe_load(open("config.yaml"))
data = np.load(config["raw_names"])
emb_matrix = np.load(config["emb_matrix"])
lci_data = yaml.safe_load(open(config["data_file"]))
model_name = config["model_name"]

model = get_model_function(model_name)
emb_fn = lambda x: model.encode(x)


def lambda_handler(event, context):

    query = event["queryStringParameters"].get("query")
    sample_emb = emb_fn(preprocess(query))

    # Compute all distances
    distances = cosine_distances(sample_emb.reshape(1, -1), emb_matrix).reshape(-1)

    # Get k closest job titles
    argmin_indices = np.argsort(distances)[:config["n_suggestions"]]

    # Return canonical form
    output = list(data[argmin_indices])
    lci_output = [lci_data[preprocess(o)]['synthese'] for o in output]

    return {
        'statusCode': 200,
        'body': json.dumps({
            'categories': output,
            'ef_score': lci_output,
            'timestamp': datetime.timestamp(datetime.now()),
        }),
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
}
