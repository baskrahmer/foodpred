from typing import Callable
from typing import List
import logging

import numpy as np
import pandas as pd
import yaml
from pandas.core.frame import Series, DataFrame
from tqdm import tqdm

from app.app_helpers import preprocess, get_model_function


def get_data(data_path: str, model_path: str, use_cached: bool = False) -> DataFrame:

    if use_cached:
        logging.info("Loading data dict...")
        preprocessed_dict = yaml.safe_load(open("../lci_data.yaml"))
        logging.info("Data dict loaded")
        raw_keys = list(preprocessed_dict.keys())

    else:
        logging.info("Loading data dict...")
        raw = yaml.safe_load(open(data_path))
        logging.info("Data dict loaded")

        preprocessed_dict = {}
        raw_keys = []

        for k, v in tqdm(raw.items(), desc="Reformatting dict"):

            raw_keys.append(v["LCI_name"])
            new_k = preprocess(v["LCI_name"])
            new_v = v["impact_environnemental"]["Score unique EF"]
            preprocessed_dict[new_k] = new_v

        yaml.dump(preprocessed_dict, open("../lci_data.yaml", "w"))

    data = pd.DataFrame(raw_keys)
    data.columns = ['raw']

    # Preprocess the data
    data['preprocessed'] = preprocess_data(data['raw'])

    model = get_model_function(model_path)
    data['embeddings'] = get_embeddings(data['preprocessed'], lambda x: model.encode(x))

    np.save("../embeddings.npy", np.array(list(data["embeddings"])))
    np.save("../preprocessed.npy", np.array(list(data['preprocessed'])))
    np.save("../raw.npy", np.array(list(data['raw'])))

    return data


def preprocess_data(data: DataFrame) -> List:
    preprocessed = []
    for i, title in enumerate(tqdm(data, desc='Preprocessing data')):
        preprocessed.append(preprocess(title))
    return preprocessed


def get_embeddings(data: Series, emb_fn: Callable) -> List:
    embeddings = []
    for i, title in enumerate(tqdm(data, desc=f'Computing embeddings')):
        embeddings.append(emb_fn(title))
    return embeddings


def main():

    get_data(
        data_path="./ciqual_dict.yaml",
        # model_path='sentence-transformers/distiluse-base-multilingual-cased-v1'
        model_path='sentence-transformers/all-MiniLM-L6-v2'
    )


if __name__ == "__main__":
    main()
