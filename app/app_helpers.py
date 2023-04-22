import logging
import re

import onnxruntime as ort
from transformers import PreTrainedTokenizerFast


def preprocess(query: str) -> str:
    # Fix HTML encoding
    query = query.replace('&amp;', '&')
    query = query.replace('&ndash;', '-')
    query = query.replace('&#39;', "'")
    query = query.replace('&#8211;', "-")
    query = query.replace('&lt;', "<")
    query = query.replace('&gt;', ">")
    query = query.replace('&#226;', "â")

    # Remove data inbetween parentheses
    query = re.sub(r" ? \([^)]+\)", "", query)
    query = re.sub(r" ?<[^>]+>", "", query)
    query = re.sub(r" ?\[[^>]+\]", "", query)

    # Remove percentages
    query = re.sub(r'[ -.\d]*%', "", query)

    # Remove leading/trailing whitespace
    query = query.strip()

    return query.lower()


def get_model_function(config):
    onnx_model_path = config["model_path"]
    tokenizer_path = config["tokenizer_path"]
    logging.info("trying to load model...")
    model = ort.InferenceSession(onnx_model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    logging.info("model loaded")
    return model, tokenizer
