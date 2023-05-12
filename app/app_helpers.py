import logging

import onnxruntime as ort
from transformers import PreTrainedTokenizerFast


def get_model_function(config):
    logging.info("loading model")
    model = ort.InferenceSession(config["model_path"])
    logging.info("loading tokenizer")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config["tokenizer_path"])
    logging.info("loading finished")
    return model, tokenizer
