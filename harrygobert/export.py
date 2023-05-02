import os

import numpy as np
import onnxruntime as ort
import torch
from transformers import PreTrainedTokenizerFast


def inference(input_text, ort_session, tokenizer):
    tokens = tokenizer(input_text, return_tensors="pt")
    input_ids = tokens["input_ids"].numpy()

    # Run the ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: input_ids}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Process the output as needed
    output = ort_outputs[0].flatten()
    return output


def export_model_and_tokenizer(cfg, model, tokenizer):
    model.eval()
    tokenizer.save_pretrained(os.path.join(cfg.save_dir, cfg.tokenizer_json_path))
    torch.onnx.export(
        model=model,
        args=torch.ones((1, cfg.max_len), dtype=torch.long, device="cpu"),
        f=os.path.join(cfg.save_dir, cfg.model_onnx_path),
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        },
    )


def test_model_inference(cfg):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(os.path.join(cfg.save_dir, cfg.tokenizer_json_path))
    ort_session = ort.InferenceSession(os.path.join(cfg.save_dir, cfg.model_onnx_path))

    def inference_fn(input_str):
        output = inference(input_str, ort_session, tokenizer)
        assert len(output) == cfg.n_classes
        assert 1 - sum(output) < 1e-5
        return np.argmax(output)

    inference_fn("Test")
