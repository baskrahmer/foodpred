import numpy as np
import onnxruntime as ort
import torch
from optimum.intel import INCQuantizer
from transformers import PreTrainedTokenizerFast


def quantize(cfg, train, trainer, val):
    from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
    model = trainer.model
    model.config = model.base_model.config
    accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
    tuning_criterion = TuningCriterion(max_trials=600)
    conf = PostTrainingQuantConfig(
        approach="static", backend="default", tuning_criterion=tuning_criterion,
        accuracy_criterion=accuracy_criterion
    )
    from neural_compressor.quantization import fit
    q_model = fit(model=model, conf=conf, calib_dataloader=val, eval_func=eval_func)
    model = trainer.model
    model.config = model.base_model.config
    # Load the quantization configuration detailing the quantization we wish to apply
    quantization_config = PostTrainingQuantConfig(approach="static")
    # Generate the calibration dataset needed for the calibration step
    quantizer = INCQuantizer.from_pretrained(model)
    # Apply static quantization and save the resulting model
    quantizer.quantize(
        quantization_config=quantization_config,
        calibration_dataset=train,
        save_directory=cfg.save_dir,
    )
    return model


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
    tokenizer.save_pretrained(cfg.tokenizer_json_path)
    torch.onnx.export(
        model=model,
        args=torch.ones((1, cfg.max_len), dtype=torch.long, device="cpu"),
        f=cfg.model_onnx_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        },
    )


def test_model_inference(cfg):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_json_path)
    ort_session = ort.InferenceSession(cfg.model_onnx_path)

    def inference_fn(input_str):
        output = inference(input_str, ort_session, tokenizer)
        return np.argmax(output)

    inference_fn("Grah")
    print()
    # todo assertions for output shape; sum of probabilities
