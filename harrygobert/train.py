import onnxruntime as ort
import torch
from lightning import Trainer
from lightning import seed_everything
from optimum.intel import INCQuantizer
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast

from harrygobert.data import get_product_loaders
from harrygobert.model.model import OFFClassificationModel
from harrygobert.util import get_callbacks, get_wandb_logger, parse_args


def main(cfg):
    seed_everything(1997)

    if cfg.use_wandb:
        wandb_logger = get_wandb_logger(cfg)

    model = OFFClassificationModel(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenize_fn = lambda x: tokenizer(
        x,
        truncation=True,
        max_length=cfg.max_len,
        return_tensors='pt',
        padding="max_length"
    )

    train, val = get_product_loaders(cfg, tokenize_fn)

    # val, train = make_agribalyse_data_loaders(config=cfg)
    # train = train.map(tokenize_fn, batched=True)
    # val = val.map(tokenize_fn, batched=True)

    trainer = Trainer(
        accelerator="auto",
        max_steps=cfg.num_steps,
        val_check_interval=cfg.eval_steps,
        check_val_every_n_epoch=None,
        logger=wandb_logger if cfg.use_wandb else None,
        callbacks=get_callbacks(cfg),
    )

    trainer.fit(
        model=model,
        train_dataloaders=train,
        val_dataloaders=val,
    )

    if cfg.grid_search:
        raise NotImplementedError

    # TODO: set up (cross-)validation procedure
    # TODO: Output mapping in correct format

    if cfg.quantize:
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

    model.eval()

    tokenizer_json_path = "./tokenizer"
    onnx_model_path = "./model.onnx"

    tokenizer.save_pretrained(tokenizer_json_path)

    # You may need to adjust the input size based on your specific model architecture
    input_size = (1, cfg.max_len)  # Example: (batch_size, sequence_length)

    dummy_input = torch.ones(input_size, dtype=torch.long, device="cpu")
    dynamic_axes = {
        "input": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"},
    }
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    ort_session = ort.InferenceSession(onnx_model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_json_path)

    input_text = "Grah"

    def inference_fn(input_str):
        import numpy as np
        output = inference(input_str, ort_session, tokenizer)
        return np.argmax(output)

    inference_fn("Grah")
    # todo assertions for output shape; sum of probabilities


def inference(input_text, ort_session, tokenizer):
    tokens = tokenizer(input_text, return_tensors="pt")
    input_ids = tokens["input_ids"].numpy()
    # Run the ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: input_ids}
    ort_outputs = ort_session.run(None, ort_inputs)
    # Process the output as needed
    output = ort_outputs[0].flatten()
    return output


if __name__ == '__main__':
    args = parse_args()
    main(args)
