import argparse
import os

import onnxruntime as ort
import torch
import wandb.errors
from lightning import Trainer
from lightning import seed_everything
from optimum.intel import INCQuantizer
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast

from harrygobert.data import get_product_loaders
from harrygobert.model.model import OFFClassificationModel


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
        logger=wandb_logger if cfg.use_wandb else None
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

    input_text = "Your input text goes here"

    tokens = tokenizer(input_text, return_tensors="pt")
    input_ids = tokens["input_ids"].numpy()

    # Run the ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: input_ids}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Process the output as needed
    output = ort_outputs[0].flatten()
    # todo assertions for output shape; sum of probabilities


def get_wandb_logger(cfg):
    try:
        wandb_logger = WandbLogger(project="harrygobert")
    except wandb.errors.UsageError:
        from getpass import getpass
        wandb.login(key=getpass("wandb API token:"))
        wandb_logger = WandbLogger(project="harrygobert")
    return wandb_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    root_path = os.path.abspath(os.path.join(__file__, "../.."))

    # Training settings
    parser.add_argument('--debug', default=True, type=bool, help='Debug mode')
    parser.add_argument('--model_name', default="distilbert-base-multilingual-cased", type=str,
                        help='Name of the pre-trained model')
    parser.add_argument('--n_accumulation_steps', default=1, type=int, help='Number of steps to accumulate gradients')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='Ratio of steps for warmup phase')
    parser.add_argument('--max_len', default=32, type=int, help='Maximum sequence length')
    parser.add_argument('--num_steps', default=1000, type=int, help='Number of steps to train for')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate for optimizer')
    parser.add_argument('--llrd', default=0.7, type=float, help='Layer-wise learning rate decay')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='Weight decay')
    parser.add_argument('--eval_steps', default=50, type=int, help='After how many steps to do evaluation')
    parser.add_argument('--grid_search', default=False, type=bool, help='Whether to run grid search')
    parser.add_argument('--n_folds', default=1, type=int,
                        help='Number of cross-validation folds. 0 trains on full data.')

    # Artefact settings
    parser.add_argument('--save_dir', default=os.path.join(root_path, 'model'), type=str,
                        help='Path to save trained model to')
    parser.add_argument('--quantize', default=False, type=bool, help='Whether or not to quantize the output model')

    # Data settings
    parser.add_argument('--translate', default=True, type=bool, help='Whether to translate text')
    parser.add_argument('--use_cached', default=False, type=bool, help='Whether to use cached data')
    parser.add_argument('--use_subcats', default=False, type=bool, help='Whether to use sub-categories')
    parser.add_argument('--n_classes', default=2473, type=int, help='Number of classes')
    parser.add_argument('--agribalyse_path',
                        default=os.path.join(root_path, 'data/product_to_ciqual.yaml'), type=str,
                        help='Path to Agribalyse data')
    parser.add_argument('--ciqual_dict', default=os.path.join(root_path, 'data/ciqual_dict.yaml'),
                        type=str,
                        help='Path to CIQUAL data')
    parser.add_argument('--csv_path', default=os.path.join(root_path, 'data/products.csv'), type=str,
                        help='Path to CSV products data')
    parser.add_argument('--cache_path', default=os.path.join(root_path, 'data/cache'), type=str,
                        help='Path to CSV products data')

    # Logging settings
    parser.add_argument('--run_name', default="HGV-debug", type=str, help='Name of the run')
    parser.add_argument('--use_wandb', default=True, type=bool, help='Whether to use wandb')

    args = parser.parse_args()

    main(args)
