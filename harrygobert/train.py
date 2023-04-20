import argparse
import numpy as np
import os
import torch
from optimum.intel import INCQuantizer
from transformers import AutoTokenizer
from transformers import TrainingArguments
from typing import Callable

from harrygobert.data import make_agribalyse_data_loaders
from harrygobert.model.model import OFFClassificationModel
from train_tools import compute_metrics, CustomCallback, CustomTrainer


def get_model_fn(cfg) -> Callable:
    def get_model():
        model = OFFClassificationModel(
            model_name=cfg.model_name,
            n_classes=cfg.n_classes
        )

        for p in model.base_model.parameters():
            p.requires_grad = False

        return model

    return get_model


def main(cfg):
    try:
        import wandb
        wandb.login()
        use_wandb = True
    except Exception:
        print("Weights & Biases not configured properly")
        use_wandb = False

    val, train = make_agribalyse_data_loaders(
        agribalyse_path=cfg.agribalyse_path,
        ciqual_path=cfg.ciqual_dict,
        config=cfg
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    tokenize_fn = lambda x: tokenizer(x['text'], truncation=True, max_length=cfg.max_len)
    train = train.map(tokenize_fn, batched=True)
    val = val.map(tokenize_fn, batched=True)

    train_class_weights = np.zeros(cfg.n_classes)
    for t in train:
        train_class_weights += np.array(t['label']) / cfg.n_classes

    val_class_weights = np.zeros(cfg.n_classes)
    for v in val:
        val_class_weights += np.array(v['label']) / cfg.n_classes

    training_args = TrainingArguments(
        output_dir="test_trainer",
        do_eval=True,
        report_to="wandb" if use_wandb else "all",
        evaluation_strategy="steps",
        logging_strategy="steps",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.n_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=1,
        log_on_each_node=False,
        num_train_epochs=cfg.num_epochs,
        overwrite_output_dir=True,
        run_name=cfg.run_name,
        warmup_ratio=cfg.warmup_ratio,
        eval_steps=cfg.eval_steps,
        fp16=True,
        weight_decay=cfg.weight_decay
    )

    trainer = CustomTrainer(
        model_init=get_model_fn(cfg),
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.add_callback(CustomCallback(trainer))

    if cfg.grid_search:
        trainer.hyperparameter_search(
            direction="maximize",
            backend="ray",
            n_trials=1
        )
    else:
        trainer.train()

    torch.save(trainer.model, f=os.path.join(cfg.save_dir, "model.pt"))
    torch.save(tokenizer, f=os.path.join(cfg.save_dir, "tokenizer.pt"))

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
    if cfg.quantize:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    root_path = os.path.abspath(os.path.join(__file__, "../.."))

    # Training settings
    parser.add_argument('--debug', default=False, type=bool, help='Debug mode')
    parser.add_argument('--model_name', default="distilbert-base-multilingual-cased", type=str,
                        help='Name of the pre-trained model')
    parser.add_argument('--n_accumulation_steps', default=1, type=int, help='Number of steps to accumulate gradients')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='Ratio of steps for warmup phase')
    parser.add_argument('--max_len', default=32, type=int, help='Maximum sequence length')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='Weight decay')
    parser.add_argument('--eval_steps', default=100, type=int, help='After how many steps to do evaluation')
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

    args = parser.parse_args()

    main(args)
