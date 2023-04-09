import argparse
import os

import numpy as np
import torch
from optimum.intel import INCQuantizer
from transformers import AutoTokenizer
from transformers import TrainingArguments

from harrygobert.data import make_agribalyse_data_loaders
from harrygobert.model.model import OFFClassificationModel
from train_tools import compute_metrics, CustomCallback, CustomTrainer


def get_model_fn(cfg):
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
        eval_steps=100,
        fp16=False,
        # weight_decay=0.01
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
    # trainer.train()

    torch.save(trainer.model, f=os.path.join(cfg.save_dir, "model.pt"))
    torch.save(tokenizer, f=os.path.join(cfg.save_dir, "tokenizer.pt"))

    state_dict = trainer.model.state_dict()

    torch.onnx.export(trainer.model)

    # torch.jit.trace(trainer.model, f=os.path.join(cfg.save_dir, "model_jit.pt"))

    # trainer.save_model(cfg.save_dir)

    # trainer.hyperparameter_search(
    #     direction="maximize",
    #     backend="ray",
    #     n_trials=1
    # )

    # # Set the accepted accuracy loss to 5%
    # accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)
    # # Set the maximum number of trials to 10
    # tuning_criterion = TuningCriterion(max_trials=10)
    # quantization_config = PostTrainingQuantConfig(
    #     approach="dynamic", accuracy_criterion=accuracy_criterion, tuning_criterion=tuning_criterion
    # )
    # quantizer = INCQuantizer.from_pretrained(trainer.model, eval_fn=eval_fn)
    # quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)

    # TODO: set up (cross-)validation procedure
    # TODO: Output mapping in correct format

    if False:
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
    if False:
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
    parser.add_argument('--save_dir', default="../model", type=str, help='Path to save trained model to')

    # Data settings
    parser.add_argument('--translate', default=True, type=bool, help='Whether to translate text')
    parser.add_argument('--use_cached', default=True, type=bool, help='Whether to use cached data')
    parser.add_argument('--use_subcats', default=False, type=bool, help='Whether to use sub-categories')
    parser.add_argument('--n_classes', default=2473, type=int, help='Number of classes')
    parser.add_argument('--agribalyse_path', default='../data/product_to_ciqual.yaml', type=str,
                        help='Path to Agribalyse data')
    parser.add_argument('--ciqual_dict', default='../data/ciqual_dict.yaml', type=str, help='Path to CIQUAL data')

    # Logging settings
    parser.add_argument('--run_name', default="HGV-debug", type=str, help='Name of the run')

    args = parser.parse_args()

    main(args)
