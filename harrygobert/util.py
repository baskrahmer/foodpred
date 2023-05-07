import argparse
import os

import lightning.pytorch as pl
import wandb.errors
from pytorch_lightning.loggers import WandbLogger


def get_callbacks(cfg):
    key = "val_loss/dataloader_idx_0"
    return [
        pl.callbacks.ModelCheckpoint(
            monitor=key,
            save_on_train_epoch_end=False,
            dirpath=cfg.save_dir,
            every_n_epochs=1
        ),
        pl.callbacks.EarlyStopping(
            monitor=key,
            min_delta=cfg.es_delta,
            patience=cfg.es_patience,
            verbose=True,
            mode="min",
            check_on_train_epoch_end=False,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    ]


def get_wandb_logger(cfg):
    try:
        wandb_logger = WandbLogger(project="harrygobert")
    except wandb.errors.UsageError:
        from getpass import getpass
        wandb.login(key=getpass("wandb API token:"))
        wandb_logger = WandbLogger(project="harrygobert")
    return wandb_logger


def parse_args():
    parser = argparse.ArgumentParser()
    root_path = os.path.abspath(os.path.join(__file__, "../.."))

    # Training settings
    parser.add_argument('--debug', default=False, type=bool, help='Debug mode')
    parser.add_argument('--precision', default=16, type=int, choices=[16, 32])
    parser.add_argument('--model_name', default="distilbert-base-multilingual-cased", type=str,
                        help='Name of the pre-trained model')
    parser.add_argument('--n_accumulation_steps', default=1, type=int, help='Number of steps to accumulate gradients')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--val_batch_size', default=256, type=int, help='Batch size for validation')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='Ratio of steps for warmup phase')
    parser.add_argument('--max_len', default=32, type=int, help='Maximum sequence length')
    parser.add_argument('--num_steps', default=10000, type=int, help='Number of steps to train for')
    parser.add_argument('--encoder_lr', default=1e-5, type=float, help='Learning rate for optimizer')
    parser.add_argument('--decoder_lr', default=1e-4, type=float, help='Learning rate for optimizer')
    parser.add_argument('--llrd', default=0.7, type=float, help='Layer-wise learning rate decay')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout rate')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='Weight decay')
    parser.add_argument('--eval_steps', default=500, type=int, help='After how many steps to do evaluation')
    parser.add_argument('--n_folds', default=0, type=int,
                        help='Number of cross-validation folds. 0 trains on full data.')

    # Artefact settings
    parser.add_argument('--save_dir', default=os.path.join(root_path, 'out_dir'), type=str,
                        help='Path to save trained model to')
    parser.add_argument('--quantize', default=False, type=bool, help='Whether or not to quantize the output model')
    parser.add_argument('--es_delta', default=0.01, type=float, help='Early stopping delta')
    parser.add_argument('--es_patience', default=10, type=int, help='Early stopping patience')
    parser.add_argument('--tokenizer_json_path', default="tokenizer")
    parser.add_argument('--model_onnx_path', default="model.onnx")

    # Data settings
    parser.add_argument('--translate', default=True, type=bool, help='Whether to translate text')
    parser.add_argument('--n_classes', default=2473, type=int, help='Number of classes')
    parser.add_argument('--agribalyse_path',
                        default=os.path.join(root_path, 'data/product_to_ciqual.yaml'), type=str,
                        help='Path to Agribalyse data')
    parser.add_argument('--ciqual_dict', default=os.path.join(root_path, 'data/ciqual_dict.yaml'),
                        type=str, help='Path to full CIQUAL data')
    parser.add_argument('--ciqual_to_name_path', default=os.path.join(root_path, 'data/ciqual_to_lci_name.yaml'),
                        type=str, help='Path to CIQUAL name dict')
    parser.add_argument('--csv_path', default=os.path.join(root_path, 'data/products.csv'), type=str,
                        help='Path to CSV products data')

    # Logging settings
    parser.add_argument('--run_name', default="HGV-debug", type=str, help='Name of the run')
    parser.add_argument('--use_wandb', default=True, type=bool, help='Whether to use wandb')
    return parser.parse_args()
