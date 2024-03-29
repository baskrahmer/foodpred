import numpy as np
import torch
import wandb
from lightning import Trainer
from lightning import seed_everything

from harrygobert.data import get_dataloaders, get_product_df
from harrygobert.export import export_model_and_tokenizer, test_model_inference
from harrygobert.model import OFFClassificationModel, get_tokenizer, get_tokenize_fn
from harrygobert.util import get_callbacks, get_wandb_logger, parse_args


def main():
    cfg = parse_args()

    seed_everything(1997)

    if cfg.debug:
        cfg.num_steps = 10
        cfg.eval_steps = 6

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    if cfg.use_wandb:
        wandb_logger = get_wandb_logger(cfg)

    tokenizer = get_tokenizer(cfg)
    tokenize_fn = get_tokenize_fn(cfg, tokenizer)

    df = get_product_df(cfg, tokenize_fn)

    fold_accuracies = []
    for fold in range(max(cfg.n_folds, 1)):
        train, val, label_weights = get_dataloaders(
            cfg=cfg,
            tokenizer_fn=tokenize_fn,
            df=df,
            fold=fold,
        )

        model = OFFClassificationModel(cfg, label_weights=label_weights)
        trainer = Trainer(
            accelerator="auto",
            max_steps=cfg.num_steps,
            val_check_interval=cfg.eval_steps,
            check_val_every_n_epoch=None,
            logger=wandb_logger if cfg.use_wandb else None,
            callbacks=get_callbacks(cfg),
            precision=cfg.precision,
        )

        trainer.fit(
            model=model,
            train_dataloaders=train,
            val_dataloaders=val,
        )

        if cfg.n_folds:
            fold_accuracies.append(trainer.callback_metrics['valid_acc/dataloader_idx_0'])

    if cfg.n_folds:
        wandb.log({"kfold_accuracy": np.mean(fold_accuracies)})
    else:
        if cfg.quantize:
            from harrygobert.quantize import quantize
            model = quantize(cfg, trainer, train, val)
        export_model_and_tokenizer(cfg, model, tokenizer)
        test_model_inference(cfg)


if __name__ == '__main__':
    main()
