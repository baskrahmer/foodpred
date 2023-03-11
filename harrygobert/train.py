import numpy as np
from transformers import AutoTokenizer
from transformers import TrainingArguments

from data import make_agribalyse_data_loaders
from model.model import OFFClassificationModel
from train_tools import compute_metrics, CustomCallback, CustomTrainer


class CFG:
    debug = False

    # Training settings
    model_name = "xlm-roberta-base"
    n_accumulation_steps = 1
    batch_size = 64
    warmup_ratio = 0.1
    max_len = 16
    num_epochs = 20
    learning_rate = 1e-2

    # Data settings
    translate = True
    use_cached = True
    use_subcats = False
    n_classes = 2473
    agribalyse_path = '../data/product_to_ciqual.yaml'
    ciqual_dict = '../data/ciqual_dict.yaml'

    # Logging settings
    run_name = "HGV-debug"


def main():
    try:
        import wandb
        wandb.login()
        use_wandb = True
    except Exception:
        print("Weights & Biases not configured properly")
        use_wandb = False

    val, train = make_agribalyse_data_loaders(
        agribalyse_path=CFG.agribalyse_path,
        ciqual_path=CFG.ciqual_dict,
        config=CFG
    )

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    def get_model():

        model = OFFClassificationModel(
            model_name=CFG.model_name,
            n_classes=CFG.n_classes
        )

        for p in model.base_model.parameters():
            p.requires_grad = False

        return model

    tokenize_fn = lambda x: tokenizer(x['text'], truncation=True, max_length=CFG.max_len)
    train = train.map(tokenize_fn, batched=True)
    val = val.map(tokenize_fn, batched=True)

    train_class_weights = np.zeros(CFG.n_classes)
    for t in train:
        train_class_weights += np.array(t['label']) / CFG.n_classes

    val_class_weights = np.zeros(CFG.n_classes)
    for v in val:
        val_class_weights += np.array(v['label']) / CFG.n_classes

    training_args = TrainingArguments(
        output_dir="test_trainer",
        do_eval=True,
        report_to="wandb" if use_wandb else "all",
        evaluation_strategy="steps",
        logging_strategy="steps",
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        gradient_accumulation_steps=CFG.n_accumulation_steps,
        learning_rate=CFG.learning_rate,
        logging_steps=1,
        log_on_each_node=False,
        num_train_epochs=CFG.num_epochs,
        overwrite_output_dir=True,
        run_name=CFG.run_name,
        warmup_ratio=CFG.warmup_ratio,
        eval_steps=100,
        fp16=False,
        # weight_decay=0.01
    )

    trainer = CustomTrainer(
        model_init=get_model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.add_callback(CustomCallback(trainer))
    trainer.train()

    trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        n_trials=1
    )

    # TODO: set up (cross-)validation procedure
    # TODO: Output mapping in correct format

    return


if __name__ == '__main__':
    main()
