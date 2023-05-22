import torch
import torchmetrics
from lightning import LightningModule
from torch import nn as nn
from transformers import AutoModel, get_linear_schedule_with_warmup, AutoTokenizer


class OFFClassificationModel(LightningModule):

    def __init__(self, cfg, label_weights=None):
        super().__init__()
        # Training settings
        self.encoder_lr = cfg.encoder_lr
        self.decoder_lr = cfg.decoder_lr
        self.llrd = cfg.llrd
        self.num_steps = cfg.num_steps
        self.weight_decay = cfg.weight_decay

        # Model settings
        self.base_model = AutoModel.from_pretrained(cfg.model_name)
        self.dropout = nn.Dropout(cfg.dropout)
        self.readout = nn.Linear(self.base_model.config.hidden_size, cfg.n_classes)

        # Loss
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor(label_weights, dtype=torch.float32))

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.n_classes)
        self.valid_acc = [torchmetrics.Accuracy(task="multiclass", num_classes=cfg.n_classes) for _ in range(2)]
        if torch.cuda.is_available():
            self.train_acc.to("cuda")
            for acc in self.valid_acc:
                acc.to("cuda")

    @staticmethod
    def mean_pooling(last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def _get_optimizer_parameters(self, no_decay=[]):

        if self.base_model.config.name_or_path == "distilbert-base-multilingual-cased":
            n_layers = self.base_model.config.n_layers
            transformer_prefix = "base_model.transformer.layer."
            readout_prefix = "readout"
            emb_id = "embeddings.word_embeddings"
            pos_id = "embeddings.position_embeddings"
            emb_norm_id = "embeddings.LayerNorm"
        elif self.base_model.config.name_or_path == "xlm-roberta-base":
            n_layers = self.base_model.config.num_hidden_layers
            transformer_prefix = "base_model.encoder.layer."
            readout_prefix = "readout"
            emb_id = "base_model.embeddings.word_embeddings"
            pos_id = "base_model.embeddings.position_embeddings"
            emb_norm_id = "base_model.embeddings.LayerNorm"

        optimizer_params = []

        # Classifier
        classifier_parameters = {
            'params': [p for n, p in self.named_parameters() if n.startswith(readout_prefix)],
            'lr': self.decoder_lr, 'weight_decay': self.weight_decay
        }
        optimizer_params.append(classifier_parameters)

        lr = self.encoder_lr
        weight_decay = self.weight_decay * (self.encoder_lr / self.decoder_lr)

        # Layers
        for i in range(n_layers - 1, -1, -1):
            decay_parameters = {
                'params': [p for n, p in self.named_parameters()
                           if n.startswith(f"{transformer_prefix}{i}.") and not any(nd in n for nd in no_decay)],
                'lr': lr, 'weight_decay': weight_decay
            }
            no_decay_parameters = {
                'params': [p for n, p in self.named_parameters()
                           if n.startswith(f"{transformer_prefix}{i}.") and any(nd in n for nd in no_decay)],
                'lr': lr, 'weight_decay': 0.0
            }

            optimizer_params.append(decay_parameters)
            optimizer_params.append(no_decay_parameters)

            lr *= self.llrd
            weight_decay *= self.llrd

        # Embeddings
        emb_parameters = {
            'params': [p for n, p in self.named_parameters() if emb_id in n],
            'lr': lr, 'weight_decay': 0.0
        }
        pos_emb_params = {
            'params': [p for n, p in self.named_parameters() if pos_id in n],
            'lr': lr, 'weight_decay': 0.0
        }
        emb_norm_parameters = {
            'params': [p for n, p in self.named_parameters() if emb_norm_id in n],
            'lr': lr, 'weight_decay': 0.0
        }
        optimizer_params.append(emb_parameters)
        optimizer_params.append(pos_emb_params)
        optimizer_params.append(emb_norm_parameters)

        return optimizer_params

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        outputs = outputs[:, 0]
        # outputs = torch.cat([outputs[:, 0], self.mean_pooling(outputs, attention_mask)], axis=-1)
        outputs = self.dropout(outputs)
        outputs = self.readout(outputs)

        if labels is not None:
            loss = self.loss(outputs, labels.long())
            if not self.training:
                outputs = nn.functional.softmax(outputs, dim=-1)
            return outputs, loss

        else:
            return nn.functional.softmax(outputs, dim=-1)

    def training_step(self, batch, batch_nb):

        if isinstance(batch, list):
            batch = batch[0]

        outputs, loss = self.forward(**batch)
        self.train_acc(outputs, batch['labels'])

        self.log("train_loss", loss, on_step=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)

        return {"loss": loss}

    def validation_step(self, batch, batch_nb, dataloader_idx):

        if isinstance(batch, list):
            batch = batch[0]

        outputs, loss = self.forward(**batch)
        acc = self.valid_acc[dataloader_idx](outputs, batch['labels'])

        self.log("val_loss", loss, on_epoch=True)
        self.log('valid_acc', acc, on_step=False, on_epoch=True)

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self._get_optimizer_parameters(),
            lr=self.encoder_lr,
            betas=(0.9, 0.95),
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=self.num_steps,
            num_warmup_steps=int(0.1 * self.num_steps)
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    return tokenizer


def get_tokenize_fn(cfg, tokenizer):
    tokenize_fn = lambda x: tokenizer(
        x,
        truncation=True,
        max_length=cfg.max_len,
        return_tensors='pt',
        padding="max_length"
    )
    return tokenize_fn
