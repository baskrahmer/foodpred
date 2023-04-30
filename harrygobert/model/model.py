import torch
import torchmetrics
from lightning import LightningModule
from torch import nn as nn
from transformers import AutoModel, get_linear_schedule_with_warmup


class OFFClassificationModel(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.lr = cfg.learning_rate
        self.num_steps = cfg.num_steps
        self.base_model = AutoModel.from_pretrained(cfg.model_name)
        self.dropout = nn.Dropout(cfg.dropout)
        self.readout = nn.Linear(self.base_model.config.hidden_size, cfg.n_classes)
        self.loss = nn.CrossEntropyLoss()
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

    def _get_optimizer_parameters(self):

        base_params = {
            "params": self.base_model.parameters(),
            "lr": self.lr * 0.1
        }
        readout_params = {
            "params": self.readout.parameters(),
            "lr": self.lr
        }

        return [base_params, readout_params]

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
        self.log('valid_acc', acc, on_step=True, on_epoch=True)

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self._get_optimizer_parameters(),
            lr=self.lr,
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
