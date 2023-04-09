import torch
from torch import nn as nn
from transformers import AutoModel


class OFFClassificationModel(nn.Module):

    def __init__(self, model_name, n_classes):
        super(OFFClassificationModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.readout = nn.Linear(self.base_model.config.hidden_size, n_classes)
        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def mean_pooling(last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        outputs = outputs[:, 0]
        # outputs = torch.cat([outputs[:, 0], self.mean_pooling(outputs, attention_mask)], axis=-1)
        outputs = self.dropout(outputs)
        outputs = self.readout(outputs)

        if labels is not None:
            loss = self.loss(outputs, labels.argmax(axis=-1))
            if not self.training:
                outputs = nn.functional.softmax(outputs, dim=-1)
            return loss, outputs

        else:
            return nn.functional.softmax(outputs, dim=-1)
