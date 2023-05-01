import os

import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from harrygobert.constants import CIQUAL_TO_IDX


class TokenizedDataset(Dataset):
    def __init__(self, data):
        super(TokenizedDataset).__init__()
        self.inputs = data["tokens"]
        self.label = data["label"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return {
            "input_ids": self.inputs[item]['input_ids'][0],
            "attention_mask": self.inputs[item]['attention_mask'][0],
            "labels": self.label[item],
        }


def get_dataloaders(cfg, tokenizer_fn):
    train_products, val_products = get_product_loaders(cfg, tokenizer_fn)
    id_products = get_identity_loader(cfg, tokenizer_fn)
    if cfg.n_folds == 0:
        # Blind training on full dataset
        return train_products + [id_products], []
    else:
        return train_products, val_products + [id_products]


def get_identity_loader(cfg, tokenizer_fn):
    ciqual_dict = yaml.safe_load(open(cfg.ciqual_to_name_path))

    tokens, labels = [], []
    for ciqual, name in ciqual_dict.items():
        tokens.append(tokenizer_fn(name))
        labels.append(CIQUAL_TO_IDX[ciqual])

    data = {"tokens": tokens, "label": labels}
    dataset = TokenizedDataset(data)

    return DataLoader(dataset, batch_size=cfg.val_batch_size, shuffle=False)


def get_product_loaders(cfg, tokenize_fn):
    train_cache = os.path.join(cfg.cache_path, "train.pt")
    val_cache = os.path.join(cfg.cache_path, "val.pt")
    if cfg.use_cached:
        if cfg.n_folds <= 1 and os.path.exists(train_cache):
            return torch.load(train_cache), []
        pass
        # TODO implement dataset caching

    df = pd.read_csv(cfg.csv_path)

    if cfg.debug:
        df = df.head(10000)

    df.dropna(subset=['name', 'ciqual'], inplace=True)
    df['label'] = df['ciqual'].apply(lambda x: CIQUAL_TO_IDX.get(x))
    df.dropna(subset=['label'], inplace=True)
    df.loc[:, 'tokens'] = df['name'].apply(tokenize_fn)

    if cfg.n_folds == 0:
        train_loader = df_to_loader(df, shuffle=True, batch_size=cfg.batch_size)
        return [train_loader], []

    elif cfg.n_folds == 1:
        train_df, val_df = train_test_split(df, test_size=0.8)
        train_loader = df_to_loader(train_df, shuffle=True, batch_size=cfg.batch_size)
        val_loader = df_to_loader(val_df, shuffle=False, batch_size=cfg.val_batch_size)
        return [train_loader], [val_loader]

    else:
        raise NotImplementedError
        # TODO: Set up cross-validation splits
        skf = StratifiedKFold(n_splits=cfg.n_folds)
        folds = {}
        for fold_idx, (train_index, test_index) in enumerate(skf.split(df, df['lang'])):
            train_loader = df_to_loader(df.iloc[train_index], shuffle=True, batch_size=cfg.batch_size)
            val_loader = df_to_loader(df.iloc[test_index], shuffle=False, batch_size=cfg.val_batch_size)
            folds[fold_idx] = {'train': train_loader, 'val': val_loader}

        return folds


def df_to_loader(df, shuffle, batch_size):
    data = {"tokens": df["tokens"].tolist(), "label": df["label"].tolist()}
    dataset = TokenizedDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
