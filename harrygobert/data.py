import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold
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


def get_dataloaders(cfg, tokenizer_fn, df, fold):
    train_products, val_products, label_weights = get_product_loaders(cfg, df, fold)
    id_products = get_identity_loader(cfg, tokenizer_fn)
    if cfg.n_folds == 0:
        # Blind training on full dataset
        return train_products + [id_products], [], label_weights
    else:
        return train_products, val_products + [id_products], label_weights


def get_identity_loader(cfg, tokenizer_fn):
    ciqual_dict = yaml.safe_load(open(cfg.ciqual_to_name_path))

    tokens, labels = [], []
    for ciqual, name in ciqual_dict.items():
        tokens.append(tokenizer_fn(name))
        labels.append(CIQUAL_TO_IDX[ciqual])

    data = {"tokens": tokens, "label": labels}
    dataset = TokenizedDataset(data)

    return DataLoader(dataset, batch_size=cfg.val_batch_size, shuffle=False)


def get_product_loaders(cfg, df, fold):
    label_weights = np.ones(cfg.n_classes)
    label, freq = np.unique(df['label'], return_counts=True)
    for l, f in list(zip(label, freq)):
        label_weights[int(l)] += f
    label_weights /= label_weights.sum() / cfg.n_classes

    if cfg.n_folds == 0:
        train_loader = df_to_loader(df, shuffle=True, batch_size=cfg.batch_size)
        return [train_loader], [], label_weights

    else:
        train_loader = df_to_loader(df[df['fold'] != fold], shuffle=True, batch_size=cfg.batch_size)
        val_loader = df_to_loader(df[df['fold'] == fold], shuffle=False, batch_size=cfg.val_batch_size)
        return [train_loader], [val_loader], label_weights


def df_to_loader(df, shuffle, batch_size):
    data = {"tokens": df["tokens"].tolist(), "label": df["label"].tolist()}
    dataset = TokenizedDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_product_df(cfg, tokenize_fn):
    df = pd.read_csv(cfg.csv_path)
    if cfg.debug:
        df = df.head(10000)
    df.dropna(subset=['name', 'ciqual'], inplace=True)
    df['label'] = df['ciqual'].apply(lambda x: CIQUAL_TO_IDX.get(x))
    df.dropna(subset=['label'], inplace=True)
    df.loc[:, 'tokens'] = df['name'].apply(tokenize_fn)
    df['lang'] = df['lang'].fillna('')

    df = df.reset_index(drop=True)
    df['fold'] = -1

    if cfg.n_folds == 1:
        skf = StratifiedKFold(n_splits=5)
        _, test_index = next(skf.split(df, df['lang']))
        df.loc[test_index, 'fold'] = 0

    elif cfg.n_folds > 1:
        skf = StratifiedKFold(n_splits=cfg.n_folds)
        for fold_idx, (train_index, test_index) in enumerate(skf.split(df, df['lang'])):
            df.loc[test_index, 'fold'] = fold_idx

    return df
