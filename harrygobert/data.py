import os
import time
from typing import List

import pandas as pd
import torch
import yaml
from datasets import Dataset, load_from_disk
from googletrans import Translator
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from util import CIQUAL_TO_IDX

SEPARATOR = ' & '


def make_val_dataset(mapping, off_cats, use_subcats, preprocess_jumbo=True, separator=SEPARATOR):
    # build up dictionary from OFF cat name "en: ... " to its integer index
    off_cat_to_idx = {}
    for i, cat in enumerate(off_cats['tags']):
        off_cat_to_idx[cat['id']] = i

    jumbo_cats = []
    off_cats = []
    for m in mapping.values():

        if m['off_category'] not in off_cat_to_idx:
            # TODO: log this
            continue

        # get/preprocess Jumbo categories
        title = m['title']
        if not use_subcats:
            title = title.split(' --> ')[-1]
        elif preprocess_jumbo:
            title = separator.join(reversed(title.split(' --> ')))
        jumbo_cats.append(title)

        # link OFF categories to idx
        off_cats.append(off_cat_to_idx[m['off_category']])

    return Dataset.from_dict({'text': jumbo_cats, 'label': off_cats})


def make_off_dataset(off_cats):
    off_names = []

    for cat in off_cats['tags']:
        off_names.append(cat['name'])

    return Dataset. \
        from_dict({'text': off_names, 'label': list(range(len(off_names)))})


def map_subcats_to_classes(jumbo_cats, separator=SEPARATOR) -> List[str]:
    classes = []

    def recursive_helper(products, cat_str):

        for product in products:

            title = product.get('title')
            class_str = title + (separator + cat_str if cat_str else '')

            if product.get('subCategories'):
                recursive_helper(product.get('subCategories'), class_str)
            else:
                classes.append(class_str)

    recursive_helper(jumbo_cats, '')

    return classes


class ValidationDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx]


class MappingDataset(Dataset):

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], idx


def make_agribalyse_data_loaders(config, languages={'nl', 'en'}):
    agribalyse_path = config.agribalyse_path
    ciqual_path = config.ciqual_path

    products_path = os.path.join(config.cache_path, 'train_ds')
    identity_path = os.path.join(config.cache_path, 'val_ds')

    if config.use_cached and os.path.exists(products_path) and os.path.exists(identity_path):
        print("Loading cached data")
        train_ds = load_from_disk(products_path)
        val_ds = load_from_disk(identity_path)

    else:
        # Load data from disk
        source_data = yaml.safe_load(open(agribalyse_path))
        ciqual_dict = yaml.safe_load(open(ciqual_path))  # This takes a while
        n_labels = len(ciqual_dict.keys())

        # Mapping between product code and label index
        ciqual_to_idx = {}
        idx_to_ciqual = {}
        for idx, ciqual_code in enumerate(ciqual_dict.keys()):
            ciqual_to_idx[ciqual_code] = idx
            idx_to_ciqual[idx] = ciqual_code

        # Extract natural language description of LCI categories
        names, labels = [], []
        for idx, (ciqual_code, product) in enumerate(tqdm(ciqual_dict.items(), desc='Mapping identity')):
            names.append(product['LCI_name'])
            labels.append([float(i == idx) for i in range(n_labels)])
        identity = Dataset.from_dict({'text': names, 'label': labels})

        names, labels = [], []
        for lang in languages:
            for product, ciqual_codes in tqdm(source_data[lang].items(), desc=f'Mapping {lang} products'):
                ciqual_codes = {c for c in ciqual_codes if c in ciqual_dict}
                ciqual_codes = {ciqual_to_idx[i] for i in ciqual_codes}
                if ciqual_codes:
                    names.append(product)
                    labels.append([int(i in ciqual_codes) / len(ciqual_codes) for i in range(n_labels)])
        products = Dataset.from_dict({'text': names, 'label': labels})

        products.save_to_disk(products_path)
        identity.save_to_disk(identity_path)

        train_ds = products
        val_ds = identity

    return train_ds, val_ds


def get_data(jumbo_path, off_path, mapping_path, config):
    # jumbo_cats = yaml.safe_load(open(jumbo_path, 'r'))
    # jumbo_flattened = map_subcats_to_classes(jumbo_cats)

    if config.use_cached:
        print("Loading cached data")
        train_ds = load_from_disk('data/cache/train_ds')
        val_ds = load_from_disk('data/cache/val_ds')

    else:

        off_cats = yaml.safe_load(open(off_path, 'rb'))
        jumbo_to_off_mapping = yaml.safe_load(open(mapping_path, 'r'))

        train_ds = make_off_dataset(off_cats)
        val_ds = make_val_dataset(jumbo_to_off_mapping, off_cats, use_subcats=config.use_subcats)

        if config.debug:
            train_ds, val_ds = sample_data(train_ds, val_ds)

        if config.translate:
            val_ds = translate_data(val_ds)

        # Cache the data
        if not config.debug:
            print("Caching data")
            train_ds.save_to_disk('data/cache/train_ds')
            val_ds.save_to_disk('data/cache/val_ds')

    return train_ds, val_ds


class DutchToEnglishTranslator:

    def __init__(self):
        self.translator = Translator()
        self.n_sleep = 60

    def __call__(self, text, max_retries=10, *args, **kwargs):

        if max_retries == 0:
            raise RuntimeError

        try:
            return self.translator.translate(text, src='nl').text
        except:
            time.sleep(self.n_sleep)
            return self(text, max_retries - 1)


def sample_data(train, val):
    return train.select(range(10)), val.select(range(10))


def translate_data(val):
    translator = DutchToEnglishTranslator()
    return val.map(lambda x: {'text': translator(x['text'])})


class ProductDataset(Dataset):
    def __init__(self, data):
        super(ProductDataset).__init__()
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


class ProductLoader(DataLoader):
    def __init__(self, dataset):
        super(ProductLoader).__init__()
        self.dataset = dataset

    def __iter__(self):
        return self.datas


def get_dataloaders(cfg, tokenizer_fn):
    train_products, val_products = get_product_loaders(cfg, tokenizer_fn)
    id_products = get_identity_loader(cfg, tokenizer_fn)
    return train_products, val_products + [id_products]


def get_identity_loader(cfg, tokenizer_fn):
    ciqual_dict = yaml.safe_load(open(cfg.ciqual_to_name_path))
    label_dict = {CIQUAL_TO_IDX[k]: v for k, v in ciqual_dict.items()}

    tokens, labels = [], []
    for ciqual, name in ciqual_dict.items():
        tokens.append(tokenizer_fn(name))
        labels.append(CIQUAL_TO_IDX[ciqual])

    data = {"tokens": tokens, "label": labels}
    dataset = ProductDataset(data)

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

    df = df[df['name'].notnull()]
    df['label'] = df['ciqual'].apply(lambda x: CIQUAL_TO_IDX.get(x))
    df = df[df['label'].notnull()]
    df = df[df['name'].notnull()]
    df['tokens'] = df['name'].apply(tokenize_fn)

    if cfg.n_folds <= 1:
        train_df, val_df = train_test_split(df, test_size=0.8)
        train_loader = df_to_loader(train_df, shuffle=True, batch_size=cfg.batch_size)
        val_loader = df_to_loader(val_df, shuffle=False, batch_size=cfg.val_batch_size)
        # torch.save([loader], f=train_cache)

        return [train_loader], [val_loader]

    else:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3)
        for fold_idx, (train_index, test_index) in enumerate(skf.split(df, df['lang'])):
            # split the dataframe into training and testing sets
            train_set = df.iloc[train_index]
            test_set = df.iloc[test_index]

        return [loader], []


def df_to_loader(df, shuffle, batch_size):
    data = {"tokens": df["tokens"].tolist(), "label": df["label"].tolist()}
    dataset = ProductDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
