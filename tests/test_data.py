import os

import pytest

from harrygobert.data import get_dataloaders, get_product_df
from harrygobert.model import get_tokenizer, get_tokenize_fn
from harrygobert.util import parse_args


# TODO make local CSV a bit bigger to allow testing n_folds=1 (i.e. standard 80-20 split)
@pytest.mark.parametrize("n_folds", [0, 2])
def test_data(n_folds):
    import sys
    sys.argv = ['']
    del sys
    cfg = parse_args()

    # Override default args
    cfg.debug = True
    cfg.csv_path = os.path.join(os.path.dirname(__file__), "test.csv")
    cfg.use_wandb = False
    cfg.n_folds = n_folds

    # Smoketest data pipeline
    tokenizer = get_tokenizer(cfg)
    tokenize_fn = get_tokenize_fn(cfg, tokenizer)
    df = get_product_df(cfg, tokenize_fn)
    get_dataloaders(cfg, tokenize_fn, df, 0)
