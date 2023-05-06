import os

from harrygobert.data import get_dataloaders
from harrygobert.model import get_tokenizer, get_tokenize_fn
from harrygobert.util import parse_args


def test_data():
    import sys
    sys.argv = ['']
    del sys

    # Smoketest data pipeline
    cfg = parse_args()
    cfg.debug = True
    cfg.csv_path = os.path.join(os.path.dirname(__file__), "test.csv")
    cfg.use_wandb = False
    tokenizer = get_tokenizer(cfg)
    tokenize_fn = get_tokenize_fn(cfg, tokenizer)
    get_dataloaders(cfg, tokenize_fn)
