import torch

from harrygobert.model import OFFClassificationModel, get_tokenizer, get_tokenize_fn
from harrygobert.util import parse_args

TEST_STRING = "test"


def test_model():
    import sys
    sys.argv = ['']
    del sys
    cfg = parse_args()
    cfg.model_name = "distilbert-base-multilingual-cased"

    # Get model
    model = OFFClassificationModel(cfg)

    # Get tokenizer
    tokenize_fn = get_tokenize_fn(cfg, get_tokenizer(cfg))

    # Run inference
    out_1 = model(**tokenize_fn(TEST_STRING))

    # Check output shape and sum to 1
    assert out_1.shape[-1] == cfg.n_classes
    assert torch.sum(out_1).isclose(torch.tensor(1.))

    # Check that outputs are different due to gradient update
    out_2 = model(**tokenize_fn(TEST_STRING))
    assert not torch.all(out_1 == out_2)

    # Check that outputs are identical in inference mode
    model.eval()
    out_3 = model(**tokenize_fn(TEST_STRING))
    out_4 = model(**tokenize_fn(TEST_STRING))
    assert torch.all(out_3 == out_4)
