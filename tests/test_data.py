from harrygobert.data import get_dataloaders
from harrygobert.util import parse_args


def test_data():
    import sys
    sys.argv = ['']
    del sys

    # Smoketest data pipeline
    cfg = parse_args()
    cfg.debug = True
    get_dataloaders(cfg)
