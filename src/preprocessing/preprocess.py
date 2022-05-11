import argparse
from pathlib import Path

from src.config import BaseConfig
from src.preprocessing.utils import read_train_metadata
from src.utils import path_type

def preprocess(config: BaseConfig):
    df = read_train_metadata(config.preprocessing)

def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        required=False,
        default=Path(__file__).parents[2] / 'configs/config.yaml',
        type=path_type,
        help="Path to yaml file with model config.",
    )
    return parser.parse_args()

def main():
    args = run_argparse()
    config = BaseConfig.from_file(args.config_path)
    
    preprocess(config)

if __name__ == '__main__':
    main()
