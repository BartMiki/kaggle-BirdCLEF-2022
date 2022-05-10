from src.config import BaseConfig
from pathlib import Path
import argparse
from src.utils import path_type

def train():
    pass

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
    print(config)

if __name__ == '__main__':
    main()