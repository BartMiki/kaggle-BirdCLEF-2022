import argparse
import torch
from pathlib import Path

from src.config import BaseConfig
from src.preprocessing.utils import read_train_metadata
from src.utils import path_type
from src.preprocessing.dataset import BirdDataset


def preprocess(config: BaseConfig):
    df, class_le = read_train_metadata(config.preprocessing)

    ds = BirdDataset(df)
    dl = torch.utils.data.DataLoader(ds, batch_size=config.training.batch_size, shuffle=True)

    return dl, class_le


def rechannel(recording, channels):
    current_channels = recording.shape[0]
    if current_channels == channels:
        return recording
    
    if channels == 1:
        return recording[:1, :]
    
    if channels == 2 and current_channels == 1:
        return torch.cat([recording, recording])
    
    if channels == 2 and current_channels == 3:
        return recording[:2, :]
    
    if channels == 3 and current_channels == 1:
        return torch.cat([recording, recording, recording])
    
    if channels == 3 and current_channels == 2:
        return torch.cat([
            recording[0:1, :],
            recording[1:2, :],
            recording.mean(axis=0)[None, :]
        ])
    
    raise ValueError(f"Unupported target channels: {channels} and audio channels {current_channels}")


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
