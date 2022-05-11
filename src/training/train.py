import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from src.config import BaseConfig
from src.preprocessing.preprocess import preprocess
from src.training.model import BaselineModel
from src.utils import path_type
from tqdm import tqdm


def train(config: BaseConfig):
    dl, class_le = preprocess(config)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device selected: {device}')
    model = BaselineModel(config.training.classes)
    model = model.to(device)

    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum=0.9)

    for epoch in tqdm(range(config.training.epochs)):
        running_loss = 0.0
        for i, data in enumerate(dl):
            inputs, labels = data
            inputs.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
        running_loss = 0.0

    return model, class_le


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
    
    train(config)


if __name__ == '__main__':
    main()
