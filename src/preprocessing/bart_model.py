from ctypes import Union
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import device, nn
import torch
import torch.nn.functional as F
from torchvision import models
import pandas as pd
from tqdm import tqdm

from src.preprocessing.training_dataset import BirdsDataset, to_records
from torch.utils.data import DataLoader


class BartModel(nn.Module):
    """ https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/ """

    def __init__(self, all_classes, scored_classes, fit_backbone=False):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = fit_backbone

        self.input_norm = nn.BatchNorm2d(3)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.05),

            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.05)
        )

        self.scored_fc = nn.Linear(512, scored_classes)
        self.all_fc = nn.Linear(512, all_classes)

    def forward(self, inputs):
        x = self.input_norm(inputs)
        x = self.backbone(x)

        all_out = self.all_fc(x)
        scored_out = self.scored_fc(x)

        return torch.sigmoid(all_out), torch.sigmoid(scored_out)


def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }


def fit(
    model, 
    epochs, 
    training, 
    optimizer, 
    criterion, 
    device, output_dir: Path, 
    save_interval=5, 
    validation=None, 
    validation_freq=5,
    scored_weight=1.0,
    all_weight=1.0
):
    model = model.to(device)
    training_history = []
    validation_history = []
    output_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(epochs):
        losses = []
        model.train()
        for spectrogram, target_all, target_scored in tqdm(training):
            spectrogram = spectrogram.to(device)
            target_all = target_all.to(device)
            target_scored = target_scored.to(device)

            optimizer.zero_grad()
            pred_all, pred_scored = model(spectrogram)

            batch_loss = (
                all_weight * criterion(pred_all, target_all.type(torch.float))
                +
                scored_weight * criterion(pred_scored, target_scored.type(torch.float))
            )

            losses.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()

        loss_value = np.mean(losses)
        training_history.append(dict(epoch=epoch, loss=loss_value))
        print(f"Epoch: {epoch:2d} train: loss: {loss_value:.3f}")

        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_value,
            }, checkpoint_dir / f"checkpoint_{epoch}.pt")

        if epoch % validation_freq == 0 and validation is not None:
            model.eval()
            with torch.no_grad():
                model_result = []
                targets = []
                for spectrogram, _, target_scored in tqdm(validation):
                    spectrogram = spectrogram.to(device)
                    target_scored = target_scored.to(device)

                    _, pred_scored = model(spectrogram)
                    model_result.extend(pred_scored.cpu().numpy())
                    targets.extend(target_scored.cpu().numpy())

                result = calculate_metrics(np.array(model_result), np.array(targets))
                result['epoch'] = epoch
                validation_history.append(result)
                
                print(f"Epoch:{epoch:2d} validation: "
                      f"micro f1: {result['micro/f1']:.3f} "
                      f"macro f1: {result['macro/f1']:.3f} "
                      f"samples f1: {result['samples/f1']:.3f}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_value,
    }, output_dir / f"model.pt")


    joblib.dump(training_history, output_dir / 'training_history.joblib')
    joblib.dump(validation_history, output_dir / 'validation_history.joblib')


def run(
    training_csv_path: Path,
    record_cache_path: Path,
    training_data_dir: Path,
    model_output_dir: Path,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    train_val_split: Optional[float] = 0.8,
    dataset_limit: Optional[int] = None,
    device = torch.device("cpu"),
    fit_backbone: bool = False,
    save_interval: int = 5,
    validation_freq: int = 5,
    scored_weight: float = 1.0,
    all_weight: float = 1.0
):
    df = pd.read_csv(training_csv_path)
    df['secondary_labels'] = [eval(r) for r in df.secondary_labels]

    if record_cache_path.exists():
        records, all_encoder, scored_encoder = joblib.load(record_cache_path)
    else:
        outputs = to_records(df)
        record_cache_path.absolute().parent.mkdir(exist_ok=True, parents=True)
        records, all_encoder, scored_encoder = outputs
        joblib.dump(outputs, record_cache_path)

    dataset = BirdsDataset(records, training_data_dir, dataset_limit)
    if train_val_split is None:
        train = DataLoader(dataset, batch_size=batch_size)
        val = None
    else:
        total = len(dataset)
        train_split = int(total * train_val_split)
        val_split = total - train_split
        train, val = torch.utils.data.random_split(dataset, [train_split, val_split], generator=torch.Generator().manual_seed(42))

        train = DataLoader(train, batch_size=batch_size)
        val = DataLoader(val, batch_size=batch_size)

    model = BartModel(len(all_encoder.classes_), len(scored_encoder.classes_), fit_backbone=fit_backbone)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    fit(model, epochs, train, optimizer, criterion, device, 
        output_dir=model_output_dir,
        validation=val, 
        validation_freq=validation_freq, 
        save_interval=save_interval,
        scored_weight=scored_weight,
        all_weight=all_weight
    )


if __name__ == '__main__':
    run(
        Path('data/custom/training.csv'),
        Path('model/outputs.gz'),
        Path('data/custom/train_audio'),
        Path('model/'),
        dataset_limit=1_000,
        save_interval=1
    )

    device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")