from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import pandas as pd
import numpy as np

from typing import List, Union
from attr import field
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset
from dataclasses import dataclass

import torch
from torchaudio.transforms import Spectrogram 
import torchaudio
from tqdm import tqdm
from src.preprocessing.transform import Seconds, TransformF, Slice, ZeroPad


@dataclass
class Record:
    filename: str
    end: Seconds
    rating: float
    is_scored: bool
    all_outputs: torch.Tensor
    scored_outputs: torch.Tensor
    duration: Seconds = Seconds(5)


@lru_cache(64)
def load_audio(path: Path):
    return torchaudio.load(path)


class BirdsDataset(Dataset):
    def __init__(self, records: List[Record], root_dir: Path, limit=None):
        self.records = records
        self.root_dir = root_dir
        self.limit = limit

    def __getitem__(self, idx):
        record = self.records[idx]
        audio, sampling_rate = load_audio(self.root_dir / record.filename)

        audio = Slice(record.end, record.duration, sampling_rate)(audio)
        audio = ZeroPad(record.duration.as_frames(sampling_rate))(audio)
        audio = Spectrogram()(audio)
        audio = torch.cat([audio for _ in range(3)])

        return audio, record.all_outputs, record.scored_outputs

    def __len__(self):
        return len(self.records) if self.limit is None else self.limit


def to_records(df: pd.DataFrame):
    all_encoder = LabelEncoder()
    scored_encoder = LabelEncoder()

    all_encoder.fit(df.primary_label)
    scored_encoder.fit(df[df.is_scored].primary_label)

    def parse_record(row):
        scored_label = torch.zeros(len(scored_encoder.classes_))
        if row.is_scored:
            scored_label[scored_encoder.transform([row.primary_label])] = 1

        all_label = torch.zeros(len(all_encoder.classes_))
        all_label[all_encoder.transform([row.primary_label])] = 1
        all_label[all_encoder.transform(row.secondary_labels)] = 0.75

        return Record(row.filename, Seconds(row.end), row.rating, row.is_scored, all_label, scored_label)

    records = [parse_record(row) for _, row in tqdm(df.iterrows(), total=len(df))]
    return records, all_encoder, scored_encoder


def main():
    df = pd.read_csv('data/custom/training.csv')
    df['secondary_labels'] = [eval(r) for r in df.secondary_labels]
    
    records, _, _ = to_records(df)
    ds = BirdsDataset(records, Path('data/custom/train_audio'))
    for x in tqdm(ds):
        pass
        # print ((x['target_all'] > 0).sum())


if __name__ == "__main__":
    main()