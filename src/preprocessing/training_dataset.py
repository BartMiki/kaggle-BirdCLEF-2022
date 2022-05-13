from __future__ import annotations
from cProfile import label
from functools import lru_cache
import gzip
from logging import root
from operator import attrgetter
from pathlib import Path
from matplotlib.transforms import Transform
import pandas as pd

from typing import List, Union
from attr import field
from torch.utils.data import Dataset
from dataclasses import dataclass
import torch
import math

import torchaudio
from tqdm import tqdm
from src.preprocessing.transform import Seconds, TransformF, Slice, ZeroPad


@dataclass
class Record:
    filename: str
    end: Seconds
    rating: float
    is_scored: bool
    label: Union[int, str, None]
    secondary_label: List[Union[int, str]] = field(factory=list)
    duration: Seconds = Seconds(5)


@lru_cache(12000)
def load_audio(path: Path):
    return torchaudio.load(path)


class BirdsDataset(Dataset):
    def __init__(self, records: List[Record], root_dir: Path):
        self.records = records
        self.root_dir = root_dir

    def __getitem__(self, idx):
        record = self.records[idx]
        audio, sampling_rate = load_audio(self.root_dir / record.filename)

        audio = Slice(record.end, record.duration, sampling_rate)(audio)
        audio = ZeroPad(record.duration.as_frames(sampling_rate))(audio)

        return audio, record.label

    def __len__(self):
        return len(self.records)


def main():
    df = pd.read_csv('data/custom/training.csv')
    
    def parse_record(row):
        return Record(row.filename, Seconds(row.end), row.rating, row.is_scored, row.primary_label, row.secondary_labels)

    records = [parse_record(row) for _, row in df.iterrows()]
    ds = BirdsDataset(records, Path('data/custom/train_audio'))
    for x in tqdm(ds):
        print (x[0].shape, x[1])


if __name__ == "__main__":
    main()