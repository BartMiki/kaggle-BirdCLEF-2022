from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from typing import Callable, List, Union
from matplotlib.pyplot import axes
import pandas as pd
import torch
import torchaudio
from joblib import Parallel, delayed
import os
from pathlib import Path
from copy import deepcopy
from itertools import chain
import math

from tqdm import tqdm
from src.preprocessing.transform import Preprocessing, Rechannel, Slice, Slice, TransformF, Seconds

pool = Parallel(-1)


@dataclass
class TrainingPreprocessing:
    df: pd.DataFrame 
    input_dir: Path
    output_dir: Path
    scored_birds: List[str]
    preprocess: Preprocessing = None


@delayed
def process(params: TrainingPreprocessing, name: str):
    info = vars(torchaudio.info(params.input_dir / name))
    file_suffix = '.flac' # '.ogg' #'.pt.gz'

    if params.preprocess:
        audio, _ = torchaudio.load(params.input_dir / name)
        audio = params.preprocess(audio)

        info['num_frames'] = audio.shape[1]
        info['sample_rate'] = params.preprocess.resample.new_freq

        path = (params.output_dir / 'train_audio' / name).with_suffix(file_suffix)
        torchaudio.save(path, audio, info['sample_rate'])
        # with gzip.open(path, 'wb') as fh:
        #     torch.save(audio, fh)

    num_frames, sample_rate = info['num_frames'], info['sample_rate']
    info['seconds'] = num_frames / sample_rate
    info['filename'] = name.replace('.ogg', file_suffix)
    info['id'] = name

    def process_window(start: int) -> dict:
        split = deepcopy(info)
        split['end'] = (start+1) * 5
        return split

    num_windows = math.ceil(info['seconds'] / 5)
    splits = [process_window(idx) for idx in range(num_windows)]
    return splits


def generate_training(params: TrainingPreprocessing):
    df = params.df.copy()

    params.output_dir.mkdir(exist_ok=True, parents=True)
    subpaths = {Path(p).parent for p in df.filename}
    for subpath in subpaths:
        (params.output_dir / 'train_audio' / subpath).mkdir(exist_ok=True, parents=True)

    df['id'] = df.filename
    values = chain.from_iterable(pool(process(params, file) for file in tqdm(df.filename)))
    info_df = pd.DataFrame.from_records(values)
    df = df.drop('filename', axis=1)

    df['is_scored'] = df['primary_label'].isin(params.scored_birds)

    info_df = pd.merge(info_df, df, on='id')
    info_df = info_df[[
        'sample_rate',
        'num_frames',
        'num_channels',
        'filename',
        'seconds',
        'end',
        'primary_label',
        'secondary_labels',
        'type',
        'rating',
        'is_scored']]

    info_df.to_csv(params.output_dir / 'training.csv', index=False)
    # with gzip.open(params.output_dir / 'training.csv.gz', 'wt') as fh:
    #     info_df.to_csv(fh, index=False)


def main():
    root = Path(__file__).parent.parent.parent.absolute()
    df = pd.read_csv(root / 'data' / 'birds' / 'train_metadata.csv')
    with open(root / 'data' / 'birds' / 'scored_birds.json') as fh:
        scored_birds = json.load(fh)


    processor = TrainingPreprocessing(df, 
        root / 'data' / 'birds' / 'train_audio',
        root / 'data' / 'custom',
        scored_birds,
        Preprocessing(1, 32000, 8000))

    generate_training(processor)


if __name__ == "__main__":
    main()