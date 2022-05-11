from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.config import BasePreprocessingConfig


def read_train_metadata(
    preprocessing_config: BasePreprocessingConfig,
    metadata_filepath=Path(__file__).parents[2] / 'data/modified_data/training.csv'):

    df = pd.read_csv(metadata_filepath)

    # Filter samples
    if preprocessing_config.min_rating:
        df = df[df.rating > 3.0]
    if preprocessing_config.scored_only:
        df = df[df.is_scored]

    # Encode labels
    class_le = LabelEncoder()
    class_le.fit(df.primary_label)
    df.primary_label = class_le.transform(df.primary_label)

    return df.reset_index(drop=True), class_le
