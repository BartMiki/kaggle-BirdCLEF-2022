from dataclasses import dataclass

import dacite
import yaml


@dataclass
class BaseTrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float

    def __post_init__(self):
        pass

@dataclass
class BaseConfig:
    training: BaseTrainingConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BaseConfig":
        return dacite.from_dict(data_class=cls, data=config_dict)

    @classmethod
    def from_file(cls, config_filepath: str) -> "BaseConfig":
        with open(config_filepath, "r") as f:
            data = yaml.load(f, yaml.UnsafeLoader)
        return cls.from_dict(data)