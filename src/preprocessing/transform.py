from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Callable, Union
from matplotlib.transforms import Transform
import torch
from torchaudio.transforms import Resample


TransformF = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Seconds:
    value: Union[int, float]

    def as_frames(self, sampling_rate: int) -> int:
        return math.floor(self.value * sampling_rate)

    def __add__(self, other) -> Seconds:
        lhs = other.value if isinstance(other, Seconds) else other
        return Seconds(self.value + lhs)

    def __sub__(self, other) -> Seconds:
        lhs = other.value if isinstance(other, Seconds) else other
        return Seconds(self.value - lhs)


@dataclass
class Rechannel:
    target_channels: int

    def __call__(self, recording: torch.Tensor) -> torch.Tensor:
        input_channels, _ = recording.shape
        if input_channels == self.target_channels:
            return recording
        
        if self.target_channels == 1:
            return recording.mean(0)[None, ...]

        if self.target_channels == 2 and input_channels == 1:
            return torch.cat([recording, recording])
    
        if self.target_channels == 2 and input_channels == 3:
            return recording[:2, :]
    
        if self.target_channels == 3 and input_channels == 1:
            return torch.cat([recording, recording, recording])
        
        if self.target_channels == 3 and input_channels == 2:
            return torch.cat([
                recording[0:1, :],
                recording[1:2, :],
                recording.mean(axis=0)[None, ...]
            ])
        
        raise ValueError(f"Unsupported target channels: {self.target_channels} and audio channels {input_channels}")


class Preprocessing:
    def __init__(self, channels: int, input_freq: int, target_freq: int):
        self.rechannel = Rechannel(channels)
        self.resample = Resample(input_freq, target_freq)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.resample(self.rechannel(tensor))


@dataclass
class ZeroPad:
    target_len: int

    def __call__(self, value: torch.Tensor, ) -> torch.Tensor:
        channels, input_len = value.shape
        if (input_len >= self.target_len):
            return value[:, :self.target_len]
        else:
            result = torch.zeros((channels, self.target_len))
            result[:, :input_len] = value
            return result


@dataclass
class Slice:
    end: Seconds
    duration: Seconds
    frequency: int

    @property
    def start(self) -> Seconds:
        return self.end - self.duration

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        start_frame =  self.start.as_frames(self.frequency)
        end_frame = self.end.as_frames(self.frequency)
        return tensor[:,start_frame:end_frame]


def test_rechannel():
    for in_c, out_c in product(range(1, 4), repeat=2):
        out = Rechannel(out_c)(torch.randn(in_c, 50))
        assert out.shape[0] == out_c