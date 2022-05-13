import torchaudio
from torchaudio.transforms import Spectrogram, Resample
import matplotlib.pyplot as plt
import torch

# https://enzokro.dev/spectrogram_normalizations/2020/09/10/Normalizing-spectrograms-for-deep-learning.html

audio, sr = torchaudio.load('data/birds/train_audio/afrsil1/XC125458.ogg')
audio = audio[:, sr*5:sr*10]

print(audio.max(), audio.min(), audio.mean(), audio.std())
print(audio.shape)
print(sr)

print(audio.shape[1] / sr)

audio = Resample(sr, 8000)(audio)
spectrogram = Spectrogram()(audio)[0]

plt.imshow(torch.log(spectrogram))
plt.title(f"Spectrogram of size: {spectrogram.shape}")
plt.savefig('spectrogram.png')