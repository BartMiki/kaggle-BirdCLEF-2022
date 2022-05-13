from pathlib import Path
import torchaudio
from tqdm import tqdm


def test(directory, ext):
    paths = list(Path(directory).rglob(f'*{ext}'))
    for path in tqdm(paths):
        audio, _ = torchaudio.load(path)
        del audio


print("Birds 32kHz .ogg", flush=True)
test('data/birds', '.ogg')

print("Custom 8kHz .ogg", flush=True)
test('data/custom', '.ogg')

print("Custom 8kHz .flac", flush=True)
test('data/custom', '.flac')