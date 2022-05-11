import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from src.config import BaseConfig
from src.preprocessing.preprocess import rechannel
from src.training.train import train
from src.utils import path_type
from torchaudio.transforms import Resample
from tqdm.notebook import tqdm


def inference_with_train(config: BaseConfig):
    model, class_le = train(config)

    model.eval()
    root_dir = Path('/kaggle/input/birdclef-2022/')
    test_dir = root_dir / 'test_soundscapes'
    files = list(test_dir.rglob('*.ogg'))
    with open(root_dir / 'scored_birds.json') as sbfile:
        scored_birds = json.load(sbfile)

    pred = {'row_id': [], 'target': []}
    for path in tqdm(files):
        afile = path.stem
        audio_file, rate = torchaudio.load(path)
        resampler = Resample(rate, 1000)
        # Transform
        audio_file = rechannel(audio_file, 2)
        audio_file = resampler(audio_file)
        # Padding
        audio_file_temp = torch.zeros((2,60*1000))
        # lewo = dopasowujemy rozmiar do pojemności temp 60*1000, jak mniejsze to 0
        # prawo = wybieramy max 1 min (60*1000)
        audio_file_temp[:,:min(audio_file.shape[1], audio_file_temp.shape[1])] = audio_file[:,:audio_file_temp.shape[1]]
        audio_file = audio_file_temp
        
        for second in range(0, 60, 5):
            audio_file_5 = audio_file[:,second*1000:(second+5)*1000]
            audio_file_5 = torch.reshape(audio_file_5, (10000,))
            pred_bird_idx = model(audio_file_5)
            pred_bird = class_le.inverse_transform([torch.argmax(pred_bird_idx)])[0]
            for bird in scored_birds:
                pred['row_id'].append(afile+'_'+bird+'_'+str(second+5))
                pred['target'].append(bird == pred_bird)

    results = pd.DataFrame(pred, columns = list(pred.keys()))
    results.to_csv("submission.csv", index=False)


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
    
    inference_with_train(config)


if __name__ == '__main__':
    main()
