from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BirdDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        audio_filepath = Path(__file__).parents[2] / 'data/modified_data/train_audio' / self.df.loc[idx, 'raw_filename']
        audio = torch.load(audio_filepath)
        
        # Preprocess audio
        # - model requires equal size input
        audio_numpy = audio.cpu().detach().numpy()
        if audio_numpy.shape[1]>5000:
            audio_numpy = audio_numpy[:,:5000]
        else:
            diff = 5000 - audio_numpy.shape[1]
            diff2 = diff//2
            audio_numpy = np.pad(audio_numpy, ((0,0), (diff2,diff-diff2)), mode='constant')

        audio_numpy = audio_numpy.reshape((10000,))
        
        audio = torch.from_numpy(audio_numpy)
        
        class_id = self.df.loc[idx, 'primary_label']
        return audio, class_id
