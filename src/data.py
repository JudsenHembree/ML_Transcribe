from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torchaudio
import pandas as pd
import numpy as np

class Data(Dataset):
    def __init__(self, csv):
        self.df = pd.read_csv(csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.df.loc[idx, 'filename']
        class_id = self.df.loc[idx, 'class_id']

        audio, _ = torchaudio.load(audio_file)
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=1024, n_mels=64)(audio)
        top_db = 80

        spec_db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)

        return spec_db, class_id

