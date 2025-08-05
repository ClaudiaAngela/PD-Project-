# src/dataloader.py

import os
import torch
import torchaudio
from torch.utils.data import Dataset

class NeuroVozSpeechCommands(Dataset):
    def __init__(self, dataframe, transform=None, sample_rate=16000, max_tries=10, max_len=128):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_tries = max_tries
        self.max_len = max_len  # numărul fix de coloane (timp) la care vrei să aduci toate exemplele

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tries = 0
        cur_idx = idx
        while tries < self.max_tries:
            audio_path = self.df.iloc[cur_idx]["audio_path"]
            label = self.df.iloc[cur_idx]["label"]
            try:
                waveform, sr = torchaudio.load(audio_path)
                if sr != self.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                if self.transform:
                    features = self.transform(waveform)
                else:
                    features = waveform
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                # --- Padding sau crop ---
                T = features.shape[-1]
                if T < self.max_len:
                    pad = self.max_len - T
                    features = torch.nn.functional.pad(features, (0, pad))
                elif T > self.max_len:
                    features = features[..., :self.max_len]
                # ---
                return features, label
            except Exception as e:
                print(f"File error: {audio_path} -> {e}")
                cur_idx = (cur_idx + 1) % len(self.df)
                tries += 1
        raise RuntimeError(f"Too many corrupt files at idx={idx}")