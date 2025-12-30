# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter

LABEL_MAP = {'Hold': 0, 'Go': 1, 'Back': 2}

def _label_to_id(x):
    if isinstance(x, str):
        s = x.strip().upper()
        for k, v in LABEL_MAP.items():
            if s == k.upper():
                return v
    return int(x)

class VoltageDataset(Dataset):
    def __init__(self, data_dir, window_size=1000):
        self.window_size = window_size
        self.segments, self.targets = [], []

        csv_list = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

        for fn in csv_list:
            df = pd.read_csv(os.path.join(data_dir, fn))

            volt = df.iloc[:, 1].astype(np.float32).values
            labels = df.iloc[:, 2].values
            labels = np.array([_label_to_id(v) for v in labels])

            for i in range(0, len(volt) - window_size + 1, window_size):
                v = volt[i:i + window_size]
                y = Counter(labels[i:i + window_size]).most_common(1)[0][0]

                x = v[np.newaxis, :]   # (1, 1000)
                self.segments.append(x)
                self.targets.append(y)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.segments[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )

def create_loaders(data_dir, batch_size, window_size=1000):
    dataset = VoltageDataset(data_dir, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
