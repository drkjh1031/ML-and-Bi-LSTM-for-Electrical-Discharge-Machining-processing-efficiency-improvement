# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

LABEL_MAP = {"Hold": 0, "Go": 1, "Back": 2}
MAX_DEPTH = 1500.0

def parse_filename(fn):
    """
    예:
    Hybrid(Average)_01-37_65_Go10_0.0.csv
    """
    name = fn.replace(".csv", "")
    parts = name.split("_")

    command_part = parts[3].upper()   # Go10
    depth = float(parts[4])           # 0.0

    if "GO" in command_part:
        label = LABEL_MAP["Go"]
    elif "HOLD" in command_part:
        label = LABEL_MAP["Hold"]
    elif "BACK" in command_part:
        label = LABEL_MAP["Back"]
    else:
        raise ValueError(f"Unknown command in filename: {fn}")

    return label, depth


class VoltageDataset(Dataset):
    def __init__(self, data_dir, window_size=1000):
        self.voltages = []
        self.depths = []
        self.targets = []

        csv_list = sorted(
            fn for fn in os.listdir(data_dir)
            if fn.lower().endswith(".csv")
        )

        for fn in csv_list:
            try:
                label, depth = parse_filename(fn)
                depth_norm = np.clip(depth / MAX_DEPTH, 0.0, 1.0)

                df = pd.read_csv(os.path.join(data_dir, fn), header=None)
                voltage = df.iloc[:window_size, 1].astype("float32").values

                if len(voltage) != window_size:
                    continue

                self.voltages.append(voltage)
                self.depths.append(depth_norm)
                self.targets.append(label)

            except Exception as e:
                print(f"[SKIP] {fn} | {e}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        v = torch.tensor(self.voltages[idx], dtype=torch.float32)  # [1000]
        d = torch.tensor([self.depths[idx]], dtype=torch.float32) # [1]
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return v, d, y


def create_loaders(data_dir, batch_size, window_size=1000, num_workers=0):
    dataset = VoltageDataset(data_dir, window_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
