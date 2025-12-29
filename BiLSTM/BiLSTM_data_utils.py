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
        try:
            return int(float(s))
        except Exception:
            pass
    try:
        return int(x)
    except Exception:
        raise ValueError(f"[label error] 해석 불가: {x!r}")

class VoltageDataset(Dataset):
    def __init__(self, data_dir, window_size=1000, oversample_B=1):
        self.window_size = int(window_size)
        self.oversample_B = int(oversample_B)
        self.segments, self.targets = [], []
        csv_list = [fn for fn in sorted(os.listdir(data_dir)) if fn.lower().endswith('.csv')]
        total_counter = Counter()

        for fn in csv_list:
            path = os.path.join(data_dir, fn)
            try:
                df = pd.read_csv(path)
                volt = df.iloc[:, 1].astype('float32').values
                labelv = df.iloc[:, 2].values
                depth = df.iloc[:, 3].astype('float32').values

                depth_norm = (depth - np.min(depth)) / (np.ptp(depth) + 1e-8)
                depth_norm *= 0.2

                labels = np.array([_label_to_id(v) for v in labelv], dtype=np.int64)
                N, W = len(volt), self.window_size
                file_counter = Counter()

                for start in range(0, N - W + 1, W):
                    end = start + W
                    seg_v = volt[start:end]
                    seg_d = depth_norm[start:end]
                    win_labs = labels[start:end]
                    lab = Counter(win_labs).most_common(1)[0][0]
                    combined = np.stack([seg_v, seg_d], axis=0)
                    repeat = self.oversample_B if lab == 2 else 1
                    for _ in range(repeat):
                        self.segments.append(combined)
                        self.targets.append(int(lab))
                        file_counter[int(lab)] += 1
                        total_counter[int(lab)] += 1

                print(f"[DEBUG] {fn} 윈도 라벨 분포: {file_counter}")

            except Exception as e:
                print(f"[ERROR] {fn} 처리 실패: {e}")

        print(f"[DEBUG] 전체 라벨 분포: {total_counter}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x = torch.tensor(self.segments[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

def create_loaders(data_dir, batch_size, window_size=1000, oversample_B=1, num_workers=0):
    dataset = VoltageDataset(data_dir, window_size, oversample_B)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader
