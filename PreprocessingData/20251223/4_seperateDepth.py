# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os, glob
from pathlib import Path

# ===== 경로 =====
input_folder = r"C:\Users\drkjh\Desktop\FinalReport\Dataset\4expandeddata"
output_folder = r"C:\Users\drkjh\Desktop\FinalReport\Dataset\5dataset"
Path(output_folder).mkdir(parents=True, exist_ok=True)

def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949", low_memory=False)

def find_runs(mask: pd.Series):
    """
    불리언 mask에서 True가 연속된 (start, end) 구간 리스트 반환
    end는 exclusive
    """
    runs = []
    n = len(mask)
    start = None
    for i in range(n):
        if mask.iloc[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i))
                start = None
    if start is not None:
        runs.append((start, n))
    return runs

# ===== 처리 =====
file_list = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
for file_path in file_list:
    df = read_csv_safe(file_path)
    if df.shape[1] < 4:
        print(f"[!] No 4th column → skip: {os.path.basename(file_path)}")
        continue

    depth = pd.to_numeric(df.iloc[:, 3], errors="coerce")
    if depth.notna().sum() == 0:
        print(f"[!] No data → skip: {os.path.basename(file_path)}")
        continue

    base = os.path.splitext(os.path.basename(file_path))[0]

    # 전체 구간 범위 계산
    max_val = np.nanmax(depth.values)
    upper_bound = int(((max_val // 100) + 1) * 100) if np.isfinite(max_val) else 100

    # 0~upper_bound까지 100단위 반복
    for start in range(0, upper_bound, 100):
        end = start + 100

        # 0-100 구간은 음수도 포함
        if start == 0:
            mask = (depth < end)
        else:
            mask = (depth >= start) & (depth < end)

        runs = find_runs(mask)

        part_idx = 1
        for s, e in runs:
            out_df = df.iloc[s:e].copy()
            out_name = f"{base}_{start}-{end}_p{part_idx:02d}.csv"
            out_path = os.path.join(output_folder, out_name)
            out_df.to_csv(out_path, index=False)
            print(f"[+] saved: {out_name}")
            part_idx += 1
