# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
import re

IN_DIR = Path(r"C:\Users\drkjh\Desktop\진하\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\1_rawdata")
OUT_DIR = Path(r"C:\Users\drkjh\Desktop\진하\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\2_")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Searching in: {IN_DIR.resolve()}")
files_lower = list(IN_DIR.glob("*.csv"))
files_upper = list(IN_DIR.glob("*.CSV"))
all_files = files_lower + files_upper
print(f"[INFO] Found {len(all_files)} CSV files")

for csv_path in all_files:
    try:
        print(f"[READ] {csv_path.name}")
        df = pd.read_csv(csv_path, header=None, names=["Time", "V"])
        fname = csv_path.stem
        parts = fname.split("_")

        if len(parts) < 5:
            print(f"[SKIP] Invalid filename format: {csv_path.name}")
            continue

        label_num = parts[-2]
        label = re.sub(r"\d+", "", label_num)

        try:
            depth = float(parts[-1])
        except ValueError:
            depth = None

        # === 수정 부분 ===
        df["Label"] = [label] * len(df)
        df["Depth"] = [None] * (len(df) - 1) + [depth]  # 마지막 행에만 Depth 입력

        save_path = OUT_DIR / csv_path.name
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"[OK] {csv_path.name} | Label={label}, Depth={depth}")

    except Exception as e:
        print(f"[ERROR] {csv_path.name}: {e}")

print("\n[INFO] Processing complete.")
