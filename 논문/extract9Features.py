# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

# ===== 경로 =====
IN_DIR = Path(
    r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\논문\Dataset"
)
OUT_CSV = Path(
    r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\논문\ML_dataset.csv"
)

# ===== 9가지 특징 =====
def extract_9_features(x: np.ndarray) -> dict:
    q25, q75 = np.percentile(x, [25, 75])
    s = pd.Series(x)

    return {
        "Max": float(np.max(x)),
        "Min": float(np.min(x)),
        "Mean": float(np.mean(x)),
        "Std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "Median": float(np.median(x)),
        "IQR": float(q75 - q25),
        "RMS": float(np.sqrt(np.mean(x ** 2))),
        "Skewness": float(s.skew()) if len(x) > 2 else 0.0,
        "Kurtosis": float(s.kurt()) if len(x) > 3 else 0.0,
    }

# ===== 파일명 파싱 =====
def parse_filename(fname: str):
    """
    (실험제목)_(실험시간)_(index)_(Command+번호)_(depth).csv
    """
    parts = fname.replace(".csv", "").split("_")
    time_tag = parts[1]
    index = int(parts[2])

    # Command 추출 (Go / Hold / Back)
    m = re.search(r"(Go|Hold|Back)", parts[3])
    command = m.group(1) if m else "Unknown"

    return time_tag, index, command

# ===== 1️⃣ 실험시간 기준 그룹핑 =====
groups = defaultdict(list)

csv_files = list(IN_DIR.glob("*.csv"))
print(f"[INFO] Found {len(csv_files)} CSV files")

for csv_path in csv_files:
    try:
        time_tag, index, command = parse_filename(csv_path.name)
        groups[time_tag].append((index, csv_path, command))
    except Exception as e:
        print(f"[SKIP] filename parse error: {csv_path.name}")

# ===== 2️⃣ 특징 계산 (실험시간 → 인덱스 순) =====
rows = []

for time_tag in sorted(groups.keys()):
    file_list = groups[time_tag]
    file_list.sort(key=lambda x: x[0])  # 인덱스 순서

    print(f"[INFO] Processing {time_tag} ({len(file_list)} files)")

    for index, csv_path, command in file_list:
        try:
            df = pd.read_csv(csv_path)

            # B열 전압
            v = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().to_numpy()
            if len(v) == 0:
                print(f"[SKIP] {csv_path.name} | empty signal")
                continue

            feats = extract_9_features(v)

            row = {
                "Filename": csv_path.name,
                **feats,
                "Label": command
            }
            rows.append(row)

        except Exception as e:
            print(f"[ERROR] {csv_path.name}: {e}")

# ===== 3️⃣ 저장 =====
dataset = pd.DataFrame(rows)
dataset.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n✅ ML dataset saved")
print(f"   path : {OUT_CSV}")
print(f"   rows : {len(dataset)}")
