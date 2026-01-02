# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# ===== 경로 설정 =====
IN_DIR = Path(
    r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\4_removedInvalidLabel"
)
OUT_CSV = Path(
    r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\ML_dataset.csv"
)

# ===== 9가지 특징 함수 =====
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

# ===== Label 결정 (파일 단위) =====
def decide_label(lbl_series: pd.Series) -> str:
    lbl_series = lbl_series.astype(str).str.strip()
    cnt = Counter(lbl_series.tolist())
    return cnt.most_common(1)[0][0]

# ===== 파일명 파싱 =====
def parse_filename(fname: str):
    """
    Hybrid(Average)_15-24_1000_Go489_443.0.csv
    → time_tag=15-24, index=1000
    """
    parts = fname.split("_")
    time_tag = parts[1]
    index = int(parts[2])
    return time_tag, index

# ===== 1️⃣ 실험시간 기준 그룹핑 =====
groups = defaultdict(list)

csv_files = list(IN_DIR.glob("*.csv"))
print(f"[INFO] Found {len(csv_files)} CSV files")

for csv_path in csv_files:
    try:
        time_tag, index = parse_filename(csv_path.name)
        groups[time_tag].append((index, csv_path))
    except Exception as e:
        print(f"[SKIP] filename parse error: {csv_path.name}")

# ===== 2️⃣ 실험시간 → 인덱스 순서대로 특징 계산 =====
rows = []

for time_tag in sorted(groups.keys()):
    file_list = groups[time_tag]

    # 인덱스 순 정렬
    file_list.sort(key=lambda x: x[0])

    print(f"[INFO] Processing experiment {time_tag} ({len(file_list)} files)")

    for index, csv_path in file_list:
        try:
            df = pd.read_csv(csv_path)

            # 전압 신호 (B열)
            v = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().to_numpy()
            if len(v) == 0:
                print(f"[SKIP] {csv_path.name} | no valid signal")
                continue

            feats = extract_9_features(v)
            label = decide_label(df["Label"])

            row = {
                "Filename": csv_path.name,
                "TimeTag": time_tag,
                "Index": index,
                **feats,
                "Label": label
            }
            rows.append(row)

        except Exception as e:
            print(f"[ERROR] {csv_path.name}: {e}")

# ===== 3️⃣ 최종 저장 =====
dataset = pd.DataFrame(rows)

dataset = (
    dataset
    .sort_values(["TimeTag", "Index"])   # ⭐ 핵심
    .drop(columns=["TimeTag", "Index"])
    .reset_index(drop=True)
)

dataset.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n✅ ML dataset saved (experiment → index order): {OUT_CSV}")
print(f"   total rows: {len(dataset)}")
