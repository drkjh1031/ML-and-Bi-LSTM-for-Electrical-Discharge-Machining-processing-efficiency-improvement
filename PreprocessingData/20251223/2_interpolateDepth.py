# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ===== 경로 =====
IN_DIR = Path(r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\2_DepthBeforeProcessing")
OUT_DIR = Path(r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\3_DepthInterpolated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 실험시간 기준 그룹핑 =====
groups = defaultdict(list)

for f in IN_DIR.glob("*.csv"):
    parts = f.stem.split("_")
    if len(parts) < 5:
        continue

    time_tag = parts[1]          # 실험시간
    idx = int(parts[2])          # 인덱스
    depth = float(parts[-1])     # 가공깊이

    groups[time_tag].append((idx, depth, f))

# ===== 그룹별 처리 =====
for time_tag, file_list in groups.items():
    # 인덱스 순 정렬
    file_list.sort(key=lambda x: x[0])

    prev_depth = None

    for idx, curr_depth, csv_path in file_list:
        df = pd.read_csv(csv_path)

        n = len(df)  # 보통 1000

        # 첫 파일: 기준 없음 → 전부 현재 깊이
        if prev_depth is None:
            df["Depth"] = np.full(n, curr_depth)
        else:
            # 이전 깊이 → 현재 깊이 선형 보간
            df["Depth"] = np.linspace(prev_depth, curr_depth, n)

        prev_depth = curr_depth

        # 저장
        out_path = OUT_DIR / csv_path.name
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"[OK] {time_tag} | idx={idx} | {prev_depth:.3f}")

print("\n[INFO] Depth interpolation complete.")
