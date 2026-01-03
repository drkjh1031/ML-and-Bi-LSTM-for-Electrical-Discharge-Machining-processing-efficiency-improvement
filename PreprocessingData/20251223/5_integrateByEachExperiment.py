# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
from collections import defaultdict

# ================= 경로 설정 =================
IN_DIR = Path(
    r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement"
    r"\PreprocessingData\20251223\4_removedInvalidLabel"
)
OUT_DIR = Path(
    r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement"
    r"\PreprocessingData\20251223\5_BiLSTM_dataset"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= 그룹 저장 =================
groups = defaultdict(list)

# ================= 파일 분류 =================
for p in IN_DIR.glob("*.csv"):
    parts = p.stem.split("_")

    # 파일명 형식 검사
    if len(parts) != 5:
        print(f"[SKIP] unexpected filename: {p.name}")
        continue

    try:
        experiment_time = parts[1]          # 예: 15-24
        index = int(parts[2])               # 예: 1647
        depth = int(float(parts[4]))        # 예: 268.0 → 268
    except Exception as e:
        print(f"[SKIP] parse error: {p.name} ({e})")
        continue

    # ===== 깊이 200 단위 bin =====
    depth_start = (depth // 200) * 200
    depth_end = depth_start + 200
    depth_bin = f"{depth_start}-{depth_end}"

    # 그룹에 추가
    groups[(experiment_time, depth_bin)].append((index, p))

# ================= 병합 =================
for (experiment_time, depth_bin), items in groups.items():

    # 인덱스 기준 정렬
    items.sort(key=lambda x: x[0])

    dfs = []

    for _, p in items:
        df = pd.read_csv(p)

        # ===== 열 순서 강제 재정렬 =====
        expected_cols = ["Time", "V", "Depth", "Label"]

        if set(expected_cols).issubset(df.columns):
            df = df[expected_cols]
        else:
            print(f"[WARN] column mismatch: {p.name}")
            print(f"       found columns: {df.columns.tolist()}")
            continue

        dfs.append(df)

    if not dfs:
        print(f"[SKIP] no valid CSVs for {experiment_time}_depth{depth_bin}")
        continue

    merged = pd.concat(dfs, ignore_index=True)

    # 저장
    out_name = f"{experiment_time}_depth{depth_bin}.csv"
    merged.to_csv(OUT_DIR / out_name, index=False)

    print(f"[DONE] {out_name} | rows={len(merged)}")

print("\n✅ integrate by experiment_time + index + depth(200bin) + column reorder DONE")
