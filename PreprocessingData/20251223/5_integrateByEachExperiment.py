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

# ================= Label 매핑 =================
LABEL_MAP = {
    "Hold": 0,
    "Go": 1,
    "Back": 2
}

# ================= 그룹 저장 =================
groups = defaultdict(list)

# ================= 파일 분류 =================
for p in IN_DIR.glob("*.csv"):
    parts = p.stem.split("_")

    if len(parts) != 5:
        print(f"[SKIP] unexpected filename: {p.name}")
        continue

    try:
        experiment_time = parts[1]
        index = int(parts[2])
        depth = int(float(parts[4]))
    except Exception as e:
        print(f"[SKIP] parse error: {p.name} ({e})")
        continue

    depth_start = (depth // 200) * 200
    depth_end = depth_start + 200
    depth_bin = f"{depth_start}-{depth_end}"

    groups[(experiment_time, depth_bin)].append((index, p))

# ================= 병합 =================
for (experiment_time, depth_bin), items in groups.items():

    items.sort(key=lambda x: x[0])
    dfs = []

    for _, p in items:
        df = pd.read_csv(p)

        expected_cols = ["Time", "V", "Depth", "Label"]

        if not set(expected_cols).issubset(df.columns):
            print(f"[WARN] column mismatch: {p.name}")
            print(f"       found columns: {df.columns.tolist()}")
            continue

        # ===== 열 순서 재정렬 =====
        df = df[expected_cols]

        # ===== Label 문자 → 숫자 변환 =====
        df["Label"] = df["Label"].map(LABEL_MAP)

        # 혹시라도 매핑 안 된 값 체크
        if df["Label"].isna().any():
            print(f"[WARN] unknown label found in {p.name}")
            continue

        df["Label"] = df["Label"].astype(int)

        dfs.append(df)

    if not dfs:
        print(f"[SKIP] no valid CSVs for {experiment_time}_depth{depth_bin}")
        continue

    merged = pd.concat(dfs, ignore_index=True)

    out_name = f"{experiment_time}_depth{depth_bin}.csv"
    merged.to_csv(OUT_DIR / out_name, index=False)

    print(f"[DONE] {out_name} | rows={len(merged)}")

print("\n✅ integrate + depth bin + label numeric DONE")
