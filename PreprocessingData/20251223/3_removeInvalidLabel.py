# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

# ===== 경로 설정 =====
IN_DIR = Path(
    r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\3_DepthInterpolated"
)
OUT_DIR = Path(
    r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\4_removedInvalidLabel"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 허용 라벨 =====
valid_labels = {"Go", "Hold", "Back"}

# ===== 파일 수집 =====
all_files = list(IN_DIR.glob("*.csv")) + list(IN_DIR.glob("*.CSV"))
print(f"[INFO] Found {len(all_files)} CSV files")

kept = 0
removed = 0

# ===== 파일 단위 검사 =====
for csv_path in all_files:
    try:
        df = pd.read_csv(csv_path, dtype=str)

        # Label 열 존재 확인
        if "Label" not in df.columns:
            print(f"[DROP] {csv_path.name} | no Label column")
            removed += 1
            continue

        # Label 정리
        labels = df["Label"].astype(str).str.strip()

        # 조건 검사: 모든 행이 valid_labels 안에 있어야 함
        if not labels.isin(valid_labels).all():
            print(f"[DROP] {csv_path.name} | invalid label detected")
            removed += 1
            continue

        # 통과 → 저장
        out_path = OUT_DIR / csv_path.name
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        kept += 1

    except Exception as e:
        print(f"[ERROR] {csv_path.name}: {e}")
        removed += 1

print(f"\n[INFO] Done | kept={kept}, removed={removed}")
