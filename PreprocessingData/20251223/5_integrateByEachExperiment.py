# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

# ===== 경로 설정 =====
IN_DIR = Path(r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\4_removedInvalidLabel")
OUT_DIR = Path(r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\5_BiLSTM_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Searching in: {IN_DIR.resolve()}")
all_files = list(IN_DIR.glob("*.csv")) + list(IN_DIR.glob("*.CSV"))
print(f"[INFO] Found {len(all_files)} CSV files")

# ===== 허용 가능한 라벨 =====
valid_labels = {"Go", "Hold", "Back"}

# ===== 파일 처리 루프 =====
for csv_path in all_files:
    try:
        df = pd.read_csv(csv_path, dtype=str)  # 문자열로 읽기 (공백, NaN 구분 정확하게)
        before = len(df)

        if "Label" not in df.columns:
            print(f"[SKIP] {csv_path.name} | No 'Label' column found")
            continue

        # Label 값 정리: 앞뒤 공백 제거
        df["Label"] = df["Label"].astype(str).str.strip()

        # 허용된 값만 남기기
        df = df[df["Label"].isin(valid_labels)]
        removed = before - len(df)

        # 결과 저장
        save_path = OUT_DIR / csv_path.name
        df.to_csv(save_path, index=False, encoding="utf-8-sig")

        if removed > 0:
            print(f"[DEL] {csv_path.name} | {removed} invalid rows removed")
        else:
            print(f"[OK] {csv_path.name} | all labels valid")

    except Exception as e:
        print(f"[ERROR] {csv_path.name}: {e}")

print("\n[INFO] Processing complete.")
