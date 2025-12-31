import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# ================= 출력 옵션 (생략 방지) =================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.max_colwidth", None)

# ---------------- 경로 설정 ----------------
INPUT_DIR = r"C:\Users\drkjh\Desktop\진하\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\ML\20251227_checkSkewness\generalGo"
OUTPUT_PATH = r"C:\Users\drkjh\Desktop\진하\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\ML\20251227_checkSkewness\generalGo.csv"

# ---------------- 결과 저장 ----------------
results = []

# ---------------- 파일 순회 ----------------
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(".csv"):
        continue

    # Go command만 처리
    if "_Go" not in filename:
        continue

    file_path = os.path.join(INPUT_DIR, filename)

    try:
        # CSV 읽기 (헤더 없음)
        df = pd.read_csv(file_path, header=None)

        # B열 (index 1)
        b = df.iloc[:, 1].astype(float).values

        # ---------------- 9가지 특징 계산 ----------------
        features = {
            "Max": np.max(b),
            "Min": np.min(b),
            "Mean": np.mean(b),
            "Std": np.std(b, ddof=1),
            "Median": np.median(b),
            "IQR": np.percentile(b, 75) - np.percentile(b, 25),
            "RMS": np.sqrt(np.mean(b ** 2)),
            "Skewness": skew(b),
            "Kurtosis": kurtosis(b)  # Fisher (정규분포 기준 0)
        }

        results.append({
            "Filename": filename,
            **features
        })

    except Exception as e:
        print(f"ERROR: {filename} → {e}")

# ---------------- DataFrame 생성 ----------------
df_result = pd.DataFrame(results)

# ---------------- CSV 저장 ----------------
df_result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(" generalGo.csv 생성 완료")

# ---------------- 2×9 행렬 생성 ----------------
feature_cols = [
    "Max", "Min", "Mean", "Std",
    "Median", "IQR", "RMS",
    "Skewness", "Kurtosis"
]

mean_row = df_result[feature_cols].mean()
std_row  = df_result[feature_cols].std(ddof=1)

matrix_2x9 = pd.DataFrame(
    [mean_row.values, std_row.values],
    index=["Mean", "Std"],
    columns=feature_cols
)

# ---------------- 터미널 출력 (절대 생략 없음) ----------------
print("\n Go 신호 특징 요약 (2×9 Matrix)")
print(matrix_2x9.to_string())
