import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# ---------------- 경로 설정 ----------------
INPUT_DIR = r"C:\Users\drkjh\Desktop\진하\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\ML\20251227_BadSignal_checkSkewness\csv"
OUTPUT_PATH = r"C:\Users\drkjh\Desktop\진하\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\ML\20251227_BadSignal_checkSkewness\SkewnessOfBadsignals\BadSignals.csv"

# ---------------- 결과 저장용 리스트 ----------------
results = []

# ---------------- CSV 파일 전체 순회 ----------------
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(".csv"):
        continue

    file_path = os.path.join(INPUT_DIR, filename)

    try:
        # 헤더 없음, B열만 사용 (index 1)
        data = pd.read_csv(file_path, header=None)
        b = data.iloc[:, 1].values.astype(float)

        # ---------------- 특징 계산 ----------------
        max_v = np.max(b)
        min_v = np.min(b)
        mean_v = np.mean(b)
        std_v = np.std(b, ddof=1)
        median_v = np.median(b)
        iqr_v = np.percentile(b, 75) - np.percentile(b, 25)
        rms_v = np.sqrt(np.mean(b ** 2))
        skew_v = skew(b)
        kurt_v = kurtosis(b)  # Fisher kurtosis (정규분포 기준 0)

        results.append([
            filename,
            max_v,
            min_v,
            mean_v,
            std_v,
            median_v,
            iqr_v,
            rms_v,
            skew_v,
            kurt_v
        ])

    except Exception as e:
        print(f"ERROR: {filename} → {e}")

# ---------------- DataFrame 생성 ----------------
columns = [
    "Filename",
    "Max",
    "Min",
    "Mean",
    "Std",
    "Median",
    "IQR",
    "RMS",
    "Skewness",
    "Kurtosis"
]

df_result = pd.DataFrame(results, columns=columns)

# ---------------- 저장 ----------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(" BadSignals.csv 생성 완료")
