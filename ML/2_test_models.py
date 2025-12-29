# 3_test_models.py
import os
import pandas as pd
import joblib
import numpy as np

# ===== 경로 =====
DATA_PATH = r"C:\Users\drkjh\Desktop\KSPE\ML\0348_300-400_p01.csv"
MODEL_DIR = r"C:\Users\drkjh\Desktop\KSPE\ML\models"

WINDOW = 500  # 윈도우 크기

# ===== 안전한 특징 추출 함수 =====
def _safe_iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)

def _safe_skew(x):
    n = x.size
    if n < 3: return 0.0
    mu = float(x.mean()); s = float(x.std(ddof=0))
    if s == 0.0: return 0.0
    m3 = float(np.mean((x - mu) ** 3))
    return m3 / (s**3 + 1e-12)

def _safe_kurtosis(x):
    n = x.size
    if n < 4: return 0.0
    mu = float(x.mean()); s = float(x.std(ddof=0))
    if s == 0.0: return 0.0
    m4 = float(np.mean((x - mu) ** 4))
    return m4 / (s**4 + 1e-12)

def extract_features(x: np.ndarray) -> np.ndarray:
    """윈도우(500개)에 대해 9개 특징 추출"""
    x = np.asarray(x, dtype=np.float32).ravel()
    return np.array([
        float(np.max(x)),               # Max
        float(np.min(x)),               # Min
        float(np.mean(x)),              # Mean
        float(np.std(x, ddof=0)),       # Std
        float(np.median(x)),            # Median
        _safe_iqr(x),                   # IQR
        float(np.sqrt(np.mean(x**2))),  # RMS
        _safe_skew(x),                  # Skewness
        _safe_kurtosis(x),              # Kurtosis
    ], dtype=np.float32)

# ===== 데이터 로드 =====
print(f"[i] 데이터 로딩: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

signal = df.iloc[:, 1].to_numpy().astype(float)  # 2번째 열
n = len(signal)
print(f"[i] 전체 샘플 수: {n}")

# ===== 모델 불러오기 =====
model_map = {
    "model_dt_w500.joblib": "dt",
    "model_knn_w500.joblib": "knn",
    "model_nb_w500.joblib": "nb",
    "model_rf_w500.joblib": "rf",
    "model_svm_linear_w500.joblib": "svm",
}

models = {}
for fname, colname in model_map.items():
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        models[colname] = joblib.load(path)
        print(f"[+] {fname} 로드 완료")
    else:
        print(f"[!] {fname} 없음, 스킵")

# ===== 예측 (500개씩 블록 → 예측 1개 → 500행 반복) =====
# 초기화
for colname in model_map.values():
    df[colname] = "-"

for start in range(0, n, WINDOW):
    end = start + WINDOW
    if end > n: break
    block = signal[start:end]
    feat = extract_features(block).reshape(1, -1)

    for colname, model in models.items():
        try:
            pred_label = int(model.predict(feat)[0])
            df.loc[start:end-1, colname] = pred_label
        except Exception as e:
            print(f"[!] {colname} 예측 실패 at block {start}-{end}: {e}")

# ===== 숫자 → 문자열 라벨 변환 =====
label_map = {0: "Hold", 1: "Go", 2: "Back"}
for colname in model_map.values():
    if colname in df.columns:
        df[colname] = df[colname].map(label_map)

# ===== 저장 =====
out_path = DATA_PATH.replace(".csv", "_pred.csv")
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"[✓] 저장 완료: {out_path}")
