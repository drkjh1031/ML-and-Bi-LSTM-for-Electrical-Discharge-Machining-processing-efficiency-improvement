# train_models_full.py
# 전체 데이터로 학습만 진행 (테스트 분리 없음)

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import joblib

# tqdm optional
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ===== 경로 =====
OUT_DIR  = Path(
    r"C:\Users\PREMA\Desktop\진하\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\compareModels\ML"
)
DATA_CSV = Path(r"C:\Users\PREMA\Desktop\진하\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\compareModels\ML_dataset.csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[i] 데이터 로딩:", DATA_CSV)
df = pd.read_csv(DATA_CSV)

# ===== 🔧 라벨 매핑 추가 =====
LABEL_MAP = {
    "Hold": 0,
    "Go":   1,
    "Back": 2
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Label이 문자열이면 숫자로 변환
if df["Label"].dtype == object:
    df["Label"] = df["Label"].map(LABEL_MAP)

# 안전 체크
if df["Label"].isnull().any():
    raise ValueError("❌ Label 컬럼에 매핑되지 않은 값이 있습니다.")

df["Label"] = df["Label"].astype(int)

print("[i] 라벨 매핑 확인:", df["Label"].value_counts().sort_index().to_dict())

# ===== 입력/출력 데이터 =====
feat_cols = ["Max", "Min", "Mean", "Std", "Median", "IQR", "RMS", "Skewness", "Kurtosis"]
X = df[feat_cols].astype(np.float32).values
y = df["Label"].values

print(f"[i] 전체 샘플 수: {len(df):,}, 특성 수: {len(feat_cols)}")

# ===== 학습 데이터 (전체 사용) =====
X_tr, y_tr = X, y

# ===== 모델 정의 =====
models = {
    "rf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))
    ]),
    "dt": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(
            max_depth=None,
            random_state=42,
            class_weight="balanced"
        ))
    ]),
    "nb": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ]),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual=False,
            max_iter=10000,
            random_state=42
        ))
    ]),
}

# ===== 학습 및 저장 =====
iterator = list(models.items())
if tqdm:
    iterator = tqdm(iterator, desc="모델 학습", unit="model")

for name, pipe in iterator:
    t0 = time.time()
    if tqdm:
        iterator.set_postfix_str(name)

    pipe.fit(X_tr, y_tr)

    took = time.time() - t0
    print(f"=== {name} === 학습 완료 ({took:.2f}s)")

    joblib.dump(
        pipe,
        OUT_DIR / f"1223model_{name}.joblib",
        compress=("xz", 5)
    )

# ===== 메타 정보 저장 =====
meta = {
    "features": feat_cols,
    "label_map": LABEL_MAP,
    "models": list(models.keys()),
    "created_at": time.ctime(),
}
with open(OUT_DIR / "models_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(" 모든 모델 학습 및 저장 완료")
