# train_models_full.py
# 전체 데이터로 학습만 진행 (테스트 분리 없음, 따로 준비한 테스트셋에서 평가)

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
OUT_DIR  = Path(r"C:\Users\PREMA\Desktop\FinalReport\ML\models")   # 모델 저장 위치
DATA_CSV = Path(r"C:\Users\PREMA\Desktop\FinalReport\Dataset\7MLdataset\edited_dataset_window1000.csv")  # 특징 CSV
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[i] 데이터 로딩:", DATA_CSV)
df = pd.read_csv(DATA_CSV)

# ===== 입력/출력 데이터 =====
feat_cols = ["Max", "Min", "Mean", "Std", "Median", "IQR", "RMS", "Skewness", "Kurtosis"]
X = df[feat_cols].astype(np.float32).values
y = df["Label"].values.astype(int)

print(f"[i] 전체 샘플 수: {len(df):,}, 특성 수: {len(feat_cols)}")
unique, counts = np.unique(y, return_counts=True)
print("[i] 라벨 분포:", dict(zip(unique.tolist(), counts.tolist())))

# 학습은 전체 데이터로 진행 (테스트셋 분리 없음)
X_tr, y_tr = X, y

# ===== 모델들 =====
models = {
    "rf":  Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        ))
    ]),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))
    ]),
    "dt":  Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=None, random_state=42, class_weight="balanced"))
    ]),
    "nb":  Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ]),
    "svm_linear": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            C=1.0, class_weight="balanced",
            dual=False, max_iter=10000, random_state=42
        ))
    ]),
}

# ===== 학습 및 저장 =====
iterator = list(models.items())
if tqdm:
    iterator = tqdm(iterator, desc="모델 학습(progress)", unit="model")

for name, pipe in iterator:
    t0 = time.time()
    if tqdm: iterator.set_postfix_str(name)

    pipe.fit(X_tr, y_tr)   # 전체 데이터로 학습

    took = time.time() - t0
    print(f"=== {name} === 학습 완료 (took {took:.2f}s)")

    # 모델 저장
    joblib.dump(pipe, OUT_DIR / f"model_{name}_w500.joblib", compress=("xz", 5))

# ===== 메타 정보 저장 =====
meta = {
    "features": feat_cols,
    "models": list(models.keys()),
    "created_at": time.ctime(),
    "notes": {
        "linear_svm_dual": False,
        "rf_n_jobs": -1,
        "compression": "xz, level=5",
    }
}
with open(OUT_DIR / "models_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("✅ 모델 저장 완료:", ", ".join([f"model_{k}_w1000.joblib" for k in models.keys()]))
