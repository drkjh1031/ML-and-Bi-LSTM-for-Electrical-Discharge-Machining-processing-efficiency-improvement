# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.inspection import permutation_importance

# ===== 데이터 로드 =====
csv_path = r"C:\Users\PREMA\Desktop\FinalReport\Dataset\7MLdataset\edited_dataset_window1000.csv"
df = pd.read_csv(csv_path)

features = ["Max", "Min", "Mean", "Std", "Median", "IQR", "RMS", "Skewness", "Kurtosis"]
X = df[features].values
y = df["Label"].values.astype(int)

# ===== 결과 저장 폴더 =====
save_dir = os.path.join(os.path.dirname(csv_path), "plots")
os.makedirs(save_dir, exist_ok=True)

# ===== Train/Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== 모델 정의 =====
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "NaiveBayes": GaussianNB(),
    "SVM": LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=10000, random_state=42)
}

# ===== 데이터 스케일링 =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 폰트 크기 설정 =====
plt.rcParams.update({
    "font.size": 24,           # 기본 폰트
    "axes.titlesize": 36,      # 차트 제목 (2배)
    "axes.labelsize": 32,      # 축 제목 (2배)
    "xtick.labelsize": 50,     # X축 폰트 (3배)
    "ytick.labelsize": 35,     # Y축 눈금
    "legend.fontsize": 48      # 범례 (1.5배)
})

# ===== Permutation Importance =====
plt.figure(figsize=(25, 12))  # 그래프 크기 2배 확대
colors = cycle(["orange", "green", "blue", "red", "purple"])

for (name, model), color in zip(models.items(), colors):
    model.fit(X_train_scaled, y_train)
    r = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importances = r.importances_mean
    plt.plot(features, importances / np.max(importances), marker='o', label=name, color=color, lw=4)

plt.title("Permutation Importance (Normalized, All Models)", fontsize=45)
plt.ylabel("Normalized Importance", fontsize=40)
plt.xlabel("Features", fontsize=40)
plt.xticks(rotation=30)
plt.legend(loc="upper left", fontsize=38)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "PermutationImportance_large.png"), dpi=400)
plt.close()

print(f"[+] Permutation Importance 그래프가 {save_dir} 폴더에 큰 사이즈(20x12인치, 400dpi)로 저장되었습니다.")
