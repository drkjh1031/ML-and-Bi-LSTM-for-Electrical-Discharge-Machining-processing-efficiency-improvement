# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===== 1. 데이터 로드 =====
csv_path = r"C:\Users\PREMA\Desktop\FinalReport\Dataset\7MLdataset\edited_dataset_window1000.csv"
df = pd.read_csv(csv_path)

# ===== 2. 특징 / 라벨 분리 =====
features = ["Max", "Min", "Mean", "Std", "Median", "IQR", "RMS", "Skewness", "Kurtosis"]
X = df[features].values

# ===== 3. 표준화 =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== 4. PCA (3차원) =====
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# ===== 5. 로딩(기여도) 행렬 =====
loadings = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=["PC1", "PC2", "PC3"]
)
print("=== Feature Contributions (Loadings) ===")
print(loadings)

# ===== 6. 3차원 로딩 벡터 시각화 =====
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# 원점에서 각 특징의 로딩벡터를 화살표로 표시
for feature in features:
    x, y, z = loadings.loc[feature, ["PC1", "PC2", "PC3"]]
    ax.quiver(0, 0, 0, x, y, z,
              arrow_length_ratio=0.1,
              color='steelblue', alpha=0.8)
    ax.text(x * 1.1, y * 1.1, z * 1.1, feature, fontsize=10, color='black')

# ===== 축 설정 =====
ax.set_xlabel("PC1", labelpad=10)
ax.set_ylabel("PC2", labelpad=10)
ax.set_zlabel("PC3", labelpad=10)
ax.set_title("3D Visualization of Feature Contributions (PCA Loadings)", fontsize=13, weight='bold')

# 보기 좋게 축 범위 조정
max_range = np.max(np.abs(loadings.values)) * 1.2
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
