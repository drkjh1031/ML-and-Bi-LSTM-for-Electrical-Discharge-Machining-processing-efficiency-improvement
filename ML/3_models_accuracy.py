import pandas as pd

# 파일 경로
csv_path = r"C:\Users\drkjh\Desktop\KSPE\ML\0348_300-400_p01_pred.csv"

# CSV 읽기
df = pd.read_csv(csv_path)

# 정답 라벨 (3번째 열)
true_col = df.columns[2]
y_true = df[true_col].astype(str).str.strip()

# 모델 예측 열 (5~9번째)
pred_cols = df.columns[4:9]

print(f"[i] 정답 열: {true_col}")
print(f"[i] 예측 열: {list(pred_cols)}\n")

# 모델별 정확도 계산
for col in pred_cols:
    y_pred = df[col].astype(str).str.strip()

    # 전체 정확도
    acc = (y_pred == y_true).mean()

    # 라벨별 정확도
    label_acc = {}
    for label in sorted(y_true.unique()):  # Go, Hold, Back
        mask = (y_true == label)
        if mask.sum() > 0:
            label_acc[label] = (y_pred[mask] == y_true[mask]).mean()
        else:
            label_acc[label] = None

    print(f"=== {col} ===")
    print(f"전체 정확도: {acc:.3f}")
    for label, a in label_acc.items():
        if a is not None:
            print(f"  {label} 정확도: {a:.3f}")
    print()
