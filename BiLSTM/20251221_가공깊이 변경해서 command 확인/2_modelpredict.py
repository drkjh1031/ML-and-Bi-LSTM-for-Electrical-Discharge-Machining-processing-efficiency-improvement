# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

# ===== 전역 변수 =====
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_initialized = False

# ===== 경로 설정 =====
MODEL_PATH = r"C:\Users\drkjh\Desktop\251221\Bi-LSTM.pth"
INPUT_DIR = r"C:\Users\drkjh\Desktop\251221\1_edit_depth\2expandeddepth"
OUTPUT_TXT = r"C:\Users\drkjh\Desktop\251221\1_edit_depth\prediction_results.txt"

# ===== 라벨 매핑 =====
ID_TO_LABEL = {0: "Hold", 1: "Go", 2: "Back"}

# ===== 깊이 정규화 기준 =====
DEPTH_MIN = -9.0
DEPTH_MAX = 700.0
SCALE_FACTOR = 0.2


def normalize_depth(depth_raw: np.ndarray):
    depth_norm = (depth_raw - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    return depth_norm * SCALE_FACTOR


# ===== BiLSTM 모델 =====
class VoltageBiLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)   # [B, 2, 1000] → [B, 1000, 2]
        out, _ = self.rnn(x)
        out = self.norm(out)
        out = out.mean(dim=1)
        return self.classifier(out)


# ===== 모델 초기화 =====
@torch.no_grad()
def init():
    global _model, _initialized

    if _initialized:
        return

    model = VoltageBiLSTM().to(_device)

    state = torch.load(MODEL_PATH, map_location=_device)
    new_state = {}
    for k, v in state.items():
        new_k = k.replace("bilstm.", "rnn.") if k.startswith("bilstm.") else k
        new_state[new_k] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()

    _model = model
    _initialized = True
    print("BiLSTM 모델 로드 완료")


# ===== 단일 파일 예측 =====
@torch.no_grad()
def predict_file(csv_path):
    df = pd.read_csv(csv_path, header=None)

    if df.shape != (1000, 2):
        raise ValueError(f"{os.path.basename(csv_path)} : 1000행 2열 아님")

    signal = df.iloc[:, 0].values.astype(np.float32)
    depth = df.iloc[:, 1].values.astype(np.float32)

    depth_norm = normalize_depth(depth)
    x = np.stack([signal, depth_norm], axis=0)
    x_tensor = torch.tensor(x).unsqueeze(0).to(_device)

    output = _model(x_tensor)
    pred_id = int(torch.argmax(output, dim=1).item())

    return ID_TO_LABEL[pred_id]


# ===== 폴더 전체 예측 =====
def run_batch_prediction():
    init()

    results = []

    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.lower().endswith(".csv"):
            continue

        fpath = os.path.join(INPUT_DIR, fname)

        try:
            label = predict_file(fpath)
            results.append(f"{fname} → {label}")
            print(f"{fname} → {label}")
        except Exception as e:
            results.append(f"{fname} → ERROR")
            print(f"{fname} → ERROR ({e})")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"\n결과 저장 완료: {OUTPUT_TXT}")


# ===== 실행 =====
if __name__ == "__main__":
    run_batch_prediction()
