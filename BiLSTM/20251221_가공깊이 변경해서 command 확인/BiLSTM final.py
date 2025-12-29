# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import os

# ===== 전역 변수 =====
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_initialized = False

# ===== 모델 경로 (수정 금지) =====
_MODEL_PATH = r"C:\Users\drkjh\Desktop\251221\Bi-LSTM.pth"

# ===== 깊이 정규화 기준 (학습 데이터 기반) =====
DEPTH_MIN = -9.0
DEPTH_MAX = 700.0
SCALE_FACTOR = 0.2


# ===== 깊이 정규화 함수 =====
def normalize_depth(depth_raw: np.ndarray):
    """
    실험에서 들어온 실제 가공깊이를
    학습 데이터 스케일(0~0.2)로 정규화
    """
    depth_norm = (depth_raw - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    return depth_norm * SCALE_FACTOR


# ===== 현재 학습된 BiLSTM 구조 정의 =====
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
        # 입력: [B, 2, 1000] → [B, 1000, 2]
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        out = self.norm(out)
        out = out.mean(dim=1)
        return self.classifier(out)


# ===== 초기화 함수 =====
@torch.no_grad()
def init():
    """루프 시작 전 1회 호출: 모델 로드 및 워밍업"""
    global _model, _initialized

    if _initialized:
        return True

    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {_MODEL_PATH}")

    model = VoltageBiLSTM(input_size=2, hidden_size=128, num_layers=2, num_classes=3).to(_device)

    # === 키 이름 매핑 (bilstm → rnn) 자동 수정 ===
    state = torch.load(_MODEL_PATH, map_location=_device)
    new_state = {}
    for k, v in state.items():
        new_k = k.replace("bilstm.", "rnn.") if k.startswith("bilstm.") else k
        new_state[new_k] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()

    # 워밍업용 dummy 입력
    dummy_signal = torch.zeros(1, 2, 1000, device=_device)
    _ = model(dummy_signal)

    _model = model
    _initialized = True
    print(" BiLSTM 모델 초기화 완료 (정규화 적용)")
    return True


# ===== 예측 함수 =====
@torch.no_grad()
def prediction(signal_1d: np.ndarray, depth_1d: np.ndarray):
    """
    signal_1d, depth_1d: 길이 1000짜리 1D numpy 배열
    return: 예측된 클래스 (0=Hold, 1=Go, 2=Back)
    """
    global _model, _initialized
    if not _initialized or _model is None:
        raise RuntimeError(" 먼저 init()을 호출해야 합니다!")

    # --- 입력 검증 ---
    signal_1d = np.asarray(signal_1d, dtype=np.float32)
    depth_1d = np.asarray(depth_1d, dtype=np.float32)

    if signal_1d.shape != (1000,) or depth_1d.shape != (1000,):
        raise ValueError("signal_1d, depth_1d 모두 길이 1000의 1차원 배열이어야 합니다.")

    # --- 깊이 정규화 ---
    depth_norm = normalize_depth(depth_1d)

    # --- 텐서 변환 ---
    arr_2d = np.stack([signal_1d, depth_norm], axis=0)  # [2, 1000]
    x_tensor = torch.tensor(arr_2d, dtype=torch.float32).unsqueeze(0).to(_device)

    # --- 예측 ---
    output = _model(x_tensor)
    predicted = int(torch.argmax(output, dim=1).item())

    return predicted


# ===== 사용 예시 =====
"""
if __name__ == "__main__":
    ok = init()  # 1회 호출
    if ok:
        while True:
            # 예시 입력
            sig = np.random.uniform(0, 6.9, 1000)  # 전압신호
            dep = np.linspace(300.0, 301.0, 1000)  # 실제 가공깊이(mm)
            label = prediction(sig, dep)
            print("예측 결과:", label)
"""
