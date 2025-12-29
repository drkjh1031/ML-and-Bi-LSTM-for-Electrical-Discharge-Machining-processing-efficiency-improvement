import torch
import torch.nn as nn
import numpy as np
import os

def predict_voltage(signal_1d: np.ndarray, depth_1d: np.ndarray):
    device = "cpu"
    model_path = r"D:\종합설계\machine learning2\cnn_bilstm.pth"

    signal_1d = np.asarray(signal_1d, dtype=np.float32)
    depth_1d = np.asarray(depth_1d, dtype=np.float32)
    if signal_1d.shape != (500,) or depth_1d.shape != (500,):
        raise ValueError("signal_1d랑 depth_1d 변수가 1차원 배열 형태가 아님~~~!")

    arr_2d = np.stack([signal_1d, depth_1d], axis=0)  
    x_tensor = torch.tensor(arr_2d, dtype=torch.float32).unsqueeze(0).to(device)  

    # 모델 구조 BiLSTM에서 여기 수정!!
    cnn = nn.Sequential(
        nn.Conv1d(2, 32, kernel_size=5, padding=2),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Dropout(0.2)
    )
    bilstm = nn.LSTM(
        input_size=32,
        hidden_size=128,
        num_layers=1,
        batch_first=True,
        dropout=0,
        bidirectional=True
    )
    classifier = nn.Sequential(
        nn.Linear(128 * 2, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 3)
    )
    #여기까지 모델

    def forward(x):
        feat = cnn(x)               # [B, C, W/2]
        feat = feat.permute(0, 2, 1)  # >[B, W/2, C]
        out, _ = bilstm(feat)       # >[B, W/2, 2H]
        out = out.mean(dim=1)       # >[B, 2H]
        return classifier(out)

    #모델 로드
    class TempModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = cnn
            self.bilstm = bilstm
            self.classifier = classifier
        def forward(self, x):
            return forward(x)

    model = TempModel()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"폴더에 모델 없음: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(x_tensor)
        predicted = torch.argmax(output, dim=1).item()

    return predicted
