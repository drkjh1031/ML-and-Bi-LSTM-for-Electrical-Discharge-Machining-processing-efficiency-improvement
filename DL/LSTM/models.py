# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class VoltageLSTM(nn.Module):
    def __init__(
        self,
        hidden_size=128,
        num_layers=2,
        num_classes=3,
        dropout=0.2
    ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, voltage, depth):
        # voltage: [B, 1000] ¡æ [B, 1000, 1]
        x = voltage.unsqueeze(-1)

        out, _ = self.rnn(x)
        h_last = out[:, -1, :]      # last timestep

        fused = torch.cat([h_last, depth], dim=1)
        return self.classifier(fused)
