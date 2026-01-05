# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

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
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        out = self.norm(out)
        out = out.mean(dim=1)
        return self.classifier(out)
