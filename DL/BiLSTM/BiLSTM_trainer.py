# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class VoltageTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(device)

    def train(self, loader, epochs, lr):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_cnt = 0
            progress = tqdm(loader, desc=f"Epoch {epoch+1}")

            for inputs, targets in progress:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                bs = targets.size(0)
                total_loss += float(loss.item()) * bs
                total_cnt += bs

                avg_loss = total_loss / max(total_cnt, 1)
                progress.set_postfix(avg_loss=f"{avg_loss:.6f}")

            epoch_loss = total_loss / max(total_cnt, 1)
            print(f"[INFO] epoch={epoch+1}, avg_loss={epoch_loss:.6f}")
