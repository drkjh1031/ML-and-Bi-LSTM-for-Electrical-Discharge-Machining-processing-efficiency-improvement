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
            total_loss, total_cnt = 0.0, 0
            progress = tqdm(loader, desc=f"Epoch {epoch+1}")

            for volt, depth, target in progress:
                volt = volt.to(self.device)
                depth = depth.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = self.model(volt, depth)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                bs = target.size(0)
                total_loss += loss.item() * bs
                total_cnt += bs

                progress.set_postfix(
                    avg_loss=f"{total_loss / total_cnt:.6f}"
                )

            print(
                f"[INFO] epoch={epoch+1}, "
                f"avg_loss={total_loss / total_cnt:.6f}"
            )
