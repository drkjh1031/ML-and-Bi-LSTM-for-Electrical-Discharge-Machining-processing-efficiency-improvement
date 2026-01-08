# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from LSTM_data_utils import create_loaders
from LSTM_models import VoltageLSTM
from LSTM_trainer import VoltageTrainer


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for volt, depth, y in loader:
            volt = volt.to(device)
            depth = depth.to(device)
            y = y.to(device)

            out = model(volt, depth)
            pred = torch.argmax(out, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    return np.array(y_true), np.array(y_pred)


def main():
    config = {
        "data_dir": r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\논문\Dataset",
        "batch_size": 32,
        "window_size": 1000,
        "save_path": r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\논문\models\20260107_LSTM.pth",
        "fig_save_dir": r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\논문\models\LSTM_figs"
    }

    os.makedirs(config["fig_save_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    loader = create_loaders(
        config["data_dir"],
        config["batch_size"],
        window_size=config["window_size"]
    )

    model = VoltageLSTM(
        hidden_size=128,
        num_layers=2,
        num_classes=3
    ).to(device)

    trainer = VoltageTrainer(model, device)

    print("\n== LSTM TRAINING (90 epochs) ==")
    trainer.train(loader, epochs=30, lr=1e-3)
    trainer.train(loader, epochs=30, lr=5e-4)
    trainer.train(loader, epochs=30, lr=1e-4)

    torch.save(model.state_dict(), config["save_path"])
    print(f"[DONE] Model saved: {config['save_path']}")

    y_true, y_pred = evaluate(model, loader, device)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Hold", "Go", "Back"],
        yticklabels=["Hold", "Go", "Back"]
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(config["fig_save_dir"], "LSTM_confusion_matrix.png"),
        dpi=400
    )
    plt.close()


if __name__ == "__main__":
    main()
