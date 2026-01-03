# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from BiLSTM_models import VoltageBiLSTM
from BiLSTM_data_utils import create_loaders
from BiLSTM_trainer import VoltageTrainer


def evaluate_and_plot(model, loader, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    classes = ["Hold", "Go", "Back"]

    # ===== Confusion Matrix =====
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("BiLSTM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "BiLSTM_confmat.png"), dpi=400)
    plt.close()

    # ===== ROC Curve =====
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(6, 5))
    for i, cls in enumerate(classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"{cls} (AUC={roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("BiLSTM ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "BiLSTM_ROC.png"), dpi=400)
    plt.close()

    # ===== Precision–Recall Curve =====
    plt.figure(figsize=(6, 5))
    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, lw=2, label=f"{cls} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("BiLSTM Precision–Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "BiLSTM_PR.png"), dpi=400)
    plt.close()

    print(f"[+] Confusion Matrix, ROC, and PR curves saved in {save_dir}")


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}")

    train_loader = create_loaders(
        config['data_dir'],
        config['batch_size'],
        window_size=config['window_size'],
        oversample_B=config.get('oversample_B', 1),
        num_workers=config.get('num_workers', 0)
    )

    model = VoltageBiLSTM(input_size=2, hidden_size=128, num_layers=2, num_classes=3).to(device)
    trainer = VoltageTrainer(model, device)

    print("\n== TRAINING START ==")
    trainer.train(train_loader, epochs=config['epochs'], lr=config['lr'])

    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    torch.save(model.state_dict(), config['save_path'])
    print(f"[DONE] Model saved at {config['save_path']}")

    # === Evaluate and plot ===
    evaluate_and_plot(model, train_loader, device, config['fig_save_dir'])


if __name__ == '__main__':
    config = {
        'data_dir': r'C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\5_BiLSTM_dataset',
        'batch_size': 32,
        'window_size': 1000,
        'oversample_B': 1,
        'num_workers': 0,
        'epochs': 15,
        'lr': 1e-3,
        'save_path': r'C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\BiLSTM\1223model_BiLSTM.pth',
        'fig_save_dir': r'C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\BiLSTM'
    }
    main(config)
