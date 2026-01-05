# -*- coding: utf-8 -*-
import shutil
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

# ================== 경로 ==================
SRC_DIR = Path(r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\0_data")

CSV_DST = Path(r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\1_rawdata\csv")

IMG_BASE = Path(r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\1_rawdata\image")
IMG_GO   = IMG_BASE / "Go"
IMG_BACK = IMG_BASE / "Back"
IMG_HOLD = IMG_BASE / "Hold"
IMG_BAD  = IMG_BASE / "Bad"

for p in [CSV_DST, IMG_GO, IMG_BACK, IMG_HOLD, IMG_BAD]:
    p.mkdir(parents=True, exist_ok=True)

# ================== 정규식 ==================
CMD_REGEX = re.compile(r"(Go|Hold|Back)\d*")

# ================== 실험시간별 그룹 ==================
groups = defaultdict(list)

print("[INFO] Scanning csv files...")

for csv_path in SRC_DIR.glob("*.csv"):
    parts = csv_path.stem.split("_")
    if len(parts) < 5:
        continue

    exp_time = parts[1]      # 하이픈 포함 실험시간
    index = parts[2]
    cmd_part = parts[3]

    m = CMD_REGEX.search(cmd_part)
    if not m:
        continue

    command = m.group(1)

    # csv 복사
    shutil.copy(csv_path, CSV_DST / csv_path.name)

    groups[exp_time].append({
        "path": csv_path,
        "index": int(index),
        "command": command
    })

print(f"[INFO] Copied {sum(len(v) for v in groups.values())} valid csv files")

# ================== 그래프 함수 ==================
def plot_and_save(csv_path, save_dir):
    try:
        df = pd.read_csv(csv_path, header=None)
        y = df.iloc[:, 1] if df.shape[1] > 1 else df.iloc[:, 0]

        plt.figure(figsize=(6, 3))
        plt.plot(y)
        plt.ylim(-0.5, 8)           #  y축 고정
        plt.title(csv_path.name)
        plt.tight_layout()
        plt.savefig(save_dir / f"{csv_path.stem}.png")
        plt.close()
    except Exception as e:
        print(f"[ERROR] Plot failed: {csv_path.name} | {e}")

# ================== 인덱스 순서 + Bad 판별 ==================
print("[INFO] Processing experiments...")

for exp_time, files in groups.items():
    files = sorted(files, key=lambda x: x["index"])

    for i, f in enumerate(files):
        cmd = f["command"]
        path = f["path"]

        if cmd == "Go":
            plot_and_save(path, IMG_GO)

            # Go → Back 판별
            if i + 1 < len(files) and files[i + 1]["command"] == "Back":
                plot_and_save(path, IMG_BAD)

        elif cmd == "Back":
            plot_and_save(path, IMG_BACK)

        elif cmd == "Hold":
            plot_and_save(path, IMG_HOLD)

print(" DONE. All graphs saved with fixed y-axis (-0.5 ~ 8).")
