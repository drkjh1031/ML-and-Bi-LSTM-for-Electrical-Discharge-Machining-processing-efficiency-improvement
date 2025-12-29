# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json

# ===== 경로/파라미터 =====
IN_DIR  = Path(r"C:\Users\drkjh\Desktop\FinalReport\Dataset\6finaldata")  # 원본 CSV 폴더
OUT_DIR = Path(r"C:\Users\drkjh\Desktop\FinalReport\Dataset\7MLdataset")                # 출력 폴더
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 1000   # 윈도우 크기
HOP    = 1000   # 겹치지 않게 → HOP == WINDOW

PAD_SHORT = True
MIN_SHORT_LEN = max(64, WINDOW // 4)

# 라벨 매핑
LBL2ID = {"H": 0, "G": 1, "B": 2}
ID2LBL = {v: k for k, v in LBL2ID.items()}

# ===== CSV 읽기 =====
def read_any_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "cp949"):
        try:
            return pd.read_csv(path, engine="python", sep=None, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, engine="python", sep=None)

# ===== 신호/라벨 추출 =====
def get_sig_lbl(df: pd.DataFrame):
    if df.shape[1] < 3:
        raise ValueError("열이 3개 미만 (Time, Signal, Label 필요)")
    sig = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    lbl = df["Label"] if "Label" in df.columns else df.iloc[:, 2]
    lbl = lbl.astype(str).str.strip()

    # 라벨 매핑: Go/Hold/Back → G/H/B, "-" 또는 빈칸 → H
    lbl = lbl.replace({
        "Go": "G",
        "Hold": "H",
        "Back": "B"
    })

    mask = sig.notna() & lbl.isin(["G", "H", "B"])
    return sig[mask].to_numpy(dtype=float), lbl[mask].to_numpy(dtype=str)

# ===== 윈도우 라벨 결정 (최빈값만 사용) =====
def window_label(lbls: np.ndarray) -> int:
    cnt = Counter(lbls.tolist())
    top = max(cnt.items(), key=lambda kv: kv[1])[0]  # 최빈값
    return LBL2ID[top]

# ===== 특징 9개 =====
def feat9(x: np.ndarray) -> dict:
    q25, q75 = np.percentile(x, [25, 75])
    s = pd.Series(x)
    return {
        "Max": float(np.max(x)),
        "Min": float(np.min(x)),
        "Mean": float(np.mean(x)),
        "Std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "Median": float(np.median(x)),
        "IQR": float(q75 - q25),
        "RMS": float(np.sqrt(np.mean(x * x))),
        "Skewness": float(s.skew()) if len(x) > 2 else 0.0,
        "Kurtosis": float(s.kurt()) if len(x) > 3 else 0.0,
    }

# ===== 메인 루프 =====
rows = []
for csv_path in sorted(IN_DIR.glob("*.csv")):
    try:
        df = read_any_csv(csv_path)
        sig, lbl = get_sig_lbl(df)
        n = len(sig)

        if n < WINDOW:
            if PAD_SHORT and n >= MIN_SHORT_LEN:
                xs = np.zeros(WINDOW, dtype=float)
                xs[:n] = sig
                ys = lbl
                f = feat9(xs); y = window_label(ys)
                rows.append({"Filename": csv_path.name, **f, "Label": y})
                print(f"[pad] {csv_path.name}: {n} -> {WINDOW}")
            else:
                print(f"[skip] {csv_path.name}: {n} < WINDOW {WINDOW}")
            continue

        start = 0
        made = 0
        while start + WINDOW <= n:
            xs = sig[start:start+WINDOW]
            ys = lbl[start:start+WINDOW]
            f = feat9(xs); y = window_label(ys)
            rows.append({"Filename": csv_path.name, **f, "Label": y})
            made += 1
            start += HOP

        # 끝 꼬리 처리 (패딩)
        tail = n - start
        if PAD_SHORT and tail >= MIN_SHORT_LEN and n >= WINDOW:
            xs = np.zeros(WINDOW, dtype=float)
            seg = sig[-WINDOW:]
            xs[:len(seg)] = seg
            ys = lbl[-WINDOW:]
            f = feat9(xs); y = window_label(ys)
            rows.append({"Filename": csv_path.name, **f, "Label": y})
            made += 1

        print(f"[OK] {csv_path.name}: windows={made}")
    except Exception as e:
        print(f"[!] {csv_path.name} 실패: {e}")

# ===== 저장 =====
dataset = pd.DataFrame(rows)
save_csv = OUT_DIR / "edited_dataset_window1000.csv"
dataset.to_csv(save_csv, index=False)

meta = {
    "WINDOW": WINDOW,
    "HOP": HOP,
    "features": ["Max", "Min", "Mean", "Std", "Median", "IQR", "RMS", "Skewness", "Kurtosis"],
    "label_rule": "window 라벨 = 최빈값"
}
with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ saved: {save_csv} (rows={len(dataset)})")
