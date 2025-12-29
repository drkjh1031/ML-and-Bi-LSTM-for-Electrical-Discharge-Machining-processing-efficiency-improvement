# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

IN_DIR = Path(r"C:\Users\drkjh\Desktop\FinalReport\Dataset\3integrateddata")
OUT_DIR = Path(r"C:\Users\drkjh\Desktop\FinalReport\Dataset\4expandeddata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for csv_path in IN_DIR.glob("*.csv"):
    df = pd.read_csv(csv_path)

    if "Depth" not in df.columns:
        print(f"[!] {csv_path.name}: No Depth column, skip")
        continue

    # 등차보간 (NaN 구간 채우기)
    df["Depth"] = df["Depth"].interpolate(method="linear", limit_direction="both")

    out_path = OUT_DIR / csv_path.name
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] {csv_path.name} interpolate complete ({len(df)}line)")
