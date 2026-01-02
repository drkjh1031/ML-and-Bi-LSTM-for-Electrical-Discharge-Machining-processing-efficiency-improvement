# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
from collections import defaultdict

IN_DIR = Path(r"C:\Users\drkjh\Desktop\FinalReport\Dataset\2editeddata")
OUT_DIR = Path(r"C:\Users\drkjh\Desktop\FinalReport\Dataset\3integrateddata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 저장시간별 그룹핑
groups = defaultdict(list)
for f in IN_DIR.glob("*.csv"):
    parts = f.stem.split("_")
    if len(parts) < 5:
        continue
    time_tag = parts[1]       # 저장시간
    idx = int(parts[2])       # 인덱스 번호
    groups[time_tag].append((idx, f))

# 그룹별 통합
for time_tag, file_list in groups.items():
    file_list.sort(key=lambda x: x[0])  # 인덱스 순 정렬
    merged = pd.concat([pd.read_csv(p) for _, p in file_list], ignore_index=True)
    out_path = OUT_DIR / f"{time_tag}_merged.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[check] {time_tag} merge complete ({len(file_list)} files, {len(merged)}lines)")
