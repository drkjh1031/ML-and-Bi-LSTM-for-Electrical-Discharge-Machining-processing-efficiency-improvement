# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

input_dir = r"C:\Users\drkjh\Desktop\251221\1_edit_depth\1editeddepth"
output_dir = r"C:\Users\drkjh\Desktop\251221\1_edit_depth\2expandeddepth"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".csv"):
        continue

    input_path = os.path.join(input_dir, filename)

    # ---------- 파일명 파싱 ----------
    name_no_ext = filename[:-4]
    parts = name_no_ext.split("_")

    command_part = parts[-2]   # Go24, Hold3, etc.
    depth = float(parts[-1])   # machining depth

    if command_part.startswith("Go"):
        command = "Go"
    elif command_part.startswith("Hold"):
        command = "Hold"
    else:
        print(f"[WARN] Command not recognized: {filename}")
        continue

    # ---------- CSV 읽기 ----------
    df = pd.read_csv(input_path, header=None)
    original_B = df.iloc[:, 1].to_numpy()  # use original B column only

    # ---------- 새 B열 생성 ----------
    if command == "Go":
        new_B = np.linspace(depth, depth + 1, 1000)
    else:  # Hold
        new_B = np.full(1000, depth)

    # ---------- 길이 1000 맞추기 ----------
    if len(original_B) < 1000:
        original_B = np.pad(
            original_B,
            (0, 1000 - len(original_B)),
            constant_values=np.nan
        )
    else:
        original_B = original_B[:1000]

    # ---------- 새 DataFrame (A, B only) ----------
    new_df = pd.DataFrame({
        "A": original_B,
        "B": new_B
    })

    # ---------- 저장 ----------
    output_path = os.path.join(output_dir, filename)
    new_df.to_csv(output_path, index=False, header=False)

print("[OK] All files processed successfully.")
