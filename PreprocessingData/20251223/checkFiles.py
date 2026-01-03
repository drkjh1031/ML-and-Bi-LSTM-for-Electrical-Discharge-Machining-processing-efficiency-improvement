import os

dir_path = r"C:\Users\PREMA\Desktop\진하\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\20251223\4_removedInvalidLabel"

count = 0
back_files = []

for fn in os.listdir(dir_path):
    if not fn.lower().endswith(".csv"):
        continue

    parts = fn.split("_")
    if len(parts) < 5:
        continue  # 형식 안 맞는 파일 방어

    command_part = parts[3]  # (command+command번호)

    if command_part.lower().startswith("go"):
        count += 1
        back_files.append(fn)

print(f"Go 파일 개수: {count}")
