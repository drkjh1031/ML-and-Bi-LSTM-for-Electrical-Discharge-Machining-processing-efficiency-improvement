import os
import shutil

# ---------------- 경로 설정 ----------------
SRC_DIR = r"C:\Users\drkjh\Desktop\진하\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\PreprocessingData\1_rawdata\csv"
DST_DIR = r"C:\Users\drkjh\Desktop\진하\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\ML\20251227_checkSkewness\generalBack"

# ---------------- 출력 폴더 생성 ----------------
os.makedirs(DST_DIR, exist_ok=True)

# ---------------- 파일 순회 ----------------
copied = 0

for filename in os.listdir(SRC_DIR):
    if not filename.lower().endswith(".csv"):
        continue

    #  command == Back 인 파일만
    # 예: _Back573_
    if "_Back" not in filename:
        continue

    src_path = os.path.join(SRC_DIR, filename)
    dst_path = os.path.join(DST_DIR, filename)

    shutil.copy2(src_path, dst_path)
    copied += 1

print(f" Back command 파일 {copied}개 복사 완료")
