import os
import shutil

# ===== 경로 설정 =====
SRC_DIR = r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\논문\20260107"
DST_DIR = r"C:\Users\drkjh\Desktop\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\논문\Dataset"

os.makedirs(DST_DIR, exist_ok=True)

# ===== 허용 커맨드 =====
valid_commands = ["_Go", "_Hold", "_Back"]

moved = 0

for filename in os.listdir(SRC_DIR):
    if not filename.lower().endswith(".csv"):
        continue

    if any(cmd in filename for cmd in valid_commands):
        src_path = os.path.join(SRC_DIR, filename)
        dst_path = os.path.join(DST_DIR, filename)

        shutil.copy2(src_path, dst_path)  # copy2 = 메타데이터 유지
        moved += 1

print(f"[DONE] 총 {moved}개 파일을 Dataset 폴더로 복사했습니다.")
