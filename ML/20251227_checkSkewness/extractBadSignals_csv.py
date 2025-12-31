import os
import shutil
from collections import defaultdict

# ================== 경로 ==================
SRC_DIR = r"C:\Users\drkjh\Desktop\2025-2\FinalReport\Dataset\1회차 원본 csv"
DST_DIR = r"C:\Users\drkjh\Desktop\진하\ML-and-Bi-LSTM-for-Electrical-Discharge-Machining-processing-efficiency-improvement\ML\20251227_checkSkewness\check\Go-Back"

os.makedirs(DST_DIR, exist_ok=True)

# ================== 파일명 파싱 ==================
def parse_filename(fname):
    """
    (실험명)_(실험시간)_(인덱스)_(command+번호)_(가공깊이).csv
    """
    name = os.path.splitext(fname)[0]
    parts = name.split("_")

    if len(parts) < 5:
        return None

    try:
        experiment_time = parts[1]
        index = int(parts[2])
        cmd_part = parts[3].lower()

        if "go" in cmd_part:
            cmd = "Go"
        elif "back" in cmd_part:
            cmd = "Back"
        else:
            return None

        return experiment_time, index, cmd, fname
    except:
        return None

# ================== 실험시간별 그룹화 ==================
groups = defaultdict(list)

for f in os.listdir(SRC_DIR):
    if f.lower().endswith(".csv"):
        parsed = parse_filename(f)
        if parsed:
            exp_time, idx, cmd, fname = parsed
            groups[exp_time].append((idx, cmd, fname))

print(f" 실험시간 그룹 수: {len(groups)}")

# ================== 그룹별 처리 ==================
pair_count = 0

for exp_time, files in groups.items():
    # 인덱스 기준 정렬
    files.sort(key=lambda x: x[0])

    for i in range(1, len(files)):
        idx_prev, cmd_prev, f_prev = files[i - 1]
        idx_curr, cmd_curr, f_curr = files[i]

        if (
            cmd_curr == "Back"
            and cmd_prev == "Go"
            and idx_curr == idx_prev + 1
        ):
            shutil.copy(
                os.path.join(SRC_DIR, f_prev),
                os.path.join(DST_DIR, f_prev)
            )
            shutil.copy(
                os.path.join(SRC_DIR, f_curr),
                os.path.join(DST_DIR, f_curr)
            )

            pair_count += 1
            print(f" [{exp_time}] Go→Back: {idx_prev} → {idx_curr}")

print(f" 총 저장된 Go–Back 쌍: {pair_count}")
print("완료.")
