import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 🔠 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 경로 설정 =====
input_folder  = r'C:\Users\PREMA\Desktop\최종가공기'   # CSV/엑셀 폴더
output_folder = r'C:\Users\PREMA\Desktop\최종가공기\Image'     # 이미지 저장 폴더
os.makedirs(output_folder, exist_ok=True)

# ===== 색상 및 범례 이름 =====
label_colors = {'Go': 'red', 'Hold': 'green', 'Back': 'blue'}
label_names  = {'Go': '공구 전진',  'Hold': '공구이송 중단',  'Back': '공구 후진'}

# ===== 메인 로직 =====
# CSV와 XLSX 둘 다 읽기
data_files = sorted(glob.glob(os.path.join(input_folder, '*.csv')) +
                    glob.glob(os.path.join(input_folder, '*.xlsx')))
total_saved = 0

for file_path in data_files:
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, engine='python')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, engine='python', encoding='cp949')

    if df.shape[1] < 4:  # 최소 4열 필요 (Time, Voltage, Label, Depth)
        print(f"[!] {os.path.basename(file_path)}: 열(<4)이라 건너뜀")
        continue

    # 열 이름 정리
    df.columns = [str(c).strip() for c in df.columns]
    time_col, signal_col, label_col, depth_col = df.columns[:4]

    # 데이터 준비
    df['Time'] = df[time_col]
    df['Voltage'] = df[signal_col]
    df['Label'] = df[label_col].astype(str).str.strip()
    df['Depth'] = df[depth_col]
    df = df.dropna(subset=['Time', 'Voltage', 'Label', 'Depth']).reset_index(drop=True)

    if df.empty:
        print(f"[!] {os.path.basename(file_path)}: 유효 데이터 없음")
        continue

    # 500행 단위로 분할 (필요하면 n=1500으로 변경)
    n = 500
    num_segments = len(df) // n

    for seg_idx in range(num_segments):
        segment = df.iloc[seg_idx*n:(seg_idx+1)*n]

        # === 파일명 생성용 Depth 첫 값 ===
        first_depth = str(segment['Depth'].iloc[0])
        safe_depth = first_depth.replace('.', 'p').replace('-', 'm')  # 안전한 파일명 변환
        save_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_Depth_{safe_depth}_seg{seg_idx+1}.jpg"
        save_path = os.path.join(output_folder, save_name)

        plt.figure(figsize=(8, 4), dpi=300)  # dpi=300: 논문용 고해상도

        times   = segment['Time'].tolist()
        volts   = segment['Voltage'].tolist()
        labels  = segment['Label'].tolist()

        start_idx = 0
        used_labels = set()

        # 라벨별 색상으로 구간 분리
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                partial_time    = times[start_idx:i+1]
                partial_voltage = volts[start_idx:i+1]
                label           = labels[i - 1]
                color           = label_colors.get(label, 'black')
                legend_name     = label_names.get(label, label)

                if label not in used_labels:
                    plt.plot(partial_time, partial_voltage, color=color,
                             label=legend_name, linewidth=0.8)
                    used_labels.add(label)
                else:
                    plt.plot(partial_time, partial_voltage, color=color, linewidth=0.8)

                start_idx = i

        # 마지막 구간
        if start_idx < len(labels):
            label = labels[-1]
            partial_time    = times[start_idx:]
            partial_voltage = volts[start_idx:]
            color       = label_colors.get(label, 'black')
            legend_name = label_names.get(label, label)

            if label not in used_labels:
                plt.plot(partial_time, partial_voltage, color=color,
                         label=legend_name, linewidth=0.8)
                used_labels.add(label)
            else:
                plt.plot(partial_time, partial_voltage, color=color, linewidth=0.8)

        # ===== 그래프 꾸미기 (논문용) =====
        plt.title(f'{os.path.basename(file_path)} - Depth={first_depth}', fontsize=10)
        plt.xlabel('Index', fontsize=9)
        plt.ylabel('Voltage [V]', fontsize=9)
        plt.ylim(0, 7)
        plt.xticks([])
        plt.yticks(fontsize=8)
        plt.tight_layout(pad=0.3)  # 여백 최소화
        if used_labels:
            plt.legend(loc='upper right', fontsize=8, frameon=False)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[+] 저장됨: {save_name} (행 {seg_idx*n}~{(seg_idx+1)*n})")
        total_saved += 1

print(f"\n[✓] 완료! 총 {total_saved}개의 이미지가 저장되었습니다. (폴더: {output_folder})")
