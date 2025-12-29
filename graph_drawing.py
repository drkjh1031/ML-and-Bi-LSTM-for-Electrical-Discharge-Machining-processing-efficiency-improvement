import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#한글 폰트
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

input_folder  = r'C:\Users\drkjh\Desktop\temp\Dataset'  
output_folder = r'C:\Users\drkjh\Desktop\temp\Image'    
os.makedirs(output_folder, exist_ok=True)

label_colors = {'G': 'red', 'H': 'green', 'B': 'blue'}
label_names  = {'G': 'Go',  'H': 'Hold',  'B': 'Back'}

def parse_time_column(series):
    base = datetime(2000, 1, 1)

    #datetime으로 자르기
    try:
        dt = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        if dt.notna().all():
            return dt
    except Exception:
        pass

    #안되면 초 단위로 파싱
    try:
        vals = pd.to_numeric(series, errors='coerce')
        if vals.notna().all():
            return pd.to_datetime([base + timedelta(seconds=float(s)) for s in vals])
    except Exception:
        pass

    #HH:MM:SS(.sss)로 파싱
    def parse_one(x):
        s = str(x).strip()
        if not s or s.lower() == 'nan':
            return pd.NaT
        parts = s.split(':')
        try:
            if len(parts) == 2:
                # MM:SS(.sss)
                m = float(parts[0])
                sec = float(parts[1])
                return base + timedelta(minutes=m, seconds=sec)
            elif len(parts) == 3:
                # HH:MM:SS(.sss)
                h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
                return base + timedelta(hours=h, minutes=m, seconds=sec)
        except Exception:
            return pd.NaT
        return pd.NaT

    parsed = series.map(parse_one)
    return pd.to_datetime(parsed)

csv_files = sorted(glob.glob(os.path.join(input_folder, '*.csv')))
segment_count = 1
total_saved = 0

if not csv_files:
    print(f"폴더에 csv파일 없음: {input_folder}")

for file_path in csv_files:
    try:
        df = pd.read_csv(file_path, engine='python')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, engine='python', encoding='cp949')

    if df.shape[1] < 3:
        print(f"[!] {os.path.basename(file_path)}: 데이터크기 3열 이하 건너뜀")
        continue

    df.columns = [str(c).strip() for c in df.columns]
    time_col    = df.columns[0]
    signal_col  = df.columns[1]
    label_col   = df.columns[2]

    #시간 단위로 파싱
    try:
        df['Time'] = parse_time_column(df[time_col])
    except Exception as e:
        print(f"[!] {os.path.basename(file_path)} 시간 파싱 실패: {e}")
        continue

    # 신호/라벨 복사
    df['Voltage'] = df[signal_col]
    df['Label']   = df[label_col].astype(str).str.strip()
    # NaN 제거
    df = df.dropna(subset=['Time', 'Voltage', 'Label'])
    if df.empty:
        print(f"[!] {os.path.basename(file_path)}: 유효 데이터 없음")
        continue

    df = df.sort_values('Time').reset_index(drop=True)
    start_time = df['Time'].min()
    end_time   = df['Time'].max()
    current_time = start_time
    base_name = os.path.basename(file_path)

    while current_time < end_time:
        next_time = current_time + pd.Timedelta(seconds=10)
        segment = df[(df['Time'] >= current_time) & (df['Time'] < next_time)]

        if not segment.empty:
            plt.figure(figsize=(8, 4))
            #command마다 색 변경
            times   = segment['Time'].tolist()
            volts   = segment['Voltage'].tolist()
            labels  = segment['Label'].tolist()

            start_idx = 0
            used_labels = set()

            for i in range(1, len(labels)):
                if labels[i] != labels[i - 1]:
                    partial_time    = times[start_idx:i+1]
                    partial_voltage = volts[start_idx:i+1]
                    label           = labels[i - 1]
                    color           = label_colors.get(label, 'black')
                    legend_name     = label_names.get(label, label)

                    if label not in used_labels:
                        plt.plot(partial_time, partial_voltage, color=color, label=legend_name)
                        used_labels.add(label)
                    else:
                        plt.plot(partial_time, partial_voltage, color=color)

                    start_idx = i

            #마지막 덩어리
            if start_idx < len(labels):
                label = labels[-1]
                partial_time    = times[start_idx:]
                partial_voltage = volts[start_idx:]
                color       = label_colors.get(label, 'black')
                legend_name = label_names.get(label, label)

                if label not in used_labels:
                    plt.plot(partial_time, partial_voltage, color=color, label=legend_name)
                    used_labels.add(label)
                else:
                    plt.plot(partial_time, partial_voltage, color=color)

            #그래프 설정
            plt.title(f'{base_name} - 구간 {segment_count} | {current_time.time()}~{next_time.time()}')
            plt.xlabel('Time')
            plt.ylabel('Voltage')
            #y축 0~7값 고정
            plt.ylim(0, 7)          
            plt.xticks([])          
            plt.tight_layout()
            if used_labels:
                plt.legend(loc='upper right')

            
            save_name = f"{segment_count:03d}.jpg"
            save_path = os.path.join(output_folder, save_name)
            plt.savefig(save_path)
            plt.close()

            print(f"저장됨: {save_name} ({current_time} ~ {next_time})")
            total_saved += 1
            segment_count += 1

        current_time = next_time

print(f"\n완료!!! 총 {total_saved}개의 이미지가 저장되었습니다. (폴더: {output_folder})")
