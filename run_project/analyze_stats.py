import os
import csv
from pathlib import Path
from collections import Counter

# --- 設定 ---
ROOT_DIRECTORY = Path('/data/elan')  
STATS_FILENAME = '3D_class_statistics.csv'
OUTPUT_FILENAME = 'summary_report.csv'
# ----------------

def parse_directory_name(dir_name):
    """
    從目錄名稱解析場景和日期。
    格式: [scene_name]_[YYYY-MM-DD-HH-MM-SS]
    """
    if len(dir_name) < 20: 
        print(f"[警告] 跳過格式錯誤的目錄: {dir_name}")
        return None, None

    date_time_str = dir_name[-19:]
    scene_name = dir_name[:-20]
    date = date_time_str[:10]
    
    if not (scene_name and date_time_str[4] == '-' and date_time_str[7] == '-'):
        print(f"[警告] 解析目錄名稱失敗: {dir_name}")
        return None, None
        
    return scene_name, date

def find_and_process_stats(root_dir):
    """
    第一階段：遍歷、解析、讀取資料。
    """
    print(f"[*] 正在掃描 {root_dir}...")
    all_records = []
    
    for stats_file in root_dir.rglob(f'**/{STATS_FILENAME}'):
        parent_dir_name = stats_file.parent.name
        
        scene, date = parse_directory_name(parent_dir_name)
        
        if not scene:
            continue 

        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader) 
                data_row = next(reader) 
                
                if not data_row:
                    print(f"[警告] 檔案為空: {stats_file}")
                    continue

                total_frames, vehicle, pedestrian, cyclist = data_row
                
                # 將所有資料暫存起來
                all_records.append({
                    'sence': scene,
                    'date': date,
                    'total_frames': total_frames,
                    'Vehicle': vehicle,
                    'Pedestrian': pedestrian,
                    'Cyclist': cyclist,
                    # --- 新增的程式碼 ---
                    # .resolve() 取得絕對路徑，確保它 "完整"
                    'full_path': str(stats_file.resolve())
                    # ---------------------
                })
                
        except Exception as e:
            print(f"[錯誤] 讀取檔案失敗 {stats_file}: {e}")
            
    return all_records

def write_summary_file(records, output_file):
    """
    第二階段：計算總數並寫入最終報告。
    """
    if not records:
        print("[!] 找不到任何資料，不產生檔案。")
        return

    scene_counts = Counter([r['sence'] for r in records])

    output_header = [
        'sence', 'total_sence', 'date', 'total_frames', 
        'Vehicle', 'Pedestrian', 'Cyclist', 
        'full_path' 
    ]

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=output_header)
            writer.writeheader()
            
            for record in records:
                output_row = {
                    **record, 
                    'total_sence': scene_counts[record['sence']] 
                }
                writer.writerow(output_row)
                
        print(f"\n[成功] 報告已產生: {output_file}")
        print(f"[*] 總共處理了 {len(records)} 筆紀錄。")

    except Exception as e:
        print(f"[致命錯誤] 無法寫入檔案 {output_file}: {e}")

# --- 執行 ---
if __name__ == "__main__":
    raw_data = find_and_process_stats(ROOT_DIRECTORY)
    write_summary_file(raw_data, OUTPUT_FILENAME)