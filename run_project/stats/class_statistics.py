import sys
import os
import pickle
import csv
from collections import Counter

# 統計的類別
CLASSES_TO_COUNT = ["Vehicle", "Pedestrian", "Cyclist"]
# 分數閾值： [Vehicle, Pedestrian, Cyclist]
SCORE_THRESHOLD = [0.7, 0.5, 0.5]
# CSV 欄位順序
CSV_FIELD_NAMES = ["total_frames"] + CLASSES_TO_COUNT

original_print = print

def print(*args, **kwargs):
    prefix = "\t\t"
    if args:
        args = (prefix + str(args[0]),) + args[1:]
    original_print(*args, **kwargs)

def append_to_csv(csv_path, row_data, field_names):
    try:
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerow(row_data)
            
        print(f"結果已寫入: {csv_path}")
        
    except IOError as e:
        print(f"寫入 CSV 失敗: {e}", file=sys.stderr)
        sys.exit(1)

def load_data(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"錯誤: 找不到 PKL 檔案: {pkl_path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError:
        print(f"錯誤: PKL 檔案已損壞: {pkl_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"讀取 PKL 檔案時發生未知錯誤: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print(f"使用方式: python {sys.argv[0]} <path_to_pkl>")
        sys.exit(1)
    pkl_path = sys.argv[1]
    
    data = load_data(pkl_path)

    # 資料夾名稱
    dir_path = os.path.dirname(pkl_path)
    csv_path = os.path.join(dir_path,"3D_class_statistics.csv")
    total_frames = len(data)

    # 計算
    name_counter = Counter()
    for frame in data:
        names = frame.get('name', [])
        scores = frame.get('score', [])
        
        # 確保 name 和 score 長度一致
        if len(names) != len(scores):
            print(f"警告: frame 中 name 和 score 長度不一致", file=sys.stderr)
            continue
        
        for name, score in zip(names, scores):
            # 只統計在 CLASSES_TO_COUNT 中的類別
            if name in CLASSES_TO_COUNT:
                class_idx = CLASSES_TO_COUNT.index(name)
                threshold = SCORE_THRESHOLD[class_idx]
                
                # 只有 score >= threshold 才計數
                if score >= threshold:
                    name_counter[name] += 1

    # csv
    report_row = {"total_frames": total_frames}
    for cls in CLASSES_TO_COUNT:
        report_row[cls] = name_counter.get(cls, 0)

    print(f"統計: {dir_path}")
    print(f"總幀數: {total_frames}\t", end='')
    for cls in CLASSES_TO_COUNT:
        print(f"{cls}: {report_row.get(cls, 0)}", end='')
    print()
    
    append_to_csv(csv_path, report_row, CSV_FIELD_NAMES)

if __name__ == "__main__":
    main()