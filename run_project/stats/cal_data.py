import pickle
import math
from pathlib import Path
import sys

# 【配置】將你的列表轉換為明確的映射
# 永遠不要在邏輯代碼中依賴列表的順序，那是災難的源頭。
THRESHOLDS = {
    'Vehicle': 0.7,
    'Pedestrian': 0.5,
    'Cyclist': 0.5
}
TARGET_CLASSES = set(THRESHOLDS.keys())

def calculate_stats(root_dir):
    def init_record(val):
        return {
            'val': val, 
            'dims': (0, 0, 0), 
            'path': 'N/A', 
            'frame': 'N/A',
            'score': 0.0 # 順便記錄下這個極值的當前分數，方便Debug
        }

    stats = {
        cls: {
            'min_vol': init_record(float('inf')),
            'max_vol': init_record(float('-inf')),
            'max_dist': init_record(0.0),
            'count': 0,
            'skipped': 0 # 追踪有多少框因為分數被過濾掉了
        }
        for cls in TARGET_CLASSES
    }

    root = Path(root_dir)
    if not root.exists():
        print(f"Error: Directory '{root_dir}' does not exist.")
        return

    print(f"Scanning directory: {root.resolve()}")
    files = list(root.rglob('3d_label.pkl'))
    
    if not files:
        print(f"Error: No '3d_label.pkl' files found.")
        return

    print(f"Processing {len(files)} files with thresholds: {THRESHOLDS}...")

    for file_path in files:
        try:
            current_full_path = str(file_path.resolve())
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, list):
                data = [data]

            for frame in data:
                frame_id = frame.get('frame_id', 'Unknown')
                names = frame.get('name', [])
                boxes = frame.get('boxes_lidar', [])
                scores = frame.get('score', []) # 獲取分數
                
                # 防禦性編程：確保三者長度一致
                if not (len(names) == len(boxes) == len(scores)):
                    continue

                # 【關鍵修改】同時 Zip 三個數據流
                for label, score, box in zip(names, scores, boxes):
                    if label not in TARGET_CLASSES:
                        continue
                    
                    # 【核心過濾邏輯】
                    # 如果分數低於閾值，直接跳過 (continue)
                    # 這是最快的路徑，不要在下面做數學運算後才判斷
                    if score < THRESHOLDS[label]:
                        stats[label]['skipped'] += 1
                        continue
                    
                    # 通過閾值，開始計算
                    dx, dy, dz = box[3], box[4], box[5]
                    volume = dx * dy * dz
                    dist = math.hypot(box[0], box[1], box[2])
                    
                    s = stats[label]
                    s['count'] += 1

                    def update_record(record, new_val, new_dims):
                        record['val'] = new_val
                        record['dims'] = new_dims
                        record['path'] = current_full_path
                        record['frame'] = frame_id
                        record['score'] = score

                    if volume < s['min_vol']['val']:
                        update_record(s['min_vol'], volume, (dx, dy, dz))

                    if volume > s['max_vol']['val']:
                        update_record(s['max_vol'], volume, (dx, dy, dz))

                    if dist > s['max_dist']['val']:
                        update_record(s['max_dist'], dist, (dx, dy, dz))

        except Exception as e:
            print(f"Failed to read {file_path}: {e}")

    # 輸出部分
    print("\n" + "="*150)
    print(f"{'CLASS':<12} | {'METRIC':<12} | {'VALUE':>10} | {'SCORE':>6} | {'DIMS (dx, dy, dz)':<22} | {'FRAME ID':<15} | {'FULL PATH'}")
    print("="*150)
    
    for cls in sorted(TARGET_CLASSES):
        s = stats[cls]
        
        # 顯示過濾統計
        total_seen = s['count'] + s['skipped']
        print(f"[{cls}] Valid: {s['count']}, Skipped (Low Score): {s['skipped']}, Total: {total_seen}")
        
        if s['count'] == 0:
            print(f"{' ':12} | No valid samples found above threshold {THRESHOLDS[cls]}.")
            print("-" * 150)
            continue

        def print_row(metric_name, record):
            dims_str = f"[{record['dims'][0]:.2f}, {record['dims'][1]:.2f}, {record['dims'][2]:.2f}]"
            # 增加 score 顯示，讓你確認這些極值是高置信度的
            print(f"{' ':12} | {metric_name:<12} | {record['val']:>10.4f} | {record['score']:>6.4f} | {dims_str:<22} | {record['frame']:<15} | {record['path']}")

        print_row("Min Volume", s['min_vol'])
        print_row("Max Volume", s['max_vol'])
        print_row("Max Dist", s['max_dist'])
        print("-" * 150)

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    calculate_stats(target_dir)