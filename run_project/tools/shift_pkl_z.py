#!/usr/bin/env python3
"""
將已儲存的 3d_label pkl 檔案 z 座標還原回感測器座標系。

適用情境：
    舊版 pipeline 輸出的 pkl 是地面座標系（z+1.8），
    新版已在輸出時還原，此工具用來補轉換舊檔案。

用法：
    # 單一檔案（覆蓋原檔）
    python shift_pkl_z.py --input /path/to/3d_label_v2.pkl

    # 單一檔案（輸出到新路徑）
    python shift_pkl_z.py --input /path/to/3d_label_v2.pkl --output /path/to/3d_label_v2_fixed.pkl

    # 批次處理整個目錄下所有 3d_label*.pkl
    python shift_pkl_z.py --input /path/to/data_dir

    # 指定不同的偏移量（預設 -1.8）
    python shift_pkl_z.py --input /path/to/3d_label_v2.pkl --shift -1.8
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def shift_pkl(input_path: Path, output_path: Path, shift: float):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        print(f"  [跳過] 格式不符（非 list）：{input_path}", file=sys.stderr)
        return False

    modified = 0
    for frame in data:
        boxes = frame.get('boxes_lidar')
        if boxes is None or len(boxes) == 0:
            continue
        boxes = np.array(boxes, dtype=np.float32)
        boxes[:, 2] += shift
        frame['boxes_lidar'] = boxes
        modified += 1

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"  z {'+' if shift >= 0 else ''}{shift:.2f}m，共 {modified} 幀有框 → {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="調整 pkl 標籤檔的 z 座標偏移")
    parser.add_argument("--input", type=str, required=True,
                        help="pkl 檔案路徑，或包含 pkl 的目錄")
    parser.add_argument("--output", type=str, default=None,
                        help="輸出路徑（單檔模式）；目錄模式下忽略此參數，直接覆蓋原檔")
    parser.add_argument("--shift", type=float, default=-1.8,
                        help="z 軸偏移量，預設 -1.8（地面座標 → 感測器座標）")
    parser.add_argument("--pattern", type=str, default="3d_label*.pkl",
                        help="目錄模式下搜尋的檔名 pattern，預設 '3d_label*.pkl'")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        output_path = Path(args.output) if args.output else input_path
        print(f"處理：{input_path}")
        shift_pkl(input_path, output_path, args.shift)

    elif input_path.is_dir():
        files = sorted(input_path.rglob(args.pattern))
        if not files:
            print(f"警告：在 {input_path} 中找不到符合 '{args.pattern}' 的檔案", file=sys.stderr)
            sys.exit(0)
        print(f"找到 {len(files)} 個檔案，開始批次處理...")
        for f in files:
            print(f"處理：{f}")
            shift_pkl(f, f, args.shift)
        print(f"\n完成，共處理 {len(files)} 個檔案。")

    else:
        print(f"錯誤：'{input_path}' 不存在", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
