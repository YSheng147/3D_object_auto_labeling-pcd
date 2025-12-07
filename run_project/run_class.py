#!/usr/bin/env python3

import sys
import os
import subprocess
import csv
from pathlib import Path
import ruamel.yaml
import argparse
import shutil

# -------------------------------
# 設定
# -------------------------------
# ex: /home/ys/MS3D/data/custom/2024-07-03/highway_cloudy_day

SCRIPT_FILE_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE_PATH.parent
# 專案目錄
BASE_PATH = SCRIPT_DIR.parent

MODEL_CFG_FILE = BASE_PATH / "run_project/all_model_config.csv"
DATASET_CFG_FILE = BASE_PATH / "tools/cfgs/dataset_configs/custom_dataset_da.yaml"
MS3D_CFG_FILE = BASE_PATH / "tools/cfgs/target_custom/label_generation/round1/cfgs/ps_config.yaml"
MODEL_RESULT_PATH = BASE_PATH / "tools/cfgs/target_custom/label_generation/round1/auto"
TOOLS_DIR = BASE_PATH / "tools"
OUTPUT_DIR = BASE_PATH / "output"
MS3D_SCRIPT_PATH = TOOLS_DIR / "cfgs/target_custom/label_generation/round1/scripts/run_ms3d.sh"

# -------------------------------
# 函式定義
# -------------------------------

def parse_arguments():
    """
    輸入參數
    """
    parser = argparse.ArgumentParser(
        description="處理 3D 點雲資料並執行模型推論",
        usage=f"python {Path(__file__).name} --source_dir /path/to/source_dir [--force-data] [--force-inference]"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="包含點雲的來源資料夾。ex: /home/ys/MS3D/data/custom/2024-07-03/highway_cloudy_day"
    )
    args = parser.parse_args()
    return args
def run_command(cmd_list: list, cwd: Path = None, capture_output: bool = False):
    """
    執行子程序
    """
    #print(f"\t執行中：{' '.join(cmd_list)}")

    # 建立 subprocess 的參數
    kwargs = {
        "cwd": cwd,
        "check": True,
        "text": True
    }

    if capture_output:
        kwargs["capture_output"] = True
    else:
        kwargs["stdout"] = sys.stdout
        kwargs["stderr"] = sys.stderr

    try:
        result = subprocess.run(cmd_list, **kwargs)
        return result
    except subprocess.CalledProcessError as e:
        print(f"錯誤：執行命令失敗: {' '.join(cmd_list)}", file=sys.stderr)
        if capture_output:
            print(f"STDOUT:\n{e.stdout}", file=sys.stderr)
            print(f"STDERR:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"錯誤：找不到命令 '{e.filename}'", file=sys.stderr)
        sys.exit(1)

# -------------------------------
# 主程式
# -------------------------------

def main():
    args = parse_arguments()

    source_dir = Path(args.source_dir)

    print(f"在 {source_dir} 中掃描場景中...")
    found_paths = list(source_dir.rglob('3d_label.pkl'))
    found_paths.sort()

    total_scenes = len(found_paths)
    if total_scenes == 0:
        print(f"警告：在 {source_dir} 中沒有找到任何 'vls128' 和 'pcd' 的資料夾", file=sys.stderr)
        sys.exit(0)

    print(f"掃描完成。找到 {total_scenes} 個場景")
    print("-" * 30)

    # 開始處理
    for i, found_path in enumerate(found_paths):
        print(f"開始處理場景 [{i + 1}/{total_scenes}]: {found_path}")
        # 統計
        run_command(["python", "class_statistics.py", str(found_path)])

    print("全部處理完畢。")

if __name__ == "__main__":
    main()