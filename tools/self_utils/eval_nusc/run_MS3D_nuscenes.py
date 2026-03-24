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

MODEL_CFG_FILE = BASE_PATH / "run_project/waymo_model_config.csv"
DATASET_CFG_FILE = BASE_PATH / "tools/cfgs/dataset_configs/nuscenes_dataset_da.yaml"
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
        "--result_dir",
        type=str,
        help="輸出pkl檔的資料夾"
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="強制重新處理資料"
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="強制重新執行模型推論"
    )
    parser.add_argument(
        "--extra_tag",
        type=str,
        default="custom",
        help="Extra tag for dataset name"
    )
    args = parser.parse_args()
    return args

def check_path(path: Path, ptype: str, check_write: bool = False):
    """
    檢查路徑
    """
    if not path.exists():
        print(f"錯誤：'{path}' 不存在", file=sys.stderr)
        sys.exit(1)
    
    if ptype == "dir" and not path.is_dir():
        print(f"錯誤：'{path}' 不是資料夾", file=sys.stderr)
        sys.exit(1)
    elif ptype == "file" and not path.is_file():
        print(f"錯誤：'{path}' 不是檔案", file=sys.stderr)
        sys.exit(1)

    if not os.access(path, os.R_OK):
        print(f"錯誤：'{path}' 無讀取權限", file=sys.stderr)
        sys.exit(1)

    if check_write and not os.access(path, os.W_OK):
        print(f"錯誤：'{path}' 無寫入權限", file=sys.stderr)
        sys.exit(1)

def modify_config_file(file_path: Path, key: str, value: str):
    """
    更改設定檔
    """
    check_path(file_path, "file", check_write=True)
    yaml = ruamel.yaml.YAML()

    yaml.preserve_quotes = True # 保留引號風格
    try:
        with open(file_path, 'r') as f:
            content = yaml.load(f)
        if key not in content:
            print(f"警告：在 {file_path} 中沒有找到頂層鍵 '{key}'。將會新增它。", file=sys.stderr)
        content[key] = value 
        
        with open(file_path, 'w') as f:
            yaml.dump(content, f)

    except yaml.YAMLError as e:
        print(f"錯誤：解析 YAML 檔案 '{file_path}' 失敗: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"錯誤：修改設定檔 '{file_path}' 失敗: {e}", file=sys.stderr)
        sys.exit(1)

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

def parse_test_output(stdout: str) -> str:
    """
    取出儲存路徑
    """
    marker = "Predictions saved to: "

    for line in stdout.splitlines():
        if marker in line:
            try:
                path_str = line.split(marker)[1]
                path_str = path_str.strip()
                return path_str
            except IndexError:
                print(f"錯誤：找到 'Predictions saved to' 但無法解析路徑: {line}", file=sys.stderr)
                return ""
            except Exception as e:
                print(f"錯誤：解析時發生未知錯誤: {e} on line: {line}", file=sys.stderr)
                return ""
    return ""

# -------------------------------
# 主程式
# -------------------------------

def main():
    args = parse_arguments()

    # 檢查路徑、權限
    check_path(MODEL_CFG_FILE, "file", check_write=True)
    check_path(DATASET_CFG_FILE, "file", check_write=True)
    check_path(MS3D_CFG_FILE, "file", check_write=True)

    MODEL_RESULT_PATH.mkdir(parents=True, exist_ok=True)
    check_path(MODEL_RESULT_PATH, "dir", check_write=True)
    (MODEL_RESULT_PATH / "cfgs").mkdir(parents=True, exist_ok=True)


    # 讀取使用模型
    with open(MODEL_CFG_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        model_list = [row for row in reader if row]
    total_models = len(model_list)
    if total_models == 0:
        print(f"\t警告：{MODEL_CFG_FILE} 中沒有找到任何模型設定。", file=sys.stderr)
        sys.exit(0)

    # dataset name
    dataset_name = f"{args.extra_tag}"
    model_list_cfg_file = MODEL_RESULT_PATH / "cfgs" / f"ensemble_detections_{dataset_name}.txt"
    model_result_dir = MODEL_RESULT_PATH / "results" / f"ensemble_detections_{dataset_name}"
    print(f"\tDataset Name：{dataset_name}")

    # 創建資料夾、檔案
    model_list_cfg_file.write_text("")
    model_result_dir.mkdir(parents=True, exist_ok=True)

    # 模型推論
    try:
        print(f"\t共 {total_models} 個模型設定。開始推論...")
        with open(model_list_cfg_file, 'a') as out_f:
            for j, row in enumerate(model_list):
                if not row: continue
                model_cfg, model_pt, veh, ped, cyc, sweeps, tta = row
                model_name = Path(model_pt).stem
                print(f"\t{j+1}/{total_models}\tRunning Model：{model_name}\tSweeps：{sweeps}\tTTA：{tta}\t", end="", flush=True)

                cmd = [
                    "python", "test.py",
                    "--cfg_file", model_cfg,
                    "--ckpt", model_pt,
                    "--eval_tag", f"{dataset_name}-nusc_s{sweeps}_tta{tta}",
                    "--target_dataset", "nuscenes",
                    "--sweeps", sweeps,
                    "--use_tta", tta,
                    "--set", "DATA_CONFIG_TAR.DATA_SPLIT.test", "train", "MODEL.POST_PROCESSING.EVAL_METRIC", "none"
                ]
            
                # 直接使用 subprocess.run 來捕捉 NotImplementedError
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=TOOLS_DIR,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                except subprocess.CalledProcessError as e:
                    # 檢查是否為 NotImplementedError (nuScenes evaluation 未實作)
                    if "NotImplementedError" in (e.stderr or ""):
                        # 忽略此錯誤，因為 result.pkl 已經成功產生
                        print("(評估步驟跳過)", end=" ")
                        # 建立一個類似的物件來解析輸出
                        class FakeResult:
                            def __init__(self, stdout, stderr):
                                self.stdout = stdout
                                self.stderr = stderr
                        result = FakeResult(e.stdout, e.stderr)
                    else:
                        # 其他錯誤則報告並繼續
                        print(f"錯誤：{e}")
                        print(f"STDERR: {e.stderr}")
                        continue

                result_path_stdout = parse_test_output(result.stdout or result.stderr)
                if not result_path_stdout:
                    print(f"錯誤：在 {model_name} 的輸出中找不到 'Predictions saved to'")
                    continue

                result_path = result_path_stdout
                
                out_f.write(f"{result_path},{veh},{ped},{cyc}\n")
                print(f"Result：{result_path}\t#DONE#")
                #break

    except FileNotFoundError:
        print(f"錯誤：找不到模型設定檔 {MODEL_CFG_FILE}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"錯誤：處理模型推論失敗: {e}", file=sys.stderr)
        sys.exit(1)

    # 改MS3D設定檔
    modify_config_file(
        MS3D_CFG_FILE,
        "DETS_TXT",
        str(model_list_cfg_file)
    )
    modify_config_file(
        MS3D_CFG_FILE,
        "SAVE_DIR",
        str(model_result_dir)
    )

    print("全部處理完畢。")

if __name__ == "__main__":
    main()