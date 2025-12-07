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

def build_expected_path(model_cfg_path: str,model_pt_path: str, dataset_name: str, sweeps: str, tta: str) -> Path:
    """
    預測輸出路徑
    """
    # 從 model_cfg 取得路徑
    cfg_part = model_cfg_path.removeprefix('./')
    cfg_part = cfg_part.lstrip('../')
    cfg_part = cfg_part.removesuffix('.yaml')

    # 從 model_pt 取得 epoch_
    pt_basename = os.path.basename(model_pt_path)
    pt_filename_stem = os.path.splitext(pt_basename)[0]
    parts = pt_filename_stem.split('_')
    target_part = [part for part in parts if part.endswith('xyzt')]
    if target_part:
        pt_part = target_part[0].removesuffix('xyzt')
    magic_part = f"default/eval/epoch_{pt_part}/val" 
    
    # 資料集、sweeps、tta
    eval_tag_part = f"{dataset_name}-custom_s{sweeps}_tta{tta}"

    file_name = "result.pkl"

    expected_path = OUTPUT_DIR / cfg_part / magic_part / eval_tag_part / file_name
    
    return expected_path

# -------------------------------
# 主程式
# -------------------------------

def main():
    args = parse_arguments()

    source_dir = Path(args.source_dir)

    # 檢查路徑、權限
    check_path(source_dir, "dir")
    check_path(MODEL_CFG_FILE, "file", check_write=True)
    check_path(DATASET_CFG_FILE, "file", check_write=True)
    check_path(MS3D_CFG_FILE, "file", check_write=True)

    MODEL_RESULT_PATH.mkdir(parents=True, exist_ok=True)
    check_path(MODEL_RESULT_PATH, "dir", check_write=True)
    (MODEL_RESULT_PATH / "cfgs").mkdir(parents=True, exist_ok=True)

    # 找點雲資料夾
    print(f"在 {source_dir} 中掃描場景中...")
    found_paths = []
    for path in source_dir.rglob('*'):
        if not path.is_dir():
            continue
        dir_name_lower = path.name.lower()
        if 'vls128' in dir_name_lower and 'pcd' in dir_name_lower:
            found_paths.append(path)
    found_paths.sort()

    total_scenes = len(found_paths)
    if total_scenes == 0:
        print(f"警告：在 {source_dir} 中沒有找到任何 'vls128' 和 'pcd' 的資料夾", file=sys.stderr)
        sys.exit(0)

    print(f"掃描完成。找到 {total_scenes} 個場景")
    print("-" * 30)

    # 讀取使用模型
    with open(MODEL_CFG_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        model_list = [row for row in reader if row]
    total_models = len(model_list)
    if total_models == 0:
        print(f"\t警告：{MODEL_CFG_FILE} 中沒有找到任何模型設定。", file=sys.stderr)
        sys.exit(0)

    # 開始處理
    for i, found_path in enumerate(found_paths):
        scene_dir = found_path.parent
        print(f"開始處理場景 [{i + 1}/{total_scenes}]: {scene_dir.name}")

        # 處理資料
        pc_path = scene_dir / "pointcloud"
        pc_pkl_path = pc_path / "sequences/sequence_0/sequence_0.pkl"
        if not args.force_data and pc_pkl_path.is_file():
            print(f"\t資料已處理：{scene_dir}")
        else:
            run_command(["bash", "data_processing.sh", str(found_path)])

        # 改資料集設定檔
        modify_config_file(
            DATASET_CFG_FILE,
            "DATA_PATH",
            str(pc_path) 
        )

        # dataset name
        dataset_name = f"{args.extra_tag}-{scene_dir.name}"
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
                    expected_path = build_expected_path(model_cfg, model_pt, dataset_name, sweeps, tta)
                    result_path = ""
                    if not args.force_inference and expected_path.is_file():
                        print(f"\t{j+1}/{total_models}\tFound (Skipping)：{model_name}\tSweeps：{sweeps}\tTTA：{tta}\t", end="", flush=True)
                        result_path = str(expected_path)
                    else:
                        print(f"\t{j+1}/{total_models}\tRunning Model：{model_name}\tSweeps：{sweeps}\tTTA：{tta}\t", end="", flush=True)

                        cmd = [
                            "python", "test.py",
                            "--cfg_file", model_cfg,
                            "--ckpt", model_pt,
                            "--eval_tag", f"{dataset_name}-custom_s{sweeps}_tta{tta}",
                            "--target_dataset", "custom",
                            "--sweeps", sweeps,
                            "--batch_size", "8",
                            "--use_tta", tta,
                            "--set", "DATA_CONFIG_TAR.DATA_SPLIT.test", "train", "MODEL.POST_PROCESSING.EVAL_METRIC", "none"
                        ]
                    
                        result = run_command(cmd, cwd=TOOLS_DIR, capture_output=True)

                        result_path_stdout = parse_test_output(result.stdout or result.stderr)
                        if not result_path_stdout:
                            print(f"錯誤：在 {model_name} 的輸出中找不到 'Predictions saved to'")
                            continue
                        
                        if Path(result_path_stdout) != expected_path:
                            print(f"警告：路徑猜測錯誤！")
                            print(f"\t猜測：{expected_path}")
                            print(f"\t實際：{result_path_stdout}")

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

        run_command(["bash", MS3D_SCRIPT_PATH], cwd=TOOLS_DIR)

        # 複製結果
        try:
            final_pkl = model_result_dir / "final_ps_dict_conv.pkl"
            if args.result_dir:
                target_pkl = Path(args.result_dir) / "3d_label.pkl"
            else:
                target_pkl = scene_dir / "3d_label.pkl"
            check_path(final_pkl, "file")
            shutil.move(final_pkl, target_pkl)
            print(f"\t結果已移動到：{target_pkl}")
        except Exception as e:
            print(f"錯誤：移動結果 pkl 失敗: {e}", file=sys.stderr)

        # 統計
        run_command(["python", "class_statistics.py", str(target_pkl)])

    print("全部處理完畢。")

if __name__ == "__main__":
    main()