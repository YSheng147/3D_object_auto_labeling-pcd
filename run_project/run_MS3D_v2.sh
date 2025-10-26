#!/bin/bash

# ex：/home/ys/MS3D/data/custom/2024-07-03/highway_cloudy_day
SOURCE_DIR="$1"

# 用絕對路徑
# 推論使用組合設定檔
MODEL_CFG_FILE="/home/ys/MS3D/data/custom/model_config.csv"
# 資料集設定檔
DATASET_CFG_FILE="/home/ys/MS3D/tools/cfgs/dataset_configs/custom_dataset_da.yaml"
# 融合設定檔
MS3D_CFG_FILE="/home/ys/MS3D/tools/cfgs/target_custom/label_generation/round1/cfgs/ps_config.yaml"
# 結果儲存路徑
RESULT_PATH="/home/ys/MS3D/tools/cfgs/target_custom/label_generation/round1/auto"

#-------------------------------
# 函式定義
#-------------------------------

check_path() {
    local path="$1"
    local type="$2"
    local check_write="$3"

    if [[ "$type" == "dir" && ! -d "$path" ]]; then
        echo "錯誤：'$path' 不存在或不是資料夾"
        exit 1
    elif [[ "$type" == "file" && ! -f "$path" ]]; then
        echo "錯誤：'$path' 不存在或不是檔案"
        exit 1
    fi
    if [[ ! -r "$path" ]]; then
        echo "錯誤：'$path' 無讀取權限"
        exit 1
    fi
    if [[ "$check_write" == "true" && ! -w "$path" ]]; then
        echo "錯誤：'$path' 無寫入權限"
        exit 1
    fi
}

#-------------------------------
# 主程式
#-------------------------------

# 檢查路徑、權限
check_path "$SOURCE_DIR" "dir"
check_path "$MODEL_CFG_FILE" "file" "true"
check_path "$DATASET_CFG_FILE" "file" "true"
check_path "$MS3D_CFG_FILE" "file" "true"
mkdir -p "$RESULT_PATH"
check_path "$RESULT_PATH" "dir" "true"

# 尋找點雲資料夾
find "$SOURCE_DIR" -type d -iname "*vls128*pcd*" | while read -r found_path; do
    echo "找到：$found_path"

    scene_dir="$(dirname "$found_path")"

    # 處理資料
    pc_path="$scene_dir/pointcloud/"
    if [ -d $pc_path ]; then
        echo -e "\t資料已處理：$scene_dir"
    else
        bash data_processing.sh $found_path
    fi
    # 改資料集設定檔
    sed -i "s|^\(DATA_PATH:\s*\).*|\1'$pc_path'|" "$DATASET_CFG_FILE"

    # dataset name
    dataset_name="$(basename $SOURCE_DIR)-$(basename $scene_dir)"
    model_list_cfg_file="${RESULT_PATH}/cfgs/ensemble_detections_${dataset_name}.txt"
    result_dir="${RESULT_PATH}/results/ensemble_detections_${dataset_name}"
    echo -e "\tDataset Name：${dataset_name}"
    # 創建資料夾、檔案
    > ${model_list_cfg_file}
    mkdir -p "$result_dir"

    pushd "../../tools"
    # 模型推論
    while IFS=',' read -r model_cfg model_pt veh ped cyc sweeps tta 
    do
        # 取得檔名
        basename="${model_pt##*/}"
        model_name="${basename%.pth}"

        echo -e -n "\tRunning Model：${model_name}\tSweeps：${sweeps}\tTTA：${tta}\t"
        
        result_path=$(python test.py --cfg_file ${model_cfg} \
                        --ckpt ${model_pt} \
                        --eval_tag "${dataset_name}_${model_name}-custom_s${sweeps}_tta${tta}" \
                        --target_dataset custom --sweeps ${sweeps} --batch_size 8 --use_tta ${tta} \
                        --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none \
                        2>&1 | grep "Predictions saved to" | awk '{print $7}')
        
        echo "${result_path},${veh},${ped},${cyc}" >> ${model_list_cfg_file}
        echo "#DONE#"
    done < <(tail -n +2 "$MODEL_CFG_FILE")


    # 改MS3D設定檔
    sed -i "s|^\(DETS_TXT:\s*\).*|\1'$model_list_cfg_file'|" $MS3D_CFG_FILE
    sed -i "s|^\(SAVE_DIR:\s*\).*|\1'$result_dir'|" $MS3D_CFG_FILE

    # run MS3D
    bash cfgs/target_custom/label_generation/round1/scripts/run_ms3d.sh
    cp "$result_dir/final_ps_dict_conv.pkl" "$scene_dir/3d_label.pkl"
    popd
    # 統計
    python class_statistics.py "$scene_dir/3d_label.pkl"
done