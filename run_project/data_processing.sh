#!/bin/bash

# 指定來源資料夾
# /home/ys/MS3D/data/custom/citystreet_sunny_day_2025-04-21-11-32-20/vls128_pcd
SOURCE_DIR="$1"

# 目標資料夾
TARGET_DIR="$SOURCE_DIR/../pointcloud/sequences"

SEQ_DIR="$TARGET_DIR/sequence_0"
PCD_DIR="$SEQ_DIR/lidar"
IMAGESETS_DIR="$TARGET_DIR/../ImageSets"

# 建立目標資料夾
mkdir -p "$PCD_DIR"
mkdir -p "$IMAGESETS_DIR"

# 移動所有 .pcd 檔案
find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.pcd" -exec cp {} "$PCD_DIR" \;

ls "$TARGET_DIR" > "$IMAGESETS_DIR/val.txt"
ls "$TARGET_DIR" > "$IMAGESETS_DIR/train.txt"

echo -e "\t已將所有 pcd 檔案從移動到 $TARGET_DIR"

./generate_lidar_odom.sh "$TARGET_DIR"

pushd "../.."
python -W ignore -m pcdet.datasets.custom.custom_dataset create_infos "$TARGET_DIR/.."
popd

echo -e "\t資料處理完成"