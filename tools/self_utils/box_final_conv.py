from collections import Counter
import copy
import os
import numpy as np
from pcdet.utils import ms3d_utils
from pathlib import Path
import pickle
from pcdet.utils import ms3d_utils, box_utils 
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import torch
import tqdm
import yaml

# 類別ID到名稱的映射
CLASS_MAP = {
    1: 'Vehicle',
    2: 'Pedestrian',
    3: 'Cyclist'
}

CLASS_MAPPING = {
    'car': 'Vehicle',
    'truck': 'Vehicle',
    'bus': 'Vehicle',
    'Vehicle': 'Vehicle',
    'motorcycle': 'Cyclist',
    'bicycle': 'Cyclist',
    'Cyclist': 'Cyclist',
    'pedestrian': 'Pedestrian',
    'Pedestrian': 'Pedestrian'
}

def dataset_frame_filename(dataset):
    frame_file_map = {}
    for info in dataset.infos:
        frame_id = info['frame_id']
        file_name = info['file_name']
        
        frame_file_map[frame_id] = file_name
    return frame_file_map

def convert_data_format(dataset_name, dataset, data_dict: dict) -> list:
    """
    將 MS3D 偽標籤字典從 pkl 格式轉換為列表格式。

    Args:
        data_dict (dict): 輸入的 pkl 檔案內容，以 frame ID 為鍵。

    Returns:
        list: 包含每個影格信息的字典列表。
    """
    converted_list = []

    frame_filename_dict = dataset_frame_filename(dataset) if dataset_name == "CustomDataset" else {}

    for key, value in data_dict.items():
        gt_boxes = value.get('gt_boxes', np.array([]))

        names = []
        scores = []
        boxes_lidar = []
        pred_labels = []

        for box in gt_boxes:
            # box 格式: [cx, cy, cz, l, w, h, heading, class_id, score]
            class_id = int(box[7])

            # 只處理 CLASS_MAP 中定義的正類別ID
            if abs(class_id) in CLASS_MAP:
                names.append(CLASS_MAP[abs(class_id)])
                scores.append(box[8])
                boxes_lidar.append(box[0:7])  # [cx, cy, cz, l, w, h, heading]
                pred_labels.append(abs(class_id))
        if key in frame_filename_dict:
            file_name = frame_filename_dict[key]
        else:
            file_name = "N/A"

        new_frame_dict = {
            'name': np.array(names, dtype='<U10'),
            'score': np.array(scores, dtype=np.float32),
            'boxes_lidar': np.array(boxes_lidar, dtype=np.float32),
            'pred_labels': np.array(pred_labels),
            'frame_id': key,
            'file_name': file_name
        }
        converted_list.append(new_frame_dict)

    return converted_list

def iou_cal(cfg, gt_anno, dt_anno):
    gt_data = copy.deepcopy(gt_anno)
    dt_data = score_filter(dt_anno, 0.7)
    
    iou_thresholds = {
        'Vehicle': 0.7,
        'Pedestrian': 0.5,
        'Cyclist': 0.5
    }
    point_cloud_range = np.array(cfg.POINT_CLOUD_RANGE, dtype=np.float32)

    calculate_recall_for_dataset(gt_data, dt_data, iou_thresholds, point_cloud_range)

def score_filter(data, score_threshold = 0.7):
    # 準備一個新的 list 來存放篩選後的結果
    filtered_results = []

    # 歷遍每一幀的資料
    for frame_data in data:
        scores = frame_data['score']
        high_score_mask = scores > score_threshold
        # 如果沒有任何一個分數 > 0.7，可以選擇跳過此幀
        if not np.any(high_score_mask):
            continue
            
        # 使用這個遮罩來篩選所有相關的 numpy array
        filtered_names = frame_data['name'][high_score_mask]
        filtered_scores = frame_data['score'][high_score_mask]
        filtered_boxes = frame_data['boxes_lidar'][high_score_mask]
        filtered_labels = frame_data['pred_labels'][high_score_mask]

        # 將篩選後的結果儲存成一個新的 dictionary
        filtered_frame = {
            'name': filtered_names,
            'score': filtered_scores,
            'boxes_lidar': filtered_boxes,
            'pred_labels': filtered_labels,
            'frame_id': frame_data['frame_id'],
            'file_name': frame_data.get('file_name', 'N/A')
        }
        # 將處理好的這一幀加到最終結果中
        filtered_results.append(filtered_frame)
    return filtered_results

def get_box_name(dataset):
    class_names = dataset.class_names
    restructured_data = {}
    print("從 Dataset Loader 中提取並處理 Ground-Truth 資料...")
    for index in tqdm.tqdm(range(len(dataset))):
        data_dict = dataset[index]
        frame_id = data_dict['frame_id']
        gt_boxes = data_dict['gt_boxes']

        mapped_gt_names = []
        gt_labels = gt_boxes[:,7].astype(int)
        # 迭代每一個物件，進行類別對應
        for label in gt_labels:
            label_name = class_names[label-1]
            # 檢查該類別是否在我們的對應規則中
            if label_name in CLASS_MAPPING:
                mapped_gt_names.append(CLASS_MAPPING[label_name])
            else:
                mapped_gt_names.append(label_name)
        
        restructured_data[frame_id] = {
            'gt_boxes': gt_boxes[:,:7],
            'gt_names': mapped_gt_names
        }
    return restructured_data

def process_single_frame_for_recall(gt_boxes, gt_names, dt_boxes, dt_names, iou_thresholds):
    """
    處理單一幀的數據，返回該幀的 TP, GT, DT 計數以及 TP 的 IoU 總和。
    """
    unique_gt_classes = np.unique(gt_names)
    dt_names_mapping = [CLASS_MAPPING.get(name, name) for name in dt_names]
    
    frame_tp_counts = Counter()
    frame_gt_counts = Counter()
    # 新增：計算偵測到的物件數量
    frame_dt_counts = Counter(dt_names_mapping)
    # 新增：計算匹配成功的 IoU 總和，用於後續計算平均值
    frame_iou_sum = Counter()
    

    for class_name in unique_gt_classes:
        class_gt_mask = (np.array(gt_names) == class_name)
        class_dt_mask = (np.array(dt_names_mapping) == class_name)

        class_gt_boxes = gt_boxes[class_gt_mask]
        class_dt_boxes = dt_boxes[class_dt_mask]

        num_gt = len(class_gt_boxes)
        frame_gt_counts[class_name] = num_gt
        
        if num_gt == 0 or len(class_dt_boxes) == 0 or iou3d_nms_utils is None:
            continue

        iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(
            torch.from_numpy(class_gt_boxes[:, :7]).float().cuda(),
            torch.from_numpy(class_dt_boxes[:, :7]).float().cuda()
        ).cpu().numpy()
        num_dt = len(class_dt_boxes)
        dt_matched = np.zeros(num_dt, dtype=bool)
        
        for i in range(num_gt):
            iou_row = iou_matrix[i, :].copy() # 使用 .copy() 避免修改原矩陣
            iou_row[dt_matched] = -1
            
            if iou_row.size > 0:
                best_match_idx = np.argmax(iou_row)
                best_match_iou = iou_row[best_match_idx]

                if best_match_iou > iou_thresholds.get(class_name, 0.5):
                    frame_tp_counts[class_name] += 1
                    # 新增：累加成功匹配的 IoU 值
                    frame_iou_sum[class_name] += best_match_iou
                    dt_matched[best_match_idx] = True
                    
    return frame_tp_counts, frame_gt_counts, frame_dt_counts, frame_iou_sum

def calculate_recall_for_dataset(gt_data_all_frames, dt_data_list, iou_thresholds, point_cloud_range):
    """
    計算整個資料集的總召回率、平均IoU和偵測數量。
    """
    dt_data_all_frames_dict = {det.get('frame_id', det.get('lidar_path', f'unknown_{i}')): det for i, det in enumerate(dt_data_list)}

    total_tp_counts = Counter()
    total_gt_counts = Counter()
    total_dt_counts = Counter()
    total_iou_sum = Counter()

    # 遍歷所有幀
    for frame_id, gt_info in gt_data_all_frames.items():
        dt_info = dt_data_all_frames_dict.get(frame_id, {})

        # 獲取原始的 GT 資訊
        gt_boxes_raw = gt_info.get('gt_boxes', np.array([]))
        gt_names_raw = gt_info.get('gt_names', np.array([]))

        # *** 核心過濾步驟 ***
        if gt_boxes_raw.shape[0] > 0:
            # 根據 point_cloud_range 產生遮罩 (mask)
            mask = box_utils.mask_boxes_outside_range_numpy(gt_boxes_raw, point_cloud_range)
            # 應用遮罩
            gt_boxes_filtered = gt_boxes_raw[mask]
            gt_names_filtered = np.array(gt_names_raw)[mask]
        else:
            gt_boxes_filtered = gt_boxes_raw
            gt_names_filtered = gt_names_raw
        # *** 過濾結束 ***

        # 獲取單幀的計算結果 (現在有4個回傳值)
        frame_tp, frame_gt, frame_dt, frame_iou = process_single_frame_for_recall(
            gt_boxes=gt_boxes_filtered,
            gt_names=gt_names_filtered,
            dt_boxes=dt_info.get('boxes_lidar', np.array([])),
            dt_names=dt_info.get('name', np.array([])),
            iou_thresholds=iou_thresholds
        )
        
        # 累加每一幀的計數
        total_tp_counts.update(frame_tp)
        total_gt_counts.update(frame_gt)
        total_dt_counts.update(frame_dt)
        total_iou_sum.update(frame_iou)
        #break # 註解掉 break 才能計算所有幀

    # --- 計算並印出最終結果 ---
    all_class_names = sorted(total_gt_counts.keys())
    
    print("\n--- 總結果分析 (Result Analysis) ---")
    print(f"{'Class':<25} | {'Total TP':<10} | {'Total DT':<10} | {'Total GT':<10} | {'Recall':<10} | {'Avg IoU (TPs)':<15}")
    print("-" * 65)
    for class_name in all_class_names:
        tp = total_tp_counts[class_name]
        total_gt = total_gt_counts[class_name]
        total_dt = total_dt_counts[class_name]
        recall = tp / total_gt if total_gt > 0 else 0
        iou_sum = total_iou_sum[class_name]
        avg_iou = iou_sum / tp if tp > 0 else 0.0
        print(f"{class_name:<25} | {tp:<10} | {total_dt:<10} | {total_gt:<10} | {recall:<10.4f} | {avg_iou:<15.4f}")
    """
    print("\n--- 偵測結果分析 (Detection Analysis) ---")
    print(f"{'Class':<25} | {'Total DT':<10} | {'Avg IoU (TPs)':<15} | {'TP Count':<10}")
    print("-" * 75)
    for class_name in all_class_names:
        tp = total_tp_counts[class_name]
        total_dt = total_dt_counts[class_name]
        iou_sum = total_iou_sum[class_name]
        
        # 計算平均 IoU，注意分母不能為零
        avg_iou = iou_sum / tp if tp > 0 else 0.0
        
        print(f"{class_name:<25} | {total_dt:<10} | {avg_iou:<15.4f} | {tp:<10}")
    """

if __name__ == '__main__':
    None