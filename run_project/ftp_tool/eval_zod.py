import pickle
import numpy as np
from shapely.geometry import Polygon
import argparse
import os

# 設定 IoU 閾值
IOU_THRESHOLDS = {
    'Vehicle': 0.5,
    'Pedestrian': 0.5,
    'Cyclist': 0.5
}

TARGET_CLASSES = ['Vehicle', 'Pedestrian', 'Cyclist']

def get_rotated_corners(box):
    """將 [x, y, z, l, w, h, yaw] 轉為 BEV 四個角點"""
    x, y, z, l, w, h, yaw = box
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners_local = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
    corners_global = (R @ corners_local.T).T + np.array([x, y])
    return corners_global

def calculate_bev_iou(box_a, box_b):
    poly_a = Polygon(get_rotated_corners(box_a))
    poly_b = Polygon(get_rotated_corners(box_b))
    
    if not poly_a.is_valid or not poly_b.is_valid: return 0.0
    if not poly_a.intersects(poly_b): return 0.0

    inter_area = poly_a.intersection(poly_b).area
    union_area = poly_a.area + poly_b.area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def reorganize_by_frame(data_list):
    data_dict = {}
    for entry in data_list:
        raw_id = entry['frame_id']
        
        # --- ID 格式統一化 ---
        try:
            # 轉 int 去除前導零，再轉 str
            frame_id = str(int(raw_id))
        except ValueError:
            frame_id = str(raw_id).strip()
        # -------------------

        filtered_boxes = []
        filtered_names = []
        filtered_scores = []
        
        scores = entry.get('score', [1.0] * len(entry['name']))
        
        for name, box, score in zip(entry['name'], entry['boxes_lidar'], scores):
            if name in TARGET_CLASSES:
                filtered_boxes.append(box)
                filtered_names.append(name)
                filtered_scores.append(score)
                
        data_dict[frame_id] = {
            'boxes': np.array(filtered_boxes),
            'names': np.array(filtered_names),
            'scores': np.array(filtered_scores)
        }
    return data_dict

def evaluate(base_path):
    gt_path = os.path.join(base_path, 'gt_label.pkl')
    pred_path = os.path.join(base_path, '3d_label_v2.pkl')

    print(f"Loading from: {base_path}")
    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        print(f"錯誤: 在 {base_path} 找不到指定的 .pkl 檔案。")
        return

    gts = reorganize_by_frame(load_pkl(gt_path))
    preds = reorganize_by_frame(load_pkl(pred_path))
    
    metrics = {cls: {'GT_Count': 0, 'TP': 0, 'FP': 0, 'Total_IoU': 0.0, 'Matched_Count': 0} for cls in TARGET_CLASSES}
    
    common_frames = set(gts.keys()) & set(preds.keys())
    print(f"GT frames: {len(gts)}, Pred frames: {len(preds)}")
    print(f"Evaluating on {len(common_frames)} common frames...")

    if len(common_frames) == 0:
        print("警告: 仍然沒有匹配到任何 Frame，請檢查 ID 格式。")
        return

    for fid in common_frames:
        gt_data = gts[fid]
        pred_data = preds[fid]
        
        for cls in TARGET_CLASSES:
            gt_indices = np.where(gt_data['names'] == cls)[0]
            pred_indices = np.where(pred_data['names'] == cls)[0]
            
            cls_gt_boxes = gt_data['boxes'][gt_indices]
            cls_pred_boxes = pred_data['boxes'][pred_indices]
            cls_pred_scores = pred_data['scores'][pred_indices]
            
            # 累加 GT 數量
            metrics[cls]['GT_Count'] += len(cls_gt_boxes)
            
            sorted_idx = np.argsort(cls_pred_scores)[::-1]
            cls_pred_boxes = cls_pred_boxes[sorted_idx]
            
            gt_matched = [False] * len(cls_gt_boxes)
            threshold = IOU_THRESHOLDS[cls]
            
            for p_box in cls_pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for i, g_box in enumerate(cls_gt_boxes):
                    if gt_matched[i]: continue
                    iou = calculate_bev_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= threshold and best_gt_idx != -1:
                    metrics[cls]['TP'] += 1
                    metrics[cls]['Total_IoU'] += best_iou
                    metrics[cls]['Matched_Count'] += 1
                    gt_matched[best_gt_idx] = True
                else:
                    metrics[cls]['FP'] += 1

    print("\n" + "="*75)
    # 修改表頭，加入 Recall
    print(f"{'Class':<12} | {'GT':<6} | {'TP':<6} | {'FP':<6} | {'Recall':<8} | {'mIoU (TP)':<10}")
    print("-" * 75)
    for cls in TARGET_CLASSES:
        gt_count = metrics[cls]['GT_Count']
        tp = metrics[cls]['TP']
        fp = metrics[cls]['FP']
        
        # 計算 Recall
        recall = tp / gt_count if gt_count > 0 else 0.0
        
        avg_iou = metrics[cls]['Total_IoU'] / metrics[cls]['Matched_Count'] if metrics[cls]['Matched_Count'] > 0 else 0
        
        print(f"{cls:<12} | {gt_count:<6} | {tp:<6} | {fp:<6} | {recall:.4f}   | {avg_iou:.4f}")
    print("="*75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='./')
    args = parser.parse_args()
    evaluate(args.base_path)