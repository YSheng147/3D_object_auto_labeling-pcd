# coding: utf-8
"""
3D框轉2D框程式
讀取3D bounding boxes並轉換為2D bounding boxes，每一幀儲存為一個txt檔案
輸出格式：class_id x_center y_center width height (YOLO格式，標準化座標)
"""
import argparse
import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm


def check_numpy_to_torch(x):
    """
    輸入 NumPy 轉換為 PyTorch Tensor
    Args:
        x (Any)
    Returns:
        tuple:
            - torch.Tensor | Any
            - bool: True: numpy → torch, False: 沒有轉換
    """
    try:
        import torch
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float(), True
        return x, False
    except ImportError:
        # 如果沒有torch，直接用numpy實現
        return x, False


def rotate_points_along_z(points, angle):
    """
    根據 heading 將 box 繞Z軸旋轉
    Args:
        points: (B, N, 3 + C)
        angle: (B)，每個 box 的旋轉角度
    Returns:
        np.ndarray | Any: (B, N, 3 + C)
    """
    try:
        import torch
        points, is_numpy = check_numpy_to_torch(points)
        angle, _ = check_numpy_to_torch(angle)

        # 計算旋轉矩陣
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa,  sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        # 旋轉
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot.numpy() if is_numpy else points_rot
    except ImportError:
        # 純numpy實現
        if len(points.shape) == 3:
            B, N, C = points.shape
            points_rot = np.zeros_like(points)
            for i in range(B):
                cosa = np.cos(angle[i])
                sina = np.sin(angle[i])
                rot_matrix = np.array([
                    [cosa, sina, 0],
                    [-sina, cosa, 0],
                    [0, 0, 1]
                ])
                points_rot[i, :, 0:3] = (rot_matrix @ points[i, :, 0:3].T).T
                if C > 3:
                    points_rot[i, :, 3:] = points[i, :, 3:]
            return points_rot
        return points


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading]，(x, y, z) 是box中心點
    Returns:
        corners3d: (N, 8, 3)，每個box的8個角點座標
    """
    try:
        import torch
        boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]

        return corners3d.numpy() if is_numpy else corners3d
    except ImportError:
        # 純numpy實現
        N = boxes3d.shape[0]
        template = np.array([
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        ]) / 2
        
        corners3d = np.zeros((N, 8, 3))
        for i in range(N):
            corners3d[i] = template * boxes3d[i, 3:6]
        
        corners3d = rotate_points_along_z(corners3d, boxes3d[:, 6])
        corners3d += boxes3d[:, None, 0:3]
        
        return corners3d


def transform_points(points, transform_matrix):
    """
    用齊次座標轉換矩陣對 3D 點進行座標轉換
    Args:
        points: (N, 3)
        transform_matrix: (4, 4) 
    Returns:
        np.ndarray: (N, 3)
    """
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed_h = (transform_matrix @ points_h.T).T
    return points_transformed_h[:, :3]


def transform_boxes_to_corners(boxes3d, transform_matrix):
    """
    7-dof 的 box 轉換為轉換後的 8 個角點
    Args:
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading]
        transform_matrix: (4, 4) 
    Returns:
        ndarray: (N, 8, 3)，每個 box 轉換後的八個角點座標。  
    """
    if not isinstance(boxes3d, np.ndarray) or boxes3d.size == 0:
        return np.zeros((0, 8, 3))

    corners3d = boxes_to_corners_3d(boxes3d)
    num_boxes = corners3d.shape[0]
    corners_flat = corners3d.reshape(-1, 3)
    corners_transformed_flat = transform_points(corners_flat, transform_matrix)
    
    return corners_transformed_flat.reshape(num_boxes, 8, 3)


def project_points_to_image(points_3d, intrinsic_matrix, distortion_coeffs=None):
    """
    將 3D 點投影到 2D 影像平面
    Args:
        points_3d : (N, 3)
        intrinsic_matrix : (3, 3)
        distortion_coeffs: 畸變係數 (預設: None)
    Returns:
        tuple:
            - points_2d : (M, 2)，對應投影後的 2D 座標 (u, v)，只包含 z > 0 的可見點
            - valid_indices : (N,)，表示哪些點被保留 (z > 0)
    """
    valid_indices = points_3d[:, 2] > 0
    points_3d_visible = points_3d[valid_indices]

    if points_3d_visible.shape[0] == 0:
        return np.zeros((0, 2)), valid_indices
    
    if distortion_coeffs is not None:
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)
        points_2d, _ = cv2.projectPoints(points_3d_visible, rvec, tvec, intrinsic_matrix, distortion_coeffs)
        points_2d = points_2d.reshape(-1, 2)
    else:
        points_homogeneous = (intrinsic_matrix @ points_3d_visible.T).T
        points_2d = points_homogeneous[:, :2] / points_homogeneous[:, 2, np.newaxis]
    return points_2d, valid_indices


def is_box_in_image_bounds(corners_2d, image_width, image_height, min_overlap_ratio=0.5):
    """
    檢查2D框是否在圖片範圍內
    Args:
        corners_2d: (8, 2) 8個角點的2D座標
        image_width: 影像寬度
        image_height: 影像高度
        min_overlap_ratio: 最小重疊比例 (預設0.1，即至少10%的框要在圖片內)
    Returns:
        bool: True 表示框在圖片範圍內，False 表示框超出範圍太多
    """
    x_coords = corners_2d[:, 0]
    y_coords = corners_2d[:, 1]
    
    # 計算框的邊界
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # 計算與圖片邊界的交集
    x_min_clipped = max(0, x_min)
    x_max_clipped = min(image_width, x_max)
    y_min_clipped = max(0, y_min)
    y_max_clipped = min(image_height, y_max)
    
    # 檢查是否有交集
    if x_min_clipped >= x_max_clipped or y_min_clipped >= y_max_clipped:
        return False  # 完全在圖片外
    
    # 計算原始框面積和交集面積
    original_area = (x_max - x_min) * (y_max - y_min)
    intersection_area = (x_max_clipped - x_min_clipped) * (y_max_clipped - y_min_clipped)
    
    # 檢查重疊比例
    overlap_ratio = intersection_area / original_area if original_area > 0 else 0
    
    return overlap_ratio >= min_overlap_ratio


def convert_to_yolo_format(corners_2d, image_width, image_height):
    """
    將 2D 角點轉換為 YOLO 格式 (歸一化的中心座標和寬高)
    Args:
        corners_2d: (8, 2) 8個角點的2D座標
        image_width: 影像寬度
        image_height: 影像高度
    Returns:
        tuple: (x_center, y_center, width, height) 歸一化座標
    """
    x_coords = corners_2d[:, 0]
    y_coords = corners_2d[:, 1]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # 計算中心點和寬高
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # 歸一化
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = width / image_width
    height_norm = height / image_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def load_3d_boxes_from_pkl(pkl_file, score_threshold=[0.7, 0.5, 0.5], z_correction=0.0):
    """
    從 pkl 檔案載入 3D 框資料
    Args:
        pkl_file: 3d_label.pkl 檔案路徑
        score_threshold: 各類別的分數閾值 [vehicle, pedestrian, cyclist]
        z_correction: Z軸補償值
    Returns:
        dict: frame_id -> {boxes_lidar, pred_labels, frame_id}
    """
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"找不到 3D 框檔案: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        detection_sets = pickle.load(f)
    
    # 整理成 frame_id -> data 的字典
    result = {}
    for frame_data in detection_sets:
        frame_id = frame_data.get('frame_id', '')
        
        # 篩選分數
        scores = np.array(frame_data.get('score', []))
        pred_labels = frame_data.get('pred_labels', [])
        boxes_lidar = frame_data.get('boxes_lidar', [])
        
        if len(scores) > 0 and len(boxes_lidar) > 0:
            score_mask = np.array([
                scores[i] >= score_threshold[pred_labels[i]-1]
                for i in range(len(scores))
            ])
            
            # 篩選後的資料
            filtered_boxes = np.array(boxes_lidar)[score_mask]
            filtered_labels = [l for i, l in enumerate(pred_labels) if score_mask[i]]
            
            # Z軸補償
            if len(filtered_boxes) > 0:
                filtered_boxes[:, 2] += z_correction
            
            result[frame_id] = {
                'boxes_lidar': filtered_boxes,
                'pred_labels': filtered_labels,
                'frame_id': frame_id
            }
        else:
            result[frame_id] = {
                'boxes_lidar': np.array([]),
                'pred_labels': [],
                'frame_id': frame_id
            }
    
    return result


def convert_3d_to_2d_boxes(pkl_file, output_folder, 
                          lidar_to_camera_extrinsics=None,
                          camera_intrinsics=None,
                          distortion_coeffs=None,
                          image_width=1280, 
                          image_height=720,
                          score_threshold=[0.7, 0.5, 0.5],
                          z_correction=0.0,
                          use_undistort=False):
    """
    將3D框轉換為2D框並儲存為txt檔案
    Args:
        pkl_file: 3d_label.pkl 檔案路徑
        output_folder: 輸出資料夾路徑
        lidar_to_camera_extrinsics: LiDAR到相機的外參矩陣 (4x4)
        camera_intrinsics: 相機內參矩陣 (3x3)
        distortion_coeffs: 畸變係數
        image_width: 影像寬度
        image_height: 影像高度
        score_threshold: 各類別的分數閾值
        z_correction: Z軸補償值
        use_undistort: 是否使用去畸變校正
    """
    # 預設參數
    if lidar_to_camera_extrinsics is None:
        lidar_to_camera_extrinsics = np.array([
            [-0.00693070, -0.99997562,  0.00083871,  0.000000],
            [-0.12013684,  0.000000,   -0.99275732, -0.200000],
            [ 0.99273312, -0.00698126, -0.12013391,  0.000000],
            [ 0.0,         0.0,         0.0,         1.0     ]
        ], dtype=np.float32)
    
    if camera_intrinsics is None:
        camera_intrinsics = np.array([
            [2453.4,    0.0, 1933.1],
            [   0.0, 2466.0, 1109.8],
            [   0.0,    0.0,    1.0]
        ], dtype=np.float32)
    
    if distortion_coeffs is None:
        distortion_coeffs = np.array([-0.4681, 0.1777, 0.0, 0.0, -0.0290], dtype=np.float32)
    
    # 建立輸出資料夾
    os.makedirs(output_folder, exist_ok=True)
    
    # 載入3D框資料
    print(f"載入3D框資料: {pkl_file}")
    boxes_dict = load_3d_boxes_from_pkl(pkl_file, score_threshold, z_correction)
    
    # 如果使用去畸變，計算新的相機內參
    K = camera_intrinsics
    if use_undistort:
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, distortion_coeffs, 
                                                    (image_width, image_height), 0, 
                                                    (image_width, image_height))
    else:
        new_K = K
    
    print(f"開始轉換 {len(boxes_dict)} 幀的3D框到2D框...")
    
    # 處理每一幀
    for frame_id, frame_data in tqdm(boxes_dict.items(), desc="轉換進度"):
        original_boxes = frame_data['boxes_lidar']
        pred_labels = frame_data['pred_labels']
        
        # 建立輸出檔案名稱
        output_file = os.path.join(output_folder, f"{frame_id.zfill(6)}.txt")
        
        # 如果該幀沒有檢測框，建立空檔案
        if original_boxes.size == 0:
            with open(output_file, 'w') as f:
                pass  # 建立空檔案
            continue
        
        # 轉換3D框到相機座標系
        corners_in_camera = transform_boxes_to_corners(original_boxes, lidar_to_camera_extrinsics)
        
        # 儲存2D框資訊
        box_2d_list = []
        
        for box_idx, box_corners in enumerate(corners_in_camera):
            # 計算box中心點，只要中心在相機前方就嘗試投影
            box_center = box_corners.mean(axis=0)
            if box_center[2] <= 0:
                continue
            
            # 投影到2D影像平面（只投影 z > 0 的角點）
            dist_to_use = None if use_undistort else distortion_coeffs
            projected_corners, valid_mask = project_points_to_image(box_corners, new_K, dist_to_use)
            
            # 至少需要4個可見角點才能形成有效的2D框
            if projected_corners.shape[0] < 4:
                continue
            
            # 檢查框是否在圖片範圍內
            if not is_box_in_image_bounds(projected_corners, image_width, image_height):
                continue
            
            # 轉換為YOLO格式
            x_center, y_center, width, height = convert_to_yolo_format(
                projected_corners, image_width, image_height
            )
            
            # 獲取類別ID (pred_labels是從1開始，轉換為從0開始)
            # (0=vehicle, 1=pedestrian, 2=cyclist)
            if isinstance(pred_labels, list):
                class_id = pred_labels[box_idx] - 1
            else:
                class_id = int(pred_labels[box_idx]) - 1
            
            # 加入列表
            box_2d_list.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # 寫入檔案
        with open(output_file, 'w') as f:
            f.write('\n'.join(box_2d_list))
    
    print(f"\n完成！2D框已儲存至: {output_folder}")
    print(f"共處理 {len(boxes_dict)} 幀")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D框轉2D框工具 - 直接從pkl檔案讀取，無需載入圖片")
    parser.add_argument("--pkl_file", type=str, required=True, 
                        help="3d_label.pkl 檔案路徑")
    parser.add_argument("--output_folder", type=str, required=True, 
                        help="輸出2D框txt檔案的資料夾路徑")
    
    # 影像參數
    parser.add_argument("--image_width", type=int, default=1280, 
                        help="影像寬度 (預設: 1280)")
    parser.add_argument("--image_height", type=int, default=720, 
                        help="影像高度 (預設: 720)")
    
    # 3D框處理參數
    parser.add_argument("--z_correction", type=float, default=0.0,
                        help="3D框Z軸補償值 (預設: -1.8)")
    parser.add_argument("--score_threshold", type=float, nargs=3, default=[0.7, 0.5, 0.5],
                        help="分數閾值 [vehicle, pedestrian, cyclist] (預設: 0.7 0.5 0.5)")
    
    # 相機參數
    parser.add_argument("--use_undistort", action="store_true", 
                        help="是否使用去畸變校正 (預設: False)")
    
    args = parser.parse_args()
    
    # 執行轉換
    try:
        convert_3d_to_2d_boxes(
            pkl_file=args.pkl_file,
            output_folder=args.output_folder,
            image_width=args.image_width,
            image_height=args.image_height,
            score_threshold=args.score_threshold,
            z_correction=args.z_correction,
            use_undistort=args.use_undistort
        )
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        exit(1)
    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
