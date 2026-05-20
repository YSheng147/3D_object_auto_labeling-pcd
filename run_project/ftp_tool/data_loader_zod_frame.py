import os
import pickle
import numpy as np
import open3d as o3d
import cv2
import ast
import re
from tqdm import tqdm

class SceneDataLoader:
    def __init__(self, base_folder):
        # --- 1. 檔案路徑設定 ---
        if not os.path.isdir(base_folder):
            raise FileNotFoundError(f"提供的基礎路徑不存在: {base_folder}")
        
        self.base_folder = base_folder
        self.pointcloud_folder = os.path.join(self.base_folder, "VLS128_pcd")
        self.imu_folder = os.path.join(self.base_folder, "imu")
        self.image_folder = os.path.join(self.base_folder, "image")
        self.calib_folder = os.path.join(self.base_folder, "calib")
        
        # 定義 檢測框 (Pred) 與 真值框 (GT) 的路徑
        self.pred_box3D_file = os.path.join(self.base_folder, "3d_label_v2.pkl")
        self.gt_box3D_file = os.path.join(self.base_folder, "gt_label.pkl")  # 新增 GT 檔案路徑

        self.imu_possible_names = ["id.txt", "id_imu.txt"]
        self.image_possible_names = ["id.jpg", "id.png", "id_image.jpg", "id_image.png"]
        self.calib_possible_names = ["id.txt", "id_calib.txt"] 

        self.distortion_coeffs = np.array([-0.463575, 0.245606, -0.000168, -0.001956])
        self.box3D_z_Correction = -1.8   

        # --- 4. 執行資料載入與預處理 ---
        self.pcd_files = sorted([f for f in os.listdir(self.pointcloud_folder) if f.endswith('.pcd')])

        if not self.pcd_files:
            raise FileNotFoundError(f"在 {self.pointcloud_folder} 中找不到任何 .pcd 檔案。")

        self._data_map = self._preprocess_and_map_data()
        self.image_height, self.image_width = self._get_default_image_size()

    def __len__(self):
        return len(self.pcd_files)

    def __getitem__(self, frame_index):
        pcd_filename = self.pcd_files[frame_index]
        frame_specific_data = self._data_map.get(pcd_filename, {})

        return { 
            'pcd_filename': frame_specific_data.get('pcd_filename'),
            'point_cloud': frame_specific_data.get('point_cloud'),
            'boxes_lidar': frame_specific_data.get('boxes_lidar', np.array([])), # 預測框
            'boxes_gt': frame_specific_data.get('boxes_gt', np.array([])),       # 新增: GT框
            'yaw_pose': frame_specific_data.get('yaw_pose', np.identity(4)),
            'imu_data': frame_specific_data.get('imu_data', np.array([])),
            'image': frame_specific_data.get('image', np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)),
            'calib': frame_specific_data.get('calib', {})
        }

    def _preprocess_and_map_data(self):
        # 分別讀取 Pred 和 GT 的字典
        pred_dict = self._parse_label_file(self.pred_box3D_file)
        gt_dict = self._parse_label_file(self.gt_box3D_file)

        # ==========================================
        # 1. 預先讀取全域 calib.txt (Fallback 機制)
        # ==========================================
        global_calib_file = os.path.join(self.base_folder, "calib.txt")
        global_calib_data = {}
        
        if os.path.exists(global_calib_file):
            print(f"發現全域校正檔: {global_calib_file}，若找不到單幀校正將使用此檔。")
            global_calib_data = self._parse_calib_data(global_calib_file)
        else:
            print(f"⚠️ 未發現全域校正檔 (calib.txt)，若無單幀校正資料將使用單位矩陣。")

        data_map = {}
        target_gt_classes = ['Vehicle', 'Pedestrian', 'Cyclist']

        for pcd_filename in tqdm(self.pcd_files, desc="處理資料讀取"):
            # 1. 解析檔名取得 Frame ID
            base_name = os.path.splitext(pcd_filename)[0]   
            raw_id = base_name.split('_')[0]                
            
            candidate_ids = [raw_id]
            if raw_id.isdigit():
                candidate_ids.append(str(int(raw_id)))
            
            def find_data_in_dict(source_dict, candidates):
                for cid in candidates:
                    if cid in source_dict:
                        return source_dict[cid]
                return {} 

            frame_data = {}
            frame_data['pcd_filename'] = pcd_filename

            # 2. 處理 Pred Boxes
            pred_entry = find_data_in_dict(pred_dict, candidate_ids)
            pred_boxes = np.array(pred_entry.get('boxes_lidar', []))
            if pred_boxes.size > 0:
                pred_boxes[:, 2] += self.box3D_z_Correction 
            frame_data['boxes_lidar'] = pred_boxes

            # 3. 處理 GT Boxes (含類別過濾)
            gt_entry = find_data_in_dict(gt_dict, candidate_ids)
            all_gt_boxes = np.array(gt_entry.get('boxes_lidar', []))
            all_gt_names = np.array(gt_entry.get('name', [])) 

            filtered_gt_boxes = np.empty((0, 7)) 
            if all_gt_boxes.size > 0 and all_gt_names.shape[0] == all_gt_boxes.shape[0]:
                mask = np.isin(all_gt_names, target_gt_classes)
                filtered_gt_boxes = all_gt_boxes[mask]
            
            frame_data['boxes_gt'] = filtered_gt_boxes
                
            # 4. Point Cloud
            pcd_path = os.path.join(self.pointcloud_folder, pcd_filename)
            pcd_legacy  = o3d.io.read_point_cloud(pcd_path)
            frame_data['point_cloud'] = pcd_legacy

            # 5. IMU
            imu_data = []
            yaw_pose = np.identity(4)
            imu_file_path = self._find_file(self.imu_folder, raw_id, self.imu_possible_names)
            if imu_file_path: 
                imu_data = self._parse_imu_data(imu_file_path)
                if imu_data and len(imu_data) > 0:
                    first_data = imu_data[0]
                    yaw = first_data[2] if len(first_data) == 6 else first_data[4]
                    yaw_pose = self._create_y_axis_rotation_matrix(yaw)  
            
            frame_data['yaw_pose'] = yaw_pose
            frame_data['imu_data'] = imu_data

            # 6. Image
            image_file_path = self._find_file(self.image_folder, raw_id, self.image_possible_names)
            if image_file_path:
                image = cv2.imread(image_file_path)
                frame_data['image'] = image
            else:
                frame_data['image'] = None

            # ==========================================
            # 7. Calibration (優先找單幀，找不到就用全域)
            # ==========================================
            calib_file_path = self._find_file(self.calib_folder, raw_id, self.calib_possible_names)
            
            if calib_file_path:
                # 情況 A: 找到該幀專屬的 calib 檔
                frame_data['calib'] = self._parse_calib_data(calib_file_path)
            else:
                # 情況 B: 找不到專屬檔 (或 calib 資料夾不存在)，使用全域 calib.txt
                # 使用 copy() 確保不會因為參照同一個字典而意外修改到全域變數
                frame_data['calib'] = global_calib_data.copy()

            data_map[pcd_filename] = frame_data
            
        return data_map

    def _find_file(self, folder, id , possible_names):
        if not os.path.exists(folder):
            return None
        for name in possible_names:
            file_name = name.replace("id", str(id))
            path = os.path.join(folder, file_name)
            if os.path.exists(path):
                return path
        return None

    # --- Parser 區域 ---

    def _parse_label_file(self, file_path):
        """
        通用解析 label pkl 檔案
        """
        if not os.path.exists(file_path):
            print(f"⚠️ 標註檔案未找到 (將略過): {file_path}")
            return {} # 若檔案不存在，回傳空字典，不讓程式崩潰

        try:
            with open(file_path, 'rb') as f:
                detection_sets = pickle.load(f)
            # 轉換為 {frame_id: data} 的字典格式
            return {frame_data.get('frame_id', ''): frame_data for frame_data in detection_sets}
        except Exception as e:
            print(f"⚠️ 讀取標註檔案錯誤 {file_path}: {e}")
            return {}

    # (其餘 _parse_calib_data, _parse_imu_data 等函式保持不變)
    def _parse_calib_data(self, calib_path):
        # ... (保持原樣) ...
        calib_dict = {'intrinsic': np.eye(3), 'extrinsic_lidar_to_camera': np.eye(4)}
        try:
            with open(calib_path, 'r', encoding='utf-8') as f:
                content = f.read()
            patterns = {
                'intrinsic': (r"Camera Intrinsic \(K\):.*?(\[\[.*?\]\])", 3),
                'extrinsic_lidar_to_camera': (r"Extrinsic LiDAR to Camera \(Tr\):.*?(\[\[.*?\]\])", 4)
            }
            for key, (pattern, dim) in patterns.items():
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    matrix_list = ast.literal_eval(match.group(1))
                    calib_dict[key] = np.array(matrix_list, dtype=np.float64)
        except Exception:
            pass
        return calib_dict

    def _parse_imu_data(self, imu_file_path):
        # ... (保持原樣) ...
        try:
            with open(imu_file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0 and "orientation_x" in lines[0]: lines = lines[1:]
                imu_data = []
                for line in lines:
                    parts = [float(x) for x in line.strip().split(',')]
                    if len(parts) in [6, 11]: imu_data.append(parts)
                return imu_data
        except: return None

    def _create_y_axis_rotation_matrix(self, angle_radians):
        # ... (保持原樣) ...
        c, s = np.cos(angle_radians), np.sin(angle_radians)
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

    def _get_default_image_size(self):
        first_frame_data = self._data_map.get(self.pcd_files[0], {})
        image = first_frame_data.get('image')
        return (image.shape[0], image.shape[1]) if image is not None else (720, 1280)