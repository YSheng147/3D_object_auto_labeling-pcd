# ==============================================================================
# 座標系定義 (Coordinate System Definitions)
# ==============================================================================
# - LiDAR 座標系 (原始點雲與標註框):
#   - X 軸: 車輛前進方向
#   - Y 軸: 車輛左側
#   - Z 軸: 垂直向上
#
# - Camera 座標系 (轉換後的點雲與視角):
#   - X 軸: 影像平面右側
#   - Y 軸: 影像平面下方
#   - Z 軸: 相機朝前方向
# ==============================================================================
import os
import pickle
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm

class SceneDataLoader:
    """
    資料格式：

    └── 240509
        └── highway_sunny_day
            └── 01
                ├──highway_sunny_day_2024-05-09-14-16-42 <- 給這一個
                    ├── image(原始圖片)
                        ├── 000000.jpg
                        ├── 000001.jpg
                        └── ...
                    ├── imu
                        ├── 000000.txt
                        ├── 000001.txt
                        └── ...
                    └── VLS128_pcdnpy
                        ├── 000000.pcd
                        ├── 000001.pcd
                        └── ...
                    └── 3d_label.pkl
            └── ... (剩下 9 個 timestamp 的資料夾)
        └── 內外參.txt
    """

    def __init__(self, base_folder):
        # --- 1. 檔案路徑設定 ---
        if not os.path.isdir(base_folder):
            raise FileNotFoundError(f"提供的基礎路徑不存在: {base_folder}")
        
        self.base_folder = base_folder # 儲存基礎路徑
        self.pointcloud_folder = os.path.join(self.base_folder, "VLS128_pcd")    # 點雲 路徑
        self.imu_folder = os.path.join(self.base_folder, "imu")                     # IMU 路徑
        self.image_folder = os.path.join(self.base_folder, "image")                 # 圖片 路徑
        self.box3D_file = os.path.join(self.base_folder, "3d_label.pkl")          # 3D框 路徑

        # 可能檔名設定
        self.imu_possible_names = ["id.txt", "id_imu.txt"]
        self.image_possible_names = ["id.jpg", "id.png", "id_image.jpg", "id_image.png"]

        # --- 2. Lidar to Camera 外參、內參 ---
        # TODO：未來這部分可以改成從檔案讀取，例如 self.lidar_to_camera_extrinsics = np.loadtxt(...)
        # self.lidar_to_camera_extrinsics = np.array([ 
        #     [0.037,  -0.999,  0.009,  0.0],
        #     [-0.094, -0.012, -0.996, -0.3],
        #     [0.995,   0.036, -0.094, -0.43],
        #     [0.0,     0.0,    0.0,    1.0]
        # ])

        # self.camera_intrinsics = np.array([
        #     [1418.667,  0.0,        640.0],
        #     [0.0,       1418.667,   360.0],
        #     [0.0,       0.0,        1.0]
        # ])

        self.lidar_to_camera_extrinsics = np.array([ 
            [  0.087505, -0.996160, 0.002753, -0.771377 ],
            [ -0.006834, -0.003364, -0.999971, 1.676000 -1.8 ],
            [  0.996141, 0.087484, -0.007102, 1.847415 ],
            [  0.0000000,  0.0000000,  0.0000000,  1.0000000 ]
        ])

        self.camera_intrinsics = np.array([
            [2596.0221329790797,    0     ,  1820.868392151553,],
            [0.000000, 2605.3715497870785,  1131.0899624075655],
            [0.000000, 0.000000, 1.000000]
        ])

        self.distortion_coeffs = np.array([-0.463575, 0.245606, -0.000168, -0.001956])
        
        # --- 3. 其他參數 ---
        # 3D框Z軸補償值(公尺)
        # 由於 3D 標註框的中心點位於地面，而 LiDAR 座標系原點位於感測器本身 (約離地 1.8 公尺)，
        # 因此需要將標註框向下平移，使其與點雲對齊。
        self.box3D_z_Correction = -1.8    

        # --- 4. 執行資料載入與預處理 ---
        # 取得所有 .pcd 檔名並排序
        self.pcd_files = sorted([f for f in os.listdir(self.pointcloud_folder) if f.endswith('.pcd')])

        if not self.pcd_files:
            raise FileNotFoundError(f"在 {self.pointcloud_folder} 中找不到任何 .pcd 檔案。")

        self._data_map = self._preprocess_and_map_data() # 執行核心的資料預處理函式
        self.image_height, self.image_width = self._get_default_image_size()

        
    def __len__(self):
        return len(self.pcd_files)

    def __getitem__(self, frame_index):
        """
        根據指定的索引，獲取一幀所需的所有資料。
        :param frame_index: 想要獲取資料的幀的索引值 (整數)。
        :return: 一個包含點雲路徑、3D邊界框和自我姿態的字典。
        """
        pcd_filename = self.pcd_files[frame_index]

        frame_specific_data = self._data_map.get(pcd_filename, {})

        return { 
            'pcd_filename': frame_specific_data.get('pcd_filename'),
            'point_cloud': frame_specific_data.get('point_cloud'),
            'boxes_lidar': frame_specific_data.get('boxes_lidar', np.array([])),
            'yaw_pose': frame_specific_data.get('yaw_pose', np.identity(4)),
            'imu_data': frame_specific_data.get('imu_data', np.array([])),
            'image': frame_specific_data.get('image', np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8))
        }

    def _preprocess_and_map_data(self):
        """
        執行所有資料的載入和對應。
        """
        temp_detection_dict = self._parse_3dlabel_data()

        data_map = {}
        
        for pcd_filename in tqdm(self.pcd_files, desc="處理資料讀取"):
            # 3d
            base_name = os.path.splitext(pcd_filename)[0] # 去掉副檔名
            frame_id = base_name.split('_')[0] # 取得frame ID
            frame_id_int = str(int(frame_id))

            boxes_data = temp_detection_dict.get(frame_id_int, {}).copy()
            boxes_data['pcd_filename'] = pcd_filename

            original_boxes = np.array(boxes_data.get('boxes_lidar', [])) # 取得 3D 邊界框
            if original_boxes.size > 0:
                original_boxes[:, 2] += self.box3D_z_Correction # 進行 Z 軸補償
                boxes_data['boxes_lidar'] = original_boxes
                
            # pcd
            pcd_path = os.path.join(self.pointcloud_folder, pcd_filename)
            pcd_legacy  = o3d.io.read_point_cloud(pcd_path)
            boxes_data['point_cloud'] = pcd_legacy

            # imu
            imu_data = []
            yaw_pose = np.identity(4)
            imu_file_path = self._find_file(self.imu_folder, frame_id, self.imu_possible_names)
            if imu_file_path: 
                imu_data = self._parse_imu_data(imu_file_path)
                if imu_data and len(imu_data) > 0:
                    first_data = imu_data[0]
                    if len(first_data) == 6:
                        # 格式 (1): orientation_x, orientation_y, orientation_z, angular_velocity_x, angular_velocity_y, angular_velocity_z
                        yaw = first_data[2]
                    elif len(first_data) == 11:
                        # 格式 (2): timestamp_ns,orientation_w,orientation_x,orientation_y,orientation_z,angular_velocity_x,angular_velocity_y,angular_velocity_z,linear_acceleration_x,linear_acceleration_y,linear_acceleration_z
                        yaw = first_data[4]
                    else:
                        print(f"⚠️ 無法解析此行資料: {first_data}")
                    # x是pitch、y是yaw、z是roll
                    yaw_pose = self._create_y_axis_rotation_matrix(yaw)  
            else:
                print(f"找不到對應的 IMU 檔案: {base_name}")

            boxes_data['yaw_pose'] = yaw_pose
            boxes_data['imu_data'] = imu_data

            # image
            image_file_path = self._find_file(self.image_folder, frame_id, self.image_possible_names)
            if image_file_path:
                image = cv2.imread(image_file_path)
                boxes_data['image'] = image
            else:
                print(f"警告：找不到對應照片 {pcd_filename}，使用黑背景代替。")

            data_map[pcd_filename] = boxes_data
        return data_map

    def _find_file(self, folder, id , possible_names):
        """
        找資料夾中的檔案
        :param folder: 尋找的路徑
        :param id: frame ID。
        :param possible_names: frame ID。
        :return: 找到的檔案路徑，或 None。
        """
        for name in possible_names:
            file_name = name.replace("id", str(id))
            path = os.path.join(folder, file_name)

            if os.path.exists(path):
                return path
        return None
# 3D Label 相關操作
    def _parse_3dlabel_data(self):
        """
        解析 3dlabel 檔案的資料
        :return: 所有幀 3D Label 資料
        """
        if not os.path.exists(self.box3D_file):
            print(f"⚠️ 檢測檔案未找到: {self.box3D_file}")
            return {pcd_file: {} for pcd_file in self.pcd_files}

        with open(self.box3D_file, 'rb') as f:
            detection_sets = pickle.load(f)
        
        temp_detection_dict = {frame_data.get('frame_id', ''): frame_data for frame_data in detection_sets}

        return temp_detection_dict

# IMU 相關操作
    def _parse_imu_data(self, imu_file_path):
        """
        解析 IMU 檔案的資料
        :param imu_file_path: IMU 檔案的路徑
        :return: 單一幀的imu data
        """
        try:
            with open(imu_file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    if "orientation_x" in lines[0]:  
                        lines = lines[1:]

                imu_data = []

                for line in lines:
                    parts = [float(x) for x in line.strip().split(',')]
                    if len(parts) == 6 or len(parts) == 11:
                        # 格式 (1): orientation_x, orientation_y, orientation_z, angular_velocity_x, angular_velocity_y, angular_velocity_z
                        # 格式 (2): timestamp_ns,orientation_w,orientation_x,orientation_y,orientation_z,angular_velocity_x,angular_velocity_y,angular_velocity_z,linear_acceleration_x,linear_acceleration_y,linear_acceleration_z
                        imu_data.append(parts)
                    else:
                        print(f"⚠️ 無法解析此行資料: {line.strip()}")  # 當資料格式不符時，跳過該行

                return imu_data
        except Exception as e:
            print(f"⚠️ 解析 IMU 檔案 {imu_file_path} 時發生錯誤: {e}")
            return None

    def _create_y_axis_rotation_matrix(self, angle_radians):
        """
        繞 Y 軸 (pitch) 建立一個 4x4 姿態矩陣
        :param angle_radians: 旋轉的角度 (弧度制)
        :return: 4x4 的 NumPy 旋轉矩陣
        """
        cos_a = np.cos(angle_radians)
        sin_a = np.sin(angle_radians)

        # 建立 3x3 的繞 Y 軸旋轉矩陣
        rotation_matrix = np.array([
            [cos_a,  0, sin_a],
            [0,      1, 0],
            [-sin_a, 0, cos_a]
        ])

        # 嵌入到 4x4 的單位矩陣中，形成姿態矩陣
        pose = np.identity(4)
        pose[:3, :3] = rotation_matrix
        return pose
# Image 相關操作
    def _get_default_image_size(self):
        """
        讀取第一幀的圖片以確定影片的預設尺寸
        :return: (高度, 寬度) 
        """
        first_frame_data = self._data_map.get(self.pcd_files[0], {})
        image = first_frame_data.get('image')

        if image is not None:
            return image.shape[0], image.shape[1]
        else:
            return (720, 1280)