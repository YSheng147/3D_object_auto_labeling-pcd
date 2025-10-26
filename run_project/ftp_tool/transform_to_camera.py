# coding: utf-8
import argparse
import os
import subprocess
import numpy as np
import open3d as o3d
import cv2
import torch
import shutil
import matplotlib.cm as cm
from data_loader import SceneDataLoader

# ==============================================================================
# 核心配置區 (CONFIG) - 全域視覺化設定
# ==============================================================================

# [X, Y, Z] -> [前後, 左右, 上下]
# 設定 3D 視窗中攝影機（眼睛）的位置
VIEW_EYE_POSITION = np.array([0.0, 10.0, 20.0])
# 設定 3D 視窗中攝影機注視的目標點位置
VIEW_CENTER_TARGET = np.array([0.0, 5.0, -10.0])
# 設定 3D 視窗中攝影機的「上方」向量
VIEW_UP_VECTOR = np.array([0.0, 1.0, 0.0])

# ==============================================================================
# 核心幾何運算函式 (Core Geometry Functions)
# ==============================================================================

def check_numpy_to_torch(x):
    """
    輸入 NumPy 轉換為 PyTorch Tensor

    Args:
        x (Any)
    Returns:
        tuple:
            - torch.Tensor | Any
            - bool: 
                True: numpy → torch 
                False: 沒有轉換
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
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
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    # 將 box 長寬高套到模板上
    # 長寬高 (N, 8, 3) * 模板 (1, 8, 3) → box (N, 8, 3)
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    # 根據 heading 將 box 繞Z軸旋轉
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    # 將 box 轉到 box 中心上
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def transform_points(points, transform_matrix):
    """
    用齊次座標轉換矩陣對 3D 點進行座標轉換

    Args:
        points: (N, 3)
        transform_matrix: (4, 4) 
    Returns:
        np.ndarray: (N, 3)
    """
    # 三維座標補成四維
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
    # 檢查輸入是否為空
    if not isinstance(boxes3d, np.ndarray) or boxes3d.size == 0:
        return np.zeros((0, 8, 3))

    # 將 7-dof boxes 轉為 (N, 8, 3) 的角點
    corners3d = boxes_to_corners_3d(boxes3d)

    # 對角點進行座標轉換
    num_boxes = corners3d.shape[0]
    corners_flat = corners3d.reshape(-1, 3)
    corners_transformed_flat = transform_points(corners_flat, transform_matrix)
    
    # 將角點重新塑形回 (N, 8, 3)
    return corners_transformed_flat.reshape(num_boxes, 8, 3)

def project_points_to_image(points_3d, intrinsic_matrix):
    """
    將 3D 點投影到 2D 影像平面

    Args:
        points_3d : (N, 3)
        intrinsic_matrix : (3, 3)
    Returns:
        tuple:
            - points_2d : (M, 2)，對應投影後的 2D 座標 (u, v)，只包含 z > 0 的可見點
            - valid_indices : (N,)，表示哪些點被保留 (z > 0)
    """
    # 取出z大於零的點
    valid_indices = points_3d[:, 2] > 0
    points_3d_visible = points_3d[valid_indices]

    if points_3d_visible.shape[0] == 0:
        return np.zeros((0, 2)), valid_indices
    
    # 3D點投影到相機座標
    points_homogeneous = (intrinsic_matrix @ points_3d_visible.T).T
    # 取得 2D 影像平面座標
    points_2d = points_homogeneous[:, :2] / points_homogeneous[:, 2, np.newaxis]
    return points_2d, valid_indices

def create_lineset_from_corners(corners, color=[1, 0, 0]):
    """
    從 3D 角點建立 Open3D LineSet
    
    args:
        corners : (N, 8, 3)，box的8個角點
        color : [R,G,B]，lineset顏色
    """
    if corners.size == 0:
        return o3d.geometry.LineSet()
        
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
             [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    
    all_points = []
    all_lines = []
    start_idx = 0
    for corner_set in corners:
        all_points.extend(corner_set)
        all_lines.extend([[i + start_idx, j + start_idx] for i, j in edges])
        start_idx += 8
        
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(all_points)),
        lines=o3d.utility.Vector2iVector(np.array(all_lines)),
    )
    line_set.paint_uniform_color(color)
    return line_set

# ==============================================================================
# 顯示相關函式
# ==============================================================================

def set_camera_view(vis, mode, pose):
    """
    控制攝影機視角

    Args:
        vis : Open3D 視覺化物件
        mode : 視角模式，'vehicle' 表示車載模式，其他為靜態模式
        pose : 4x4 姿態矩陣，描述車輛的旋轉和位置
    """
    view_ctl = vis.get_view_control()
    cam_params = view_ctl.convert_to_pinhole_camera_parameters()

    # 將靜態視角向量應用車輛的旋轉
    if mode == 'vehicle':
        rotation_matrix = pose[:3, :3]

        eye = rotation_matrix @ VIEW_EYE_POSITION
        center = rotation_matrix @ VIEW_CENTER_TARGET
        up = rotation_matrix @ VIEW_UP_VECTOR
    # 退回到靜態模式
    else: 
        eye = VIEW_EYE_POSITION
        center = VIEW_CENTER_TARGET
        up = VIEW_UP_VECTOR


    # --- 計算 LookAt 矩陣 ---
    # 前進方向向量 (center - eye)
    forward = center - eye
    forward /= np.linalg.norm(forward)

    # 右向量 (forward 與 up 的叉積)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)

    # 上向量 (right 與 forward 的叉積)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)

    # 構建相機的外部參數矩陣並應用更新
    extrinsic = np.identity(4)
    extrinsic[0, :3] = right
    extrinsic[1, :3] = new_up
    extrinsic[2, :3] = -forward
    extrinsic[:3, 3] = eye
    
    cam_params.extrinsic = extrinsic
    view_ctl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

def update_geometries(vis, state, camera_mode="vehicle"):
    """
    o3d 繪製 3D 俯視圖(單幀)

    Args:
        vis : Open3D 視覺化物件
        state : 狀態控制
        camera_mode :
    """
    # 1. 讀資料
    frame_idx = state.current_index
    frame_data = state.data_loader[frame_idx]

    pcd_legacy = frame_data['point_cloud']
    original_boxes = frame_data['boxes_lidar']
    yaw_pose = frame_data['yaw_pose'] 
    pcd_filename = frame_data['pcd_filename'] 

    print(f"顯示幀: {frame_idx} / {state.total_frames - 1}\t檢測框形狀: {original_boxes.shape} - {pcd_filename}", end="\r")
    if original_boxes.size == 0:
        print("-> 警告: 此幀沒有檢測框。")
    
    # 2. 執行座標轉換
    points_in_camera = transform_points(np.asarray(pcd_legacy.points), state.lidar_to_camera_extrinsics)
    corners_in_camera = transform_boxes_to_corners(original_boxes, state.lidar_to_camera_extrinsics)
    
    # 3. 準備視覺化
    pcd_camera = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_in_camera))
    #TODO: 改以下code
    # pcd_camera.colors = pcd_legacy.colors
    # --- 開始：根據 Z 軸高度上色 ---
    z_values = points_in_camera[:, 2] # 提取所有點的 Z 值
    
    # 1. 定義您關心的高度範圍 (例如 -2.0米 到 3.0米)
    #    這有助於將顏色"拉伸"到您感興趣的區域
    z_min = -100
    z_max = 100
    
    # 2. 將 Z 值正規化到 [0, 1] 範圍
    norm_z = (z_values - z_min) / (z_max - z_min)
    norm_z = np.clip(norm_z, 0.0, 1.0) # 將超出範圍的值限制在 0 或 1
    
    # 3. 使用 matplotlib 的 colormap (例如 'jet' 或 'viridis')
    #    cmap 會返回 (R, G, B, A)，我們只需要前三個 (RGB)
    cmap = cm.jet 
    colors = cmap(norm_z)[:, :3]
    
    # 4. 將計算出的顏色賦予點雲
    pcd_camera.colors = o3d.utility.Vector3dVector(colors)
    # --- 結束：根據 Z 軸高度上色 ---
    
    lineset_camera = create_lineset_from_corners(corners_in_camera, color=[1.0, 0.0, 0.0])
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # 4. 更新到視窗
    vis.clear_geometries()
    vis.add_geometry(pcd_camera)
    vis.add_geometry(lineset_camera)
    vis.add_geometry(coordinate_frame)

    set_camera_view(vis, camera_mode, yaw_pose)

def render_video_sequence(state, output_filename, framerate):
    """
    渲染 3D 點雲俯視影片

    Args:
        state :
        output_filename : 輸出檔名，包括檔案路徑
        framerate : 目標幀率
    returns:
    """
    
    # 1. 建立一個臨時資料夾來儲存圖片
    temp_image_folder = "temp_render_3d_images"
    if not os.path.exists(temp_image_folder):
        os.makedirs(temp_image_folder)
    print(f"3D 渲染圖片將暫存於: '{temp_image_folder}/'")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Video Renderer", width=1280, height=720, visible=False) 
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 1.5
    
    # 2. 繪圖
    total_frames = state.total_frames
    for i in range(total_frames):
        state.current_index = i
        update_geometries(vis, state)

        vis.poll_events()
        vis.update_renderer()
        
        # 儲存當前畫面
        image_path = os.path.join(temp_image_folder, f"frame_{i:05d}.png")
        vis.capture_screen_image(image_path, do_render=True)

    vis.destroy_window()
    print(f"\n成功渲染 {total_frames} 幀 3D 圖片。")

    # 3. 使用 ffmpeg 將圖片序列合成為影片
    print("正在使用 ffmpeg 生成影片...")
    ffmpeg_command = [
        'ffmpeg',
        '-r', str(framerate),
        '-i', f'{temp_image_folder}/frame_%05d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y', # 覆蓋已存在的檔案
        output_filename
    ]
    
    try:
        # 隱藏 ffmpeg 的輸出
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"影片成功儲存至: '{output_filename}'")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("\n" + "="*50)
        print("錯誤: ffmpeg 執行失敗。")
        print("請確保你已經安裝了 ffmpeg 並將其加入了系統的 PATH 環境變數。")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"FFmpeg 錯誤訊息:\n{e.stderr.decode('utf-8')}")
        print("="*50)

    # 4. 清理臨時圖片
    shutil.rmtree(temp_image_folder)
    print(f"已刪除臨時資料夾: '{temp_image_folder}'")

# ==============================================================================
# 2D 顯示
# ==============================================================================

def draw_projection_on_image(image, points_2d, corners_2d):
    """
    2D 影像上面繪製投影點雲和框

    Args:
        image : 
        points_2d : 2D 投影點
        corners_2d : 2D 投影框的角點
    Returns:
        numpy.ndarray: 繪製了投影點和框的影像
    """
    # 畫上投影後的點雲
    image_height, image_width, _ = image.shape
    in_bounds = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_width) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_height)
    
    for point in points_2d[in_bounds].astype(np.int32):
        cv2.circle(image, tuple(point), radius=1, color=(0, 255, 0), thickness=-1) # 綠色的點

    # 畫上 3D 框的 2D 投影線條
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), 
        (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for box_corners in corners_2d:
        pts = box_corners.astype(np.int32)
        for i, j in edges:
            cv2.line(image, tuple(pts[i]), tuple(pts[j]), color=(0, 0, 255), thickness=2) # 紅色的框

    return image

def render_2d_video_sequence(state, output_filename, framerate):
    """
    渲染 2D 影片

    Args:
        state :
        output_filename : 輸出檔名，包括檔案路徑
        framerate : 目標幀率
    Returns:
        None :
    """
    # 1. 建立一個臨時資料夾來儲存圖片
    temp_image_folder = "temp_render_2d_images"
    if not os.path.exists(temp_image_folder):
        os.makedirs(temp_image_folder)
    print(f"2D 渲染圖片將暫存於: '{temp_image_folder}/'") 


    height, width = state.data_loader.image_height, state.data_loader.image_width

    K = state.camera_intrinsics
    D = state.distortion_coeffs

    # 2. 遍歷所有幀
    for i in range(state.total_frames):
        print(f"\r處理幀: {i+1}/{state.total_frames}", end="")
        # a. 從 DataLoader 獲取所有需要的資料
        frame_data = state.data_loader[i]
        pcd_legacy = frame_data['point_cloud']
        original_boxes = frame_data['boxes_lidar']
        canvas = frame_data['image'].copy()

        if canvas is None or canvas.size == 0:
            print(f"\n 警告：幀 {i} 找不到對應照片，使用黑背景代替。")
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # b. 去畸變
        # 計算新的、最佳化的相機內參矩陣
        #new_camera_intrinsics, roi = cv2.getOptimalNewCameraMatrix(K, D, (width, height), 0, (width, height))
        #undistorted_canvas = cv2.undistort(canvas, K, D, None, new_camera_intrinsics)
        
        # c. 執行座標轉換和投影
        points_in_camera = transform_points(np.asarray(pcd_legacy.points), state.lidar_to_camera_extrinsics)
        corners_in_camera = transform_boxes_to_corners(original_boxes, state.lidar_to_camera_extrinsics)
        projected_points, _ = project_points_to_image(points_in_camera, K)

        projected_corners = np.zeros((0, 8, 2)) # 8個角點的2D投影

        if corners_in_camera.shape[0] > 0:
            valid_box_mask = (corners_in_camera[:, :, 2] > 0).all(axis=1)
            valid_3d_corners = corners_in_camera[valid_box_mask]
            if valid_3d_corners.shape[0] > 0:
                projected_valid_corners, _ = project_points_to_image(valid_3d_corners.reshape(-1, 3), K)
                projected_corners = projected_valid_corners.reshape(-1, 8, 2)

        # c. 在影像上繪圖並儲存
        output_frame = draw_projection_on_image(canvas, projected_points, projected_corners)
        image_path = os.path.join(temp_image_folder, f"frame_{i:05d}.png")
        cv2.imwrite(image_path, output_frame)

    print(f"\n成功渲染 {state.total_frames} 幀 2D 圖片。") 

    # 3. 使用 ffmpeg 將圖片序列合成為影片
    print("正在使用 ffmpeg 生成影片...") 
    ffmpeg_command = [ 
        'ffmpeg',
        '-r', str(framerate),
        '-i', f'{temp_image_folder}/frame_%05d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y', 
        output_filename 
    ]
    
    try:
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"影片成功儲存至: '{output_filename}'")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("\n" + "="*50)
        print("錯誤: ffmpeg 執行失敗。")
        print("請確保你已經安裝了 ffmpeg 並將其加入了系統的 PATH 環境變數。")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"FFmpeg 錯誤訊息:\n{e.stderr.decode('utf-8')}")
        print("="*50)

    # 4. 清理臨時圖片
    shutil.rmtree(temp_image_folder)
    print(f"已刪除臨時資料夾: '{temp_image_folder}'")

# ==============================================================================
# 主程式 (Main)
# ==============================================================================

class AppState:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.current_index = 0
        self.total_frames = len(data_loader)

        self.lidar_to_camera_extrinsics = self.data_loader.lidar_to_camera_extrinsics
        self.distortion_coeffs = self.data_loader.distortion_coeffs 
        self.camera_intrinsics = self.data_loader.camera_intrinsics



def run_interactive_mode(state):
    """ 互動式視覺化 """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="LiDAR Interactive Viewer")
    
    def next_frame_callback(vis):
        state.current_index = (state.current_index + 1) % state.total_frames
        update_geometries(vis, state)

    def prev_frame_callback(vis):
        state.current_index = (state.current_index - 1 + state.total_frames) % state.total_frames
        update_geometries(vis, state)
    
    vis.register_key_callback(ord('N'), next_frame_callback)
    vis.register_key_callback(ord('B'), prev_frame_callback)
    
    print("\n" + "="*50)
    print("互動模式已啟動:\n  N: 下一幀\n  B: 上一幀\n  Q: 退出")
    print("="*50 + "\n")

    update_geometries(vis, state)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 1.5

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="LiDAR 視覺化工具 (已使用 DataLoader 重構)")
    parser.add_argument("--base_folder", type=str, required=True, help="包含 VLS128_pcdnpy/, imu/, image/, 3d_label.pkl 的基礎資料夾路徑")
    parser.add_argument("--render_video", type=str, metavar="OUTPUT_FILE", help="渲染 3D 影片")
    parser.add_argument("--render_2d_video", type=str, metavar="OUTPUT_2D_FILE", help="渲染 2D 投影影片")
    parser.add_argument("--fps", type=int, default=10, help="輸出影片的幀率")

    args = parser.parse_args()

    # DataLoader
    try:
        data_loader = SceneDataLoader(base_folder=args.base_folder)
        print(f"成功載入 {len(data_loader)} 幀資料。")
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        exit(1)
    app_state = AppState(data_loader)

    if args.render_2d_video:
        print(app_state.data_loader.image_width)
        render_2d_video_sequence(app_state, args.render_2d_video, args.fps)
    if args.render_video:
        render_video_sequence(app_state, args.render_video, args.fps)
    #else:
        #run_interactive_mode(app_state)