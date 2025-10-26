# project_3d_to_2d.py
import os
import argparse
import json
import subprocess
import numpy as np
import cv2
import torch
import shutil
from tqdm import tqdm

# 依賴你寫好的 DataLoader
from data_loader import SceneDataLoader

# ==============================================================================
# 核心幾何運算函式 (從你的 transform_to_camera.py 複製過來)
# 這些函數寫得還行，可以直接用。它們只依賴 numpy 和 torch，很乾淨。
# ==============================================================================

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    """
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading]
    Returns:
        corners3d: (N, 8, 3)
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d.numpy() if is_numpy else corners3d

def transform_points(points, transform_matrix):
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed_h = (transform_matrix @ points_h.T).T
    return points_transformed_h[:, :3]

def project_points_to_image(points_3d, intrinsic_matrix):
    valid_indices = points_3d[:, 2] > 0
    if not np.any(valid_indices):
        return np.zeros((0, 2)), valid_indices
        
    points_3d_visible = points_3d[valid_indices]
    points_homogeneous = (intrinsic_matrix @ points_3d_visible.T).T
    points_2d = points_homogeneous[:, :2] / points_homogeneous[:, 2, np.newaxis]
    
    # 返回所有點的投影結果，即使它們在圖像外
    full_points_2d = np.zeros((points_3d.shape[0], 2))
    full_points_2d[valid_indices] = points_2d
    return full_points_2d, valid_indices
    
def corners_to_2d_bbox(corners_2d, image_height, image_width):
    if corners_2d.size == 0:
        return np.zeros((0, 4))

    x_coords = corners_2d[:, :, 0]
    y_coords = corners_2d[:, :, 1]

    x_min = np.min(x_coords, axis=1)
    y_min = np.min(y_coords, axis=1)
    x_max = np.max(x_coords, axis=1)
    y_max = np.max(y_coords, axis=1)

    x_min = np.clip(x_min, 0, image_width)
    y_min = np.clip(y_min, 0, image_height)
    x_max = np.clip(x_max, 0, image_width)
    y_max = np.clip(y_max, 0, image_height)

    return np.stack([x_min, y_min, x_max, y_max], axis=1)
# ==============================================================================
# 新增的核心邏輯: 從 2D 角點計算 BBox
# 這就是我說的 "第 5 步"。
# ==============================================================================

def calculate_all_projections(state):
    """
    純計算函式。
    它只做一件事：遍歷所有幀，計算 2D BBox，然後返回一個包含所有結果的字典。
    它不關心如何儲存或顯示這些結果。
    """
    K = state.camera_intrinsics
    D = state.distortion_coeffs
    img_height = state.data_loader.image_height
    img_width = state.data_loader.image_width
    
    # 計算一次新的內參即可，因為它對所有幀都一樣
    new_camera_intrinsics, _ = cv2.getOptimalNewCameraMatrix(K, D, (img_width, img_height), 0, (img_width, img_height))

    all_results = {}
    for i in tqdm(range(len(state.data_loader)), desc="計算 3D->2D 投影"):
        frame_data = state.data_loader[i]
        pcd_filename = frame_data['pcd_filename']
        boxes_lidar = frame_data['boxes_lidar']
        frame_key = os.path.splitext(os.path.basename(pcd_filename))[0]

        if not isinstance(boxes_lidar, np.ndarray) or boxes_lidar.size == 0:
            all_results[frame_key] = []
            continue

        corners_3d_lidar = boxes_to_corners_3d(boxes_lidar)
        corners_flat = corners_3d_lidar.reshape(-1, 3)
        corners_3d_camera = transform_points(corners_flat, state.lidar_to_camera_extrinsics).reshape(-1, 8, 3)
        
        visible_mask = (corners_3d_camera[:, :, 2] > 0).all(axis=1)
        if not np.any(visible_mask):
            all_results[frame_key] = []
            continue
        
        visible_corners_3d_camera = corners_3d_camera[visible_mask]
        projected_corners_flat, _ = project_points_to_image(
            visible_corners_3d_camera.reshape(-1, 3), new_camera_intrinsics
        )
        projected_corners = projected_corners_flat.reshape(-1, 8, 2)
        bboxes_2d = corners_to_2d_bbox(projected_corners, state.data_loader.image_height, state.data_loader.image_width)
        all_results[frame_key] = bboxes_2d.tolist()

    return all_results

def write_json_output(results, output_filename):
    """
    純 I/O 函式。接收計算結果並寫入 JSON 檔。
    """
    print(f"正在將結果寫入 JSON 檔案: {output_filename}")
    output_json = json.dumps(results, indent=2)
    with open(output_filename, 'w') as f:
        f.write(output_json)
    print("JSON 檔案寫入成功。")

def draw_2d_bboxes_on_image(image, bboxes, color=(0, 255, 0), thickness=2):
    """
    輔助函式，在影像上畫框。
    """
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = map(int, bbox)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    return image

def render_2d_video(state, all_results, output_filename, framerate):
    """
    純 I/O 函式。
    接收計算結果和 AppState，讀取圖片，繪製 BBox，並合成影片。
    它不再執行任何投影計算。
    """
    temp_image_folder = "temp_render_images"
    if os.path.exists(temp_image_folder):
        shutil.rmtree(temp_image_folder)
    os.makedirs(temp_image_folder)
    print(f"2D 渲染圖片將暫存於: '{temp_image_folder}/'")

    height, width = state.data_loader.image_height, state.data_loader.image_width

    K = state.camera_intrinsics
    D = state.distortion_coeffs
    new_camera_intrinsics, _ = cv2.getOptimalNewCameraMatrix(K, D, (width, height), 0, (width, height))

    for i in tqdm(range(len(state.data_loader)), desc="渲染 2D 影片幀"):
        frame_data = state.data_loader[i]
        pcd_filename = frame_data['pcd_filename']
        frame_key = os.path.splitext(os.path.basename(pcd_filename))[0]
        
        # 直接從預算結果中獲取 BBox
        bboxes_2d = all_results.get(frame_key, [])

        canvas = frame_data['image'].copy()
        if canvas is None or canvas.size == 0:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        undistorted_canvas = cv2.undistort(canvas, K, D, None, new_camera_intrinsics)
        output_frame = draw_2d_bboxes_on_image(undistorted_canvas, bboxes_2d)
        image_path = os.path.join(temp_image_folder, f"frame_{i:05d}.png")
        cv2.imwrite(image_path, output_frame)

    print(f"\n成功渲染 {len(state.data_loader)} 幀 2D 圖片。")

    print("正在使用 ffmpeg 生成影片...")
    ffmpeg_command = [
        'ffmpeg', '-framerate', str(framerate), '-i', f'{temp_image_folder}/frame_%05d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', output_filename
    ]

    try:
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"影片成功儲存至: '{output_filename}'")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("\n" + "="*60)
        print("錯誤: ffmpeg 執行失敗。請確保已安裝 ffmpeg 並加入 PATH。")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"FFmpeg 錯誤訊息:\n{e.stderr.decode('utf-8', errors='ignore')}")
        print("="*60)
    finally:
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

def main():
    parser = argparse.ArgumentParser(description="將 3D 標註框投影到 2D 影像平面並輸出 2D BBox 座標。")
    parser.add_argument("--base_folder", type=str, help="包含 VLS128_pcdnpy/, imu/, image/, 3d_label.pkl 的基礎資料夾路徑")
    parser.add_argument("--output", "-o", type=str, help="輸出 JSON 檔案的路徑。如果未提供，則直接印在終端機上。")
    parser.add_argument("--render_video", type=str, help="輸出 2D 渲染影片的檔案路徑 (例如: output.mp4)。")
    parser.add_argument("--framerate", type=int, default=10, help="渲染影片時使用的幀率 (FPS)。僅在 --render-video 時有效。")
    args = parser.parse_args()


    try:
        data_loader = SceneDataLoader(base_folder=args.base_folder)
        print(f"成功載入 {len(data_loader)} 幀資料。")
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        exit(1)
    app_state = AppState(data_loader)

    # 1. 執行核心計算
    all_projections = calculate_all_projections(app_state)
    
    # 2. 根據參數執行對應的輸出操作
    if args.output:
        write_json_output(all_projections, args.output)
    
    if args.render_video:
        render_2d_video(app_state, all_projections, args.render_video, args.framerate)

if __name__ == "__main__":
    main()