import open3d as o3d
import numpy as np
import math

# 您的數據
data1 = np.array([
    [3.73518608e+01, 6.43973387e+01, 4.50991675e-01, 4.63300000e+00, 2.01100000e+00, 1.57300000e+00, 3.08940784e+00],
    [5.97927366e+00, 3.50087253e+01, 4.40587610e-02, 4.01000000e+00, 1.70800000e+00, 1.63100000e+00, 1.50189087e+00],
    [-4.49864330e+00, 1.52533225e+01, 3.96393503e-01, 1.02010000e+01, 2.87700000e+00, 3.59500000e+00, 1.59520502e+00],
    [-8.27170431e+00, 7.76698402e+01, 2.04984263e+00, 4.95600000e+00, 2.13500000e+00, 2.17000000e+00, -5.57437416e-02],
    [3.30117512e+00, 4.03404947e+01, 1.45804100e-01, 4.11500000e+00, 1.84700000e+00, 1.52600000e+00, 1.50278136e+00],
    [2.95259452e+01, 6.50114578e+01, 5.75882031e-01, 4.81900000e+00, 1.93900000e+00, 1.73600000e+00, 3.08940784e+00],
    [3.78552674e+01, 7.09529966e+01, 6.90733989e-01, 4.69800000e+00, 1.97200000e+00, 1.58100000e+00, 3.11330812e+00],
    [6.70496413e+00, 4.57677899e+01, 6.48678724e-01, 4.53500000e+00, 1.78700000e+00, 2.05900000e+00, 1.48532098e+00],
    [-2.05323576e+00, 3.80260869e+01, 2.70261055e-01, 4.72700000e+00, 1.90700000e+00, 1.95700000e+00, 1.58046689e+00]
])

data2 = np.array([
    [21.685732, 14.607297, 1.0547465, 5.3309956, 2.2460496, 1.8873115, 4.6773157],
    [-4.4560833, 15.050034, 2.1918504, 10.281255, 2.9600334, 3.52405, 1.5846227],
    [7.639847, 47.255882, 2.3109293, 4.696842, 2.0560691, 1.7317303, 1.4993845]
])

def create_oriented_box(x, y, z, dx, dy, dz, ry, color):
    """
    創建一個帶有旋轉的長方體
    x, y, z: 中心位置
    dx, dy, dz: 尺寸（長、寬、高）
    ry: 繞Y軸的旋轉角度（弧度）
    color: 顏色
    """
    # 創建長方體
    box = o3d.geometry.OrientedBoundingBox()
    box.center = np.array([x, y, z])
    box.extent = np.array([dx, dy, dz])
    
    # 設置旋轉矩陣 (繞Y軸旋轉)
    rotation_matrix = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
    box.R = rotation_matrix
    
    # 轉換為TriangleMesh以便設置顏色
    mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box)
    mesh.paint_uniform_color(color)
    
    # 添加wireframe效果
    wireframe = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
    wireframe.paint_uniform_color([0, 0, 0])  # 黑色邊框
    
    return mesh, wireframe

def visualize_data():
    """可視化兩組數據"""
    geometries = []
    
    # 顏色設置
    color1 = [0.8, 0.2, 0.2]  # 紅色 - 第一組數據
    color2 = [0.2, 0.2, 0.8]  # 藍色 - 第二組數據
    
    # 處理第一組數據
    print("處理第一組數據...")
    for i, row in enumerate(data1):
        x, y, z, dx, dy, dz, ry = row
        mesh, wireframe = create_oriented_box(x, y, z, dx, dy, dz, ry, color1)
        geometries.extend([mesh, wireframe])
        print(f"  物體 {i+1}: 位置=({x:.2f}, {y:.2f}, {z:.2f}), 尺寸=({dx:.2f}, {dy:.2f}, {dz:.2f}), 旋轉={ry:.2f}")
    
    # 處理第二組數據
    print("\n處理第二組數據...")
    for i, row in enumerate(data2):
        x, y, z, dx, dy, dz, ry = row
        mesh, wireframe = create_oriented_box(x, y, z, dx, dy, dz, ry, color2)
        geometries.extend([mesh, wireframe])
        print(f"  物體 {i+1}: 位置=({x:.2f}, {y:.2f}, {z:.2f}), 尺寸=({dx:.2f}, {dy:.2f}, {dz:.2f}), 旋轉={ry:.2f}")
    
    # 添加坐標軸
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    geometries.append(coordinate_frame)
    
    # 創建可視化窗口
    print("\n正在創建3D可視化窗口...")
    print("控制說明:")
    print("- 滑鼠左鍵拖拽: 旋轉視角")
    print("- 滑鼠右鍵拖拽: 平移視角") 
    print("- 滑鼠滾輪: 縮放")
    print("- 按 'Q' 或關閉窗口退出")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="3D物體可視化 - 紅色:第一組數據, 藍色:第二組數據",
        width=1200,
        height=800
    )

def save_as_ply():
    """保存為PLY文件以備後用"""
    geometries = []
    
    # 處理第一組數據
    for i, row in enumerate(data1):
        x, y, z, dx, dy, dz, ry = row
        mesh, wireframe = create_oriented_box(x, y, z, dx, dy, dz, ry, [0.8, 0.2, 0.2])
        geometries.extend([mesh, wireframe])
    
    # 處理第二組數據
    for i, row in enumerate(data2):
        x, y, z, dx, dy, dz, ry = row
        mesh, wireframe = create_oriented_box(x, y, z, dx, dy, dz, ry, [0.2, 0.2, 0.8])
        geometries.extend([mesh, wireframe])
    
    # 合併所有mesh
    combined_mesh = o3d.geometry.TriangleMesh()
    for geom in geometries:
        if isinstance(geom, o3d.geometry.TriangleMesh):
            combined_mesh += geom
    
    # 保存為PLY文件
    o3d.io.write_triangle_mesh("visualization.ply", combined_mesh)
    print("已保存為 visualization.ply 文件")

def try_alternative_backend():
    """嘗試使用替代的渲染後端"""
    try:
        # 方法1: 設置環境變量
        import os
        os.environ['DISPLAY'] = ':0'
        visualize_data()
    except Exception as e1:
        print(f"方法1失敗: {e1}")
        
        try:
            # 方法2: 使用headless模式並保存圖片
            print("嘗試headless模式...")
            geometries = []
            
            # 創建幾何體
            for i, row in enumerate(data1):
                x, y, z, dx, dy, dz, ry = row
                mesh, wireframe = create_oriented_box(x, y, z, dx, dy, dz, ry, [0.8, 0.2, 0.2])
                geometries.extend([mesh, wireframe])
            
            for i, row in enumerate(data2):
                x, y, z, dx, dy, dz, ry = row
                mesh, wireframe = create_oriented_box(x, y, z, dx, dy, dz, ry, [0.2, 0.2, 0.8])
                geometries.extend([mesh, wireframe])
            
            # 添加坐標軸
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
            geometries.append(coordinate_frame)
            
            # 使用offscreen渲染
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1200, height=800)
            
            for geom in geometries:
                vis.add_geometry(geom)
            
            vis.run()
            vis.capture_screen_image("visualization.png")
            vis.destroy_window()
            print("已保存截圖為 visualization.png")
            
        except Exception as e2:
            print(f"方法2也失敗: {e2}")
            print("正在保存PLY文件...")
            save_as_ply()

if __name__ == "__main__":
    print("檢測到圖形界面問題，嘗試替代方案...")
    try_alternative_backend()