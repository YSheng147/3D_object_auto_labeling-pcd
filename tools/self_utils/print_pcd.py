import open3d as o3d
import numpy as np
import torch

pcd = o3d.io.read_point_cloud("/home/ys/MS3D/data/custom/colored_pcd_img1/000000.pcd")
print(torch.__version__)
print(o3d.__version__)
print(pcd)  # 會顯示點雲資料概要
points = np.asarray(pcd.points)  # 轉成 numpy 陣列
print(points.shape)  # (N,3)
print(points)  # 印出點的座標
#o3d.visualization.draw_geometries([pcd])

# 創建視覺化器
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="My Window", width=800, height=600)

# 添加點雲
vis.add_geometry(pcd)

# 自定義渲染選項（可選）
render_option = vis.get_render_option()
render_option.point_size = 5.0  # 設置點的大小
# 獲取視圖控制器
ctr = vis.get_view_control()
if ctr is None:
    print("Failed to get view control!")
else:
    print(f"View control type: {type(ctr)}")
# 運行視覺化窗口
vis.run()

# 關閉窗口
vis.destroy_window()