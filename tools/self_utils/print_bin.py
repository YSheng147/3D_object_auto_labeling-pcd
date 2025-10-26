import numpy as np

# 讀取 bin 檔案，假設每點有 4 個 float (x, y, z, intensity)
points = np.fromfile("/home/ys/MS3D/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin", dtype=np.float32)
points = points.reshape(-1, 5)  # 每列為一個點
print(points.shape)
print(points[:5])  # 印出前五筆點資料
