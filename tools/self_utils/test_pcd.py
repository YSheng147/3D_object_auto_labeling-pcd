import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector()
o3d.visualization.draw_geometries([pcd])