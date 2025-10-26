import pickle
import numpy as np
import torch

"""
[{  'name':             #預測3D框標籤名稱
        ['第1個框', '第2個框', ...],
    'pred_labels':      #預測3D框標籤ID
        ['第1個框', '第2個框', ...],
    'score':            #預測置信度
        ['第1個框', '第2個框', ...],
    'boxes_lidar':      #預測3D框位置
        [[x, y, z, dx, dy, dz, heading], #第1個框
        [中心x, 中心y, 中心z, x軸長度(長), y軸長度(寬), z軸長度(高), z軸旋轉角度], #第2個框
         ...
        ],
    'frame_id':         #對應幀ID
        'xxxx',
    'file_name':        #對應檔案名
        'xxxx.pcd'
}, 
{  'name': 
        ['Vehicle', 'Cyclist', 'Vehicle', 'Pedestrian', ...],
    'pred_labels': 
        [1, 3, 1, 2, ...],
    'score': 
        [0.9543666, 0.91705304, 0.8858577, ...],
    'boxes_lidar': 
        [[14.008743, 23.693468, 0.18879847, 5.2519183, 2.1775723, 2.5923324, 3.2007687],
         [17.905642, 21.211912, -0.026869845, 4.7511477, 2.0320444, 1.7714549, 3.1628015],
         ...
        ],
    'frame_id': 
        '1745206340',
    'file_name':
        '01745206340_pointcloud.pcd'
},...
]
"""

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
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
#3D空間框8點計算
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
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
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

def main():
    #文件加載
    file_name = "/home/ys/MS3D/data/custom/2024-07-03/highway_cloudy_day/#00_csd/result.pkl"

    with open(file_name,'rb') as f:
        detection_sets = pickle.load(f)

    #單一幀讀取
    frame = detection_sets[-1]
    print(f"Frame ID：{frame['frame_id']}\tPCD File Name：{frame['file_name']}")
    for i in range(len(frame['pred_labels'])):
        print(f"ID：${i}\tLabel name：{frame['name'][i]}\tLabel ID：{frame['pred_labels'][i]}\tScore：{frame['score'][i]}")
        print(f"   Box：[\
x：{frame['boxes_lidar'][i][0]},\t\
y：{frame['boxes_lidar'][i][1]},\t\
z：{frame['boxes_lidar'][i][2]},\t\n\t\
dx：{frame['boxes_lidar'][i][3]},\t\
dy：{frame['boxes_lidar'][i][4]},\t\
dz：{frame['boxes_lidar'][i][5]}\t\
heading：{frame['boxes_lidar'][i][6]}]"
        )

    #3D空間框8點計算
    box_pts = boxes_to_corners_3d(frame['boxes_lidar'])
    print(box_pts[0])
    #BEV點轉換
    box_pts_bev = box_pts[:,:5,:2]     
    print(box_pts_bev[0])
    

if __name__ == '__main__':
    main()