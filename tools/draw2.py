import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import math

# 您的數據
data1 = np.array([[ 3.73518608e+01,6.43973387e+01,4.50991675e-01,4.63300000e+00,2.01100000e+00,1.57300000e+00,3.08940784e+00],
 [ 5.97927366e+00,3.50087253e+01,4.40587610e-02,4.01000000e+00,1.70800000e+00,1.63100000e+00,1.50189087e+00],
 [-4.49864330e+00,1.52533225e+01,3.96393503e-01,1.02010000e+01,2.87700000e+00,3.59500000e+00,1.59520502e+00],
 [-8.27170431e+00,7.76698402e+01,2.04984263e+00,4.95600000e+00,2.13500000e+00,2.17000000e+00,-5.57437416e-02],
 [ 3.30117512e+00,4.03404947e+01,1.45804100e-01,4.11500000e+00,1.84700000e+00,1.52600000e+00,1.50278136e+00],
 [ 2.95259452e+01,6.50114578e+01,5.75882031e-01,4.81900000e+00,1.93900000e+00,1.73600000e+00,3.08940784e+00],
 [ 3.78552674e+01,7.09529966e+01,6.90733989e-01,4.69800000e+00,1.97200000e+00,1.58100000e+00,3.11330812e+00],
 [ 6.70496413e+00,4.57677899e+01,6.48678724e-01,4.53500000e+00,1.78700000e+00,2.05900000e+00,1.48532098e+00],
 [-2.05323576e+00,3.80260869e+01,2.70261055e-01,4.72700000e+00,1.90700000e+00,1.95700000e+00,1.58046689e+00]])

data2 = np.array([
    [21.685732,14.607297, 1.0547465,5.3309956,2.2460496,1.8873115,4.6773157],
    [-4.4560833,15.050034, 2.1918504,10.281255, 2.9600334,3.52405,1.5846227],
    [ 7.639847,47.255882, 2.3109293,4.696842, 2.0560691,1.7317303,1.4993845]])

def create_box_vertices(x, y, z, dx, dy, dz, ry):
    """創建旋轉長方體的頂點"""
    # 長方體的8個頂點（相對於中心）
    vertices = np.array([
        [-dx/2, -dy/2, -dz/2],
        [dx/2, -dy/2, -dz/2],
        [dx/2, dy/2, -dz/2],
        [-dx/2, dy/2, -dz/2],
        [-dx/2, -dy/2, dz/2],
        [dx/2, -dy/2, dz/2],
        [dx/2, dy/2, dz/2],
        [-dx/2, dy/2, dz/2]
    ])
    
    # 旋轉矩陣（繞Y軸）
    rotation_matrix = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
    
    # 應用旋轉
    rotated_vertices = vertices @ rotation_matrix.T
    
    # 平移到指定位置
    final_vertices = rotated_vertices + np.array([x, y, z])
    
    return final_vertices

def create_box_faces(vertices):
    """創建長方體的面"""
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 頂面
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 後面
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
        [vertices[4], vertices[7], vertices[3], vertices[0]]   # 左面
    ]
    return faces

def plot_3d_boxes():
    """繪製3D長方體"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 設置顏色
    color1 = 'red'
    color2 = 'blue'
    alpha = 0.6
    
    print("繪製第一組數據（紅色）...")
    # 處理第一組數據
    for i, row in enumerate(data1):
        x, y, z, dx, dy, dz, ry = row
        vertices = create_box_vertices(x, y, z, dx, dy, dz, ry)
        faces = create_box_faces(vertices)
        
        # 創建3D多邊形集合
        poly3d = [[face for face in faces]]
        ax.add_collection3d(Poly3DCollection(poly3d[0], alpha=alpha, 
                                           facecolor=color1, edgecolor='black', linewidth=0.5))
        
        # 添加標籤
        ax.text(x, y, z+dz/2+0.5, f'1-{i+1}', fontsize=8)
        print(f"  物體 {i+1}: 位置=({x:.1f}, {y:.1f}, {z:.1f})")
    
    print("\n繪製第二組數據（藍色）...")
    # 處理第二組數據
    for i, row in enumerate(data2):
        x, y, z, dx, dy, dz, ry = row
        vertices = create_box_vertices(x, y, z, dx, dy, dz, ry)
        faces = create_box_faces(vertices)
        
        # 創建3D多邊形集合
        poly3d = [[face for face in faces]]
        ax.add_collection3d(Poly3DCollection(poly3d[0], alpha=alpha, 
                                           facecolor=color2, edgecolor='black', linewidth=0.5))
        
        # 添加標籤
        ax.text(x, y, z+dz/2+0.5, f'2-{i+1}', fontsize=8)
        print(f"  物體 {i+1}: 位置=({x:.1f}, {y:.1f}, {z:.1f})")
    
    # 設置坐標軸範圍
    all_data = np.vstack([data1, data2])
    x_min, x_max = all_data[:, 0].min() - 5, all_data[:, 0].max() + 5
    y_min, y_max = all_data[:, 1].min() - 5, all_data[:, 1].max() + 5
    z_min, z_max = all_data[:, 2].min() - 2, all_data[:, 2].max() + 5
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # 設置標籤和標題
    ax.set_xlabel('X軸')
    ax.set_ylabel('Y軸')
    ax.set_zlabel('Z軸')
    ax.set_title('3D物體可視化\n紅色: 第一組數據 (9個物體), 藍色: 第二組數據 (3個物體)', fontsize=14)
    
    # 添加圖例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=alpha, label='第一組數據'),
                      Patch(facecolor='blue', alpha=alpha, label='第二組數據')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 設置視角
    ax.view_init(elev=20, azim=45)
    
    # 顯示圖形
    plt.tight_layout()
    plt.show()
    
    # 保存圖形
    plt.savefig('3d_visualization.png', dpi=300, bbox_inches='tight')
    print("\n圖形已保存為 '3d_visualization.png'")

def print_summary():
    """打印數據摘要"""
    print("="*50)
    print("數據摘要:")
    print("="*50)
    print(f"第一組數據: {len(data1)} 個物體")
    print(f"第二組數據: {len(data2)} 個物體")
    print(f"總共: {len(data1) + len(data2)} 個物體")
    print("\n數據格式: [x, y, z, dx, dy, dz, ry]")
    print("x, y, z: 物體中心位置")
    print("dx, dy, dz: 物體尺寸（長、寬、高）")
    print("ry: 繞Y軸旋轉角度（弧度）")
    print("="*50)

if __name__ == "__main__":
    print_summary()
    plot_3d_boxes()