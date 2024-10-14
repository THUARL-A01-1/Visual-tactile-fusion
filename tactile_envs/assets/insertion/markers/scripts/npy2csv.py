import numpy as np
import csv
import os

def npy_to_csv(npy_file_path, csv_file_path):
    """
    将 .npy 文件转换为 .csv 文件

    参数:
    npy_file_path: str, .npy 文件的路径
    csv_file_path: str, 保存 .csv 文件的路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(npy_file_path):
        raise FileNotFoundError(f"找不到 .npy 文件: {npy_file_path}")
    
    # 读取 .npy 文件中的数据
    data = np.load(npy_file_path)
    
    # 打开或创建一个 .csv 文件
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # 如果数据是一维数组，直接写入
        if data.ndim == 1:
            csv_writer.writerow(data)
        # 如果数据是多维数组，逐行写入
        else:
            csv_writer.writerows(data)
    
    print(f"成功将 {npy_file_path} 转换为 {csv_file_path}")

if __name__ == '__main__':
    # 设置 .npy 文件路径和 .csv 文件保存路径
    npy_file = '/Users/xiaokangsun/local_py_proj/Visual-tactile-fusion/tactile_envs/assets/insertion/markers/Markers.npy'  # 替换为你的 .npy 文件路径
    csv_file = '/Users/xiaokangsun/local_py_proj/Visual-tactile-fusion/tactile_envs/assets/insertion/markers/ideal_markers.csv'  # 替换为你希望保存的 .csv 文件路径

    # 调用转换函数
    npy_to_csv(npy_file, csv_file)
