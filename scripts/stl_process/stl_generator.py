import trimesh
import numpy as np

def generate_parametric_cube(length, width, height, filename="cube.stl"):
    """
    根据指定的长、宽和高生成一个立方体，并将其底面放置在 z=0 平面上，使长宽中心投影在 z 轴上。
    
    参数:
    - length: 立方体的长度 (X 方向)
    - width: 立方体的宽度 (Y 方向)
    - height: 立方体的高度 (Z 方向)
    - filename: 保存的 STL 文件名
    """
    # 计算顶点偏移，底面在 z=0
    x_offset = length / 2
    y_offset = width / 2
    z_offset = height  # 由于底面在 z=0，因此 z 偏移为正高

    # 定义立方体的八个顶点，底面在 z=0
    vertices = np.array([
        [-x_offset, -y_offset, 0], [x_offset, -y_offset, 0],
        [x_offset, y_offset, 0], [-x_offset, y_offset, 0],    # 底面
        [-x_offset, -y_offset, z_offset], [x_offset, -y_offset, z_offset],
        [x_offset, y_offset, z_offset], [-x_offset, y_offset, z_offset]       # 顶面
    ])

    # 定义立方体的12个三角面
    faces = np.array([
        [0, 1, 2], [0, 2, 3],            # 底面
        [4, 5, 6], [4, 6, 7],            # 顶面
        [0, 1, 5], [0, 5, 4],            # 前面
        [2, 3, 7], [2, 7, 6],            # 后面
        [1, 2, 6], [1, 6, 5],            # 右侧
        [0, 3, 7], [0, 7, 4]             # 左侧
    ])

    # 创建立方体网格
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 修正法线方向
    mesh.rezero()  # 确保重心对齐
    mesh.fix_normals()  # 确保法线朝外

    # 保存为 STL 文件
    mesh.export(filename)
    print(f"STL 文件已保存为 {filename}")

if __name__ == "__main__":
    generate_parametric_cube(150, 150, 20, "../tactile_envs/assets/insertion/assets/custom_table.stl") # table
    generate_parametric_cube(200, 20, 20, "../tactile_envs/assets/insertion/assets/custom_obj.stl") # example 1d obj
    generate_parametric_cube(40, 40, 25, "../tactile_envs/assets/insertion/assets/unit_cube.stl") # unit_cube