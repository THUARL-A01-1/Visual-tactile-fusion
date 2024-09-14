import trimesh
import matplotlib.pyplot as plt

'''STL 文件可视化'''

# 读取 STL 文件
mesh = trimesh.load('/Users/xiaokangsun/local_py_proj/Xiaokang-Sun/Asset/softbody.stl')

# 渲染 3D 模型
mesh.show()