import numpy as np
from stl import mesh

your_mesh = mesh.Mesh.from_file('STL_trans/old_pad.stl')

# 读取 STL 文件
# your_mesh = mesh.Mesh.from_file('/Users/xiaokangsun/local_py_proj/Visual-tactile-fusion/STL_trans/old_pad.stl')
# your_mesh.save('ascii_output.stl', mode=stl.Mode.ASCII)
# 调换坐标轴（例如，交换 Z 和 Y 轴）
your_mesh.vectors = np.dot(your_mesh.vectors, [[-1, 0, 0], [0, 0, -1], [0, -1, 0]])

# 保存为新的 STL 文件
print("success")
your_mesh.save('/Users/xiaokangsun/local_py_proj/Visual-tactile-fusion/tactile_envs/assets/insertion/assets/silicone_pad.stl')
