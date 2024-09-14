# import numpy as np
import pandas as pd

# Load the CSV file
data = pd.read_csv('/Users/xiaokangsun/local_py_proj/Visual-tactile-fusion/tactile_envs/assets/insertion/old_marker_coordinate.csv')

print(data.keys())
# data = np.load('/Users/xiaokangsun/local_py_proj/Visual-tactile-fusion/tactile_envs/assets/insertion/old_marker_coordinate.csv', allow_pickle=True)

# Swap the y and z coordinates
data['y'], data['z'] = -data['z'], -data['y']
# Negate the x coordinates
data['x'] = -data['x']

# Save the modified data to a new CSV file
print("...")
data.to_csv('marker_coordinate.csv', index=False)