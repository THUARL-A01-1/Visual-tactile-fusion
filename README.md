# Visual-tactile Fusion-based Failure Recovery RL
## Stucture
### scripts
``` text
scripts/
├── stl_process/
│   ├── stl_files/
│   ├── stl_generator.py
│   └── stl_trans.py
├── test_random_grasp.py # 环境测试调试脚本
└── train_from_failure.py # 训练主脚本
```

### models
``` text
models/
├── ppo_grasp_1d_v1/
├── custom_policies.py
└── networks.py
```
### tactile_envs
``` text
tactile_envs/
├── __init__.py
├── envs/
│   ├── __init__.py
│   ├── insertion.py        # 最重要，定义 GraspPhase 类
│   ├── regrasp1d.py       # 1D抓取环境
│   └── regrasp2d.py       # 2D抓取环境，未实现
└── assets/
│   ├── ...
│   └── insertion
│       ├── assets          # stl 模型文件
│       ├── complex_objects # 自定义的待抓取物体
│       │   ├── continue_1d # 质心位置连续随机分布的 1d 物体
│       │   ├── discrete_2d # 质心位置离散随机分布（小质量块连接）的 2d 物体
│       │   ├── continue_1d_gen.py # 生成 continue_1d
│       │   ├── discrete_2d_gen.py # 生成 discrete_2d
│       │   ├── shape_generator.py # 形状生成器
│       ├── markers # 触觉传感器标记点
│       ├── custom_touch_sensors.xml 
│       ├── generate_pads_collisions.py  # 生成 xml
│       ├── left_custom_pad_collisions.xml
│       ├── right_custom_pad_collisions.xml
│       └── scene.xml 
└── utils/
   └── ...
```

## Installation
### Choice 1: venv
```bash
cd tactile_envs # 按原仓库教程安装 tactile_envs 环境，并且进入 tactile_envs 文件夹工作区
source ../tactile_envs/bin/activate # 激活虚拟环境
``` 
### Choice 2: conda 

To install `tactile_envs` in a fresh conda env:
```
conda create --name tactile_envs python=3.11
conda activate tactile_envs
pip install -r requirements.txt
```

Before running the environment code, make sure that you generate the tactile sensor collision meshes for the desired resolution. E.g., for 32x32 sensors:
``` bash
python tactile_envs/assets/insertion/generate_pad_collisions.py --nx 32 --ny 32
```


## Preparation
### 3D Model Coordinate Transformation
To rotate and transform the coordinate of tactile sensor .stl file, run:
``` bash
python STL_trans/stl_trans.py
```
### Markers Coordinate Transformation
run: `python tactile_envs/assets/insertion/markers/scripts/coordinate_trans.py`

### Modify and Generate Sensor Pads Configurations
The script `generate_pads_collisions.py` generates custom XML files for pad collisions based on marker coordinates from a CSV file. It allows users to configure parameters such as the number of rows and columns of markers, the scaling factor, and the size of each tactile pad. You can modify parameters including `--num_rows`, `--num_cols`, `--scale`, `--size_x`, `--size_y`, `--size_z`. 
Example Usage: 
``` bash
python tactile_envs/assets/insertion/generate_pads_collisions.py --num_rows 20 --num_cols 20 --scale 0.0011
```

### Modify Other Simulation Configurations
In the files below you can modify object attributes, movement modes, etc. Refer to files:
1. `tactile_envs/envs/insertion.py`
2. `tactile_envs/assets/insertion/scene.xml`







## Test the available environment:
The script `train_from_failure.py` tests a gripping environment with tactile and visual information using `gymnasium` and `tactile_envs`. The main function, `test_env`, simulates robotic gripping tasks where you can control key environment parameters such as the number of episodes, steps, state type (e.g., `vision_and_touch`, `vision`, `touch`), and image size through command-line arguments.
### Example Usage
``` bash
python scripts/train_from_failure.py --n_episodes 50 --n_steps 500 --show_highres --state_type vision_and_touch
```
This command runs 50 episodes, with 500 steps per episode, using tactile information only and high-resolution images.
Available parameters including: `--n_episodes`, `--n_steps`, `--show_highres`, `--seed`, `--env_name`, `--state_type`, `--multiccd`, `--im_size`, `--no_gripping`, `--no_rotation`, `--tactile_shape`, `--max_delta`. To see all available options and defaults, use:
``` bash
python scripts/test_env_grip.py --help
```






## Citation
If you use these environments in your research, please cite the following paper:
```
@article{sferrazza2023power,
  title={The power of the senses: Generalizable manipulation from vision and touch through masked multimodal learning},
  author={Sferrazza, Carmelo and Seo, Younggyo and Liu, Hao and Lee, Youngwoon and Abbeel, Pieter},
  year={2023}
}
```

## Additional resources
Are you interested in more complex robot environments with high-dimensional tactile sensing? Check out [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench)!

## References
This codebase contains some files adapted from other sources:
* Gymnasium-Robotics: https://github.com/Farama-Foundation/Gymnasium-Robotics
* robosuite: https://github.com/ARISE-Initiative/robosuite/tree/master
* TactileSimulation: https://github.com/eanswer/TactileSimulation
