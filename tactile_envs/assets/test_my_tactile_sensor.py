import mujoco
import my_tactile_sensor

# 加载 MuJoCo 模型
model =('/Users/xiaokangsun/local_py_proj/tactile_env/tactile_envs/assets/my_tactile_sensor.xml')
data = mujoco.MjData(model)

# 初始化传感器插件
sensor = my_tactile_sensor.MyTactileSensor()

# 运行模拟
while True:
    mujoco.mj_step(model, data)
    sensor_data = sensor.update(model, data)
    print(sensor_data)