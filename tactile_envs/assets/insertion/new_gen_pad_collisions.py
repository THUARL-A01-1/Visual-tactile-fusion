import shutil
import numpy as np
import pandas as pd
from absl import app

def edit_pad_collisions(path,coordinates,num_points,scale,size_x,size_y,size_z):
    # 创建XML文件
    with open(path, "w") as f:
        f.write("<mujoco>\n")
        for i in range(num_points):
            pos_x = (coordinates.iloc[i, 0] - 0.4216) * scale
            # pos_y = (coordinates.iloc[i, 1]) * scale
            pos_y = (coordinates.iloc[i, 1] + 6.847836) * scale
            # print(f">>>>>>>>>>>>>> init {coordinates.iloc[i, 1]} pos_y: {pos_y}")
            pos_z = coordinates.iloc[i, 2] * scale
            # pos_z = -0.0026
            # pos_z = -0.001
            
            # 生成rgba值，这里使用线性渐变作为示例
            rgb = 0.6 + 0.1 * i / (num_points)
            
            # 生成XML字符串
            xml_string = f'<geom class="pad" pos="{pos_x} {pos_z} {pos_y}" size="{size_x} {size_z} {size_y}" rgba="{rgb} {rgb} {rgb} 1"/>'
            f.write(xml_string + '\n')

            # 可视化每一个pad
            xml_string2 = f'<geom class="visual" type="box" pos="{pos_x} {pos_z} {pos_y}" size="{size_x} {size_z} {size_y}" rgba="{rgb} {rgb} {rgb} 0.5"/>'
            f.write(xml_string2 + '\n')
        
        f.write("</mujoco>")


def main(_):
    # 从CSV文件读取XYZ坐标
    coordinates = pd.read_csv("tactile_envs/assets/insertion/marker_coordinate.csv", header=None)  
    # print(len(coordinates))
    # exit(0)
    num_rows = 20
    num_cols = 20
    num_points = num_rows * num_cols
    # print(f"Number of points: {num_points}")
    # exit(0)

    scale = 0.0011  # 1mm = 0.001m
    # scale = 0.0012

    # 仿照原始文件debug
    size_x = 9.6745/(num_rows-1) * scale
    size_y = 9.6261/(num_rows-1)* scale

    # print(f"init size_y: {9.6261/(num_rows-1)}")
    size_z = 0.004
    # size_z = 0.007

    # 定义size的默认值
    # size_x = 2.000000 * scale # 例如，固定尺寸
    # size_y = 2.000000 * scale
    # size_z = 0.500000 * scale

    # size_x = 1.000000 * scale # 例如，固定尺寸
    # size_y = 1.000000 * scale
    # size_z = 1.000000 * scale

    # 创建XML文件
    edit_pad_collisions(path="tactile_envs/assets/insertion/left_custom_pad_collisions.xml",
                        coordinates=coordinates,num_points=num_points,scale=scale,size_x=size_x,size_y=size_y,size_z=size_z)
    
    # 创建XML文件
    edit_pad_collisions(path="tactile_envs/assets/insertion/right_custom_pad_collisions.xml",
                        coordinates=coordinates,num_points=num_points,scale=scale,size_x=size_x,size_y=size_y,size_z=size_z)

    # 复制文件（如有需要）
    # shutil.copyfile("tactile_envs/assets/insertion/custom_pad_collisions.xml", 
    #                 "tactile_envs/assets/insertion/custom_pad_collisions_mirror.xml")

    # 生成传感器配置的XML文件
    touch_sensor_string = """
    <mujoco>
    <sensor>
        <plugin name="touch_right" plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch_right">
        <config key="size" value="{0} {0}"/>
            <config key="fov" value="18 18"/>
        <config key="gamma" value="0"/>
        <config key="nchannel" value="3"/>
        </plugin>
    </sensor>
    <sensor>
        <plugin name="touch_left" plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch_left">
        <config key="size" value="{0} {0}"/>
            <config key="fov" value="18 18"/>
        <config key="gamma" value="0"/>
        <config key="nchannel" value="3"/>
        </plugin>
    </sensor>
    </mujoco>
    """.format(num_rows)
    
    # 写入传感器配置
    with open("tactile_envs/assets/insertion/custom_touch_sensors.xml", "w") as f:
        f.write(touch_sensor_string)

if __name__ == "__main__":
    app.run(main)