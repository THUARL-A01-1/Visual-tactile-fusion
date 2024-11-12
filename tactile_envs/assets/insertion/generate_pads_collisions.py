import shutil
import numpy as np
import pandas as pd
import argparse

def edit_pad_collisions(path, coordinates, num_points, scale, size_x, size_y, size_z):
    """Creates an XML file for pad collisions."""
    with open(path, "w") as f:
        f.write("<mujoco>\n")
        for i in range(num_points):
            offsetx = 0
            offsety = 0
            offsetz = 0

            pos_x = (coordinates.iloc[i, 0] + offsetx) * scale
            pos_y = (coordinates.iloc[i, 1] + offsety) * scale
            pos_z = (coordinates.iloc[i, 2] + offsetz) * scale
            
            rgb = 0.6 + 0.1 * i / num_points
            
            xml_string = f'<geom class="pad" pos="{pos_x} {pos_z} {pos_y}" size="{size_x} {size_z} {size_y}" rgba="{rgb} {rgb} {rgb} 1"/>'
            f.write(xml_string + '\n')
            xml_string2 = f'<geom class="visual" type="box" pos="{pos_x} {pos_z} {pos_y}" size="{size_x} {size_z} {size_y}" rgba="{rgb} {rgb} {rgb} 0.5"/>'
            f.write(xml_string2 + '\n')

        f.write("</mujoco>")

def parse_arguments():
    """Parse command-line arguments for pad collision editing."""
    parser = argparse.ArgumentParser(description="Generate pad collision XML files with custom parameters.")
    
    parser.add_argument("--num_rows", type=int, default=20, help="Number of rows of markers")
    parser.add_argument("--num_cols", type=int, default=20, help="Number of columns of markers")
    parser.add_argument("--scale", type=float, default=0.0011, help="Scale factor for the coordinates")
    parser.add_argument("--size_x", type=float, help="Size of the pad in the x direction")
    parser.add_argument("--size_y", type=float, help="Size of the pad in the y direction")
    parser.add_argument("--size_z", type=float, default=0.0001, help="Thickness of each tactile pad")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Fixed CSV path
    csv_path = "/Users/xiaokangsun/local_py_proj/Visual-tactile-fusion/tactile_envs/assets/insertion/markers/t_ideal_markers.csv"
    
    # Load coordinates from the fixed CSV file
    coordinates = pd.read_csv(csv_path, header=None)
    
    num_points = args.num_rows * args.num_cols

    # Default calculation for size_x and size_y if not provided
    size_x = args.size_x if args.size_x else 10 / (args.num_rows - 1) * args.scale
    size_y = args.size_y if args.size_y else 10 / (args.num_cols - 1) * args.scale

    # Paths are hardcoded as requested
    path_left = "tactile_envs/assets/insertion/left_custom_pad_collisions.xml"
    path_right = "tactile_envs/assets/insertion/right_custom_pad_collisions.xml"

    # Print the settings used
    print("Running with the following settings:")
    print(f"CSV path: {csv_path}")
    print(f"Number of rows: {args.num_rows}")
    print(f"Number of columns: {args.num_cols}")
    print(f"Number of points: {num_points}")
    print(f"Scale: {args.scale}")
    print(f"Size (x): {size_x}")
    print(f"Size (y): {size_y}")
    print(f"Size (z): {args.size_z}")
    print(f"Path (left): {path_left}")
    print(f"Path (right): {path_right}")
    print("---------------------------------------------------")

    # Create XML files for left and right pads
    edit_pad_collisions(path=path_left, coordinates=coordinates, num_points=num_points, 
                        scale=args.scale, size_x=size_x, size_y=size_y, size_z=args.size_z)

    edit_pad_collisions(path=path_right, coordinates=coordinates, num_points=num_points, 
                        scale=args.scale, size_x=size_x, size_y=size_y, size_z=args.size_z)

    # Create sensor configuration file
    touch_sensor_string = f"""<mujoco>
<sensor>
    <plugin name="touch_right" plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch_right">
        <config key="size" value="{args.num_rows} {args.num_rows}"/>
        <config key="fov" value="18 18"/>
        <config key="gamma" value="0"/>
        <config key="nchannel" value="3"/>
    </plugin>
</sensor>
<sensor>
    <plugin name="touch_left" plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch_left">
        <config key="size" value="{args.num_rows} {args.num_rows}"/>
        <config key="fov" value="18 18"/>
        <config key="gamma" value="0"/> 
        <config key="nchannel" value="3"/>
    </plugin>
</sensor>
</mujoco>
"""

    # Save the sensor configuration file
    with open("tactile_envs/assets/insertion/custom_touch_sensors.xml", "w") as f:
        f.write(touch_sensor_string)

    print("Success! Sensor configuration written to 'custom_touch_sensors.xml'.")

if __name__ == "__main__":
    main()
