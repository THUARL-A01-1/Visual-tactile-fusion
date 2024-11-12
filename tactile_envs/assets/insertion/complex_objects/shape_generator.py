import os
import numpy as np

def generate_complex_object_xml(filename="exp_complex_object.xml", cube_count=5, cube_size=0.1, 
                                cube_positions=None, cube_rotations=None, cube_masses=None, 
                                rgba=(0.8, 0.8, 0.1, 1), friction=0.8, diaginertia_factor=0.01):
    """
    Generate the XML file for the complex object, which is composed of multiple unit_cube.

    Parameters:
    - filename: name of the output XML file.
    - cube_count: number of cubes to be connected.
    - cube_size: size of each cube.
    - cube_positions: relative positions of the cubes, like [(x1, y1, z1), ...], if None, the cubes will be arranged automatically.
    - cube_rotations: rotation angles of each cube, like [(rx1, ry1, rz1), ...].
    - cube_masses: masses of each cube, the length should be the same as cube_count. If None, the default mass is 1.0.
    - rgba: color of the cubes, like (r, g, b, a).
    - friction: friction coefficient of the cubes.
    - diaginertia_factor: factor used to calculate the moment of inertia.
    """
    # If the positions are not specified, generate a series of offset cubes
    if cube_positions is None:
        cube_positions = [(i * cube_size, 0, 0) for i in range(cube_count)]
    if cube_rotations is None:
        cube_rotations = [(0, 0, 0) for _ in range(cube_count)]

    # Check if the lengths of cube_positions and cube_rotations match cube_count
    if len(cube_positions) != cube_count:
        raise ValueError("The length of cube_positions should match cube_count.")
    if len(cube_rotations) != cube_count:
        raise ValueError("The length of cube_rotations should match cube_count.")

    # If the masses are not specified, give all cubes the same mass
    if cube_masses is None:
        cube_masses = [1.0] * cube_count
    elif len(cube_masses) != cube_count:
        raise ValueError("The length of cube_masses should match cube_count.")

    # XML file content starts
    xml_content = '''<mujoco model="complex_object">\n'''

    # Generate the main <body> that contains all the cubes
    xml_content += '''    <body name="complex_object" pos="0 0 0">\n'''
    xml_content += '''        <joint type="free" name="complex_object_jnt"/>\n'''

    # Generate the XML definition for each cube, put each cube into an independent <body>
    for i, ((x, y, z), (rx, ry, rz)) in enumerate(zip(cube_positions, cube_rotations)):
        mass = cube_masses[i]
        inertial_value = mass * diaginertia_factor
        axisangle = f'{rx} {ry} {rz} {1.0}' if any([rx, ry, rz]) else "0 0 1 0"

        xml_content += f'        <body name="cube_{i+1}" pos="{x} {y} {z}" axisangle="{axisangle}">\n'
        xml_content += f'            <geom class="visual" name="cube_visual_{i+1}" mesh="unit_cube" size="{cube_size/2}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" friction="{friction}"/>\n'
        xml_content += f'            <geom class="collision" name="cube_collision_{i+1}" mesh="unit_cube" size="{cube_size/2}" friction="{friction}"/>\n'
        xml_content += f'            <inertial mass="{mass}" pos="0 0 0" diaginertia="{inertial_value} {inertial_value} {inertial_value}"/>\n'
        xml_content += '        </body>\n'

    # XML file content ends
    xml_content += '''    </body>\n</mujoco>'''

    # Write the content to the file
    with open(filename, 'w') as file:
        file.write(xml_content)

    print(f"XML file has been saved as {filename}")

def continue_1D_object(filename, cube_size, random_seed=10):
    """
    Generate a 1D object, with the center of mass randomly distributed along the long axis.
    
    Parameters:
    - filename: name of the output XML file.
    - cube_size: size of the object.
    """
    # XML file content starts
    xml_content = '''<mujoco model="1d_object">\n'''
    
    # Generate the main <body>
    xml_content += '''    <body name="1d_object" pos="0 0 0">\n'''
    xml_content += '''        <joint type="free" name="1d_object_jnt"/>\n'''
    
    # Generate a complete rectangular object
    length = cube_size * 5  # Total length
    width = cube_size       # Width and height keep cube_size
    height = cube_size
    
    # Add visual and collision geometry
    xml_content += f'        <geom class="visual" name="object_visual" type="box" size="{length/2} {width/2} {height/2}" rgba="0.8 0.2 0.2 1" friction="1.0"/>\n'
    xml_content += f'        <geom class="collision" name="object_collision" type="box" size="{length/2} {width/2} {height/2}" friction="1.0"/>\n'
    
    # Set the center of mass position - randomly offset along the x-axis
    np.random.seed(random_seed)
    com_offset_x = np.random.uniform(-length/2, length/2)  # Randomly offset within the entire length
    mass = 0.5  # Total mass is 0.5kg
    inertial_value = mass * 0.01  # Inertia scaling factor
    
    # Add inertia properties, change the center of mass by setting the com position
    xml_content += f'        <inertial mass="{mass}" pos="{com_offset_x} 0 0" diaginertia="{inertial_value} {inertial_value} {inertial_value}"/>\n'
    
    # XML file content ends
    xml_content += '''    </body>\n</mujoco>'''
    
    # Write the content to the file
    with open(filename, 'w') as file:
        file.write(xml_content)
        
    print(f"1D object with center of mass offset has been generated: {filename}")
    print(f"Center of mass x-axis offset: {com_offset_x}")




