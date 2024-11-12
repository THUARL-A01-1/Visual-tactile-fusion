from tactile_envs.assets.insertion.complex_objects.shape_generator import generate_complex_object_xml



if __name__ == "__main__":
    # The unit of object mass is kg
    # The gravity of the object is mass * 9.8

    # I-shaped object
    # I1: Uniform mass distribution, total mass is 0.5kg
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/I1.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(-0.06, 0, 0), (-0.03, 0, 0), (0, 0, 0), (0.03, 0, 0), (0.06, 0, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.1, 0.1, 0.1, 0.1, 0.1] 
    )
    ## I2: The 1st/5th block is the heaviest
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/I2.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(-0.06, 0, 0), (-0.03, 0, 0), (0, 0, 0), (0.03, 0, 0), (0.06, 0, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.3, 0.05, 0.05, 0.05, 0.05]
    )

    ## I3: The 2nd/4th block is the heaviest
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/I3.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(-0.06, 0, 0), (-0.03, 0, 0), (0, 0, 0), (0.03, 0, 0), (0.06, 0, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.05, 0.3, 0.05, 0.05, 0.05]
    )
    ## I4: The 3rd block is the heaviest
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/I4.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(-0.06, 0, 0), (-0.03, 0, 0), (0, 0, 0), (0.03, 0, 0), (0.06, 0, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.05, 0.05, 0.3, 0.05, 0.05]
    )
    ## I5: The 3rd/4th block is the heaviest
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/I5.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(-0.06, 0, 0), (-0.03, 0, 0), (0, 0, 0), (0.03, 0, 0), (0.06, 0, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.05, 0.05, 0.175, 0.175, 0.05]
    )


    # ------------------------------------------------------------------------------------------------
    # T-shaped object
    ## T1: Uniform mass distribution
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/T1.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(0, 0, 0), (0.03, 0, 0), (-0.03, 0, 0), (0.03, 0.03, 0), (0.03, -0.03, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.1, 0.1, 0.1, 0.1, 0.1]
    )
    ## T2: The 5th block is the heaviest
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/T2.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(0, 0, 0), (0.03, 0, 0), (-0.03, 0, 0), (0.03, 0.03, 0), (0.03, -0.03, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.3, 0.05, 0.05, 0.05, 0.05]
    )
    ## T3: The 1st block is the heaviest
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/T3.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(0, 0, 0), (0.03, 0, 0), (-0.03, 0, 0), (0.03, 0.03, 0), (0.03, -0.03, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.05, 0.05, 0.05, 0.05, 0.3]
    )
    ## T4: The 3rd block is the heaviest
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/T4.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(0, 0, 0), (0.03, 0, 0), (-0.03, 0, 0), (0.03, 0.03, 0), (0.03, -0.03, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.05, 0.05, 0.3, 0.05, 0.05]
    )
    ## T5: The 3rd/4th block is the heaviest
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/T5.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(0, 0, 0), (0.03, 0, 0), (-0.03, 0, 0), (0.03, 0.03, 0), (0.03, -0.03, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.05, 0.05, 0.175, 0.175, 0.05]
    )
    # ------------------------------------------------------------------------------------------------

    # L-shaped object
    ## L1: Uniform mass distribution
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/L1.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(0, 0, 0), (0.03, 0, 0), (-0.03, 0, 0), (0.03, 0.03, 0), (0.03, 0.06, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.1, 0.1, 0.1, 0.1, 0.1]
    )
    ## L2: The 1st block is the heaviest
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/L2.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(0, 0, 0), (0.03, 0, 0), (-0.03, 0, 0), (0.03, 0.03, 0), (0.03, 0.06, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.3, 0.05, 0.05, 0.05, 0.05]
    )
    ## L3: The 3rd block is the heaviest
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/L3.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(0, 0, 0), (0.03, 0, 0), (-0.03, 0, 0), (0.03, 0.03, 0), (0.03, 0.06, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.05, 0.05, 0.3, 0.05, 0.05]
    )  
    ## L4: The 3rd/4th block is the heaviest
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/L4.xml",
        cube_count=5,
        cube_size=0.04,
        cube_positions=[(0, 0, 0), (0.03, 0, 0), (-0.03, 0, 0), (0.03, 0.03, 0), (0.03, 0.06, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.05, 0.05, 0.175, 0.175, 0.05]
    )         
    ## L5: The 2nd block is the heaviest
    generate_complex_object_xml(
        filename="tactile_envs/assets/insertion/complex_objects/discrete_2d/L5.xml",
        cube_count=5,
        cube_size=0.04, 
        cube_positions=[(0, 0, 0), (0.03, 0, 0), (-0.03, 0, 0), (0.03, 0.03, 0), (0.03, 0.06, 0)],
        cube_rotations=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        cube_masses=[0.05, 0.3, 0.05, 0.05, 0.05]
    )           
