"""
Generate a continuous 1D object with random center of mass distribution.
"""
from tactile_envs.assets.insertion.complex_objects.shape_generator import continue_1D_object

if __name__ == "__main__":
    num_objs = 10
    for i in range(num_objs):
        continue_1D_object(
            filename=f"tactile_envs/assets/insertion/complex_objects/continue_1d/obj_{i}.xml",
            cube_size=0.04,
            random_seed=i
        )