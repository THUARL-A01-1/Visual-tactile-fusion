import gymnasium as gym
import tactile_envs
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning) # ignore the warning of mujoco

def parse_arguments():
    """
    Parse command-line arguments for the tactile environment testing.
    
    Returns:
    argparse.Namespace: Parsed arguments with their values.
    """
    parser = argparse.ArgumentParser(description="Test the gripping environment with tactile information")

    # Adding arguments
    parser.add_argument("--n_episodes", type=int, default=100, help="Number of episodes to run (default: 100)")
    parser.add_argument("--n_steps", type=int, default=300, help="Number of steps per episode (default: 300)")
    parser.add_argument("--show_highres", action='store_true', help="Show high-resolution images (default: False)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generator (default: None)")
    parser.add_argument("--env_name", type=str, default="tactile_envs/Regrasp-v0", help="Environment name (default: tactile_envs/Regrasp-v0)")
    parser.add_argument("--state_type", type=str, default="vision_and_touch", help="Type of state information (default: vision_and_touch), options: vision, vision_and_touch, touch, privileged")
    parser.add_argument("--multiccd", action='store_true', help="Enable multi-CCD (default: False)")
    parser.add_argument("--im_size", type=int, default=480, help="Image size for the environment (default: 480)")
    parser.add_argument("--no_gripping", action='store_true', help="Disable gripping (default: False)")
    parser.add_argument("--no_rotation", action='store_true', help="Disable rotation (default: False)")
    parser.add_argument("--tactile_shape", type=int, nargs=2, default=(20, 20), help="Shape of tactile sensor (height, width) (default: (20, 20))")
    parser.add_argument("--max_delta", type=float, default=None, help="Maximum delta for tactile data (default: None)")

    return parser.parse_args()

def test_env(n_episodes=100, n_steps=300, show_highres=False, seed=42, 
                  env_name="tactile_envs/Regrasp-v0", state_type='vision_and_touch',
                  multiccd=False, im_size=480, no_gripping=False, no_rotation=False,
                  tactile_shape=(20,20), max_delta=None):
    """
    Function to test the gripping with tactile information in a given environment.

    Parameters:
    - n_episodes (int): Number of episodes to run.
    - n_steps (int): Maximum steps per episode.
    - show_highres (bool): Whether to show high-resolution images.
    - seed (int or None): Random seed for environment reset. If None, a random seed is generated.
    - env_name (str): The name of the environment to create.
    - state_type (str): Type of state information used (e.g., 'vision_and_touch').
    - multiccd (bool): Whether to enable multi-CCD.
    - im_size (int): Image size used in the environment.
    - no_gripping (bool): Whether to disable gripping.
    - no_rotation (bool): Whether to disable rotation.
    - tactile_shape (tuple): Shape of the tactile sensor input.
    - max_delta (float or None): Maximum delta for tactile data.
    """

    def print_parameters():
        # Print all the parameters to show what values are being used
        print("\nRunning with the following configuration")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"env_name: {env_name}")
        print(f"state_type: {state_type}")
        print(f"multiccd: {multiccd}")
        print(f"no_gripping: {no_gripping}")
        print(f"no_rotation: {no_rotation}")
        print(f"show_highres: {show_highres}")
        print(f"random seed: {seed}")
        print(f"n_episodes: {n_episodes}")
        print(f"n_steps: {n_steps}")
        print(f"tactile_shape: {tactile_shape}")
        print(f"im_size: {im_size}")
        print(f"max_delta: {max_delta}")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    
    print_parameters()

    env = gym.make(env_name, state_type=state_type, multiccd=multiccd, im_size=im_size, 
                   no_gripping=no_gripping, no_rotation=no_rotation, 
                   tactile_shape=tactile_shape, max_delta=max_delta)

    np.random.seed(seed)  # Set the random seed for numpy operations
    
    for j in range(n_episodes):
        env.reset()
        tic = time.time()
        print("\nEpisode ", j+1, " started.")

        for i in range(n_steps):
            print("[step:", i,"], phase:", env.grasp_sequence[env.grasp_phase])
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action) # execute the action

            tactile_data = obs['tactile']
            img_data = obs['image']

            temp_image_save_path = f"data/temp_obs/obs_{j}_{i}_image.npy"
            temp_tactile_save_path = f"data/temp_obs/obs_{j}_{i}_tactile.npy"

            # save the image data
            np.save(temp_image_save_path, img_data)
            print(f"Obs image saved to \"{temp_image_save_path}\"")

            # save the tactile data
            np.save(temp_tactile_save_path, tactile_data)
            print(f"Obs tactile saved to \"{temp_tactile_save_path}\"")

            # tactile_data shape is (6,20,20), the first 3 channels are the right tactile sensors, 
            # the last 3 channels are the left tactile sensors
            # y direction tactile sum
            y_sum_right = np.sum(tactile_data[1,:,:]) # right sensor y direction sum
            y_sum_left = np.sum(tactile_data[4,:,:])  # left sensor y direction sum
            total_y_sum = y_sum_right + y_sum_left
            print("Y direction tactile sum: ", -total_y_sum, "N (Gravity direction)")
            # x direction tactile sum
            x_sum_right = np.sum(tactile_data[0,:,:]) # right sensor x direction sum
            x_sum_left = np.sum(tactile_data[3,:,:])  # left sensor x direction sum
            total_x_sum = x_sum_right + x_sum_left
            print("X direction tactile sum: ", total_x_sum, "N")
            # z direction tactile sum
            z_sum_right = np.sum(tactile_data[2,:,:]) # right sensor z direction sum
            z_sum_left = np.sum(tactile_data[5,:,:])  # left sensor z direction sum
            total_z_avg = (z_sum_right + z_sum_left)/2
            print("Z direction tactile avg: ", total_z_avg, "N")
            
            
            done = terminated or truncated # check if the episode is done
            if done:
                print(f"Episode {j+1} done after {i+1} steps.")
                break
        
        toc = time.time() # time cost of the episode
        print(f"Episode {j+1} completed in {toc - tic:.2f} seconds.\n")
    
    if show_highres:
        print("Displaying high-resolution images (Note: This might be slow).")
        # Add any additional code to display highres images here if needed.

def main():
    ''' Main function to test the gripping with tactile information in the environment. '''
    # Parse the arguments
    args = parse_arguments()

    # Call the main function with parsed arguments
    test_env(n_episodes=args.n_episodes, n_steps=args.n_steps, show_highres=args.show_highres, 
                  seed=args.seed, env_name=args.env_name, state_type=args.state_type, 
                  multiccd=args.multiccd, im_size=args.im_size, no_gripping=args.no_gripping, 
                  no_rotation=args.no_rotation, tactile_shape=tuple(args.tactile_shape), 
                  max_delta=args.max_delta)


if __name__ == "__main__":
    main()


