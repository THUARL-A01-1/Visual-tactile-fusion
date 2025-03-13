"""
暂时不用处理此脚本，用于测试随机抓取环境
"""

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

def print_parameters(env_name, state_type, multiccd, no_gripping, no_rotation, 
                    show_highres, seed, n_episodes, n_steps, tactile_shape, 
                    im_size, max_delta):
    """打印运行参数"""
    print("\nconfig:")
    print("="*50)
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
    print("="*50 + "\n")

def test_env(n_episodes=100, n_steps=300, show_highres=False, seed=42, 
             env_name="tactile_envs/Regrasp-v0", state_type='vision_and_touch',
             multiccd=False, im_size=480, no_gripping=False, no_rotation=False,
             tactile_shape=(20,20), max_delta=None):
    
    print_parameters(env_name, state_type, multiccd, no_gripping, no_rotation,
                    show_highres, seed, n_episodes, n_steps, tactile_shape,
                    im_size, max_delta)
    
    try:
        print("creating environment...")
        env = gym.make(env_name, state_type=state_type, multiccd=multiccd, im_size=im_size, 
                      no_gripping=no_gripping, no_rotation=no_rotation, 
                      tactile_shape=tactile_shape, max_delta=max_delta)
        
        print("environment created successfully, start testing...")
        for episode in range(n_episodes):
            print(f"\nstart episode {episode + 1}")
            print("="*50)
            
            obs, info = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # 生成随机动作
                action = env.action_space.sample()
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                # 渲染环境
                if show_highres:
                    env.render()
            
            print(f"episode {episode + 1} finished")
            print(f"total reward: {episode_reward}")
            print("="*50)
            
    except Exception as e:
        print(f"error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("\ntesting finished")

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


