import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback

from models.custom_policies import CustomPPOPolicy
from tactile_envs.envs import RLGraspEnv1D, RLGraspEnv2D # 导入环境类

def make_env(rank, seed=0):
    """
    创建环境的辅助函数
    """
    def _init():
        try:
            env = RLGraspEnv1D()
            env.reset()
            return env
        except Exception as e:
            print(f"环境 {rank} 初始化错误: {str(e)}")
            raise e
    return _init

def main():
    # 设置实验名称和日志目录
    exp_name = "ppo_grasp"
    log_dir = f"./logs/{exp_name}"
    model_dir = f"./models/{exp_name}"
    
    # 创建必要的目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 初始化wandb
    wandb.init(
        project="grasp-rl",
        name=exp_name,
        config={
            "algorithm": "PPO",
            "n_envs": 8,
            "n_steps": 2048,
            "batch_size": 128,
            "n_epochs": 10,
            "learning_rate": 1e-4,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "total_timesteps": 100000,
            "action_space": "1D",  # 添加这行来记录动作空间的变化
            "action_type": "x_only"  # 添加这行来说明只使用x方向
        }
    )
    
    ############################################################
    # 先尝试使用单个环境
    n_envs = 1  # 临时改为1个环境
    try:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        print("环境创建成功！")
    except Exception as e:
        print(f"创建环境时出错: {str(e)}")
        raise e

    # 创建多进程环境
    # n_envs = 8
    # env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    ############################################################
    
    # 创建评估环境
    eval_env = Monitor(RLGraspEnv1D())
    
    # 设置回调函数
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model",
        log_path=log_dir,
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="ppo_grasp_model"
    )
    
    # 创建PPO模型
    model = PPO(
        CustomPPOPolicy,
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        device="cuda"  # 如果有GPU的话使用GPU
    )
    
    # 训练模型
    try:
        model.learn(
            total_timesteps=100000,
            callback=[
                eval_callback,
                checkpoint_callback,
                WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f"{model_dir}/wandb/{exp_name}",
                    verbose=2
                )
            ]
        )
    except KeyboardInterrupt:
        print("训练被手动中断")
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 关闭环境
    env.close()
    eval_env.close()
    wandb.finish()

def evaluate_model(model_path, n_eval_episodes=100):
    """
    评估已训练的模型
    """
    env = RLGraspEnv1D()
    model = PPO.load(model_path)
    
    success_count = 0
    total_rewards = 0
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if info.get("success", False):
                success_count += 1
        
        total_rewards += episode_reward
    
    success_rate = success_count / n_eval_episodes
    mean_reward = total_rewards / n_eval_episodes
    
    print(f"评估结果 ({n_eval_episodes} 回合):")
    print(f"成功率: {success_rate:.3f}")
    print(f"平均奖励: {mean_reward:.3f}")
    
    return success_rate, mean_reward

if __name__ == "__main__":
    # 训练模型
    main()
    
    # 评估最终模型
    model_path = "./data/trained_models/ppo_grasp/final_model.zip"
    if os.path.exists(model_path):
        print("\n评估最终模型:")
        evaluate_model(model_path)