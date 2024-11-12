'''
RLGraspEnv1D 类定义了1维抓取环境
'''
import gymnasium as gym
import numpy as np
from tactile_envs.envs.insertion import RegraspEnv

class RLGraspEnv1D(RegraspEnv):
    def __init__(self, **kwargs):
        # 继承 RegraspEnv 的初始化
        super().__init__(**kwargs)
        
        # # 定义动作空间 (与原始 RegraspEnv 保持一致)
        # self.action_space = self.action_space
        # 重新定义动作空间为一维 (仅x方向)
        self.action_space = gym.spaces.Box(
            low=self.action_space.low[0:1],  # 只取x方向的最小值
            high=self.action_space.high[0:1], # 只取x方向的最大值
            shape=(1,),  # 一维空间
            dtype=np.float32
        )
        
        # 定义观察空间 (根据 state_type 定义)
        self.observation_space = self.observation_space

    def reset(self, seed=None, options=None):
        # 设置随机种子
        if seed is not None:
            self.seed(seed)
            
        # 调用父类的 reset
        obs, info = super().reset()
        
        return obs, info

    def step(self, action):
        # 将一维动作转换为原始环境需要的多维动作格式
        full_action = np.zeros(super().action_space.shape[0], dtype=np.float32)
        full_action[0] = action[0]  # 只设置x方向的值
        
        # 调用父类的 step
        obs, reward, done, info = super().step(full_action)
        
        # 添加 truncated 参数以符合 Gymnasium 接口
        truncated = False
        
        return obs, reward, done, truncated, info

    def render(self):
        # 调用父类的 render
        return super().render()

    def close(self):
        # 实现环境清理
        pass
        # super().close()