'''
自定义策略网络，继承自 ActorCriticPolicy
'''
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
from torch import nn
import numpy as np
from torch.distributions import Normal

class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        **kwargs
    ):
        super(CustomPPOPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
        
        # 网络参数
        self.net_arch = [dict(pi=[64, 128, 64], vf=[64, 128, 64])]
        
        # PPO特定参数
        self.clip_range = 0.2
        self.ent_coef = 0.01
        self.learning_rate = 1e-4
        self.n_steps = 2048  # 每次更新的步数
        self.batch_size = 128
        self.n_epochs = 10   # 每批数据的训练轮数
        
    def forward(self, obs, deterministic=False):
        """前向传播"""
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        mean_actions = self.action_net(latent_pi)
        
        # 获取动作分布
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        if deterministic:
            actions = mean_actions
        else:
            actions = distribution.sample()
            
        # 添加动作噪声以提高鲁棒性
        if not deterministic:
            noise = th.normal(
                mean=th.zeros_like(actions),
                std=0.01
            )
            actions = actions + noise
            
        return actions, None
    
    def evaluate_actions(self, obs, actions):
        """评估动作"""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        
        return values, log_prob, entropy