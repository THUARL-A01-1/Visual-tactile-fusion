'''
PPO使用的网络架构
'''
import torch
import torch.nn as nn

class SharedMLPNetwork(nn.Module):
    """PPO使用的共享特征提取器"""
    def __init__(self, input_dim):
        super(SharedMLPNetwork, self).__init__()
        
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.shared_net(x)

class PolicyNetwork(nn.Module):
    """PPO的策略网络"""
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        
        self.shared_net = SharedMLPNetwork(input_dim)
        
        # 动作均值输出层
        self.action_mean = nn.Linear(64, action_dim)
        # 动作标准差输出层 (log_std以确保标准差为正)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, x):
        features = self.shared_net(x)
        action_mean = self.action_mean(features)
        action_std = self.action_log_std.exp()
        return action_mean, action_std

class ValueNetwork(nn.Module):
    """PPO的价值网络"""
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        
        self.shared_net = SharedMLPNetwork(input_dim)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x):
        features = self.shared_net(x)
        value = self.value_head(features)
        return value