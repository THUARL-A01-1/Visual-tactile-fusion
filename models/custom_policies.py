'''
Policy Network inherit from ActorCriticPolicy
'''
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
from torch import nn
import numpy as np

class HistoryEncoder(nn.Module):
    """LSTM Network for encoding history failure experiences"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
    def forward(self, x, lengths):
        """
        Args:
            x: [batch_size, max_len, input_dim] history experience sequence
            lengths: [batch_size] actual sequence length
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        output, (hidden, cell) = self.lstm(packed)
        
        return hidden[-1]

class GraspActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
        
        # history experience encoder
        self.history_encoder = HistoryEncoder(
            input_dim=observation_space["current_obs"].shape[0] + action_space.shape[0],
            hidden_dim=128
        )
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(128 + observation_space["current_obs"].shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.shape[0])
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(128 + observation_space["current_obs"].shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, obs):
        current_obs = obs["current_obs"]

        history = obs["history"]  # [batch_size, max_attempts, feature_dim]
        history_lengths = obs["history_lengths"]  # [batch_size]
        
        history_encoding = self.history_encoder(history, history_lengths)

        combined_features = torch.cat([current_obs, history_encoding], dim=1)

        action_mean = self.actor(combined_features)
        action_logstd = self.log_std

        values = self.critic(combined_features)
        
        return action_mean, action_logstd, values

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
        
        self.net_arch = [dict(pi=[64, 128, 64], vf=[64, 128, 64])]
        
        self.clip_range = 0.2
        self.ent_coef = 0.01
        self.learning_rate = 1e-4
        self.n_steps = 2048  
        self.batch_size = 128
        self.n_epochs = 10      
        
    def forward(self, obs):
        """
        Forward pass in all the networks (actor and critic)
        """
        features = self.extract_features(obs)
        
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)
        
        values = self.value_net(latent_vf)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions()
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob    
    
    def evaluate_actions(self, obs, actions):
        """评估动作"""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        
        return values, log_prob, entropy
    
class CustomSACPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.net_arch = [dict(pi=[64, 128, 64], vf=[64, 128, 64])]
        self.learning_rate = 1e-4
        self.n_steps = 2048  
        self.batch_size = 128
        self.n_epochs = 10      

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                        net_arch=[dict(pi=[256, 256], vf=[256, 256])])
