import os
import numpy as np
import torch
import torch.nn as nn
import wandb
from tactile_envs.envs import RLGraspEnv1D

class FailureHistoryPolicy(nn.Module):
    """Policy Network based on failure history"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # LSTM encoder for processing failure history
        self.lstm = nn.LSTM(
            input_size=state_dim + action_dim,  # state dimension + action dimension
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
        # current state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # policy network
        self.policy_net = nn.Sequential(
            nn.Linear(256, 128),  # 256 = 128(lstm) + 128(state)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # value network
        self.value_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state, failure_history):
        """
        Args:
            state: current state
            failure_history: failure history sequence [batch_size, seq_len, state_dim + action_dim]
        """
        # encode current state
        state_features = self.state_encoder(state)
        
        # encode failure history
        if len(failure_history) > 0:
            history_features, _ = self.lstm(failure_history)
            history_features = history_features[:, -1, :]  # take the last time step
        else:
            history_features = torch.zeros(state.size(0), 128, device=state.device)
        
        # merge features
        combined_features = torch.cat([state_features, history_features], dim=1)
        
        # output action and value
        action_mean = self.policy_net(combined_features)
        value = self.value_net(combined_features)
        
        return action_mean, value

class FailureHistoryTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_env()
        self.setup_model()
        self.setup_wandb()
        
    def setup_env(self):
        """Initialize environment"""
        try:
            self.env = RLGraspEnv1D(
                no_rotation=True,
                no_gripping=False,
                state_type='vision_and_touch',
                camera_idx=0,
                symlog_tactile=True,
                im_size=480,
                tactile_shape=(20, 20),
                skip_frame=5
            )
            
            # get object list and initialize current object index
            self.object_list = self.env.objects  # get object list
            self.current_obj_idx = 0  # initialize object index
            print(f"\nobject list: {self.object_list}")
            
            test_state, _ = self.env.reset()
            print("\ntest environment:")
            print(f"state type: {type(test_state)}")
            if isinstance(test_state, dict):
                print(f"state keys: {test_state.keys()}")
                print(f"tactile shape: {test_state['tactile'].shape}")
            
        except Exception as e:
            print(f"environment initialization failed: {e}")
            raise
        
    def setup_model(self):
        """Initialize policy network"""
        # get state and action dimension
        obs_space = self.env.observation_space
        
        # calculate state dimension (only keep tactile feature)
        tactile_dim = np.prod(obs_space["tactile"].shape)
        state_dim = tactile_dim
        
        # get action dimension
        action_dim = self.env.action_space.shape[0]
        
        print(f"\nstate space information:")
        print(f"tactile dimension: {obs_space['tactile'].shape}")
        print(f"merged state dimension: {state_dim}")
        print(f"action dimension: {action_dim}")
        
        self.policy = FailureHistoryPolicy(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        
    def setup_wandb(self):
        """Initialize wandb"""
        wandb.init(
            project=self.config['project_name'],
            name=self.config['exp_name'],
            config=self.config
        )
        
    def train_episode(self):
        """Train a single episode"""
        # update current object
        self.env.object = self.object_list[self.current_obj_idx]
        print(f"\ncurrent training object: {self.env.object}")
        self.current_obj_idx = (self.current_obj_idx + 1) % len(self.object_list)
        
        failure_history = []  # current object's failure history
        episode_reward = 0
        max_attempts = 10
        current_attempt = 0
        
        def process_state(state):
            """Process state, convert it to a one-dimensional array"""
            if 'tactile' in state:
                return state['tactile'].flatten()
            raise ValueError("state lacks tactile information")
        
        # try multiple times for the current object
        while current_attempt < max_attempts:
            current_attempt += 1
            print(f"\nstart the {current_attempt}/{max_attempts}th grasp attempt")
            
            # reset environment for each attempt
            state, _ = self.env.reset()
            if state is None:
                print("warning: environment reset returned None state")
                continue
            
            # generate action: first attempt use the object center position, then based on failure history
            if len(failure_history) == 0:
                action = np.zeros(self.env.action_space.shape[0])
                print("first attempt: use the object center position")
            else:
                try:
                    # use current state and failure history to predict new grasp position
                    history_tensor = torch.FloatTensor(failure_history).unsqueeze(0)
                    state_tensor = torch.FloatTensor(process_state(state)).unsqueeze(0)
                    
                    with torch.no_grad():
                        action_mean, _ = self.policy(state_tensor, history_tensor)
                        action = action_mean.numpy()[0]
                        # add some noise to explore
                        action += np.random.normal(0, 0.01, size=action.shape)
                    print(f"predict new position based on failure history: {action}")
                except Exception as e:
                    print(f"error when predicting new position: {e}")
                    action = np.zeros(self.env.action_space.shape[0])
            
            # execute grasp attempt
            next_state, reward, done, _, info = self.env.step(action)
            episode_reward += reward
            
            print(f"grasp result:")
            print(f"position: {action}")
            print(f"reward: {reward:.3f}")
            print(f"success: {info.get('is_success', False)}")
            
            # if success, update policy and end training for current object
            if info.get('is_success', False):
                print(f"grasp success! attempts: {current_attempt}")
                if len(failure_history) > 0:
                    self.update_policy(failure_history, reward)
                return True, current_attempt, episode_reward
            
            # record failure experience (state and corresponding grasp position)
            try:
                state_array = process_state(state)
                failure_exp = np.concatenate([state_array, action])
                failure_history.append(failure_exp)
                print(f"record failure position and tactile feedback")
            except Exception as e:
                print(f"error when recording failure experience: {e}")
            
            print(f"failure reason: {info.get('failure_reason', 'unknown')}")
        
        print(f"reach max attempts ({max_attempts}), training failed")
        return False, max_attempts, episode_reward
    
    def update_policy(self, failure_history, final_reward):
        """Update policy network"""
        history_tensor = torch.FloatTensor(failure_history).unsqueeze(0)
        
        # compute discounted rewards for each time step
        rewards = np.zeros(len(failure_history))
        rewards[-1] = final_reward
        discounted_rewards = self.compute_returns(rewards, gamma=0.99)
        
        # convert to tensor
        returns = torch.FloatTensor(discounted_rewards).unsqueeze(1)
        
        # compute loss
        state_tensor = torch.FloatTensor(failure_history)[:, :self.env.observation_space.shape[0]]
        action_mean, values = self.policy(state_tensor.unsqueeze(0), history_tensor)
        
        # compute policy loss and value loss
        advantage = returns - values
        policy_loss = -(advantage.detach() * action_mean).mean()
        value_loss = advantage.pow(2).mean()
        
        # total loss
        loss = policy_loss + 0.5 * value_loss
        
        # update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def compute_returns(self, rewards, gamma=0.99):
        """计算折扣奖励"""
        returns = np.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        return returns
    
    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            success, attempts, episode_reward = self.train_episode()

            wandb.log({
                'episode': episode,
                'success': success,
                'attempts': attempts,
                'episode_reward': episode_reward
            })
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Success={success}, Attempts={attempts}, Reward={episode_reward:.2f}")
    
    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))

def main():
    config = {
        "exp_name": "grasp_from_failure_v1",
        "project_name": "grasp-from-failure",
        "num_episodes": 100,
        "max_attempts": 20,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    trainer = FailureHistoryTrainer(config)
    trainer.train(num_episodes=config['num_episodes'])
    
    # save final model
    trainer.save_model(f"models/{config['exp_name']}/final_model.pth")

if __name__ == "__main__":
    main() 