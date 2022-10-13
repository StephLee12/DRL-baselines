import numpy as np 

import torch
import torch.nn as nn 
import torch.optim as optim

from net import TD3_PolicyNetwork,TD3_QNetwork

class TD3_Agent():
    def __init__(
        self,
        obs_dim, 
        action_dim, 
        device,
        replay_buffer, 
        action_space,
        hidden_dim=64,
        action_range=1.0,
        policy_delay_update_interval=3
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.replay_buffer = replay_buffer
        self.action_space = action_space
        self.policy_delay_update_interval = policy_delay_update_interval

        self.q_net1 = TD3_QNetwork(obs_dim,action_dim,hidden_dim).to(self.device)
        self.q_net2 = TD3_QNetwork(obs_dim,action_dim,hidden_dim).to(self.device)
        self.target_q_net1 = TD3_QNetwork(obs_dim,action_dim,hidden_dim).to(self.device)
        self.target_q_net2 = TD3_QNetwork(obs_dim,action_dim,hidden_dim).to(self.device)
        self.policy_net = TD3_PolicyNetwork(device,action_space,obs_dim,action_dim,hidden_dim,action_range).to(device)
        self.target_policy_net = TD3_PolicyNetwork(device,action_space,obs_dim,action_dim,hidden_dim,action_range).to(self.device)

        for target_param,param in zip(self.target_q_net1.parameters(),self.q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param,param in zip(self.target_q_net2.parameters(),self.q_net2.parameters()):
            target_param.data.copy_(param.data)
        for target_param,param in zip(self.target_policy_net.parameters(),self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        q_lr = 3e-4
        policy_lr = 3e-4
        self.update_cnt = 0

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
    
    def target_soft_update(self, net, target_net, soft_tau):
        # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return target_net
    
    def update(self, batch_size, deterministic, eval_noise_scale, reward_scale=10., gamma=0.9,soft_tau=1e-2):
        self.update_cnt += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        q_val1 = self.q_net1(state, action)
        q_val2 = self.q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, _, _, _, _ = self.target_policy_net.evaluate(next_state, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        # reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_min = torch.min(self.target_q_net1(next_state, new_next_action),self.target_q_net2(next_state, new_next_action))
        target_q_val = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((q_val1 - target_q_val.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((q_val2 - target_q_val.detach())**2).mean()
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.q_optimizer1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.q_optimizer2.step()

        if self.update_cnt % self.policy_delay_update_interval == 0:
            # This is the **Delayed** update of policy and all targets (for Q and policy). 
            # Training Policy Function
            ''' implementation 1 '''
            # new_q_val = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            new_q_val = self.q_net1(state, new_action)

            policy_loss = - new_q_val.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
            # Soft update the target nets
            self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        return q_val1.mean()
    
    def save_model(self, path):
        torch.save(self.q_net1.state_dict(), path+'_q1')
        torch.save(self.q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.q_net1.load_state_dict(torch.load(path+'_q1'))
        self.q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))
        self.q_net1.eval()
        self.q_net2.eval()
        self.policy_net.eval()


