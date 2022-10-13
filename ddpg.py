import numpy as np 

import torch
import torch.nn as nn 
import torch.optim as optim

from net import DDPG_PolicyNetwork,DDPG_QNetwork

class DDPG_Agent():
    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        replay_buffer,
        action_space,
        hidden_dim=64,
        action_range=1.
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.replay_buffer = replay_buffer
        self.action_space = action_space

        self.qnet = DDPG_QNetwork(obs_dim,action_dim,hidden_dim).to(device)
        self.target_qnet = DDPG_QNetwork(obs_dim,action_dim,hidden_dim).to(device)
        self.policy_net = DDPG_PolicyNetwork(device,action_space,obs_dim,action_dim,hidden_dim,action_range).to(device)
        self.target_policy_net = DDPG_PolicyNetwork(device,action_space,obs_dim,action_dim,hidden_dim,action_range).to(device)

        for target_param,param in zip(self.target_qnet.parameters(),self.qnet.parameters()):
            target_param.data.copy_(param.data)
        
        self.q_criterion = nn.MSELoss()
        q_lr = 8e-4
        policy_lr = 8e-4
        self.update_cnt = 0

        self.q_optimizer = optim.Adam(self.qnet.parameters(),lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(),lr=policy_lr)

    def target_soft_update(self, net, target_net, soft_tau):
        # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return target_net
    
    def update(self,batch_size,noise_scale,reward_scale=10.0, gamma=0.99, soft_tau=1e-2, policy_up_itr=10, target_update_delay=3, warmup=True):
        self.update_cnt += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        predict_q = self.qnet(state, action) # for q 
        new_next_action = self.target_policy_net.evaluate(next_state,noise_scale=noise_scale)  # for q
        new_action = self.policy_net.evaluate(state,noise_scale=noise_scale) # for policy
        predict_new_q = self.qnet(state, new_action) # for policy
        target_q = reward+(1-done)*gamma*self.target_qnet(next_state, new_next_action)  # for q
        # reward = reward_scale * (reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std

        # train qnet
        q_loss = self.q_criterion(predict_q, target_q.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # train policy_net
        policy_loss = -torch.mean(self.qnet(state,new_action))
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # update the target_qnet
        if self.update_cnt % target_update_delay ==0:
            self.target_qnet=self.target_soft_update(self.qnet, self.target_qnet, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        return q_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path+'_q')
        torch.save(self.target_qnet.state_dict(), path+'_target_q')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path+'_q'))
        self.target_qnet.load_state_dict(torch.load(path+'_target_q'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))
        self.qnet.eval()
        self.target_qnet.eval()
        self.policy_net.eval()