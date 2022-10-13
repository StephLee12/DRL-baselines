import numpy as np 

import torch
import torch.nn as nn 
import torch.optim as optim

from net import PPO_PolicyNetwork,PPO_ValueNetwork

class PPO_Agent():
    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        action_space,
        duration_t,
        hidden_dim=64,
        action_range=1.,
        alpha_1=100,
        alpha_2=100,
        mode='ES',
        min_soc=0,
        max_soc=1,
        energy_capacity=10
    ) -> None:
        self.device = device 
        self.action_space = action_space
        self.action_range = action_range
        self.alpha_1 = alpha_1 # coeff for regularizer 1
        self.alpha_2 = alpha_2 # coeff for regularizer 2

        self.mode = mode
        self.duration_t = duration_t
        self.energy_capacity = energy_capacity
        self.min_soc = min_soc
        self.max_soc = max_soc

        self.policy_net = PPO_PolicyNetwork(obs_dim,action_dim,action_space,device,hidden_dim,action_range).to(device)
        self.v_net = PPO_ValueNetwork(obs_dim,hidden_dim).to(device)

        policy_lr = 1e-4
        q_lr = 1e-4

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(),lr=policy_lr)
        self.v_optimizer = optim.Adam(self.v_net.parameters(),lr=q_lr)
        # self.q_criterion = nn.MSELoss()

        self.memory = []

        self.gamma = 0.99
        self.Lambda = 0.95
        self.eps_clip = 0.2
        # self.batch_size = 256
        self.K_epoch = 10
    
    def push(self,transition):
        self.memory.append(transition)

    def sample_batch(self):
        obs_lst,action_lst,reward_lst,next_obs_lst,log_prob_lst,done_lst= [],[],[],[],[],[]
        for transition in self.memory:
            obs,action,reward,next_obs,prob,done = transition

            obs_lst.append(obs)
            action_lst.append(action)
            reward_lst.append(reward)
            next_obs_lst.append(next_obs)
            log_prob_lst.append(np.array([prob]))
            done_lst.append(np.array([done]))
        
        obs_t = torch.tensor(np.array(obs_lst),dtype=torch.float32,device=self.device)
        action_t = torch.tensor(np.array(action_lst),dtype=torch.float32,device=self.device)
        reward_t = torch.tensor(reward_lst,dtype=torch.float32,device=self.device)
        next_obs_t = torch.tensor(np.array(next_obs_lst),dtype=torch.float32,device=self.device)
        log_prob_lst = torch.tensor(np.array(log_prob_lst),dtype=torch.float32,device=self.device)
        done_t = torch.tensor(np.array(done_lst),dtype=torch.float32,device=self.device)

        self.data = []

        return obs_t,action_t,reward_t,next_obs_t,log_prob_lst,done_t

    def update(self):
        obs,action,reward,next_obs,log_prob,done = self.sample_batch()
        flipped_done = torch.flip(done,dims=(0,))

        for _ in range(self.K_epoch):
            td_target = reward + self.gamma * self.v_net(next_obs) * done
            delta = td_target - self.v_net(obs)
            delta = delta.detach().cpu().numpy()

            advantages_lst = []
            advantage = 0.0
            for delta_t,mask in zip(delta[::-1],flipped_done):
                advantage = self.gamma * self.Lambda * advantage * mask + delta_t[0]
                advantages_lst.append([advantage])
            advantages_lst.reverse()
            advantage = torch.tensor(advantages_lst,dtype=torch.float32,device=self.device)
            if not torch.isnan(advantage.std()):
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

            new_log_prob = self.policy_net.evaluate(obs,action)
            ratio = torch.exp(new_log_prob-log_prob)

            surr1 = ratio * advantage
            surr2 = advantage * torch.clamp(ratio,1-self.eps_clip,1+self.eps_clip)

            policy_loss = -torch.min(surr1,surr2).mean()

            ## regularizer 1: the constraint of power capacity 
            sum_action = torch.sum(action[:,1:],dim=1)
            upbound_action = torch.ones_like(sum_action)
            reg_loss_1 = torch.sum(torch.max(torch.zeros_like(sum_action),sum_action-upbound_action))
            
            # # regularizer 2: the constraint of energy capacity 
            # soc = obs[:,0]
            # dch_ch = torch.sign(action[:,0])
            # energy_change = torch.mul(action[:,1:],self.duration_t) # energy change in MWh in each market
            # sum_energy_change = torch.mul(torch.sum(energy_change,dim=1),dch_ch) # total energy change
            # soc_change = sum_energy_change / self.energy_capacity # soc change
            # new_soc = soc_change + soc # soc after action 
            # min_soc = torch.ones_like(new_soc) * self.min_soc # lobound soc 
            # max_soc = torch.ones_like(new_soc) * self.max_soc # upbound soc 
            # new_soc_minus_upbound = new_soc - max_soc # the excel amount of soc 

            # '''
            # For instance: new_soc = [-0.1,0.9,1.1,0.8] new_soc_minus_upbound = [-1.1,-0.1,0.1,-0.2]
            # tmp_t = [abs(-0.1),0.0,0.0,0.0], final_t = [abs(-0.1),0.0,0.1,0.0]
            # '''
            # tmp_t = torch.where(new_soc<min_soc,torch.abs(new_soc),torch.zeros_like(new_soc))
            # final_t = torch.where(new_soc_minus_upbound>torch.zeros_like(new_soc),new_soc_minus_upbound,tmp_t)

            # reg_loss_2 = torch.sum(final_t)
            policy_loss += self.alpha_1 * reg_loss_1
            # policy_loss += self.alpha_2 * reg_loss_2

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            v_loss = (self.v_net(obs)-td_target.detach()).pow(2).mean()
            self.v_optimizer.zero_grad()
            v_loss.backward()
            self.v_optimizer.step()
    
    def save_model(self, path):
        torch.save(self.v_net.state_dict(), path+'_v')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.v_net.load_state_dict(torch.load(path+'_v'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))
        self.v_net.eval()
        self.policy_net.eval()



