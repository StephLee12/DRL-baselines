import numpy as np 

import torch
import torch.nn as nn 
import torch.optim as optim

from itertools import chain 

from rlalgo_net import DDPG_PolicyNetwork,DDPG_QNetwork,BaseAttentiveConvExtraCritic,AttentiveConvDDPG

class DDPG_Agent():
    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        replay_buffer,
        is_advanced,
        embedding_size,
        nhead,
        out_channels,
        kernel_size_lst,
        critic_ffdim,
        hidden_size,
        action_range=1.
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.replay_buffer = replay_buffer
        self.is_advanced = is_advanced


        q_lr = 8e-4
        policy_lr = 8e-4

        if self.is_advanced:
            qnet = BaseAttentiveConvExtraCritic(kernel_size_lst=kernel_size_lst,out_channels=out_channels,action_dim=action_dim,critic_ffdim=critic_ffdim).to(device)
            target_qnet = BaseAttentiveConvExtraCritic(kernel_size_lst=kernel_size_lst,out_channels=out_channels,action_dim=action_dim,critic_ffdim=critic_ffdim).to(device)
            self.policy_net = AttentiveConvDDPG(
                feature_size=obs_dim,
                embedding_size=embedding_size,
                nhead=nhead,
                kernel_size_lst=kernel_size_lst,
                out_channels=out_channels,
                action_dim=action_dim,
                qnet=qnet,
                target_qnet=target_qnet,
                action_range=action_range
            ).to(device)
            self.target_policy_net = AttentiveConvDDPG(
                feature_size=obs_dim,
                embedding_size=embedding_size,
                nhead=nhead,
                kernel_size_lst=kernel_size_lst,
                out_channels=out_channels,
                action_dim=action_dim,
                qnet=qnet,
                target_qnet=target_qnet,
                action_range=action_range
            ).to(device)
        
            for target_param,param in zip(self.policy_net.target_qnet.parameters(),self.policy_net.qnet.parameters()):
                target_param.data.copy_(param.data)
            for target_param,param in zip(self.target_policy_net.parameters(),self.policy_net.parameters()):
                target_param.data.copy_(param.data)

            chained_params = chain(
                self.policy_net.embedding_layer.parameters(),
                self.policy_net.trans_encoder.parameters(),
                self.policy_net.multi_grain_conv.parameters(),
                self.policy_net.actor_fc_mean.parameters()
            )
            self.q_optimizer = optim.Adam(self.policy_net.qnet.parameters(), lr=q_lr)
            self.policy_optimizer = optim.Adam(chained_params,lr=policy_lr)
        else:
            self.qnet = DDPG_QNetwork(obs_dim=obs_dim,action_dim=action_dim,hidden_size=hidden_size).to(device)
            self.target_qnet = DDPG_QNetwork(obs_dim=obs_dim,action_dim=action_dim,hidden_size=hidden_size).to(device)
            self.policy_net = DDPG_PolicyNetwork(obs_dim=obs_dim,action_dim=action_dim,hidden_size=hidden_size,action_range=action_range).to(device)
            self.target_policy_net = DDPG_PolicyNetwork(obs_dim=obs_dim,action_dim=action_dim,hidden_size=hidden_size,action_range=action_range).to(device)

            for target_param,param in zip(self.target_qnet.parameters(),self.qnet.parameters()):
                target_param.data.copy_(param.data)
            for target_param,param in zip(self.target_policy_net.parameters(),self.policy_net.parameters()):
                target_param.data.copy_(param.data)

            self.q_optimizer = optim.Adam(self.qnet.parameters(),lr=q_lr)
            self.policy_optimizer = optim.Adam(self.policy_net.parameters(),lr=policy_lr)

            
        self.qloss_func = nn.MSELoss()
        self.update_cnt = 0

    
    def update(self,batch_size,noise_scale,reward_scale=10.0, gamma=0.99, soft_tau=1e-2, target_update_delay=3):
        self.update_cnt += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        if self.is_advanced:
            qval = self.policy_net.forward_critic(obs=state,action=action,qnet_type='qnet')
            new_next_action = self.target_policy_net.evaluate(obs=next_state,noise_scale=noise_scale)
            new_action = self.policy_net.evaluate(obs=state,noise_scale=noise_scale)
            target_next_qval = self.policy_net.forward_critic(obs=next_state,action=new_next_action,qnet_type='target_qnet')
            target_qval = reward + (1-done)*gamma*target_next_qval

            qloss = self.qloss_func(qval,target_qval.detach())
            self.q_optimizer.zero_grad()
            qloss.backward()
            self.q_optimizer.step()

            new_qval = self.policy_net.forward_critic(obs=state,action=new_action,qnet_type='qnet')
            policy_loss = -new_qval.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            if self.update_cnt % target_update_delay ==0:
                for target_param, param in zip(self.policy_net.target_qnet.parameters(), self.policy_net.qnet.parameters()):
                    target_param.data.copy_(  # copy data value into target parameters
                        target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                    )
                for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(  # copy data value into target parameters
                        target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                    )

        else:        
        
            qval = self.qnet(state, action) # for q 
            new_next_action = self.target_policy_net.evaluate(obs=next_state,noise_scale=noise_scale)  # for q
            new_action = self.policy_net.evaluate(obs=state,noise_scale=noise_scale) # for policy
            target_next_qval = self.target_qnet(next_state, new_next_action) 
            target_qval = reward + (1-done)*gamma*target_next_qval # for q

            # train qnet
            qloss = self.qloss_func(qval, target_qval.detach())
            self.q_optimizer.zero_grad()
            qloss.backward()
            self.q_optimizer.step()

            # train policy_net
            new_qval = self.qnet(state,new_action)
            policy_loss = -new_qval.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # update the target_qnet
            if self.update_cnt % target_update_delay ==0:
                for target_param, param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
                    target_param.data.copy_(  # copy data value into target parameters
                        target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                    )
                for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(  # copy data value into target parameters
                        target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                    )


    def save_model(self, path):
        if self.is_advanced:
            torch.save(self.policy_net.state_dict(),path+'_advpolicy')
        else:
            torch.save(self.qnet.state_dict(), path+'_q')
            torch.save(self.target_qnet.state_dict(), path+'_target_q')
            torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        if self.is_advanced:
            self.policy_net.load_state_dict(torch.load(path+'_advpolicy'))
            self.policy_net.eval()
        else:
            self.qnet.load_state_dict(torch.load(path+'_q'))
            self.target_qnet.load_state_dict(torch.load(path+'_target_q'))
            self.policy_net.load_state_dict(torch.load(path+'_policy'))

            self.qnet.eval()
            self.target_qnet.eval()
            self.policy_net.eval()