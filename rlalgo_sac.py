import numpy as np 

import torch
import torch.nn as nn 
import torch.optim as optim

from itertools import chain 

from rlalgo_net import SoftQNetwork,SoftPolicyNetwork,BaseAttentiveConvExtraCritic,AttentiveConvSAC

class SAC_Agent():
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
        action_range=1.0
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.replay_buffer = replay_buffer
        self.is_advanced = is_advanced

        q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr  = 3e-4    

        if is_advanced:
            qnet1 = BaseAttentiveConvExtraCritic(kernel_size_lst=kernel_size_lst,out_channels=out_channels,action_dim=action_dim,critic_ffdim=critic_ffdim).to(device)
            qnet2 = BaseAttentiveConvExtraCritic(kernel_size_lst=kernel_size_lst,out_channels=out_channels,action_dim=action_dim,critic_ffdim=critic_ffdim).to(device)
            target_qnet1 = BaseAttentiveConvExtraCritic(kernel_size_lst=kernel_size_lst,out_channels=out_channels,action_dim=action_dim,critic_ffdim=critic_ffdim).to(device)
            target_qnet2 = BaseAttentiveConvExtraCritic(kernel_size_lst=kernel_size_lst,out_channels=out_channels,action_dim=action_dim,critic_ffdim=critic_ffdim).to(device)
            self.policy_net = AttentiveConvSAC(
                feature_size=obs_dim,
                embedding_size=embedding_size,
                nhead=nhead,
                kernel_size_lst=kernel_size_lst,
                out_channels=out_channels,
                action_dim=action_dim,
                qnet1=qnet1,
                qnet2=qnet2,
                target_qnet1=target_qnet1,
                target_qnet2=target_qnet2,
                action_range=action_range
            ).to(device)
            
            for target_param, param in zip(self.policy_net.target_qnet1.parameters(), self.policy_net.qnet1.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(self.policy_net.target_qnet2.parameters(), self.policy_net.qnet2.parameters()):
                target_param.data.copy_(param.data)

            chained_params = chain(
                self.policy_net.embedding_layer.parameters(),
                self.policy_net.trans_encoder.parameters(),
                self.policy_net.multi_grain_conv.parameters(),
                self.policy_net.actor_fc_mean.parameters(),
                self.policy_net.actor_fc_log_std.parameters()
            )

            self.policy_optimizer = optim.Adam(chained_params, lr=policy_lr)
            self.q1_optimizer = optim.Adam(self.policy_net.qnet1.parameters(), lr=q_lr)
            self.q2_optimizer = optim.Adam(self.policy_net.qnet2.parameters(), lr=q_lr)
        else:
            self.qnet1 = SoftQNetwork(obs_dim=obs_dim,action_dim=action_dim,hidden_size=hidden_size).to(device)
            self.qnet2 = SoftQNetwork(obs_dim=obs_dim,action_dim=action_dim,hidden_size=hidden_size).to(device)
            self.target_qnet1 = SoftQNetwork(obs_dim=obs_dim,action_dim=action_dim,hidden_size=hidden_size).to(device)
            self.target_qnet2 = SoftQNetwork(obs_dim=obs_dim,action_dim=action_dim,hidden_size=hidden_size).to(device)
            self.policy_net = SoftPolicyNetwork(obs_dim=obs_dim,action_dim=action_dim,hidden_size=hidden_size,action_range=action_range).to(device)

            for target_param, param in zip(self.target_qnet1.parameters(), self.qnet1.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(self.target_qnet2.parameters(), self.qnet2.parameters()):
                target_param.data.copy_(param.data)

            self.q1_optimizer = optim.Adam(self.qnet1.parameters(), lr=q_lr)
            self.q2_optimizer = optim.Adam(self.qnet2.parameters(), lr=q_lr)
            self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        

        self.q1_loss_func = nn.MSELoss()
        self.q2_loss_func = nn.MSELoss()

        
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    
    def update(self, batch_size, auto_entropy, target_entropy, reward_scale=10., gamma=0.99,soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        if self.is_advanced:
            qval1 = self.policy_net.forward_critic(obs=state,action=action,qnet_type='qnet1')
            qval2 = self.policy_net.forward_critic(obs=state,action=action,qnet_type='qnet2')
            new_action,log_prob,_,_,_ = self.policy_net.evaluate(obs=state)
            new_next_action,next_log_prob,_,_,_ = self.policy_net.evaluate(obs=next_state)

            if auto_entropy is True:
                alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()
            else:
                self.alpha = 1.
                alpha_loss = 0
            
            target_next_qval1 = self.policy_net.forward_critic(obs=next_state,action=new_next_action,qnet_type='target_qnet1')
            target_next_qval2 = self.policy_net.forward_critic(obs=next_state,action=new_next_action,qnet_type='target_qnet2')
            target_next_qmin = torch.min(target_next_qval1,target_next_qval2)
            target_qval = reward + (1-done)*gamma*target_next_qmin
            qloss1 = self.q1_loss_func(qval1,target_qval.detach())
            qloss2 = self.q2_loss_func(qval2,target_qval.detach())
            self.q1_optimizer.zero_grad()
            qloss1.backward()
            self.q1_optimizer.step()
            self.q2_optimizer.zero_grad()
            qloss2.backward()
            self.q2_optimizer.step()

            new_qval1 = self.policy_net.forward_critic(obs=state,action=new_action,qnet_type='qnet1')
            new_qval2 = self.policy_net.forward_critic(obs=state,action=new_action,qnet_type='qnet2')
            new_qval = torch.min(new_qval1,new_qval2)
            policy_loss = (self.alpha*log_prob - new_qval).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            for target_param, param in zip(self.policy_net.target_qnet1.parameters(), self.policy_net.qnet1.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )
            for target_param, param in zip(self.policy_net.target_qnet2.parameters(), self.policy_net.qnet2.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )


        else:
            # with torch.autograd.set_detect_anomaly(True):
            qval1 = self.qnet1(state, action)
            qval2 = self.qnet2(state, action)
            new_action, log_prob, _, _, _ = self.policy_net.evaluate(obs=state)
            new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(obs=next_state)
            
            # Updating alpha wrt entropy
            # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
            if auto_entropy is True:
                alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
                # print('alpha loss: ',alpha_loss)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()
            else:
                self.alpha = 1.
                alpha_loss = 0

            # Training Q Function
            target_next_qmin = torch.min(self.target_qnet1(next_state, new_next_action),self.target_qnet2(next_state, new_next_action)) - self.alpha * next_log_prob
            target_qval = reward + (1 - done) * gamma * target_next_qmin # if done==1, only reward
            qloss1 = self.q1_loss_func(qval1, target_qval.detach())  # detach: no gradients for the variable
            qloss2 = self.q2_loss_func(qval2, target_qval.detach())

            self.q1_optimizer.zero_grad()
            qloss1.backward()
            self.q1_optimizer.step()
            self.q2_optimizer.zero_grad()
            qloss2.backward()
            self.q2_optimizer.step()  

            # Training Policy Function
            new_qval = torch.min(self.qnet1(state, new_action),self.qnet2(state, new_action))
            policy_loss = (self.alpha * log_prob - new_qval).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Soft update the target value net
            for target_param, param in zip(self.target_qnet1.parameters(), self.qnet1.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )
            for target_param, param in zip(self.target_qnet2.parameters(), self.qnet2.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )

    
    def save_model(self, path):
        if self.is_advanced:
            torch.save(self.policy_net.state_dict(),path+'_advpolicy')
        else:
            torch.save(self.qnet1.state_dict(), path+'_q1')
            torch.save(self.qnet2.state_dict(), path+'_q2')
            torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        if self.is_advanced:
            self.policy_net.load_state_dict(torch.load(path+'_advpolicy'))
            self.policy_net.eval()
        else:
            self.qnet1.load_state_dict(torch.load(path+'_q1'))
            self.qnet2.load_state_dict(torch.load(path+'_q2'))
            self.policy_net.load_state_dict(torch.load(path+'_policy'))

            self.qnet1.eval()
            self.qnet2.eval()
            self.policy_net.eval()