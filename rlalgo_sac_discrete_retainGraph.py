import numpy as np 
import gym 
import os 
import logging 

import torch
import torch.nn as nn 
import torch.optim as optim

from itertools import chain 
from collections import deque 

from rlalgo_net import SAC_QDiscreteSingleAction,SAC_QDiscreteMultiAction
from rlalgo_net import SAC_PolicyDiscreteSingleAction,SAC_PolicyDiscreteMultiAction
from rlalgo_utils import ReplayBuffer


class SAC_Discrete():
    def __init__(
        self,
        device,
        is_single_multi_out,
        obs_dim,
        hidden_dim,
        action_dim,
        q_lr,
        policy_lr,
        alpha_lr
    ) -> None:
        self.device = device 
        self.is_single_multi_out = is_single_multi_out

        if is_single_multi_out == 'single_out':
            self.critic1 = SAC_QDiscreteSingleAction(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim=action_dim).to(device)(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim=action_dim).to(device)
            self.critic2 = SAC_QDiscreteSingleAction(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim=action_dim).to(device)
            self.tar_critic1 = SAC_QDiscreteSingleAction(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim=action_dim).to(device)
            self.tar_critic2 = SAC_QDiscreteSingleAction(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim=action_dim).to(device)
            self.policy = SAC_PolicyDiscreteSingleAction(device=device,obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim=action_dim).to(device)

            self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = optim.Adam([self.log_alpha],lr=alpha_lr)
        
        else: 
            self.critic1 = SAC_QDiscreteMultiAction(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim_lst=action_dim).to(device)
            self.critic2 = SAC_QDiscreteMultiAction(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim_lst=action_dim).to(device)
            self.tar_critic1 = SAC_QDiscreteMultiAction(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim_lst=action_dim).to(device)
            self.tar_critic2 = SAC_QDiscreteMultiAction(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim_lst=action_dim).to(device)
            self.policy = SAC_PolicyDiscreteMultiAction(device=device,obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim_lst=action_dim).to(device)

            self.log_alpha_lst = [torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device) for _ in range(action_dim)]
            self.alpha_lst = [log_alpha.exp() for log_alpha in self.log_alpha_lst]
            self.alpha_optim_lst = [optim.Adam(log_alpha,lr=alpha_lr) for log_alpha in self.log_alpha_lst]

        for tar_param,param in zip(self.tar_critic1.parameters(),self.critic1.parameters()):
                tar_param.data.copy_(param.data)
        for tar_param,param in zip(self.tar_critic2.parameters(),self.critic2.parameters()):
            tar_param.data.copy_(param.data)

        self.critic1_optim = optim.Adam(self.critic1.parameters(),lr=q_lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(),lr=q_lr)
        self.policy_optim = optim.Adam(self.policy.parameters(),lr=policy_lr)


    def update(self, replay_buffer, batch_size, target_entropy, reward_scale=10., gamma=0.99,soft_tau=1e-2):
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)

        obs = torch.FloatTensor(obs).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        if self.is_single_multi_out == 'single_out':
            _,log_probs = self.policy.evaluate(obs=obs)

            # cal alpha loss 
            alpha_loss = -(self.log_alpha * (log_probs + target_entropy).detach()).mean()

            # cal q loss 
            _,next_log_probs = self.policy.evaluate(obs=next_obs)
            tar_next_q1 = self.tar_critic1(obs=next_obs)
            tar_next_q2 = self.tar_critic2(obs=next_obs)
            tar_next_q = (next_log_probs.exp() * (torch.min(tar_next_q1,tar_next_q2) - self.alpha*next_log_probs)).sum(dim=-1).unsqueeze(-1)
            tar_q = reward + (1-done) * gamma * tar_next_q

            q1 = self.critic1(obs=obs).gather(1,action.unsqueeze(-1).long())
            q2 = self.critic2(obs=obs).gather(1,action.unsqueeze(-1).long())

            q_loss_func = nn.MSELoss()
            q1_loss = q_loss_func(q1,tar_q.detach())
            q2_loss = q_loss_func(q2,tar_q.detach())

            # cal policy loss 
            new_q = torch.min(self.critic1(obs=obs),self.critic2(obs=obs))
            # new_q = self.critic1(obs=obs,action=new_action)
            policy_loss = (log_probs.exp()*(self.alpha.detach()*log_probs - new_q.detach())).sum(dim=-1).mean()

            # update critic
            self.critic1_optim.zero_grad()
            q1_loss.backward(retain_graph=True)
            self.critic1_optim.step()
            self.critic2_optim.zero_grad()
            q2_loss.backward(retain_graph=True)
            self.critic2_optim.step()

            # update policy 
            self.policy.zero_grad()
            policy_loss.backward()
            self.policy.step()
            
            # update alpha 
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()            
        
        else:
            _,log_probs_lst = self.policy.evaluate(obs=obs) 

            # update alpha
            self.alpha_lst = []
            for idx,log_alpha in enumerate(self.log_alpha_lst):
                alpha_loss = -(log_alpha * (log_probs[:,idx] + target_entropy[idx]).detach()).mean()
                self.alpha_optim_lst[idx].zero_grad()
                alpha_loss.backward()
                self.alpha_optim_lst[idx].step()
                self.alpha_lst.append(self.log_alpha_lst[idx].exp())
            
            # update q 
            _,next_log_probs_lst = self.policy.evaluate(obs=next_obs)
            tar_next_q1_lst = self.tar_critic1(obs=next_obs)
            tar_next_q2_lst = self.tar_critic2(obs=next_obs)
            tar_next_q_lst = [(next_log_probs.exp()*(torch.min(tar_next_q1,tar_next_q2) - alpha*next_log_probs)).sum(dim=-1).unsqueeze(-1)  for tar_next_q1,tar_next_q2,next_log_probs,alpha in zip(tar_next_q1_lst,tar_next_q2_lst,next_log_probs_lst,self.alpha_lst)]
            tar_q_lst = [reward + (1-done) * gamma * tar_next_q for tar_next_q in tar_next_q_lst]

            q1_lst = self.critic1(obs=obs)
            q2_lst = self.critic2(obs=obs)
            for idx,q1 in enumerate(zip(q1_lst,q2_lst)):
                q1 = q1.gather(1,action[:,idx].unsqueeze(-1).long())
                q2 = q2.gather(1,action[:,idx].unsqueeze(-1).long())
                q1_lst[idx] = q1
                q2_lst[idx] = q2 

            q_loss_func = nn.MSELoss()
            q1_loss,q2_loss = 0,0
            for q1,q2,tar_q in zip(q1_lst,q2_lst,tar_q_lst):
                q1_loss += q_loss_func(q1,tar_q.detach())
                q2_loss += q_loss_func(q2,tar_q.detach())

            self.critic1_optim.zero_grad()
            q1_loss.backward()
            self.critic1_optim.step()
            self.critic2_optim.zero_grad()
            q2_loss.backward()
            self.critic2_optim.step()

            # update policy 
            new_q_lst = [torch.min(new_q1,new_q2) for new_q1,new_q2 in zip(self.critic1(obs=obs),self.critic2(obs=obs))]
            # new_q_lst = self.critic1(obs=obs,action=new_action)
            policy_loss = 0
            for new_q,alpha,log_probs in zip(new_q_lst,self.alpha_lst,log_probs_lst):
                policy_loss += (log_probs.exp()*(alpha.detach()*log_probs - new_q.detach())).sum(dim=-1).mean()

            self.policy.zero_grad()
            policy_loss.backward()
            self.policy.step()

        for tar_param, param in zip(self.tar_critic1.parameters(), self.critic1.parameters()):
            tar_param.data.copy_(tar_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for tar_param, param in zip(self.tar_critic2.parameters(), self.critic2.parameters()):
            tar_param.data.copy_(tar_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    
    def save_model(self, path):
        torch.save(self.critic1.state_dict(), path+'_critic1')
        torch.save(self.critic2.state_dict(), path+'_critic2')
        torch.save(self.policy.state_dict(), path+'_policy')

    def load_model(self, path):
        self.critic1.load_state_dict(torch.load(path+'_critic1'))
        self.critic2.load_state_dict(torch.load(path+'_critic2'))
        self.policy.load_state_dict(torch.load(path+'_policy'))

        self.critic1.eval()
        self.critic2.eval()
        self.policy.eval()


def train_or_test(train_or_test):
    is_single_multi_out = 'single_out'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 512
    q_lr = 3e-4
    policy_lr = 3e-4 
    alpha_lr = 3e-4
    
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = SAC_Discrete(
        device=device,
        is_single_multi_out=is_single_multi_out,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        q_lr=q_lr,
        policy_lr=policy_lr,
        alpha_lr=alpha_lr
    )

    model_save_folder = 'trained_models'
    os.makedirs(model_save_folder,exist_ok=True)
    save_name = 'sac_discrete_{}_demo'.format(env_name)
    save_path = os.path.join(model_save_folder,save_name)

    if train_or_test == 'train':
        save_interval = 1000

        log_folder = 'logs'
        os.makedirs(log_folder,exist_ok=True)
        log_name = 'sac_discrete_train_{}'.format(env_name)
        log_path = os.path.join(log_name,log_name)
        logging.basicConfig(
            filename=log_path,
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.DEBUG
        )
        logger = logging.getLogger(log_name)
        log_interval = 1000

        replay_buffer = ReplayBuffer(int(1e5))
        batch_size = 512
        max_timeframe = int(1e6)
        update_times = 4

        deterministic = False
        reward_window = deque(maxlen=100)
        cum_reward = 10
        obs = env.reset()
        for step in range(1,max_timeframe+1):
            action = agent.policy.get_action(obs=obs,deterministic=deterministic)
            next_obs,reward,done,info = env.step(action)
            replay_buffer.push(obs,action,reward,next_obs,done)
            reward_window.append(reward)
            cum_reward = 0.95*cum_reward + 0.05*reward 
            if done: obs = env.reset()
            else: obs = next_obs 

            if len(replay_buffer) > batch_size:
                for _ in range(update_times):
                    agent.update(replay_buffer=replay_buffer,batch_size=batch_size,target_entropy=-1.0*action_dim)

            if step % save_interval == 0:
                agent.save_model(save_path)
            
            if step % log_interval == 0:
                reward_mean = np.mean(reward_window)
                print('---Current step:{}----Mean Reward:{:.2f}----Cumulative Reward:{:.2f}'.format(step,reward_mean,cum_reward))
                logger.info('---Current step:{}----Mean Reward:{:.2f}----Cumulative Reward:{:.2f}'.format(step,reward_mean,cum_reward))

        agent.save_model(save_path)
    
    else: # == 'test'
        eval_timeframe = 100

        log_folder = 'logs'
        os.makedirs(log_folder,exist_ok=True)
        log_name = 'sac_discrete_test_{}'.format(env_name)
        log_path = os.path.join(log_name,log_name)
        logging.basicConfig(
            filename=log_path,
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.DEBUG
        )
        logger = logging.getLogger(log_name)
        log_interval = 1 
        
        res_save_folder = 'eval_res'
        os.makedirs(res_save_folder,exist_ok=True)
        res_save_name = 'sac_discrete_{}'.format(env_name)
        res_save_path = os.path.join(res_save_folder,res_save_name)

        agent.load_model(save_path)

        deterministic = True
        reward_lst = []
        obs = env.reset()
        for step in range(1,eval_timeframe+1):
            action = agent.policy.get_action(obs=obs,deterministic=deterministic)
            next_obs,reward,done,info = env.step(action)
            reward_lst.append(reward)
            obs = next_obs 
            
            if step % log_interval == 0:
                cum_reward = np.sum(reward_lst)
                print('---Current step:{}----Cumulative Reward:{:.2f}'.format(step,cum_reward))
                logger.info('---Current step:{}----Cumulative Reward:{:.2f}'.format(step,cum_reward))

        np.savetxt(res_save_path,reward_lst)
