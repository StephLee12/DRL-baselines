import numpy as np 
import os 
import gym
import logging 

import torch
import torch.nn as nn 
import torch.optim as optim

from rlalgo_utils import ReplayBuffer

from collections import deque 
from itertools import chain 

from rlalgo_net import DDPG_QContinuousMultiActionSingleOutLayer,DDPG_QContinuousMultiActionMultiOutLayer
from rlalgo_net import DDPG_DeterministicContinuousPolicyMultiActionSingleOutLayer,DDPG_DeterministicContinuousPolicyMultiActionMultiOutLayer
from rlalgo_net import DDPG_GaussianContinuousPolicyMultiActionSingleOutLayer,DDPG_GaussianContinuousPolicyMultiActionMultiOutLayer


class DDPG_DeterministicContinuous():
    def __init__(
        self,
        device,
        is_single_or_multi_out,
        obs_dim,
        hidden_dim,
        action_dim,
        q_lr,
        policy_lr
    ) -> None:
        self.device = device
        self.is_single_or_multi_out = is_single_or_multi_out

        if is_single_or_multi_out == 'single_out':
            self.critic = DDPG_QContinuousMultiActionSingleOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
            self.tar_critic = DDPG_QContinuousMultiActionSingleOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
            self.policy = DDPG_DeterministicContinuousPolicyMultiActionSingleOutLayer(device=device,obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
            self.tar_policy = DDPG_DeterministicContinuousPolicyMultiActionSingleOutLayer(device=device,obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
        else: # == 'multi_out'
            self.critic = DDPG_QContinuousMultiActionMultiOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
            self.tar_critic = DDPG_QContinuousMultiActionMultiOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
            self.policy = DDPG_DeterministicContinuousPolicyMultiActionMultiOutLayer(device=device,obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
            self.tar_policy = DDPG_DeterministicContinuousPolicyMultiActionMultiOutLayer(device=device,obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)

        for tar_param,param in zip(self.tar_critic.parameters(),self.critic.parameters()):
            tar_param.data.copy_(param.data)
        for tar_param,param in zip(self.tar_policy.parameters(),self.policy.parameters()):
            tar_param.data.copy_(param.data)
        
        self.critic_optim = optim.Adam(self.critic.parameters(),lr=q_lr)
        self.policy_optim = optim.Adam(self.policy.parameters(),lr=policy_lr)

        self.update_cnt = 0

    def update(self,replay_buffer,batch_size,noise_scale,reward_scale=10.0, gamma=0.99, soft_tau=1e-2, tar_update_delay=3):
        self.update_cnt += 1

        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)

        obs = torch.FloatTensor(obs).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        if self.is_single_or_multi_out == 'single_out':
            # update q
            new_next_action = self.tar_policy.evaluate(obs=next_obs,noise_scale=noise_scale)
            tar_next_q = self.tar_critic(obs=next_obs,action=new_next_action)
            tar_q = reward + (1-done)*gamma*tar_next_q

            q = self.critic(obs=obs,action=action)

            q_loss_func = nn.MSELoss()
            q_loss = q_loss_func(q,tar_q.detach())
            self.critic_optim.zero_grad()
            q_loss.backward()
            self.critic_optim.step()

            # update policy
            new_action = self.policy.evaluate(obs=obs,noise_scale=noise_scale)
            new_q = self.critic(obs=obs,action=new_action)
            policy_loss = -new_q.mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

        else: # == 'multi_out'
            new_next_action = self.tar_policy.evaluate(obs=next_obs,noise_scale=noise_scale) 
            tar_next_q_lst = self.tar_critic.evalaute(obs=next_obs,action=new_next_action)
            tar_q_lst = [reward + (1-done)*gamma*tar_next_q for tar_next_q in tar_next_q_lst]

            q_lst = self.critic(obs=obs,action=action)

            q_loss_func = nn.MSELoss()
            q_loss = 0
            for q,tar_q in zip(q_lst,tar_q_lst):
                q_loss += q_loss_func(q,tar_q.detach())
            self.critic_optim.zero_grad()
            q_loss.backward()
            self.critic_optim.step()

            new_action = self.policy.evaluate(obs=obs,noise_scale=noise_scale)
            new_q_lst = self.critic(obs=obs,action=new_action)
            policy_loss = 0
            for new_q in new_q_lst:
                policy_loss += -new_q.mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()
 

        # update the target nets 
        if self.update_cnt % tar_update_delay == 0:
            for tar_param,param in zip(self.tar_critic.parameters(), self.critic.parameters()):
                tar_param.data.copy_(tar_param.data * (1.0 - soft_tau) + param.data * soft_tau)
            for tar_param, param in zip(self.tar_policy.parameters(), self.policy.parameters()):
                tar_param.data.copy_(tar_param.data * (1.0 - soft_tau) + param.data * soft_tau)


    def save_model(self,path):
        torch.save(self.policy.state_dict(),path+'_policy')
        torch.save(self.critic.state_dict(),path+'_critic')

    def load_model(self,path):
        self.policy.load_state_dict(torch.load(path+'_policy'))
        self.critic.load_state_dict(torch.load(path+'_critic'))

        self.policy.eval()
        self.critic.eval()



class DDPG_GaussianContinuous():
    def __init__(
        self,
        device,
        is_single_or_multi_out,
        obs_dim,
        hidden_dim,
        action_dim,
        log_std_min,
        log_std_max,
        q_lr,
        policy_lr
    ) -> None:
        self.device = device 
        self.is_single_or_multi_out = is_single_or_multi_out

        if is_single_or_multi_out == 'single_out':
            self.critic = DDPG_QContinuousMultiActionSingleOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
            self.tar_critic = DDPG_QContinuousMultiActionSingleOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
            self.policy = DDPG_GaussianContinuousPolicyMultiActionSingleOutLayer(device=device,obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim,log_std_min=log_std_min,log_std_max=log_std_max).to(device)
            self.tar_policy = DDPG_GaussianContinuousPolicyMultiActionSingleOutLayer(device=device,obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim,log_std_min=log_std_min,log_std_max=log_std_max).to(device)
        else: # == 'multi_out'
            self.critic = DDPG_QContinuousMultiActionMultiOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
            self.tar_critic = DDPG_QContinuousMultiActionMultiOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
            self.policy = DDPG_GaussianContinuousPolicyMultiActionMultiOutLayer(device=device,obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim,log_std_min=log_std_min,log_std_max=log_std_max).to(device)
            self.tar_policy = DDPG_GaussianContinuousPolicyMultiActionMultiOutLayer(device=device,obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim,log_std_min=log_std_min,log_std_max=log_std_max).to(device)
        
        for tar_param,param in zip(self.tar_critic.parameters(),self.critic.parameters()):
            tar_param.data.copy_(param.data)
        for tar_param,param in zip(self.tar_policy.parameters(),self.policy.parameters()):
            tar_param.data.copy_(param.data)
        
        self.critic_optim = optim.Adam(self.critic.parameters(),lr=q_lr)
        self.policy_optim = optim.Adam(self.policy.parameters(),lr=policy_lr)

        self.update_cnt = 0


    def update(self,replay_buffer,batch_size,noise_scale,reward_scale=10.0, gamma=0.99, soft_tau=1e-2, tar_update_delay=3):
        self.update_cnt += 1

        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)

        obs = torch.FloatTensor(obs).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        if self.is_single_or_multi_out == 'single_out':
            # update q
            new_next_action = self.tar_policy.evaluate(obs=next_obs,noise_scale=noise_scale)
            tar_next_q = self.tar_critic(obs=next_obs,action=new_next_action)
            tar_q = reward + (1-done)*gamma*tar_next_q

            q = self.critic(obs=obs,action=action)

            q_loss_func = nn.MSELoss()
            q_loss = q_loss_func(q,tar_q.detach())
            self.critic_optim.zero_grad()
            q_loss.backward()
            self.critic_optim.step()

            # update policy
            new_action = self.policy.evaluate(obs=obs,noise_scale=noise_scale)
            new_q = self.critic(obs=obs,action=new_action)
            policy_loss = -new_q.mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

        else: # == 'multi_out'
            new_next_action = self.tar_policy.evaluate(obs=next_obs,noise_scale=noise_scale) 
            tar_next_q_lst = self.tar_critic.evalaute(obs=next_obs,action=new_next_action)
            tar_q_lst = [reward + (1-done)*gamma*tar_next_q for tar_next_q in tar_next_q_lst]

            q_lst = self.critic(obs=obs,action=action)

            q_loss = 0
            q_loss_func = nn.MSELoss()
            for q,tar_q in zip(q_lst,tar_q_lst):
                q_loss += q_loss_func(q,tar_q.detach())
            self.critic_optim.zero_grad()
            q_loss.backward()
            self.critic_optim.step()

            new_action = self.policy.evaluate(obs=obs,noise_scale=noise_scale)
            new_q_lst = self.critic(obs=obs,action=new_action)
            policy_loss = 0
            for new_q in new_q_lst:
                policy_loss += -new_q.mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()
 

        # update the target nets 
        if self.update_cnt % tar_update_delay == 0:
            for tar_param,param in zip(self.tar_critic.parameters(), self.critic.parameters()):
                tar_param.data.copy_(tar_param.data * (1.0 - soft_tau) + param.data * soft_tau)
            for tar_param, param in zip(self.tar_policy.parameters(), self.policy.parameters()):
                tar_param.data.copy_(tar_param.data * (1.0 - soft_tau) + param.data * soft_tau)


    def save_model(self,path):
        torch.save(self.policy.state_dict(),path+'_policy')
        torch.save(self.critic.state_dict(),path+'_critic')

    def load_model(self,path):
        self.policy.load_state_dict(torch.load(path+'_policy'))
        self.critic.load_state_dict(torch.load(path+'_critic'))

        self.policy.eval()
        self.critic.eval()



def train_or_test(train_or_test):
    is_single_multi_out = 'single_out'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 512
    q_lr = 3e-4
    policy_lr = 3e-4 
    log_std_min = -20
    log_std_max = 2
    
    env_name = 'Pendulum'
    env = gym.make(env_name)
    obs_dim = env.num_observations
    action_dim = env.num_actions

    agent = DDPG_GaussianContinuous(
        device=device,
        is_single_or_multi_out=is_single_multi_out,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        q_lr=q_lr,
        policy_lr=policy_lr
    )

    model_save_folder = 'trained_models'
    os.makedirs(model_save_folder,exist_ok=True)
    save_name = 'ddpg_continuous_{}_demo'.format(env_name)
    save_path = os.path.join(model_save_folder,save_name)

    if train_or_test == 'train':
        save_interval = 1000

        log_folder = 'logs'
        os.makedirs(log_folder,exist_ok=True)
        log_name = 'ddpg_continuous_train_{}'.format(env_name)
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
        log_name = 'ddpg_continuous_test_{}'.format(env_name)
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
        res_save_name = 'ddpg_continuous_{}'.format(env_name)
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
