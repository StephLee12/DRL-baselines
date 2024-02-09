import numpy as np 
import gym 
import os 
import logging 

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

from itertools import chain 
from collections import deque

from rlalgo_net import SAC_QContinuousMultiActionSingleOutLayer
from rlalgo_net import SAC_GaussianContinuousPolicyMultiActionSingleOutLayer

from rlalgo_utils import LagReplayBuffer

class SACLag_GaussianContinuous():
    def __init__(
        self,
        device,
        obs_dim,
        hidden_dim,
        action_dim,
        log_std_min,
        log_std_max,
        q_lr=5e-4,
        policy_lr=5e-4,
        alpha_lr=3e-6,
        lamda_lr=3e-2,
        init_lamda=10.0,
        lamda_update_interval=12,
    ) -> None:
        self.device = device 

        # task q 
        self.task_critic1 = SAC_QContinuousMultiActionSingleOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
        self.task_critic2 = SAC_QContinuousMultiActionSingleOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
        self.tar_task_critic1 = SAC_QContinuousMultiActionSingleOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
        self.tar_task_critic2 = SAC_QContinuousMultiActionSingleOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
        
        # safe q 
        self.safe_critic1 = SAC_QContinuousMultiActionSingleOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
        self.safe_critic2 = SAC_QContinuousMultiActionSingleOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
        self.tar_safe_critic1 = SAC_QContinuousMultiActionSingleOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
        self.tar_safe_critic2 = SAC_QContinuousMultiActionSingleOutLayer(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)

        self.policy = SAC_GaussianContinuousPolicyMultiActionSingleOutLayer(device=device,obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim,log_std_min=log_std_min,log_std_max=log_std_max).to(device)
        
        # temperature 
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha],lr=alpha_lr)

        # lag lambda 
        self.lamda = torch.tensor(init_lamda, requires_grad=True, device=device)
        self.lamda_optim = optim.Adam([self.lamda], lr=lamda_lr)
        self.lamda_update_interval = lamda_update_interval
        self.interval_cnt = 0
        self.cost_lim = -5e-4


        for tar_param,param in zip(self.tar_task_critic1.parameters(),self.task_critic1.parameters()):
            tar_param.data.copy_(param.data)
        for tar_param,param in zip(self.tar_task_critic2.parameters(),self.task_critic2.parameters()):
            tar_param.data.copy_(param.data)
        for tar_param,param in zip(self.tar_safe_critic1.parameters(),self.safe_critic1.parameters()):
            tar_param.data.copy_(param.data)
        for tar_param,param in zip(self.tar_safe_critic2.parameters(),self.safe_critic2.parameters()):
            tar_param.data.copy_(param.data)

        self.task_critic1_optim = optim.Adam(self.task_critic1.parameters(),lr=q_lr)
        self.task_critic2_optim = optim.Adam(self.task_critic2.parameters(),lr=q_lr)
        
        self.safe_critic1_optim = optim.Adam(self.safe_critic1.parameters(),lr=q_lr)
        self.safe_critic2_optim = optim.Adam(self.safe_critic2.parameters(),lr=q_lr)

        self.policy_optim = optim.Adam(self.policy.parameters(),lr=policy_lr)


    def update(self, replay_buffer, batch_size, target_entropy, reward_scale=10., gamma=0.99,soft_tau=1e-2):
        obs, action, reward, cost, next_obs, done = replay_buffer.sample(batch_size)

        obs = torch.FloatTensor(obs).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        cost = torch.FloatTensor(cost).unsqueeze(1).to(self.device) # cost w.r.t safe critic; need normalization or not?
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        new_action,log_probs = self.policy.evaluate(obs=obs)

        # cal alpha loss
        # dual gradient: max_{alpha} min_{a*} alpha*log_probs + lamda*safe_q - task_q* -> min{alpha} -alpha*log_probs 
        alpha_loss = -(self.log_alpha * (log_probs + target_entropy).detach()).mean()

        # cal task q loss 
        next_action,next_log_probs = self.policy.evaluate(obs=next_obs)
        tar_next_task_q1 = self.tar_task_critic1.forward(obs=next_obs,action=next_action)
        tar_next_task_q2 = self.tar_task_critic2.forward(obs=next_obs,action=next_action)
        tar_next_task_q = torch.min(tar_next_task_q1,tar_next_task_q2) - self.alpha * next_log_probs
        tar_task_q = reward + (1-done) * gamma * tar_next_task_q 

        task_q1 = self.task_critic1.forward(obs=obs,action=action)
        task_q2 = self.task_critic2.forward(obs=obs,action=action)

        q_loss_func = nn.MSELoss()
        task_q1_loss = q_loss_func(task_q1,tar_task_q.detach())
        task_q2_loss = q_loss_func(task_q2,tar_task_q.detach())

        # cal safe q loss 
        tar_next_safe_q1 = self.tar_safe_critic1.forward(obs=obs, action=next_action)
        tar_next_safe_q2 = self.tar_safe_critic2.forward(obs=obs, action=next_action)
        tar_next_safe_q = torch.min(tar_next_safe_q1, tar_next_safe_q2) - self.alpha * next_log_probs
        tar_safe_q = cost + (1-done) * gamma * tar_next_safe_q

        safe_q1 = self.safe_critic1.forward(obs=obs, action=action)
        safe_q2 = self.safe_critic2.forward(obs=obs, action=action)

        safe_q1_loss = q_loss_func(safe_q1, tar_safe_q.detach())
        safe_q2_loss = q_loss_func(safe_q2, tar_safe_q.detach())

        # cal policy loss 
        new_task_q = torch.min(self.task_critic1.forward(obs=obs,action=new_action),self.task_critic2.forward(obs=obs,action=new_action))
        task_loss = self.alpha.detach()*log_probs - new_task_q
        new_safe_q = torch.min(self.safe_critic1.forward(obs=obs, action=new_action), self.safe_critic2.forward(obs=obs, action=new_action))
        safe_loss = self.lamda * new_safe_q
        policy_loss = (task_loss + safe_loss).mean() # max task_q - alpha*log_probs - safe_q -> min alpha*log_probs + lamda*safe_q - task_q 

        # cal lag multiplier loss 
        self.interval_cnt += 1
        if self.interval_cnt % self.lamda_update_interval == 0:
            safe_q1 = self.safe_critic1.forward(obs=obs, action=action)
            safe_q2 = self.safe_critic2.forward(obs=obs, action=action)

            violation = torch.min(safe_q1, safe_q2) - self.cost_lim

            self.log_lamda = F.softplus(self.lamda)
            # dual gradient max_{lambda} min_{a*} alpha*log_probs* + lamda*safe_q - task_q* -> min_{lambda} -lambda * safe_q 
            lamda_loss = -(self.log_lamda * violation.detach()).sum(dim=-1) 

        # update task critic 
        self.task_critic1_optim.zero_grad()
        task_q1_loss.backward(retain_graph=True)
        self.task_critic1_optim.step()
        self.task_critic2_optim.zero_grad()
        task_q2_loss.backward(retain_graph=True)
        self.task_critic2_optim.step()

        # update safe critic 
        self.safe_critic1_optim.zero_grad()
        safe_q1_loss.backward(retain_graph=True)
        self.safe_critic1_optim.step()
        self.safe_critic2_optim.zero_grad()
        safe_q2_loss.backward(retain_graph=True)
        self.safe_critic2_optim.step()

        # update policy 
        self.policy.zero_grad()
        policy_loss.backward()
        self.policy.step()

        # update alpha 
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # update lag multiplier 
        if self.interval_cnt % self.lamda_update_interval == 0:
            self.lamda_optim.zero_grad()
            lamda_loss.backward()
            self.lamda_optim.step()
        
        for tar_param, param in zip(self.tar_task_critic1.parameters(), self.task_critic1.parameters()):
            tar_param.data.copy_(tar_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for tar_param, param in zip(self.tar_task_critic2.parameters(), self.task_critic2.parameters()):
            tar_param.data.copy_(tar_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for tar_param, param in zip(self.tar_safe_critic1.parameters(), self.safe_critic1.parameters()):
            tar_param.data.copy_(tar_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for tar_param, param in zip(self.tar_safe_critic2.parameters(), self.safe_critic2.parameters()):
            tar_param.data.copy_(tar_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    
    def save_model(self, path):
        torch.save(self.task_critic1.state_dict(), path+'_task_critic1')
        torch.save(self.task_critic2.state_dict(), path+'_task_critic2')
        torch.save(self.safe_critic1.state_dict(), path+'_safe_critic1')
        torch.save(self.safe_critic2.state_dict(), path+'_safe_critic2')
        torch.save(self.policy.state_dict(), path+'_policy')

    def load_model(self, path):
        self.task_critic1.load_state_dict(torch.load(path+'_task_critic1'))
        self.task_critic2.load_state_dict(torch.load(path+'_task_critic2'))
        self.safe_critic1.load_state_dict(torch.load(path+'_safe_critic1'))
        self.safe_critic2.load_state_dict(torch.load(path+'_safe_critic2'))
        self.policy.load_state_dict(torch.load(path+'_policy'))

        self.task_critic1.eval()
        self.task_critic2.eval()
        self.safe_critic1.eval()
        self.safe_critic2.eval()
        self.policy.eval()


def train_or_test(train_or_test):

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 512
    q_lr = 3e-4
    policy_lr = 3e-4 
    alpha_lr = 3e-4
    log_std_min = -20
    log_std_max = 2
    
    env_name = 'Pendulum'
    env = gym.make()
    obs_dim = env.num_observations
    action_dim = env.num_actions

    agent = SACLag_GaussianContinuous(
        device=device,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        q_lr=q_lr,
        policy_lr=policy_lr,
        alpha_lr=alpha_lr
    )

    model_save_folder = 'trained_models'
    os.makedirs(model_save_folder,exist_ok=True)
    save_name = 'sac_continuous_{}_demo'.format(env_name)
    save_path = os.path.join(model_save_folder,save_name)
    

    if train_or_test == 'train':
        save_interval = 1000

        log_folder = 'logs'
        os.makedirs(log_folder,exist_ok=True)
        log_name = 'sac_lag_continuous_train_{}'.format(env_name)
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

        replay_buffer = LagReplayBuffer(int(1e5))
        batch_size = 512
        max_timeframe = int(1e6)
        update_times = 4
        target_entropy = -1.0 * action_dim

        deterministic = False
        reward_window = deque(maxlen=100)
        cum_reward = 10
        obs = env.reset()
        for step in range(1,max_timeframe+1):
            action = agent.policy.get_action(obs=obs,deterministic=deterministic)
            next_obs,reward,cost,done,info = env.step(action)
            replay_buffer.push(obs,action,reward,cost,next_obs,done)
            reward_window.append(reward)
            cum_reward = 0.95*cum_reward + 0.05*reward 
            if done: obs = env.reset()
            else: obs = next_obs 

            if len(replay_buffer) > batch_size:
                for _ in range(update_times):
                    agent.update(replay_buffer=replay_buffer,batch_size=batch_size,target_entropy=target_entropy)

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
        log_name = 'sac_continuous_test_{}'.format(env_name)
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
        res_save_name = 'sac_continuous_{}'.format(env_name)
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
