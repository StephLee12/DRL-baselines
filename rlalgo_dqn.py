import numpy as np 
import gym 
import os 
import logging 

import torch
import torch.nn as nn 
import torch.optim as optim

from itertools import chain 
from collections import deque 

from rlalgo_net import QDiscreteSingleAction, QDiscreteMultiAction
from rlalgo_utils import ReplayBuffer

class DQN():
    def __init__(
        self,
        device,
        is_single_multi_out,
        obs_dim,
        hidden_dim,
        action_dim,
        q_lr
    ) -> None:
        self.device = device 
        self.is_single_multi_out = is_single_multi_out

        if is_single_multi_out == 'single_out':
            self.q = QDiscreteSingleAction(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim=action_dim).to(self.device)
            self.tar_q = QDiscreteSingleAction(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim=action_dim).to(self.device)
        else: 
            self.q = QDiscreteMultiAction(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim_lst=action_dim).to(self.device)
            self.tar_q = QDiscreteMultiAction(obs_dim=obs_dim,hidden_dim=hidden_dim,action_dim_lst=action_dim).to(self.device)
        
        for tar_param,param in zip(self.tar_q.parameters(),self.q.parameters()):
            tar_param.data.copy_(param.data)

        self.q_optim = optim.Adam(self.q.parameters(),lr=q_lr)

    def update(self, replay_buffer, batch_size, epsilon, reward_scale=10., gamma=0.99):
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)

        obs = torch.FloatTensor(obs).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        if self.is_single_multi_out == 'single_out':

            q = self.q(obs)
            q_a = q.gather(1,action)
            max_next_q = self.tar_q(next_obs).max(-1)[0].unsqueeze(-1)

            tar_q = reward + gamma * (1-done) * max_next_q
            
            q_loss_func = nn.MSELoss()
            q_loss = q_loss_func(q_a,tar_q.detach())

        else:
            q_lst = self.obs(obs) 
            q_a_lst = [q.gather(1,action[:,idx]) for idx,q in enumerate(q_lst)]
            next_q_lst = self.tar_q(next_obs)
            max_next_q_lst = [next_q.max(-1)[0].unsqueeze(-1) for next_q in next_q_lst]

            tar_q_lst = [reward + gamma*(1-done)*max_next_q for max_next_q in max_next_q_lst]

            q_loss_func = nn.MSELoss()
            q_loss = 0
            for q_a,tar_q in zip(q_a_lst,tar_q_lst):
                q_loss += q_loss_func(q_a,tar_q.detach())


        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

    
    def save_model(self, path):
        torch.save(self.q.state_dict(), path+'_critic')

    def load_model(self, path):
        self.q.load_state_dict(torch.load(path+'_critic'))
        
        self.q.eval()



def train_or_test(train_or_test):
    is_single_multi_out = 'single_out'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 512
    q_lr = 3e-4 
    
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(
        device=device,
        is_single_multi_out=is_single_multi_out,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        q_lr=q_lr
    )

    model_save_folder = 'trained_models'
    os.makedirs(model_save_folder,exist_ok=True)
    save_name = 'dqn_discrete_{}_demo'.format(env_name)
    save_path = os.path.join(model_save_folder,save_name)

    if train_or_test == 'train':
        save_interval = 1000

        log_folder = 'logs'
        os.makedirs(log_folder,exist_ok=True)
        log_name = 'dqn_discrete_train_{}'.format(env_name)
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
        update_times = 1

        deterministic = False
        reward_window = deque(maxlen=100)
        cum_reward = 10
        obs = env.reset()
        for step in range(1,max_timeframe+1):
            epsilon = max(0.01, 0.08-0.01*(step/200))
            action = agent.q.get_action(obs=obs,epsilon=epsilon,deterministic=deterministic)
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
        log_name = 'dqn_discrete_test_{}'.format(env_name)
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
        res_save_name = 'dqn_discrete_{}'.format(env_name)
        res_save_path = os.path.join(res_save_folder,res_save_name)

        agent.load_model(save_path)

        deterministic = True
        reward_lst = []
        obs = env.reset()
        for step in range(1,eval_timeframe+1):
            action = agent.q.get_action(obs=obs,epsilon=epsilon,deterministic=deterministic)
            next_obs,reward,done,info = env.step(action)
            reward_lst.append(reward)
            obs = next_obs 
            
            if step % log_interval == 0:
                cum_reward = np.sum(reward_lst)
                print('---Current step:{}----Cumulative Reward:{:.2f}'.format(step,cum_reward))
                logger.info('---Current step:{}----Cumulative Reward:{:.2f}'.format(step,cum_reward))

        np.savetxt(res_save_path,reward_lst)
