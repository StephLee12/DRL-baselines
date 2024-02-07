import numpy as np 
import gym 
import os 
import logging 

import torch
import torch.nn as nn 
import torch.optim as optim

from collections import deque 

from rlalgo_net import C51QDiscreteSingleAction
from rlalgo_utils import ReplayBuffer

class C51:
    def __init__(
        self,
        device,
        obs_dim,
        hidden_dim,
        action_dim,
        q_lr,
        v_min=0.0,
        v_max=200.0,
        atom_size=51
    ) -> None:
        self.device = device 

        self.atom_size = atom_size
        
        self.support = torch.linspace(start=v_min, end=v_max, steps=atom_size).to(device)
        self.q = C51QDiscreteSingleAction(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim, atom_size=atom_size, support=self.support).to(device)
        self.tar_q = C51QDiscreteSingleAction(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim, atom_size=atom_size, support=self.support).to(device)

        for tar_param,param in zip(self.tar_q.parameters(),self.q.parameters()):
            tar_param.data.copy_(param.data)

        self.q_optim = optim.Adam(self.q.parameters(),lr=q_lr)

        self.v_min, self.v_max = v_min, v_max
        self.delta_z = float(v_max-v_min) / (atom_size-1) # C51, distance between two neighboring atoms

    def update(self, replay_buffer, batch_size, reward_scale=10., gamma=0.99):
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)

        obs = torch.FloatTensor(obs).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device) 

        with torch.no_grad():
            next_action = self.tar_q.forward(obs=next_obs).argmax(dim=-1) # still use Q(s,a) to get action
            next_dist = self.tar_q.get_dist(obs=obs)
            next_dist = next_dist.index_select(dim=1, index=next_action) # get dist corresponding to a*

            t_z = reward + (1-done) * gamma * self.support 
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(0, (batch_size-1)*self.atom_size, batch_size).long().unsqueeze(-1) # (batch_size, 1)
            offset = offset.expand(batch_size, self.atom_size).to(self.device) # (batch_size, atom_size)

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                dim=0,
                index=(l+offset).view(-1),
                source=(next_dist*(u.float()-b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                dim=0,
                index=(u+offset).view(-1),
                source=(next_dist*(b-l.float())).view(-1)
            )
        
        dist = self.q.get_dist(obs=obs)
        log_p = torch.log(dist.index_select(dim=1, index=action))

        loss = -(proj_dist * log_p).sum(-1).mean()

        self.q_optim.zero_grad()
        loss.backward()
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

    agent = C51(
        device=device,
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
