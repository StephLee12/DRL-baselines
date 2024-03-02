import numpy as np 
# import gym 
import gymnasium as gym 
import os 
import logging 

import torch
import torch.nn as nn 
import torch.optim as optim

from rlalgo_net import QDiscreteSingleAction
from rlalgo_utils import PER

class PERDQN():
    def __init__(
        self,
        device,
        obs_dim,
        hidden_dim,
        action_dim,
        q_lr
    ) -> None:
        self.device = device 

        self.q = QDiscreteSingleAction(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(self.device)
        self.tar_q = QDiscreteSingleAction(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(self.device)

        for tar_param, param in zip(self.tar_q.parameters(), self.q.parameters()):
            tar_param.data.copy_(param.data)

        self.q_optim = optim.Adam(self.q.parameters(), lr=q_lr)
        
        self.update_cnt = 0

    def update(self, replay_buffer, batch_size, reward_scale=10., gamma=0.99, prior_eps=1e-6, update_interval=32, soft_tau=0.005, update_manner='hard'):
        self.update_cnt += 1
        
        obs, action, reward, next_obs, dw, weights, indices = replay_buffer.sample(batch_size)

        obs = torch.FloatTensor(obs).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        dw = torch.FloatTensor(np.float32(dw)).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        q = self.q.forward(obs)
        q_a = q.gather(1, action.unsqueeze(-1).long())
        max_next_q_a = self.tar_q.forward(next_obs).max(-1)[0].unsqueeze(-1)

        tar_q_a = reward + gamma * (1-dw) * max_next_q_a
        
        q_loss_func = nn.SmoothL1Loss(reduction='none') # calculate element-wise loss 
        q_element_wise_loss = q_loss_func(q_a, tar_q_a.detach())

        # aggregate above loss with weights 
        q_loss = torch.mean(q_element_wise_loss * weights)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        ## PER: update priorities
        loss_for_PER = q_element_wise_loss.detach().cpu().numpy()
        new_priorties = loss_for_PER + prior_eps # based on TD-error 
        replay_buffer.update_priorities(indices=indices, priorities=new_priorties)
        
        if update_manner == 'hard':
            if self.update_cnt % update_interval == 0:
                self._target_hard_update()
            else:
                self._soft_hard_update(soft_tau=soft_tau)  
    
    def save_model(self, path):
        torch.save(self.q.state_dict(), path+'_critic')

    def load_model(self, path):
        self.q.load_state_dict(torch.load(path+'_critic'))
        
        self.q.eval()
        
    def _target_hard_update(self):
        for tar_param, param in zip(self.tar_q.parameters(), self.q.parameters()):
            tar_param.data.copy_(param.data)
            
    def _soft_hard_update(self, soft_tau):
        for tar_param, param in zip(self.tar_q.parameters(), self.q.parameters()):
            tar_param.data.copy_(param.data*soft_tau + tar_param*(1-soft_tau))



def train_or_test(train_or_test):
    is_single_multi_out = 'single_out'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 512
    q_lr = 3e-4 
    
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PERDQN(
        device=device,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        q_lr=q_lr
    )

    model_save_folder = 'trained_models'
    os.makedirs(model_save_folder, exist_ok=True)
    save_name = 'per_dqn_discrete_{}_demo'.format(env_name)
    save_path = os.path.join(model_save_folder, save_name)

    if train_or_test == 'train':
        save_interval = 1000

        log_folder = 'logs'
        os.makedirs(log_folder,exist_ok=True)
        log_name = 'per_dqn_discrete_train_{}'.format(env_name)
        log_path = os.path.join(log_folder, log_name)
        logging.basicConfig(
            filename=log_path,
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.DEBUG
        )
        logger = logging.getLogger(log_name)
        log_interval = 1000

        replay_buffer = PER(capacity=int(1e5))
        batch_size = 512
        max_timeframe = int(1e6)
        update_times = 1

        deterministic = False
        score_lst = []
        score = 0
        obs, _ = env.reset()
        for step in range(1, max_timeframe+1):
            epsilon = max(0.01, 0.08-0.01*(step/10000))
            action = agent.q.get_action(obs=obs, epsilon=epsilon, device=device, action_space=env.action_space, deterministic=deterministic)
            next_obs, reward, dw, tr, info = env.step(action)
            done = (dw or tr)
            replay_buffer.push(obs, action, reward, next_obs, dw)
            score += reward 
            if done: 
                obs, _ = env.reset()
                score_lst.append(score)
                score = 0
            else: 
                obs = next_obs 

            if len(replay_buffer) > batch_size:
                for _ in range(update_times):
                    agent.update(replay_buffer=replay_buffer, batch_size=batch_size, update_manner='hard')

            if step % save_interval == 0:
                agent.save_model(save_path)
            
            if step % log_interval == 0:
                score_mean = np.mean(score_lst)
                print('---Current step:{}----Mean Score:{:.2f}'.format(step, score_mean))
                logger.info('---Current step:{}----Mean Score:{:.2f}'.format(step, score_mean))

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


if __name__ == "__main__":
    train_or_test(train_or_test='train')