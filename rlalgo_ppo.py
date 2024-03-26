import numpy as np 
import math
import os 
import logging 
# import gym 
import gymnasium as gym 

import torch
import torch.nn as nn 
import torch.optim as optim

from rlalgo_net import MLPHead, PPOMLPHead

from rlalgo_utils import _save_model, _load_model, _eval_mode, _train_mode

from rlalgo_ppo_utils import _get_action, _get_adv, _minibatch_update


class GaussianPPO():
    def __init__(
        self,
        device,
        MC_or_GAE,
        obs_dim,
        mlp_hidden_dim,
        action_dim,
        action_range,
        policy_layer_num,
        critic_layer_num,
        policy_lr,
        critic_lr,
        K_epochs,
        entropy_coeff,
        epsilon_clip,
        GAE_lamda,
        T_horizon,
        log_std_min,
        log_std_max,
        log_prob_min,
        log_prob_max
    ) -> None:
        self.device = device 
        self.MC_or_GAE = MC_or_GAE
        
        self.trajectory_buffer = []
        
        self.val_net = MLPHead(input_dim=obs_dim, mlp_hidden_dim=mlp_hidden_dim, output_dim=1, layer_num=critic_layer_num).to(device)
        self.policy = PPOMLPHead(input_dim=obs_dim, mlp_hidden_dim=mlp_hidden_dim, output_dim=action_dim, layer_num=policy_layer_num, log_std_min=log_std_min, log_std_max=log_std_max).to(device)
        
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.val_net_optim = optim.Adam(self.val_net.parameters(), lr=critic_lr)
        
        self.action_range = action_range 
        
        self.policy_lr = policy_lr
        self.K_epochs = K_epochs
        self.entropy_coeff = entropy_coeff
        self.epsilon_clip = epsilon_clip
        self.GAE_lamda = GAE_lamda
        self.T_horizon = T_horizon
        self.log_std_min = log_std_min 
        self.log_std_max = log_std_max 
        self.log_prob_min = log_prob_min
        self.log_prob_max = log_prob_max
        
        self.trajectory_length = 0
        
    
    
    def get_action(self, obs, deterministic):
        return _get_action(
            obs=obs,
            deterministic=deterministic,
            device=self.device,
            policy=self.policy,
            action_range=self.action_range,
            log_prob_min=self.log_prob_min,
            log_prob_max=self.log_prob_max
        ) 
    
    
    
    
    def update(
        self, 
        batch_size, 
        is_clip_gradient=True,
        clip_gradient_val=4., 
        gamma=0.99, 
    ):
        obs, next_obs, action, reward, old_probs_a, done, dw = self._ensemble_trajectory()
        
        # get advantage functions 
        adv, tar_v = _get_adv(
            MC_or_GAE=self.MC_or_GAE,
            device=self.device,
            obs=obs,
            next_obs=next_obs,
            reward=reward,
            done=done,
            dw=dw,
            lamda=self.GAE_lamda,
            gamma=gamma,
            val_net=self.val_net
        )
        
        adv = (adv - adv.mean(dim=0)) / (adv.std(dim=0) + 1e-6) # advantage function normalization
        
        # minibatch update 
        _minibatch_update(
            batch_size=batch_size,
            K_epoch=self.K_epochs,
            device=self.device,
            obs=obs,
            action=action,
            tar_v=tar_v,
            adv=adv,
            old_probs_a=old_probs_a,
            policy=self.policy,
            val_net=self.val_net,
            epsilon_clip=self.epsilon_clip,
            policy_optim=self.policy_optim,
            val_net_optim=self.val_net_optim,
            is_clip_gradient=is_clip_gradient,
            clip_gradient_val=clip_gradient_val,
            log_prob_min=self.log_prob_min,
            log_prob_max=self.log_prob_max
        )
        
        
        
        
    
    
    def _ensemble_trajectory(self):
        obs_lst, action_lst, reward_lst, next_obs_lst, old_probs_a_lst, done_lst, dw_lst = [], [], [], [], [], [], []
        for (obs, action, reward, next_obs, old_probs_a, done, dw) in self.trajectory_buffer:
            obs_lst.append(obs) 
            action_lst.append(action)
            reward_lst.append(reward)
            next_obs_lst.append(next_obs)
            old_probs_a_lst.append(old_probs_a)
            done_lst.append(done)
            dw_lst.append(dw)

        obs = torch.FloatTensor(obs_lst).to(self.device)
        next_obs = torch.FloatTensor(next_obs_lst).to(self.device)
        action = torch.FloatTensor(action_lst).to(self.device)
        reward = torch.FloatTensor(reward_lst).unsqueeze(-1).to(self.device)
        old_probs_a = torch.FloatTensor(old_probs_a_lst).to(self.device)
        done = torch.FloatTensor(np.float32(done_lst)).unsqueeze(-1).to(self.device)
        dw = torch.FloatTensor(np.float32(dw_lst)).unsqueeze(-1).to(self.device)

        self.trajectory_buffer = [] # clear buffer 

        return obs, next_obs, action, reward, old_probs_a, done, dw
    





    
    def save_model(self, path):
        _save_model(
            net_lst=[self.val_net, self.policy],
            path_lst=[path+'_v', path+'_policy']
        )
        
    
    def load_model(self, path):
        _load_model(
            net_lst=[self.val_net, self.policy],
            path_lst=[path+'_v', path+'_policy']
        )
        
        
    def eval_mode(self):
        _eval_mode(net_lst=[self.val_net, self.policy])
    
    
    def train_mode(self):
        _train_mode(net_lst=[self.val_net, self.policy])



def train_or_test(train_or_test):
    is_single_multi_out = 'single_out'

    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    is_use_MC_or_GAE = 'GAE'
    hidden_dim = 512
    v_lr = 3e-4
    policy_lr = 3e-5
    log_std_min = -20 
    log_std_max = 2
    
    env_name = 'Pendulum-v1'
    # env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high[0]

    agent = PPO_GaussianContinuous(
        device=device,
        is_single_or_multi_out=is_single_multi_out,
        is_use_MC_or_GAE=is_use_MC_or_GAE,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        action_range=action_range,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        policy_lr=policy_lr,
        v_lr=v_lr
    )

    model_save_folder = 'trained_models'
    os.makedirs(model_save_folder, exist_ok=True)
    save_name = 'ppo_continuous_{}_demo'.format(env_name)
    save_path = os.path.join(model_save_folder, save_name)

    if train_or_test == 'train':
        save_interval = 1000

        log_folder = 'logs'
        os.makedirs(log_folder, exist_ok=True)
        log_name = 'ppo_continuous_train_{}'.format(env_name)
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

        batch_size = 512
        max_timeframe = int(1e6)
        T_horizon = 2048
        K_epoch = 20

        deterministic = False
        score_lst = []
        score = 0
        trajectory_length = 0
        obs, _ = env.reset()
        for step in range(1, max_timeframe+1):
            trajectory_length += 1
            
            action, probs = agent.policy.get_action(obs=obs, deterministic=deterministic)
            next_obs, reward, dw, tr, info = env.step(action)
            done = (dw or tr)
            agent.trajectory_buffer.append([list(obs), list(action), reward, list(next_obs), list(probs), done, dw])
            score += reward 
            if done: 
                obs, _ = env.reset()
                score_lst.append(score)
                score = 0
            else: 
                obs = next_obs 

            if trajectory_length % T_horizon == 0:
                agent.update(batch_size=batch_size, K_epoch=K_epoch)
                trajectory_length = 0

            if step % save_interval == 0:
                agent.save_model(save_path)
            
            if step % log_interval == 0:
                score_mean = np.mean(score_lst[-10:])
                print('---Current step:{}----Mean Score:{:.2f}'.format(step, score_mean))
                logger.info('---Current step:{}----Mean Score:{:.2f}'.format(step, score_mean))


        agent.save_model(save_path)
    
    else: # == 'test'
        eval_timeframe = 100

        log_folder = 'logs'
        os.makedirs(log_folder,exist_ok=True)
        log_name = 'ppo_continuous_test_{}'.format(env_name)
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
        res_save_name = 'ppo_continuous_{}'.format(env_name)
        res_save_path = os.path.join(res_save_folder,res_save_name)

        agent.load_model(save_path)

        deterministic = True
        reward_lst = []
        obs = env.reset()
        for step in range(1,eval_timeframe+1):
            action,_ = agent.policy.get_action(obs=obs,deterministic=deterministic)
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


