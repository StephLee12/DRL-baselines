import numpy as np 
# import gym 
import gymnasium as gym 
import os 
import logging 

import torch
import torch.nn as nn 
import torch.optim as optim

from rlalgo_utils import ReplayBuffer


from rlalgo_net import ContinuousCritic, TD3GaussianMLPHead

from rlalgo_utils import _target_hard_update, _target_update
from rlalgo_utils import _save_model, _load_model, _eval_mode, _train_mode
from rlalgo_utils import _sample_batch
from rlalgo_utils import _network_update

from rlalgo_td3_utils import _get_action, _get_target_q, _get_q, _get_qloss, _get_policy_loss




class GaussianTD3():
    def __init__(
        self,
        device,
        obs_dim,
        mlp_hidden_dim,
        action_dim,
        policy_layer_num,
        critic_layer_num,
        action_range,
        log_std_min,
        log_std_max,
        critic_lr,
        policy_lr,
        policy_delay_update_interval,
        eval_noise_scale,
        eval_noise_clip,
        explore_noise_std,
    ) -> None:
        self.device = device 
        self.action_range = action_range

        # init critic 
        self.critic_head1 = ContinuousCritic(input_dim=obs_dim+action_dim, mlp_hidden_dim=mlp_hidden_dim, output_dim=1, layer_num=critic_layer_num).to(device)
        self.critic_head2 = ContinuousCritic(input_dim=obs_dim+action_dim, mlp_hidden_dim=mlp_hidden_dim, output_dim=1, layer_num=critic_layer_num).to(device)
        self.tar_critic_head1 = ContinuousCritic(input_dim=obs_dim+action_dim, mlp_hidden_dim=mlp_hidden_dim, output_dim=1, layer_num=critic_layer_num).to(device)
        self.tar_critic_head2 = ContinuousCritic(input_dim=obs_dim+action_dim, mlp_hidden_dim=mlp_hidden_dim, output_dim=1, layer_num=critic_layer_num).to(device)
        # init policy 
        self.policy = TD3GaussianMLPHead(input_dim=obs_dim, mlp_hidden_dim=mlp_hidden_dim, output_dim=action_dim, layer_num=policy_layer_num, action_range=action_range, eval_noise_scale=eval_noise_scale, eval_noise_clip=eval_noise_clip, log_std_min=log_std_min, log_std_max=log_std_max).to(device)
        self.tar_policy = TD3GaussianMLPHead(input_dim=obs_dim, mlp_hidden_dim=mlp_hidden_dim, output_dim=action_dim, layer_num=policy_layer_num, action_range=action_range, eval_noise_scale=eval_noise_scale, eval_noise_clip=eval_noise_clip, log_std_min=log_std_min, log_std_max=log_std_max).to(device)
        _target_hard_update(tar_net_lst=[self.tar_critic_head1, self.tar_critic_head2, self.tar_policy], net_lst=[self.critic_head1, self.critic_head2, self.policy])

        # init optim 
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.critic1_optim = optim.Adam(self.critic_head1.parameters(), lr=critic_lr)
        self.critic2_optim = optim.Adam(self.critic_head2.parameters(), lr=critic_lr)
        
        self.policy_lr = policy_lr 
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.eval_noise_scale = eval_noise_scale
        self.eval_noise_clip = eval_noise_clip
        self.explore_noise_std = explore_noise_std
        self.policy_delay_update_interval = policy_delay_update_interval
        
        self.update_cnt = 0 

        
        
    def get_action(self, obs, deterministic):
        return _get_action(
            obs=obs,
            explore_noise_std=self.explore_noise_std,
            deterministic=deterministic,
            device=self.device,
            policy=self.policy,
            action_range=self.action_range
        ) 
        
    
    def update(
        self, 
        replay_buffer, 
        batch_size, 
        is_safe,
        safe_type,
        reward_scale=10., 
        gamma=0.99, 
        hard_update_interval=32,
        soft_tau=1e-2,
        is_clip_gradient=True, 
        clip_gradient_val=40
    ):
        self.update_cnt += 1
        
        
        obs, next_obs, action, reward, dw = _sample_batch(
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            reward_scale=reward_scale,
            device=self.device,
            is_safe=is_safe,
            safe_type=safe_type
        ) 
        
        
        # update critic 
        tar_q = _get_target_q(
            task_or_safe='task',
            tar_policy=self.tar_policy,
            tar_critic_head1=self.tar_critic_head1,
            tar_critic_head2=self.tar_critic_head2,
            next_obs=next_obs,
            reward=reward,
            cost=None,
            dw=dw,
            gamma=gamma
        )
        
        q1, q2 = _get_q(
            obs=obs,
            action=action,
            critic_head1=self.critic_head1,
            critic_head2=self.critic_head2
        )
        
        q1loss, q2loss = _get_qloss(
            q1=q1,
            q2=q2,
            tar_q=tar_q
        )
        
        for optimizer, loss, params in zip(
            [self.critic1_optim, self.critic2_optim], 
            [q1loss, q2loss], 
            [self.critic_head1.parameters(), self.critic_head2.parameters()]
        ):
            _network_update(
                optimizer=optimizer,
                loss=loss,
                is_clip_gradient=is_clip_gradient,
                clip_parameters=params,
                clip_gradient_val=clip_gradient_val
            )
        
        
        _target_update(
            update_manner='soft',
            hard_update_interval=hard_update_interval,
            update_cnt=self.update_cnt,
            tar_net_lst=[self.tar_critic_head1, self.tar_critic_head2],
            net_lst=[self.critic_head1, self.critic_head2],
            soft_tau=soft_tau
        )
        
        
        if self.update_cnt % self.policy_delay_update_interval == 0:
            policy_loss = _get_policy_loss(
                task_or_safe='task',
                policy=self.policy,
                obs=obs,
                critic_head1=self.critic_head1,
                critic_head2=self.critic_head2,
                safe_critic_head1=None,
                safe_critic_head2=None,
                lamda=None
            ) 
            _network_update(
                optimizer=self.policy_optim,
                loss=policy_loss,
                is_clip_gradient=is_clip_gradient,
                clip_parameters=self.policy.parameters(),
                clip_gradient_val=clip_gradient_val
            )
            
            _target_update(
                update_manner='soft',
                hard_update_interval=hard_update_interval,
                update_cnt=self.update_cnt,
                tar_net_lst=[self.tar_policy],
                net_lst=[self.policy],
                soft_tau=soft_tau
            )
            
        
    
    
    def save_model(self, path):
        _save_model(
            net_lst=[self.critic_head1, self.critic_head2, self.policy],
            path_lst=[path+'_critic1', path+'_critic2', path+'policy']
        )
    
    
    def load_model(self, path):
        _load_model(
            net_lst=[self.critic_head1, self.critic_head2, self.policy],
            path_lst=[path+'_critic1', path+'_critic2', path+'_policy']
        )
        _target_hard_update(
            tar_net_lst=[self.tar_critic_head1, self.tar_critic_head2, self.tar_policy],
            net_lst=[self.critic_head1, self.critic_head2, self.policy]
        ) 
        
    
    def eval_mode(self):
        _eval_mode(net_lst=[self.critic_head1, self.critic_head2, self.policy])
    
    
    def train_mode(self):
        _train_mode(net_lst=[self.critic_head1, self.critic_head2, self.policy])



def train_or_test(train_or_test):
    is_single_multi_out = 'single_out'

    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 512
    q_lr = 3e-4
    policy_lr = 3e-4 
    log_std_min = -20
    log_std_max = 2
    policy_delay_update_interval = 3
    
    env_name = 'Pendulum-v1'
    # env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high[0]

    agent = TD3_GaussianContinuous(
        device=device,
        is_single_or_multi_out=is_single_multi_out,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        action_range=action_range,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        q_lr=q_lr,
        policy_lr=policy_lr,
        policy_delay_update_interval=policy_delay_update_interval
    )

    # agent = TD3_DeterministicContinuous(
    #     device=device,
    #     is_single_or_multi_out=is_single_multi_out,
    #     obs_dim=obs_dim,
    #     hidden_dim=hidden_dim,
    #     action_dim=action_dim,
    #     action_range=action_range,
    #     q_lr=q_lr,
    #     policy_lr=policy_lr,
    #     policy_delay_update_interval=policy_delay_update_interval
    # )

    model_save_folder = 'trained_models'
    os.makedirs(model_save_folder, exist_ok=True)
    save_name = 'td3_continuous_{}_demo'.format(env_name)
    save_path = os.path.join(model_save_folder, save_name)

    if train_or_test == 'train':
        save_interval = 1000

        log_folder = 'logs'
        os.makedirs(log_folder, exist_ok=True)
        log_name = 'td3_continuous_train_{}'.format(env_name)
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

        replay_buffer = ReplayBuffer(int(1e5))
        batch_size = 512
        max_timeframe = int(1e6)
        update_times = 4

        deterministic = False
        score_lst = []
        score = 0
        explore_noise_std = 0.1
        obs, _ = env.reset()
        for step in range(1, max_timeframe+1):
            action = agent.policy.get_action(obs=obs, deterministic=deterministic, explore_noise_std=explore_noise_std)
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
                    agent.update(replay_buffer=replay_buffer, batch_size=batch_size)

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
        log_name = 'td3_continuous_test_{}'.format(env_name)
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
        res_save_name = 'td3_continuous_{}'.format(env_name)
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


if __name__ == "__main__":
    train_or_test(train_or_test='train')