import numpy as np 
import os 
import gymnasium as gym 
import logging 

import torch 
import torch.optim as optim 

from rlalgo_net import ContinuousCritic, DDPGMLPHead

from rlalgo_utils import _target_hard_update, _target_update
from rlalgo_utils import _save_model, _load_model, _eval_mode, _train_mode
from rlalgo_utils import _sample_batch
from rlalgo_utils import _network_update

from rlalgo_utils import ReplayBuffer

from rlalgo_ddpg_utils import _get_action, _get_target_q, _get_q, _get_qloss, _get_policy_loss



class DDPG():
    def __init__(
        self,
        device,
        obs_dim,
        mlp_hidden_dim,
        action_dim,
        policy_layer_num,
        critic_layer_num,
        action_range,
        critic_lr,
        policy_lr,
        noise_std
    ) -> None:
        self.device = device 
        self.action_range = action_range
        
        # init critic 
        self.critic_head = ContinuousCritic(input_dim=obs_dim+action_dim, mlp_hidden_dim=mlp_hidden_dim, output_dim=1, layer_num=critic_layer_num).to(device)
        self.tar_critic_head = ContinuousCritic(input_dim=obs_dim+action_dim, mlp_hidden_dim=mlp_hidden_dim, output_dim=1, layer_num=critic_layer_num).to(device)
        # init policy 
        self.policy = DDPGMLPHead(input_dim=obs_dim, mlp_hidden_dim=mlp_hidden_dim, output_dim=action_dim, layer_num=policy_layer_num, action_range=action_range, noise_std=noise_std).to(device)
        self.tar_policy = DDPGMLPHead(input_dim=obs_dim, mlp_hidden_dim=mlp_hidden_dim, output_dim=action_dim, layer_num=policy_layer_num, action_range=action_range, noise_std=noise_std).to(device)
        _target_hard_update(tar_net_lst=[self.tar_critic_head, self.tar_policy], net_lst=[self.critic_head, self.policy])
        
        
        # init optim 
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.critic_optim = optim.Adam(self.critic_head.parameters(), lr=critic_lr)
        
        self.policy_lr = policy_lr 
        self.noise_std = noise_std 
        
        self.update_cnt = 0 
        
        
    def get_action(self, obs, deterministic):
        return _get_action(
            obs=obs,
            device=self.device,
            deterministic=deterministic,
            policy=self.policy,
            action_range=self.action_range,
            noise_std=self.noise_std
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
            next_obs=next_obs,
            noise_std=self.noise_std,
            tar_critic_head=self.tar_critic_head,
            reward=reward,
            cost=None,
            dw=dw,
            gamma=gamma
        )
        
        q = _get_q(
            obs=obs,
            action=action,
            critic_head=self.critic_head
        )
        
        qloss = _get_qloss(
            q=q,
            tar_q=tar_q
        )
        
        _network_update(
            optimizer=self.critic_optim,
            loss=qloss,
            is_clip_gradient=is_clip_gradient,
            clip_parameters=self.critic_head.parameters(),
            clip_gradient_val=clip_gradient_val
        )
        
        # update policy 
        policy_loss = _get_policy_loss(
            task_or_safe='task',
            policy=self.policy,
            obs=obs,
            noise_std=self.noise_std,
            critic_head=self.critic_head,
            safe_critic_head=None,
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
            tar_net_lst=[self.tar_policy, self.tar_critic_head],
            net_lst=[self.policy, self.critic_head],
            soft_tau=soft_tau
        )
        
    
    def save_model(self, path):
        _save_model(
            net_lst=[self.critic_head, self.policy],
            path_lst=[path+'_critic', path+'policy']
        )
    
    
    def load_model(self, path):
        _load_model(
            net_lst=[self.critic_head, self.policy],
            path_lst=[path+'_critic', path+'_policy']
        )
        _target_hard_update(
            tar_net_lst=[self.tar_critic_head, self.tar_policy],
            net_lst=[self.critic_head, self.policy]
        ) 
        
    
    def eval_mode(self):
        _eval_mode(net_lst=[self.critic_head, self.policy])
    
    
    def train_mode(self):
        _train_mode(net_lst=[self.critic_head, self.policy])





def train_or_test(train_or_test):
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 512
    critic_lr = 3e-4
    policy_lr = 3e-5
    policy_layer_num = 2
    critic_layer_num = 2
    action_range = 1.
    log_std_min = -20
    log_std_max = 2
    noise_std = 0.1
    
    env_name = 'Pendulum-v1'
    # env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high[0]

    agent = DDPG(
        device=device,
        obs_dim=obs_dim,
        mlp_hidden_dim=hidden_dim,
        action_dim=action_dim,
        policy_layer_num=policy_layer_num,
        critic_layer_num=critic_layer_num,
        action_range=action_range,
        critic_lr=critic_lr,
        policy_lr=policy_lr,
        noise_std=noise_std
    )

    model_save_folder = 'trained_models'
    os.makedirs(model_save_folder, exist_ok=True)
    save_name = 'ddpg_continuous_{}_demo'.format(env_name)
    save_path = os.path.join(model_save_folder, save_name)

    if train_or_test == 'train':
        save_interval = 1000

        log_folder = 'logs'
        os.makedirs(log_folder, exist_ok=True)
        log_name = 'ddpg_continuous_train_{}'.format(env_name)
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
        noise_std = 0.1
        score_lst = []
        score = 0
        obs, _ = env.reset()
        for step in range(1, max_timeframe+1):
            action = agent.get_action(obs=obs, deterministic=deterministic)
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
                    agent.update(replay_buffer=replay_buffer, batch_size=batch_size, noise_std=noise_std)

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
            action = agent.get_action(obs=obs)
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