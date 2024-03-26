import numpy as np 
# import gym 
import gymnasium as gym 
import os 
import logging 

import torch
import torch.nn as nn 
import torch.optim as optim


from rlalgo_net import MLPHead

from rlalgo_utils import _target_hard_update, _target_update, _load_model, _save_model, _train_mode, _eval_mode
from rlalgo_utils import _sample_batch, _network_update

from rlalgo_dqn_utils import _get_action, _get_target, _get_q, _get_qloss


from rlalgo_utils import MultiStepReplayBuffer



class MultiStepDQN():
    def __init__(
        self,
        device,
        obs_dim,
        hidden_dim,
        critic_layer_num,
        action_dim,
        critic_lr
    ) -> None:
        self.device = device 

        self.critic_head = MLPHead(input_dim=obs_dim, mlp_hidden_dim=hidden_dim, output_dim=action_dim, layer_num=critic_layer_num).to(self.device)
        self.tar_critic_head = MLPHead(input_dim=obs_dim, mlp_hidden_dim=hidden_dim, output_dim=action_dim, layer_num=critic_layer_num).to(self.device)

        _target_hard_update(tar_net_lst=[self.tar_critic_head], net_lst=[self.critic_head])

        self.critic_optim = optim.Adam(self.critic_head.parameters(), lr=critic_lr)
        
        self.update_cnt = 0
        
    
    def get_action(self, obs, batch_input=False):
        return _get_action(
            critic_head=self.critic_head,
            obs=obs,
            batch_input=batch_input,
            device=self.device
        )
        
    

    def update(
        self, 
        replay_buffer, 
        batch_size, 
        reward_scale=10., 
        gamma=0.99, 
        hard_update_interval=32, 
        soft_tau=0.005, 
        update_manner='soft', 
        is_clip_gradient=True, 
        clip_gradient_val=40
    ):
        self.update_cnt += 1
        
        obs, next_obs, action, reward, dw = _sample_batch(
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            reward_scale=reward_scale,
            device=self.device
        )
        
        # target q 
        tar_q_a = _get_target(
            tar_critic_head=self.tar_critic_head,
            next_obs=next_obs,
            reward=reward,
            dw=dw,
            gamma=gamma
        )
        # q 
        q_a = _get_q(
            critic_head=self.critic_head,
            obs=obs,
            action=action
        )
        
        qloss = _get_qloss(
            q_a=q_a,
            tar_q_a=tar_q_a
        )
        
        
        _network_update(
            optimizer=self.critic_optim,
            loss=qloss,
            is_clip_gradient=is_clip_gradient,
            clip_parameters=self.critic_head.parameters(),
            clip_gradient_val=clip_gradient_val
        )
            

        _target_update(
            update_manner=update_manner,
            hard_update_interval=hard_update_interval,
            update_cnt=self.update_cnt,
            tar_net_lst=[self.tar_critic_head], 
            net_lst=[self.critic_head], 
            soft_tau=soft_tau
        )


    
    def save_model(self, path):
        torch.save(self.critic_head.state_dict(), path+'_critic')

    def load_model(self, path):
        self.critic_head.load_state_dict(torch.load(path+'_critic'))
        
        self.critic_head.eval()
       
       
    def save_model(self, path):
        _save_model(net_lst=[self.critic_head], path_lst=[path+'_critic'])
        
    
    
    def load_model(self, path):
        _load_model(net_lst=[self.critic_head], path_lst=[path+'_critic'])
        _target_hard_update(tar_net_lst=[self.tar_critic_head], net_lst=[self.critic_head])
               
               
    
    def eval_mode(self):
        _eval_mode(net_lst=[self.critic_head, self.tar_critic_head])
        
        


    def train_mode(self):
        _train_mode(net_lst=[self.critic_head, self.tar_critic_head])



def train_or_test(train_or_test):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 512
    critic_lr = 3e-4 
    critic_layer_num = 2
    
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = MultiStepDQN(
        device=device,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        critic_layer_num=critic_layer_num,
        action_dim=action_dim,
        critic_lr=critic_lr
    )

    model_save_folder = 'trained_models'
    os.makedirs(model_save_folder, exist_ok=True)
    save_name = 'multi_step_dqn_discrete_{}_demo'.format(env_name)
    save_path = os.path.join(model_save_folder, save_name)

    if train_or_test == 'train':
        save_interval = 1000

        log_folder = 'logs'
        os.makedirs(log_folder, exist_ok=True)
        log_name = 'multi_step_dqn_discrete_train_{}'.format(env_name)
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

        replay_buffer = MultiStepReplayBuffer(int(1e5)) ## MultiStep DQN
        batch_size = 512
        max_timeframe = int(1e6)
        update_times = 1

        deterministic = False
        score_lst = []
        score = 0
        obs, _ = env.reset()
        for step in range(1, max_timeframe+1):
            epsilon = max(0.01, 0.08-0.01*(step/10000))
            action = agent.get_action(obs=obs)
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
                    agent.update(replay_buffer=replay_buffer, batch_size=batch_size, update_manner='soft')

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
            action = agent.critic_head.get_action(obs=obs,epsilon=epsilon,deterministic=deterministic)
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