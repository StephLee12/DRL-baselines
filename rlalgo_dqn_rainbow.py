import numpy as np 
# import gym 
import gymnasium as gym 
import os 
import logging 

import torch
import torch.optim as optim


from rlalgo_net import RainbowCritic

from rlalgo_utils import MultiStepPER
from rlalgo_utils import _target_hard_update, _target_update
from rlalgo_utils import _save_model, _load_model, _eval_mode, _train_mode
from rlalgo_utils import _network_update, _sample_PER_batch

from rlalgo_dqn_c51_utils import _get_log_p

from rlalgo_dqn_rainbow_utils import _get_target, _get_loss

from rlalgo_dqn_per_utils import _update_PER

from rlalgo_dqn_utils import _get_action

from rlalgo_dqn_noisynet_utils import _reset_noisy


class RainbowDQN():
    def __init__(
        self,
        device,
        obs_dim,
        mlp_hidden_dim,
        action_dim,
        critic_layer_num,
        critic_lr,
        task_v_min,
        task_v_max,
        atom_size=51
    ) -> None:
        self.device = device 

        self.atom_size = atom_size

        self.support = torch.linspace(start=task_v_min, end=task_v_max, steps=atom_size).to(device)

        self.critic_head = RainbowCritic(obs_dim=obs_dim, hidden_dim=mlp_hidden_dim, action_dim=action_dim, layer_num=critic_layer_num, atom_size=atom_size, support=self.support).to(device)
        self.tar_critic_head = RainbowCritic(obs_dim=obs_dim, hidden_dim=mlp_hidden_dim, action_dim=action_dim, layer_num=critic_layer_num, atom_size=atom_size, support=self.support).to(device)

        _target_hard_update(tar_net_lst=[self.tar_critic_head], net_lst=[self.critic_head])

        self.critic_optim = optim.Adam(self.critic_head.parameters(), lr=critic_lr)

        self.task_v_min, self.task_v_max = task_v_min, task_v_max
        self.delta_z = float(task_v_max-task_v_min) / (atom_size-1) # C51, distance between two neighboring atoms
        
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
        prior_eps=1e-6,
        reward_scale=10., 
        gamma=0.99, 
        hard_update_interval=32, 
        soft_tau=0.005, 
        update_manner='soft', 
        is_clip_gradient=True, 
        clip_gradient_val=40
    ):
        self.update_cnt += 1
        
        obs, action, reward, next_obs, dw, weights, indices = _sample_PER_batch(
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            reward_scale=reward_scale,
            device=self.device
        )
        
        proj_dist = _get_target(
            support=self.support,
            v_min=self.task_v_min,
            v_max=self.task_v_max,
            delta_z=self.delta_z,
            critic_head=self.critic_head,
            tar_critic_head=self.tar_critic_head,
            next_obs=next_obs,
            reward=reward,
            dw=dw,
            gamma=gamma,
            batch_size=batch_size,
            atom_size=self.atom_size,
            device=self.device
        )
        
        log_p = _get_log_p(obs=obs, critic_head=self.critic_head, action=action, atom_size=self.atom_size)
        elementwise_loss, loss = _get_loss(proj_dist=proj_dist, log_p=log_p, weights=weights)
        
        _network_update(
            optimizer=self.critic_optim,
            loss=loss,
            is_clip_gradient=is_clip_gradient,
            clip_parameters=self.critic_head.parameters(),
            clip_gradient_val=clip_gradient_val
        )



        # PER update priorities
        _update_PER(
            q_element_wise_loss=elementwise_loss,
            prior_eps=prior_eps,
            replay_buffer=replay_buffer,
            indices=indices
        )


        # NoisyNet -> reset noise 
        _reset_noisy(
            critic_head=self.critic_head,
            tar_critic_head=self.tar_critic_head
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
        _save_model(net_lst=[self.critic_head], path_lst=[path+'_critic'])
        
    
    
    def load_model(self, path):
        _load_model(net_lst=[self.critic_head], path_lst=[path+'_critic'])
        _target_hard_update(tar_net_lst=[self.tar_critic_head], net_lst=[self.critic_head])
               
               
    
    def eval_mode(self):
        _eval_mode(net_lst=[self.critic_head, self.tar_critic_head])
        
        


    def train_mode(self):
        _train_mode(net_lst=[self.critic_head, self.tar_critic_head])




def train_or_test(train_or_test):
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 512
    critic_lr = 3e-4 
    critic_layer_num = 2
    task_v_min = 0
    task_v_max = 200
    
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = RainbowDQN(
        device=device,
        obs_dim=obs_dim,
        mlp_hidden_dim=hidden_dim,
        action_dim=action_dim,
        critic_layer_num=critic_layer_num,
        critic_lr=critic_lr,
        task_v_min=task_v_min,
        task_v_max=task_v_max,
    )

    model_save_folder = 'trained_models'
    os.makedirs(model_save_folder, exist_ok=True)
    save_name = 'rainbow_dqn_discrete_{}_demo'.format(env_name)
    save_path = os.path.join(model_save_folder, save_name)

    if train_or_test == 'train':
        save_interval = 1000

        log_folder = 'logs'
        os.makedirs(log_folder, exist_ok=True)
        log_name = 'rainbow_dqn_discrete_train_{}'.format(env_name)
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

        replay_buffer = MultiStepPER(int(1e5))
        batch_size = 512
        max_timeframe = int(1e6)
        update_times = 1

        deterministic = False
        score_lst = []
        score = 0
        obs, _ = env.reset()
        for step in range(1, max_timeframe+1):
            # epsilon = max(0.01, 0.08-0.01*(step/200))
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