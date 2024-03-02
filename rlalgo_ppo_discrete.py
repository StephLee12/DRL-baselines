import numpy as np 
import math
import os 
import logging 
# import gym 
import gymnasium as gym 


import torch
import torch.nn as nn 
import torch.optim as optim



from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical

from rlalgo_net import PPO_ValueNet, PPO_PolicyDiscreteSingleAction, PPO_PolicyDiscreteMultiAction


class PPO_Discrete():
    def __init__(
        self,
        device,
        is_single_or_multi_out,
        is_use_MC_or_GAE,
        obs_dim,
        hidden_dim,
        action_dim,
        policy_lr,
        v_lr
    ) -> None:

        self.device = device 
        self.is_single_or_multi_out = is_single_or_multi_out
        self.is_use_MC_or_GAE = is_use_MC_or_GAE

        self.trajectory_buffer = []

        self.v = PPO_ValueNet(obs_dim=obs_dim, hidden_dim=hidden_dim).to(device)
        if self.is_single_or_multi_out == 'single_out':
            self.policy = PPO_PolicyDiscreteSingleAction(device=device, obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)
        else: # == 'multi_out'
            self.policy = PPO_PolicyDiscreteMultiAction(device=device, obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim_lst=action_dim).to(device)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.v_optim = optim.Adam(self.v.parameters(), lr=v_lr)


    def ensemble_trajectory(self):
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
        action = torch.FloatTensor(action_lst).unsqueeze(-1).to(self.device)
        reward = torch.FloatTensor(reward_lst).unsqueeze(-1).to(self.device)
        old_probs_a = torch.FloatTensor(old_probs_a_lst).unsqueeze(-1).to(self.device)
        done = torch.FloatTensor(np.float32(done_lst)).unsqueeze(-1).to(self.device)
        dw = torch.FloatTensor(np.float32(dw_lst)).unsqueeze(-1).to(self.device)

        self.trajectory_buffer = [] # clear buffer 

        return obs, next_obs, action, reward, old_probs_a, done, dw

    def update(self, batch_size, K_epoch, gradient_clip=40, entropy_coeff=0.01, epsilon_clip=0.2, gamma=0.99, lamda=0.95):
        obs, next_obs, action, reward, old_probs_a, done, dw = self.ensemble_trajectory()
        
        if self.is_use_MC_or_GAE == 'MC': # use monte-carlo estimation 
            trajectory_reward_lst = []
            discounted_reward = 0
            for step_reward, step_done in zip(reward.detach().cpu().numpy().flatten()[::-1], done.detach().cpu().numpy().flatten()[::-1]):
                if step_done: discounted_reward = 0
                discounted_reward = step_reward + gamma * discounted_reward
                trajectory_reward_lst.append(discounted_reward)
            trajectory_reward_lst.reverse()
            trajectory_reward_lst = torch.FloatTensor(trajectory_reward_lst).to(self.device).unsqueeze(-1)

            adv = trajectory_reward_lst - self.v.forward(obs=obs).detach()

            tar_v = trajectory_reward_lst

        else: # use general advantage estimation (GAE) using TD rather than MC
            v = self.v.forward(obs)
            next_v = self.v.forward(next_obs)

            deltas = reward + gamma * (1-dw) * next_v.detach() - v.detach()
            deltas = deltas.detach().cpu().numpy().flatten()

            adv_lst = []
            adv = 0.0
            for delta_t, done_t in zip(deltas[::-1], done.detach().cpu().numpy().flatten()[::-1]):
                # adv = gamma * lamda * adv * (1-done) + delta_t
                adv = gamma * lamda * adv * (1-done_t) + delta_t
                adv_lst.append(adv)
            adv_lst.reverse()
            adv = torch.FloatTensor(adv_lst).to(self.device).unsqueeze(-1)

            tar_v = adv + v.detach()
        
        adv = (adv - adv.mean(dim=0)) / (adv.std(dim=0) + 1e-6)


        # minibatch update 
        minibatch_iter_num = int(math.ceil(obs.shape[0] / batch_size))
        for _ in range(K_epoch):            
            # shuffle trajectory
            perm = np.arange(obs.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)

            obs, action, tar_v, adv, old_probs_a = obs[perm].clone(), action[perm].clone(), tar_v[perm].clone(), adv[perm].clone(), old_probs_a[perm].clone()

            for iter_num in range(minibatch_iter_num):
                index = slice(iter_num*batch_size, min((iter_num+1)*batch_size,obs.shape[0]))

                if self.is_single_or_multi_out == 'single_out':
                    probs = self.policy.evaluate(obs=obs[index])

                    dist_entropy = Categorical(probs).entropy().unsqueeze(-1)
                    probs_a = probs.gather(1, action[index].long())

                    ratio = torch.exp(torch.log(probs_a) - torch.log(old_probs_a[index])) # a/b = exp(log(a)-log(b)) this kind of operation is faster than division

                    surr1 = ratio * adv[index]
                    surr2 = torch.clamp(ratio, 1-epsilon_clip, 1+epsilon_clip) * adv[index]

                    policy_loss = (-torch.min(surr1, surr2) - entropy_coeff * dist_entropy).mean()
                    self.policy_optim.zero_grad()
                    policy_loss.backward()
                    # clip_grad_norm_(self.policy.parameters(),gradient_clip)
                    self.policy_optim.step()

                    
                    v_loss_func = nn.MSELoss()
                    v_loss = v_loss_func(self.v.forward(obs=obs[index]), tar_v[index])
                    self.v_optim.zero_grad()
                    v_loss.backward()
                    # clip_grad_norm_(self.v.parameters(), clip_grad_norm_)
                    self.v_optim.step()
                else: 
                    probs_lst = self.policy.evaluate(obs=obs[index])
                    dist_entropy_lst = [Categorical(probs).entropy().sum(dim=0,keepdim=True) for probs in probs_lst]
                    probs_a_lst = [probs.gather(1,action[index][:,idx]) for idx,probs in enumerate(probs_lst)]
                    ratio_lst = [torch.exp(torch.log(probs_a) - torch.log(old_probs_a[index])) for probs_a,old_probs_a in zip(probs_a_lst,old_probs_a)]

                    policy_loss = 0
                    for ratio,dist_entropy in zip(ratio_lst,dist_entropy_lst):
                        surr1 = ratio * adv[index]
                        surr2 = torch.clamp(ratio,1-epsilon_clip,1+epsilon_clip) * adv[index] 

                        policy_loss += (-torch.min(surr1,surr2) - entropy_coeff * dist_entropy).mean()
                    self.policy_optim.zero_grad()
                    policy_loss.backward()
                    # clip_grad_norm_(self.policy.parameters(),gradient_clip)
                    self.policy_optim.step()

                    v_loss_func = nn.MSELoss()
                    v_loss = v_loss_func(self.v(obs=obs[index]) - tar_v[index])
                    self.v_optim.zero_grad()
                    v_loss.backward()
                    self.v_optim.step()
                    
        
    def save_model(self, path):
        torch.save(self.v.state_dict(), path+'_v')
        torch.save(self.policy.state_dict(), path+'_policy')

    def load_model(self, path):
        self.v.load_state_dict(torch.load(path+'_v'))
        self.policy.load_state_dict(torch.load(path+'_policy'))
        
        self.v.eval()
        self.policy.eval()

    

def train_or_test(train_or_test):
    is_single_multi_out = 'single_out'

    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    is_use_MC_or_GAE = 'GAE'
    hidden_dim = 512
    v_lr = 3e-4
    policy_lr = 3e-4 
    
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO_Discrete(
        device=device,
        is_single_or_multi_out=is_single_multi_out,
        is_use_MC_or_GAE=is_use_MC_or_GAE,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        policy_lr=policy_lr,
        v_lr=v_lr,
    )

    model_save_folder = 'trained_models'
    os.makedirs(model_save_folder, exist_ok=True)
    save_name = 'ppo_discrete_{}_demo'.format(env_name)
    save_path = os.path.join(model_save_folder, save_name)

    if train_or_test == 'train':
        save_interval = 1000

        log_folder = 'logs'
        os.makedirs(log_folder, exist_ok=True)
        log_name = 'ppo_discrete_train_{}'.format(env_name)
        log_path = os.path.join(log_folder,log_name)
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
            
            action, probs_a = agent.policy.get_action(obs=obs, deterministic=deterministic)
            next_obs, reward, dw, tr, info = env.step(action)
            done = (dw or tr)
            agent.trajectory_buffer.append([list(obs), action, reward, list(next_obs), probs_a, done, dw])
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
                score_mean = np.mean(score_lst)
                print('---Current step:{}----Mean Score:{:.2f}'.format(step,score_mean))
                logger.info('---Current step:{}----Mean Score:{:.2f}'.format(step,score_mean))


            # if done: obs = env.reset()
            # else: obs = next_obs 

        agent.save_model(save_path)
    
    else: # == 'test'
        eval_timeframe = 100

        log_folder = 'logs'
        os.makedirs(log_folder,exist_ok=True)
        log_name = 'ppo_discrete_test_{}'.format(env_name)
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
        res_save_name = 'ppo_discrete_{}'.format(env_name)
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