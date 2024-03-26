import torch 
import torch.nn as nn 

import math 
import numpy as np 


from torch.distributions import Normal 
from torch.nn.utils import clip_grad_norm_



from rlalgo_utils import _network_update


def _get_action(
    obs, 
    deterministic,
    device,
    policy,
    action_range,
    log_prob_min,
    log_prob_max
):
    obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
    mean, std = policy.forward(obs=obs)

    if deterministic:
        return action_range * mean.detach().cpu().numpy().flatten(), None
    else:
        dist = Normal(mean, std)
        action = action_range * dist.sample()
        action = action.clamp(-action_range, action_range)
        log_probs = dist.log_prob(action).clamp(-log_prob_min, log_prob_max).sum(dim=-1, keepdim=True)
        probs = torch.exp(log_probs + 1e-10)

        return action.detach().cpu().numpy().flatten(), probs.detach().cpu().numpy().flatten()
    
    

def _get_adv(
    MC_or_GAE,
    device,
    obs,
    next_obs,
    reward,
    done,
    dw,
    lamda,
    gamma,
    val_net
):
    if MC_or_GAE == 'MC':
        trajectory_reward_lst = []
        discounted_reward = 0
        for step_reward, step_done in zip(reward.detach().cpu().numpy().flatten()[::-1], done.detach().cpu().numpy().flatten()[::-1]):
            if step_done: discounted_reward = 0
            discounted_reward = step_reward + gamma * discounted_reward
            trajectory_reward_lst.append(discounted_reward)
        trajectory_reward_lst.reverse()
        trajectory_reward_lst = torch.FloatTensor(trajectory_reward_lst).to(device).unsqueeze(-1)

        adv = trajectory_reward_lst - val_net(obs=obs).detach()

        tar_v = trajectory_reward_lst
    else:
        v = val_net.forward(obs)
        next_v = val_net.forward(next_obs)

        deltas = reward + gamma * (1-dw) * next_v.detach() - v.detach()
        deltas = deltas.detach().cpu().numpy().flatten()

        adv_lst = []
        adv = 0.0
        for delta_t, done_t in zip(deltas[::-1], done.detach().cpu().numpy().flatten()[::-1]):
            adv = gamma * lamda * adv * (1-done_t) + delta_t
            adv_lst.append(adv)
        adv_lst.reverse()
        adv = torch.FloatTensor(adv_lst).to(device).unsqueeze(-1)

        tar_v = adv + v.detach()
        
        
    return adv, tar_v


def _minibatch_update(
    batch_size,
    K_epoch,
    device,
    obs,
    action,
    tar_v,
    adv,
    old_probs_a,
    policy,
    val_net,
    epsilon_clip,
    policy_optim,
    val_net_optim,
    is_clip_gradient,
    clip_gradient_val,
    log_prob_min,
    log_prob_max,
):
    minibatch_iter_num = int(math.ceil(obs.shape[0] / batch_size))
    for _ in range(K_epoch):
        obs, action, tar_v, adv, old_probs_a = _shuffle_trajectory(
            device=device,
            obs=obs,
            action=action,
            tar_v=tar_v,
            adv=adv,
            old_probs_a=old_probs_a
        )

        for iter_num in range(minibatch_iter_num):
            index = slice(iter_num*batch_size, min((iter_num+1)*batch_size,obs.shape[0]))
            
            policy_loss = _get_policy_loss(
                policy=policy,
                index=index,
                obs=obs,
                action=action,
                old_probs_a=old_probs_a,
                adv=adv,
                epsilon_clip=epsilon_clip,
                log_prob_min=log_prob_min,
                log_prob_max=log_prob_max
            )            
            
            
            _network_update(
                optimizer=policy_optim,
                loss=policy_loss,
                is_clip_gradient=is_clip_gradient,
                clip_parameters=policy.parameters(),
                clip_gradient_val=clip_gradient_val
            )
            

            vloss = _get_v_loss(
                index=index,
                val_net=val_net,
                tar_v=tar_v,
                obs=obs
            )
            
            _network_update(
                optimizer=val_net_optim,
                loss=vloss,
                is_clip_gradient=is_clip_gradient,
                clip_parameters=val_net.parameters(),
                clip_gradient_val=clip_gradient_val
            )



def _get_policy_loss(
    policy,
    index,
    obs,
    action,
    old_probs_a,
    adv,
    epsilon_clip,
    log_prob_min,
    log_prob_max,
):
    dist = policy.evaluate(obs=obs[index])
    log_probs_a = dist.log_prob(action[index]).clamp(log_prob_min, log_prob_max).sum(dim=-1, keepdim=True)
    probs_a = torch.exp(log_probs_a + 1e-10)
    # dist_entropy = dist.entropy()
    ratio = 1 + (torch.log(probs_a) - torch.log(old_probs_a[index])) # taylor expansion: e^x = 1+x when x->0
    # ratio = torch.exp(torch.log(probs_a) - torch.log(old_probs_a[index]))
    
    
    surr1 = ratio * adv[index]
    surr2 = torch.clamp(ratio, 1-epsilon_clip, 1+epsilon_clip) * adv[index] 

    policy_loss = -torch.min(surr1, surr2).mean() 
    # policy_loss = (-torch.min(surr1, surr2) - entropy_coeff*dist_entropy).mean()

    
    return policy_loss



def _get_v_loss(
    index,
    val_net,
    tar_v,
    obs
):
    v_loss_func = nn.MSELoss()
    
    return v_loss_func(val_net.forward(obs=obs[index]), tar_v[index]) 
    



def _shuffle_trajectory(
    device,
    obs,
    action,
    tar_v,
    adv,
    old_probs_a
):
    # shuffle trajectory
    perm = np.arange(obs.shape[0])
    np.random.shuffle(perm)
    perm = torch.LongTensor(perm).to(device)
    
    return obs[perm].clone(), action[perm].clone(), tar_v[perm].clone(), adv[perm].clone(), old_probs_a[perm].clone()