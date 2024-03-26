import torch 
import torch.nn as nn 

from torch.distributions import Normal

def _get_action(
    obs,
    device,
    deterministic,
    policy,
    action_range,
    noise_std
):
    obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
    
    if deterministic:
        return torch.tanh(policy.forward(obs)).detach().cpu().numpy().flatten()
    else:
        action = action_range * torch.tanh(policy.forward(obs))
        noise = Normal(0, noise_std * action_range).sample(action.shape).to(device)
        action = action + noise
        action = action.clamp(-action_range, action_range)

        return action.detach().cpu().numpy().flatten()
    


def _get_target_q(
    task_or_safe,
    tar_policy,
    next_obs,
    noise_std,
    tar_critic_head,
    reward,
    cost,
    dw,
    gamma
):
    new_next_action = tar_policy.evaluate(obs=next_obs, noise_std=noise_std)
    tar_next_q = tar_critic_head.forward(obs=next_obs, action=new_next_action)
    
    if task_or_safe == 'task':
        tar_q = reward + (1-dw)*gamma*tar_next_q 
    else:
        tar_q = cost + (1-dw)*gamma*tar_next_q

    return tar_q


def _get_q(
    obs,
    action,
    critic_head
):
    return critic_head.forward(obs=obs, action=action) 


def _get_qloss(
    q,
    tar_q
):
    qloss_func = nn.MSELoss()
    
    return qloss_func(q, tar_q.detach())




def _get_policy_loss(
    task_or_safe,
    policy,
    obs,
    noise_std,
    critic_head,
    safe_critic_head,
    lamda
):
    new_action = policy.evaluate(obs=obs, noise_std=noise_std)
    new_q = critic_head.forward(obs=obs, action=new_action)
    
    if task_or_safe == 'safe':
        new_safe_q = safe_critic_head.forward(obs=obs, action=new_action)
    
    if task_or_safe == 'task': return -new_q.mean()
    else: return (-new_q + lamda*new_safe_q).mean()



