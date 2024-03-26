import torch 
import torch.nn as nn 

from torch.distributions import Normal 

def _get_action(
    device,
    obs,
    policy,
    deterministic,
    action_range
):
    obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
    mean, log_std = policy.forward(obs)

    if deterministic:
        return action_range * torch.tanh(mean).detach().cpu().numpy().flatten()
    else:
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = action_range * torch.tanh(mean+std*z)

        return action.detach().cpu().numpy().flatten()





def _get_target_q(
    task_or_safe,
    policy,
    tar_critic_head1,
    tar_critic_head2,
    next_obs,
    reward,
    cost,
    dw,
    gamma,
    alpha
):
    next_action, next_log_probs = policy.evaluate(obs=next_obs)
    tar_next_q1 = tar_critic_head1.forward(obs=next_obs, action=next_action)
    tar_next_q2 = tar_critic_head2.forward(obs=next_obs, action=next_action)
    tar_next_q = torch.min(tar_next_q1, tar_next_q2) - alpha * next_log_probs
    if task_or_safe == 'task':
        tar_q = reward + (1-dw) * gamma * tar_next_q 
    else:
        tar_q = cost + (1-dw) * gamma * tar_next_q

    return tar_q


def _get_q(
    obs,
    action,
    critic_head1,
    critic_head2
):
    q1 = critic_head1.forward(obs=obs, action=action) 
    q2 = critic_head2.forward(obs=obs, action=action)
    
    return q1, q2


def _get_qloss(
    q1,
    q2,
    tar_q
):
    qloss_func = nn.MSELoss()
    
    q1loss = qloss_func(q1, tar_q.detach())
    q2loss = qloss_func(q2, tar_q.detach())
    
    return q1loss, q2loss 


def _get_policy_loss(
    task_or_safe,
    policy,
    obs,
    critic_head1,
    critic_head2,
    safe_critic_head1,
    safe_critic_head2,
    alpha,
    lamda
):
    new_action, log_probs = policy.evaluate(obs=obs)
    
    new_q1 = critic_head1.forward(obs=obs, action=new_action)
    new_q2 = critic_head2.forward(obs=obs, action=new_action)
    new_q = torch.min(new_q1, new_q2)
    
    if task_or_safe == 'safe':
        new_safe_q1 = safe_critic_head1.forward(obs=obs, action=new_action)
        new_safe_q2 = safe_critic_head2.forward(obs=obs, action=new_action)
        new_safe_q = torch.min(new_safe_q1, new_safe_q2)
    
    if task_or_safe == 'task':
        policy_loss = (alpha.detach()*log_probs - new_q).mean()
    else:
        policy_loss = (alpha.detach()*log_probs - new_q + lamda*new_safe_q).mean() 
    
    
    return policy_loss, log_probs


def _get_alpha_loss(
    log_alpha,
    log_probs,
    tar_entropy
):
    return -(log_alpha * (log_probs + tar_entropy).detach()).mean()