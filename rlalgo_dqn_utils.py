import torch 
import torch.nn as nn 


def _get_action(
    critic_head,
    obs,
    batch_input,
    device
):
    if not batch_input: obs = torch.FloatTensor(obs).to(device).unsqueeze(0)
    else: obs = torch.FloatTensor(obs).to(device)

    qval = critic_head.forward(obs=obs)

    if not batch_input: return qval.argmax().detach().cpu().numpy().flatten() 
    else: return qval.argmax(dim=-1).detach().cpu().numpy().flatten()  
    
    

def _get_target(
    tar_critic_head,
    next_obs,
    reward,
    dw,
    gamma
):
    max_next_q_a = tar_critic_head.forward(obs=next_obs).max(dim=-1)[0].unsqueeze(-1)

    return  reward + gamma * (1-dw) * max_next_q_a 



def _get_q(
    critic_head,
    obs,
    action
):
    q = critic_head.forward(obs=obs)
    return q.gather(dim=1, index=action.unsqueeze(-1).long()) 
    
    
    
def _get_qloss(
    q_a,
    tar_q_a
):
    q_loss_func = nn.MSELoss()
    return q_loss_func(q_a, tar_q_a.detach()) 