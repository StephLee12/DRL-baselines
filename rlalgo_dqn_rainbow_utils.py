import torch 

def _get_target(
    support,
    v_min,
    v_max,
    delta_z,
    critic_head,
    tar_critic_head,
    next_obs,
    reward,
    dw,
    gamma,
    batch_size,
    atom_size,
    device
):
    with torch.no_grad():
        # Double DQN; use q to select action and use tar_q to evaluate 
        next_action = critic_head.forward(obs=next_obs).argmax(dim=-1)
        next_dist = tar_critic_head.get_dist(obs=next_obs)
        next_dist = next_dist.gather(dim=1, index=next_action.unsqueeze(-1).expand(-1, atom_size).unsqueeze(1)).squeeze(1)
        
        t_z = reward + (1-dw) * gamma * support
        
        t_z = t_z.clamp(min=v_min, max=v_max)
        b = (t_z - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size-1)*atom_size, batch_size).long().unsqueeze(-1).to(device) # (batch_size, 1)

        proj_dist = torch.zeros(next_dist.size(), device=device)
        proj_dist.view(-1).index_add_(
            dim=0,
            index=(l+offset).view(-1),
            source=((u.float() + (l==u)-b) * next_dist).view(-1)
        )
        proj_dist.view(-1).index_add_(
            dim=0,
            index=(u+offset).view(-1),
            source=((b - l.float()) * next_dist).view(-1)
        )

    return proj_dist



def _get_loss(
    proj_dist,
    log_p,
    weights
):
    elementwise_loss = -(proj_dist * log_p).sum(-1) # for PER update
    loss = torch.mean(elementwise_loss * weights) # multiply with PER weights
    
    return elementwise_loss, loss 