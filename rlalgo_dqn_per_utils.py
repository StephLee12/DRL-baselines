import torch 
import torch.nn as nn 

def _get_qloss(
    q_a,
    tar_q_a,
    weights
):
    q_loss_func = nn.SmoothL1Loss(reduction='none') # calculate element-wise loss 
    q_element_wise_loss = q_loss_func(q_a, tar_q_a.detach())

    # aggregate above loss with weights 
    qloss = torch.mean(q_element_wise_loss * weights)
    
    return q_element_wise_loss, qloss



def _update_PER(
    q_element_wise_loss,
    prior_eps,
    replay_buffer,
    indices
):
    ## PER: update priorities
    loss_for_PER = q_element_wise_loss.detach().cpu().numpy()
    new_priorties = loss_for_PER + prior_eps # based on TD-error 
    replay_buffer.update_priorities(indices=indices, priorities=new_priorties) 
    
   