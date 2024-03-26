import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np 

from torch.distributions import Normal

from rlalgo_utils import NoisyLinear




class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp_hidden_dim,
        output_dim,
        layer_num
    ) -> None:
        super().__init__()

        mlp_head = []
        for idx in range(layer_num):
            if idx == 0: i_dim = input_dim
            else: i_dim = mlp_hidden_dim

            if idx == layer_num -1: o_dim = output_dim
            else: o_dim = mlp_hidden_dim

            mlp_head.append(nn.Linear(i_dim, o_dim))
            if idx != layer_num -1: mlp_head.append(nn.ReLU())
        
        self.mlp_head = nn.Sequential(*mlp_head)


    def forward(self, obs):
        return self.mlp_head(obs)
    


class DuelingCritic(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp_hidden_dim,
        output_dim,
        layer_num
    ) -> None:
        super().__init__()
        
        mlp_head = []
        for idx in range(layer_num):
            if idx == 0: i_dim = input_dim
            else: i_dim = mlp_hidden_dim

            mlp_head.append(nn.Linear(i_dim, mlp_hidden_dim))
            mlp_head.append(nn.ReLU())
        
        self.mlp_head = nn.Sequential(*mlp_head)
        
        
        self.adv_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        ) # an advantage head 

        self.v_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        ) # a value head 
        
    
    def forward(self, obs):
        obs = self.mlp_head(obs)

        v = self.v_head(obs)
        adv = self.adv_head(obs)

        # adv = q - v; to solve the "identification" problem, the adv is divived by the mean to make sure that the gradient descent updates both v_head and adv_head
        q = v + adv - adv.mean(dim=-1, keepdim=True)

        return q 
        


class NoisyCritic(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp_hidden_dim,
        output_dim,
        layer_num,
        noisy_layer_num
    ) -> None:
        super().__init__()
        
        mlp_head = []
        for idx in range(layer_num):
            if idx == 0: i_dim = input_dim
            else: i_dim = mlp_hidden_dim

            mlp_head.append(nn.Linear(i_dim, mlp_hidden_dim))
            mlp_head.append(nn.ReLU())
            
        self.noisy_lst = []
        for idx in range(noisy_layer_num):
            if idx != noisy_layer_num -1: 
                self.noisy_lst.append(NoisyLinear(mlp_hidden_dim, mlp_hidden_dim))
            else:
                self.noisy_lst.append(NoisyLinear(mlp_hidden_dim, output_dim))
        
        for idx, noisy_layer in enumerate(self.noisy_lst):
            mlp_head.append(noisy_layer)
            if idx != len(self.noisy_lst)-1: mlp_head.append(nn.ReLU())


        self.mlp_head = nn.Sequential(*mlp_head)
        
    
    def forward(self,obs):
        return self.mlp_head(obs)
    

    def reset_noise(self):
        for noisy_layer in self.noisy_lst:
            noisy_layer.reset_noise()


class C51MLPHead(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        action_dim,
        layer_num,
        atom_size,
        support
    ) -> None:
        super().__init__()
        
        self.action_dim = action_dim 
        self.atom_size = atom_size 
        self.support = support
        
        mlp_head = []
        for idx in range(layer_num):
            if idx == 0: input_dim = obs_dim
            else: input_dim = hidden_dim
            
            if idx == layer_num-1: output_dim = action_dim * atom_size
            else: output_dim = hidden_dim
            
            mlp_head.append(nn.Linear(input_dim, output_dim))
            if idx != layer_num-1: mlp_head.append(nn.ReLU())
            
        self.mlp_head = nn.Sequential(*mlp_head)
        
    
    def forward(self, obs):
        dist = self.get_dist(obs=obs) 
        q = torch.sum(dist*self.support, dim=-1) # dist*support -> p*z

        return q


    def get_dist(self, obs):
        q_atoms = self.mlp_head(obs).view(-1, self.action_dim, self.atom_size) # (batch_size, action_dim, atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3) # avoid zero elements

        return dist


class RainbowCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        action_dim,
        layer_num,
        atom_size,
        support
    ) -> None:
        super().__init__()
        
        # C51 
        self.action_dim = action_dim 
        self.atom_size = atom_size
        self.support = support
        
        
        mlp_head = []
        for idx in range(layer_num):
            if idx == 0: input_dim = obs_dim
            else: input_dim = hidden_dim
        
            mlp_head.append(nn.Linear(input_dim, hidden_dim))
            mlp_head.append(nn.ReLU())
            
        self.mlp_head = nn.Sequential(*mlp_head)
        
        
        # dueling & Noisy & C51
        adv_head, v_head = [], []
        self.adv_noisy_lst = [NoisyLinear(hidden_dim, hidden_dim), NoisyLinear(hidden_dim, action_dim*atom_size)] # C51
        self.v_noisy_lst = [NoisyLinear(hidden_dim, hidden_dim), NoisyLinear(hidden_dim, atom_size)] # C51

        for idx, (adv_noisy_layer, v_noisy_layer) in enumerate(zip(self.adv_noisy_lst, self.v_noisy_lst)):
            adv_head.append(adv_noisy_layer)
            v_head.append(v_noisy_layer)
            if idx != len(self.adv_noisy_lst)-1:
                adv_head.append(nn.ReLU())
                v_head.append(nn.ReLU())

        self.adv_head = nn.Sequential(*adv_head)
        self.v_head = nn.Sequential(*v_head)
        
    
    def forward(self, obs):
        # C51 
        dist = self.get_dist(obs=obs) 
        q = torch.sum(dist*self.support, dim=-1) # dist*support -> p*z

        return q 
    
    
    def get_dist(self, obs):
        obs = self.mlp_head(obs)
        # Dueling 
        adv_atoms = self.adv_head(obs).view(-1, self.action_dim, self.atom_size)
        v_atoms = self.v_head(obs).view(-1, 1, self.atom_size)
        q_atoms = v_atoms + adv_atoms - adv_atoms.mean(dim=1, keepdim=True) # get mean along the action axis 

        # C51 
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3) # avoid zero elements

        return dist 
    

    def reset_noise(self):
        # Noisy 
        for adv_noise_layer, v_noise_layer in zip(self.adv_noisy_lst, self.v_noisy_lst):
            adv_noise_layer.reset_noise()
            v_noise_layer.reset_noise()
    
    
    
        




class ContinuousCritic(MLPHead):
    def __init__(self, input_dim, mlp_hidden_dim, output_dim, layer_num) -> None:
        super().__init__(input_dim, mlp_hidden_dim, output_dim, layer_num)
     

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.mlp_head(x)



class PPOMLPHead(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp_hidden_dim,
        output_dim,
        layer_num,
        log_std_min,
        log_std_max
    ) -> None:
        super().__init__()
        
        mlp_head = []
        for idx in range(layer_num):
            if idx == 0: i_dim = input_dim
            else: i_dim = mlp_hidden_dim

            mlp_head.append(nn.Linear(i_dim, mlp_hidden_dim))
            mlp_head.append(nn.Tanh())
        
        self.mlp_head = nn.Sequential(*mlp_head)
        
        self.mean_layer = nn.Linear(mlp_hidden_dim, output_dim)
        self.std_layer = nn.Linear(mlp_hidden_dim, output_dim)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    
    def forward(self, obs):
        x = self.mlp_head(obs)
        
        mean = torch.tanh(self.mean_layer(x))
        std = F.softplus(self.std_layer(x)).clamp(np.exp(self.log_std_min), np.exp(self.log_std_max))
        
        return mean, std 
    
    
    def evaluate(self, obs):
        mean, std = self.forward(obs=obs)
        
        dist = Normal(mean, std)
        
        return dist 



class DDPGMLPHead(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp_hidden_dim,
        output_dim,
        layer_num,
        action_range,
        noise_std
    ) -> None:
        super().__init__()
        
        self.mlp_head = MLPHead(
            input_dim=input_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            output_dim=output_dim,
            layer_num=layer_num
        )
        
        self.action_range = action_range 
        self.noise_std = noise_std 
        
        
    def forward(self, obs):
        return self.mlp_head.forward(obs=obs)
    
    
    def evaluate(self, obs, noise_std):
        action = self.action_range * torch.tanh(self.mlp_head.forward(obs))
        noise = Normal(0, noise_std * self.action_range).sample(action.shape).to(action.device)
        action = action + noise
        action = action.clamp(-self.action_range, self.action_range)

        return action
    
    
    
class TD3GaussianMLPHead(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp_hidden_dim,
        output_dim,
        layer_num,
        action_range,
        eval_noise_scale,
        eval_noise_clip,
        log_std_min=-20,
        log_std_max=2
    ) -> None:
        super().__init__()  
        
        self.action_range = action_range 
        self.eval_noise = eval_noise_scale * action_range
        self.eval_noise_clip = eval_noise_clip * action_range
        
        mlp_head = []
        for idx in range(layer_num):
            if idx == 0: i_dim = input_dim
            else: i_dim = mlp_hidden_dim

            mlp_head.append(nn.Linear(i_dim, mlp_hidden_dim))
            mlp_head.append(nn.ReLU())
            
        self.mlp_head = nn.Sequential(*mlp_head)
        
        self.mean_layer = nn.Linear(mlp_hidden_dim, output_dim)
        self.log_std_layer = nn.Linear(mlp_hidden_dim, output_dim)
        
        self.log_std_min = log_std_min 
        self.log_std_max = log_std_max
        

    def forward(self, obs):
        x = self.mlp_head(obs)
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    

    def evaluate(self, obs, is_tar=False):
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(mean.device)
        action = self.action_range * torch.tanh(mean+std*z)
        
        if is_tar:
            noise = (torch.randn_like(action) * self.eval_noise).clamp(-self.eval_noise_clip, self.eval_noise_clip)
            action = action + noise 
            action = action.clamp(-self.action_range, self.action_range)

        return action
    
    
    
class SACGaussianMLPHead(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp_hidden_dim,
        output_dim,
        layer_num,
        action_range,
        log_std_min=-20,
        log_std_max=2
    ) -> None:
        super().__init__()
        
        self.action_range = action_range
        
        mlp_head = []
        for idx in range(layer_num):
            if idx == 0: i_dim = input_dim
            else: i_dim = mlp_hidden_dim

            mlp_head.append(nn.Linear(i_dim, mlp_hidden_dim))
            mlp_head.append(nn.ReLU())
            
        self.mlp_head = nn.Sequential(*mlp_head)
        
        self.mean_layer = nn.Linear(mlp_hidden_dim, output_dim)
        self.log_std_layer = nn.Linear(mlp_hidden_dim, output_dim)
        
        self.log_std_min = log_std_min 
        self.log_std_max = log_std_max
    
    
    def forward(self, obs):
        x = self.mlp_head(obs)
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    
    def evaluate(self, obs, epsilon=1e-6):
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(mean.device)
        action = torch.tanh(mean+std*z)
        log_probs = Normal(mean, std).log_prob(mean+std*z) - torch.log(1-action.pow(2)+epsilon)
        log_probs = log_probs.sum(dim=-1,keepdim=True) # sum in log is equal to probability multiplication 
        
        action = self.action_range * action

        return action, log_probs
    
    


