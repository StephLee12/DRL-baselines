import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import random 

from rlalgo_utils import NoisyLinear
from torch.distributions import Categorical,Normal

# ---------  Discrete Action Space -----------
# ******** Single Action **********
class ValueNet(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim
    ) -> None:
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self,obs):
        return self.mlp_head(obs)

class QDiscreteSingleAction(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        action_dim
    ) -> None:
        super().__init__() 

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim)
        )

    def forward(self,obs):
        return self.mlp_head(obs)
    

    def get_action(self,obs,epsilon,deterministic=False): # epsilon greedy
        q = self.forward(obs) 
        if deterministic:
            return q.argmax().detach().cpu().item()
        else:
            rnd = random.random()
            if rnd < epsilon:
                return random.randint(0,1)
            else:
                return q.argmax().detach().cpu().item()

# Dueling DQN 
class DuelingQDiscreteSingleAction(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        action_dim
    ) -> None:
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
        )

        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ) # an advantage head 

        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ) # a value head 

    def forward(self, obs):
        obs = self.mlp_head(obs)

        v = self.v_head(obs)
        adv = self.adv_head(obs)

        # adv = q - v; to solve the "identification" problem, the adv is divived by the mean to make sure that the gradient descent updates both v_head and adv_head
        q = v + adv - adv.mean(dim=-1, keepdim=True)

        return q 
    
    def get_action(self,obs,epsilon,deterministic=False): # epsilon greedy
        q = self.forward(obs) 
        if deterministic:
            return q.argmax().detach().cpu().item()
        else:
            rnd = random.random()
            if rnd < epsilon:
                return random.randint(0,1)
            else:
                return q.argmax().detach().cpu().item()

# Noisy Net
class NoisyQDiscreteSingleAction(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        action_dim
    ) -> None:
        super().__init__()

        modules = [nn.Linear(obs_dim,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,hidden_dim), nn.ReLU()]

        self.noisy_lst = [NoisyLinear(hidden_dim, hidden_dim), NoisyLinear(hidden_dim, action_dim)]

        for idx, noise_layer in enumerate(self.noisy_lst):
            modules.append(noise_layer)
            if idx != len(self.noisy_lst)-1: modules.append(nn.ReLU())

        self.mlp_head = nn.Sequential(*modules)


    def forward(self,obs):
        return self.mlp_head(obs)

    def reset_noise(self):
        for noise_layer in self.noisy_lst:
            noise_layer.reset_noise()
    

    def get_action(self,obs): # replace epsilon greedy
        q = self.forward(obs) 
        return q.argmax().detach().cpu().item()


# C51
class C51QDiscreteSingleAction(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        action_dim,
        atom_size,
        support
    ) -> None:
        super().__init__()
        
        self.action_dim = action_dim 
        self.atom_size = atom_size 
        self.support = support

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim * atom_size)
        ) 

    def forward(self, obs):
        dist = self.get_dist(obs=obs) 
        q = torch.sum(dist*self.support, dim=-1) # dist*support -> p*z

        return q


    def get_dist(self, obs):
        q_atoms = self.mlp_head(obs).view(-1, self.action_dim, self.atom_size) # (batch_size, action_dim, atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3) # avoid zero elements

        return dist

    def get_action(self,obs,epsilon,deterministic=False): # epsilon greedy
        q = self.forward(obs) 
        if deterministic:
            return q.argmax().detach().cpu().item()
        else:
            rnd = random.random()
            if rnd < epsilon:
                return random.randint(0,1)
            else:
                return q.argmax().detach().cpu().item()

# Rainbow 
class RainbowQDiscreteSingleAction(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        action_dim,
        atom_size,
        support
    ) -> None:
        super().__init__()

        # C51 
        self.action_dim = action_dim 
        self.atom_size = atom_size
        self.support = support

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
        )

        # dueling & Noisy & C51
        adv_head_modules, v_head_modules = [], []
        self.adv_noisy_lst = [NoisyLinear(hidden_dim, hidden_dim), NoisyLinear(hidden_dim, action_dim*atom_size)] # C51
        self.v_noisy_lst = [NoisyLinear(hidden_dim, hidden_dim), NoisyLinear(hidden_dim, atom_size)] # C51

        for idx, (adv_noise_layer, v_noise_layer) in enumerate(zip(self.adv_noisy_lst, self.v_noisy_lst)):
            adv_head_modules.append(adv_noise_layer)
            v_head_modules.append(v_noise_layer)
            if idx != len(self.adv_noisy_lst)-1:
                adv_head_modules.append(nn.ReLU())
                v_head_modules.append(nn.ReLU())

        self.adv_head = nn.Sequential(*adv_head_modules)
        self.v_head = nn.Sequential(*v_head_modules)
        
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

    def get_action(self, obs): # replace epsilon greedy
        # Noisy 
        q = self.forward(obs) 
        return q.argmax().detach().cpu().item()

    def reset_noise(self):
        # Noisy 
        for adv_noise_layer, v_noise_layer in zip(self.adv_noisy_lst, self.v_noisy_lst):
            adv_noise_layer.reset_noise()
            v_noise_layer.reset_noise()


class PolicyDiscreteSingleAction(nn.Module):
    def __init__(
        self,
        device,
        obs_dim,
        hidden_dim,
        action_dim
    ) -> None:
        super().__init__()

        self.device = device 

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self,obs):
        probs = self.mlp_head(obs)

        return probs

    def evaluate(self,obs,epsilon=1e-6):
        probs = self.forward(obs=obs)
        dist = Categorical(probs)
        action = dist.sample()

        z = (probs==0.0).float()*epsilon # avoid numerical instability 
        log_probs = torch.log(probs+z)

        return action,log_probs 
    

    def get_action(self,obs,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs = self.forward(obs=obs)
        if deterministic:
            action = np.argmax(probs.detach().cpu().numpy())
        else:
            dist = Categorical(probs)
            action = np.array(dist.sample().squeeze().detach().cpu().item())

        return action 



# ********* Multiple Action ********
class QDiscreteMultiAction(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        action_dim_lst
    ) -> None:
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU()
        )

        self.q_layer_lst = nn.ModuleList([nn.Linear(hidden_dim,action_dim) for action_dim in action_dim_lst])
    
    def forward(self,obs):
        x = self.mlp_head(obs)
        q_lst = [q_layer(x) for q_layer in self.q_layer_lst]

        return q_lst
    
    def get_action(self,obs,epsilon,deterministic):
        q_lst = self.forward(obs)

        action = []
        for q in q_lst:
            if deterministic:
                action.append(q.argmax().detach().cpu().item())
            else:
                rnd = random.random()
                if rnd < epsilon: action.append(random.randint(0,1))
                else: action.append(q.argmax().detach().cpu().item())

        return np.array(action)
                

class PolicyDiscreteMultiAction(nn.Module):
    def __init__(
        self,
        device,
        obs_dim,
        hidden_dim,
        action_dim_lst
    ) -> None:
        super().__init__()

        self.device = device 

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU()
        )

        self.action_layer_lst = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim,action_dim),
                nn.Softmax(dim=-1)
            )
        ] for action_dim in action_dim_lst)
    
    def forward(self,obs):
        x = self.mlp_head(obs)
        probs_lst = [action_layer(x) for action_layer in self.action_layer_lst]

        return probs_lst

    def evaluate(self,obs,epsilon):
        probs_lst = self.forward(obs=obs)
        dist_lst = [Categorical(probs) for probs in probs_lst]
        action = torch.cat([dist.sample().unsqueeze(-1) for dist in dist_lst],dim=-1)

        z_lst = [(probs==0.0).float()*epsilon for probs in probs_lst]
        log_probs_lst = [torch.log(probs+z) for probs,z in zip(probs_lst,z_lst)]

        return action,log_probs_lst

    def get_action(self,obs,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs_lst = self.forward(obs=obs)
        if deterministic:
            action = np.concat([np.argmax(probs.detach().cpu().numpy()) for probs in probs_lst],dim=-1) 
        else:
            dist_lst = [Categorical(probs) for probs in probs_lst]
            action = torch.cat([dist.sample().unsqueeze(-1) for dist in dist_lst],dim=-1).detach().cpu().numpy()

        return action





# ------- Proximal Policy Gradient --------------------
class PPO_ValueNet(ValueNet):
    def __init__(self, obs_dim, hidden_dim) -> None:
        super().__init__(obs_dim, hidden_dim)

class PPO_PolicyDiscreteSingleAction(PolicyDiscreteSingleAction):
    def __init__(self, device, obs_dim, hidden_dim, action_dim) -> None:
        super().__init__(device, obs_dim, hidden_dim, action_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,action_dim),
            nn.Softmax(dim=-1)
        )

    def evaluate(self,obs):
        probs = self.forward(obs=obs)

        return probs
    
    def get_action(self,obs,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs = self.forward(obs=obs)
        if deterministic:
            action = np.argmax(probs.detach().cpu().numpy())
            return action 
        else:
            dist = Categorical(probs)
            action = np.array(dist.sample().squeeze().detach().cpu().item())
            probs_a = probs[action[0]].detach().cpu().item()

            return action,probs_a

class PPO_PolicyDiscreteMultiAction(PolicyDiscreteMultiAction):
    def __init__(self, device, obs_dim, hidden_dim, action_dim_lst) -> None:
        super().__init__(device, obs_dim, hidden_dim, action_dim_lst)
    
    def evaluate(self,obs):
        probs_lst = self.forward(obs=obs)

        return probs_lst 

    def get_action(self,obs,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs_lst = self.forward(obs=obs)
        if deterministic:
            action = np.concatenate([np.argmax(probs.detach().cpu().numpy()) for probs in probs_lst])
            return action 
        else:
            dist_lst = [Categorical(probs) for probs in probs_lst] 
            action = np.concatenate([list(dist.sample().squeeze().detach().cpu().item()) for dist in dist_lst])
            probs_a_lst = [probs[action[idx][0]].detach().cpu().item() for idx,probs in enumerate(probs_lst)]

            return action,probs_a_lst



# --------- Soft Actor Critic -------
class SAC_QDiscreteSingleAction(QDiscreteSingleAction):
    def __init__(self, obs_dim, hidden_dim, action_dim) -> None:
        super().__init__(obs_dim, hidden_dim, action_dim)

class SAC_PolicyDiscreteSingleAction(PolicyDiscreteSingleAction):
    def __init__(self, device, obs_dim, hidden_dim, action_dim) -> None:
        super().__init__(device, obs_dim, hidden_dim, action_dim)

class SAC_QDiscreteMultiAction(QDiscreteMultiAction):
    def __init__(self, obs_dim, hidden_dim, action_dim_lst) -> None:
        super().__init__(obs_dim, hidden_dim, action_dim_lst)

class SAC_PolicyDiscreteMultiAction(PolicyDiscreteMultiAction):
    def __init__(self, device, obs_dim, hidden_dim, action_dim_lst) -> None:
        super().__init__(device, obs_dim, hidden_dim, action_dim_lst)






# --------- Continuous Action Space -----------
# ********** Single Action *******
class QContinuousMultiActionSingleOutLayer(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim
    ) -> None:
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim+action_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self,obs,action):
        x = torch.cat([obs,action],dim=-1)
        q = self.mlp_head(x)

        return q

class DeterministicContinuousPolicyMultiActionSingleOutLayer(nn.Module):
    def __init__(
        self,
        device,
        obs_dim,
        action_dim,
        hidden_dim
    ) -> None:
        super().__init__()

        self.device = device 

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim)
        )

    def forward(self,obs):
        return torch.tanh(self.mlp_head(obs))
    
    def evaluate(self,obs):
        return torch.tanh(self.mlp_head(obs))

    def get_action(self,obs):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return torch.tanh(self.mlp_head(obs)).detach().cpu().numpy() 

class GaussianContinuousPolicyMultiActionSingleOutLayer(nn.Module):
    def __init__(
        self,
        device,
        obs_dim,
        action_dim,
        hidden_dim,
        log_std_min=-20,
        log_std_max=2
    ) -> None:
        super().__init__()

        self.device = device 

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(hidden_dim,action_dim)
        self.log_std_layer = nn.Linear(hidden_dim,action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max 
    
    def forward(self,obs):
        x = self.mlp_head(obs)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std,self.log_std_min,self.log_std_max)

        return mean,log_std 

    
    def evaluate(self,obs,epsilon=1e-6):
        mean,log_std = self.forward(obs)
        std = log_std.exp()


        normal = Normal(0,1)
        z = normal.sample(mean.shape).to(self.device)
        action = torch.tanh(mean+std*z)
        log_probs = Normal(mean,std).log_prob(mean+std*z) - torch.log(1-action.pow(2)+epsilon)
        log_probs = log_probs.sum(dim=-1,keepdim=True)

        return action,log_probs

    
    def get_action(self,obs,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean,log_std = self.forward(obs)

        if deterministic:
            return torch.tanh(mean).detach().cpu().numpy()
        else:
            std = log_std.exp()
            normal = Normal(0,1)
            z = normal.sample(mean.shape).to(self.device)
            action = torch.tanh(mean+std*z)

            return action.detach().cpu().numpy()



# ********* Multiple Action ********
class QContinuousMultiActionMultiOutLayer(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim
    ) -> None:
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim+action_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU()
        )

        self.q_layer_lst = nn.ModuleList([nn.Linear(hidden_dim,1) for _ in range(action_dim)])

    def forward(self,obs,action):
        x = torch.cat([obs,action],dim=-1)
        x = self.mlp_head(x)

        q_lst = [q_layer(x) for q_layer in self.q_layer_lst]

        return q_lst

class DeterministicContinuousPolicyMultiActionMultiOutLayer(nn.Module):
    def __init__(
        self,
        device,
        obs_dim,
        action_dim,
        hidden_dim
    ) -> None:
        super().__init__()

        self.device = device 

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU()
        )

        self.action_layer_lst = nn.ModuleList([nn.Linear(hidden_dim,1) for _ in range(action_dim)])

    def forward(self,obs):
        x = self.mlp_head(obs)
        return torch.cat([torch.tanh(action_layer(x)) for action_layer in self.action_layer_lst],dim=-1)

    def evaluate(self,obs):
        x = self.mlp_head(obs)
        return torch.cat([torch.tanh(action_layer(x)) for action_layer in self.action_layer_lst],dim=-1)

    def get_action(self,obs):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        x = self.mlp_head(obs)
        return torch.cat([torch.tanh(action_layer(x)) for action_layer in self.action_layer_lst],dim=-1).detach().cpu().numpy()

class GaussianContinuousPolicyMultiActionMultiOutLayer(nn.Module):
    def __init__(
        self,
        device,
        obs_dim,
        action_dim,
        hidden_dim,
        log_std_min=-20,
        log_std_max=2
    ) -> None:
        super().__init__()

        self.device = device 

        self.mlp_head = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU()
        )

        self.mean_layer_lst = nn.ModuleList([nn.Linear(hidden_dim,1) for _ in range(action_dim)])
        self.log_std_layer_lst = nn.ModuleList([nn.Linear(hidden_dim,1) for _ in range(action_dim)])

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max 
    
    def forward(self,obs):
        x = self.mlp_head(obs)

        mean = torch.concat([mean_layer(x) for mean_layer in self.mean_layer_lst],dim=-1)
        log_std = torch.concat([torch.clamp(log_std_layer(x),self.log_std_min,self.log_std_max) for log_std_layer in self.log_std_layer_lst],dim=-1)

        return mean,log_std 

    
    def evaluate(self,obs,epsilon=1e-6):
        mean,log_std = self.forward(obs)
        std = log_std.exp()

        action_lst = []
        log_probs_lst = []
        for each_mean,each_std in zip(mean,std):
            normal = Normal(0,1)
            z = normal.sample(each_mean.shape).to(self.device)
            action = torch.tanh(each_mean+each_std*z)
            log_probs = Normal(each_mean,each_std).log_prob(each_mean+each_std*z) - torch.log(1-action.pow(2)+epsilon)
            log_probs = log_probs.sum(dim=-1,keepdim=True)

            action_lst.append(action)
            log_probs_lst.append(log_probs)
        
        action = torch.concat(action_lst,dim=-1)

        return action,log_probs_lst
    

    def get_action(self,obs,determinisitc=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean,log_std = self.forward(obs)

        if determinisitc:
            return torch.tanh(mean).detach().cpu().numpy() 
        else:
            std = log_std.exp()
            action_lst = []
            for each_mean,each_std in zip(mean,std):
                normal = Normal(0,1)
                z = normal.sample(each_mean.shape).to(self.device)
                action = torch.tanh(each_mean+each_std*z)

                action_lst.append(action)
            action = torch.concat(action_lst,dim=-1)

            return action.detach().cpu().numpy()


# ----------- Proximal Policy Optimization ------------------------
class PPO_GaussianContinuousPolicyMultiActionSingleOutLayer(GaussianContinuousPolicyMultiActionSingleOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max)


    def evaluate(self, obs, epsilon=1e-6):
        mean, log_std = self.forward(obs=obs)
        std = log_std.exp()

        normal = Normal(mean, std)
        action = normal.sample()
        log_probs = normal.log_prob(action) - torch.log(1.-action.pow(2)+epsilon)
        log_probs = log_probs.sum(dim=-1,keepdim=True)

        return log_probs

    def get_action(self,obs,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean,log_std = self.forward(obs)

        if deterministic:
            return torch.tanh(mean).detach().cpu().numpy()
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())

            return action.detach().cpu().numpy()
        
class PPO_GaussianContinuousPolicyMultiActionMultiOutLayer(GaussianContinuousPolicyMultiActionMultiOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max)

    def evaluate(self, obs, epsilon=1e-6):
        mean,log_std = self.forward(obs)
        std = log_std.exp()

        log_probs_lst = []
        for each_mean,each_std in zip(mean,std):
            normal = Normal(each_mean, each_std)
            action = normal.sample()
            log_probs = normal.log_prob(action) - torch.log(1.-action.pow(2)+epsilon)
            log_probs = log_probs.sum(dim=-1,keepdim=True)

            log_probs_lst.append(log_probs)

        return log_probs_lst
    

    def get_action(self,obs,determinisitc=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean,log_std = self.forward(obs)

        if determinisitc:
            return torch.tanh(mean).detach().cpu().numpy() 
        else:
            std = log_std.exp()
            action_lst = []
            for each_mean,each_std in zip(mean,std):
                normal = Normal(each_mean, each_std)
                action = torch.tanh(normal.sample())
                action_lst.append(action)

            action = torch.concat(action_lst,dim=-1)

            return action.detach().cpu().numpy()




# --------- Deep Deterministic Policy Gradient -------------
class DDPG_QContinuousMultiActionSingleOutLayer(QContinuousMultiActionSingleOutLayer):
    def __init__(self, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, hidden_dim)
    
class DDPG_DeterministicContinuousPolicyMultiActionSingleOutLayer(DeterministicContinuousPolicyMultiActionSingleOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim)
    
    def evaluate(self,obs,noise_scale=1.0):
        
        action = self.forward(obs)
        noise = noise_scale * Normal(0,1).sample(action.shape).to(self.device)
        action = action + noise
        action = action.clamp(-1.,1.)

        return action

    def get_action(self,obs,noise_scale=1.0,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if deterministic:
            return self.forward(obs).detach().cpu().numpy()
        else:
            action = self.forward(obs)
            noise = noise_scale * Normal(0,1).sample(action.shape).to(self.device)
            action = action + noise
            action = action.clamp(-1.,1.)

            return action.detach().cpu().numpy()
    
class DDPG_GaussianContinuousPolicyMultiActionSingleOutLayer(GaussianContinuousPolicyMultiActionSingleOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max)
    

    def evaluate(self,obs,noise_scale=1.0):
        mean,log_std = self.forward(obs)
        std = log_std.exp()


        normal = Normal(0,1)
        z = normal.sample(mean.shape).to(self.device)
        action = torch.tanh(mean+std*z)

        noise = noise_scale * Normal(0,1).sample(action.shape).to(self.device)
        action = action + noise 
        action = action.clamp(-1.,1.)

        return action
    
    def get_action(self,obs,noise_scale=1.0,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean,log_std = self.forward(obs)

        if deterministic:
            return mean.detach().cpu().numpy()
        else:
            std = log_std.exp()
            
            normal = Normal(0,1)
            z = normal.sample(mean.shape).to(self.device)
            action = torch.tanh(mean+std*z)

            noise = noise_scale * Normal(0,1).sample(action.shape).to(self.device)
            action = action + noise 
            action = action.clamp(-1.,1.)

            return action.detach().cpu().numpy()

class DDPG_QContinuousMultiActionMultiOutLayer(QContinuousMultiActionMultiOutLayer):
    def __init__(self, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, hidden_dim)

class DDPG_DeterministicContinuousPolicyMultiActionMultiOutLayer(DeterministicContinuousPolicyMultiActionMultiOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim)
    
    def evaluate(self,obs,noise_scale=1.0):
        action = self.forward(obs)
        noise = noise_scale * Normal(0,1).sample(action.shape).to(self.device)
        action = action + noise
        action = action.clamp(-1.,1.)

        return action 
    
    def get_action(self,obs,noise_scale=1.0,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if deterministic:
            return self.forward(obs).detach().cpu().numpy()
        else:
            action = self.forward(obs)
            noise = noise_scale * Normal(0,1).sample(action.shape).to(self.device)
            action = action + noise
            action = action.clamp(-1.,1.)

            return action.detach().cpu().numpy()

class DDPG_GaussianContinuousPolicyMultiActionMultiOutLayer(GaussianContinuousPolicyMultiActionMultiOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max)

    def evaluate(self,obs,noise_scale=1.0):
        mean,log_std = self.forward(obs)
        std = log_std.exp()


        normal = Normal(0,1)
        z = normal.sample(mean.shape).to(self.device)
        action = torch.tanh(mean+std*z)

        noise = noise_scale * Normal(0,1).sample(action.shape).to(self.device)
        action = action + noise 
        action = action.clamp(-1.,1.)

        return action
    
    def get_action(self,obs,noise_scale=1.0,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean,log_std = self.forward(obs)

        if deterministic:
            return mean.detach().cpu().numpy()
        else:
            std = log_std.exp()
            
            normal = Normal(0,1)
            z = normal.sample(mean.shape).to(self.device)
            action = torch.tanh(mean+std*z)

            noise = noise_scale * Normal(0,1).sample(action.shape).to(self.device)
            action = action + noise 
            action = action.clamp(-1.,1.)

            return action.detach().cpu().numpy()




# -------- Twined Delayed Deep Deterministic Policy Gradient ---------
class TD3_QContinuousMultiActionSingleOutLayer(QContinuousMultiActionSingleOutLayer):
    def __init__(self, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, hidden_dim)

class TD3_DeterministicContinuousPolicyMultiActionSingleOutLayer(DeterministicContinuousPolicyMultiActionSingleOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim)
    
    def evaluate(self,obs,eval_noise_scale):
        action = self.forward(obs)
        noise = eval_noise_scale * Normal(0,1).sample(action.shape).to(self.device)
        noise = noise.clamp(-2*eval_noise_scale,2*eval_noise_scale)
        action = action + noise
        action = action.clamp(-1.,1.)

        return action 

    def get_action(self,obs,explore_noise_scale,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if deterministic:
            return self.forward(obs).detach().cpu().numpy()
        else:
            action = self.forward(obs)
            noise = explore_noise_scale * Normal(0,1).sample(action.shape).to(self.device)
            action = action + noise
            action = action.clamp(-1.,1.)

            return action.detach().cpu().numpy() 

class TD3_GaussianContinuousPolicyMultiActionSingleOutLayer(GaussianContinuousPolicyMultiActionSingleOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max)

    
    def evaluate(self,obs,eval_noise_scale):
        mean,log_std = self.forward(obs)
        std = log_std.exp()


        normal = Normal(0,1)
        z = normal.sample(mean.shape).to(self.device)
        action = torch.tanh(mean+std*z)

        noise = eval_noise_scale * Normal(0,1).sample(action.shape).to(self.device)
        noise = noise.clamp(-2*eval_noise_scale,2*eval_noise_scale)
        action = action + noise 
        action = action.clamp(-1.,1.)

        return action
    

    def get_action(self,obs,explore_noise_scale=1.0,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean,log_std = self.forward(obs)

        if deterministic:
            return mean.detach().cpu().numpy()
        else:
            std = log_std.exp()
            
            normal = Normal(0,1)
            z = normal.sample(mean.shape).to(self.device)
            action = torch.tanh(mean+std*z)

            noise = explore_noise_scale * Normal(0,1).sample(action.shape).to(self.device)
            action = action + noise 
            action = action.clamp(-1.,1.)

            return action.detach().cpu().numpy()

class TD3_QContinuousMultiActionMultiOutLayer(QContinuousMultiActionMultiOutLayer):
    def __init__(self, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, hidden_dim)

class TD3_DeterministicContinuousPolicyMultiActionMultiOutLayer(DeterministicContinuousPolicyMultiActionMultiOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim)

    def evaluate(self,obs,eval_noise_scale):
        action = self.forward(obs)
        noise = eval_noise_scale * Normal(0,1).sample(action.shape).to(self.device)
        noise = noise.clamp(-2*eval_noise_scale,2*eval_noise_scale)
        action = action + noise
        action = action.clamp(-1.,1.)

        return action 

    def get_action(self,obs,explore_noise_scale,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if deterministic:
            return self.forward(obs).detach().cpu().numpy()
        else:
            action = self.forward(obs)
            noise = explore_noise_scale * Normal(0,1).sample(action.shape).to(self.device)
            action = action + noise
            action = action.clamp(-1.,1.)

            return action.detach().cpu().numpy() 

class TD3_GaussianContinuousPolicyMultiActionMultiOutLayer(GaussianContinuousPolicyMultiActionMultiOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max)

    def evaluate(self,obs,eval_noise_scale):
        mean,log_std = self.forward(obs)
        std = log_std.exp()


        normal = Normal(0,1)
        z = normal.sample(mean.shape).to(self.device)
        action = torch.tanh(mean+std*z)

        noise = eval_noise_scale * Normal(0,1).sample(action.shape).to(self.device)
        noise = noise.clamp(-2*eval_noise_scale,2*eval_noise_scale)
        action = action + noise 
        action = action.clamp(-1.,1.)

        return action
    

    def get_action(self,obs,explore_noise_scale=1.0,deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean,log_std = self.forward(obs)

        if deterministic:
            return mean.detach().cpu().numpy()
        else:
            std = log_std.exp()
            
            normal = Normal(0,1)
            z = normal.sample(mean.shape).to(self.device)
            action = torch.tanh(mean+std*z)

            noise = explore_noise_scale * Normal(0,1).sample(action.shape).to(self.device)
            action = action + noise 
            action = action.clamp(-1.,1.)

            return action.detach().cpu().numpy()



# --------- Soft Actor Critic -----------------
class SAC_QContinuousMultiActionSingleOutLayer(QContinuousMultiActionSingleOutLayer):
    def __init__(self, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, hidden_dim)

class SAC_GaussianContinuousPolicyMultiActionSingleOutLayer(GaussianContinuousPolicyMultiActionSingleOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max)

    def evaluate(self,obs,epsilon=1e-6):
        mean,log_std = self.forward(obs)
        std = log_std.exp()


        normal = Normal(0,1)
        z = normal.sample(mean.shape).to(self.device)
        action = torch.tanh(mean+std*z)
        log_probs = Normal(mean,std).log_prob(mean+std*z) - torch.log(1-action.pow(2)+epsilon)
        log_probs = log_probs.sum(dim=-1,keepdim=True) # sum in log is equal to probability multiplication 

        return action,log_probs
    
    def get_action(self,obs,deterministic):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean,log_std = self.forward(obs)

        if deterministic:
            return mean.detach().cpu().numpy()
        else:
            std = log_std.exp()
            
            normal = Normal(0,1)
            z = normal.sample(mean.shape).to(self.device)
            action = torch.tanh(mean+std*z)

            return action.detach().cpu().numpy()
  
class SAC_QContinuousMultiActionMultiOutLayer(QContinuousMultiActionMultiOutLayer):
    def __init__(self, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, hidden_dim)

class SAC_GaussianContinuousPolicyMultiActionMultiOutLayer(GaussianContinuousPolicyMultiActionMultiOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max)
    
    def evaluate(self,obs,epsilon=1e-6):
        mean,log_std = self.forward(obs)
        std = log_std.exp()


        normal = Normal(0,1)
        z = normal.sample(mean.shape).to(self.device)
        action = torch.tanh(mean+std*z)

        log_probs_lst = []
        for idx in range(action.shape[1]):
            log_probs = Normal(mean[:,idx],std[:,idx]).log_prob(mean[:,idx]+std[:,idx]*z[:,idx]) - torch.log(1-action[:,idx].pow(2)+epsilon)
            # log_probs = log_probs.sum(dim=-1,keepdim=True) # only one dim, no need sum 
            log_probs_lst.append(log_probs)

        return action,log_probs_lst
    
    def get_action(self,obs,deterministic):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean,log_std = self.forward(obs)

        if deterministic:
            return mean.detach().cpu().numpy()
        else:
            std = log_std.exp()
            
            normal = Normal(0,1)
            z = normal.sample(mean.shape).to(self.device)
            action = torch.tanh(mean+std*z)

            return action.detach().cpu().numpy()