import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from torch.distributions import Normal

## PPO
class PPO_ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_size=64, init_w=0.003) -> None:
        super(PPO_ValueNetwork,self).__init__()

        self.linear1 = nn.Linear(obs_dim,hidden_size)
        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size,hidden_size)
        self.linear4 = nn.Linear(hidden_size,1)

        self.linear4.weight.data.uniform_(-init_w,init_w)
        self.linear4.weight.data.uniform_(-init_w,init_w)
    
    def forward(self,x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        x = self.linear4(x)

        return x        

class PPO_PolicyNetwork(nn.Module):
    def __init__(self,obs_dim,action_dim,action_space,device,hidden_size=64,action_range=1.,init_w=0.003) -> None:
        super(PPO_PolicyNetwork,self).__init__()
        self.action_space = action_space
        self.action_range = action_range
        self.device = device

        self.linear1 = nn.Linear(obs_dim,hidden_size)
        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size,hidden_size)
        self.mean_linear = nn.Linear(hidden_size,action_dim)
        self.log_std_linear = nn.Linear(hidden_size,action_dim)

        self.mean_linear.weight.data.uniform_(-init_w,init_w)
        self.mean_linear.weight.data.uniform_(-init_w,init_w)
        self.log_std_linear.weight.data.uniform_(-init_w,init_w)
        self.log_std_linear.weight.data.uniform_(-init_w,init_w)
    
    def forward(self,x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        mean = torch.tanh(self.mean_linear(x))
        log_std = self.log_std_linear(x)

        return mean,log_std
    
    def evaluate(self,obs,action):
        mean, log_std = self.forward(obs)
        action = action / self.action_range
        log_prob = Normal(mean,log_std.exp()).log_prob(action)
        log_prob = log_prob.sum(dim=-1,keepdim=True)

        return log_prob

    def get_action(self,obs):
        obs = torch.tensor(obs,device=self.device,dtype=torch.float32).unsqueeze(0)
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(0,1)
        z = normal.sample()
        action = mean + std * z 
        log_prob = Normal(mean,std).log_prob(action)
        log_prob = log_prob.sum(dim=-1,keepdim=True)

        action = self.action_range * action

        return action.detach().cpu().numpy()[0],log_prob.detach().cpu().numpy()[0]
    
    def sample_action(self):
        return self.action_space.sample()

class BaseQNetwork(nn.Module):
    def __init__(self,obs_dim,action_dim,hidden_size,init_w=3e-3) -> None:
        super(BaseQNetwork,self).__init__()

        self.linear1 = nn.Linear(obs_dim+action_dim,hidden_size)
        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size,hidden_size)
        self.linear4 = nn.Linear(hidden_size,1)

        self.linear4.weight.data.uniform_(-init_w,init_w)
        self.linear4.bias.data.uniform_(-init_w,init_w)
    
    def forward(self,obs,action):
        x = torch.cat([obs,action],dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x

class BaseValueNetwork(nn.Module):
    def __init__(self,obs_dim,hidden_size,init_w=3e-3) -> None:
        super(BaseValueNetwork,self).__init__()

        self.linear1 = nn.Linear(obs_dim,hidden_size)
        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size,hidden_size)
        self.linear4 = nn.Linear(hidden_size,1)

        self.linear4.weight.data.uniform_(-init_w,init_w)
        self.linear4.bias.data.uniform_(-init_w,init_w)
    
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x 

class BasePolicyNetwork(nn.Module):
    def __init__(self) -> None:
        super(BasePolicyNetwork,self).__init__()

    def forward(self):
        pass

    def evaluate(self):
        pass

    def get_action(self):
        pass

    def sample_action(self):
        pass

## Deep deterministic policy gradient
class DDPG_QNetwork(BaseQNetwork):
    def __init__(self, obs_dim, action_dim, hidden_size, init_w=0.003) -> None:
        super().__init__(obs_dim, action_dim, hidden_size, init_w)

class DDPG_PolicyNetwork(BasePolicyNetwork):
    def __init__(
        self,
        device,
        action_space,
        obs_dim,
        action_dim,
        hidden_size,
        action_range=1.,
        init_w=3e-3
    ) -> None:
        super().__init__()
        self.device = device
        self.action_space = action_space
        self.action_range = action_range

        self.linear1 = nn.Linear(obs_dim,hidden_size)
        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size,hidden_size)
        self.linear4 = nn.Linear(hidden_size,action_dim)

        self.linear4.weight.data.uniform_(-init_w,init_w)
        self.linear4.bias.data.uniform_(-init_w,init_w)
    
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = torch.tanh(self.linear4(x))

        return x
    
    def evaluate(self,obs,noise_scale=0.1):
        normal = Normal(0,1)
        action = self.forward(obs)
        noise = noise_scale * normal.sample(action.shape).to(self.device)
        action = self.action_range * action + noise

        return action 
    
    def get_action(self,obs,noise_scale=0.1):
        obs = torch.tensor(obs,device=self.device,dtype=torch.float32).unsqueeze(0)
        normal = Normal(0,1)
        action = self.forward(obs)
        noise = noise_scale * normal.sample(action.shape).to(self.device)
        action = self.action_range * action + noise 
        
        return action.detach().cpu().numpy()[0]
    
    def sample_action(self):
        return self.action_space.sample()   

## Twin Delayed Deep Deterministic Policy Gradients (TD3)
class TD3_QNetwork(BaseQNetwork):
    def __init__(self, obs_dim, action_dim, hidden_size, init_w=0.003) -> None:
        super().__init__(obs_dim, action_dim, hidden_size, init_w)

class TD3_PolicyNetwork(BasePolicyNetwork):
    def __init__(
        self,
        device,
        action_space,
        obs_dim,
        action_dim,
        hidden_size,
        action_range=1.,
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2
    ) -> None:
        super().__init__()
        self.device = device
        self.action_space = action_space
        self.action_range = action_range
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
    
    def forward(self,obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mean = torch.tanh(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std,self.log_std_min,self.log_std_max)

        return mean,log_std

    def evaluate(self,obs,deterministic=True,eval_noise_scale=0.1,epsilon=1e-6):
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = Normal(0, 1)
        z      = normal.sample() 
        action_0 = torch.tanh(mean + std*z.to(self.device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*mean if deterministic else self.action_range*action_0
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(self.device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        ''' add noise '''
        eval_noise_clip = 2*eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = torch.clamp(noise,-eval_noise_clip, eval_noise_clip)
        action = action + noise.to(self.device)

        return action, log_prob, z, mean, log_std

    def get_action(self,obs, deterministic=True, explore_noise_scale=0.1):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = Normal(0, 1)
        z      = normal.sample().to(self.device)
        
        action = mean.detach().cpu().numpy()[0] if deterministic else torch.tanh(mean + std*z).detach().cpu().numpy()[0]

        ''' add noise '''
        noise = normal.sample(action.shape) * explore_noise_scale
        action = self.action_range*action + noise.numpy()

        return action
    
    def sample_action(self):
        return self.action_space.sample()

## Soft Actor Critic
class SoftQNetwork(BaseQNetwork):
    def __init__(self, obs_dim, action_dim, hidden_size, init_w=0.003) -> None:
        super().__init__(obs_dim, action_dim, hidden_size, init_w)

class SoftPolicyNetwork(BasePolicyNetwork):
    def __init__(
        self,
        device, 
        action_space, 
        obs_dim, 
        action_dim, 
        hidden_size, 
        action_range=1., 
        init_w=3e-3, 
        log_std_min=-20, 
        log_std_max=2
    ) -> None:
        super().__init__()
        self.device = device
        self.action_space = action_space
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.action_dim = action_dim

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # x = F.relu(self.linear4(x))

        mean    = (self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, obs, epsilon=1e-6):
        '''
        generate sampled action with obs as input wrt the policy network;
        '''
        mean, log_std = self.forward(obs)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape) 
        action_0 = torch.tanh(mean + std*z.to(self.device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(self.device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std
    
    def get_action(self, obs, deterministic=True):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(self.device)
        action = self.action_range * torch.tanh(mean + std*z)
        
        action = self.action_range * torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        return action

    def sample_action(self):

        return self.action_space.sample()
