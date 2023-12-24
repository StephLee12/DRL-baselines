import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from torch.distributions import Categorical,Normal

# ---------  Discrete Action Space -----------

# ******** Single Action **********
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
class QDiscretePMultiAction(nn.Module):
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

        return action,log_probs
    

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

    def get_action(self,obs,noise_scale,deterministic):
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
    

    def evaluate(self,obs,noise_scale):
        mean,log_std = self.forward(obs)
        std = log_std.exp()


        normal = Normal(0,1)
        z = normal.sample(mean.shape).to(self.device)
        action = torch.tanh(mean+std*z)

        noise = noise_scale * Normal(0,1).sample(action.shape).to(self.device)
        action = action + noise 
        action = action.clamp(-1.,1.)

        return action
    
    def get_action(self,obs,noise_scale,deterministic=False):
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
    
    def evaluate(self)


class DDPG_GaussianContinuousPolicyMultiActionMultiOutLayer(GaussianContinuousPolicyMultiActionMultiOutLayer):
    def __init__(self, device, obs_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2) -> None:
        super().__init__(device, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max)





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





# Deep Deterministic Policy Gradient (DDPG)
class DDPG_QNetwork(BaseQNetwork):
    def __init__(self, obs_dim, action_dim, hidden_size, init_w=0.003) -> None:
        super().__init__(obs_dim, action_dim, hidden_size, init_w)

class DDPG_PolicyNetwork(BasePolicyNetwork):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_size,
        action_range=1.,
        init_w=3e-3
    ) -> None:
        super().__init__()
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
    
    def evaluate(self,obs,noise_scale):
        action = self.forward(obs)
        
        normal = Normal(0,1)
        noise = normal.sample(action.shape) * noise_scale

        action = self.action_range * action + noise.to(action.device)
        action = torch.clamp(action,-self.action_range,self.action_range)

        return action 
    
    def get_action(self,obs,deterministic,noise_scale,device):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

        mean = self.forward(obs)
        
        normal = Normal(0,1)
        noise = normal.sample(mean.shape) * noise_scale

        action = mean + noise.to(device)
        action = action * self.action_range
        action = torch.clamp(action,-self.action_range,self.action_range)

        action = self.action_range*mean if deterministic else action 
        
        return action.detach().cpu().numpy().flatten()


class AttentiveConvDDPG(BaseAttentiveConvNetwork):
    def __init__(
        self, 
        feature_size, 
        embedding_size, 
        nhead, 
        kernel_size_lst, 
        out_channels, 
        action_dim,
        qnet,
        target_qnet,
        action_range=1.
    ) -> None:
        super(AttentiveConvDDPG,self).__init__(
            feature_size, 
            embedding_size,
            nhead, 
            kernel_size_lst, 
            out_channels, 
            action_dim, 
        )
        self.qnet = qnet
        self.target_qnet = target_qnet

        self.action_range = action_range
    
    def forward_actor(self,obs):
        x = self.forward_share(obs)
        x = torch.tanh(self.actor_fc_mean(x))

        return x

    def forward_critic(self,obs,action,qnet_type):
        x = self.forward_share(obs)
        x = torch.cat([x,action],dim=1)
        if qnet_type == 'qnet':
            q_val = self.qnet(x)
        else:
            q_val = self.target_qnet(x)

        return q_val

    def evaluate(self,obs,noise_scale):
        action = self.forward_actor(obs)
        
        normal = Normal(0,1)
        noise = normal.sample(action.shape) * noise_scale

        action = action + noise.to(action.device)
        action = action * self.action_range
        action = torch.clamp(action,-self.action_range,self.action_range)

        return action 

    def get_action(self,obs,deterministic,noise_scale,device):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

        mean = self.forward_actor(obs)

        normal = Normal(0,1)
        noise =  normal.sample(mean.shape) * noise_scale
        action = mean + noise.to(device)
        action = action * self.action_range
        action = torch.clamp(action,-self.action_range,self.action_range)

        action = self.action_range*mean if deterministic else action 

        return action.detach().cpu().numpy().flatten()
    


# Twin Delayed Deep Deterministic Policy Gradients (TD3)
class TD3_QNetwork(BaseQNetwork):
    def __init__(self, obs_dim, action_dim, hidden_size, init_w=0.003) -> None:
        super().__init__(obs_dim, action_dim, hidden_size, init_w)


class TD3_PolicyNetwork(BasePolicyNetwork):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_size,
        action_range=1.,
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2
    ) -> None:
        super().__init__()
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

    def evaluate(self,obs,eval_noise_scale,epsilon=1e-6):
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = Normal(0, 1)
        z      = normal.sample(mean.shape) 
        action = torch.tanh(mean + std*z.to(mean.device)) # TanhNormal distribution as actions; reparameterization trick
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(mean.device)) - torch.log(1. - action.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        ''' add noise '''
        eval_noise_clip = 2*eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = torch.clamp(noise,-eval_noise_clip, eval_noise_clip)
        action = action + noise.to(action.device)
        action = action * self.action_range
        action = torch.clamp(action,-self.action_range,self.action_range)

        return action, log_prob, z, mean, log_std

    def get_action(self,obs, deterministic, explore_noise_scale, device):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = Normal(0, 1)
        z      = normal.sample(mean.shape)

        action = torch.tanh(mean + std*z.to(device))
        noise = normal.sample(action.shape) * explore_noise_scale
        action = action + noise.to(device)
        action = action * self.action_range
        action = torch.clamp(action,-self.action_range,self.action_range)

        action = self.action_range*mean if deterministic else action 

        return action.detach().cpu().numpy().flatten()
    


class AttentiveConvTD3(BaseAttentiveConvNetwork):
    def __init__(
        self, 
        feature_size, 
        embedding_size, 
        nhead, 
        kernel_size_lst, 
        out_channels, 
        action_dim, 
        qnet1,
        qnet2,
        target_qnet1,
        target_qnet2,
        action_range=1.,
        log_std_min=-20,
        log_std_max=2
    ) -> None:
        super(AttentiveConvTD3,self).__init__(
            feature_size, 
            embedding_size,
            nhead, 
            kernel_size_lst, 
            out_channels, 
            action_dim, 
        )
        self.qnet1 = qnet1
        self.qnet2 = qnet2
        self.target_qnet1 = target_qnet1
        self.target_qnet2 = target_qnet2

        self.actor_fc_log_std = nn.Linear(in_features=len(kernel_size_lst)*out_channels,out_features=action_dim)
        self.actor_fc_log_std.weight.data.uniform_(-3e-3,3e-3)
        self.actor_fc_log_std.bias.data.uniform_(-3e-3,3e-3)

        self.action_range = action_range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    
    def forward_actor(self,obs):
        x = self.forward_share(obs)
        mean = self.actor_fc_mean(x)
        log_std = self.actor_fc_log_std(x)
        log_std = torch.clamp(log_std,self.log_std_min,self.log_std_max)

        return mean,log_std 

    def forward_critic(self,obs,action,qnet_type):
        x = self.forward_share(obs)
        x = torch.cat([x,action],dim=1)
        if qnet_type == 'qnet1':
            q_val = self.qnet1(x)
        elif qnet_type == 'qnet2':
            q_val = self.qnet2(x)
        elif qnet_type == 'target_qnet1':
            q_val = self.target_qnet1(x)
        else:
            q_val = self.target_qnet2(x)

        return q_val
    

    def evaluate(self,obs,eval_noise_scale,epsilon=1e-6):
        mean, log_std = self.forward_actor(obs)
        std = log_std.exp()

        normal = Normal(0, 1)
        z      = normal.sample(mean.shape) 
        action = torch.tanh(mean + std*z.to(mean.device)) # TanhNormal distribution as actions; reparameterization trick
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(mean.device)) - torch.log(1. - action.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        ''' add noise '''
        eval_noise_clip = 2*eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = torch.clamp(noise,-eval_noise_clip, eval_noise_clip)
        action = action + noise.to(action.device)
        action = action * self.action_range
        action = torch.clamp(action,-self.action_range,self.action_range)

        return action, log_prob, z, mean, log_std


    def get_action(self,obs, deterministic, explore_noise_scale, device):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

        mean, log_std = self.forward_actor(obs)
        std = log_std.exp()

        normal = Normal(0, 1)
        z      = normal.sample(mean.shape)

        action = torch.tanh(mean + std*z.to(device))
        noise = normal.sample(action.shape) * explore_noise_scale
        action = action + noise.to(device)
        action = action * self.action_range
        action = torch.clamp(action,-self.action_range,self.action_range)

        action = self.action_range*mean if deterministic else action 

        return action.detach().cpu().numpy().flatten()



# Soft Actor Critic (SAC)
class SoftQNetwork(BaseQNetwork):
    def __init__(self, obs_dim, action_dim, hidden_size, init_w=0.003) -> None:
        super().__init__(obs_dim, action_dim, hidden_size, init_w)

class SoftPolicyNetwork(BasePolicyNetwork):
    def __init__(
        self,
        obs_dim, 
        action_dim, 
        hidden_size, 
        action_range=1., 
        init_w=3e-3, 
        log_std_min=-20, 
        log_std_max=2
    ) -> None:
        super().__init__()
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
    
    def evaluate(self,obs,epsilon=1e-6):
        '''
        generate sampled action with obs as input wrt the policy network;
        '''
        mean, log_std = self.forward(obs)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape) 
        action_0 = torch.tanh(mean + std*z.to(mean.device)) # TanhNormal distribution as actions; reparameterization trick
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
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(mean.device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std
    
    def get_action(self,obs,deterministic,device):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action = self.action_range * torch.tanh(mean) if deterministic else self.action_range * torch.tanh(mean + std*z.to(device)) 

        return action.detach().cpu().numpy().flatten()



class AttentiveConvSAC(BaseAttentiveConvNetwork):
    def __init__(
        self, 
        feature_size, 
        embedding_size, 
        nhead, 
        kernel_size_lst, 
        out_channels, 
        action_dim,
        qnet1,
        qnet2,
        target_qnet1,
        target_qnet2,
        action_range=1.,
        log_std_min=-20,
        log_std_max=2
    ) -> None:
        super(AttentiveConvSAC,self).__init__(
            feature_size=feature_size, 
            embedding_size=embedding_size, 
            nhead=nhead, 
            kernel_size_lst=kernel_size_lst, 
            out_channels=out_channels, 
            action_dim=action_dim
        )
        self.qnet1 = qnet1
        self.qnet2 = qnet2
        self.target_qnet1 = target_qnet1
        self.target_qnet2 = target_qnet2


        self.actor_fc_log_std = nn.Linear(in_features=len(kernel_size_lst)*out_channels,out_features=action_dim)
        self.actor_fc_log_std.weight.data.uniform_(-3e-3,3e-3)
        self.actor_fc_log_std.bias.data.uniform_(-3e-3,3e-3)

        self.action_range = action_range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
    
    
    def forward_actor(self,obs):
        x = self.forward_share(obs)
        mean = self.actor_fc_mean(x)
        log_std = self.actor_fc_log_std(x)
        log_std = torch.clamp(log_std,self.log_std_min,self.log_std_max)

        return mean,log_std 

    def forward_critic(self,obs,action,qnet_type):
        x = self.forward_share(obs)
        x = torch.cat([x,action],dim=1)
        if qnet_type == 'qnet1':
            q_val = self.qnet1(x)
        elif qnet_type == 'qnet2':
            q_val = self.qnet2(x)
        elif qnet_type == 'target_qnet1':
            q_val = self.target_qnet1(x)
        else:
            q_val = self.target_qnet2(x)

        return q_val
    
    def evaluate(self,obs,epsilon=1e-6):
        mean, log_std = self.forward_actor(obs)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape) 
        action_reparam = torch.tanh(mean + std*z.to(mean.device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_reparam
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
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(mean.device)) - torch.log(1. - action_reparam.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self,obs,deterministic,device):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device) # shape is 1*feature_size

        mean, log_std = self.forward_actor(obs)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        # deterministic return mean 
        action = self.action_range * torch.tanh(mean) if deterministic else self.action_range * torch.tanh(mean + std*z.to(device)) 

        return action.detach().cpu().numpy().flatten()












    