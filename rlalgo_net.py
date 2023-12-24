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