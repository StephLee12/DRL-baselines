import random
import numpy as np 
import math 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from collections import deque 

class ReplayBuffer:
    def __init__(self,capacity) -> None:
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, obs, action, reward, next_obs, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (obs, action, reward, next_obs, done)
        self.pos = int((self.pos + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = map(np.stack, zip(*batch)) # stack for each element

        return obs, action, reward, next_obs, done

    def __len__(self):
        return len(self.buffer)
    

class MultiStepReplayBuffer(ReplayBuffer):
    def __init__(
        self, 
        capacity,
        n_step=3
    ) -> None:
        super().__init__(capacity)

        self.n_step = n_step 
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, obs, action, reward, next_obs, done):
        transition = [obs, action, reward, next_obs, done]
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) == self.n_step: # n-step ready
            # when n-step is ready 
            reward, next_obs, done = self._get_n_step_info() # get discounted reward 
            obs, action = self.n_step_buffer[0][:2] # get the first step info during the n-step 

            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.pos] = (obs, action, reward, next_obs, done)
            self.pos = int((self.pos + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        return super().sample(batch_size)

    def _get_n_step_info(self, gamma=0.99):
        reward, next_obs, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]): # track back 
            prev_reward, prev_next_obs, prev_done = transition[-3:]

            reward = prev_reward + gamma * (1-prev_done) * reward 
            next_obs, done = (prev_next_obs, prev_done) if done else (next_obs, done)

        return reward, next_obs, done

    def __len__(self):
        return super().__len__()



class PER(ReplayBuffer):
    def __init__(
        self, 
        capacity,
        alpha=0.6 # priority level
    ) -> None:
        super().__init__(capacity)

        self.max_priority = 1.0
        self.tree_pos = 0
        self.alpha = alpha 

        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity) # for fast retreival 
        self.min_tree = MinSegmentTree(tree_capacity)


    def push(self, obs, action, reward, next_obs, done):
        super().push(obs, action, reward, next_obs, done)

        self.sum_tree[self.tree_pos] = self.max_priority ** self.alpha
        self.min_tree[self.tree_pos] = self.max_priority ** self.alpha
        self.tree_pos = (self.tree_pos + 1) % self.capacity

    
    def sample(self, batch_size, beta=0.4): # beta is related to the importance sampling weight , annealing from start to 1.0
        indices = self._sample_proportional(batch_size=batch_size)

        obs = np.array([self.buffer[idx][0] for idx in indices])
        action = np.array([self.buffer[idx][1] for idx in indices])
        reward = np.array([self.buffer[idx][2] for idx in indices])
        next_obs = np.array([self.buffer[idx][3] for idx in indices])
        done = np.array([self.buffer[idx][4] for idx in indices])
        weights = np.array([self._calculate_weight(idx=idx, beta=beta) for idx in indices])

        return obs, action, reward, next_obs, done, weights, np.array(indices)

    
    def update_priorities(self, indices, priorities): # priority is based on TD-error
        for idx, priority in zip(indices, priorities):
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha 

            self.max_priority = max(self.max_priority, priority)


    def _sample_proportional(self, batch_size): # sample based on priorities
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices


    def _calculate_weight(self, idx, beta): # calculate IS weight 
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

    def __len__(self):
        return super().__len__()


class MultiStepPER(MultiStepReplayBuffer):
    def __init__(
        self, 
        capacity, 
        n_step=3,
        alpha=0.6
    ) -> None:
        super().__init__(capacity, n_step)

        self.max_priority = 1.0
        self.tree_pos = 0
        self.alpha = alpha 

        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity) # for fast retreival 
        self.min_tree = MinSegmentTree(tree_capacity)


    def push(self, obs, action, reward, next_obs, done):
        super().push(obs, action, reward, next_obs, done)

        self.sum_tree[self.tree_pos] = self.max_priority ** self.alpha
        self.min_tree[self.tree_pos] = self.max_priority ** self.alpha
        self.tree_pos = (self.tree_pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        indices = self._sample_proportional(batch_size=batch_size)

        obs = np.array([self.buffer[idx][0] for idx in indices])
        action = np.array([self.buffer[idx][1] for idx in indices])
        reward = np.array([self.buffer[idx][2] for idx in indices])
        next_obs = np.array([self.buffer[idx][3] for idx in indices])
        done = np.array([self.buffer[idx][4] for idx in indices])
        weights = np.array([self._calculate_weight(idx=idx, beta=beta) for idx in indices])

        return obs, action, reward, next_obs, done, weights, np.array(indices) 

    
    def update_priorities(self, indices, priorities): # priority is based on TD-error
        for idx, priority in zip(indices, priorities):
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha 

            self.max_priority = max(self.max_priority, priority)


    def _sample_proportional(self, batch_size): # sample based on priorities
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices


    def _calculate_weight(self, idx, beta): # calculate IS weight 
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

    def __len__(self):
        return super().__len__()


class LagReplayBuffer(ReplayBuffer):
    def __init__(self, capacity) -> None:
        super().__init__(capacity)
    
    def push(self, obs, action, reward, cost, next_obs, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (obs, action, reward, cost, next_obs, done)
        self.pos = int((self.pos + 1) % self.capacity)  # as a ring buffer


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, cost, next_obs, done = map(np.stack, zip(*batch)) # stack for each element

        return obs, action, reward, cost, next_obs, done
    
    
    def __len__(self):
        return super().__len__()

# change nn.Linear to NoisyLinear 
class NoisyLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        std_init=0.5
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()


    def forward(self, x):
        return F.linear(
            input=x, 
            weight=self.weight_mu+self.weight_sigma*self.weight_epsilon, 
            bias=self.bias_mu+self.bias_sigma*self.bias_epsilon
        )

    def reset_noise(self):
        epsilon_in = self._scale_size(self.in_features)
        epsilon_out = self._scale_size(self.out_features) # generate two vectors of noise, their product can generate row*col noises. It is more efficient

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )


    def _scale_size(self, size):
        x = torch.rand(size)

        return x.sign().mul(x.abs().sqrt())




########## Segment Tree ################### 



# -*- coding: utf-8 -*-
"""Segment tree for Prioritized Replay Buffer."""

import operator
from typing import Callable


class SegmentTree:
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)