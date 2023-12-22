import random
import numpy as np 

class ReplayBuffer:
    def __init__(self,capacity) -> None:
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = int((self.pos + 1) % self.capacity)  # as a ring buffer
    
    def sample(self,batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)