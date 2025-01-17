
import collections
import numpy as np 

class ReplayBuffer:
    def __init__(self,size):
        self.buffer=collections.deque(maxlen=size)
    
    def __len__(self):
        return len(self.buffer)
    
    
    
    def append(self,exp):
        self.buffer.append(exp)
        
    def sample(self,batch_size):
        indices = np.random.choice(len(self.buffer),batch_size,replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)  
        