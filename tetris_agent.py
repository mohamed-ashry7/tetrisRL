
import collections
from collections import namedtuple
import argparse
import numpy as np 
import torch
import torch.nn as nn
from utils.drawing_utils import *
from controllers import best_action





Experience=namedtuple('Experience',field_names=['state','action','reward','done','next_state'])
GAMMA=0.99
class Agent():

    
    
  def __init__(self,envs,replay_buffer):
    self.envs=envs
    self.replay_buffer=replay_buffer
    for env in envs:
      env.clear()
    self.actions=[0]*40
    self.index=0
  
  def _set_env_state(self):

    self.env =self.envs[self.index]
    self.state=self.env.calc_state()
    self.index=(self.index+1)%len(self.envs)

  @torch.no_grad()
  def play_step(self,net,eps=0.0,device='cpu',mode='train'):
        net.eval()
        self._set_env_state()
   
        FloatTensor= torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor
        eps_agent=np.random.random() 

        if eps_agent<eps:
            action=best_action(self.env,'el-tetris')
        else :
            state_v= torch.tensor(np.array([self.state],copy=False)).type(FloatTensor).to(device)
            Q_values=net(state_v)
            max_Q_value,action_index=torch.max(Q_values,dim=1)
            action=int(action_index.item())
            self.actions[action]+=1

        next_state,reward,done = self.env.step(action)
        exp=Experience(state=self.state,action=action,reward=reward,done=done,next_state=next_state)
        self.replay_buffer.append(exp)
        if done:
            done_reward=self.env.total_reward
            tetrominos=self.env.tetrominos
            cleared_lines=self.env.cleared_lines
            self.env.clear()
            return done_reward,cleared_lines,tetrominos
        else:
          return None,None,None
  


      
        
        
   

