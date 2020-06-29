
import collections
from collections import namedtuple
import argparse
import numpy as np 
import torch
import torch.nn as nn
from utils.drawing_utils import *
from controllers import best_action





Experience=namedtuple('Experience',field_names=['state','action','reward','done','next_state'])
BETA=0.9
GAMMA=0.99
class Agent():

    
    
  def __init__(self,env,replay_buffer):
    self.env=env
    self.replay_buffer=replay_buffer
    self._reset()
    self.actions=[0]*40

  def _reset(self):

    self.state=self.env.clear() 
    self.total_reward=0.0
    
  
  
  @torch.no_grad()
  def play_step(self,net,eps=0.0,device='cpu',mode='train'):
        done_reward=None
        tetrominos=0
        cleared_lines=0

        net.eval()

        FloatTensor= torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor
        # eps_rand=np.random.random()
        eps_agent=np.random.random() 

        # if eps_rand<eps and eps_agent<eps:
        #     if eps_agent>eps_rand:
        #         action=best_action(self.env,'near')
        #     else:
        #         action = self.env.random_action()
        # elif eps_rand<eps:
        #     action = self.env.random_action()
        if eps_agent<eps:
            action=best_action(self.env,'dellacherie')
        else :
            state_v= torch.tensor(np.array([self.state],copy=False)).type(FloatTensor).to(device)
            Q_values=net(state_v)
            max_Q_value,action_index=torch.max(Q_values,dim=1)
            # max_Q_value = max_Q_value.item()
            # Q_values=Q_values[0].cpu().data.numpy() -max_Q_value
            # boltzmann_dist = np.exp(BETA * Q_values)/np.sum(np.exp(BETA * Q_values))
            # action = np.random.choice(a=self.env.number_actions(),p=boltzmann_dist)
            action=int(action_index.item())
            self.actions[action]+=1

        next_state,reward,done = self.env.step(action)
        self.total_reward+=reward
        exp=Experience(state=self.state,action=action,reward=reward,done=done,next_state=next_state)
        self.replay_buffer.append(exp)
        if done:
            done_reward=self.total_reward
            tetrominos=self.env.tetrominos
            cleared_lines=self.env.cleared_lines
            self._reset()
        if mode=='play':
            print(f'Action taken = {action}, Reward={reward}')
            if done:
                print(f'Total Reward= {done_reward}')
            return done_reward
        return done_reward,cleared_lines,tetrominos
  


      
        
        
   

