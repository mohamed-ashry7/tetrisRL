from dqn_model import DQN

from tetris_engine import TetrisEngine
import collections
from collections import namedtuple
import argparse
import numpy as np 
import torch
import torch.nn as nn
import time

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 10000
MEAN_REWARD_BOUND=10
EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

Experience=namedtuple('Experience',field_names=['state','action','reward','done','next_state'])

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
        return np.array(states), np.array(actions), 
        np.array(rewards, dtype=np.float32), 
        np.array(dones, dtype=np.uint8), 
        np.array(next_states)  
        
        
        
class Agent():
    
    
    def __init__(self,env,replay_buffer):
        self.env=env
        self.replay_buffer=replay_buffer
        self._reset()
    
    def _reset(self):
        self.state=self.env.clear() 
        self.total_reward=0.0
        
    
    
    @torch.no_grad()
    def play_step(self,net,eps=0.0,device='cpu'):
        done_reward=None
        
        if np.random.random()<eps:
            action = self.env.random_action()
        else :
            state_a=np.array([self.state],copy=False)
            state_v=torch.tensor(state_a).to(device)
            q_vals_v=net(state_v)
            _,action_v=torch.max(q_vals_v,dim=1)
            action=int(action_v.item())
        
        next_state,reward,done = self.env.step(action)
        self.total_reward+=reward
        exp=Experience(state=self.state,action=action,reward=reward,done=done,next_state=next_state)
        self.replay_buffer.append(exp)
        
        if done:
            done_reward=self.total_reward
            self._reset()
        
        return done_reward
    
    def calc_loss(self,batch,net,target_net,device='cpu'):
        
        states,actions,rewards,dones,next_states=batch
        
        states_v = torch.tensor(np.array(states, copy=False)).to(device)
        next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)
                
        Q_values=net(state_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_Q_values=target_net(next_states_v).max(1)[0]
        next_Q_values[done_mask]=0.0        
        next_Q_values=next_Q_values.detach()
        expected_Q_value=rewards_v+GAMMA*next_Q_values
        
        return nn.MSELoss()(Q_values,expected_Q_value)
        
    
    
if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--cuda",default=False,help="Enable cuda",action="store_true")
    parser.add_argument("--width",default=10,help="Board Width",action="store")
    parser.add_argument("--height",default=20,help="Board Height",action="store")

    args=parser.parse_args()

    device=torch.device("cuda" if args.cuda else "cpu")
    width = args.width
    height=args.height

    env=TetrisEngine(width,height)

    env_shape=torch.Size([1,width,height])
    number_actions=env.number_actions()
    
    net=DQN(env_shape,number_actions).to(device)
    target_net=DQN(env_shape,number_actions).to(device)
    replay_buffer=ReplayBuffer(REPLAY_SIZE)

    print(net)
    agent=Agent(env,replay_buffer)

    epsilon= EPSILON_START
    optimizer=torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, reward %.3f, "
            "eps %.2f, speed %.2f f/s" % (frame_idx, len(total_rewards), m_reward, epsilon,speed))
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), "tetrisJay-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                    best_m_reward, m_reward))
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(replay_buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
        optimizer.zero_grad()
        batch = replay_buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()