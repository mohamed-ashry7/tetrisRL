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
BATCH_SIZE = 64
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 10000
MEAN_REWARD_BOUND=100
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
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)  
        
        
        
class Agent():
    
    
    def __init__(self,env,replay_buffer):
        self.env=env
        self.replay_buffer=replay_buffer
        self._reset()
    
    def _reset(self):
        self.state=self.env.clear() 
        self.total_reward=0.0
        
    
    
    @torch.no_grad()
    def play_step(self,net,eps=0.0,device='cpu',mode='train'):
        done_reward=None
        FloatTensor= torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor

        
        if np.random.random()<eps:
            action = self.env.random_action()
        else :
            state_a=np.array([self.state],copy=False)
            state_v=torch.tensor(state_a).unsqueeze(0).type(FloatTensor).to(device)
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
        if mode=='play':
            print(f'Action taken = {action}, Reward={reward}')
            if done:
                print(f'Total Reward= {done_reward}')
        return done_reward
    
    def calc_loss(self,batch,net,target_net,device='cpu'):
        FloatTensor= torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor

        states,actions,rewards,dones,next_states=batch
        
        states_v = torch.tensor(np.array(states, copy=False)).type(FloatTensor).to(device)
        next_states_v = torch.tensor(np.array(next_states, copy=False)).type(FloatTensor).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)
        # Adding unsqueeze to adjust the size of the input of the network to be [BatchSize, channels, W,H]       
        Q_values=net(states_v.unsqueeze(1)).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_Q_values=target_net(next_states_v.unsqueeze(1)).max(1)[0]
        next_Q_values[done_mask]=0.0        
        next_Q_values=next_Q_values.detach()
        expected_Q_values=rewards_v+GAMMA*next_Q_values
        
        return nn.MSELoss()(Q_values,expected_Q_values)
        
    
    
if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--cuda",default=False,help="Enable cuda",action="store_true")
    parser.add_argument("--cuda-version",default=0,help="Determining cuda version",action="store")
    parser.add_argument("--width",default=10,help="Board Width",action="store")
    parser.add_argument("--height",default=20,help="Board Height",action="store")
    parser.add_argument("--mode",default='train',help="Define the mode of the model play or train",action="store")
    parser.add_argument("--model",default=None,help="Model file to load",action='store')
    
    args=parser.parse_args()
    
    device=torch.device(f"cuda:{args.cuda_version}" if args.cuda else "cpu")
    env=TetrisEngine(args.width,args.height)
    
    net=DQN(env.env_shape(),env.number_actions()).to(device)
    target_net=DQN(env.env_shape(),env.number_actions()).to(device)
    replay_buffer=ReplayBuffer(REPLAY_SIZE)
    print(net)
    agent=Agent(env,replay_buffer)
    
#     x= torch.zeros([1,1,10,20]).to(device)
#     vv=net(x)


    if args.mode =='train':
        epsilon = EPSILON_START
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
                    torch.save(net.state_dict(), "./dqn_models_stats/tetris_cnn_Della-best_%.0f.dat" % m_reward)
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
                target_net.load_state_dict(net.state_dict())
            optimizer.zero_grad()
            batch = replay_buffer.sample(BATCH_SIZE)
            
            loss_t = agent.calc_loss(batch, net, target_net, device=device)
            loss_t.backward()
            optimizer.step()
    else:
        
        if args.model==None:
            parser.error("--mode play requires --model")
        
        state = torch.load(args.model, map_location=lambda stg,_: stg)
        net.load_state_dict(state)
        while agent.play_step(net,mode='play')==None:
            print(env)
            
            