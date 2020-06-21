from dqn_model import DQN
from tetris_engine import TetrisEngine
import collections
from collections import namedtuple
import argparse
import numpy as np 
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt 
from utils.drawing_utils import *

GAMMA = 0.99
BATCH_SIZE = 360
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-5
SYNC_TARGET_FRAMES = 10000
MEAN_REWARD_BOUND=100
EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
DRAWING_RATE=25000
Experience=namedtuple('Experience',field_names=['state','action','reward','done','next_state'])

        
        
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
        tetrominos=0
        cleared_lines=0
        FloatTensor= torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor

        net.eval()
        if np.random.random()<eps:
            action = self.env.random_action()
        else :
            state_a=np.array([self.state],copy=False)
            state_v=torch.tensor(state_a).type(FloatTensor).to(device)
            q_vals_v=net(state_v)
            _,action_v=torch.max(q_vals_v,dim=1)
            action=int(action_v.item())

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
    
    def calc_loss(self,batch,net,target_net,device='cpu'):
        net.train()
        FloatTensor= torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor

        states,actions,rewards,dones,next_states=batch
        
        states_v = torch.tensor(np.array(states, copy=False)).type(FloatTensor).to(device)
        next_states_v = torch.tensor(np.array(next_states, copy=False)).type(FloatTensor).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)
        Q_values=net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_Q_values=target_net(next_states_v).max(1)[0]
            next_Q_values[done_mask]=0.0        
        expected_Q_values=rewards_v+GAMMA*next_Q_values
        
        return nn.SmoothL1Loss()(Q_values,expected_Q_values)
        
        
        
   



    


if __name__=="__main__":
        
    parser=argparse.ArgumentParser()
    parser.add_argument("--cuda",default=False,help="Enable cuda",action="store_true")
    parser.add_argument("--width",default=10,help="Board Width",action="store")
    parser.add_argument("--height",default=20,help="Board Height",action="store")
    parser.add_argument("--mode",default='train',help="Define the mode of the model play or train",action="store")
    parser.add_argument("--model",default=None,help="Model file to load",action='store')
    parser.add_argument("--model-dir",default=None,help="dir to save models at it",action='store')
    parser.add_argument("--play-times",default=None,help="Testing the model how many times",action='store')


    args=parser.parse_args()
    
    device=torch.device("cuda" if args.cuda else "cpu")
    env=TetrisEngine(args.width,args.height)

    net=DQN(env.state_shape(),env.number_actions()).to(device)
    target_net=DQN(env.state_shape(),env.number_actions()).to(device)
    replay_buffer=ReplayBuffer(REPLAY_SIZE)
    print(net)
    agent=Agent(env,replay_buffer)
    

    model_path="./dqn_models_stats/" if args.model_dir==None else args.model_dir
    
    if args.mode =='train':
        epsilon = EPSILON_START
        optimizer=torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)
        total_rewards = []
        mean_rewards=[]
        mean_q_losses=[]
        q_losses=[]
        cleared_lines =[]
        mean_cleared_lines=[]
        tetrominos =[]
        mean_tetrominos=[]
        frame_idx = 0
        ts_frame = 0
        ts = time.time()
        best_m_reward = None

        while True:
            frame_idx += 1
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
            reward,c,t = agent.play_step(net, epsilon, device=device)
            
            if reward is not None:
                cleared_lines.append(c)
                mean_cleared_lines.append(np.mean(cleared_lines[-100:]))

                tetrominos.append(t)
                mean_tetrominos.append(np.mean(tetrominos[-100:]))

                total_rewards.append(reward)
                m_reward = np.mean(total_rewards[-100:])
                mean_rewards.append(m_reward)

                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                
                print("%d: done %d games, reward %.3f, "
                "eps %.2f, speed %.2f f/s" % (frame_idx, len(total_rewards), m_reward, epsilon,speed))
                if best_m_reward is None or best_m_reward < m_reward:
                    torch.save(net.state_dict(), f"{model_path}tetris_best_%.0f.dat" % m_reward)
                    if best_m_reward is not None:
                        print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                    best_m_reward = m_reward
                

            if len(replay_buffer) < REPLAY_START_SIZE:
                continue
            if frame_idx % SYNC_TARGET_FRAMES == 0:
                target_net.load_state_dict(net.state_dict())
            
            optimizer.zero_grad()
            batch = replay_buffer.sample(BATCH_SIZE)
            loss_t = agent.calc_loss(batch, net, target_net, device=device)
            q_losses.append(loss_t.item())
            mean_q_losses.append(np.mean(q_losses[-100:]))
            loss_t.backward()
            optimizer.step()
            
            if frame_idx %DRAWING_RATE ==0:
                games_number=len(total_rewards)
                plot_data(model_path+f"Rewards_of_{games_number}_games.png",mean_rewards,'Rewards',games_number)
                plot_data(model_path+f"Losses_of_{games_number}_games.png",mean_q_losses,'Q losses',games_number)
                plot_data(model_path+f"Cleared_lines_of_{games_number}_games.png",mean_cleared_lines,'Cleared Lines',games_number)
                plot_data(model_path+f"Tetrominos_of_{games_number}_games.png",mean_tetrominos,'Tetrominos',games_number)
                torch.save(net.state_dict(), f"{model_path}tetris_best_%d.dat" % games_number)
    else:
        
        if args.model==None:
            parser.error("--mode play requires --model")
        
        state = torch.load(args.model, map_location=lambda stg,_: stg)
        net.load_state_dict(state)
        cleared_lines =0
        for i in range (args.play_time):
            while agent.play_step(net,mode='play')==None:
                cleared_lines+=env.cleared_lines
                print(env)
        print(cleared_lines)
            