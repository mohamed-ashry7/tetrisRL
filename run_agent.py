
from models.dqn_model import DQN
from tetris_engine import TetrisEngine
import collections
from tetris_agent import Agent
from replay_buffer import ReplayBuffer
from utils.drawing_utils import *
import numpy as np 
import torch
import torch.nn as nn
import time
import argparse


GAMMA=0.99
BATCH_SIZE = 320
REPLAY_SIZE = 20000
REPLAY_START_SIZE = 20000
LEARNING_RATE = 1e-5
SYNC_TARGET_FRAMES = 5000
EPSILON_DECAY_LAST_FRAME = 50000
EPSILON_START = 0.0
EPSILON_FINAL = 0.01
DRAWING_RATE=25000
N_ENVS=2

def unpack_batch(batch,device):

  FloatTensor= torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor
  states,actions,rewards,dones,next_states=batch
  states_v = torch.tensor(np.array(states, copy=False)).type(FloatTensor).to(device)
  next_states_v = torch.tensor(np.array(next_states, copy=False)).type(FloatTensor).to(device)
  actions_v = torch.tensor(actions).to(device)
  rewards_v = torch.tensor(rewards).to(device)
  done_mask = torch.BoolTensor(dones).to(device)
  return states_v,next_states_v,actions_v,rewards_v,done_mask

def calc_loss(batch,net,target_net,device,ddqn):
    net.train()
    target_net.eval()
    states_v,next_states_v,actions_v,rewards_v,done_mask= unpack_batch(batch,device)

    Q_values=net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    with torch.no_grad():
        if ddqn:
            next_state_actions = net(next_states_v).max(1)[1].unsqueeze(-1)
            next_Q_values=target_net(next_states_v).gather(1, next_state_actions).squeeze(-1)
        else:
            next_Q_values=target_net(next_states_v).max(1)[0]
        next_Q_values[done_mask]=-100.0       
    expected_Q_values=rewards_v+GAMMA*next_Q_values.detach()
    
    return nn.SmoothL1Loss()(Q_values,expected_Q_values)    
   
def optimize_network(optimizer,batch,net,target_net,device='cpu',ddqn=True): 
  optimizer.zero_grad()
  loss_fn =calc_loss(batch, net, target_net, device,ddqn)
  loss_fn.backward()
  optimizer.step()
  return loss_fn.item()



   


if __name__=="__main__":




  parser=argparse.ArgumentParser()
  parser.add_argument("--cuda",default=False,help="Enable cuda",action="store_true")
  parser.add_argument("--width",default=10,help="Board Width",action="store")
  parser.add_argument("--height",default=20,help="Board Height",action="store")
  parser.add_argument("--mode",default='train',help="Define the mode of the model play or train",action="store")
  parser.add_argument("--model",default=None,help="Model file to load",action='store')
  parser.add_argument("--model-dir",default="./dqn_models_stats/",help="Directory to save models at it",action='store')
  parser.add_argument("--play-times",default=None,help="Testing the model how many times",action='store')
  
  args=parser.parse_args()
    
  device=torch.device("cuda" if args.cuda else "cpu")
  env=TetrisEngine(args.width,args.height)  


  envs =[] 
  for _ in range(N_ENVS):
    env=TetrisEngine(args.width,args.height)  
    envs.append(env)

  net=DQN(env.state_shape(),env.number_actions()).to(device)
  target_net=DQN(env.state_shape(),env.number_actions()).to(device)
  
  replay_buffer=ReplayBuffer(REPLAY_SIZE)
  print(net)
  agent=Agent(envs,replay_buffer)


  model_path=args.model_dir
  mode = args.mode
  model = 
  if mode =='train':

    if args.model != None:
        state = torch.load(model_path+args.model, map_location=lambda stg,_: stg)# to train from a previous data.
        net.load_state_dict(state)
    epsilon = EPSILON_START
    optimizer=torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)

    total_rewards, mean_rewards = [], []
    q_losses, mean_q_losses=[], []
    cleared_lines, mean_cleared_lines=[], []
    tetrominos, mean_tetrominos =[], []
    frame_idx, ts_frame = 0, 0
    
    ts = time.time()

    while True:
        frame_idx += 1
        for _ in range(N_ENVS):
          epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
          r,c,t = agent.play_step(net, epsilon, device=device)
          
          if r is not None:
              cleared_lines.append(c)
              mean_cleared_lines.append(np.mean(cleared_lines[-100:]))

              tetrominos.append(t)
              mean_tetrominos.append(np.mean(tetrominos[-100:]))

              total_rewards.append(r)
              m_reward = np.mean(total_rewards[-100:])
              mean_rewards.append(m_reward)

              speed = (frame_idx - ts_frame) / (time.time() - ts)
              ts_frame = frame_idx
              ts = time.time()
              ql=0.0 if len(mean_q_losses)==0 else mean_q_losses[-1]
              if len(total_rewards)%100==0:
                print("%d: done %d games, reward %.3f, "
                "eps %.2f, lines %.3f, Q-losses %.3f, speed %.2f f/s" % (frame_idx, len(total_rewards), m_reward, epsilon, mean_cleared_lines[-1],ql,speed))
            

        if len(replay_buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())
        

        batch =replay_buffer.sample(BATCH_SIZE)
        loss=optimize_network(optimizer,batch ,net,target_net,device )
        q_losses.append(loss)
        mean_q_losses.append(np.mean(q_losses[-100:]))
        
        
        if frame_idx %DRAWING_RATE ==0:
            games_number=len(total_rewards)
            plot_data(model_path+f"Rewards_of_{games_number}_games.png",mean_rewards,'Rewards',games_number)
            plot_data(model_path+f"Losses_of_{games_number}_games.png",mean_q_losses,'Q losses',games_number)
            plot_data(model_path+f"Cleared_lines_of_{games_number}_games.png",mean_cleared_lines,'Cleared Lines',games_number)
            plot_data(model_path+f"Tetrominos_of_{games_number}_games.png",mean_tetrominos,'Tetrominos',games_number)
            hist_data(model_path+f"actions_distributions{games_number}.png",agent.actions,"Actions")
            torch.save(net.state_dict(), f"{model_path}tetris_best_%d.dat" % games_number)
else:
    if args.model==None:
          parser.error("--mode play requires --model")
    state = torch.load(model_path, map_location=lambda stg,_: stg)
    net.load_state_dict(state)
    for i in range (args.play_time):
      r,c,t=agent.play_step(net,mode='play',device=device)
      while r==None:
        # print(envs[0])
        r,c,t=agent.play_step(net,mode='play',device=device)
      print (f'Reward= {r}, Cleared_lines={c}, Tetrominos{t}')
    hist_data(model_path+"actions_distributions.png",agent.actions,"Actions")  
