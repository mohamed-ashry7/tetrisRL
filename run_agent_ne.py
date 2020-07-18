# if __name__=="__main__":
        
#     parser=argparse.ArgumentParser()
#     parser.add_argument("--cuda",default=False,help="Enable cuda",action="store_true")
#     parser.add_argument("--width",default=10,help="Board Width",action="store")
#     parser.add_argument("--height",default=20,help="Board Height",action="store")
#     parser.add_argument("--mode",default='train',help="Define the mode of the model play or train",action="store")
#     parser.add_argument("--model",default=None,help="Model file to load",action='store')
#     parser.add_argument("--model-dir",default=None,help="dir to save models at it",action='store')
#     parser.add_argument("--play-times",default=None,help="Testing the model how many times",action='store')


#     args=parser.parse_args()
    
#     device=torch.device("cuda" if args.cuda else "cpu")
#     env=TetrisEngine(args.width,args.height)

#     net=DQN(env.state_shape(),env.number_actions()).to(device)
#     target_net=DQN(env.state_shape(),env.number_actions()).to(device)
#     replay_buffer=ReplayBuffer(REPLAY_SIZE)
#     print(net)
#     agent=Agent(env,replay_buffer)
    

#     model_path="./dqn_models_stats/" if args.model_dir==None else args.model_dir
    
#     if args.mode =='train':
#         epsilon = EPSILON_START
#         optimizer=torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)
#         total_rewards = []
#         mean_rewards=[]
#         mean_q_losses=[]
#         q_losses=[]
#         cleared_lines =[]
#         mean_cleared_lines=[]
#         tetrominos =[]
#         mean_tetrominos=[]
#         frame_idx = 0
#         ts_frame = 0
#         ts = time.time()
#         best_m_reward = None

#         while True:
#             frame_idx += 1
#             epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
#             reward,c,t = agent.play_step(net, epsilon, device=device)
            
#             if reward is not None:
#                 cleared_lines.append(c)
#                 mean_cleared_lines.append(np.mean(cleared_lines[-100:]))

#                 tetrominos.append(t)
#                 mean_tetrominos.append(np.mean(tetrominos[-100:]))

#                 total_rewards.append(reward)
#                 m_reward = np.mean(total_rewards[-100:])
#                 mean_rewards.append(m_reward)

#                 speed = (frame_idx - ts_frame) / (time.time() - ts)
#                 ts_frame = frame_idx
#                 ts = time.time()
                
#                 print("%d: done %d games, reward %.3f, "
#                 "eps %.2f, speed %.2f f/s" % (frame_idx, len(total_rewards), m_reward, epsilon,speed))
#                 if best_m_reward is None or best_m_reward < m_reward:
#                     torch.save(net.state_dict(), f"{model_path}tetris_best_%.0f.dat" % m_reward)
#                     if best_m_reward is not None:
#                         print("Best reward updated %.3f -> %.3f" % (
#                         best_m_reward, m_reward))
#                     best_m_reward = m_reward
                

#             if len(replay_buffer) < REPLAY_START_SIZE:
#                 continue
#             if frame_idx % SYNC_TARGET_FRAMES == 0:
#                 target_net.load_state_dict(net.state_dict())
            
#             optimizer.zero_grad()
#             batch = replay_buffer.sample(BATCH_SIZE)
#             loss_t = agent.calc_loss(batch, net, target_net, device=device)
#             q_losses.append(loss_t.item())
#             mean_q_losses.append(np.mean(q_losses[-100:]))
#             loss_t.backward()
#             optimizer.step()
            
#             if frame_idx %DRAWING_RATE ==0:
#                 games_number=len(total_rewards)
#                 plot_data(model_path+f"Rewards_of_{games_number}_games.png",mean_rewards,'Rewards',games_number)
#                 plot_data(model_path+f"Losses_of_{games_number}_games.png",mean_q_losses,'Q losses',games_number)
#                 plot_data(model_path+f"Cleared_lines_of_{games_number}_games.png",mean_cleared_lines,'Cleared Lines',games_number)
#                 plot_data(model_path+f"Tetrominos_of_{games_number}_games.png",mean_tetrominos,'Tetrominos',games_number)
#                 torch.save(net.state_dict(), f"{model_path}tetris_best_%d.dat" % games_number)
#     else:
        
#         if args.model==None:
#             parser.error("--mode play requires --model")
        
#         state = torch.load(args.model, map_location=lambda stg,_: stg)
#         net.load_state_dict(state)
#         cleared_lines =0
#         for i in range (args.play_time):
#             while agent.play_step(net,mode='play')==None:
#                 cleared_lines+=env.cleared_lines
#                 print(env)
#         print(cleared_lines)