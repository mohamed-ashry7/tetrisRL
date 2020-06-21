import numpy as np
import copy 









def basic_evaluation_fn(env,controller,abs_value=True):
    state=np.copy(env.board).T
    holes=0
    wells=0
    col_heights=[]
    for j in range(state.shape[1]):
        well_depth=0
        col=state[:,j]
        arr_ind=np.where(col==1)[0]
        col_height=0 if len(arr_ind)==0 else len(col)-arr_ind[0]
        col_heights.append(col_height)
        for i in range(state.shape[0]-1,0,-1):
            holes+=1 if state[i,j]==0 and state[i-1,j]==1 else 0
            if state[i,j]==0:
                if j-1 <0 and state[i,j+1] or j+1 >=state.shape[1] and state[i,j-1] or state[i,j-1] and state[i,j+1]:
                    well_depth+=1
                else:
                    wells+=(well_depth+1)*well_depth/2
                    well_depth=0
        wells+=(well_depth+1)*well_depth/2
    

    if controller=='schwenker':
        # These feature came from [2]
        avgh= np.mean(col_heights)
        qu =sum([(col_heights[i]-col_heights[i-1])**2 for i in range(1,len(col_heights))])
        return -5*avgh-16*holes-qu if abs_value else (avgh,holes,qu,col_heights)
    elif controller=='near':
        # These features came from [1]
        agg_height=sum(col_heights)
        complete_lines=env.cleared_lines_per_move
        bumpiness=sum([abs(col_heights[i]-col_heights[i-1]) for i in range(1,len(col_heights))])
        return -0.51*agg_height+0.76*complete_lines-0.36*holes-0.18*bumpiness if abs_value else (agg_height,complete_lines,holes,bumpiness)
    else :
        return None
    

def best_action(env,controller):


    evaluation=None
    action_chosen=None 
    for action in range(env.group_actions_number):
        dumm = copy.deepcopy(env)
        dumm.step(action)

        value = basic_evaluation_fn(dumm,controller,True)
        if evaluation==None or evaluation<value:
            evaluation=value
            action_chosen=action
    return action_chosen


# if __name__ == '__main__':

# done =False 
# action_chosen=None  
# env=TetrisEngine(10,20)
# adding_rate=10
# drawing_rate=5000
# cleared_lines=[]
# tetrominos=[]
# m_cleared_lines=[]
# m_tetrominos=[]
# t=0
# games=1
# while True: 
#     action = best_action(env,'near')
#     _,_,done=env.step(action)
#     print(env)
#     if done:
#         print(env.cleared_lines)
#         break
# games+=1
# cleared_lines.append(env.cleared_lines)
# tetrominos.append(env.tetrominos)
# env.clear()
        
#if len(cleared_lines)>len(m_cleared_lines):
#     m_cleared_lines.append(np.mean(cleared_lines[-3:]))
#     m_tetrominos.append(np.mean(tetrominos[-3:]))
#if t%drawing_rate==0:
#     plot_data(f"./Graphs/cleared_lines_after_{games}_games.png",m_cleared_lines,"Cleared Lines",games)
#     plot_data(f"./Graphs/tetrominos_after_{games}_games.png",m_tetrominos,"Tetrominos",games)



