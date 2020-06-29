import numpy as np
import copy 




def basic_evaluation_fn(env,controller,abs_value=True):
    state=np.copy(env.board).T
    holes=0
    wells=0
    row_trans=0
    col_trans=0
    col_heights=[]
    for j in range(state.shape[1]):
        well_depth=0
        col=state[:,j]
        arr_ind=np.where(col==1)[0]
        col_height=0 if len(arr_ind)==0 else len(col)-arr_ind[0]
        col_heights.append(col_height)
        for i in range(state.shape[0]-1,0,-1):
            if j>0:
                row_trans+=1 if state[i,j]!=state[i,j-1] else 0
            col_trans+=1 if state[i,j]!=state[i-1,j] else 0 
            holes+=1 if state[i,j]==0 and state[i-1,j]==1 else 0
            if state[i,j]==0:
                if j-1 <0 and state[i,j+1] or j+1 >=state.shape[1] and state[i,j-1] or state[i,j-1] and state[i,j+1]:
                    well_depth+=1
                else:
                    wells+=(well_depth+1)*well_depth/2
                    well_depth=0
        wells+=(well_depth+1)*well_depth/2
    
    qu =sum([(col_heights[i]-col_heights[i-1])**2 for i in range(1,len(col_heights))])
    if controller=='schwenker':
        # These feature came from [2]
        avgh= np.mean(col_heights)
        return -5*avgh-16*holes-qu if abs_value else (avgh,holes,qu,col_heights)
    elif controller=='near':
        # These features came from [1]
        agg_height=sum(col_heights)
        complete_lines=env.cleared_lines_per_move
        bumpiness=sum([abs(col_heights[i]-col_heights[i-1]) for i in range(1,len(col_heights))])
        return -0.51*agg_height+0.76*complete_lines-0.36*holes-0.18*bumpiness if abs_value else (agg_height,complete_lines,holes,bumpiness)
    elif controller=='dellacherie':
        estimated_evaluation= -env.landing_height+50*(env.cleared_lines_per_move)**2-row_trans\
            -col_trans-4*holes-wells
        return estimated_evaluation if abs_value else (row_trans,col_trans,holes,wells) # The reason that landing height and cleared lines are not returned,
                                                                                        # That they can be got from the env directly
    elif controller=='lundgaard':
        avgh=sum(col_heights)//len(col_heights)
        holes= holes//5 + 1 
        if holes >5 :
            holes = 5 
        qu = qu//20+1
        if qu>20:
            qu=20
        single_valley = 1 if wells>0 else 0 
        return avgh,qu,single_valley,holes
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



