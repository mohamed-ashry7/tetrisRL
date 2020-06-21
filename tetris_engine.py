#from __future__ import print_function



# This File is Originally taken from https://github.com/jaybutera/tetrisRL. 
# There are some modification to this environment to be fine the model chosen.

import numpy as np
import random

shapes = {
    'T': [(0, 0), (-1, 0), (1, 0), (0, -1)],
    'J': [(0, 0), (-1, 0), (0, -1), (0, -2)],
    'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
    'Z': [(0, 0), (-1, 0), (0, -1), (1, -1)],
    'S': [(0, 0), (-1, -1), (0, -1), (1, 0)],
    'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
    'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}
shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']


def rotated(shape, cclk=False):
    if cclk:
        return [(-j, i) for i, j in shape]
    else:
        return [(j, -i) for i, j in shape]


def is_occupied(shape, anchor, board,h=False):
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        if y < 0:
            continue
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            if h:
                return True, max(board.shape[1]-y,0)
            return True
    if h:
        return False,None
    return False


def left(shape, anchor, board):
    new_anchor = (anchor[0] - 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def right(shape, anchor, board):
    new_anchor = (anchor[0] + 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def soft_drop(shape, anchor, board):
    new_anchor = (anchor[0], anchor[1] + 1)
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def hard_drop(shape, anchor, board):
    while True:
        _, anchor_new = soft_drop(shape, anchor, board)
        if anchor_new == anchor:
            return shape, anchor_new
        anchor = anchor_new


def rotate_left(shape, anchor, board):
    new_shape = rotated(shape, cclk=False)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def rotate_right(shape, anchor, board):
    new_shape = rotated(shape, cclk=True)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)

def idle(shape, anchor, board):
    return (shape, anchor)


class TetrisEngine:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=np.float)

        # actions are triggered by letters
        self.value_action_map = {
            0: left,
            1: right,
            2: hard_drop,
            3: soft_drop,
            4: rotate_left,
            5: rotate_right,
            6: idle,
        }
        self.action_value_map = dict([(j, i) for i, j in self.value_action_map.items()])
        self.nb_actions = len(self.value_action_map)
        
        
        
        self.group_actions_number=4*(self.width) # Means that there are 4 possible rotations and width-number of translations +1 for idle translations
        
        # for running the engine
        self.time = -1
        self.score = -1
        self.anchor = None
        self.shape = None
        
        self.prev_state_evaluation=0
        self.landing_height=None
        self.cleared_lines=0
        self.cleared_lines_per_move=0
        # used for generating shapes
        self._shape_counts = [0] * len(shapes)
        
        self.piece_number=None
        self.tetrominos=0
        # clear after initializing
        self.clear()
    
    
    
    def _choose_shape(self):
        maxm = max(self._shape_counts)
        m = [5 + maxm - x for x in self._shape_counts]
        r = random.randint(1, sum(m))
        for i, n in enumerate(m):
            r -= n
            if r <= 0:
                self._shape_counts[i] += 1
                self.piece_number=i
                return shapes[shape_names[i]]

    def _new_piece(self):
        # Place randomly on x-axis with 2 tiles padding
        #x = int((self.width/2+1) * np.random.rand(1,1)[0,0]) + 2
        self.tetrominos+=1
        self.anchor = (self.width //2, 0)
        #self.anchor = (x, 0)
        self.shape = self._choose_shape()
        self.shape, self.anchor = soft_drop(self.shape, self.anchor, self.board)

        
    # Modification
    
    
    def _has_dropped(self):
        is_occ,self.landing_height=is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board,h=True)
        return is_occ
    
    
    #Modification
    def _clear_lines(self):
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1
        self.score += sum(can_clear)
        self.board = new_board
        self.cleared_lines_per_move=sum(can_clear)
        self.cleared_lines += sum(can_clear)


    #Modification
    
    #Modification
    
    def _rotate_grouped_action(self,rotations):
        action_taken =5 if rotations==1 else 4
        
        self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
        self.shape, self.anchor = self.value_action_map[action_taken](self.shape, self.anchor, self.board)
        
        if rotations==3:# face up so that means 2 rotate_left or 2 rotate_right
            self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
            self.shape, self.anchor = self.value_action_map[action_taken](self.shape, self.anchor, self.board)
    
    def _translate_grouped_action(self,translations,is_right):
        action_taken=1 if is_right else 0
        for _ in range(translations):
            self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
            self.shape, self.anchor = self.value_action_map[action_taken](self.shape, self.anchor, self.board)
    
    
    def _exec_grouped_actions(self,action):
        
        
        #Basic Actions
        #0: left,
        #1: right,
        #2: hard_drop,
        #3: soft_drop,
        #4: rotate_left,
        #5: rotate_right,
        #6: idle,
        
        #0->10  idle, for all v in first five values means move v+1 to the right and the last 5 moves means v+1-5 to the left
        # the same for the other grouped actions 
        #it is stated that it is 0->10 because the expected width is 10 
        is_right=False
        rotations =action//(self.width)
        translations =action%(self.width)
        if translations>=self.width//2:
            is_right=True
            translations-=self.width//2
        else:
            translations=self.width//2 - translations
        
        if rotations>0:
            self._rotate_grouped_action(rotations)
        if translations>0:    
            self._translate_grouped_action(translations,is_right)
        
        self.shape, self.anchor = hard_drop(self.shape, self.anchor, self.board)

    
   
    def calc_state(self):
        # state= np.copy(self.board).T
        # prev_height=0
        # col_heights =[]
        # edge=3
        # for j in range(state.shape[1]):
        #     col=state[:,j]
        #     arr_ind=np.where(col==1)[0]
        #     col_height=0 if len(arr_ind)==0 else len(col)-arr_ind[0]
        #     diff =col_height-prev_height
        #     if diff>edge:
        #         diff=edge
        #     elif diff < -edge:
        #         diff=-edge
        #     if j>0:
        #         col_heights.append(diff)
        #     prev_height=col_height
            
        
        
        # col_heights.append(self.piece_number)

        state=np.copy(self.board).T

        self.holes=0
        self.wells=0
        self.qu=0
        avgh=0
        maxh=0
        self.col_heights=[]
        prev_h=0
        for j in range(state.shape[1]):
            
            well_depth=0
            col=state[:,j]
            arr_ind=np.where(col==1)[0]
            col_height=0 if len(arr_ind)==0 else len(col)-arr_ind[0]
            self.col_heights.append(col_height)
            if j>0:
                self.qu+=(col_height-prev_h)**2
            prev_h=col_height
            for i in range(state.shape[0]-1,0,-1):
                self.holes+=1 if state[i,j]==0 and state[i-1,j]==1 else 0
                if state[i,j]==0:
                    if j-1 <0 and state[i,j+1] or j+1 >=state.shape[1] and state[i,j-1] or state[i,j-1] and state[i,j+1]:
                        well_depth+=1
                    else:
                        self.wells+=(well_depth+1)*well_depth/2
                        well_depth=0
            self.wells+=(well_depth+1)*well_depth/2


        avgh=np.mean(self.col_heights)
        maxh=max(self.col_heights)
        diffs=[self.col_heights[i]-self.col_heights[i-1] for i in range(1,len(self.col_heights))]
        
        state=np.copy(diffs)
        state=np.append(state,self.holes)
        state=np.append(state,self.qu)
        # state=np.append(state,self.wells)
        state=np.append(state,self.piece_number)
        return state
    
    
    def sigmoid(self,r):
        # r/=25 
        # return (1/(1+np.exp(-r))-0.5)*10
        return r/15
    def calc_reward(self):
        
        state_evaluation= self.calc_state_evaluation()
        reward=state_evaluation-self.prev_state_evaluation + self.tetrominos + 30*self.cleared_lines_per_move
        self.prev_state_evaluation=state_evaluation
        return reward
    
    
    def step(self, action):
        
        
       
        self._exec_grouped_actions(action)
        
        # Update time and reward
        self.time += 1
        reward=0

        done = False
        if self._has_dropped():
            
            self._set_piece(True)
            self._clear_lines()
            state = self.calc_state()
            if np.any(self.board[:, 0]):
                done = True
            else:
                self._new_piece()
            self._set_piece(False)

        
        #calcualte the Reward based on the Evaluation of the states. 
        
        
        reward = -1 if done else self.calc_reward() 
        return state, round(reward,3), done

    def clear(self):
        self.time = 0
        self.score = 0
        self._new_piece()
        self.board = np.zeros_like(self.board)
        #Modification
        self.prev_state_evaluation=0
        self.landing_height=None
        self.cleared_lines=0
        self.cleared_lines_per_move=0
        self.tetrominos=0
        return self.calc_state()

    def _set_piece(self, on=False):
        for i, j in self.shape:
            x, y = i + self.anchor[0], j + self.anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                self.board[int(self.anchor[0] + i), int(self.anchor[1] + j)] = on

    def __repr__(self):
        self._set_piece(True)
        s = 'o' + '-' * self.width + 'o\n'
        s += '\n'.join(['|' + ''.join(['X' if j else ' ' for j in i]) + '|' for i in self.board.T])
        s += '\no' + '-' * self.width + 'o'
        self._set_piece(False)
        return s
    
    
    # ADDED Functions
    #Modification   
    def random_action(self):
        return int(np.random.random()*self.group_actions_number)
    
    def number_actions(self):
        return self.group_actions_number
    
    def state_shape(self):
        return np.prod(self.calc_state().shape)
    
    
    
    def calc_state_evaluation(self):
        # This Function is based on the features of Dr. Schwenker and Dellachereie
        #Schwenker Features are average height->avgh , holes -> holes, maximum height->maxh, Quadratic UnEvenness ->qu
        #You can review the paper that discusses this feature 
        # "A Reinforcement Learning Algorithm to Train a Tetris Playing Agent"
        #    Patrick Thiam, Viktor Kessler, and Friedhelm Schwenker
        # The Dellacherie features can be found in that paper ->Fahey, C. P. (2003). Tetris AI, Computer plays Tetris
        
        # The chosen features would be only qu,avg,holes,wells,maxh
        
                
        avgh=np.mean(self.col_heights)
                    
        estimated_evaluation= -5*avgh - self.qu - 16*self.holes
        return self.sigmoid(estimated_evaluation)
                


# if __name__ == '__main__':
#     env = TetrisEngine(10,20)
#     while True:
#         action =np.random.randint(0,40)
#         state , reward, done = env.step(action)
#         print(env)
#         print(f"Reward {reward} , state{state}, Action {action},Lines {env.cleared_lines}")
#         if done:
#             if env.cleared_lines>0:
#                 break
#             else:
#                 env.clear()

