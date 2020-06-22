
from tetris_engine import TetrisEngine
import controllers
import utils.drawing_utils as dd
import numpy as np 
if __name__ == '__main__':

    done =False 
    action_chosen=None  
    env=TetrisEngine(10,20)
    drawing_rate=5000
    cleared_lines=[]
    tetrominos=[]
    m_cleared_lines=[]
    m_tetrominos=[]
    t=0
    games=1
while True: 
    t+=1
    action = controllers.best_action(env,'near')
    _,_,done=env.step(action)
    if done:
        games+=1
        cleared_lines.append(env.cleared_lines)
        tetrominos.append(env.tetrominos)
        m_cleared_lines.append(np.mean(cleared_lines[-5:]))
        m_tetrominos.append(np.mean(tetrominos[-5:]))
        env.clear()
        
    
    if t%drawing_rate==0:
        dd.plot_data(f"./Graphs/cleared_lines_after_{games}_games.png",m_cleared_lines,"Cleared Lines",games)
        dd.plot_data(f"./Graphs/tetrominos_after_{games}_games.png",m_tetrominos,"Tetrominos",games)


