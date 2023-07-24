import matplotlib.pyplot as plt
from Environment import Env
from Run_Sim import run_sim
import numpy as np
import sys
from utils import circle_maker, square_maker


if __name__ == '__main__':
    # Shape we wish to from: Points MUST BE ORDERED!!! 
    rectangle_8 = np.array([[10,0],[10,10],[0,10],[-10,10],[-10,0],[-10,-10],[0,-10],[10,-10]], dtype=float)
    circle = circle_maker(20,50)
    rectangle, n_agents = square_maker(100,100,6,6)
    # Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
    env = Env(100,rectangle,n_agents,30,CONNECTED=False)
    run_sim(env, margins=100)

    

    
    






        
    

