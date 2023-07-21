import matplotlib.pyplot as plt
from Environment import Env
from Run_Sim import run_sim
import numpy as np
import sys
from utils import circle_maker, Centroid


if __name__ == '__main__':
    np.random.seed(123)
    # Shape we wish to from: Points MUST BE ORDERED!!! 
    rectangle_8 = np.array([[10,0],[10,10],[0,10],[-10,10],[-10,0],[-10,-10],[0,-10],[10,-10]], dtype=float)
    circle_20 = circle_maker(20,50)
    # Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
    env = Env(100,circle_20,20,30,CONNECTED=False)
    run_sim(env, margins=100)

    

    
    






        
    

