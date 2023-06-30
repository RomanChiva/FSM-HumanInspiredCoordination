import matplotlib.pyplot as plt
from Environment import Env
from Run_Sim import run_sim
import numpy as np
import sys



if __name__ == '__main__':
    np.random.seed(123)
    # Shape we wish to from: Points MUST BE ORDERED!!! 

    # EG for a square: Any starting point is fine, but they must be ordered like that

    # 1    2    3

    # 8         4

    # 7    6    5

    rectangle_1 = np.array([[10,0],[10,10],[0,10],[-10,10],[-10,0],[-10,-10],[0,-10],[10,-10]], dtype=float)
    

    
    # Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
    env = Env(100,rectangle_1,20,50, CONNECTED=True)
    run_sim(env)

    

    
    






        
    

