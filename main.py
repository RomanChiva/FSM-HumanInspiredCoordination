import matplotlib.pyplot as plt
from Environment import Env
from Run_Sim import run_sim
import numpy as np


if __name__ == '__main__':
    np.random.seed(123)
    # Shape we wish to from
    rectangle_1 = np.array([[10,0],[10,10],[0,10],[-10,10],[-10,0],[-10,-10],[0,-10],[10,-10]], dtype=float)

    # Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
    env = Env(100,rectangle_1,20,50, CONNECTED=True)
    run_sim(env)

    

    
    






        
    

