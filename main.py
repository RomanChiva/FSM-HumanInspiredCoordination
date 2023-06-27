import matplotlib.pyplot as plt
from Environment import Env
from Run_Sim import run_sim



if __name__ == '__main__':

    # Env SIze, N Agents, Nbrhd Radius, Spawn as Connected Graph

    env = Env(100,20,30, CONNECTED=True)
    run_sim(env)

    

    
    






        
    

