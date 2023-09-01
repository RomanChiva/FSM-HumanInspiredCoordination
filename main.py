import matplotlib.pyplot as plt
from Environment import Env
from Run_Sim import viz_sim
import numpy as np
import sys
from utils import circle_maker, square_maker
from gen_params import generate_agent_parameters

if __name__ == '__main__':

    # Shape we wish to from: Points MUST BE ORDERED!!! 
    sensing_radius = 30
    spread = 2
    shape_main_axis=2
    AR = 1

    circle = circle_maker(20,45)
    rectangle, n_agents = square_maker(sensing_radius*shape_main_axis,sensing_radius*shape_main_axis/AR,6,6)
    
    angle = 0.75 # More or less 90 degrees
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    
    #rotated = np.dot(rectangle,rotation_matrix)
    #plt.scatter(rotated[:,0],rotated[:,1])
    #plt.show()


    # Import Agent Parameters
    parameters = generate_agent_parameters()

    # DEFINE ENVIRONMENT: Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
    env = Env(100,circle,20,sensing_radius,parameters,spread*sensing_radius)

    viz_sim(env, margins=100)
    
    

    

    
    






        
    

