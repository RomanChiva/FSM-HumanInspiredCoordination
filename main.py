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

    circle = circle_maker(20,60)
    rectangle, n_agents = square_maker(sensing_radius*shape_main_axis,sensing_radius*shape_main_axis/AR,4,4)
    
    angle = 0.75 # More or less 90 degrees
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    
    #rotated = np.dot(rectangle,rotation_matrix)
    #plt.scatter(rotated[:,0],rotated[:,1])
    #plt.show()


    # Import Agent Parameters
    parameters = generate_agent_parameters(sensing_radius,spread)

    # DEFINE ENVIRONMENT: Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
    env = Env(100,circle,20,sensing_radius,parameters,spread*sensing_radius)
    t = env.run_sim()
    print(t)

    viz_sim(env, margins=100)
    
    

    

    
# Function that generates a circle of points taking as input number of agents and inter agent distance, so the circle grows proportionally larger with more agents

def generate_circle(n_agents, inter_agent_distance):
    # Generate a circle of points
    # n_agents: number of agents
    # inter_agent_distance: distance between agents
    # returns: array of points in a circle

    # Generate points in a circle
    # https://stackoverflow.com/questions/50731785/how-to-generate-random-points-in-a-circle
    # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly

    # Generate random angles
    angles = np.random.uniform(0, 2*np.pi, n_agents)
    # Generate random radii
    radii = np.random.uniform(0, inter_agent_distance, n_agents)
    # Convert polar to cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    # Return points
    return np.array([x,y]).T









        
    

