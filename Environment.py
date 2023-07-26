import numpy as np
from Agent import Agent
from utils import connected_pos_gen
import random


class Env:

    def __init__(self, env_size,shape, n_agents,neighborhood_radius, params, CONNECTED=False) -> None:

        self.env_size = env_size
        self.n_agents = n_agents
        self.n_radius = neighborhood_radius
        self.shape = shape
        self.CONNECTED = CONNECTED
        self.params = params
        self.n_radius = neighborhood_radius
        self.t = 0
        self.agents = []


        

    def generate_agents(self):

        if self.CONNECTED:
            initial_positions = connected_pos_gen(self.env_size,self.n_agents, self.n_radius)
            self.agents = [{'agent':Agent(self.params,self.shape,i), 
                            'pos':pos} 
                            for i, pos in enumerate(initial_positions)]

        else:
            self.agents = [{'agent':Agent(self.params,self.shape, agent), 
                            'pos':(np.random.random((1,2))[0]-0.5)*2*self.env_size} 
                            for agent in range(self.n_agents)]



    def timestep(self):

        # Perform all actions simultaneously (although they are computed separately)
        # ASSUMPTION: Agents operate synchronously
        
        # Every agents broadcasts
        current_broadcasts = self.get_broadcasts()
        # scan neighborhood and get the broadcasts of neighborhood
        for agent in self.agents:
            # Get neighborhood and broadcast

            n,b=self.get_local_info(agent['pos'],current_broadcasts)

            agent['agent'].get_broadcast(n,b,self.t)
        
        # Move

        for agent in self.agents:

            v = agent['agent'].move()
            agent['pos'] += v
        
        self.t +=1
        
        
    def get_local_info(self, pos, current_broadcasts):

        # Remove self from list
        positions = self.make_positions_list()
        mask1 = np.all(positions!=pos, axis=1)
        positions = positions[mask1]
        # Compute norm on relative position vectors
        relative_positions = positions - pos
        norm = np.linalg.norm(relative_positions, axis=1)

        mask2 = norm <= self.n_radius

        # Neighborhood positions
        neighborhood = relative_positions[mask2]

        # Neighborhood broadcasts
        broadcast = [b for a, b in zip(mask1, current_broadcasts) if a]
        broadcast = [b for a, b in zip(mask2, broadcast) if a]

        return neighborhood, broadcast


    def make_positions_list(self):

        # Make a list containing the positions of all the agents

        positions = [agent['pos'] for agent in self.agents]
        
        return np.squeeze(np.array(positions))
    

    def get_broadcasts(self):

        broadcasts = [agent['agent'].send_broadcast(self.t) for agent in self.agents]

        return broadcasts
    
    def done_assuming_convergence(self):

        # Simplest CHeck possible Really
        agent_sampled = random.choice(self.agents)
        if len(agent_sampled['agent'].vertices_covered) == self.shape.shape[0]:
            return True
        else:
            return False
        
    
    def run_sim(self):

        #Generate Fresh Batch of Agents
        self.generate_agents()
        self.t = 0
        
        DONE = False
        # Run the simulation
        while not DONE:
            self.timestep()
            DONE = self.done_assuming_convergence()

        # Reutn really simple loss
        return -self.t





    
