import numpy as np
from Agent import Agent
from utils import connected_pos_gen

class Env:

    def __init__(self, env_size,shape, n_agents,neighborhood_radius, CONNECTED=False) -> None:

        self.env_size = env_size
        self.n_radius = neighborhood_radius
        self.shape = shape


        if CONNECTED:
            initial_positions = connected_pos_gen(self.env_size,n_agents, neighborhood_radius)
            self.agents = [{'agent':Agent(self.shape,i), 
                            'pos':pos} 
                            for i, pos in enumerate(initial_positions)]

        else:
            
            self.agents = [{'agent':Agent(self.shape, agent), 
                            'pos':(np.random.random((1,2))[0]-0.5)*2*self.env_size} 
                            for agent in range(n_agents)]


    def timestep(self):

        # Perform all actions simultaneously (although they are computed separately)
        # ASSUMPTION: Agents operate synchronously
        
        # Exchange_messages and scan neighborhood

        for agent in self.agents:
            # Get neighborhood and broadcast

            n,b=self.get_local_info(agent['pos'])

            agent['agent'].get_broadcast(n,b)
        
        # Move

        for agent in self.agents:

            v = agent['agent'].move()
            agent['pos'] += v
        

        
        
        
    def get_local_info(self, pos):

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
        broadcast = [b for a, b in zip(mask1, self.get_broadcasts()) if a]
        broadcast = [b for a, b in zip(mask2, broadcast) if a]

        return neighborhood, broadcast


    def make_positions_list(self):

        # Make a list containing the positions of all the agents

        positions = [agent['pos'] for agent in self.agents]
        
        return np.squeeze(np.array(positions))
    

    def get_broadcasts(self):

        broadcasts = [agent['agent'].send_broadcast() for agent in self.agents]

        return broadcasts


    
