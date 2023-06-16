import numpy as np
from Agent import Agent


class Env:

    def __init__(self, env_size, n_agents,neighborhood_radius, CONNECTED=False) -> None:

        self.env_size = env_size
        self.n_radius = neighborhood_radius


        if CONNECTED:
            initial_positions = self.connected_pos_gen(n_agents, neighborhood_radius)
            self.agents = [{'agent':Agent(), 
                            'pos':pos} 
                            for pos in initial_positions]

        else:
            
            self.agents = [{'agent':Agent(), 
                            'pos':(np.random.random((1,2))-0.5)*2*self.env_size} 
                            for agent in range(n_agents)]


    def timestep(self):

        # Perform all actions simultaneously (although they are computed separately)
        # ASSUMPTION: Agents operate synchronously
        

        # Gather Actions

        tstep_actions = []

        for agent in self.agents:

            n = self.get_neighborhood(agent['pos'])
            v = agent['agent'].move(n)
            tstep_actions.append(v)
        

        # Perform actions
        for i, action in enumerate(tstep_actions):
            self.agents[i]['pos'] += action

        
            

    def get_neighborhood(self, pos):

        # Remove self from list
        positions = self.make_positions_list()
        positions = positions[np.all(positions!=pos, axis=1)]
        
        # Compute norm on relative position vectors
        relative_positions = positions - pos
        norm = np.linalg.norm(relative_positions, axis=1)

        # Neighborhood
        neighborhood = relative_positions[norm <= self.n_radius]
        return neighborhood


    def make_positions_list(self):

        # Make a list containing the positions of all the agents

        positions = [agent['pos'] for agent in self.agents]
        
        return np.squeeze(np.array(positions))
    

    def connected_pos_gen(self, n_agents, r, CENTERED=True):
        
        if CENTERED:
            pos_list = np.array([np.array([0,0])])
        else:
            pos_list = (np.random.random((1,2))-0.5)*2*self.env_size

        while pos_list.shape[0] < n_agents:

            new_random_point = (np.random.random((1,2))-0.5)*2*self.env_size

            relative_distances = pos_list - new_random_point
            distances = np.linalg.norm(relative_distances, axis=1)

            if np.any(distances <= r):
                
                pos_list = np.concatenate((pos_list,new_random_point), axis=0)
        
        return pos_list
