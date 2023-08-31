import numpy as np
import random
from utils import create_adjacency_matrix, generate_random_string, generate_lobe_trajectory, distance_between_vertices, find_furthest_value, Centroid, maximum_distance
from scipy.sparse.csgraph import shortest_path
from message import message
from Potential import RadialBasisFunction
import matplotlib.pyplot as plt
import sys

class Agent:

    def __init__(self, p, shape, ID) -> None:
        
        # Global
        self.ID = ID # Only for Visualization and Debug
        self.shape = shape
        self.graph = create_adjacency_matrix(self.shape.shape[0])
        

        # Shape Related
        self.shape_ID = None
        self.root_index = None
        self.parent_index = None 
        self.parent_position = None
        self.index = None
        self.child1_index = None # [Index, Bool]
        self.child1_there = False
        # Only Relevant when you are Root, otherwise remains None
        self.child2_index = None
        self.child2_there = False
        # Current Members of shape
        self.vertices_covered = {}
        self.belief_persistence = self.shape.shape[0]
       
        # Current State
        self.state = 'Random_Tour'
        self.neighborhood = None

        # Transition probabilities
        self.p_root = p.p_r_0
        self.p_give_up_root = p.p_r_GU # Constant
        self.p_accept = {'p0':p.p_a_0, 'sharpness':p.p_a_s, 'center':p.p_a_c} # Center govern as decimal of number of agents in the shape
        self.p_give_up = {'p0':p.p_GU_0, 'sharpness':p.p_GU_s, 'center':p.p_GU_c} # Sigmoid like before
        
        # Init Variables for random tour
        self.tour_table = [{'Name':'Init','length':p.init_L, 'width':p.init_W, 'v':p.init_D, 'step':p.init_step},# Init
                           {'Name':'Orbit','length':maximum_distance(self.shape,Centroid(self.shape)), 
                            'width':maximum_distance(self.shape,Centroid(self.shape)), 
                            'v':p.orbit_D, 'total_laps':4,'n_laps':0,'step':p.orbit_step}, # Center
                           {'Name':'Explore','length':p.exp_L, 'width':p.exp_W, 'v':p.exp_D,'step':p.exp_step}] # Explore
        self.tour_params = self.tour_table[0].copy()
        self.shapes_seen = []
        self.tour_history = []
        self.current_traj = None
        self.current_index = 0
        self.travel_to_centroid ={'switch':False, 'shape_ID':None, 'COM':np.array([0,0])}

        # Avoid Rejoining the same shape
        self.shape_to_avoid = {'ID':'random_stringgg', 'patience':p.patience, 'counter':0, 'avoiding':False}


    #===================================
    #++++++++++ MOVEMENT FUNCTIONS +++++
    #===================================

    def move(self):
        # Comment
        
        if self.state == 'Random_Tour':
            v = self.random_tour()
        elif self.state == 'In_Place':
            v = self.in_place()
        elif self.state == 'Root':
            v = self.root()

        repulsion = self.avoid_neighbors()

        print(repulsion)

        v = v + repulsion

        return v
        #print(self.ID, self.state,self.index, self.child1_index,self.child1_there,self.child2_index,self.child2_there, self.tour_params['Name'])
    
    def random_tour(self):

        if self.travel_to_centroid['switch']:
            v = self.travel_to_centroid['COM']/np.linalg.norm(self.travel_to_centroid['COM'])*2
            self.travel_to_centroid['COM'] -= v
            if np.linalg.norm(self.travel_to_centroid['COM']) <= 2:
                self.travel_to_centroid ={'switch':False, 'shape_ID':None, 'COM':np.array([0,0])}
                self.tour_params = self.tour_table[1].copy()
                self.tour_history = []
                self.current_traj = None
                self.current_index = 0


        else:
            if len(self.tour_history) == 0 or self.current_index >= self.current_traj.shape[0] -2:

                # Limit the number of times you can spend obiting a shape, in case it dissociates
                if self.tour_params['Name'] == 'Orbit':
                    if self.tour_params['n_laps'] >= self.tour_params['total_laps']:
                        self.tour_params = self.tour_table[2].copy()
                    else:
                        self.tour_params['n_laps'] +=1
                        
                
                orientation = (random.random()-0.5)*2*np.pi if len(self.tour_history)==0 else find_furthest_value(self.tour_history)
                self.tour_history.append(orientation)
                self.tour_params['length'] += self.tour_params['step']
                self.current_traj = generate_lobe_trajectory(self.tour_params['length'],self.tour_params['width'],self.tour_params['v'],orientation)
                self.current_index = 0
                
            v = self.current_traj[self.current_index+1] - self.current_traj[self.current_index]
            
            self.current_index += 1
            
        return v
    

    def in_place(self):
        # Identify target using graph indices and parent's position
        from_parent_to_target = (self.shape[self.index] - self.shape[self.parent_index])
        resultant = self.parent_position+from_parent_to_target
        return resultant*0.1


    def root(self):
        # The root drone doesn't move
        v = np.array([0,0])
        return v
             

    # ===============================
    #++++++++++LISTEN/Broadcast FUNCTIONS+++
    #================================

    def send_broadcast(self,t):
        # Update your beielfs about the neigborhood before you broadcast them
        self.update_shape_beliefs(t)

        if self.state == 'Root':
            return [message(self.shape_ID,
                       self.root_index,
                       self.index, 
                       self.child1_index, 
                       self.child1_there, self.vertices_covered
                       ), 
                       message(self.shape_ID,self.root_index,
                       self.index, 
                       self.child2_index, 
                       self.child2_there, self.vertices_covered
                       )]
        else:
            # Share you knowledge on the shape so far
            return [message(self.shape_ID,self.root_index,
                        self.index, 
                        self.child1_index, 
                        self.child1_there, self.vertices_covered
                        )]


    def process_broadcast(self, N,B):

        n = []
        b = []

        for i, message in enumerate(B):
            if len(message) == 1:
                n.append(N[i])
                b.append(message[0])
            if len(message) == 2:
                n.append(N[i])
                n.append(N[i])
                b.append(message[0])
                b.append(message[1])
        return np.array(n), b 


    def get_broadcast(self, n, b,t):

        n,b = self.process_broadcast(n,b)
        
        if self.state == 'Random_Tour':
            self.random_tour_listen(n,b,t)

        elif self.state == 'In_Place':
            self.in_place_listen(n,b,t)
            self.shape_unsuccessful_check()

        elif self.state == 'Root':
            self.root_listen(n,b,t)
            self.shape_unsuccessful_check()
        


    def find_available_spots(self, B):
        offers = []
        for i, message in enumerate(B):
            if self.shape_to_avoid['ID'] == message.shape_ID:
                pass
            elif not message.child1_there and message.self_index != None:
                offers.append(i)

        return offers

    def random_tour_listen(self,N,B,t):

        # Store Neighborhood
        self.neighborhood = N
        rand = random.random()
        fragments_in_neighborhood = [1 for message in B if message.self_index != None]
        

        # === Become Root only if you receive no broadcasts ROOT ===========
        if len(fragments_in_neighborhood) == 0 and N.shape[0] > 2 and self.tour_params['Name'] != 'Orbit':
            if rand < self.p_root:
                # Generate ID for new shape
                self.shape_ID = generate_random_string(5)
                # Pick RootVertex/random select
                self.index = random.randint(0,self.shape.shape[0]-1)
                self.root_index = self.index
                # Identify Children
                children = np.argwhere(self.graph[self.index] > 0)
                self.child1_index = children[0][0]
                self.child1_there = False
                self.child2_index = children[1][0]
                self.child2_there = False
                self.vertices_covered.update({self.index:t})
                # Set state to root
                self.state = 'Root'


        #===== JOIN SHAPE ======
        # Find available offers
        available_spots = self.find_available_spots(B)
        
        # If there are no available offers len(offers) = 0 , thus probability you join is 0
        accepted_offer = self.evaluate_offers(available_spots, B)
        if accepted_offer[0]:
            # Find Parent: Pick an offer
            j = accepted_offer[1]
            self.shape_ID = B[j].shape_ID # Set ID of your shape
            self.root_index = B[j].root_index
            self.index = B[j].child1_index # Identify your own index in the graph
            self.vertices_covered[self.index] = t
            self.distance_to_root = distance_between_vertices(self.index, self.root_index, self.graph)
            # Read parents info from the offer and its relative position in the neighborhood
            self.parent_index = B[j].self_index
            self.parent_position = N[j]
            # Figure out child
            connections = np.argwhere(self.graph[self.index] > 0).T[0] # Check connections
            mask = connections != self.parent_index
            self.child1_index = int(connections[mask])
            self.child1_there = False
            self.vertices_covered.update(B[j].vertices_covered)
            
            # Set State to InPlace
            self.state = 'In_Place'

        # Check what type of motion shoul;d follow
        self.check_new_shape(B,N)

        


    def in_place_listen(self,N,B,t):
        
        # Parent Check
        neighbord_indices = []
        for message in B:
            neighbord_indices.append(message.self_index)
        if self.parent_index not in neighbord_indices:
            self.state = 'Random_Tour'
            self.shape_to_avoid['ID'] = self.shape_ID
            self.shape_to_avoid['avoiding'] = True
            self.shapes_seen = [self.shape_ID]
            self.reset()
            return None


        # Store Neighborhood
        self.neighborhood = N 
        # Assume your kid worn be there
        self.child1_there = False
        # Iterate over all messages
        for i,message in enumerate(B):


            # Check for new members in the shape
            if message.shape_ID == self.shape_ID:
                self.vertices_covered.update(message.vertices_covered)

            # Update your parent's position
            if message.self_index == self.parent_index and message.shape_ID == self.shape_ID:
                self.parent_position = N[i]
            # Check for yourself: If you find someone with the same index you might have toi give up your index
            elif message.self_index == self.index and message.shape_ID == self.shape_ID: #and random.random() < 0.5:
                self.reset()
                self.state = 'Random_Tour'
            # Check if if your children are there
            elif message.self_index == self.child1_index and message.shape_ID == self.shape_ID:
                    self.child1_there = True

        
        

            


    def root_listen(self,N,B,t):
        # Store Neighborhood
        self.neighborhood = N

        self.child1_there = False
        self.child2_there = False
        for message in B:

            # Check for new members in the shape
            if message.shape_ID == self.shape_ID:
                self.vertices_covered.update(message.vertices_covered)
            if message.self_index == self.child1_index and message.shape_ID == self.shape_ID:
                self.child1_there = True
            elif message.self_index == self.child2_index and message.shape_ID == self.shape_ID:
                self.child2_there = True
        




    # ====================================
    # +++++++++MISC FUNCTIONS+++++++++++++
    #=====================================

    def reset(self):
        self.shape_ID = None
        self.root_index = None
        self.parent_index = None 
        self.parent_position = None
        self.index = None
        self.distance_to_root = None
        self.child1_index = None
        self.child1_there = False
        self.child2_index = None
        self.child2_there = False
        self.vertices_covered = {}
        # Init Variables for random tour
        self.tour_params = self.tour_table[2].copy()
        self.tour_history = []
        self.current_traj = None
        self.current_index = 0
        self.travel_to_centroid ={'switch':False, 'shape_ID':None, 'COM':np.array([0,0])}
        
        

    
    def inv_sigmoid(self,p0,sharpness,centroid,x):

        return p0*(1-(1/(1+np.exp(-sharpness*(x-centroid*self.shape.shape[0])))))
    
    def sigmoid(self,p0,p_end,sharpness,centroid,x):

        return p0 + p_end*((1/(1+np.exp(-sharpness*(x-centroid*self.shape.shape[0])))))
    

    def check_new_shape(self, B,N):

        for i, b in enumerate(B):

            if b.shape_ID != self.shape_to_avoid['ID'] and b.shape_ID != None and b.shape_ID not in self.shapes_seen:
                # Find Relative vecotr to shape centroid
                self.shapes_seen = b.shape_ID
                vec_b = self.shape[b.self_index]
                shape_centroid = Centroid(self.shape)

                b_relative = N[i]
                b_abs = vec_b - shape_centroid

                centroid_relative = b_relative - b_abs
                # Set Values in dictionary
                self.travel_to_centroid['switch'] = True
                self.travel_to_centroid['shape_ID'] = b.shape_ID
                self.travel_to_centroid['COM'] = centroid_relative
            else:
                pass

        if self.shape_to_avoid['avoiding']:
            self.shape_to_avoid['counter'] += 1
            if self.shape_to_avoid['counter'] >= self.shape_to_avoid['patience']:
                self.shape_to_avoid['ID'] = 'random_stringg'
                self.shape_to_avoid['counter'] = 0
                self.shape_to_avoid['avoiding'] = False

    def shape_unsuccessful_check(self):

        rand = random.random()

        # Condition for breaking out of formation as child of root
        if self.state == 'In_Place':
            if not self.child1_there:
                thresh = self.inv_sigmoid(self.p_give_up['p0'],
                                              self.p_give_up['sharpness'],
                                              self.p_give_up['center'],
                                              len(self.vertices_covered))
                if rand < thresh:
                    self.state = 'Random_Tour'
                    self.shape_to_avoid['ID'] = self.shape_ID
                    self.shape_to_avoid['avoiding'] = True
                    self.shapes_seen = [self.shape_ID]
                    self.reset()
                    

        # Condition for giving up as root
        elif self.state == 'Root':

            if not self.child1_there and not self.child2_there:
                if rand < self.p_give_up_root:
                    self.state = 'Random_Tour'
                    self.reset()

    def evaluate_offers(self, available, B):

        # No offers in the first place
        if len(available) == 0:
            return (False, None)
        
        random.shuffle(available)
        for i in available:
            rand = random.random()
            p = self.sigmoid(self.p_accept['p0'],
                                              1-self.p_accept['p0'], 
                                              self.p_accept['sharpness'],
                                              self.p_accept['center'],
                                              len(B[i].vertices_covered))
            
            if  rand < p: 
                return (True,i)
            
        # None passed
        return (False, None)

        
    def update_shape_beliefs(self,t):

        to_remove = []
        
        for vertex in self.vertices_covered:
            if vertex == self.index:
                self.vertices_covered[vertex] = t
            if t - self.vertices_covered[vertex] > self.belief_persistence:
                to_remove.append(vertex)
        
        for vertex in to_remove:
            del self.vertices_covered[vertex]



    def rbf_gaussian(self,distance, width=3, height=10):
        return -height*np.exp(-(distance ** 2) / (2 * width ** 2))
        
        
    def avoid_neighbors(self):

        if self.neighborhood.shape[0] > 0:
            # Find Magnitude
            distances = np.linalg.norm(self.neighborhood, axis=1)
            # Apply RBF centered at each neighbor
            rbf_mag = self.rbf_gaussian(distances)
            # FInd repulsions, same direction
            forces = (self.neighborhood/distances[:,np.newaxis])*rbf_mag[:,np.newaxis]

            return np.sum(forces, axis=0)
        
        else:
            return np.array([0.0,0.0])


        

    def plot_sigmoids(self):

        a = np.arange(0,self.shape.shape[0])
        result1 = self.sigmoid(self.p_accept['p0'],
                                              1-self.p_accept['p0'], 
                                              self.p_accept['sharpness'],
                                              self.p_accept['center'],
                                              a)
        result2 = self.inv_sigmoid(self.p_give_up['p0'], 
                                              self.p_give_up['sharpness'],
                                              self.p_give_up['center'],
                                              a)
        fig, axs = plt.subplots(1,2)
        axs[0].scatter(a,result1)
        axs[1].scatter(a, result2)
        plt.show()

   
    




        

