import numpy as np
import random
from utils import create_adjacency_matrix, generate_random_string, generate_lobe_trajectory, distance_between_vertices, find_furthest_value, Centroid
from scipy.sparse.csgraph import shortest_path
from message import message
import smallestenclosingcircle
from Potential import RadialBasisFunction
import matplotlib.pyplot as plt

class Agent:

    def __init__(self, shape, ID) -> None:
        
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
        self.vertices_covered = set()
       
        # Current State
        self.state = 'Random_Tour'
        self.neighborhood = None

        # Transition probabilities
        self.p_root = 0.01
        self.p_accept = {'p0':0.3, 'sharpness':0.7, 'center':0.5} # Center govern as decimal of number of agents in the shape
        self.p_give_up_root = 0.2 # Constant
        self.p_give_up = {'p0':0.1, 'sharpness':0.6, 'center':0.4} # Sigmoid like before
        
        # Init Variables for random tour
        self.tour_params = {'length':50, 'width':30, 'v':40}
        self.tour_history = []
        self.current_traj = None
        self.current_index = 0
        self.travel_to_centroid ={'switch':False, 'shape_ID':None, 'COM':np.array([0,0])}

        # Avoid Rejoining the same shape
        self.shape_to_avoid = {'ID':'random_stringgg', 'patience':100, 'counter':0, 'avoiding':False}
        

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
        

        return v
    

    def RendezVous(self, neighborhood):
        
        cx,cy,r = smallestenclosingcircle.make_circle(neighborhood)

        heading = np.array([cx,cy])

        v = heading/np.linalg.norm(heading)

        if random.random() < self.RV_RT:
            self.state = 'RandomTour'

        self.curr_n_size = neighborhood.shape[0]

        return v
    
    def random_tour(self):

        if self.travel_to_centroid['switch']:
            v = self.travel_to_centroid['COM']/np.linalg.norm(self.travel_to_centroid['COM'])*2
            self.travel_to_centroid['COM'] -= v
            if np.linalg.norm(self.travel_to_centroid['COM']) <= 5:
                self.travel_to_centroid['switch'] = False
                self.travel_to_centroid['shape_ID'] = None
                self.travel_to_centroid['COM'] = np.array([0,0])
                self.tour_params = {'length':60, 'width':30, 'v':40}
                self.tour_history = []
                self.current_traj = None
                self.current_index = 0


        else:
            if len(self.tour_history) == 0 or self.current_index >= self.current_traj.shape[0] -2:
                orientation = (random.random()-0.5)*2*np.pi if len(self.tour_history)==0 else find_furthest_value(self.tour_history)
                self.tour_history.append(orientation)
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

    def send_broadcast(self):

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


    def get_broadcast(self, n, b):

        n,b = self.process_broadcast(n,b)
        
        if self.state == 'Random_Tour':
            self.random_tour_listen(n,b)

        elif self.state == 'In_Place':
            self.in_place_listen(n,b)
            self.shape_unsuccessful_check()

        elif self.state == 'Root':
            self.root_listen(n,b)
            self.shape_unsuccessful_check()


    def find_available_spots(self, B):
        offers = []
        for i, message in enumerate(B):
            if self.shape_to_avoid == message.shape_ID:
                pass
            elif not message.child1_there and message.self_index != None:
                offers.append(i)

        return offers

    def random_tour_listen(self,N,B):

        # Store Neighborhood
        self.neighborhood = N
        rand = random.random()
        fragments_in_neighborhood = [1 for message in B if message.self_index != None]


        # === Become Root only if you receive no broadcasts ROOT ===========
        if len(fragments_in_neighborhood) == 0 and N.shape[0] > 2:
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
                self.vertices_covered.add(self.index)
                # Set state to root
                self.state = 'Root'


        #===== JOIN SHAPE ======
        # Find available offers
        available_spots = [i for i, message in enumerate(B) if not message.child1_there and message.self_index != None]
        available_spots = self.find_available_spots(B)
        # If there are no available offers len(offers) = 0 , thus probability you join is 0
        accepted_offer = self.evaluate_offers(available_spots, B)
        if accepted_offer[0]:

            # Find Parent: Pick an offer
            j = accepted_offer[1]
            self.shape_ID = B[j].shape_ID # Set ID of your shape
            self.root_index = B[j].root_index
            self.index = B[j].child1_index # Identify your own index in the graph
            self.vertices_covered.add(self.index)
            self.distance_to_root = distance_between_vertices(self.index, self.root_index, self.graph)
            # Read parents info from the offer and its relative position in the neighborhood
            self.parent_index = B[j].self_index
            self.parent_position = N[j]
            # Figure out child
            connections = np.argwhere(self.graph[self.index] > 0).T[0] # Check connections
            mask = connections != self.parent_index
            self.child1_index = int(connections[mask])
            self.child1_there = False
            self.vertices_covered = self.vertices_covered.union(B[j].vertices_covered)
            
            # Set State to InPlace
            self.state = 'In_Place'

        # Check what type of motion shoul;d follow
        self.check_new_shape(B,N)

        


    def in_place_listen(self,N,B):
        
        # Store Neighborhood
        self.neighborhood = N 
        # Assume your kid worn be there
        self.child1_there = False
        # Iterate over all messages
        for i,message in enumerate(B):

            # Check for new members in the shape
            if message.shape_ID == self.shape_ID:
                self.vertices_covered = self.vertices_covered.union(message.vertices_covered)


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

            


    def root_listen(self,N,B):
        # Store Neighborhood
        self.neighborhood = N

        self.child1_there = False
        self.child2_there = False
        for message in B:

            # Check for new members in the shape
            if message.shape_ID == self.shape_ID:
                self.vertices_covered = self.vertices_covered.union(message.vertices_covered)
            if message.self_index == self.child1_index:
                self.child1_there = True
            elif message.self_index == self.child2_index:
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
        # Init Variables for random tour
        self.tour_params = {'length':100, 'width':50, 'v':40}
        self.tour_history = []
        self.current_traj = None
        self.current_index = 0

    
    def inv_sigmoid(self,p0,sharpness,centroid,x):

        return p0*(1-(1/(1+np.exp(-sharpness*(x-centroid*self.shape.shape[0])))))
    
    def sigmoid(self,p0,p_end,sharpness,centroid,x):

        return p0 + p_end*((1/(1+np.exp(-sharpness*(x-centroid*self.shape.shape[0])))))
    

    def check_new_shape(self, B,N):

        for i, b in enumerate(B):

            if b.shape_ID != self.shape_to_avoid['ID'] and b.shape_ID != None:
                # Find Relative vecotr to shape centroid
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
            self.shape_to_avoid['counter'] +=1
            if self.shape_to_avoid['counter'] >= self.shape_to_avoid['counter']:
                print('yo')
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
            if random.random() > self.sigmoid(self.p_accept['p0'],
                                              1-self.p_accept['p0'], 
                                              self.p_accept['sharpness'],
                                              self.p_accept['center'],
                                              len(B[i].vertices_covered)):
                return (True,i)
            
        # None passed
        return (False, None)

        



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
        
        plt.scatter(a,result1)
        plt.scatter(a, result2)
        plt.show()

   
    


        

