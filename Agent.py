import numpy as np
import random
from utils import make_graph

class Agent:

    def __init__(self, shape, ID) -> None:
        
        # Global
        self.ID = ID
        self.shape = shape
        self.graph = make_graph(self.shape)

        # Shape Related
        self.parent = None #(Index, Pos)
        self.index = None
        self.child1 = None # (Index, Bool)
        # Only Relevant when you are Root, otherwise remains None
        self.child2 = None

        # Current State
        self.state = 'Random_Tour'
        self.neighborhood = None

        # Transition probabilities
        self.p_root = 0.001
        self.p_accept = 0.05
        self.p_give_up = 0.01

        # States for random tour
        self.rand_motion_duration = 10
        self.random_heading = (random.random()-0.5)*np.pi*2
        self.current_heading = (random.random()-0.5)*np.pi*2
        self.step_size = (self.random_heading - self.current_heading)/self.rand_motion_duration

    def move(self):

        if self.state == 'Random_Tour':
            v = self.random_tour()
        elif self.state == 'In_Place':
            v = self.in_place()
        elif self.state == 'Root':
            v = self.root()

        return v
    

    def RendezVous(self, neighborhood):
        
        import smallestenclosingcircle

        cx,cy,r = smallestenclosingcircle.make_circle(neighborhood)

        heading = np.array([cx,cy])

        v = heading/np.linalg.norm(heading)

        if random.random() < self.RV_RT:
            self.state = 'RandomTour'

        self.curr_n_size = neighborhood.shape[0]

        return v
    
    def random_tour(self):

        # Switch Heading roughly every 50 timesteps
        rand = random.random()

        if rand <= 1/self.rand_motion_duration:
            # Update new step size and rnadom motion duration
            self.random_heading = (random.random()-0.5)*np.pi*2
            self.step_size = (self.random_heading - self.current_heading)/self.rand_motion_duration
        
        # Update current heading
        self.current_heading += self.step_size

        v = np.array([np.cos(self.current_heading), np.sin(self.current_heading)])

        return v
    

    def in_place(self):
        # Identify target using graph indices and parent's position
        # target = parent relative position + (self_place_in_shape  - parent_place_in_shape)
        target = self.parent[1] + (self.shape[self.index] - self.shape[self.parent[0]])

        v = target/np.linalg.norm(target)
        return v


    def root(self):
        # The root drone doesn't move
        v = np.array([0,0])
        return v
        


    def send_broadcast(self):
        # Info contained in broadcasted message depending on state
        if self.state == 'Root':

            return [self.index, self.child1],[self.index,self.child2]
        elif self.state == 'In_Place':
            return [self.index, self.child1]
        else:
            return None
                
    
    def get_broadcast(self, n, b):

        if self.state == 'Random_Tour':
            self.random_tour_listen(n,b)

        elif self.state == 'In_Place':
            self.in_place_listen(n,b)

            self.shape_unsuccessful_check()

        elif self.state == 'Root':
            self.root_listen(n,b)
            self.shape_unsuccessful_check()



        
    def shape_unsuccessful_check(self):

        # Condition for breaking out of formation as child of root
        if self.state == 'In_Place':
            if not self.child[1]:
                rand = random.random()
                if rand < self.p_give_up:
                    self.state = 'Random_Tour'

        # Condition for giving up as root
        else:
            if not self.child1[1] and not self.child2[1]:
                rand = random.random()
                if rand < self.p_give_up:
                    self.state = 'Random_Tour'




        

    def random_tour_listen(self,n,b):

        # Store Neighborhood
        self.neighborhood = n 

        # ===========
        # TRANSITIONS
        # ===========
        rand = random.random()

        # === ROOT ===========
        if len(b) == 0 and len(n) != 0:
            if rand < self.p_root:
                
                # Pick Root Vertex/random select
                self.index = random.randint(0,range(self.shape.shape[0]))
                # Identify Children
                children = np.argwhere(self.graph[self.index] > 0)
                
                self.child1 = (children[0],False)
                self.child2 = (children[1],False)

                self.state = 'Root'


        #===== JOIN SHAPE ======

        self.offers = []

        for i,message in enumerate(b):
            if message[1][1]:
                self.offers.append(i)
        # If there are no available offers len(offers) = 0 , thus probability you join is 0
        if rand < self.p_accept*len(self.offers):

            # Find Parent: Pick an offer
            j = random.choice(self.offers)
            # Read parents info from the offer and its relative position in the neighborhood
            self.parent = (b[j][0],n[j])

            # Figure out child
            self.index = b[j][0] # Identify your own index in the graph
            connections = np.argwhere(self.graph[self.ID] > 0) # Check connections
            child_index = connections[connections != self.parent[0]]
            self.child1 = (child_index, False)

            self.state == 'in_place'




    def in_place_listen(self,n,b):
        # Store Neighborhood
        self.neighborhood = n 
        rand = random.random()

        self.child1[1] = False

        for i,message in enumerate(b):

            if message[0] == self.parent[0]:
                self.parent[1] == n[i]

            elif message[0] == self.child[0]:
                self.child[1] = True
            # Check for yourself:
            elif message[0] == self.index and rand < 0.5:
                self.state = 'Random Tour'
            else:
                pass

            


    def root_listen(self,n,b):
        # Store Neighborhood
        self.neighborhood = n 

        self.child1[1] = False
        self.child2[1] = False

        for message in b:
            if message[0] == self.child1[0]:
                self.child1[1] = True
            if message[0] == self.child2[0]:
                self.child2[1] = True



        

