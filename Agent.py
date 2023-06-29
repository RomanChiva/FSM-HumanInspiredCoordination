import numpy as np
import random
from utils import make_graph
np.random.seed(123)
class Agent:

    def __init__(self, shape, ID) -> None:
        
        # Global
        self.ID = ID # Only for Visualization and Debug
        self.shape = shape
        self.graph = make_graph(self.shape)

        # Shape Related
        self.parent = None #[Index, Pos]
        self.index = None
        self.child1 = None # [Index, Bool]
        # Only Relevant when you are Root, otherwise remains None
        self.child2 = None

        # Current State
        self.state = 'Random_Tour'
        self.neighborhood = None

        # Transition probabilities
        self.p_root = 0.01
        self.p_accept = 0.05
        self.p_give_up = 0.01
        self.p_give_up_root = 0.1

        # States for random tour
        self.rm_v = 2
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

        return v*self.rm_v
    

    def in_place(self):
        # Identify target using graph indices and parent's position
        # target = parent relative position + (self_place_in_shape  - parent_place_in_shape)
        parent_relative_pos = self.parent[1]  
        from_parent_to_target = (self.shape[self.index] - self.shape[self.parent[0]])
        resultant = parent_relative_pos-from_parent_to_target
        return resultant*0.1


    def root(self):
        # The root drone doesn't move
        v = np.array([0,0])
        return v
        


    def send_broadcast(self):
        # Info contained in broadcasted message depending on state
        if self.state == 'Root':
            return [self.index, self.child1,self.index,self.child2]
        elif self.state == 'In_Place':
            if self.child1 == None:
                # Signal virtual agent -1 is already occupyiong this place
                return [self.index,[-1,True]]
            else:
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
        if self.state == 'In_Place' and self.child1:
            if not self.child1[1]:
                rand = random.random()
                if rand < self.p_give_up:
                    self.state = 'Random_Tour'

        # Condition for giving up as root
        elif self.state == 'Root':

            if self.child2 is not None:
                if not self.child1[1] and not self.child2[1]:
                    rand = random.random()
                    if rand < self.p_give_up_root*2:
                        self.state = 'Random_Tour'

            else:
                if not self.child1[1]:
                    rand = random.random()
                    if rand < self.p_give_up:
                        self.state = 'Random_Tour'




        

    def random_tour_listen(self,N,B):

        # Store Neighborhood
        self.neighborhood = N

        n,b = self.format_raw_broadcast(N,B)
        
        # ===========
        # TRANSITIONS
        # ===========
        rand = random.random()

        # === ROOT ===========
        if len(b) == 0:
            if rand < self.p_root:
                
                # Pick Root Vertex/random select
                self.index = random.randint(0,self.shape.shape[0]-1)
                # Identify Children
                children = np.argwhere(self.graph[self.index] > 0)
                
                self.child1 = [children[0][0],False]
                if children.shape[0] >1:
                    self.child2 = [children[1][0],False]
               
                self.state = 'Root'


        #===== JOIN SHAPE ======

        self.offers = []
        for i,message in enumerate(b):
            if not message[1][1]:
                self.offers.append(i)

        # If there are no available offers len(offers) = 0 , thus probability you join is 0
        if rand < self.p_accept*len(self.offers):

            # Find Parent: Pick an offer
            j = random.choice(self.offers)
            self.index = b[j][1][0] # Identify your own index in the graph
            
            # Read parents info from the offer and its relative position in the neighborhood
            self.parent = [b[j][0],n[j]]
            
            # Figure out child
            connections = np.argwhere(self.graph[self.index] > 0).T[0] # Check connections
            if connections.shape[0] <2:
                self.child1 = None
            else:
                child_index = connections[connections != self.parent[0]][0]
                self.child1 = [child_index, False]
            
            self.state = 'In_Place'




    def in_place_listen(self,N,B):
        
        # Store Neighborhood
        self.neighborhood = N 

        n,b = self.format_raw_broadcast(N,B)
        if self.child1 != None:
            self.child1[1] = False
        for i,message in enumerate(b):
            if message[0] == self.parent[0]:
                self.parent[1] = n[i]
            # Check for yourself:
            elif message[0] == self.index: #and random.random() < 0.5:
                self.state = 'Random_Tour'
            # Check if if you have children
            if self.child1 != None:
                if message[0] == self.child1[0]:
                    self.child1[1] = True

            


    def root_listen(self,N,B):
        # Store Neighborhood
        self.neighborhood = N

        n,b = self.format_raw_broadcast(N,B)

        self.child1[1] = False
        for message in b:
            if message[0] == self.child1[0]:
                self.child1[1] = True

        if self.child2 is not None:
            self.child2[1] = False
            for message in b:
                if message[0] == self.child2[0]:
                    self.child2[1] = True


    def format_raw_broadcast(self, N,B):

        
        
        b = []
        n = []

        for i, message in enumerate(B):
            if message is not None:
                if len(message) > 2:
                    b.append(message[:2])
                    n.append(N[i])
                    if message[3] is not None:
                        b.append(message[2:])
                        n.append(N[i])
                else:
                    b.append(message)
                    n.append(N[i])

        return np.array(n),b



        

