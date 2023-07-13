import numpy as np
import random
from Potential import RadialBasisFunction

class Agent:

    def __init__(self) -> None:
        
        # Global
        self.t_step = 0

        # Current State
        self.state = 'RandomTour'

        # Transition variables 
        self.curr_n_size = 0

        # Random Tour States
        self.target_count = 0
        self.RBF_center = np.array([0,0], dtype=float)
        self.relative_to_target = np.array([0,0],dtype=float)
        self.RBF = RadialBasisFunction(self.RBF_center,1)

    def move(self, neighborhood):

        self.t_step +=1


        if self.state == 'RendezVous':
            v = self.RendezVous(neighborhood)
        elif self.state == 'RandomTour':
            v = self.random_tour(neighborhood)

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
    
    def random_tour(self, neighborhood):
      
        current_value = self.RBF.evaluate(self.relative_to_target)
        print(current_value)
        # Check if you have to generate a new RBF
        if current_value > 0.9:
            self.target_count += 1
            # Generate random vector and set it relative to origin
            rbf_width = self.target_count*10
            rand = (np.random.rand(1,2)*2-1)*rbf_width
            self.RBF_center = rand -self.relative_to_target
            # Define the new RBF
            self.RBF = RadialBasisFunction(self.RBF_center,rbf_width)
            print('NewRBF')
            
        # Use gradient to get a velocity_vector 
        grad = self.RBF.gradient(self.relative_to_target)
        # Velocity Range (1 to 10) (Assumin g gradient vector magnitudes range from 0 to 0.15 approximtely)
        v = -grad[0]*(1/np.linalg.norm(grad[0]) + 60)
        print(v)
        self.relative_to_target += v

        return v
    


        

