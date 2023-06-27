import numpy as np
import random


class Agent:

    def __init__(self) -> None:
        
        # Global
        self.t_step = 0

        # Current State
        self.state = 'RandomTour'

        # Transition variables 
        self.curr_n_size = 0

        self.RV_RT = 0.0025

        # States for random tour
        self.rand_motion_duration = 10
        self.random_heading = (random.random()-0.5)*np.pi*2
        self.current_heading = (random.random()-0.5)*np.pi*2
        self.step_size = (self.random_heading - self.current_heading)/self.rand_motion_duration

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

        if self.curr_n_size < neighborhood.shape[0]:
            self.curr_n_size = neighborhood.shape[0]
            self.state = 'RendezVous'

        self.curr_n_size = neighborhood.shape[0]
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


        

