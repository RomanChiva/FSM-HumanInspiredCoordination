import numpy as np


class generate_agent_parameters():

    def __init__(self,sensing_radius,spread) -> None:

        # Transition Probs (Params, 0)
        self.p_r_0 = 0.1
        self.p_r_GU = 0.02

        self.p_a_0 = 0.05
        self.p_a_s = 3
        self.p_a_c = 0.2

        self.p_GU_0 = 0.05
        self.p_GU_s = 0.7
        self.p_GU_c = 0.3

        # Tour Params
        self.init_L = int(np.random.normal(1.3*spread*sensing_radius, 10, 1))#100)
        self.init_W = int(np.random.normal(0.75*spread*sensing_radius,5,1))#60
        self.init_D = int(np.random.normal(0.75*spread*sensing_radius,5,1))#60#60
        self.init_step = 5

        self.orbit_D = 25
        self.orbit_step = 0

        self.exp_L = int(np.random.normal(2*spread*sensing_radius, 10, 1))#120
        self.exp_W = int(np.random.normal(1.2*spread*sensing_radius,5,1))#75
        self.exp_D = int(np.random.normal(spread*sensing_radius,5,1))#60#60
        self.exp_step = 10

        self.patience = 60

        # Repulsion Gains
        self.tour_gain = 1.5
        self.in_place_gain = 0.3
        self.root_gain = 0.3


    def return_dictionary(self):

        return vars(self)
    
    def input_dictionary(self,input_dict):

        for key, value in input_dict.items():
            setattr(self,key,value)





