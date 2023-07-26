import numpy as np


class generate_agent_parameters():

    def __init__(self) -> None:

        # Transition Probs
        self.p_r_0 = 0.05
        self.p_r_GU = 0.02

        self.p_a_0 = 0.3
        self.p_a_s = 0.5
        self.p_a_c = 0.5

        self.p_GU_0 = 0.07
        self.p_GU_s = 0.7
        self.p_GU_c = 0.3

        # Tour Params
        self.init_L = 100
        self.init_W = 30
        self.init_D = 50
        self.init_step = 5

        self.orbit_D = 25
        self.orbit_step = 0

        self.exp_L = 140
        self.exp_W = 30
        self.exp_D = 50
        self.exp_step = 5

        self.patience = 100