import numpy as np


class message:

    def __init__(self,ID,root_index, self_index, child1_index,child1_there,VC) -> None:

        # Broadcast the index of your root
        self.shape_ID = ID
        self.root_index = root_index
        # Broadcast What index you are in the grap
        self.self_index = self_index
        self.vertices_covered = VC
        # Children 
        self.child1_index = child1_index
        self.child1_there = child1_there
        
        