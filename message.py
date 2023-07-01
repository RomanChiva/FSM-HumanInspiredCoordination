import numpy as np


class message:

    def __init__(self,root_index, self_index, child1_index,child1_there) -> None:

        # Broadcast the index of your root
        self.root_index = root_index
        # Broadcast What index you are in the graph
        self.self_index = self_index

        # Children 
        self.child1_index = child1_index
        self.child1_there = child1_there
        
        