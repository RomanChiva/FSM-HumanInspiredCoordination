import numpy as np

class RadialBasisFunction:
    def __init__(self, center, width):
        
        self.center = center
        self.width = width


    def activation(self, pos):
        return np.exp(-0.5 * (pos / self.width) ** 2)

    def evaluate(self, position):
        distance = np.linalg.norm(position - self.center)
        return self.activation(distance)

    def gradient(self, position):
        
        diff = position - self.center
        distance = np.linalg.norm(diff)
        activation = self.activation(distance)
        gradient = (activation / self.width**2) * diff
        return gradient