import numpy as np
import matplotlib.pyplot as plt

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

# Example usage
center = np.array([0,0])
width = 5
rbf_nav = RadialBasisFunction(center, width)
current_position = np.array([-2,0])
potential = rbf_nav.evaluate(current_position)
gradient = rbf_nav.gradient(current_position)
velocity = -gradient  # Adjust the scaling and sign as per your needs

print("Potential:", potential)
print("Gradient:", gradient)
print("Velocity:", velocity)