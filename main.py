import matplotlib.pyplot as plt
from Environment import Env
from Run_Sim import viz_sim
import numpy as np
import sys
from utils import circle_maker, square_maker
from gen_params import generate_agent_parameters

if __name__ == '__main__':
    # Shape we wish to from: Points MUST BE ORDERED!!! 
    circle = circle_maker(15,40)
    rectangle, n_agents = square_maker(100,100,7,7)

    # Import Agent Parameters
    parameters = generate_agent_parameters()

    # DEFINE ENVIRONMENT: Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
    env = Env(100,circle,15,35,parameters,CONNECTED=True)




    def create_optimal_histogram(data, plot=True):
        # Calculate the interquartile range (IQR)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1

        # Calculate the Freedman-Diaconis bin width
        num_data_points = len(data)
        bin_width = 2 * iqr / (num_data_points ** (1/3))

        # Calculate the number of bins
        data_range = max(data) - min(data)
        num_bins = int(data_range / bin_width) + 1

        # Plot the histogram if requested
        if plot:
            plt.hist(data, bins=num_bins, edgecolor='black')
            plt.xlabel('Data')
            plt.ylabel('Frequency')
            plt.title('Optimal Histogram')
            plt.show()

        return num_bins, bin_width

    viz_sim(env, margins=100)
    sys.exit()
    # Run Numerically
    loss_list = [env.run_sim() for x in range(50)]
    print(loss_list)

    # Example usage:

    create_optimal_histogram(loss_list)

    # Visualize Game
    

    

    
    






        
    

