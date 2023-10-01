import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path
import random
import string
import scipy.stats as stats
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
from scipy.stats import mode
from scipy.stats import lognorm
# Funcs to turn point cloud into a graph

def make_undirected(AM):

    # For some dumbass reason the scipy MST spits out a directed graph

    n = AM.shape[0]
    symm = np.zeros((n,n))
    
    for i in range(n):
        for j in range(i):
            if AM[i,j] or AM[j,i] != 0:
                symm[i,j] = 1
                symm[j,i] = 1
            else:
                symm[i,j] == symm[j,i] == 0

    return symm


def circle_maker(n_agents, radius):

    root = np.linspace(0,2*np.pi, n_agents+1)
    x = np.sin(root)*radius
    y = np.cos(root)*radius

    resultant = np.array([x,y]).T
    return resultant[:-1]




def square_maker(width, height, points_on_width, points_on_height):
    # Calculate half of the width and height to position the rectangle centered at the origin (0, 0)
    half_width = width / 2.0
    half_height = height / 2.0

    # Generate points along the width and height of the rectangle
    width_points = np.linspace(-half_width, half_width, points_on_width)
    height_points = np.linspace(-half_height, half_height, points_on_height)

    # Calculate points on the corners
    top_left_corner = np.array([width_points[0], half_height])
    top_right_corner = np.array([width_points[-1], half_height])
    bottom_right_corner = np.array([width_points[-1], -half_height])
    bottom_left_corner = np.array([width_points[0], -half_height])

    # Combine the points to form the outline of the rectangle
    rectangle_outline = np.vstack((
        np.column_stack((width_points[1:], np.full(points_on_width - 1, half_height))),
        np.column_stack((np.full(points_on_height - 1, half_width), height_points[:-1][::-1])),
        np.column_stack((width_points[:-1][::-1], np.full(points_on_width - 1, -half_height))),
        np.column_stack((np.full(points_on_height - 1, -half_width), height_points[1:])),
    ))

    return rectangle_outline, rectangle_outline.shape[0]




def make_graph(shape):
    fully_connected = sp.spatial.distance_matrix(shape, shape)
    mst = sp.sparse.csgraph.minimum_spanning_tree(fully_connected)
    mst = mst.toarray()

    return make_undirected(mst)


def show_graph(shape,graph):

    plt.scatter(shape[:,0],shape[:,1])
    
    for i,row in enumerate(graph):
        ind = np.nonzero(row)[0]
        for j in ind:
            x = np.array([shape[i,0],shape[j,0]])
            y = np.array([shape[i,1],shape[j,1]])
            plt.plot(x,y)
    plt.show()





def create_adjacency_matrix(num_points):
    # Create an identity matrix
    adjacency_matrix = np.eye(num_points)

    # Set zeros in adjacent diagonals
    adjacency_matrix += np.diag(np.ones(num_points - 1), k=1)
    adjacency_matrix += np.diag(np.ones(num_points - 1), k=-1)

    adjacency_matrix = adjacency_matrix - np.eye(num_points)
    adjacency_matrix[0,num_points-1] = 1
    adjacency_matrix[num_points-1,0] = 1

    return adjacency_matrix




def distance_between_vertices(v1, v2, adjacency_matrix):
    # Calculate the shortest path lengths between all pairs of vertices
    shortest_distances = shortest_path(adjacency_matrix, directed=False)

    # Retrieve the distance between the two vertices
    distance = shortest_distances[v1, v2]

    return distance
            
## Connected position generation

def connected_pos_gen(env_size, n_agents, r, CENTERED=True):
        
    if CENTERED:
        pos_list = np.array([np.array([0,0], dtype=float)])
    else:
        pos_list = (np.random.random((1,2))-0.5)*2*env_size

    while pos_list.shape[0] < n_agents:

        new_random_point = (np.random.random((1,2))-0.5)*2*env_size

        relative_distances = pos_list - new_random_point
        distances = np.linalg.norm(relative_distances, axis=1)

        if np.any(distances <= r):
            
            pos_list = np.concatenate((pos_list,new_random_point), axis=0)
    
    return pos_list


def generate_random_string(length):
    letters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(letters) for _ in range(length))



def custom_sigmoid(p0,sharpness,centroid,x):

    return p0*(1-(1/(1+np.exp(-sharpness*(x-centroid)))))


def generate_lobe_trajectory(length, width, num_points, orientation):
    angles = np.linspace(0, 2 * np.pi, num_points)
    
    x_coords = (length/2)*np.cos(angles)*-1 + length/2
    y_coords = np.sin(angles)* width/2
    
    result = np.column_stack((x_coords,y_coords))

    angle = orientation*-1
    # Create a rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])


    # Apply the rotation to the translated trajectory
    result = np.dot(result, rotation_matrix)

    return result


def perpendicular_gen(start_angle_radians):
    angles_radians = [start_angle_radians]
    for _ in range(3):
        start_angle_radians += np.pi / 2
        angles_radians.append(start_angle_radians % (2 * np.pi))  # Ensure angles stay within the range [0, 2π)
    return angles_radians



def find_furthest_value(values):
    values = np.array(values)
    candidate_values = np.linspace(-np.pi,np.pi, num=100)  # Adjust the 'num' parameter as needed
    
    distances = np.abs(candidate_values[:, np.newaxis] - values)  # Calculate distances for all candidate values
    min_distances = np.min(distances, axis=1)  # Find the minimum distances
    
    furthest_index = np.argmax(min_distances)  # Get the index of the furthest value
    furthest_value = candidate_values[furthest_index]  # Retrieve the furthest value
    
    return furthest_value


def Centroid(shape):
    return np.mean(shape, axis=0)


def maximum_distance(points,centroid):
    
    relative = points - centroid
    dist = np.linalg.norm(relative,axis=1)
    # Find the maximum distance
    max_distance = np.max(dist)
    return max_distance

def weib(x,n,a):
    return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)



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

def fit_lognormal_and_plot_histogram(data, bins=15, plot=False, Print=False):

    num_bins, bin_width = create_optimal_histogram(data, plot=False)
    num_bins = max(num_bins,bins)
    # Create a histogram of the data
    counts, bin_edges = np.histogram(data, bins=num_bins, density=True)

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit a lognormal distribution to the histogram centroids
    def lognormal_fit(x, mu, sigma):
        return lognorm.pdf(x, sigma, scale=np.exp(mu))

    initial_params = [0, 1]  # Initial guess for mean and standard deviation
    fitted_params, _ = curve_fit(lognormal_fit, bin_centers, counts, p0=initial_params)

    if plot:
        # Plot the histogram
        plt.hist(data, bins=num_bins, density=True, alpha=0.5, label='Histogram')

        # Plot the fitted lognormal distribution
        x_range = np.linspace(min(data), max(data), 1000)
        plt.plot(x_range, lognormal_fit(x_range, *fitted_params), 'r-', label='Lognormal Fit')

        # Add labels and legend
        plt.xlabel('Game Duration')
        plt.ylabel('Probability')
        plt.legend()
        plt.suptitle('Convergence Time Distrbution')

        # Show the plot
        plt.show()
    if Print:
        print({'mu': fitted_params[0], 'sigma': fitted_params[1]})

    # Return the parameters of the fitted lognormal distribution
    return {'mu': fitted_params[0], 'sigma': fitted_params[1]}


def filt_above(input_list, threshold):
    # Use list comprehension to filter out values above the threshold
    filtered_list = [x for x in input_list if x <= threshold]
    
    # Calculate the number of removed values
    removed_count = len(input_list) - len(filtered_list)
    
    return filtered_list#, removed_count



def plot_dual_y_axis(x, y1, y2,x_label, label1, label2, title, y1_lower_limit, y1_upper_limit, y2_lower_limit, y2_upper_limit):
    # Create a figure and the first axes (left y-axis)
    fig, ax1 = plt.subplots()

    # Plot the first series on the left y-axis
    ax1.plot(x, y1, color='b', label=label1)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(label1, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(y1_lower_limit, y1_upper_limit)  # Set the limits for the left y-axis

    # Create a second axes (right y-axis) sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot the second series on the right y-axis
    ax2.plot(x, y2, color='r', label=label2)
    ax2.set_ylabel(label2, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(y2_lower_limit, y2_upper_limit)  # Set the limits for the right y-axis

    # Adding legends for both series
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, loc='upper left')
    plt.grid()
    plt.title(title)
    plt.tight_layout()
    plt.savefig('figs/'+x_label+'2'+'.png')
    plt.show()