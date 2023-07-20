import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path
import random
import string

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
        pos_list = np.array([np.array([0,0])])
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
