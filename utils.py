import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



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

            


