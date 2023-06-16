import numpy as np




class Agent:

    def __init__(self) -> None:
    
        self.state = 0

    def move(self, neighborhood):



        if neighborhood.shape[0] > 0:
            v = self.RendezVous(neighborhood)
        else:
            v =  (np.random.random((1,2))-0.5)*2*5

        return v
    
        print('ryhds')

    def RendezVous(self, neighborhood):
        
        import smallestenclosingcircle

        cx,cy,r = smallestenclosingcircle.make_circle(neighborhood)

        heading = np.array([cx,cy])

        v = heading/np.linalg.norm(heading)

        return v