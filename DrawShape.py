import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from matplotlib import cm
import copy

def draw_shape(canvas_size, pixel_size):

    # Create a canvas with a white background
    canvas = 255 * np.ones((canvas_size[0], canvas_size[1]), dtype=np.uint8)
    drawing = False

    def draw_shape(event, x, y, flags, param):

          # true if mouse is pressed

        global drawing
        

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(canvas, (x, y), pixel_size, 0, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    # Create a window and set the mouse callback function
    cv2.namedWindow('Draw Shape')
    cv2.setMouseCallback('Draw Shape', draw_shape)

    while True:
        cv2.imshow('Draw Shape', canvas)

        key = cv2.waitKey(1) & 0xFF

        # Press 'r' to reset the canvas
        if key == ord('r'):
            canvas = 255 * np.ones((canvas_size[0], canvas_size[1]), dtype=np.uint8)

        # Press 'q' to quit and obtain the array
        elif key == ord('q'):
            break

    # Convert the canvas to a binary array (0s and 1s)
    binary_array = (canvas == 0).astype(int)

    plt.imshow(binary_array, cmap='gray')
    plt.show()

    cv2.destroyAllWindows()
    return binary_array


# Blur image using mean kernel
def blur(array, kernel_size):
    blurred_array = cv2.blur(array, (kernel_size, kernel_size))
    return blurred_array

def resize_image(pic, size):
    pic = cv2.resize(pic, size, interpolation=cv2.INTER_NEAREST)
    return pic



def Make_RBF(pic, kernel_size,size_shrink,size_game_scale, epsilon, function):

    # Get Raw Pic and BLUR
    pic = pic.astype(np.uint8)*255
    
    pic = resize_image(pic, (size_shrink, size_shrink))
    plt.imshow(pic, cmap='gray')
    plt.show()
    # Scale Pic Down to manageable ammount of points for RBF Interpolation
    pic = blur(pic, kernel_size)
    plt.imshow(pic, cmap='gray')
    plt.show()

    # Get indices and values of non-zero points
    indices = np.argwhere(pic != 0)
    values = copy.deepcopy(pic[indices[:, 0], indices[:, 1]])

    # Shift indices so origin coincides with the center of the image array
    indices = indices - size_shrink/2

    # Scale indices to size they should be in the game
    size_up = size_game_scale/size_shrink
    print(size_up)
    indices = indices*(size_game_scale/size_shrink)
    
    # Make RBF
    x = indices[:, 0]
    y = indices[:, 1]
    z = -1*values/255

    print(x.shape)
    rbf = Rbf(x, y, z, epsilon=epsilon, function=function, smooth=0.0)

    return rbf

def show_rbf(RBF,size,z_lim):


    xx = np.linspace(-size, size, 1000)
    yy = np.linspace(-size, size, 1000)
    xx, yy = np.meshgrid(xx, yy)
    zz = RBF(xx, yy)


    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    plt.imshow(zz, cmap=cm.jet, origin='lower', extent=[-size, size, -size, size])
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='viridis')
    # Set range of z axis from 0 to 5
    ax.set_zlim(0, z_lim)
    plt.show()




# Take derivative of RBF at a point and return vector of direction speepest descent

# Find gradient of RBF at a point
def gradient_RBF(RBF, point, step_size):

    # Get x and y coordinates of point
    x = point[0]
    y = point[1]

    # Get value of RBF at point
    z = RBF(x, y)

    # Get value of RBF at point shifted by step size in x direction
    z_dx = RBF(x + step_size, y)

    # Get value of RBF at point shifted by step size in y direction
    z_dy = RBF(x, y + step_size)

    # Calculate partial derivatives
    dz_dx = (z_dx - z) / step_size
    dz_dy = (z_dy - z) / step_size

    # Return gradient vector
    return np.array([dz_dx, dz_dy])




pic = draw_shape((500, 500), 15)
rbf = Make_RBF(pic, 3, 30, 200, 5, 'gaussian')
show_rbf(rbf,200,5)

derivative = gradient_RBF(rbf, (0,0, 0), 0.2)
print(derivative)
