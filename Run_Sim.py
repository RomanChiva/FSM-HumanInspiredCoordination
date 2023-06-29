import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
#plt.style.use('dark_background')



np.random.seed(123)

def run_sim(env):

    # Run the simulation and show an animation

    # Create figure
    fig = plt.figure()
    fig.suptitle('Decentralized Multi-Agent Template')
    # Create plot inside figure
    ax1 = fig.add_subplot(1,1,1)
    
    colors_index = {'Root':'blue', 'In_Place':'green', 'Random_Tour':'red'}

    def update(i):

        env.timestep()

        positions = env.make_positions_list()
        colors = [colors_index[agent['agent'].state] for agent in env.agents]
        ax1.clear()

        # Set Axis Limits
        ax1.set_xlim(-env.env_size, env.env_size)
        ax1.set_ylim(-env.env_size, env.env_size)

        # Set Ticks and Make Grid

        # Major ticks every 10, minor ticks every 5
        major_ticks = np.arange(-env.env_size, env.env_size,20)
        minor_ticks = np.arange(-env.env_size, env.env_size,10)

        ax1.set_xticks(major_ticks)
        ax1.set_xticks(minor_ticks, minor=True)
        ax1.set_yticks(major_ticks)
        ax1.set_yticks(minor_ticks, minor=True)

        # And a corresponding grid
        ax1.grid(which='both')

        # Plot Scatter
        ax1.scatter(positions.T[0], positions.T[1], c=colors)
        #ax1.scatter(positions.T[0], positions.T[1], s=150, c='red', alpha=0.3)
        for agent in env.agents:
            ax1.annotate(agent['agent'].ID, (agent['pos'][0], agent['pos'][1]))

    ani = animation.FuncAnimation(fig, update, interval = 100)
    plt.show()