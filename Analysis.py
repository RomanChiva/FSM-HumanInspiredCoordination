import numpy as np
import matplotlib.pyplot as plt
from Environment import Env
from utils import circle_maker, square_maker, create_optimal_histogram, fit_lognormal_and_plot_histogram, filt_above, plot_dual_y_axis
from gen_params import generate_agent_parameters
import pickle
import copy
import sys

def sensitivity_plot(param, lower, upper, steps, n_runs):

    ###################### GAME PARAMS ####################
    sensing_radius = 30
    spread = 2
    shape_main_axis=1
    AR = 1

    circle = circle_maker(10,30)


    param_values = np.linspace(lower,upper,steps)

    mu = []
    sigma = []
    times = []

    for value in param_values:

        # Import Agent Parameters
        parameters = generate_agent_parameters(sensing_radius, spread)
        p_dict = parameters.return_dictionary()
        p_dict[param] = value
        parameters.input_dictionary(p_dict)

        # DEFINE ENVIRONMENT: Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
        env = Env(100,circle,10,sensing_radius,parameters,spread*sensing_radius)
        #####################################################################
        
        times.append([env.run_sim() for x in range(n_runs)])
        
        
        
        #results = fit_lognormal_and_plot_histogram(t,bins=30)

        #mu.append(results['mu'])
        #sigma.append(results['sigma'])

    file_path = 'SA/SA_' + param +'.pkl'
    
    with open(file_path, "wb") as file:
        pickle.dump((param_values,times), file)
    



def sensitivity_spread(lower, upper, steps, n_runs):

    ###################### GAME PARAMS ####################
   
    param_values = np.linspace(lower,upper,steps)
    
    times = []
    for value in param_values:

        sensing_radius = 30
        spread = value
       

        circle = circle_maker(10,30)
        

        # Import Agent Parameters
        parameters = generate_agent_parameters(sensing_radius, spread)

        # DEFINE ENVIRONMENT: Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
        env = Env(100,circle,10,sensing_radius,parameters,spread*sensing_radius)
        #####################################################################
        
        times.append([env.run_sim() for x in range(n_runs)])
        
        print('Sensitivity_spread:',str(value), ' Done ',times[-1])
        #results = fit_lognormal_and_plot_histogram(t,bins=30)

        #mu.append(results['mu'])
        #sigma.append(results['sigma'])

    file_path = 'SA/SA_' + 'spreadC1030' +'.pkl'
    
    with open(file_path, "wb") as file:
        pickle.dump((param_values,times), file)



def sensitivity_agent_distance(lower, upper, steps, n_runs):

    ###################### GAME PARAMS ####################
   
    param_values = np.linspace(lower,upper,steps)
    times = []
    for value in param_values:

        sensing_radius = 30
        spread = 2
        n_ag = 10
        iner_ag_dist = value
        r = iner_ag_dist*n_ag/(2*np.pi)
      

        circle = circle_maker(n_ag,r)

        

        # Import Agent Parameters
        parameters = generate_agent_parameters(sensing_radius, spread)

        # DEFINE ENVIRONMENT: Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
        env = Env(100,circle,10,sensing_radius,parameters,spread*sensing_radius)
        #####################################################################
        
        times.append([env.run_sim() for x in range(n_runs)])
        print('Sensitivity_Dist:',str(value), ' Done ',times[-1])
        
        #results = fit_lognormal_and_plot_histogram(t,bins=30)

        #mu.append(results['mu'])
        #sigma.append(results['sigma'])

    file_path = 'SA/SA_' + 'ag_distC10' +'.pkl'
    
    with open(file_path, "wb") as file:
        pickle.dump((param_values,times), file)



def sensitivity_n_agents(lower, upper, steps, n_runs):

    ###################### GAME PARAMS ####################
   
    param_values = np.linspace(lower,upper,steps)
    print(param_values)
    times = []
    for value in param_values:

        sensing_radius = 30
        spread = 2
        n_ag = int(value)
        iner_ag_dist = 6*np.pi
        r = iner_ag_dist*n_ag/(2*np.pi)
      

        circle = circle_maker(n_ag,r)


        # Import Agent Parameters
        parameters = generate_agent_parameters(sensing_radius, spread)

        # DEFINE ENVIRONMENT: Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
        env = Env(100,circle,n_ag,sensing_radius,parameters,spread*sensing_radius)
        #####################################################################
        
        times.append([env.run_sim() for x in range(n_runs)])
        print('Sensitivity_N_ag:',str(value), ' Done ',times[-1])
        
        #results = fit_lognormal_and_plot_histogram(t,bins=30)

        #mu.append(results['mu'])
        #sigma.append(results['sigma'])

    file_path = 'SA/SA_' + 'n_ag6pi' +'.pkl'
    
    with open(file_path, "wb") as file:
        pickle.dump((param_values,times), file)



def sensitivity_rectangle_n_agents(lower, upper, steps, n_runs):

    ###################### GAME PARAMS ####################
   
    param_values = np.linspace(lower,upper,steps)
    print(param_values)
    times = []
    for value in param_values:

        sensing_radius = 30
        spread = 2

        size = 15*value

        rectangle,n_ag = square_maker(size,size,int(value),int(value))

       

        # Import Agent Parameters
        parameters = generate_agent_parameters(sensing_radius, spread)

        # DEFINE ENVIRONMENT: Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
        env = Env(100,rectangle,n_ag,sensing_radius,parameters,spread*sensing_radius)
        #####################################################################
        
        times.append([env.run_sim() for x in range(n_runs)])
        print('Sensitivity_RectNAg:',str(value), ' Done ',times[-1])
        
        #results = fit_lognormal_and_plot_histogram(t,bins=30)

        #mu.append(results['mu'])
        #sigma.append(results['sigma'])

    file_path = 'SA/SA_' + 'rect_n_ag' +'.pkl'
    
    with open(file_path, "wb") as file:
        pickle.dump((param_values,times), file)


def sensitivity_rectangle_spread(lower, upper, steps, n_runs):

    ###################### GAME PARAMS ####################
   
    param_values = np.linspace(lower,upper,steps)
    times = []
    for value in param_values:

        sensing_radius = 30
        spread = value
        
        inter_agent_dist = 15
        ag_side = 4

        rectangle,n_ag = square_maker(inter_agent_dist*ag_side,inter_agent_dist*ag_side,4,4)

        

        # Import Agent Parameters
        parameters = generate_agent_parameters(sensing_radius, spread)
      

        # DEFINE ENVIRONMENT: Env SIze, shape, N Agents, Nbrhd Radius, Spawn as Connected Graph
        env = Env(100,rectangle,n_ag,sensing_radius,parameters,spread*sensing_radius)
        #####################################################################
        
        times.append([env.run_sim() for x in range(n_runs)])
        print('Sensitivity_RektSpred:',str(value), ' Done ',times[-1])
        
        #results = fit_lognormal_and_plot_histogram(t,bins=30)

        #mu.append(results['mu'])
        #sigma.append(results['sigma'])

    file_path = 'SA/SA_' + 'rect_spred' +'.pkl'
    
    with open(file_path, "wb") as file:
        pickle.dump((param_values,times), file)



# sensitivity_plot('p_r_0',0.01,0.8,50,400)
# sensitivity_plot('p_r_GU',0.005,0.4,40,400)
# sensitivity_plot('p_a_0',0.01,0.5,40,400)
# sensitivity_plot('p_a_s',0.5,5,20,400)
# sensitivity_plot('p_a_c',0.01,0.5,40,400)
# sensitivity_plot('p_GU_0',0.01,0.5,50,400)
# sensitivity_plot('p_GU_s',0.2,5,40,400)
# sensitivity_plot('p_GU_c',0.1,0.7,40,400)
# sensitivity_plot('patience',5,200,60,400)
    

# sensitivity_spread(0.5,5,50,100)
# sensitivity_agent_distance(2,25,50,100)
# sensitivity_n_agents(5,50,46,100)
# sensitivity_rectangle_n_agents(3,15,13,100)
# sensitivity_rectangle_spread(0.5,5,50,100)

def make_plot(param_values,times, cutoff):

    # Remove all values 2500 or greater
    times = [filt_above(time,cutoff) for time in times] 

    mean = np.array([np.mean(np.array(time)) for time in times])
    std = np.array([np.std(np.array(time)) for time in times])

    plt.plot(param_values,mean)
    plt.fill_between(param_values,mean-std,mean+std,alpha=0.2)
    plt.xlabel(param)
    plt.ylabel('timesteps')
    plt.show()
    

def make_histogram(times):


    # Flatten the list of lists
    #t = [item for sublist in times for item in sublist]
    t = np.array(times)
    # Natural logarithm of the data
    #t = np.log(t)

    # Create the optimal histogram
    #num_bins, bin_width = create_optimal_histogram(t, plot=True)

    # Fit a lognormal distribution to the data and plot the histogram
    fit_lognormal_and_plot_histogram(t, bins=60, plot=True)



def lognormal_fit(param_values,times):

    mu = []
    sigma = []

    for i,time in enumerate(times):
        print(i)
        results = fit_lognormal_and_plot_histogram(time,bins=20,plot=True,Print=True)
        mu.append(results['mu'])
        sigma.append(results['sigma'])

    
    mu = np.array(mu)
    sigma = np.array(sigma)

    # Expectation of lognormal distribution
    mu_scaled = np.exp(mu + (sigma**2)/2)
    # https://en.wikipedia.org/wiki/Log-normal_distribution
    # Variance of lognormal distribution
    sigma_scaled = (np.exp(sigma**2)-1)

    plot_dual_y_axis(param_values, mu_scaled, sigma_scaled, param,'mu', 'sigma', 'Lognormal Fit Mean and Variance:'+ 'N Agents Rectangle' , 0, 5000, 0, 5)




param = 'p_a_s'

file_path = 'SA/SA_' + param +'.pkl'

with open(file_path, "rb") as file:
    param_values,times = pickle.load(file)



# FIlter out 10000+ values
cutoff = 9999
times = [filt_above(time,cutoff) for time in times]

# Remove last two items of list
trim_edges = 0
param_values = param_values[:len(param_values)-trim_edges]
times = [times[i] for i in range(len(times)-trim_edges)]

#make_histogram(t)

lognormal_fit(param_values,times)


# Write a function to plot the mean and variance of the lognormal distribution as a function of the parameter   












