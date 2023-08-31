import numpy as np
from Environment import Env
from gen_params import generate_agent_parameters
import matplotlib.pyplot as plt
from utils import circle_maker, square_maker

class SensitivityAnalysis:

    def __init__(self, env_size,shape,n_agents,radius) -> None:

        # Create Base_Line
        self.env = Env(env_size,shape,n_agents,radius, None)




    def local_sensitivity(self,param,range_low, range_high, steps, CV_step=5,thresh=0.05, lookback=5):

        param_range = np.linspace(range_low,range_high,steps)
        # Store Results
        loss_for_value = np.zeros_like(param_range)
        n_iters = np.zeros_like(param_range)
        # Generate Parameters
        params = generate_agent_parameters()
        params_dict = params.return_dictionary()

        for i,value in enumerate(param_range):
            # Load Params into the environment
            params_dict[param] = value
            params.input_dictionary(params_dict)
            self.env.params = params

            results_for_value = [self.env.run_sim() for x in range(CV_step*lookback)]
            CVs = [self.coefficient_of_variation(results_for_value[0:CV_step*x]) for x in range(1,lookback+1)]
            print('Done Prelim')
            while self.check_CV_stab(CVs,lookback,thresh):

                new_results = [self.env.run_sim() for x in range(CV_step)]
                results_for_value += new_results
                CVs.append(self.coefficient_of_variation(results_for_value))
                print('New Set')
            
            loss_for_value[i] = sum(results_for_value)/len(results_for_value)
            n_iters[i] = len(results_for_value)
            print(CVs)
            print('DONE iter',value)

        plt.plot(param_range,loss_for_value,label='Loss for Parameter Value')
        plt.plot(param_range,n_iters,label='n_iters')
        plt.legend()
        plt.grid()
        plt.suptitle(param + ' variation')
        plt.show()

        return param_range,loss_for_value

        

    def check_CV_stab(self,CV,lookback,thresh):

        last = CV[-lookback:]

        avg = sum(last)/lookback
        mx = max(last)
        mn = min(last)

        if (mn-mx)/avg < thresh:
            return True
        else:
            return False


    def coefficient_of_variation(self,data, epsilon=0.01):

        data = np.array(data)
        # Calculate the mean and standard deviation
        mean = np.mean(data)
        std_dev = np.std(data)

        # Calculate the stabilized coefficient of variation (SCV)
        cv = (std_dev + epsilon) / (mean + epsilon) * 100
        print(cv)
        return cv



if __name__ == '__main__':

    circle = circle_maker(20,50)
    SA = SensitivityAnalysis(100,circle,20,30)
    SA.local_sensitivity('p_root',0.01,0.2,20)

