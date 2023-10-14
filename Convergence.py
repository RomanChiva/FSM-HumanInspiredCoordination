import numpy as np
import matplotlib.pyplot as plt
import pickle


param = 'p_GU_c'
## Load Data

path = 'SA/SA_'+param+'.pkl'

with open(path,'rb') as file:
    data = pickle.load(file)

x = np.array(data[0])
y = np.array(data[1])

x_conv = y < 10000
x_nconv = y > 10000

conv = np.sum(x_conv,axis=-1)
n_conv = np.sum(x_nconv, axis=-1)

print(conv)
print(n_conv)
title = 'Convergence Plot: ' + param
plt.suptitle(title)
plt.plot(x,conv)
plt.grid()
plt.xlabel(param)
plt.ylabel('Convergence Rate Percentage')
plt.show()
plt.savefig('figs/ConvPlot'+param+'.png')




