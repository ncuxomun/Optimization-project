#%%
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import subprocess
import sys
import gc
import time
import os

#%%
start = time.time()
# params = np.load(r"D:\Optim Proj\maps_hat.npy")
params = np.load(r"D:\Work\Gaussian\gaussian_maps_new.npy")
params = params[:, 11:31]

#%%
porosity = np.round(np.dot(params, 0.05) + 0.24, 4)
# permeability = np.round(np.add(5, np.dot(10000000, np.power(porosity, 10))), 4) 
permeability = np.round(50 + 10e6 * np.power(porosity, 10), 4)

# plt.imshow(porosity[:, 22].reshape(100, 100), cmap = 'jet'); plt.colorbar()
# # plt.imshow(porosity[:, 2].reshape(100, 100), cmap = 'jet')

#%%
# plt.imshow(np.log(permeability[:, 0]).reshape(100, 100), cmap='jet')
# plt.colorbar()

RANGE = params.shape[1]

#%%
for i in range(RANGE):
    # Read in the file
    with open(r'D:\Optim Proj\Simulations\Maps for NPV\GAUSS.DATA', 'r') as file:
        filedata = file.read()

    file_1 = open(r'D:\Optim Proj\Simulations\Maps for NPV\PORO.INC', 'w')
    file_1.write('FILEUNIT \n')
    file_1.write('FIELD / \n')
    file_1.write('PORO \n')
    np.set_printoptions(threshold=sys.maxsize)
    file_1.write(np.array2string(porosity[:, i].reshape(100, 100)).replace('[', '').replace(']', ''))
    file_1.write('\n')
    file_1.write('/')
    file_1.close()

    # open poro file
    with open(r'D:\Optim Proj\Simulations\Maps for NPV\PORO.INC', 'r') as p_file:
        p_filedata = p_file.read()

    # write a new one
    with open(f'D:\\Optim Proj\\Simulations\\Maps for NPV\\PORO_TAB_{i+1}.INC', 'w') as new_p_file:
        new_p_file.write(p_filedata)

    file_2 = open(r'D:\Optim Proj\Simulations\Maps for NPV\PERM.INC', 'w')
    file_2.write('FILEUNIT \n')
    file_2.write('FIELD / \n')
    file_2.write('PERMX \n')
    np.set_printoptions(threshold=sys.maxsize)
    file_2.write(np.array2string(permeability[:, i].reshape(100, 100)).replace('[', '').replace(']', ''))
    file_2.write('\n')
    file_2.write('/')
    file_2.close()

    # open perm file
    with open(r'D:\Optim Proj\Simulations\Maps for NPV\PERM.INC', 'r') as perm_file:
        perm_filedata = perm_file.read()

    # write a new one
    with open(f'D:\\Optim Proj\\Simulations\\Maps for NPV\\PERM_TAB_{i+1}.INC', 'w') as new_perm_file:
        new_perm_file.write(perm_filedata)

    # Replace the target string
    filedata = filedata.replace('PORO_TAB', f'PORO_TAB_{i+1}')
    filedata = filedata.replace('PERM_TAB', f'PERM_TAB_{i+1}')

    # Write the file out again
    with open(f'D:\\Optim Proj\\Simulations\\Maps for NPV\\GAUSS_{i+1}.DATA', 'w') as new_file:
        new_file.write(filedata)

    print(f'Realization: {i+1}')

gc.collect()

# %%
print(f'{params.shape[1]} took {(time.time() - start) / 60.0} minutes')
