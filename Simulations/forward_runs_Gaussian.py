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

#%%
start = time.time()
# RANGE = 500
# params = np.load(r"D:\Work\Gaussian\gaussian_maps_new.npy")
# params = params.reshape(10000, -1)
# params = params[:, 21]
# params = params.reshape(-1, 1)

# params = np.load('all_ms.npy').T
params = np.load('cc_ms.npy').T
#%%
porosity = np.round(np.dot(params, 0.05) + 0.24, 4)
# permeability = np.round(np.add(5, np.dot(10000000, np.power(porosity, 10))), 4) 
permeability = np.round(50 + 10e6 * np.power(porosity, 10), 4)

# plt.imshow(porosity[:, 22].reshape(100, 100), cmap = 'jet'); plt.colorbar()
# # plt.imshow(porosity[:, 2].reshape(100, 100), cmap = 'jet')

#%%
# plt.imshow(np.log(permeability[:, 0]).reshape(100, 100), cmap='jet')
# plt.colorbar()

#%%
WOPR_out = []
WOPT_out = []
WWPR_out = []
WBHP_out = []
WWCT_out = []

RANGE = params.shape[1]

#%%
# porosity.shape[1]
for i in range(RANGE):
    file_1 = open(r'D:\Optim Proj\Simulations\PORO.INC', 'w')
    file_1.write('FILEUNIT \n')
    file_1.write('FIELD / \n')
    file_1.write('PORO \n')
    np.set_printoptions(threshold=sys.maxsize)
    file_1.write(np.array2string(porosity[:, i].reshape(100, 100)).replace('[', '').replace(']', ''))
    file_1.write('\n')
    file_1.write('/')
    file_1.close()

    file_2 = open(r'D:\Optim Proj\Simulations\PERM.INC', 'w')
    file_2.write('FILEUNIT \n')
    file_2.write('FIELD / \n')
    file_2.write('PERMX \n')
    np.set_printoptions(threshold=sys.maxsize)
    file_2.write(np.array2string(permeability[:, i].reshape(100, 100)).replace('[', '').replace(']', ''))
    file_2.write('\n')
    file_2.write('/')
    file_2.close()

    subprocess.call([r'D:\tNavigator\tNavigator-con', '--ecl-rsm', # '--use-gpu',
                     r'D:\Optim Proj\Simulations\BO_GAUSS_trial.DATA'])

    # subprocess.call([r'D:\tNavigator\tNavigator-con', '--ecl-rsm', '-d', '-e', '-m', '-r', '-u', # '--use-gpu',
    #                  r'D:\Optim Proj\Simulations\BO_GAUSS_trial.DATA'])

    # subprocess.call([r'D:\tNavigator\tNavigator-con', '--convert-ecl-bin-to-text',
    #                  r'D:\Optim Proj\Simulations\RESULTS\BO_GAUSS_trial.SMSPEC'])

    with open(r'D:\Optim Proj\Simulations\RESULTS\BO_GAUSS_trial.RSM') as file_in:
        data = []
        for line in file_in:
            line = line.split()
            data.append(line)
    data = pd.DataFrame(data)

    # WWPR = data.iloc[9:35, 5:9].astype(float).values
    # WWPR = WWPR.flatten(order='A').reshape(-1, 1)
    # WWPR_out.append(WWPR)

    WOPR = data.iloc[46:73, 7:10].astype(float).values
    # temp = data.iloc[45:71, 1:4].astype(float).values
    # WOPR = np.concatenate((WOPR, temp), axis=1)
    WOPR = WOPR.flatten(order='A').reshape(-1, 1)
    WOPR_out.append(WOPR)

    # WOPT = data.iloc[45:71, 8:10].astype(float).values
    # temp_ = data.iloc[81:107, 1:3].astype(float).values
    # WOPT = np.concatenate((WOPT, temp_), axis=1)
    # WOPT = np.multiply(1000, WOPT.flatten(order='A').reshape(-1, 1))
    # WOPT_out.append(WOPT)

    WWCT = data.iloc[119:146, 1:4].astype(float).values
    WWCT = WWCT.flatten(order='A').reshape(-1, 1)
    WWCT_out.append(WWCT)

    # WBHP = data.iloc[117:143, 2:6].astype(float).values
    # if WBHP.shape == (25, 4):
    #     WBHP = np.vstack((np.zeros((1, 4)), WBHP))

    # _ = np.where(WBHP > 100, WBHP, WBHP * 1000)
    # WBHP = _

    # WBHP = WBHP.flatten(order='A').reshape(-1, 1)
    # WBHP_out.append(WBHP)

    print(f"Realization {i+1} completed.")

gc.collect()

#%%
np.save('WOPR_cc.npy', WOPR_out)
# np.save('WOPT.npy', WOPT_out)
# np.save('WWPR_or.npy', WWPR_out)
np.save('WWCT_cc.npy', WWCT_out)
# np.save('WBHP.npy', WBHP_out)

# %%
# obs = sio.loadmat(r'C:\Users\ulugb\Desktop\Test\observation.mat')['observation']

# # %%
# plt.plot(obs[:75, :], 'r--', label='loaded')
# plt.plot(WOPR_out[0][:75, :], label='Simulated')
# plt.legend()
# plt.show()

# %%
print(f'{params.shape[1]} took {(time.time() - start) / 60.0} minutes')

# %%
