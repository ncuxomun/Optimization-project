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
import skopt

#%%
start = time.time()
params = np.load(r"D:\Work\Gaussian\gaussian_maps_new.npy")
params = params.reshape(10000, -1)
params = params[:, 21]
params = params.reshape(-1, 1)

#%%
porosity = np.round(np.dot(params, 0.05) + 0.24, 4)
# permeability = np.round(np.add(5, np.dot(10000000, np.power(porosity, 10))), 4) 
permeability = np.round(50 + 10e6 * np.power(porosity, 10), 4)

#%%
file_1 = open(r'D:\Optim Proj\Simulations\PORO.INC', 'w')
file_1.write('FILEUNIT \n')
file_1.write('FIELD / \n')
file_1.write('PORO \n')
np.set_printoptions(threshold=sys.maxsize)
file_1.write(np.array2string(porosity.reshape(100, 100)).replace('[', '').replace(']', ''))
file_1.write('\n')
file_1.write('/')
file_1.close()

file_2 = open(r'D:\Optim Proj\Simulations\PERM.INC', 'w')
file_2.write('FILEUNIT \n')
file_2.write('FIELD / \n')
file_2.write('PERMX \n')
np.set_printoptions(threshold=sys.maxsize)
file_2.write(np.array2string(permeability.reshape(100, 100)).replace('[', '').replace(']', ''))
file_2.write('\n')
file_2.write('/')
file_2.close()

#%%
# CONTROLS
path = 'D:\Optim Proj\CSVs/'
# controls = pd.read_csv(path+'controls_new_limits.csv').iloc[:, 1:].values
controls = np.load('proxy_control.npy')

n_samples = 300
r = [(3500, 5500), ] * 20
p = [(1150, 1400), ] * (80-20)

space = skopt.Space(r + p)
method = skopt.sampler.Lhs(criterion="maximin", iterations=10000)
# x = method.generate(space.dimensions, n_samples)
# controls = np.array(x)

# np.save('controls_300.npy', controls)
# controls = np.load('controls_300.npy')

# controls = np.ones((3, 104)) * 3000
# controls[:, 26:] = 1200
# %%
u0 = np.zeros((80, )) + 0.5  # NORMED_CTRLS[2]

ub_0 = u0.reshape(-1, 1).T
ub_0 = ub_0.reshape(4, 20).T.astype('int32')

c_list = list()

for p in range(21, 31):
    for i in range(1, 3):
        c_list.append(f'@I1_RTE_{p}_{i}@')
        c_list.append(f'@P1_PRS_{p}_{i}@')
        c_list.append(f'@P2_PRS_{p}_{i}@')
        c_list.append(f'@P3_PRS_{p}_{i}@')

c_list = np.reshape(c_list, ub_0.shape)

# %%
# TODO: Calculation of NVP

CONTROLS = controls.copy()

NPV_DATA = np.zeros((CONTROLS.shape[0], 21, 1)) # +1 first CAPEX period
FOE_DATA = np.zeros_like(NPV_DATA)
SOIL = np.zeros((CONTROLS.shape[0], 10000, 2)) # first and last steps
PRESSURE = np.zeros_like(SOIL)

discount_rate = 15.000000  # %
oil_sale_price = 40.000000  # $/stb
gas_sale_price = 2.500000  # $/mscf
oil_prod_cost = 10.000000  # $/stb
water_prod_cost = 6.000000  # $/stb
gas_prod_cost = 2.000000  # $/mscf
wat_inj_cost = 5.000000  # $/stb

CAPEX = 4 * 10e6

#%%
for c in range(CONTROLS.shape[0]):  # CONTROLS.shape[0]
    ub_0 = CONTROLS[c].reshape(4, 20).T

    with open(r'D:\Optim Proj\Simulations\BO_GAUSS.DATA', 'r') as file:
        data_data = file.read() 

    for i in range(0, 1):
        for j in range(ub_0.shape[1]):
            data_data = data_data.replace(c_list[i, j], f'{ub_0[i, j]}')
    
    # replacing the schedule
    data_data = data_data.replace('CONTROL_SCHEDULE.INC', 'CONTROL_SCHEDULE_TRIAL.INC')

    with open(r'D:\Optim Proj\Simulations\BO_GAUSS_trial.DATA', 'w') as new_file:
        new_file.write(data_data)

    #############################################################################
    with open(r'D:\Optim Proj\Simulations\CONTROL_SCHEDULE.INC', 'r') as cs_file:
        filedata = cs_file.read()

    for i in range(1, ub_0.shape[0]):
        for j in range(ub_0.shape[1]):
            filedata = filedata.replace(c_list[i, j], f'{ub_0[i, j]}')

    with open(r'D:\Optim Proj\Simulations\CONTROL_SCHEDULE_TRIAL.INC', 'w') as w_cs_file:
        w_cs_file.write(filedata)

    #############################################################################
    
    # subprocess.call([r'D:\tNavigator\tNavigator-con', '--ecl-rsm', # --use-gpu',
    #                 r'D:\Optim Proj\Simulations\BO_GAUSS_trial.DATA'])

    # dumping all binaries
    subprocess.call([r'D:\tNavigator\tNavigator-con', '--ecl-rsm', '--ecl-unrst',  # '--use-gpu',
                     r'D:\Optim Proj\Simulations\BO_GAUSS_trial.DATA'])

    with open(r'D:\Optim Proj\Simulations\RESULTS\BO_GAUSS_trial.RSM') as file_in:
        data = []
        for line in file_in:
            line = line.split()
            data.append(line)
    data = pd.DataFrame(data)

    FGOR = data.iloc[10:31, 5].astype(float).values.reshape(-1, 1) # mscf/stb
    FOPT = data.iloc[10:31, 6].astype(float).values.reshape(-1, 1) * 10e6 # stb
    FWPT = data.iloc[10:31, 7].astype(float).values.reshape(-1, 1) * 10e3 # stb
    FWIT = data.iloc[10:31, 8].astype(float).values.reshape(-1, 1) *10e6 # stb
    FOE = data.iloc[10:31, 9].astype(float).values.reshape(-1, 1)

    REVENUE = oil_sale_price * FOPT + (FOPT * FGOR) * gas_sale_price
    COST = oil_prod_cost * FOPT + water_prod_cost * FWPT + wat_inj_cost * FWIT + (FOPT * FGOR) * gas_prod_cost

    period = np.arange(0.0, 10.5, 0.5) # discount periods

    # single value
    NPV = (REVENUE - COST) / ((1 + discount_rate*0.01/2) ** (2*period[-1])).reshape(-1, 1) - CAPEX

    NPV_DATA[c, :] = NPV
    FOE_DATA[c, :] = FOE

    print(f"Realization {c+1} completed.")

    # converting binary to text
    subprocess.call([r'D:\tNavigator\tNavigator-con', '--convert-ecl-bin-to-text',
                     r'D:\Optim Proj\Simulations\RESULTS\BO_GAUSS_trial.UNRST'])

    with open(r'D:\Optim Proj\Simulations\RESULTS\BO_GAUSS_trial.FUNRST') as all_file_in:
        grids = []
        for line in all_file_in:
            line = line.split()
            grids.append(line)
    grids = pd.DataFrame(grids)

    skip_value = 2500
    # saturations = [2643, 10645, 18647, 26649, 34651, 42653, 50655, 58657, 66659, 74661, 82663, \
    #                90665, 98667, 106669, 114671, 122673, 130675, 138677, 146679, 154681, 162683, \
    #                170685, 178687, 186689, 194691, 202693, 210695]

    # pressures = [142, 8144, 16146, 24148, 32150, 40152, 48154, 56156, 64158, 72160, 80162,
    #              88164, 96166, 104168, 112170, 120172, 128174, 136176, 144178, 152180, 160182,
    #              168184, 176186, 184188, 192190, 200192, 208194]

    saturations = [2643, 10645]

    pressures = [142, 8144]

    soil_map = np.zeros((10000, 2))
    pressure_map = np.zeros_like(soil_map)

    for k, s in enumerate(saturations):
        soil_map[:, k] = grids.iloc[s:s+2500, 0:4].astype(float).values.reshape(-1, )
    SOIL[c] = soil_map

    for g, p in enumerate(pressures):
        pressure_map[:, g] = grids.iloc[p:p+2500, 0:4].astype(float).values.reshape(-1, )
    PRESSURE[c] = pressure_map

gc.collect()

# %%
print(f'{CONTROLS.shape[0]} took {(time.time() - start) / 60.0} minutes')

# np.save('NPV_proxy.npy', NPV_DATA[:, -1, 0])
# np.save('FOE.npy', FOE_DATA)
# np.save('SOIL.npy', SOIL)
# np.save('PRESSURE.npy', PRESSURE)

#%%
