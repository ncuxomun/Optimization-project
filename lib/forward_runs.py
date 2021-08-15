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

# time start
start = time.time()

class SIM:
    def __init__(self, x):
        # pick a map to run optimization for
        MAP = 21 

        params = np.load("gaussian_maps_new.npy") # should be unpacked first
        params = params.reshape(10000, -1)
        self.params = params[:, MAP].reshape(-1, 1)

        # rock properties
        self.porosity = np.round(np.dot(self.params, 0.05) + 0.24, 4)
        self.permeability = np.round(50 + 10e6 * np.power(self.porosity, 10), 4)

        # controls
        self.x = x
        no_wells = 4

        u0 = np.zeros((self.x.shape[1], )) + 0.5
        ub_0 = u0.reshape(-1, 1).T
        ub_0 = ub_0.reshape(self.x.shape[1]//no_wells, no_wells).astype('int32')
        self.u_control = ub_0.copy()

        c_list_temp = list()

        for p in range(21, 31):
            for i in range(1, 3):
                c_list_temp.append(f'@I1_RTE_{p}_{i}@')
                c_list_temp.append(f'@P1_PRS_{p}_{i}@')
                c_list_temp.append(f'@P2_PRS_{p}_{i}@')
                c_list_temp.append(f'@P3_PRS_{p}_{i}@')

        self.c_list = np.reshape(c_list_temp, ub_0.shape)

    def npv_calculation(self):

        CONTROLS = self.x.copy()
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

        for c in range(CONTROLS.shape[0]):  # CONTROLS.shape[0]
            # x_0 = CONTROLS[c].reshape(self.x.shape[-2], self.x.shape[-1]).T
            x_0 = CONTROLS[c].reshape(*self.u_control.T.shape).T

            with open(os.getcwd() + '\Simulations\BO_GAUSS.DATA', 'r') as file:
                data_data = file.read() 

            for i in range(0, 1):
                for j in range(x_0.shape[1]):
                    data_data = data_data.replace(self.c_list[i, j], f'{x_0[i, j]}')
            
            # replacing the schedule
            data_data = data_data.replace('CONTROL_SCHEDULE.INC', 'CONTROL_SCHEDULE_TRIAL.INC')

            with open(os.getcwd() + '\Simulations\BO_GAUSS_trial.DATA', 'w') as new_file:
                new_file.write(data_data)

            #############################################################################
            with open(os.getcwd() + '\Simulations\CONTROL_SCHEDULE.INC', 'r') as cs_file:
                filedata = cs_file.read()

            for i in range(1, x_0.shape[0]):
                for j in range(x_0.shape[1]):
                    filedata = filedata.replace(self.c_list[i, j], f'{x_0[i, j]}')

            with open(os.getcwd() + '\Simulations\CONTROL_SCHEDULE_TRIAL.INC', 'w') as w_cs_file:
                w_cs_file.write(filedata)

            #############################################################################
            
            # subprocess.call([r'D:\tNavigator\tNavigator-con', '--ecl-rsm', # --use-gpu',
            #                 r'D:\Optim Proj\Simulations\BO_GAUSS_trial.DATA'])

            # dumping all binaries
            subprocess.call([r'D:\tNavigator\tNavigator-con', '--ecl-rsm', '--ecl-unrst',  # '--use-gpu',
                             os.getcwd() + '\Optim Proj\Simulations\BO_GAUSS_trial.DATA'])

            with open(os.getcwd() + '\Simulations\RESULTS\BO_GAUSS_trial.RSM') as file_in:
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
                             os.getcwd() + '\Simulations\RESULTS\BO_GAUSS_trial.UNRST'])

            with open(os.getcwd() + '\Simulations\RESULTS\BO_GAUSS_trial.FUNRST') as all_file_in:
                grids = []
                for line in all_file_in:
                    line = line.split()
                    grids.append(line)
            grids = pd.DataFrame(grids)

            skip_value = 2500

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


        # print(f'{CONTROLS.shape[0]} took {(time.time() - start) / 60.0} minutes')

        return NPV_DATA, FOE_DATA, SOIL, PRESSURE
