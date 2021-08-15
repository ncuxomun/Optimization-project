#%%
from torch.autograd.functional import jacobian, hessian
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import torch
from torch import nn
import pandas as pd
import pytorch_lightning as pl
import torchvision
from sklearn.metrics import r2_score, mean_squared_error
from adabelief_pytorch import AdaBelief
import torch.nn.functional as F
import os
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import torchmetrics
from lib.proxy_model import LitReg
from lib.data_module import DataModule
from lib.optim_module import OPTIM
from lib.forward_runs import SIM
from lib.train_function import TRAIN

# seed = 999

# np.random.seed(seed)
# torch.manual_seed(seed)
# pl.seed_everything(seed)

#%%
################################## INITIAL DATA LOADING
def _data_load():
    # Path
    path_npy = os.getcwd() + '\Saved Models and Files/'

    # Examples vs Controls (Inj rate --> P1-P3 BHP), 27 x 4 = 108
    controls = np.load(path_npy+'controls_300.npy')
    npv = np.load(path_npy+'NPV.npy').squeeze()

    return controls, npv

CONTROLS, NPV = _data_load()

#%%
################################## SOME CONSTANTS
total_add_sim_runs = []
controls_new = []
npv_new = []
max_iter = 300
r_2_value = 0.0
count = 1
discr = 1  # initial discrepancy
tol_rmse = 2e-3 # discrepancy tolerance
scaler_controls = MinMaxScaler() # scaler for CONTROLS
scaler_npv = MinMaxScaler() # scaler for NPV

#%%
################################## MAIN LOOP
for i in range(max_iter):

    # normed input and outputs
    NORMED_CTRLS = scaler_controls.fit(CONTROLS).transform(CONTROLS)
    NORMED_NPV = scaler_npv.fit(NPV[:, -1:]).transform(NPV[:, -1:])

    # Reshaping
    NORMED_CTRLS = NORMED_CTRLS.reshape(-1, 4, 20)

    print("NORMED_CTRLS shape: ", NORMED_CTRLS.shape)
    print("NORMED_NPV shape: ", NORMED_NPV.shape)

    batch_size = 24
    lr = 1e-4
    in_size = NORMED_CTRLS.shape[-1]
    out_size = NORMED_NPV.shape[-1]
    in_out_size = in_size
    epochs = 5000

    # dataset for proxy
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(NORMED_CTRLS).float(),
                                             torch.from_numpy(NORMED_NPV).float())

    N = len(dataset)
    split = [int(2/3 * N), int(N - int(2/3 * N)), 0]

    ######################################
    if count == 1:
        seed = 999
        
    np.random.seed(seed); torch.manual_seed(seed); pl.seed_everything(seed)

    train = TRAIN(count, seed, dataset, split, batch_size, lr, epochs, in_out_size, r_2_value)
    model, x, y, y_hat, folder = train.forward()
    r_2_value = r2_score(y.flatten(), y_hat.flatten()).round(4)

    #####################################
    while r_2_value < 0.90:
        seed = np.random.randint(0, 1e3) + 10
        np.random.seed(seed); torch.manual_seed(seed); pl.seed_everything(seed)

        train = TRAIN(count, seed, dataset, split, batch_size, lr, epochs, in_out_size, r_2_value)
        model, x, y, y_hat, folder = train.forward()
        r_2_value = r2_score(y.flatten(), y_hat.flatten()).round(4)
    
    ######################################

    def plot(y, y_hat):
        plt.figure(10, figsize=(6, 4))
        plt.plot(y, 'r.--')
        plt.plot(y_hat, 'b')
        plt.ylabel('NPV')
        plt.ylim(0, 1)
        plt.savefig(os.getcwd() + "\lib\Model_Pred_{count}.png", dpi=150)
        # plt.show()
        plt.close(10)

        plt.figure(33, figsize=(4, 4))
        plt.plot(y, y_hat, 'bo', alpha=0.25)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f'$R^2$: {r2_score(y.flatten(), y_hat.flatten()).round(4)}')
        plt.savefig(os.getcwd() + "\lib\Model_Pred_R2_{count}.png", dpi=150)
        # plt.show()
        plt.close(33)

    plot(y, y_hat)

    print(folder)
    print(f"RMSE: {np.sqrt(mean_squared_error(y.flatten(), y_hat.flatten()))}")
    print(f'R2: {r2_score(y.flatten(), y_hat.flatten()).round(4)}')

    ######### RUN OPTIMIZATION
    plotProgress = True
    optim = OPTIM(x, model, count, plotProgress)
    new_proxy_controls = optim.optimization()

    # last 23 points
    new_sim_proxy_controls = scaler_controls.inverse_transform(new_proxy_controls[-2:])

    ######### RUN NPVs by PROXY
    model.eval()
    with torch.no_grad():
        # last 2 points
        new_npvs = model(torch.FloatTensor(new_proxy_controls[-2:].reshape(-1, x.shape[-2], x.shape[-1])))

    # proxy NPV inverse-transformed
    _NPV_p = scaler_npv.inverse_transform(new_npvs)

    # simulation of proxy-produced controls
    _NPV_p_sim, _FOE_SIM, _SOIL_SIM, _PRESSURE_SIM = SIM(new_sim_proxy_controls).npv_calculation()

    # count additional runs
    total_add_sim_runs.append(_NPV_p_sim.shape[0])

    def plot_npvs(npv_proxy, npv_proxy_sim):
        plt.figure(num=11, figsize=(4, 4))
        plt.plot(npv_proxy, 'b', label='Proxy')
        plt.plot(npv_proxy_sim[:, -1:, 0], 'r', label='Sim')
        plt.ylabel('NPV')
        plt.xlabel('# Iterations')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.getcwd() + "\lib\Comparison_{count}.png", dpi=150)
        plt.close(11)
        # plt.show()

    plot_npvs(_NPV_p, _NPV_p_sim)

    ######### ADDING NEW DATAPOINTS TO TRAINING STACK
    NEW_NPV = np.vstack((NPV[:, -1:], _NPV_p_sim[:, -1:, 0]))
    NEW_CONTROLS = np.vstack((CONTROLS, new_sim_proxy_controls))

    ######### COMPUTE DISCREPANCY
    discr = np.sqrt(mean_squared_error(_NPV_p_sim[:, -1:, 0]/1e8, _NPV_p/1e8))
    print('Discrepancy, RMSE: ', discr)

    if discr <= tol_rmse:
        break

    count += 1
    ######### SETTING NEW CONTROLs AND NPVs
    controls_new.append(new_sim_proxy_controls)
    npv_new.append(_NPV_p_sim[:, -1:, 0])
    CONTROLS = NEW_CONTROLS
    NPV = NEW_NPV

#%%
# np.save(r"D:\Optim Proj\lib\controls_new.npy", np.array(controls_new))
# np.save(r"D:\Optim Proj\lib\npv_new.npy", np.array(npv_new))
# np.save(r"D:\Optim Proj\lib\total_add_sim_runs.npy", np.array(total_add_sim_runs))
