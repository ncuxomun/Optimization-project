from torch.autograd.functional import jacobian, hessian
from scipy.optimize import minimize
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

class OPTIM:
    def __init__(self, x, model, count=1, figure=True):
        self.x = x # input x
        self.model = model # trained model
        self.count = count
        self.figure_plot = figure

        self.cost_values = []
        self.controls_list = []

    def nvp_eval_function(self, u):
        global norm_npv
        global u_track

        u = u.reshape(-1, self.x.shape[1], self.x.shape[2])
        u_opt = torch.FloatTensor(u)

        self.model.eval()
        with torch.no_grad():
            npv_hat = self.model(u_opt)

        # mean neg npv to optimize/minimize
        avg_npv = -1 * npv_hat
        u_track = u_opt.detach().numpy()
        norm_npv = avg_npv.detach().numpy().astype('float64')
        return norm_npv

    def log_cost(self, a, b):
        self.cost_values.append(norm_npv.item())
        self.controls_list.append(u_track)

    def jac_function(self, u):
        u = u.reshape(-1, self.x.shape[1], self.x.shape[2])
        u_opt = u

        self.model.eval()
        npv_hat = self.model(u_opt)

        # mean neg npv to optimize/minimize
        avg_npv = -1 * npv_hat
        norm_avg = avg_npv
        return avg_npv

    def model_der(self, u):
        der = jacobian(self.jac_function, torch.from_numpy(u).float())
        der = np.reshape(der.T.detach().numpy(), (self.x.shape[-2] * self.x.shape[-1],))
        der = der.astype('float64')
        # TODO: for safety, remove if needed
        der = np.nan_to_num(der, copy=False, nan=1e-4)
        return der

    def model_hess(self, u):
        hess = hessian(self.jac_function, torch.from_numpy(u).float())
        hess = hess.detach().numpy()
        hess = hess.astype('float64')
        # TODO: for safety, remove if needed
        hess = np.nan_to_num(hess, copy=False, nan=1e-4)
        return hess

    def optimization(self):
        u0 = np.zeros((self.x.shape[-2] * self.x.shape[-1], )) + 0.50
        bnds = ([0, 1], ) * self.x.shape[-2] * self.x.shape[-1]

        # self.cost_values = []
        # self.controls_list = []

        res = minimize(self.nvp_eval_function, u0, method='trust-constr', jac=self.model_der, hess=self.model_hess, 
                       callback=self.log_cost, bounds=bnds, tol=2e-3, options={'verbose': 3})

        u = u0.reshape(-1, self.x.shape[1], self.x.shape[2])
        u_opt = torch.FloatTensor(u)

        self.model.eval()
        with torch.no_grad():
            npv_ = self.model(u_opt)

        proxy_control = np.squeeze(np.asarray(self.controls_list))
        proxy_control = proxy_control.reshape(-1, self.x.shape[-2] * self.x.shape[-1])

        if self.figure_plot:
            plt.figure(num=22, figsize=(4, 4))
            plt.plot(-1*np.array(self.cost_values), 'k*-')
            plt.xlabel('# Iterations')
            plt.ylabel('Norm Obj Function')
            plt.savefig(f"D:\Optim Proj\lib\Progress_{self.count}.png", dpi=150)
            plt.tight_layout()
            plt.close(22)
            # plt.show()

        return proxy_control
