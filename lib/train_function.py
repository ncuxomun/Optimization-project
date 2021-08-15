import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import torchvision
from sklearn.metrics import r2_score, mean_squared_error
import torch.nn.functional as F
import os
from lib.data_module import DataModule
from lib.proxy_model import LitReg

class TRAIN:
    def __init__(self, count, seed, dataset, split, batch_size, lr, epochs, in_out_size, r_2_value):
        self.count = count
        self.seed = seed
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.in_out_size = in_out_size
        self.r_2_value = r_2_value
    
    def forward(self):
        pl.seed_everything(self.seed)

        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(self.dataset, self.split, generator=torch.Generator().manual_seed(self.seed))

        dm = DataModule(self.dataset, self.batch_size, self.split, self.seed)

        ######### MODEL TRAINING
        early_stopping = pl.callbacks.EarlyStopping('val_loss', patience=30)
        model = LitReg(self.in_out_size, self.lr)

        # trainer = pl.Trainer(fast_dev_run=True)
        trainer = pl.Trainer(max_epochs = self.epochs, progress_bar_refresh_rate = 50, callbacks = [early_stopping], stochastic_weight_avg=True)
        trainer.fit(model, datamodule=dm)

        # save model for run
        torch.save(model.state_dict(), os.getcwd() + "\lib\pte_model_{self.count}")

        ######### MODEL QC
        def data_access(folder=None):
            if folder == 'train':
                data_set = train_dataset; n = 0
            elif folder == 'val':
                data_set = valid_dataset; n = 1
            elif folder == 'test':
                data_set = test_dataset; n = 2

            dims_x, dims_y = (self.split[n], 4, self.in_out_size), (self.split[n], 1)

            x_ = torch.zeros(dims_x)
            y_ = torch.zeros(dims_y)

            for i, (x_in, y_in) in enumerate(data_set):
                x_[i, :] = x_in
                y_[i, :] = y_in

            return x_, y_

        folder = "val"  # "train", "val", "test"

        x, y = data_access(folder)

        # model.load_state_dict(torch.load(r"D:\Optim Proj\lib\pte_model_{}"))
        model.eval()
        with torch.no_grad():
            y_hat = model(x)

        x, y, y_hat = x.numpy(), y.numpy(), y_hat.numpy()
        

        return model, x, y, y_hat, folder
