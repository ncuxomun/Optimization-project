import torch
import pytorch_lightning as pl
import numpy as np
import os

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset=None, batch_size=25, split=None, seed=0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.split = split

    def prepare_data(self):
        # dataset for NPV
        # dataset = torch.utils.data.TensorDataset(torch.from_numpy(NORMED_CTRLS).float(),
        #                                          torch.from_numpy(NORMED_NPV).float())
        self.dataset = self.dataset

    def setup(self, stage=None):
        # train/valid/test split
        # and assign to use in dataloaders via self
        train_set, valid_set, test_set = torch.utils.data.random_split(self.dataset, self.split, generator=torch.Generator().manual_seed(self.seed))

        if stage == 'fit' or stage is None:
            self.train_set = train_set
            self.valid_set = valid_set
        if stage == 'test' or stage is None:
            self.test_set = test_set

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)