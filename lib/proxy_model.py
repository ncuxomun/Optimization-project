import torch
from torch import nn
import pytorch_lightning as pl
import torchvision
from adabelief_pytorch import AdaBelief

class LitReg(pl.LightningModule):
    def __init__(self, in_out_dims, lr=2e-4):
        super().__init__()
        self.in_out_dims = in_out_dims
        self.lr = lr
        self.c = 64

        # model
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=self.c, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.SELU(inplace=True),
            nn.Conv1d(self.c, self.c//2, 3, 1, 1, padding_mode='replicate'),
            nn.BatchNorm1d(self.c//2),
            nn.SELU(inplace=True),
            nn.Conv1d(self.c//2, self.c//4, 3, 1, 1, padding_mode='replicate'),
            nn.BatchNorm1d(self.c//4),
            nn.SELU(inplace=True),
            nn.Conv1d(self.c//4, self.c//8, 3, 1, 1, padding_mode='replicate'),
            nn.BatchNorm1d(self.c//8),
            nn.SELU(inplace=True),
            nn.Conv1d(self.c//8, self.c//16, 3, 1, 1, padding_mode='replicate'),
            nn.SELU(inplace=True),
            nn.Conv1d(self.c//16, 1, 3, 1, 1, padding_mode='replicate'),
            nn.SELU(inplace=True),
            nn.Flatten(),
            nn.Dropout(0.10),
            nn.Linear(20, 20),
            nn.Linear(20, 1)
        )

        self.model.apply(self.weights_init)

        self.criterion = nn.MSELoss()

    def forward(self, data):
        out = self.model(data)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self.model(x)
        train_loss = self.criterion(pred_y, y)
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'test')

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        pred_y = self.model(x)
        loss = self.criterion(pred_y, y)
        self.log(f'{prefix}_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), self.lr)
        optimizer = AdaBelief(self.parameters(), lr=self.lr, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=True)
        return optimizer

    def weights_init(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
