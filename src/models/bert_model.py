import torch
import pytorch_lightning as pl
from torch import nn


class VGG16(pl.LightningModule):
    def __init__(self, num_classes, lr=2e-4, num_epochs=10, steps_per_epoch=35, div_factor=1e+3):
        super().__init__()
        self.save_hyperparameters()
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.div_factor = div_factor
        self.lr = lr
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            epochs=self.num_epochs,
            steps_per_epoch=self.steps_per_epoch,
            max_lr=self.lr,
            pct_start=0.1,
            anneal_strategy='linear',
            final_div_factor=self.div_factor
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(torch.argmax(preds, dim=1), y)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(torch.argmax(preds, dim=1), y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        acc = self.accuracy(torch.argmax(preds, dim=1), y)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)
