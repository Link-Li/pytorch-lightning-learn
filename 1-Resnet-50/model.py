
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.modle = resnet50(pretrained=True, progress=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        for param in self.modle.parameters():
            param.requires_grad = False
    
    def forward(self, imgs):
        return self.classifier(self.modle(imgs))


class CIFARModule(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.modle = ResNet50()
        self.loss = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        return self.modle(imgs)

    def configure_optimizers(self):
        optimizer = None
        scheduler = None
        if self.args.optimizer_name == "Adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=0.9)
        
        if self.args.scheduler_name == "lr_schedule":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=self.args.warmup_step,
                num_training_steps=self.args.total_steps)

        if optimizer and scheduler:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif optimizer:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.modle(imgs)
        loss = self.loss(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("train_acc", acc, on_step=True)
        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.modle(imgs)
        loss = self.loss(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("val_acc", acc, on_step=True)
        self.log("val_loss", loss, on_step=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.modle(imgs)
        loss = self.loss(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("test_acc", acc, on_step=True)
        self.log("test_loss", loss, on_step=True)

