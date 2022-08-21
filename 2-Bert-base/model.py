
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer


class TextModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.text_model = None
        self.output_dim = 768
        if args.text_model == "roberta-wwm":
            self.text_model = BertModel.from_pretrained("model/chinese_roberta_wwm/")
            self.output_dim = self.text_model.pooler.dense.out_features
        
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, 2)
        )
        
        # print(self.text_model)

    def forward(self, input_ids, attention_mask):
        model_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(model_output.pooler_output)


class CIFARModule(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.modle = TextModel(args)
        self.loss = nn.CrossEntropyLoss()
        # self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

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
        input_ids, attention_mask, labels = batch
        preds = self.modle(input_ids, attention_mask)
        loss = self.loss(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        preds = self.modle(input_ids, attention_mask)
        loss = self.loss(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("val_acc", acc, on_step=True)
        self.log("val_loss", loss, on_step=True)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        preds = self.modle(input_ids, attention_mask)
        loss = self.loss(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("test_acc", acc, on_step=True)
        self.log("test_loss", loss, on_step=True)

