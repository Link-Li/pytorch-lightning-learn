import os
import json

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from transformers import T5ForConditionalGeneration
import numpy as np
from util.cal_evaluate import compute_rouge


class TextModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.text_model = None
        self.output_dim = 768
        if args.text_model == "mengzi-t5":
            self.text_model = T5ForConditionalGeneration.from_pretrained(args.load_model_path)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, 2)
        )

    def forward(self, input_ids, attention_mask, labels):
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output
    
    def generate(self, input_ids, attention_mask):
        return self.text_model.generate(inputs=input_ids, attention_mask=attention_mask, max_length=200, num_beams=3)


class CIFARModule(pl.LightningModule):
    def __init__(self, args, tokenizer) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.tokenizer = tokenizer
        self.model = TextModel(args)

    def forward(self, imgs):
        return self.model(imgs)

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
        preds = self.model(input_ids, attention_mask, labels)
        self.log("train_loss", preds.loss, on_step=True, on_epoch=True)
        return preds.loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        labels[labels == -100] = self.tokenizer.pad_token_id
        preds = self.model.generate(input_ids, attention_mask)
        labels_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        preds_text = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        rouge_res = compute_rouge(labels_text, preds_text, mode="all")
        for key, value in rouge_res.items():
            self.log("val-" + key, value, on_step=True)
        return rouge_res

    def validation_epoch_end(self, outputs) -> None:
        rouge_res = {}
        for key in outputs[0]:
            rouge_res[key] = np.mean([out[key] for out in outputs])

        for k, v in rouge_res.items():
            self.log("val-" + k + "-epoch", v, prog_bar=True)   
        print(rouge_res)   
        return rouge_res

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids, attention_mask, labels = batch
        labels[labels == -100] = self.tokenizer.pad_token_id
        preds = self.model.generate(input_ids, attention_mask)
        preds_text = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        return {"pre": preds_text, "source": input_text, "target": labels_text}

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        with open(self.args.save_file_path, "a", encoding="utf-8") as f_write:
            temp_save = {}
            for pre, source, target in zip(outputs["pre"], outputs["source"], outputs["target"]):
                temp_save["source"] = source
                temp_save["target"] = target
                temp_save["pre"] = pre
                f_write.write(json.dumps(temp_save, ensure_ascii=False))
            

    # def test_step(self, batch, batch_idx):
    #     input_ids, attention_mask, labels = batch
    #     preds = self.model(input_ids, attention_mask)
    #     self.log("test_loss", preds.loss, on_step=True)

