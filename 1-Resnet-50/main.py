from datetime import datetime
import math
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor
import seaborn as sns
import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision

from image_data_loader import get_data_loader
from model import CIFARModule


# PyTorch
# Torchvision


def train_model(args, model, train_loader, val_loader, test_loader):
    checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename="{epoch:02d}-{val_acc:.4f}",
        save_weights_only=False,
        save_on_train_epoch_end=True,
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    device_stats = DeviceStatsMonitor()
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint, lr_monitor])
    trainer.fit(model, train_loader, val_loader)
    print("trainer.checkpoint_callback.best_model_path: ", str(trainer.checkpoint_callback.best_model_path))

    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    # train param
    parse.add_argument("--seed", type=int, default=13, help="random seed num")
    parse.add_argument("--max_epochs", type=int, default=100)
    parse.add_argument("--accumulate_grad_batches", type=int, default=1)
    parse.add_argument("--checkpoint_path", type=str, default="checkpoint")
    parse.add_argument("--add_note", type=str, default="debug", help='add descriptions to checkpoint_path')
    parse.add_argument("--default_root_dir", type=str, default="logs")
    parse.add_argument("--gpus", type=int, default=1, help="-1: use all gpus")

    parse.add_argument("--optimizer_name", type=str, default="Adamw")
    parse.add_argument("--scheduler_name", type=str, default="none")
    parse.add_argument("--lr", type=float, default=1e-3)
    parse.add_argument("--warmup_step", type=int, default=100)
    
    # model_param

    # data param
    parse.add_argument("--batch_size", type=int, default=128)

    args = parse.parse_args()

    # Function for setting the seed
    pl.seed_everything(args.seed)

    dt = datetime.now()
    save_path = "/" + dt.strftime('%m-%d-%H-%M-%S') + '-' + args.add_note + "-"
    args.checkpoint_path += save_path
    args.default_root_dir += save_path

    train_loader, val_loader, test_loader = get_data_loader(args)
    args.total_steps = math.ceil(args.max_epochs / args.accumulate_grad_batches * len(train_loader))
    if args.gpus == -1:
        args.gpus = max(0, torch.cuda.device_count())

    model = CIFARModule(args=args)

    print(args)
    train_model(args, model, train_loader, val_loader, test_loader)

