from datetime import datetime
import argparse
import math

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor
import torch
from transformers import T5Tokenizer

from get_data_loader import get_data_loader
from model import CIFARModule
from util.util import seed_everything

# python main.py --scheduler_name lr_schedule --lr 2e-4
# PyTorch
# Torchvision



def train_model(args, model, train_loader, val_loader):
    checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename="{epoch:02d}-{val-rouge-1:.4f}",
        save_weights_only=False,
        save_on_train_epoch_end=True,
        monitor='val-rouge-1',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    device_stats = DeviceStatsMonitor()
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint, lr_monitor])
    trainer.fit(model, train_loader, val_loader)
    print("trainer.checkpoint_callback.best_model_path: ", str(trainer.checkpoint_callback.best_model_path))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    # train param
    parse.add_argument("--seed", type=int, default=13, help="random seed num")
    parse.add_argument("--max_epochs", type=int, default=20)
    parse.add_argument("--checkpoint_path", type=str, default="checkpoint")
    parse.add_argument("--add_note", type=str, default="debug", help='add descriptions to checkpoint_path')
    parse.add_argument("--default_root_dir", type=str, default="logs")
    parse.add_argument("--gpus", type=int, default=1, help="-1: use all gpus")
    parse.add_argument("--optimizer_name", type=str, default="Adamw")
    parse.add_argument("--scheduler_name", type=str, default="none")
    parse.add_argument("--lr", type=float, default=2e-5)
    parse.add_argument("--warmup_step", type=int, default=100)
    parse.add_argument("--num_sanity_val_steps", type=int, default=0, 
        help="Sanity check runs n validation batches before starting the training routine.")
    parse.add_argument("--check_val_every_n_epoch", type=int, default=1)
    
    # model_param
    parse.add_argument("--text_model", type=str, default="mengzi-t5")
    parse.add_argument("--load_model_path", type=str, default="model/mengzi-t5-base/")

    # data param
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--accumulate_grad_batches", type=int, default=8)
    parse.add_argument("--data_path", type=str, default="data/csl_data.json")
    parse.add_argument("--num_workers", type=int, default=8)

    # debug
    parse.add_argument("--limit_train_batches", type=float, default=1.0)
    parse.add_argument("--limit_val_batches", type=float, default=1.0)

    args = parse.parse_args()

    # Function for setting the seed
    pl.seed_everything(args.seed)
    seed_everything(args.seed)

    dt = datetime.now()
    save_path = "/" + dt.strftime('%m-%d-%H-%M-%S') + '-' + args.add_note + "-"
    args.checkpoint_path += save_path
    args.default_root_dir += save_path

    tokenizer = None
    if args.text_model == "mengzi-t5":
        tokenizer = T5Tokenizer.from_pretrained(args.load_model_path)
    train_loader, val_loader = get_data_loader(args, tokenizer)
    args.total_steps = math.ceil(args.max_epochs / args.accumulate_grad_batches * len(train_loader))
    if args.gpus == -1:
        args.gpus = max(0, torch.cuda.device_count())

    model = CIFARModule(args=args, tokenizer=tokenizer)

    print(args)
    train_model(args, model, train_loader, val_loader)

