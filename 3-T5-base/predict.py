from datetime import datetime
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor
import torch
from transformers import T5Tokenizer

from get_data_loader import get_data_loader
from model import CIFARModule


# PyTorch
# Torchvision



if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    # train param
    parse.add_argument("--seed", type=int, default=13, help="random seed num")
    parse.add_argument("--max_epochs", type=int, default=100)
    parse.add_argument("--checkpoint_path", type=str, default="checkpoint")
    parse.add_argument("--add_note", type=str, default="debug", help='add descriptions to checkpoint_path')
    parse.add_argument("--default_root_dir", type=str, default="logs")
    parse.add_argument("--gpus", type=int, default=1, help="-1: use all gpus")
    parse.add_argument("--optimizer_name", type=str, default="Adamw")
    parse.add_argument("--scheduler_name", type=str, default="none")
    parse.add_argument("--lr", type=float, default=2e-5)
    parse.add_argument("--warmup_step", type=int, default=100)
    parse.add_argument("--num_sanity_val_steps", type=int, default=2, 
        help="Sanity check runs n validation batches before starting the training routine.")
    
    # model_param
    parse.add_argument("--text_model", type=str, default="mengzi-t5")
    parse.add_argument("--load_model_path", type=str, default="model/mengzi-t5-base/")
    parse.add_argument("--loader_ckpt_path", type=str, default="", help="ckpt file from train")

    # data param
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--accumulate_grad_batches", type=int, default=8)
    parse.add_argument("--data_path", type=str, default="data/csl_data.json")
    parse.add_argument("--num_workers", type=int, default=8)
    parse.add_argument("--save_file_path", type=str, default="output")

    # debug
    parse.add_argument("--limit_predict_batches", type=float, default=0.05)

    args = parse.parse_args()

    if args.loader_ckpt_path == "":
        args.loader_ckpt_path = "checkpoint/08-16-16-07-32-debug-/epoch=08-rouge-1=0.6598.ckpt"
    args.save_file_path += "/" + args.loader_ckpt_path.split("/")[-2] + ".json"
    if os.path.exists(args.save_file_path):
        os.remove(args.save_file_path)

    # Function for setting the seed
    pl.seed_everything(args.seed)

    dt = datetime.now()
    save_path = "/" + dt.strftime('%m-%d-%H-%M-%S') + '-' + args.add_note + "-"
    args.checkpoint_path += save_path
    args.default_root_dir += save_path

    tokenizer = None
    if args.text_model == "mengzi-t5":
        tokenizer = T5Tokenizer.from_pretrained(args.load_model_path)
    train_loader, val_loader = get_data_loader(args, tokenizer)
    args.total_steps = args.max_epochs // args.accumulate_grad_batches * len(train_loader)
    if args.gpus == -1:
        args.gpus = max(0, torch.cuda.device_count())

    model = CIFARModule(args=args, tokenizer=tokenizer)
    # model = model.load_from_checkpoint(checkpoint_path=args.loader_ckpt_path, map_location="cpu")

    print(args)
    trainer = pl.Trainer.from_argparse_args(args, logger=False)
    trainer.predict(model, val_loader, ckpt_path=args.loader_ckpt_path)
    # trainer.predict(model, val_loader)

