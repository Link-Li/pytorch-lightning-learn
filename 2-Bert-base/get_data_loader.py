
import random

import torch
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, random_split


class TextDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        self.data_list = []
        self.label_list = []
        with open(args.data_path, "r", encoding="utf-8") as f:
            file_content = f.readlines()[1:]
            random.shuffle(file_content)
            for data_line in file_content:
                self.data_list.append(data_line[2:])
                self.label_list.append(int(data_line[0]))

    def __len__(self, ):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index], self.label_list[index]


class Collate():
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        data_list = [b[0] for b in batch_data]
        labels = torch.LongTensor([b[1] for b in batch_data])

        input_tokenizer = self.tokenizer(data_list, return_tensors="pt", truncation=True, padding=True, max_length=512)
        return input_tokenizer.input_ids, input_tokenizer.attention_mask, labels



def get_data_loader(args, tokenizer):
    dataset = TextDataset(args)

    train_set, val_set = random_split(dataset, [len(dataset)-2000, 2000])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, collate_fn=Collate(tokenizer), pin_memory=True)
    
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, collate_fn=Collate(tokenizer), pin_memory=True)

    return train_loader, val_loader