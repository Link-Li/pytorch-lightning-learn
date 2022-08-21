
import random
import json

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class TextDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        self.source_text_list = []
        self.target_text_list = []
        with open(args.data_path, "r", encoding="utf-8") as f:
            file_content = json.load(f)
            # random.shuffle(file_content)
            for data_line in file_content:
                self.source_text_list.append(data_line["content"])
                self.target_text_list.append(data_line["title"])

    def __len__(self, ):
        return len(self.source_text_list)

    def __getitem__(self, index):
        return self.source_text_list[index], self.target_text_list[index]


class Collate():
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        source_text_list = [b[0] for b in batch_data]
        target_text_list = [b[1] for b in batch_data]

        source_text_ids = self.tokenizer(source_text_list, return_tensors="pt", truncation=True, padding=True, max_length=512)
        target_text_ids = self.tokenizer(target_text_list, return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids
        target_text_ids[target_text_ids == self.tokenizer.pad_token_id] = -100

        return source_text_ids.input_ids, source_text_ids.attention_mask, target_text_ids


def get_data_loader(args, tokenizer):
    dataset = TextDataset(args)

    train_set, val_set = random_split(dataset, [int(len(dataset)*0.9), len(dataset) - int(len(dataset)*0.9)])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, collate_fn=Collate(tokenizer), pin_memory=True)
    
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, collate_fn=Collate(tokenizer), pin_memory=True)

    return train_loader, val_loader