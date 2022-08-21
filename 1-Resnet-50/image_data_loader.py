

import torch
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.utils.data as data

def get_data_loader(args):

    DATASET_PATH = "../data/"

    train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)

    return train_loader, val_loader, test_loader