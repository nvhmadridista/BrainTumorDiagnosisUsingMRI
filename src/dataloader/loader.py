import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

def get_data_loaders(data_dir, batch_size=32):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.classes
