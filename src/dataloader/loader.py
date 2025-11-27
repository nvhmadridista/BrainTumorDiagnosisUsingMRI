import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_data_loaders(data_dir, batch_size=32, img_size=224):
    """
    Hàm tạo DataLoader cho Train, Val và Test
    Args:
        data_dir (str): Đường dẫn đến thư mục dataset (chứa folder Training và Testing)
        batch_size (int): Kích thước batch
        img_size (int): Kích thước ảnh resize
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    
    # Chuẩn hóa ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Transform cho Train (Có Augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        normalize
    ])

    # Transform cho Val/Test (Không Augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Load dữ liệu
    train_dir = os.path.join(data_dir, 'Training')
    test_dir = os.path.join(data_dir, 'Testing')

    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transforms)

    # Chia Train/Val (85/15)
    train_size = int(0.85 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, full_train_dataset.classes