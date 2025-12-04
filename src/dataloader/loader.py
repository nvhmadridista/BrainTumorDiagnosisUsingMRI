import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

def get_data_loaders(data_dir, batch_size=32, img_size=224, seed=42):
    """
    Hàm tạo DataLoader chuẩn: Train có Augment, Val/Test thì KHÔNG.
    """
    
    # 1. Định nghĩa Transforms
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

    # Transform cho Val và Test (Sạch, chỉ Resize + Normalize)
    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    # 2. Load dữ liệu
    train_dir = os.path.join(data_dir, 'Training')
    test_dir = os.path.join(data_dir, 'Testing')

    # MẸO QUAN TRỌNG: Load thư mục Training 2 lần với 2 transform khác nhau
    # dataset_for_train: Dùng để lấy dữ liệu train (có augment)
    # dataset_for_val: Dùng để lấy dữ liệu val (không augment)
    dataset_for_train = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    dataset_for_val = datasets.ImageFolder(root=train_dir, transform=val_test_transforms)
    
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transforms)

    # 3. Chia chỉ số (Indices) thủ công để đảm bảo không bị trùng lặp
    num_train = len(dataset_for_train)
    indices = list(range(num_train))
    split = int(np.floor(0.15 * num_train)) # 15% cho Validation

    # Đặt seed để lần nào chạy cũng chia giống nhau (quan trọng để tái lập kết quả)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    # 4. Tạo Subset từ đúng dataset tương ứng
    # Train subset lấy từ dataset có augment
    train_dataset = Subset(dataset_for_train, train_idx)
    # Val subset lấy từ dataset sạch (không augment)
    val_dataset = Subset(dataset_for_val, val_idx)

    # 5. Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, dataset_for_train.classes