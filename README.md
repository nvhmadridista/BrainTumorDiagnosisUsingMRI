# Ví dụ cách dùng trong file training.ipynb

# 1. Import modul đã viết
from src.models.hybrid_model import BrainTumorModel

from src.dataloader.loader import get_data_loaders

from src.utils.helpers import init_weights

import torch

# 2. Lấy dữ liệu
data_dir = './brain-tumor-mri-dataset' # Sửa lại đường dẫn nếu cần

train_loader, val_loader, test_loader, classes = get_data_loaders(data_dir, batch_size=32)

# 3. Khởi tạo Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BrainTumorModel(num_classes=4).to(device)

# 4. Khởi tạo trọng số (Bước này quan trọng để tránh lỗi model không học)

model.apply(init_weights)

# 5. Bắt đầu viết vòng lặp training ở đây...
