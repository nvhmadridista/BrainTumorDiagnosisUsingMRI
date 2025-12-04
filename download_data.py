import opendatasets as od
import os
import shutil

# Link dataset
dataset_url = 'https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset'

# Thư mục đích theo cấu trúc PDF
data_dir = './data/kaggle'

# Tạo thư mục nếu chưa có
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Tải dữ liệu
print("⏳ Đang tải dữ liệu từ Kaggle...")
od.download(dataset_url, data_dir)

# Xử lý thư mục lồng nhau (Kaggle thường tải về dạng data/kaggle/brain-tumor-mri-dataset/Training)
# Đoạn này giúp di chuyển file ra đúng chỗ data/kaggle/Training
downloaded_folder = os.path.join(data_dir, 'brain-tumor-mri-dataset')
if os.path.exists(downloaded_folder):
    for item in os.listdir(downloaded_folder):
        shutil.move(os.path.join(downloaded_folder, item), data_dir)
    os.rmdir(downloaded_folder) # Xóa thư mục rỗng

print(f"✅ Đã tải và sắp xếp dữ liệu xong tại: {data_dir}")