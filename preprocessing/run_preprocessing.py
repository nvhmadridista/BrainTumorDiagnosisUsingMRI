import os
import cv2
from tqdm import tqdm

from resize_normalize import load_and_resize, normalize_minmax
from noise_removal import remove_noise


# RAW_DIR = "dataset/Training"
RAW_DIR = "dataset/Validation"

# OUT_DIR = "data/processed/Training"
OUT_DIR = "data/processed/Validation"
SIZE = 224   # hoặc 256

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess_image(input_path, output_path):
    # 1. Load & resize
    img = load_and_resize(input_path, size=SIZE)

    # 2. Remove noise
    img = remove_noise(img)

    # 3. Normalize → [0,1]
    img = normalize_minmax(img)

    # 4. Convert lại về dạng 0–255 để lưu
    img = (img * 255).astype("uint8")

    # 5. Save
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def preprocess_folder():
    classes = os.listdir(RAW_DIR)

    for cls in classes:
        input_folder = os.path.join(RAW_DIR, cls)
        output_folder = os.path.join(OUT_DIR, cls)
        ensure_dir(output_folder)

        print(f"Preprocessing class: {cls}")

        for file in tqdm(os.listdir(input_folder)):
            in_path = os.path.join(input_folder, file)
            out_path = os.path.join(output_folder, file)

            try:
                preprocess_image(in_path, out_path)
            except:
                print("Lỗi ảnh:", in_path)

if __name__ == "__main__":
    preprocess_folder()
    print("\n DONE! Preprocessed images saved in data/processed/")
