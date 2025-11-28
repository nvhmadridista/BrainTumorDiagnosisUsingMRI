import os
import cv2
from tqdm import tqdm

from resize_normalize import load_and_resize, normalize_minmax
from noise_removal import remove_noise


SIZE = 224  

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess_image(input_path, output_path):

    img = load_and_resize(input_path, size=SIZE)

    img = remove_noise(img)
    
    img = normalize_minmax(img)

    img = (img * 255).astype("uint8")
    
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
def run_main_preprocess(RAW_DIR_IN, OUT_DIR_IN):
    global RAW_DIR, OUT_DIR 
    RAW_DIR = RAW_DIR_IN
    OUT_DIR = OUT_DIR_IN
    
    preprocess_folder()
    print(f"\n DONE! Preprocessed images from {RAW_DIR_IN} saved in {OUT_DIR_IN}")

if __name__ == "__main__":
    preprocess_folder()
    print("\n DONE! Preprocessed images saved in data/processed/")