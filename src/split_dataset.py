import os
import shutil
import random
from tqdm import tqdm

def split_dataset(
    raw_dir="data/kaggle/raw",
    output_dir="data",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    random.seed(seed)

    classes = os.listdir(raw_dir)
    classes = [c for c in classes if os.path.isdir(os.path.join(raw_dir, c))]

    print("Found classes:", classes)

    # Create output folders
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    # Process each class independently
    for cls in classes:
        cls_path = os.path.join(raw_dir, cls)
        images = os.listdir(cls_path)

        images = [img for img in images if img.lower().endswith((".jpg", ".png", ".jpeg"))]

        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        print(f"\nClass: {cls}")
        print(f"Total: {n_total} → Train {len(train_imgs)}, Val {len(val_imgs)}, Test {len(test_imgs)}")

        # Copy images
        for img_name, split in [
            (train_imgs, "train"),
            (val_imgs, "val"),
            (test_imgs, "test")
        ]:
            for img in tqdm(img_name, desc=f"Copying {cls} to {split}"):
                src = os.path.join(cls_path, img)
                dst = os.path.join(output_dir, split, cls, img)
                shutil.copy(src, dst)

    print("\nDataset splitting completed successfully!")


if __name__ == "__main__":
    split_dataset()
