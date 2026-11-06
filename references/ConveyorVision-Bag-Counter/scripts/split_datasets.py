import os
import random
import shutil

def split_dataset(source_dir, train_dir, val_dir, test_dir,
                  train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
        raise ValueError("Train, validation, and test ratios must sum to 1")
    files = [name for name in os.listdir(source_dir)
             if os.path.isfile(os.path.join(source_dir, name))]
    random.Random(seed).shuffle(files)
    n_train = int(len(files) * train_ratio)
    n_val = int(len(files) * val_ratio)
    splits = ((files[:n_train], train_dir),
              (files[n_train:n_train+n_val], val_dir),
              (files[n_train+n_val:], test_dir))
    for names, destination in splits:
        os.makedirs(destination, exist_ok=True)
        for name in names:
            shutil.copy2(os.path.join(source_dir, name),
                         os.path.join(destination, name))

if __name__ == "__main__":
    split_dataset("path/to/images", "path/to/train",
                  "path/to/val", "path/to/test")
