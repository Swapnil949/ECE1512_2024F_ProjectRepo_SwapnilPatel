from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MHISTDataset(Dataset):
    def __init__(self, split_dir, train=True, transform=None):
        """
        Args:
            split_dir (str): Directory containing the split images (e.g., 'images-split/train' or 'images-split/test').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split_dir = Path(split_dir)
        if train:
            self.split_dir = self.split_dir / 'train'
        else:
            self.split_dir = self.split_dir / 'test'
    
        self.transform = transform

        # Map folder names to class labels
        self.label_map = {"HP": 0, "SSA": 1}

        # Gather image file paths and their corresponding labels
        self.image_paths = []
        self.labels = []

        for label_name, label_value in self.label_map.items():
            class_dir = self.split_dir / label_name
            if class_dir.exists():
                for img_path in class_dir.rglob('*.*'):  # Recursively list all image files
                    self.image_paths.append(img_path)
                    self.labels.append(label_value)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the image path and corresponding label
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        label = self.labels[idx]
        # map label to class name, if 0 then HP, if 1 then SSA

        name = "HP" if label == 0 else "SSA"

        if self.transform:
            image = self.transform(image)

        return image, label