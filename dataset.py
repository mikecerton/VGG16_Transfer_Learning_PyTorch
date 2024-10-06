import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]  # Get the image path
        image = Image.open(img_path).convert("RGB")  # Load image

        label = self.data_frame.iloc[idx, 1]  # Get the label

        if self.transform:
            image = self.transform(image)  # Apply transformations if any

        return image, label