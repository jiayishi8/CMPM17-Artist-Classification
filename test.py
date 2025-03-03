import pandas as pd
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("artists.csv")

# Define the dataset class
class ArtDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.artist_to_idx = {}

        artist_folders = [f for f in os.listdir(self.root_dir) if os.path.isdir(self.root_dir / f)]
       
        # Reduce artwork count for Van Gogh, Degas, and Picasso
        max_images_per_artist = {"Vincent_van_Gogh": 300, "Edgar_Degas": 300, "Pablo_Picasso": 300}
       
        for idx, artist in enumerate(artist_folders):
            self.artist_to_idx[artist] = idx
            image_files = os.listdir(self.root_dir / artist)
           
            if artist in max_images_per_artist:
                image_files = image_files[:max_images_per_artist[artist]]
           
            for image in image_files:
                self.image_paths.append(self.root_dir / artist / image)
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
       
        if self.transform:
            image = self.transform(image)
       
        return image, label

# Define transformations with data augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop and resize
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),  # Rotation, translation, and skew
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color distortions
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images horizontally
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Random blurring
    transforms.ToTensor()
])

# Load dataset
dataset = ArtDataset("images", transform=transform)

# Split into train and test sets (80/20 split)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
def train_loop(dataloader):
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: ")
        print(f"Inputs (Images Tensor): {images.shape}")
        print(f"Outputs (Labels): {labels}")
        if batch_idx == 1:  # Print only first 2 batches for readability
            break

# Testing loop
def test_loop(dataloader):
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: ")
        print(f"Inputs (Images Tensor): {images.shape}")
        print(f"Outputs (Labels): {labels}")
        if batch_idx == 1:
            break

print("Training Data:")
train_loop(train_loader)

print("Testing Data:")
test_loop(test_loader)
