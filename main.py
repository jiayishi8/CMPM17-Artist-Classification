import pandas as pd
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

#dataset class
class ArtDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.artist_to_idx = {}

        #refered to stackoverflow
        # artist_folders = os.listdir(self.root_dir)
        artist_folders = []  
        for folder in os.listdir(self.root_dir):  #go trhu the directory
            folder_path = self.root_dir / folder    #get the path
            if os.path.isdir(folder_path):          #see if the path is a folder
                artist_folders.append(folder)   

        
        # Van Gogh, Degas, and Picasso artwork tm --> max to 300
        max_images_per_artist = {"Vincent_van_Gogh": 300, "Edgar_Degas": 300, "Pablo_Picasso": 300}
        
        for idx, artist in enumerate(artist_folders):       #go the the index and artist
            self.artist_to_idx[artist] = idx                
            image_files = os.listdir(self.root_dir / artist)        #get all the file
            
            if artist in max_images_per_artist:
                image_files = image_files[:max_images_per_artist[artist]]           #limti the file
            
            for image in image_files:
                self.image_paths.append(self.root_dir / artist / image)     #loops through each img file and stores path and its artist
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]            #get the img path given the idx
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


#  transformations & data augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  #  crop and resize
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),  # rotation, translation, and skew
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # color distortions
    transforms.RandomHorizontalFlip(p=0.5),  # flip images horizontally
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Randm blurring
    transforms.ToTensor()
])
# load dataset
dataset = ArtDataset("images", transform=transform)

# split into train and test sets (80/20 split)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# create dataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# seeing if the split is done for each artisit or nah
# counters for training & testing labels
train_label_counts = Counter()
test_label_counts = Counter()

# loop trhu and count occurrences in training set
for _, labels in train_loader:
    train_label_counts.update(labels.numpy())

# count occurrences in testing set
for _, labels in test_loader:
    test_label_counts.update(labels.numpy())

# verify 80:20 ratio for each artist
print()
print("Checking 80:20 Split per Artist:")
for label in sorted(train_label_counts.keys()):
    train_count = train_label_counts[label]
    test_count = test_label_counts[label]
    total = train_count + test_count
    
    expected_train = int(0.8 * total)
    expected_test = total - expected_train  # Should be ~20%

    print(f"Artist {label}: Train={train_count}, Test={test_count}, Expected Train={expected_train}, Expected Test={expected_test}")


# training loop
def train_loop(dataloader):
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: ")
        print(f"Inputs (Images Tensor): {images.shape}")
        print(f"Outputs (Labels): {labels}")
        if batch_idx == 1:  # print only first 2 batches cuz rest would take too long (jsut seeing if works)
            break

# testing loop
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
