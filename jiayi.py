#This is Jiayi's code file

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay,precision_score,recall_score,f1_score



#not using artist.csv anymore, all alternations and augmentations done to the file are dropped, since file is not being used anymore
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

# define the model
class ArtistClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ArtistClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1)  # Dynamically adjusts spatial size
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),  # No need for 128 * 28 * 28 anymore
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# get number of artists
num_classes = len(dataset.artist_to_idx)
model = ArtistClassifier(num_classes)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training function
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # prints loss every 10 batches instead of breaking
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss/len(train_loader):.4f}")


# test function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    batch_idx = 1
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print("hellp")
            accuracy = 100 * correct / total
            print(accuracy)
            
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# train and evaluate the model
epochs = 15
train_model(model, train_loader, criterion, optimizer, epochs)
test_model(model, test_loader)
def compute_confusion_matrix(model, test_loader, class_names):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.xticks(rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

# call the function after training and testing
compute_confusion_matrix(model, test_loader, list(dataset.artist_to_idx.keys()))
