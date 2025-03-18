#This is Jiayi's code file

import torch
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
from PIL import Image

class ArtistClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ArtistClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "artist_classifier.pth"  
num_classes = 50  

model = ArtistClassifier(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

print("Model loaded successfully!")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

artist_folders = sorted([d for d in os.listdir("images") if os.path.isdir(os.path.join("images", d))])
artist_to_idx = {artist: idx for idx, artist in enumerate(artist_folders)}
idx_to_artist = {v: k for k, v in artist_to_idx.items()}

def predict_artist(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  
    
    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)
    
    predicted_artist = idx_to_artist[predicted_idx.item()]
    return predicted_artist

# example Usage
image_path = "VVGSelfPortrait.jpg"  
predicted_artist = predict_artist(image_path, model)
print(f"Predicted Artist: {predicted_artist}")




# Define the same transformation as used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test dataset (adjust path accordingly)
test_dataset = datasets.ImageFolder(root="images", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Ensure model is in evaluation mode
model.eval()

all_preds = []
all_labels = []

# Iterate through the test set and get predictions
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Create confusion matrix
cm = confusion_matrix(all_labels, all_preds)
artist_names = list(test_dataset.class_to_idx.keys())

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=artist_names, yticklabels=artist_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix of Artist Classifier")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()
