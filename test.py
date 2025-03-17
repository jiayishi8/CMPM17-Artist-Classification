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
image_path = "dove.jpg"  
predicted_artist = predict_artist(image_path, model)
print(f"Predicted Artist: {predicted_artist}")
