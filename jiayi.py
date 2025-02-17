#This is Jiayi's code file

import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os



csv_path = "C:/Users/theet/OneDrive/art-dataset/artists.csv"
df = pd.read_csv(csv_path)

print(df.head())



art_path = "C:/Users/theet/OneDrive/art-dataset/images/images"

#this resizes the images to standard 224,224
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=art_path, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# checking if everything works
for images, labels in train_loader:
    print(images.shape, labels)
    break



# Print what's inside the image folder
print("Folders inside resized:", os.listdir(art_path))

artist_folders = [f for f in os.listdir(art_path) if os.path.isdir(os.path.join(art_path, f))]
print("Filtered artist folders:", artist_folders)

artist_counts = {artist: len(os.listdir(os.path.join(art_path, artist))) for artist in artist_folders}



# Convert dictionary to DataFrame
artist_df = pd.DataFrame(list(artist_counts.items()), columns=['Artist', 'Number of Artworks'])

# Sort by number of artworks
artist_df = artist_df.sort_values(by='Number of Artworks', ascending=False)

# Plot full dataset
plt.figure(figsize=(15, 6))
plt.bar(artist_df['Artist'], artist_df['Number of Artworks'], color='skyblue')

plt.xticks(rotation=90)  # Rotate artist names for visibility
plt.xlabel("Artists")
plt.ylabel("Number of Artworks")
plt.title("Number of Artworks per Artist")

plt.show()
