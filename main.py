import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image
import numpy as np
import math



df = pd.read_csv("artists.csv")

print(df.head())



art_path = Path('images')

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
print("Folders inside images:", os.listdir(art_path))

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





# Displays images.
num_images_to_display = 100  # num of images
num_cols = 10  # num of columns
num_rows = math.ceil(num_images_to_display / num_cols)  # num rows

plt.figure(figsize=(15, num_rows * 3))  # Set figure size
image_count = 0  # track number of images displayed

for images, labels in train_loader:
    for idx in range(len(images)):
        if image_count >= num_images_to_display:
            break  # Stop when target is reached

        # Convert tensor image to NumPy format
        image = images[idx].permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)  # Convert range [0,1] → [0,255]

        plt.subplot(num_rows, num_cols, image_count + 1)
        plt.imshow(image)
        plt.title(f"Label: {labels[idx].item()}")
        plt.axis('off')

        image_count += 1

    if image_count >= num_images_to_display:
        break  

plt.tight_layout()
plt.show()

print(df.info())
df = df.dropna(ignore_index=True) #drops all null values, also resets the index after dropping rows with missing values
df = df.drop_duplicates(ignore_index=True) #drops all duplicate values, also resets the index after dropping rows with missing values
print(df.info())
df = df.drop(columns=['bio', 'wikipedia'])
print(df['nationality'].unique())
print(df['genre'].unique())
print(df['years'].unique())
print(df.head())