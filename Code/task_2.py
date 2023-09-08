# Implement a program which extracts and stores feature descriptors
# for all the images in the data set.

# WARNING: Running this script will overwrite the contents of
# <git_root>/Code/tensor_database/


import os

import torch
from torchvision.datasets import Caltech101
from torchvision.transforms import Resize, Grayscale, transforms

from feature_extractors.color_moments import get_color_vector
from feature_extractors.HOG import get_hog_vector
from feature_extractors.resnet_50 import get_resnet50_feature_vectors
from utils.utilities import partition_to_grid


print("Downloading dataset if not present.")

dataset = Caltech101(root=os.path.abspath(os.path.join(os.path.dirname( __file__ ))), download=True)

print(dataset, "\n")

color_tensors = {}
hog_tensors = {}
avgpool_tensors = {}
layer3_tensors = {}
fc1000_tensors = {}

for i in range(len(dataset)):
    print(f"Processing image {i+1}/{len(dataset)} \t ({int(100*(i+1)/len(dataset))} %)", end='\r')

    image = dataset[i][0]

    # Skip images which are not RGB
    if image.mode != 'RGB':
        continue

    # Resize image to 300x100.
    image300x100 = Resize(size=(100, 300))(image)

    # Partition image into 10x10 grid.
    grid = partition_to_grid(image300x100, num_rows=10, num_cols=10, hor_pixels=30, ver_pixels=10)

    # Compute color moments vector.
    color_tensors[i] = get_color_vector(grid)

    # Convert image to grayscale.
    gs_image = Grayscale()(image300x100)

    # Partition grayscale image into 10x10 grid.
    gs_grid = partition_to_grid(gs_image, num_rows=10, num_cols=10, hor_pixels=30, ver_pixels=10)

    # Compute HOG moments vector.
    hog_tensors[i] = get_hog_vector(gs_grid)

    # Resize original image to 244x244.
    image224x224 = Resize(size=(224, 224))(image)

    # Convert to float tensor.
    img_tensor = transforms.Compose([transforms.ToTensor()])(image224x224)

    # Extract features from ResNet-50.
    avgpool_vector, layer3_vector, fc1000_vector = get_resnet50_feature_vectors(img_tensor)

    avgpool_tensors[i] = avgpool_vector
    layer3_tensors[i] = layer3_vector
    fc1000_tensors[i] = fc1000_vector


db_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'tensor_database'))

print("\n\nSaving color feature descriptors to '<db_dir>/color_tensors.pt")

torch.save(color_tensors, os.path.join(db_dir, 'color_tensors.pt'))

print("\nSaving HOG feature descriptors to '<db_dir>/hog_tensors.pt'")

torch.save(hog_tensors, os.path.join(db_dir, 'hog_tensors.pt'))

print("\nSaving ResNet-50 AvgPool feature descriptors to '<db_dir>/avgpool_tensors.pt'")

torch.save(avgpool_tensors, os.path.join(db_dir, 'avgpool_tensors.pt'))

print("\nSaving ResNet-50 Layer3 feature descriptors to '<db_dir>/layer3_tensors.pt'")

torch.save(layer3_tensors, os.path.join(db_dir, 'layer3_tensors.pt'))

print("\nSaving ResNet-50 Fc1000 feature descriptors to '<db_dir>/fc1000_tensors.pt'")

torch.save(fc1000_tensors, os.path.join(db_dir, 'fc1000_tensors.pt'))

print(
"""
> Feature vectors for all images are stored in a tensor dictionary per descriptor type.
> Stored in binary form, the key is the image ID, val is feature vector.
"""
)
