# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("soumikrakshit/nyu-depth-v2")

# print("Path to dataset files:", path)

import kagglehub
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

 # Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Download the dataset
dataset_path = kagglehub.dataset_download("soumikrakshit/nyu-depth-v2")
print(f"Dataset downloaded to: {dataset_path}")