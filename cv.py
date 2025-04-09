import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
class Config:
    DATA_DIR = '/home/d24csa005/.cache/kagglehub/datasets/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/data'  # Root directory containing nyu2_train and nyu2_test
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    EPOCHS = 20
    IMAGE_SIZE = (256, 256)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_MODEL_PATH = 'nyu_depth_estimation_model.pth'

# Custom Dataset for NYU Depth V2 with corrected structure
class NYUDataset(Dataset):
    def __init__(self, rgb_paths, depth_paths, transform=None):
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        # Load RGB image
        rgb_image = Image.open(self.rgb_paths[idx]).convert('RGB')
        
        # Load depth map (16-bit PNG)
        depth_image = Image.open(self.depth_paths[idx])
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)
        
        # Convert depth to numpy array and normalize
        depth_array = np.array(depth_image).astype(np.float32)
        depth_array = depth_array / 10000.0  # NYU depth is in millimeters, convert to meters
        depth_array = torch.from_numpy(depth_array).unsqueeze(0)  # Add channel dimension
        
        return rgb_image, depth_array

# Model Architecture (Same U-Net as before)
class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        
        # Encoder
        self.encoder1 = self._block(3, 64)
        self.encoder2 = self._block(64, 128)
        self.encoder3 = self._block(128, 256)
        self.encoder4 = self._block(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.decoder4 = self._block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.decoder3 = self._block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.decoder2 = self._block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.decoder1 = self._block(128, 64)
        
        # Output
        self.conv_out = nn.Conv2d(64, 1, 1)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv_out(dec1))

# Data Preparation for corrected NYU structure
def prepare_nyu_data():
    rgb_images = []
    depth_images = []
    
    # Scan through training directory
    train_dir = os.path.join(Config.DATA_DIR, 'nyu2_train')
    for scene_dir in os.listdir(train_dir):
        scene_path = os.path.join(train_dir, scene_dir)
        if os.path.isdir(scene_path):
            for file in os.listdir(scene_path):
                if file.endswith('.jpg'):
                    rgb_path = os.path.join(scene_path, file)
                    depth_path = os.path.join(scene_path, file.replace('.jpg', '.png'))
                    if os.path.exists(depth_path):
                        rgb_images.append(rgb_path)
                        depth_images.append(depth_path)
    
    # Scan through test directory
    test_dir = os.path.join(Config.DATA_DIR, 'nyu2_test')
    for file in os.listdir(test_dir):
        if file.endswith('_colors.png'):
            rgb_path = os.path.join(test_dir, file)
            depth_path = os.path.join(test_dir, file.replace('_colors.png', '_depth.png'))
            if os.path.exists(depth_path):
                rgb_images.append(rgb_path)
                depth_images.append(depth_path)
    
    # Split data (80% train, 20% validation)
    train_rgb, val_rgb, train_depth, val_depth = train_test_split(
        rgb_images, depth_images, test_size=0.2, random_state=42
    )
    
    return train_rgb, val_rgb, train_depth, val_depth

# Training Function
def train_nyu_model():
    # Prepare data
    train_rgb, val_rgb, train_depth, val_depth = prepare_nyu_data()
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = NYUDataset(train_rgb, train_depth, transform)
    val_dataset = NYUDataset(val_rgb, val_depth, transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = DepthEstimationModel().to(Config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        
        for rgb, depth in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS}'):
            rgb = rgb.to(Config.DEVICE)
            depth = depth.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(rgb)
            loss = criterion(outputs, depth)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * rgb.size(0)
        
        # Calculate average training loss
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for rgb, depth in val_loader:
                rgb = rgb.to(Config.DEVICE)
                depth = depth.to(Config.DEVICE)
                
                outputs = model(rgb)
                loss = criterion(outputs, depth)
                epoch_val_loss += loss.item() * rgb.size(0)
        
        # Calculate average validation loss
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f'\nEpoch {epoch+1}/{Config.EPOCHS}')
        print(f'Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), Config.SAVE_MODEL_PATH)
            print('Model saved!')
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('nyu_training_curve.png')
    plt.show()

if __name__ == '__main__':
    train_nyu_model()