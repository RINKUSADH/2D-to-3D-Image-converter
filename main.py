import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as n
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from cv import DepthEstimationModel
import numpy as np

# Load the trained model
@st.cache_resource
def load_nyu_model():
    model = DepthEstimationModel()  # Using the same class definition as above
    model.load_state_dict(torch.load('nyu_depth_estimation_model.pth', map_location='cpu'))
    model.eval()
    return model

# Function to process image and predict depth
def predict_nyu_depth(model, image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        depth = model(image)
    return depth.squeeze().numpy() * 10.0  # Scale to meters

# Function to create 3D point cloud
def create_nyu_point_cloud(rgb_image, depth_map):
    h, w = depth_map.shape
    rgb_image = np.array(rgb_image.resize((w, h))) / 255.0
    
    # Create grid of coordinates
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    x, y = np.meshgrid(x, y)
    
    # Use depth directly (converted to meters)
    z = depth_map
    
    # Flatten arrays
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    colors = rgb_image.reshape(-1, 3)
    
    # Downsample for performance
    step = 4
    return x[::step], y[::step], z[::step], colors[::step]

# Streamlit app
def nyu_app():
    st.title("NYU Depth V2 - 2D to 3D Converter")
    st.write("Upload a 2D image to convert it to a 3D depth map using NYU-trained model")
    
    # Load model
    model = load_nyu_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display original image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict depth
        depth_map = predict_nyu_depth(model, image)
        
        # Display depth map
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(depth_map, cmap='plasma')
        plt.colorbar(im, ax=ax, label='Depth (meters)')
        st.pyplot(fig)
        
        # Create 3D visualization
        st.subheader("3D Point Cloud Visualization")
        
        # Create point cloud
        x, y, z, colors = create_nyu_point_cloud(image, depth_map)
        
        # Create 3D plot
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        # Plot points with original colors
        ax_3d.scatter(x, y, z, c=colors, s=1, alpha=0.5)
        
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Depth (meters)')
        ax_3d.set_title('3D Point Cloud')
        ax_3d.view_init(elev=30, azim=45)  # Better initial view for indoor scenes
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Display 3D plot
        st.image(buf, caption="3D Point Cloud", use_column_width=True)
        
        # Add rotation animation
        st.write("Rotating 3D View")
        
        frames = []
        for angle in range(0, 360, 15):
            fig_anim = plt.figure(figsize=(8, 6))
            ax_anim = fig_anim.add_subplot(111, projection='3d')
            ax_anim.scatter(x, y, z, c=colors, s=1, alpha=0.5)
            ax_anim.view_init(elev=30, azim=angle)
            ax_anim.set_axis_off()
            
            buf_anim = io.BytesIO()
            plt.savefig(buf_anim, format='png', dpi=100, bbox_inches='tight')
            buf_anim.seek(0)
            frames.append(buf_anim)
            plt.close(fig_anim)
        
        st.image([Image.open(frame) for frame in frames], width=400)

if __name__ == '__main__':
    nyu_app()