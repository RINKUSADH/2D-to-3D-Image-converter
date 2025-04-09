# utils/download_model.py
import requests
import os

def download_model():
    model_path = "nyu_depth_estimation_model.pth"
    if not os.path.exists(model_path):
        dropbox_url = "https://dl.dropboxusercontent.com/scl/fi/9up2a8m1rg7cb9rk3uvep/nyu_depth_estimation_model.pth?rlkey=p2072fpgz18b2u2tdwpbw1y2s&st=wuoncs2y&dl=0/nyu_depth_estimation_model.pth"
        response = requests.get(dropbox_url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return model_path
