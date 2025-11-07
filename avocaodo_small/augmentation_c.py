import os
from PIL import Image
import torchvision.transforms as T
import torch

# Pre-processing for images

class ImageTransform:
    def __init__(self, size=(224, 224)):
        self.size = size
        self.vis_transform = T.Compose([
            T.Resize(self.size),
            T.ToTensor()
        ])

    def __call__(self, vis_path: str):
        vis = Image.open(vis_path).convert('RGB')

        vis_t = self.vis_transform(vis)

        channels = vis_t
        return channels
