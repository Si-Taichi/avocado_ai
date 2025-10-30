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
        self.nir_transform = T.Compose([
            T.Resize(self.size),
            T.ToTensor()
        ])

    def __call__(self, vis_path: str, nir_path: str):
        vis = Image.open(vis_path).convert('RGB')
        nir = Image.open(nir_path).convert('L')

        vis_t = self.vis_transform(vis)
        nir_t = self.nir_transform(nir) 

        four_channel = torch.cat((vis_t, nir_t), dim=0)
        return four_channel
