import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
from augmentation import FourChannelImageTransform

class CNN4Channel(nn.Module):
    def __init__(self, num_classes=9):
        super(CNN4Channel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32 * 53 * 53, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def predict(vis_path, nir_path, model_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN4Channel(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = FourChannelImageTransform(size=(224,224))
    image = transform(vis_path, nir_path).unsqueeze(0).to(device)  # batch size 1
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output.data, 1)
        print(f"Predicted class (day): {class_names[pred.item()]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict avocado day using 4-channel image")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--visible", type=str, required=True, help="Path to visible image")
    parser.add_argument("--nir", type=str, required=True, help="Path to NIR image")
    parser.add_argument("--classes", type=str, nargs='+', default=[str(i) for i in range(1,10)],
                        help="List of class names (e.g. days 1-9)")
    args = parser.parse_args()

    predict(args.visible, args.nir, args.model, args.classes)
