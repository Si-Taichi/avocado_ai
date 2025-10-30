import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import time
from augmentation import ImageTransform



class ChannelDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for class_label in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_label)
            if not os.path.isdir(class_dir):
                continue
            label_int = int(class_label) - 1 
            for fname in os.listdir(class_dir):
                if fname.startswith('vis_'):
                    vis_path = os.path.join(class_dir, fname)
                    nir_fname = fname.replace('vis_', 'nir_')
                    nir_path = os.path.join(class_dir, nir_fname)
                    if os.path.exists(nir_path):
                        self.samples.append((vis_path, nir_path, label_int))
        print(f"Found {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vis_path, nir_path, label = self.samples[idx]
        image = self.transform(vis_path, nir_path)
        return image, label

# Model 
class CNN(nn.Module):
    def __init__(self, num_classes=9):
        super(CNN, self).__init__()
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

def main():
    root_dir = 'day'  # folder containing sub-folders “1”, “2”,… “9”
    transform = ImageTransform(size=(224, 224))
    dataset = ChannelDataset(root_dir=root_dir, transform=transform)

    writer = SummaryWriter(log_dir='logs/avocado_model')
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 5


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    epochs = 30

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Calculate current loss
            current_loss = running_loss / total
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), 'models/best_model.pth')
            else:
                epochs_no_improve += 1
                
            # Log metrics to tensorboard
            writer.add_scalar('Loss/train', current_loss, epoch)
            writer.add_scalar('Accuracy/train', correct / total, epoch)

            if epochs_no_improve >= early_stop_patience:
                print("Early stopping: no improvement for %d epochs" % early_stop_patience)
                break


        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_avocado.pth")
    print("Model saved to models/cnn_avocado.pth")

if __name__ == '__main__':
    main()
