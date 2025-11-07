import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import time
from augmentation_c import ImageTransform



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
                if fname.startswith('image_'):
                    vis_path = os.path.join(class_dir, fname)
                    self.samples.append((vis_path, label_int))
        print(f"Found {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vis_path, label = self.samples[idx]
        image = self.transform(vis_path)
        return image, label

# Model 
class CNN(nn.Module):
    def __init__(self, num_classes=9):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(93312, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():
    transform = ImageTransform(size=(224, 224))
    # dataset = ChannelDataset(root_dir=root_dir, transform=transform)

    train_dataset = ChannelDataset(root_dir='data/train/day', transform=transform)
    # val_dataset   = ChannelDataset(root_dir='data/val/day',   transform=transform)
    # test_dataset  = ChannelDataset(root_dir='data/test/day',  transform=transform)


    writer = SummaryWriter(log_dir='logs/avocado_model')
    best_val_loss = float('inf')
    best_val_acc  = 0.0
    epochs_no_improve = 0
    early_stop_patience = 30


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device : ", device)
    model = CNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    epochs = 300
    

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

        current_loss = running_loss / total
        val_acc = correct / total
            
        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
        val_acc = f"{(val_correct / val_total):.4f}"

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {current_loss:.4f} Train Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"Epoch {epoch+1}: New best model saved (val_acc = {val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_avocado.pth")
    print("Model saved to models/cnn_avocado.pth")

if __name__ == '__main__':
    main()
