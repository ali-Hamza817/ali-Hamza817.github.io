import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from glob import glob

# --- CONFIGURATION ---
DATA_DIR = "../../datasets/dataset_4_kaggle_validator/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
NEGATIVE_DIR = "../../datasets/dataset_4_kaggle_validator/Negative-Examples"
MODEL_SAVE_DIR = "../../models"
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(NEGATIVE_DIR, exist_ok=True)

# --- DATASET ---
class KidneyValidatorDataset(Dataset):
    def __init__(self, data_dir, negative_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        # 1. Load Positive (Kidney) Images
        # The Kaggle dataset has Normal, Cyst, Tumor, Stone folders. All are valid kidney CTs.
        if os.path.exists(data_dir):
            for cls_folder in os.listdir(data_dir):
                cls_path = os.path.join(data_dir, cls_folder)
                if os.path.isdir(cls_path):
                    for img_name in os.listdir(cls_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(cls_path, img_name))
                            self.labels.append(1) # 1 = Kidney
                            
        # 2. Load Negative (Not Kidney) Images
        if os.path.exists(negative_dir):
            for img_name in os.listdir(negative_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(negative_dir, img_name))
                    self.labels.append(0) # 0 = Not Kidney
                    
        print(f"Loaded {len([l for l in self.labels if l==1])} positive and {len([l for l in self.labels if l==0])} negative samples.")
        if len([l for l in self.labels if l==0]) == 0:
            print("WARNING: You have 0 negative examples! Please add non-kidney images to:", negative_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_transforms():
    # Standard EfficientNet-B0 resolution is 224x224
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform

# --- TRAINING SCRIPT ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = KidneyValidatorDataset(DATA_DIR, NEGATIVE_DIR, transform=get_transforms())
    
    if len(dataset) == 0:
        print("Dataset is empty. Run 01_setup_datasets.sh first.")
        return
        
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Init Model
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Replace classification head for 2 classes (Kidney vs Not-Kidney)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Starting Training Loop...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(MODEL_SAVE_DIR, "ct_validator.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved new best model to {save_path}")

if __name__ == "__main__":
    train()
