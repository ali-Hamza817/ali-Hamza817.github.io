import os
import torch
import torch.optim as optim
from glob import glob

import monai
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset, list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandAffined, EnsureTyped
)

# --- CONFIGURATION ---
DATA_DIR = "../../datasets/dataset_5_kits23/kits23/data"
MODEL_SAVE_DIR = "../../models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

EPOCHS = 300
BATCH_SIZE = 2
LR = 1e-4
ROI_SIZE = (96, 96, 96)

# --- DATASET PREPARATION ---
def get_transforms():
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-135, a_max=215,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=ROI_SIZE,
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=0.5, spatial_size=ROI_SIZE,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1)
        ),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-135, a_max=215,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])
    return train_transforms, val_transforms

def get_data_dicts():
    # KiTS23 structure: kits23/data/case_00000/imaging.nii.gz and segmentation.nii.gz
    cases = sorted(glob(os.path.join(DATA_DIR, "case_*")))
    data_dicts = []
    for case in cases:
        img_path = os.path.join(case, "imaging.nii.gz")
        seg_path = os.path.join(case, "segmentation.nii.gz")
        if os.path.exists(img_path) and os.path.exists(seg_path):
            data_dicts.append({"image": img_path, "label": seg_path})
            
    print(f"Found {len(data_dicts)} valid KiTS23 cases.")
    return data_dicts

# --- TRAINING SCRIPT ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dicts = get_data_dicts()
    if len(data_dicts) == 0:
        print("No data found! Please run the 01_setup_datasets.sh script to download KiTS23 first.")
        return

    # Split 80/20
    train_size = int(0.8 * len(data_dicts))
    train_files, val_files = data_dicts[:train_size], data_dicts[train_size:]
    
    train_transforms, val_transforms = get_transforms()
    
    # Use standard MONAI Dataset for robust custom dataloading
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=list_data_collate)
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    
    # 3D U-Net Model
    # 4 Output classes: 0=bg, 1=kidney, 2=tumor, 3=cyst
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = optim.Adam(model.parameters(), LR)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    best_metric = -1
    best_metric_epoch = -1
    
    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= step
        print(f"Epoch {epoch + 1}/{EPOCHS}, Average Loss: {epoch_loss:.4f}")
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    # Using SlidingWindowInferer in real app, but direct forward pass for validation patches
                    val_outputs = model(val_inputs)
                    # compute metric
                    val_outputs = [monai.transforms.AsDiscrete(argmax=True, to_onehot=4)(i) for i in val_outputs]
                    val_labels = [monai.transforms.AsDiscrete(to_onehot=4)(i) for i in val_labels]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                print(f"  -> Validation Mean Dice: {metric:.4f}")
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    save_path = os.path.join(MODEL_SAVE_DIR, "3d_unet_kits23.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f"  -> Saved new best model to {save_path}")

if __name__ == "__main__":
    train()
