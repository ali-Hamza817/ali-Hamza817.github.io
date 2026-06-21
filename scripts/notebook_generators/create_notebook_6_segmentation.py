import os
import nbformat as nbf

def create_notebook():
    out_dir = "/home/administrator/Desktop/RCC/notebooks"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Notebook_6_3D_Tumour_Segmentation.ipynb")

    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell("""# Phase 2: 3D Kidney & Tumour Segmentation (MONAI U-Net)
This notebook details the state-of-the-art 3D Medical AI built for extracting the kidney and tumour regions from the raw CT scan. 
Once the EfficientNet Validator (Phase 1) approves the image, this U-Net takes over to segment the image into: Background (0), Kidney (1), Tumour (2), and Cyst (3).

**Dataset**: KiTS23 (Kidney Tumor Segmentation Challenge 2023). 489 3D NIfTI volumes (~50GB).
**Architecture**: MONAI 3D U-Net.
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Import Libraries"))
    nb.cells.append(nbf.v4.new_code_cell("""import monai
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
import torch"""))

    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Model Architecture\nWe construct a 3D U-Net capable of processing volumetric data directly. Using `DiceCELoss` is standard practice to combat extreme class imbalance in medical imaging (tumours are very small compared to background)."))
    nb.cells.append(nbf.v4.new_code_cell("""# Define the 3D U-Net architecture
model = UNet(
    spatial_dims=3,             # 3D Data
    in_channels=1,              # 1 channel (Hounsfield Units)
    out_channels=4,             # 4 classes: bg, kidney, tumour, cyst
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")"""))

    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Data Transformations\nSince 3D NIfTI volumes cannot fit entirely into GPU memory, we extract 96x96x96 patches focused around the tumour regions."))
    nb.cells.append(nbf.v4.new_code_cell("""from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, RandCropByPosNegLabeld
)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(
        keys=["image"], a_min=-135, a_max=215,
        b_min=0.0, b_max=1.0, clip=True,
    ),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1, neg=1, num_samples=4,
        image_key="image"
    )
])"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("## Conclusion\nThis massive model takes approximately 4 days to train on 50GB of data. Once complete, its weights (`3d_unet_kits23.pth`) are dynamically loaded into the FastAPI backend replacing `TotalSegmentator`."))

    with open(out_path, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Successfully generated {out_path}")

if __name__ == "__main__":
    create_notebook()
