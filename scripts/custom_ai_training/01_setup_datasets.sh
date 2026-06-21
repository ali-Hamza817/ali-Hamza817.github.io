#!/bin/bash
# 01_setup_datasets.sh
# Downloads and organizes the Kaggle and KiTS23 datasets for custom AI training.

set -e

BASE_DIR="/home/administrator/Desktop/RCC"
DATASETS_DIR="$BASE_DIR/datasets"

# 1. Setup Kaggle Dataset
KAGGLE_DIR="$DATASETS_DIR/dataset_4_kaggle_validator"
echo "=== Setting up Kaggle CT Kidney Dataset ==="
mkdir -p "$KAGGLE_DIR"
cd "$KAGGLE_DIR"

if [ ! -d "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone" ]; then
    echo "Downloading from Kaggle... (Ensure kaggle.json is in ~/.kaggle/)"
    pip install kaggle
    kaggle datasets download nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone
    unzip -q ct-kidney-dataset-normal-cyst-tumor-and-stone.zip
    rm ct-kidney-dataset-normal-cyst-tumor-and-stone.zip
else
    echo "Kaggle dataset already exists."
fi

# 2. Setup KiTS23 Dataset
KITS_DIR="$DATASETS_DIR/dataset_5_kits23"
echo "=== Setting up KiTS23 Segmentation Dataset ==="
mkdir -p "$KITS_DIR"
cd "$KITS_DIR"

if [ ! -d "kits23" ]; then
    echo "Cloning KiTS23 repository..."
    git clone https://github.com/neheller/kits23
    cd kits23
    echo "Installing KiTS23 toolkit..."
    pip install -e .
    
    echo ""
    echo "============================================================"
    echo "NOTE: The KiTS23 download requires ~50GB of disk space."
    echo "To start the multi-hour download, run this command manually:"
    echo "  cd $KITS_DIR/kits23 && kits23_download_data"
    echo "============================================================"
else
    echo "KiTS23 repository already cloned."
fi

echo "Setup script completed successfully!"
