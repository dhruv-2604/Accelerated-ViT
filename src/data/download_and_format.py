"""
Tiny ImageNet Dataset Preparation Script

This script handles the messy reality of research datasets:
1. Downloads the dataset from Stanford's servers
2. Extracts the zip file
3. Reorganizes the validation set into a usable structure

Why is this needed?
- Tiny ImageNet's validation folder is FLAT (all images in one directory)
- PyTorch's ImageFolder expects: val/class_name/image.jpg
- We need to parse val_annotations.txt and move files accordingly

This script runs ONCE on the PACE login node (not the compute node!)
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import urllib.request
import zipfile


# ============================================================================
# CONSTANTS
# ============================================================================

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_NAME = "tiny-imagenet-200"


# ============================================================================
# STEP 1: Download with Progress Bar
# ============================================================================

class DownloadProgressBar(tqdm):
    """
    Custom progress bar for urllib downloads.

    This is a nice UX touch - shows download progress in real-time.
    """
    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_dataset(output_dir: Path):
    """
    Downloads Tiny ImageNet zip file if not already present.

    Args:
        output_dir: Directory where dataset will be stored
    """
    zip_path = output_dir / f"{DATASET_NAME}.zip"

    # Check if already downloaded
    if zip_path.exists():
        print(f"✅ Dataset zip already exists at: {zip_path}")
        return zip_path

    print(f"📥 Downloading Tiny ImageNet from {TINY_IMAGENET_URL}")
    print(f"   This may take 5-10 minutes (~237 MB)...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download with progress bar
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as pbar:
        urllib.request.urlretrieve(
            TINY_IMAGENET_URL,
            zip_path,
            reporthook=pbar.update_to
        )

    print(f"✅ Download complete: {zip_path}")
    return zip_path


# ============================================================================
# STEP 2: Extract Zip File
# ============================================================================

def extract_dataset(zip_path: Path, output_dir: Path):
    """
    Extracts the Tiny ImageNet zip file.

    Args:
        zip_path: Path to the downloaded zip file
        output_dir: Directory where dataset will be extracted
    """
    extracted_dir = output_dir / DATASET_NAME

    # Check if already extracted
    if extracted_dir.exists() and (extracted_dir / "train").exists():
        print(f"✅ Dataset already extracted at: {extracted_dir}")
        return extracted_dir

    print(f"📦 Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract with progress bar
        members = zip_ref.namelist()
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, output_dir)

    print(f"✅ Extraction complete: {extracted_dir}")
    return extracted_dir


# ============================================================================
# STEP 3: The Critical Part - Reorganize Validation Set
# ============================================================================

def reorganize_validation_set(dataset_dir: Path):
    """
    Reorganizes Tiny ImageNet validation set for PyTorch ImageFolder.

    PROBLEM:
        Raw structure:
            val/images/val_0.JPEG, val_1.JPEG, ...  (10,000 images in one folder!)
            val/val_annotations.txt  (maps image -> class)

        ImageFolder expects:
            val/n01443537/val_0.JPEG
            val/n01443537/val_5.JPEG
            val/n01629819/val_1.JPEG
            ...

    SOLUTION:
        1. Parse val_annotations.txt to create image -> class mapping
        2. Create class subdirectories in val/
        3. Move images from val/images/ to val/class_name/

    Args:
        dataset_dir: Root directory of extracted Tiny ImageNet
    """
    val_dir = dataset_dir / "val"
    val_images_dir = val_dir / "images"
    val_annotations_file = val_dir / "val_annotations.txt"

    # Check if already reorganized
    # If val/images doesn't exist, we've already done the reorganization
    if not val_images_dir.exists():
        print("✅ Validation set already reorganized")
        return

    print("🔧 Reorganizing validation set...")
    print(f"   Reading annotations from: {val_annotations_file}")

    # ========================================================================
    # Step 3a: Parse the annotations file
    # ========================================================================
    # File format (tab-separated):
    # val_0.JPEG    n01443537    1    2    3    4
    # val_1.JPEG    n01629819    5    6    7    8
    # ...
    # Columns: [image_name, class_id, bbox coordinates (we ignore these)]

    image_to_class = {}

    with open(val_annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            image_name = parts[0]  # e.g., "val_0.JPEG"
            class_id = parts[1]    # e.g., "n01443537"
            image_to_class[image_name] = class_id

    print(f"   Found {len(image_to_class)} image-to-class mappings")

    # ========================================================================
    # Step 3b: Create class subdirectories
    # ========================================================================
    unique_classes = set(image_to_class.values())
    print(f"   Creating {len(unique_classes)} class subdirectories...")

    for class_id in unique_classes:
        class_dir = val_dir / class_id
        class_dir.mkdir(exist_ok=True)

    # ========================================================================
    # Step 3c: Move images to their class folders
    # ========================================================================
    print("   Moving images to class folders...")

    for image_name, class_id in tqdm(image_to_class.items(), desc="Moving images"):
        src = val_images_dir / image_name
        dst = val_dir / class_id / image_name

        if src.exists():
            shutil.move(str(src), str(dst))
        else:
            print(f"⚠️  Warning: {src} not found")

    # ========================================================================
    # Step 3d: Clean up - remove now-empty images directory
    # ========================================================================
    if val_images_dir.exists() and not any(val_images_dir.iterdir()):
        val_images_dir.rmdir()
        print("   Removed empty val/images/ directory")

    # Also remove the annotations file (no longer needed)
    if val_annotations_file.exists():
        val_annotations_file.unlink()
        print("   Removed val_annotations.txt (no longer needed)")

    print("✅ Validation set reorganization complete!")


# ============================================================================
# STEP 4: Verify Dataset Structure
# ============================================================================

def verify_dataset(dataset_dir: Path):
    """
    Sanity check: Ensure dataset has correct structure after preparation.

    Expected structure:
        train/
            n01443537/images/*.JPEG  (500 images per class)
            n01629819/images/*.JPEG
            ... (200 classes total)
        val/
            n01443537/*.JPEG  (50 images per class)
            n01629819/*.JPEG
            ... (200 classes total)
    """
    print("\n🔍 Verifying dataset structure...")

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    # Count classes
    train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
    val_classes = [d for d in val_dir.iterdir() if d.is_dir()]

    print(f"   Train classes: {len(train_classes)}")
    print(f"   Val classes: {len(val_classes)}")

    # Count images in first class (as sample)
    if train_classes:
        sample_train_class = train_classes[0]
        # Train images are in class/images/ subdirectory
        train_images_dir = sample_train_class / "images"
        if train_images_dir.exists():
            train_img_count = len(list(train_images_dir.glob("*.JPEG")))
            print(f"   Sample train class ({sample_train_class.name}): {train_img_count} images")

    if val_classes:
        sample_val_class = val_classes[0]
        # Val images are directly in class directory (after our reorganization)
        val_img_count = len(list(sample_val_class.glob("*.JPEG")))
        print(f"   Sample val class ({sample_val_class.name}): {val_img_count} images")

    print("✅ Dataset verification complete!")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare Tiny ImageNet dataset for training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where dataset will be downloaded and extracted (e.g., ~/scratch/tiny_imagenet_data)"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    print("="*70)
    print("Tiny ImageNet Dataset Preparation")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print("="*70)

    # Pipeline: Download -> Extract -> Reorganize -> Verify
    zip_path = download_dataset(output_dir)
    dataset_dir = extract_dataset(zip_path, output_dir)
    reorganize_validation_set(dataset_dir)
    verify_dataset(dataset_dir)

    print("\n" + "="*70)
    print("✅ ALL DONE! Dataset is ready for training.")
    print("="*70)
    print(f"\nDataset location: {dataset_dir}")
    print("\nNext steps:")
    print("  1. Update your Hydra config to point to this directory")
    print("  2. Run training with: sbatch scripts/train.sh")


if __name__ == "__main__":
    main()
