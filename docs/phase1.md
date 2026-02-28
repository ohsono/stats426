# Phase_1_Data_Harmonization.md

## 1. Overview
The goal of this phase is to unify three distinct datasets—GTSRB, LISA, and BDD100K—into a single, robust pipeline using a 70-10-10-10 split (Train, Validation, In-Domain Test, Out-of-Distribution/Challenge Test). To handle the massive scale of the combined data, staging the raw and processed images in an AWS S3 bucket is recommended before pulling them to the local compute environment.

## 2. Dataset Specifics & PyTorch Implementation


### 2.1 GTSRB (High-Quality Base)
* **Characteristics:** Clean, centered crops, predominantly 30x30 to 50x50 pixels.
* **Pipeline Action:** Resize to a uniform `64x64`. Map the 43 German class IDs to the unified global index (0-57, with gaps handled). 

### 2.2 LISA (US Domain Bridge)
* **Characteristics:** Clean crops of US traffic signs.
* **Pipeline Action:** Map string labels (e.g., `stop`, `yield`) to the unified integer index. Apply standard `torchvision.transforms` such as `RandomAffine` and `ColorJitter` to prevent overfitting on specific lighting conditions.

### 2.3 BDD100K (Real-World Dashcam)
* **Characteristics:** 720p full frames, severe class imbalance, heavy motion blur, varying weather.
* **Pipeline Action:** * Write a pre-processing script to parse the bounding box JSONs and extract padded crops of the signs.
    * Implement a PyTorch `WeightedRandomSampler` in the `DataLoader` to oversample minority classes (e.g., construction signs) and undersample common ones (e.g., speed limits).
    * Apply aggressive augmentations: `GaussianBlur`, heavy contrast shifts, and artificial noise to simulate dashcam sensor artifacts.

## 3. The Unified DataLoader
Construct a custom PyTorch `Dataset` class that can seamlessly ingest from the S3 bucket or local storage, applying the dataset-specific transformations dynamically based on the image's source metadata.