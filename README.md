# End-to-End DCGAN Pipeline for Robust Image Generation

 
*(PyTorch · Fault-Tolerant Data Pipelines · Reproducible ML Systems)*

---

## Overview
This project implements a **robust, end-to-end GAN-based image generation pipeline** built for a Kaggle GAN Challenge.  
The goal is to generate realistic **32×32 RGB images** from a highly noisy dataset of **base64-encoded image shards** and evaluate the generated distribution using **Fréchet Inception Distance (FID)**.

Unlike clean benchmark datasets, the data intentionally contains **corrupt records, missing keys, mixed image formats, EXIF rotations, inversions, and optional alpha masks**, making robustness and fault tolerance core to the solution.

---

## Key Features
- Robust decoding of corrupted **base64-encoded image shards**
- Handles **16-bit grayscale, RGB, RGBA, palette images**, and alpha masks
- Applies **EXIF rotation** and inversion flags with safe fallbacks
- Standardizes all images to **32×32 RGB**, normalized to **[-1, 1]**
- Trains a **DCGAN** using PyTorch
- Deterministic generation with fixed seeds
- **Inception-V3 (pool3) feature extraction** for FID-compatible evaluation
- Produces a competition-ready `submission.csv`

---

## Dataset Description
Each image is stored as a JSONL record with metadata and base64-encoded bytes.

### Common Fields
| Key          | Description |
|--------------|-------------|
| `id`         | Unique image identifier |
| `mode`       | Image mode (e.g., `I;16`, `RGBA`) |
| `size`       | Image dimensions `[H, W]` |
| `exif_rot`   | Rotation (0, 90, 180, 270) |
| `invert`     | Pixel inversion flag |
| `img_b64`    | Base64-encoded image |
| `alpha_b64`  | Optional alpha mask |

The pipeline is designed to **never crash** on malformed or incomplete records.

---

## Project Pipeline

### 1. Robust Decoding & Preprocessing
- Extracts base64 image data from inconsistent keys
- Safely decodes bytes with error handling
- Applies:
  - EXIF orientation correction
  - Inversion flags
  - Alpha-mask merging (when present)
- Converts all images to **RGB**
- Resizes to **32×32**
- Normalizes pixel values to **[-1, 1]**

---

### 2. Dataset & DataLoader
A custom PyTorch `Dataset`:
- Loads multiple JSONL shards
- Gracefully handles corrupt samples
- Enables uninterrupted training on real-world noisy data

---

### 3. Model Architecture (DCGAN)

**Generator**
- Input: 128-D latent noise vector
- Transposed convolutions
- Output: 32×32 RGB image
- Activation: `Tanh`

**Discriminator**
- Strided convolutions
- LeakyReLU activations
- Binary real/fake prediction

Label smoothing is used to improve training stability.

---

### 4. Training
- Optimizer: Adam (`lr=2e-4`, `betas=(0.5, 0.999)`)
- Loss: Binary Cross Entropy with logits
- GPU-enabled training
- Periodic sample visualization
- Model checkpointing after each epoch

---

### 5. Image Generation
- Generates **1000 images** using fixed random seeds
- Saves images as PNG files
- Ensures deterministic evaluation

---

### 6. FID-Compatible Feature Extraction
- Uses pretrained **Inception-V3**
- Extracts **2048-D pool3 features**
- Handles torchvision version incompatibilities
- Applies ImageNet normalization

---

### 7. Submission Generation
Creates a `submission.csv` with:
- 1000 rows
- 1 `id` column + 2048 feature columns (`f0`–`f2047`)

Example format:
```csv
id,f0,f1,...,f2047
dig-000000,0.012,-0.034,...,0.015
...
````

---

## Dataset Access (Competition Ended)

The dataset used in this project was released as part of the **Kaggle “test-gan-competition”**, which has now concluded.

Due to Kaggle’s competition licensing rules, the dataset is **not publicly redistributable** and therefore cannot be included in this repository.

To reproduce this work:

1. Join the competition on Kaggle (if access is still permitted)
2. Download the data using the Kaggle CLI:

```bash
kaggle competitions download -c test-gan-competition
unzip test-gan-competition.zip
```

If the competition is no longer accessible, this repository remains valuable as a **reference implementation** for robust GAN training and FID-based evaluation on noisy image datasets.

---

## Results

* Successfully trained a DCGAN on noisy, real-world data
* Generated visually coherent 32×32 images
* Produced a valid FID-compatible submission
* Demonstrated robustness beyond assumption-heavy pipelines

---

## Tech Stack

* **Language:** Python
* **Frameworks:** PyTorch, Torchvision
* **Models:** DCGAN, Inception-V3
* **Libraries:** PIL, NumPy, Pandas
* **Evaluation:** Fréchet Inception Distance (FID)
* **Platform:** Kaggle

---

## What This Project Demonstrates

* Fault-tolerant data engineering
* Deep generative modeling
* Image preprocessing and normalization
* Reproducible ML workflows
* Competition-grade evaluation pipelines

---


## License

This project is intended for educational and research purposes.

```
```
