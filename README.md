# AI-Based Restoration of Degraded Images for Semiconductor Inspection

Deep learning restoration pipeline engineered to simultaneously eliminate speckle noise, remove Gaussian blur, and perform 2× super-resolution on nanoscale semiconductor microscopic inspection imagery.

---

## 📌 Project Overview

In semiconductor manufacturing, microscopic inspection images are critical for identifying sub-nanometer defects in gates, vias, and interconnects. In production environments, sensor noise and hardware throughput constraints introduce:
1. **Speckle Noise:** High-variance pixel-level noise pushing intensity values beyond true dynamic ranges.
2. **Gaussian Noise:** Soft edge blurring and structural contrast loss.
3. **Spatial Resolution Loss (2× Downsampling):** High-frequency pattern degradation (256×256 → 128×128 or 512×512 → 256×256).

This repository provides an end-to-end single-pass restoration network utilizing **Residual Channel Attention Networks (RCAN)** combined with a **Bicubic Global Residual Base** and a **Hybrid L1 + SSIM Loss** formulation to reconstruct high-fidelity inspection imagery in real time.

---

## 🏗️ Model Architecture & Methodology

Degraded Input [1, H, W] ───────────┬──────────────── (Bicubic Upsample 2x) ───────────────┐│                                                      │▼                                                      ▼[Conv 3x3 Head (64 Filters)]                                        ││                                                      │▼                                                      │[8x Residual Channel Attention Blocks (RCAB)]                             │└── Conv -> ReLU -> Conv -> CALayer -> Scale (0.1x)                 ││                                                      │▼                                                      │[PixelShuffle Upsampler (2x)]                                       ││                                                      │▼                                                      │[Conv 3x3 Tail (1 Channel)]                                         ││                                                      │└─────────────────────► (+) ◄──────────────────────────┘│▼Sigmoid Clamping [0, 1]│▼Restored Output [1, 2H, 2W]
### Key Technical Highlights:
* **Channel Attention Mechanism (`CALayer`):** Adaptively recalibrates feature maps to suppress grain/speckle artifacts while amplifying structural edges.
* **Residual Scaling (0.1×):** Prevents signal saturation and vanishing gradients during deep residual feature extraction.
* **Sub-Pixel Convolution (`nn.PixelShuffle`):** Reconstructs 2× spatial resolution with minimal compute overhead compared to transposed convolutions.
* **Hybrid Objective Function:** Combines pixel-level accuracy (L1) with perceptual structural consistency (SSIM):
  $$\mathcal{L}_{\text{total}} = 0.8 \cdot \mathcal{L}_{1} + 0.2 \cdot (1.0 - \text{SSIM})$$
* **8-Geometric Self-Ensemble (TTA):** Test-Time Augmentation over 8 spatial rotations and flips for out-of-distribution robustness.

---

## 📂 Repository Structure

├── evaluate.py               # Standalone evaluation & inference CLI script├── train.py                  # End-to-end training pipeline├── final_model_weights.pth   # Pre-trained model weights├── requirements.txt          # Python dependencies├── restored_test_outputs/    # Restored test set outputs (.npy / .png)└── README.md                 # Project documentation and run guide
---

## ⚙️ Installation & Setup

Ensure Python 3.9+ and CUDA-compatible PyTorch are installed:

```bash
git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git
cd <YOUR_REPOSITORY_NAME>
pip install -r requirements.txt
🚀 Running Evaluation & BenchmarkingThe evaluation script evaluate.py accepts input directories containing either .npy arrays or .png/.jpg images, runs model inference, and writes the restored outputs directly to the specified target folder.1. Standard Inference (with 8-Fold TTA)Bashpython evaluate.py --input_dir ./test_images --output_dir ./restored_outputs --weights final_model_weights.pth
2. High-Speed Benchmark Mode (Single Forward-Pass for H100 GPU)Bashpython evaluate.py --input_dir ./test_images --output_dir ./restored_outputs --weights final_model_weights.pth --disable_tta
Command-Line Arguments:ArgumentTypeDefaultDescription--input_dirstrRequiredPath to folder containing degraded input images (.npy, .png, .jpg).--output_dirstrRequiredPath to folder where restored outputs will be saved.--weightsstrfinal_model_weights.pthPath to pre-trained PyTorch weights checkpoint.--disable_ttaflagFalseDisables 8-fold test-time augmentation for maximum inference speed.🏋️ Training the Model from ScratchTo reproduce the training process:Bashpython train.py --data_dir /path/to/train_dataset --epochs 100 --batch_size 16 --lr 2e-4
📊 Performance & SpecificationsModel Parameters: ~0.82 Million (< 5 MB weights file)Input Resolution: Single-channel grayscale ($128\times128$ or $256\times256$)Output Resolution: Single-channel grayscale ($256\times256$ or $512\times512$)Inference Latency: < 15 ms per sample on NVIDIA T4 / < 2 ms on NVIDIA H100📚 ReferencesRCAN: Zhang, Y., et al. (2018). Image Super-Resolution Using Very Deep Residual Channel Attention Networks. ECCV.EDSR: Lim, B., et al. (2017). Enhanced Deep Residual Networks for Single Image Super-Resolution. CVPRW.ESPCN: Shi, W., et al. (2016). Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. CVPR.Loss Design: Zhao, H., et al. (2017). Loss Functions for Image Restoration with Neural Networks. IEEE TCI.
