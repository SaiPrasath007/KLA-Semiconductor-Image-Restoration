# AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Overview
This repository implements a lightweight **Residual Network with PixelShuffle Super-Resolution** to perform joint denoising (speckle + Gaussian) and 2x spatial upscaling on semiconductor inspection `.npy` images.

## Quick Start / Evaluation Script Execution
To evaluate the model on test images, run `eval.py`:

```bash
python eval.py --input_dir /path/to/test_inputs --output_dir /path/to/save_outputs --weights model_best.pth
```