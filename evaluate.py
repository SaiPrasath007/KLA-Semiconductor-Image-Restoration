import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# --- 1. Architecture Definition ---
class CALayer(nn.Module):
    def __init__(self, channels=64, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.conv_du(self.avg_pool(x))

class ResBlock(nn.Module):
    def __init__(self, channels=64, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ca = CALayer(channels)
    def forward(self, x):
        return x + self.ca(self.conv2(self.relu(self.conv1(x)))) * self.res_scale

class SuperResolutionNet(nn.Module):
    def __init__(self, num_res_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(1, 64, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(64) for _ in range(num_res_blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.ReLU(inplace=True)
        )
        self.tail = nn.Conv2d(64, 1, 3, padding=1)
    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        feat = self.upsample(self.body(self.head(x)))
        return torch.sigmoid(base + self.tail(feat))

# --- 2. Data Loader Helper ---
def load_input_tensor(file_path):
    if file_path.endswith('.npy'):
        arr = np.load(file_path).astype(np.float32)
        tensor = torch.from_numpy(arr)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0) if tensor.shape[0] == 1 else tensor.permute(2, 0, 1).unsqueeze(0)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor
    else:
        img = Image.open(file_path).convert('L')
        return TF.to_tensor(img).unsqueeze(0)

# --- 3. Fast Self-Ensemble / TTA Predictor ---
def predict_with_tta(model, tensor_input):
    preds = []
    for rot in [0, 1, 2, 3]:
        for flip in [False, True]:
            x = torch.rot90(tensor_input, k=rot, dims=[-2, -1])
            if flip:
                x = torch.flip(x, dims=[-1])
            pred = model(x)
            if flip:
                pred = torch.flip(pred, dims=[-1])
            pred = torch.rot90(pred, k=-rot, dims=[-2, -1])
            preds.append(pred)
    return torch.stack(preds).mean(dim=0)

# --- 4. Main CLI Runner ---
def main():
    parser = argparse.ArgumentParser(description="KLA Image Restoration Benchmark Evaluation Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input degraded images (.npy / .png / .jpg)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save restored output images")
    parser.add_argument("--weights", type=str, default="final_model_weights.pth", help="Path to model weights file")
    parser.add_argument("--disable_tta", action="store_true", help="Disable TTA for ultra-fast benchmark inference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running evaluation on: {device}")

    # Load Model
    model = SuperResolutionNet(num_res_blocks=8).to(device)
    if os.path.exists(args.weights):
        state_dict = torch.load(args.weights, map_location=device)
        # Handle cases where state_dict is inside a checkpoint dict
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print(f"[+] Loaded weights from {args.weights}")
    else:
        raise FileNotFoundError(f"Weights file not found at: {args.weights}")

    model.eval()

    # Find Test Files
    valid_exts = ('.npy', '.png', '.jpg', '.jpeg')
    test_files = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(args.input_dir)
        for f in files if f.endswith(valid_exts) and not f.startswith('.')
    ])

    print(f"[+] Found {len(test_files)} samples to restore in '{args.input_dir}'")

    with torch.no_grad():
        for idx, file_path in enumerate(test_files):
            filename = os.path.basename(file_path)
            base_name, ext = os.path.splitext(filename)

            tensor = load_input_tensor(file_path).to(device)
            
            if args.disable_tta:
                pred = model(tensor)
            else:
                pred = predict_with_tta(model, tensor)

            pred = torch.clamp(pred, 0.0, 1.0)
            output_np = pred.squeeze().cpu().numpy().astype(np.float32)

            # Save in matching format
            if ext == '.npy':
                np.save(os.path.join(args.output_dir, f"{base_name}.npy"), output_np)
            else:
                out_img = Image.fromarray((output_np * 255.0).astype(np.uint8))
                out_img.save(os.path.join(args.output_dir, f"{base_name}.png"))

            if (idx + 1) % 50 == 0 or (idx + 1) == len(test_files):
                print(f"  Processed [{idx + 1}/{len(test_files)}] -> {base_name}")

    print(f"[+] Inference complete! Restored files saved to: '{args.output_dir}'")

if __name__ == "__main__":
    main()
