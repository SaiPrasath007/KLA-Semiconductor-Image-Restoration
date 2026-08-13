import argparse
import os
import glob
import numpy as np
import torch
from model import RestorationSRNet

def run_evaluation(input_dir, output_dir, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    model = RestorationSRNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    lr_files = sorted(glob.glob(os.path.join(input_dir, '*.npy')))
    
    with torch.no_grad():
        for file_path in lr_files:
            filename = os.path.basename(file_path)
            img = np.load(file_path).astype(np.float32)
            
            p_max = max(np.percentile(img, 99.9), 1.0)
            p_min = min(img.min(), 0.0)
            norm_img = (img - p_min) / (p_max - p_min + 1e-8)
            
            tensor_in = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(device)
            output_tensor = model(tensor_in)
            output_arr = output_tensor.squeeze().cpu().numpy()
            
            restored_img = output_arr * (p_max - p_min + 1e-8) + p_min
            np.save(os.path.join(output_dir, filename), restored_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KLA Image Restoration Benchmarking Script')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input test directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--weights', type=str, default='model_best.pth', help='Path to model weights')
    
    args = parser.parse_args()
    run_evaluation(args.input_dir, args.output_dir, args.weights)