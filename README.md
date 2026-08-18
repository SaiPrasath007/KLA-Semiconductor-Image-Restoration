# AI-Based Restoration of Degraded Images for Semiconductor Inspection

Deep learning restoration pipeline engineered to simultaneously eliminate speckle noise, remove Gaussian blur, and perform 2x super-resolution on nanoscale semiconductor microscopic inspection imagery.

---

## 📂 Repository Structure
team_name/
├── run.py                 # Main automated inference script
├── requirements.txt       # Python dependencies
├── README.md              # Documentation & benchmark execution guide
└── models/
└── final_model_weights.pth  # Offline trained PyTorch model weights


---

## ⚡ Execution Command

To run restoration inference on any input directory:

```bash
python run.py <input-dir> <output-dir>
Key Specifications:
Input: Grayscale degraded .npy arrays or image files.

Output: Clean, super-resolved .npy arrays with shape (H, W) in [0.0, 1.0] range.

Offline execution: Loads weights locally from models/ without internet or API key dependencies.
