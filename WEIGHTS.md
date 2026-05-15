# Trained Model Weights

The trained checkpoints are too large for GitHub. Download from Google Drive:

| File | Stage | Download |
|---|---|---|
| `best_efficientnet_b3_v4.pth` | Stage 1 — EfficientNet-B3 classifier | [Download](https://drive.google.com/file/d/10v4MgVqkviQ3ePSmvNAEuQj3ISsIT4uV/view?usp=drive_link) |
| `best_yolov8_fracture.pt` | Stage 2 — YOLOv8s detector | [Download](https://drive.google.com/file/d/1VPeLTqJTHyxyjURq5MkyGYUWxwZopTgl/view?usp=drive_link) |
| `best_fasterrcnn_fracture.pth` | Stage 2 — Faster R-CNN detector | [Download](https://drive.google.com/file/d/1S11Yv3mklxNkY1h1dgFemPICJhkLy0TP/view?usp=drive_link) |
| `class_info_v4.pkl` | Class names + calibrated thresholds | [Download](https://drive.google.com/file/d/1jEt1UpMXqbAZu2Y8KomE6QvxKN-890bW/view?usp=drive_link) |

## How to use

1. Download all 4 files
2. Place them in a `weights/` folder at the root of this repo (or anywhere)
3. Update the path at the top of the relevant notebook to point to your folder
