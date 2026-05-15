# Automated Bone Fracture Detection & Localization

> A 3-stage deep learning pipeline for fracture detection and localization on plain-film X-rays, trained on the **FracAtlas** dataset (Islam et al., *Scientific Data*, Nature 2023).

**Computer Vision · Final Project · Saint Joseph University · Spring 2026**

**Team:** Maria Bouchi · Theresa Kozah · Alaa Hajjar
**Supervisor:** Mr. Ahmad Awde

---

## 📊 Headline Results

| Stage | Model | Metric | Value |
|---|---|---|---|
| 1. Classification | EfficientNet-B3 | Test Accuracy | **86.63%** |
| 1. Classification | EfficientNet-B3 | Macro F1 | **0.7625** |
| 2. Detection | YOLOv8s | F1 @ IoU 0.5 | 0.591  (32 ms / image) |
| 2. Detection | Faster R-CNN | F1 @ IoU 0.5 | **0.623**  (126 ms / image) |
| 3. Explainability | Grad-CAM | — | Visual saliency map per prediction |

A post-hoc threshold calibration on the validation set raised Macro F1 from 0.69 to 0.76 — a ~7 percentage point gain with zero retraining.

---

## 🧠 Project Description

Bone fractures are among the most common injuries presented at emergency departments — the WHO estimates more than 178 million new cases globally each year. X-ray remains the first-line imaging modality, but hairline cracks, displaced fragments, and subtle cortical disruptions are routinely missed by junior clinicians under time pressure.

This project answers two clinical questions:
1. **Is a fracture present in this X-ray?** (classification)
2. **If so, where exactly on the bone?** (localization)

### Pipeline Overview

```
   ┌─────────────────┐     ┌──────────────────────┐     ┌──────────────┐
   │  Stage 1        │     │  Stage 2             │     │  Stage 3     │
   │  Classification │ ──▶ │  Localization        │ ──▶ │  Explain.    │
   │                 │     │                      │     │              │
   │  EfficientNet   │     │  YOLOv8s /           │     │  Grad-CAM    │
   │  -B3            │     │  Faster R-CNN        │     │              │
   └─────────────────┘     └──────────────────────┘     └──────────────┘
       4-class output         Bounding boxes               Saliency map
       (hand/leg ×            around fracture              over input
        fractured/normal)
```

- **Stage 1 — Classification:** EfficientNet-B3 fine-tuned in two phases on FracAtlas. Predicts one of four classes: hand_fractured, hand_normal, leg_fractured, leg_normal. Calibrated with per-class probability thresholds.
- **Stage 2 — Localization:** Two detectors trained and compared side by side. YOLOv8s (single-stage) optimized for speed and precision; Faster R-CNN with a ResNet-50 FPN backbone (two-stage) optimized for recall.
- **Stage 3 — Explainability:** Grad-CAM saliency maps targeting the last MBConv block of EfficientNet-B3.

Stage 2 only runs when Stage 1 predicts a fractured class — non-fractured X-rays bypass detection compute, matching the clinical priority of catching positives.

Full methodology, dataset construction, ablations, threshold calibration, and discussion are in **[docs/CV_FinalReport.pdf](docs/CV_FinalReport.pdf)**.

---

## 📁 Repository Structure

```
bone-fracture-computer-vision-project/
│
├── README.md                          (this file)
├── requirements.txt                   (Python dependencies)
│
├── notebooks/
│   ├── 01_preprocessing.ipynb         FracAtlas download, cleaning, 4-class splits
│   ├── 02_dataloaders_split.ipynb     PyTorch DataLoaders + class_info_v4.pkl
│   ├── 03_train_efficientnet.ipynb    Stage 1 — EfficientNet-B3 training
│   ├── 04_train_yolov8.ipynb          Stage 2 — YOLOv8s training
│   └── 05_evaluation_pipeline.ipynb   Test-set eval, Grad-CAM, FRCNN comparison
│
├── app/
│   ├── fracture_app.py                Streamlit web application
│   └── fracture_app_launcher.ipynb    One-click Colab launcher (with localtunnel)
│
├── docs/
│   ├── CV_FinalReport.pdf             Full project report
│   └── CV_Defense_Presentation.pdf    Defense slide deck
│
└── figures/                           Key result figures used in the report
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU recommended (we used a single NVIDIA Tesla T4 on Google Colab)
- ~5 GB of free disk space for the dataset and weights

### 1. Clone the repository

```bash
git clone https://github.com/TKozah/bone-fracture-computer-vision-project.git
cd bone-fracture-computer-vision-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

FracAtlas is publicly available:

📥 **Official source:** https://figshare.com/articles/dataset/FracAtlas/22363012

After download, extract the archive into a folder named `FracAtlas/` at a location of your choice. You will reference this path in the notebooks.

### 4. Download the trained weights

The trained model checkpoints are hosted on Google Drive (they exceed GitHub's file-size limit):

EDIT THIS

The folder contains:
- `best_model_v4.pth` — EfficientNet-B3 classifier (~120 MB)
- `yolov8s_best.pt` — YOLOv8s detector (~22 MB)
- `faster_rcnn_best.pth` — Faster R-CNN detector (~165 MB)
- `class_info_v4.pkl` — class names, weights, and calibrated thresholds

Place these files in a folder of your choice (e.g., `weights/`) and update the paths at the top of the relevant notebooks.

---

## 🚀 How to Run Inference

There are two ways to run inference on a new X-ray.

### Option A — Web Application (recommended)

The fastest way to test the pipeline on real X-rays. Launch the Streamlit app:

```bash
cd app/
streamlit run fracture_app.py
```

Your browser will open automatically (or visit http://localhost:8501). Upload any X-ray image — the app will return:

- **Stage 1:** body region + fracture status with confidence scores
- **Stage 2:** bounding boxes from YOLOv8 (with toggle to Faster R-CNN)
- **Stage 3:** Grad-CAM attention heatmap overlay

Make sure the weight paths at the top of `fracture_app.py` point to your downloaded checkpoints.

#### Running the app on Google Colab

If you don't have a local GPU, the launcher notebook spins up the app on Colab via localtunnel:

```
app/fracture_app_launcher.ipynb
```

Open it in Colab, run all cells, and follow the printed tunnel URL.

### Option B — Programmatic inference (notebook)

For batch evaluation or custom inputs, use `notebooks/05_evaluation_pipeline.ipynb`. Update the paths at the top of the notebook, then run all cells. The notebook produces:

- Test-set confusion matrix
- Per-class F1 chart
- Grad-CAM visualizations
- YOLOv8 vs Faster R-CNN side-by-side comparison

---

## 🔬 How to Reproduce Training

Run the notebooks in order. Each notebook has a configuration block at the top where you set paths and (optionally) hyperparameters.

| # | Notebook | Purpose | Runtime (T4 GPU) |
|---|---|---|---|
| 01 | `01_preprocessing.ipynb` | Download FracAtlas, exclude hip/shoulder (insufficient samples), build 4-class splits | ~10 min |
| 02 | `02_dataloaders_split.ipynb` | Build PyTorch DataLoaders, save `class_info_v4.pkl` (class weights + thresholds) | ~2 min |
| 03 | `03_train_efficientnet.ipynb` | Two-phase fine-tune of EfficientNet-B3 (Phase 1: frozen backbone, Phase 2: unfrozen at LR 1e-5) | ~45 min |
| 04 | `04_train_yolov8.ipynb` | Train YOLOv8s at 640×640 with grayscale-safe augmentation | ~60 min |
| 05 | `05_evaluation_pipeline.ipynb` | Run threshold tuning, test-set evaluation, Grad-CAM, Faster R-CNN comparison | ~10 min |

### Key training choices

- **No ColorJitter** — X-ray pixel intensity encodes bone density; brightness jitter would corrupt the diagnostic signal
- **Class-weighted cross-entropy** with label smoothing (0.1)
- **WeightedRandomSampler** to address 9× imbalance (leg_normal vs leg_fractured)
- **Two-phase fine-tuning** — backbone frozen for the first 15 epochs, then unfrozen at LR = 1e-5
- **Post-hoc threshold calibration** — per-class thresholds `[0.50, 0.32, 0.50, 0.10]` swept on validation, applied at test time

---

## 📈 Selected Results

### Stage 1 — Test set per-class F1 (after threshold tuning)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| hand_fractured | 0.62 | 0.77 | 0.69 | 39 |
| hand_normal | 0.90 | 0.79 | 0.84 | 91 |
| leg_fractured | 0.54 | 0.59 | 0.57 | 22 |
| leg_normal | 0.95 | 0.95 | 0.95 | 192 |
| **Macro avg** | **0.75** | **0.78** | **0.76** | **344** |

### Stage 2 — Detector comparison (hand+leg filtered test set, IoU 0.5)

| Metric | YOLOv8s | Faster R-CNN |
|---|---|---|
| Precision | **0.7857** | 0.7027 |
| Recall | 0.4731 | **0.5591** |
| F1 | 0.5906 | **0.6228** |
| Inference time (T4) | **32 ms** | 126 ms |
| Parameters | 11.1 M | ~41.8 M |

**Clinical verdict:** Faster R-CNN is preferable for fracture detection because missing a fracture is more costly than a false alarm. In a real deployment, the optimal setup uses both — YOLO for real-time triage at intake, Faster R-CNN for the radiologist's second-read worklist.

---

## 📚 Citation

If you use this work, please cite the FracAtlas dataset:

```bibtex
@article{abedeen2023fracatlas,
  title   = {FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs},
  author  = {Abedeen, Iftekharul and Rahman, Md. Ashiqur and Prottyasha, Fatema Zohra and others},
  journal = {Scientific Data},
  volume  = {10},
  number  = {521},
  year    = {2023},
  publisher = {Nature Publishing Group},
  doi     = {10.1038/s41597-023-02432-4}
}
```

Full reference list is in **[docs/CV_FinalReport.pdf](docs/CV_FinalReport.pdf)**.

---

## 🛠 Built With

- [PyTorch](https://pytorch.org/) — deep learning framework
- [torchvision](https://pytorch.org/vision/) — Faster R-CNN implementation
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — YOLOv8 training and inference
- [grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Grad-CAM implementation
- [Streamlit](https://streamlit.io/) — web application framework
- [Google Colab](https://colab.research.google.com/) — training environment (Tesla T4)

---

## 📜 License

This project is released for academic purposes only. The FracAtlas dataset is licensed by its authors under its own terms — please consult the original publication.

---

## 🙏 Acknowledgments

We thank our supervisor **Mr. Ahmad Awde** for his guidance throughout the semester, and the FracAtlas team (Abedeen et al., 2023) for releasing a high-quality, peer-reviewed musculoskeletal radiograph dataset that made this project possible.
