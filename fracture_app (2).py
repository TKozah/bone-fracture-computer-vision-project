"""
═══════════════════════════════════════════════════════════════════════════════
  FractureAI — Streamlit Web Application
  Saint Joseph University · Computer Vision Final Project · Spring 2026
───────────────────────────────────────────────────────────────────────────────
  A 3-stage bone fracture detection AI pipeline:
    Stage 1 — EfficientNet-B3        4-class classification
    Stage 2 — YOLOv8s / Faster R-CNN bounding box localization
    Stage 3 — Grad-CAM               attention heatmap

  Run in Colab:
      !pip install -q streamlit pytorch-grad-cam ultralytics
      !streamlit run fracture_app.py &>/content/log.txt &
      !npx localtunnel --port 8501
═══════════════════════════════════════════════════════════════════════════════
"""

import io
import time
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models import efficientnet_b3
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — paths, thresholds, metrics
# ═══════════════════════════════════════════════════════════════════════════════
BASE        = Path('/content/drive/MyDrive/CV_project_v4')
CLF_CKPT    = BASE / 'best_efficientnet_b3_v4.pth'
YOLO_CKPT   = BASE / 'best_yolov8_fracture.pt'
FRCNN_CKPT  = BASE / 'best_fasterrcnn_fracture.pth'
INFO_PKL    = BASE / 'class_info_v4.pkl'

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP     = torch.cuda.is_available()

YOLO_CONF   = 0.25
YOLO_IOU    = 0.45
YOLO_IMGSZ  = 640
FRCNN_CONF  = 0.5

# Latest test-set metrics (from Notebook E, on GPU)
METRICS = {
    'clf_acc':      0.8663,          # 86.63%
    'clf_f1':       0.7625,
    'clf_params':   '10.7M',
    'per_class_f1': {
        'hand_normal':    0.84,
        'leg_normal':     0.95,
        'hand_fractured': 0.69,
        'leg_fractured':  0.57,
    },
    'yolo_map50':   0.4058,
    'yolo_prec':    0.7857,
    'yolo_rec':     0.4731,
    'yolo_f1':      0.5906,
    'yolo_ms':      32.3,
    'yolo_params':  '11.1M',
    'frcnn_prec':   0.7027,
    'frcnn_rec':    0.5591,
    'frcnn_f1':     0.6228,
    'frcnn_ms':     126.1,
    'frcnn_params': '~41.8M',
}

FUN_FACTS = [
    "The human hand contains 27 bones — more than any other part of the body.",
    "X-rays were discovered by Wilhelm Röntgen in 1895. He won the first Nobel Prize in Physics.",
    "A broken bone begins healing within 48 hours of the fracture occurring.",
    "FracAtlas was annotated by 2 expert radiologists and an orthopedic surgeon.",
    "EfficientNet-B3 was originally trained on 1.2 million ImageNet images before fine-tuning.",
    "The tibia is the most commonly fractured long bone in the human body.",
    "Grad-CAM was introduced in 2017 to make neural networks visually explainable.",
    "Over 178 million fractures occur globally each year according to WHO.",
    "AI-assisted diagnosis can reduce radiologist reading time by up to 30%.",
    "YOLOv8 can process images in under 10 milliseconds on modern GPU hardware.",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG + CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title='FractureAI — CV Final Project',
    page_icon='🦴',
    layout='wide',
    initial_sidebar_state='expanded'
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');

/* hide streamlit chrome */
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; display: none !important; }
.viewerBadge_container__1QSob { display: none !important; }

/* global background */
.stApp {
    background: #0a0e1a;
    color: #f1f5f9;
    font-family: 'DM Sans', sans-serif;
}
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1400px; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: #0e1424 !important;
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] > div { background: #0e1424 !important; }

/* hero header */
.hero-wrap { margin-bottom: 28px; }
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f1f5f9 0%, #3b82f6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -2px;
    line-height: 1.1;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 6px;
    font-family: 'Space Mono', monospace;
}

/* card */
.card {
    background: #1a2235;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 14px;
    font-weight: 700;
}

/* diagnosis banners */
.banner {
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    margin: 18px 0 22px 0;
    position: relative;
    overflow: hidden;
}
.banner-fracture {
    background: linear-gradient(135deg, #1a0e10 0%, #2a1015 100%);
    border: 2px solid #ef4444;
    box-shadow: 0 0 40px rgba(239, 68, 68, 0.3), inset 0 0 30px rgba(239, 68, 68, 0.08);
}
.banner-fracture .banner-icon { font-size: 2.2rem; }
.banner-fracture .banner-title { color: #ef4444; }
.banner-normal {
    background: linear-gradient(135deg, #0a1a14 0%, #0e2618 100%);
    border: 2px solid #10b981;
    box-shadow: 0 0 40px rgba(16, 185, 129, 0.25), inset 0 0 30px rgba(16, 185, 129, 0.08);
}
.banner-normal .banner-icon { font-size: 2.2rem; }
.banner-normal .banner-title { color: #10b981; }
.banner-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    margin: 6px 0 0 0;
    letter-spacing: 2px;
}
.banner-meta {
    color: #cbd5e1;
    margin-top: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    letter-spacing: 1px;
}

/* metric tiles */
.metric-tile {
    background: #1a2235;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}
.metric-tile::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #3b82f6, #06b6d4);
}
.metric-label {
    color: #94a3b8;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}
.metric-value {
    color: #f1f5f9;
    font-size: 1.45rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}

/* confidence bars */
.conf-section { margin-top: 8px; }
.conf-row { margin: 10px 0; }
.conf-label {
    color: #cbd5e1;
    font-size: 0.85rem;
    font-family: 'Space Mono', monospace;
    margin-bottom: 5px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}
.conf-label .pct { color: #f1f5f9; font-weight: 700; }
.conf-bar {
    background: #0e1424;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    height: 14px;
    overflow: hidden;
    position: relative;
}
.conf-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 6px currentColor;
}
.conf-fill.frac { background: linear-gradient(90deg, #ef4444, #f87171); color: #ef4444; }
.conf-fill.normal { background: linear-gradient(90deg, #3b82f6, #06b6d4); color: #3b82f6; }
.conf-fill.predicted { box-shadow: 0 0 18px currentColor; }

/* pipeline step card */
.pipeline-step {
    background: #1a2235;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 14px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.pipeline-step:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15); }
.step-number {
    color: #3b82f6;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 3px;
}
.step-title {
    color: #f1f5f9;
    font-family: 'Space Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    margin: 6px 0 10px 0;
}
.step-desc { color: #94a3b8; font-size: 0.92rem; line-height: 1.55; }
.step-meta {
    margin-top: 12px;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    padding-top: 10px;
    border-top: 1px solid rgba(30, 58, 95, 0.5);
}

/* disclaimer box */
.disclaimer {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 18px;
    color: #fbbf24;
    margin-top: 24px;
    font-size: 0.9rem;
    line-height: 1.55;
}
.disclaimer strong { color: #f59e0b; }

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    border-bottom: 1px solid #1e3a5f;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8;
    font-family: 'Space Mono', monospace;
    font-weight: 500;
    padding: 12px 22px;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #f1f5f9 !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* buttons */
.stButton button, .stDownloadButton button {
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    color: white !important;
    border: none;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border-radius: 8px;
    padding: 10px 22px;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton button:hover, .stDownloadButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
}

/* radio buttons */
.stRadio > div {
    background: #0e1424;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 6px 10px;
}
.stRadio label { color: #cbd5e1 !important; font-family: 'Space Mono', monospace; }

/* file uploader */
.stFileUploader > section {
    background: #0e1424 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
    color: #cbd5e1 !important;
}
.stFileUploader > section:hover { border-color: #3b82f6 !important; }
.stFileUploader label { color: #cbd5e1 !important; }
.stFileUploader small, .stFileUploader span { color: #94a3b8 !important; }
.stFileUploader p, .stFileUploader div { color: #cbd5e1 !important; }

/* the "Browse files" button inside the uploader — was white before */
.stFileUploader button,
.stFileUploader section button,
section[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, #3b82f6, #06b6d4) !important;
    color: white !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 8px 22px !important;
    transition: transform 0.15s, box-shadow 0.15s;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
}
.stFileUploader button:hover,
.stFileUploader section button:hover,
section[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.45) !important;
    color: white !important;
}
/* the file pill that appears after upload */
[data-testid="stFileUploaderFile"] {
    background: #1a2235 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #cbd5e1 !important;
}
[data-testid="stFileUploaderFile"] * { color: #cbd5e1 !important; }

/* scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: #3b82f6; }

/* fun fact box */
.fun-fact {
    background: linear-gradient(135deg, #0e1424 0%, #1a2235 100%);
    border: 1px solid #1e3a5f;
    border-left: 3px solid #06b6d4;
    border-radius: 10px;
    padding: 14px 16px;
    color: #cbd5e1;
    font-style: italic;
    font-size: 0.85rem;
    line-height: 1.5;
    margin-top: 14px;
}
.fun-fact-icon { color: #06b6d4; font-weight: 700; font-style: normal; }

/* comparison table */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-family: 'Space Mono', monospace;
}
.compare-table th, .compare-table td {
    padding: 11px 14px;
    text-align: left;
    border-bottom: 1px solid #1e3a5f;
    font-size: 0.88rem;
}
.compare-table th {
    background: #0e1424;
    color: #94a3b8;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
}
.compare-table td { color: #cbd5e1; }
.compare-table td:first-child { color: #94a3b8; }
.compare-table .winner { color: #10b981; font-weight: 700; }
.compare-table tr:hover td { background: rgba(59, 130, 246, 0.04); }

/* sidebar typography */
.sidebar-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f1f5f9, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2px;
    line-height: 1.1;
    letter-spacing: -1px;
}
.sidebar-subtitle {
    color: #64748b;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
    margin-bottom: 22px;
}
.sidebar-section-title {
    color: #94a3b8;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 18px 0 10px 0;
    font-weight: 700;
}
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
    margin: 16px 0;
}
.sidebar-model-card {
    background: #1a2235;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.sidebar-model-name {
    color: #06b6d4;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.sidebar-kv {
    display: flex;
    justify-content: space-between;
    color: #94a3b8;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    padding: 2px 0;
}
.sidebar-kv .v { color: #f1f5f9; font-weight: 700; }

/* fact card grid */
.fact-card {
    background: #1a2235;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #06b6d4;
    border-radius: 8px;
    padding: 14px 16px;
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.55;
    height: 100%;
    min-height: 90px;
}

/* reference list */
.ref-item {
    color: #cbd5e1;
    font-size: 0.88rem;
    padding: 9px 0;
    border-bottom: 1px solid rgba(30, 58, 95, 0.5);
    line-height: 1.5;
}
.ref-item:last-child { border-bottom: none; }
.ref-item .ref-title { color: #f1f5f9; font-weight: 700; }
.ref-item .ref-venue { color: #06b6d4; font-family: 'Space Mono', monospace; font-size: 0.8rem; }

/* code block */
.code-block {
    background: #0e1424;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px;
    color: #cbd5e1;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.65;
    white-space: pre-wrap;
    overflow-x: auto;
}
.code-block .kw { color: #c084fc; }
.code-block .str { color: #34d399; }
.code-block .com { color: #64748b; font-style: italic; }

/* section header */
.section-header {
    color: #94a3b8;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin: 26px 0 14px 0;
    font-weight: 700;
}

/* design decision card */
.decision-card {
    background: #1a2235;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.decision-title {
    color: #06b6d4;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.decision-desc { color: #94a3b8; font-size: 0.85rem; line-height: 1.5; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING (cached — runs once per session)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_info():
    if not INFO_PKL.exists():
        return None
    with open(INFO_PKL, 'rb') as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_classifier():
    if not CLF_CKPT.exists():
        return None, None
    model = efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(1536, 4)
    )
    ckpt = torch.load(CLF_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model, ckpt


@st.cache_resource(show_spinner=False)
def load_yolo():
    if not YOLO_CKPT.exists():
        return None
    return YOLO(str(YOLO_CKPT))


@st.cache_resource(show_spinner=False)
def load_frcnn():
    if not FRCNN_CKPT.exists():
        return None
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(FRCNN_CKPT, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def get_gradcam():
    clf, _ = load_classifier()
    if clf is None:
        return None
    return GradCAM(model=clf, target_layers=[clf.features[-1]])


def check_assets():
    """Verify all required model files exist. Return a status dict."""
    return {
        'info':  INFO_PKL.exists(),
        'clf':   CLF_CKPT.exists(),
        'yolo':  YOLO_CKPT.exists(),
        'frcnn': FRCNN_CKPT.exists(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def run_inference(pil_img):
    """Run the complete 3-stage pipeline. Returns dict with all results."""
    info = load_info()
    clf_model, _ = load_classifier()
    yolo_model = load_yolo()
    frcnn_model = load_frcnn()
    cam = get_gradcam()

    eval_transform  = info['eval_transform']
    best_thresholds = info['optimal_thresholds']
    IDX_TO_CLASS    = info['IDX_TO_CLASS']
    ORDERED_CLASSES = info['ORDERED_CLASSES']
    IMAGENET_MEAN   = info['IMAGENET_MEAN']
    IMAGENET_STD    = info['IMAGENET_STD']

    # ─── Stage 1: Classification ────────────────────────────────────────────
    t0 = time.time()
    tensor = eval_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            logits = clf_model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy().astype(float)

    pred_idx = int(np.argmax(probs))
    for i, t in enumerate(best_thresholds):
        if probs[i] > t:
            pred_idx = i
            break

    pred_label   = IDX_TO_CLASS[pred_idx]
    is_fractured = 'fractured' in pred_label
    body_part    = pred_label.replace('_fractured', '').replace('_normal', '')
    confidence   = float(probs[pred_idx])
    t_clf = (time.time() - t0) * 1000

    # ─── Stage 3 (run early for caching): Grad-CAM ─────────────────────────
    t0 = time.time()
    tensor_cam = eval_transform(pil_img).unsqueeze(0).to(DEVICE)
    grayscale = cam(input_tensor=tensor_cam)[0]
    rgb = tensor_cam.squeeze(0).cpu().clone()
    for i in range(3):
        rgb[i] = rgb[i] * IMAGENET_STD[i] + IMAGENET_MEAN[i]
    rgb = rgb.clamp(0, 1).permute(1, 2, 0).numpy()
    cam_array = show_cam_on_image(rgb, grayscale, use_rgb=True)
    cam_image = Image.fromarray(cam_array)  # convert to PIL so st.image expands it
    t_cam = (time.time() - t0) * 1000

    # ─── Stage 2a: YOLOv8 ───────────────────────────────────────────────────
    t0 = time.time()
    yolo_boxes = []
    if is_fractured:
        preds = yolo_model.predict(
            np.array(pil_img), conf=YOLO_CONF, iou=YOLO_IOU,
            imgsz=YOLO_IMGSZ, verbose=False
        )
        for det in preds[0].boxes:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            yolo_boxes.append((float(x1), float(y1), float(x2), float(y2),
                               float(det.conf[0])))
    t_yolo = (time.time() - t0) * 1000

    # ─── Stage 2b: Faster R-CNN ─────────────────────────────────────────────
    t0 = time.time()
    frcnn_boxes = []
    if is_fractured:
        frcnn_tensor = T.ToTensor()(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds = frcnn_model(frcnn_tensor)
        for box, score, label in zip(
            preds[0]['boxes'].cpu().numpy(),
            preds[0]['scores'].cpu().numpy(),
            preds[0]['labels'].cpu().numpy()
        ):
            if label == 1 and score >= FRCNN_CONF:
                frcnn_boxes.append((float(box[0]), float(box[1]),
                                    float(box[2]), float(box[3]), float(score)))
    t_frcnn = (time.time() - t0) * 1000

    return {
        'pred_label':   pred_label,
        'pred_idx':     pred_idx,
        'is_fractured': is_fractured,
        'body_part':    body_part,
        'confidence':   confidence,
        'probs':        probs.tolist(),
        'classes':      ORDERED_CLASSES,
        'cam_image':    cam_image,
        'yolo_boxes':   yolo_boxes,
        'frcnn_boxes':  frcnn_boxes,
        't_clf':        t_clf,
        't_cam':        t_cam,
        't_yolo':       t_yolo,
        't_frcnn':      t_frcnn,
    }


def get_or_cache_result(file_bytes):
    """Hash the uploaded bytes and only re-run inference when image changes."""
    file_hash = hash(file_bytes)
    if (st.session_state.get('result_hash') == file_hash
        and 'result' in st.session_state):
        return st.session_state.result
    pil_img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    res = run_inference(pil_img)
    res['pil_img'] = pil_img
    st.session_state.result = res
    st.session_state.result_hash = file_hash
    return res


# ═══════════════════════════════════════════════════════════════════════════════
#  RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def fig_with_boxes(pil_img, boxes, color):
    """Render an image with bounding boxes using matplotlib (dark theme)."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#1a2235')
    ax.set_facecolor('#1a2235')
    ax.imshow(pil_img)
    for (x1, y1, x2, y2, conf) in boxes:
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 8, 4),
            f'fracture  {conf:.0%}',
            color='white', fontsize=11, fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.85, pad=4, edgecolor='none',
                      boxstyle='round,pad=0.4')
        )
    ax.axis('off')
    plt.tight_layout()
    return fig


def render_confidence_bars(probs, classes, pred_idx):
    """Custom HTML confidence bars. NOTE: kept on single line to prevent
    Streamlit's markdown parser from treating indented blocks as code."""
    parts = ['<div class="conf-section">']
    order = sorted(range(len(classes)), key=lambda i: -probs[i])
    for i in order:
        cls = classes[i]
        p = probs[i]
        is_frac = 'fractured' in cls
        is_pred = (i == pred_idx)
        bar_class = ('frac' if is_frac else 'normal') + (' predicted' if is_pred else '')
        clean = cls.replace('_', ' ').title()
        marker = '  ◄ predicted' if is_pred else ''
        parts.append(
            f'<div class="conf-row">'
            f'<div class="conf-label"><span>{clean}{marker}</span>'
            f'<span class="pct">{p*100:.1f}%</span></div>'
            f'<div class="conf-bar">'
            f'<div class="conf-fill {bar_class}" style="width: {p*100:.1f}%;"></div>'
            f'</div></div>'
        )
    parts.append('</div>')
    return ''.join(parts)


def render_per_class_f1_fig():
    """Per-class F1 bar chart with dark theme."""
    classes = list(METRICS['per_class_f1'].keys())
    f1s = list(METRICS['per_class_f1'].values())
    pairs = sorted(zip(classes, f1s), key=lambda x: x[1])
    classes, f1s = zip(*pairs)
    colors = ['#10b981' if f >= 0.75 else '#f59e0b' if f >= 0.60 else '#ef4444' for f in f1s]

    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor='#1a2235')
    ax.set_facecolor('#1a2235')
    bars = ax.barh(classes, f1s, color=colors, edgecolor='none', height=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('F1 Score', color='#94a3b8', fontfamily='monospace', fontsize=10)
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, f in zip(bars, f1s):
        ax.text(f + 0.015, bar.get_y() + bar.get_height() / 2,
                f'{f:.2f}', va='center', color='#f1f5f9',
                fontfamily='monospace', fontweight='bold', fontsize=10)
    ax.axvline(0.6, ls='--', color='#ef4444', alpha=0.3, lw=1)
    ax.axvline(0.75, ls='--', color='#10b981', alpha=0.3, lw=1)
    plt.tight_layout()
    return fig


def render_comparison_table():
    return f'''
    <table class="compare-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>YOLOv8s</th>
                <th>Faster R-CNN</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Precision</td><td class="winner">{METRICS["yolo_prec"]:.4f}</td><td>{METRICS["frcnn_prec"]:.4f}</td></tr>
            <tr><td>Recall</td><td>{METRICS["yolo_rec"]:.4f}</td><td class="winner">{METRICS["frcnn_rec"]:.4f}</td></tr>
            <tr><td>F1 Score</td><td>{METRICS["yolo_f1"]:.4f}</td><td class="winner">{METRICS["frcnn_f1"]:.4f}</td></tr>
            <tr><td>Avg Inference</td><td class="winner">{METRICS["yolo_ms"]:.1f}&nbsp;ms</td><td>{METRICS["frcnn_ms"]:.1f}&nbsp;ms</td></tr>
            <tr><td>Parameters</td><td>{METRICS["yolo_params"]}</td><td>{METRICS["frcnn_params"]}</td></tr>
            <tr><td>Architecture</td><td>1-stage</td><td>2-stage</td></tr>
        </tbody>
    </table>
    '''


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">🦴 FractureAI</div>'
        '<div class="sidebar-subtitle">SAINT JOSEPH UNIVERSITY · CV · SPRING 2026</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section-title">Model Performance</div>',
                unsafe_allow_html=True)

    st.markdown(f'''
    <div class="sidebar-model-card">
        <div class="sidebar-model-name">⚙ EfficientNet-B3</div>
        <div class="sidebar-kv"><span>Accuracy</span><span class="v">{METRICS["clf_acc"]*100:.2f}%</span></div>
        <div class="sidebar-kv"><span>Macro F1</span><span class="v">{METRICS["clf_f1"]:.4f}</span></div>
        <div class="sidebar-kv"><span>Parameters</span><span class="v">{METRICS["clf_params"]}</span></div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="sidebar-model-card">
        <div class="sidebar-model-name">⚡ YOLOv8s</div>
        <div class="sidebar-kv"><span>mAP@50</span><span class="v">{METRICS["yolo_map50"]*100:.1f}%</span></div>
        <div class="sidebar-kv"><span>Precision</span><span class="v">{METRICS["yolo_prec"]*100:.1f}%</span></div>
        <div class="sidebar-kv"><span>Recall</span><span class="v">{METRICS["yolo_rec"]*100:.1f}%</span></div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="sidebar-model-card">
        <div class="sidebar-model-name">🎯 Faster R-CNN</div>
        <div class="sidebar-kv"><span>Precision</span><span class="v">{METRICS["frcnn_prec"]*100:.1f}%</span></div>
        <div class="sidebar-kv"><span>Recall</span><span class="v">{METRICS["frcnn_rec"]*100:.1f}%</span></div>
        <div class="sidebar-kv"><span>F1 Score</span><span class="v">{METRICS["frcnn_f1"]*100:.1f}%</span></div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Dataset</div>',
                unsafe_allow_html=True)
    st.markdown('''
    <div class="sidebar-model-card">
        <div class="sidebar-model-name">📚 FracAtlas</div>
        <div class="sidebar-kv"><span>X-rays</span><span class="v">4,083</span></div>
        <div class="sidebar-kv"><span>Hospitals</span><span class="v">3</span></div>
        <div class="sidebar-kv"><span>Published</span><span class="v">Nature 2023</span></div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Did You Know?</div>',
                unsafe_allow_html=True)
    sidebar_fact = random.choice(FUN_FACTS)
    st.markdown(f'''
    <div class="fun-fact">
        <span class="fun-fact-icon">▸</span> {sidebar_fact}
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    device_label = '⚡ GPU' if USE_AMP else '🖥 CPU'
    st.markdown(
        f'<div class="sidebar-kv"><span>Device</span>'
        f'<span class="v">{device_label}</span></div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER + ASSET CHECK
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('''
<div class="hero-wrap">
    <div class="hero-title">FractureAI</div>
    <div class="hero-subtitle">> 3-stage deep-learning pipeline for bone-fracture detection from X-ray images</div>
</div>
''', unsafe_allow_html=True)

assets = check_assets()
if not all(assets.values()):
    missing = [k for k, v in assets.items() if not v]
    st.error(
        f'**Missing model files:** {", ".join(missing)}\n\n'
        f'Make sure Google Drive is mounted at `/content/drive` and that all '
        f'checkpoints exist under `{BASE}`. Required:\n\n'
        f'- `best_efficientnet_b3_v4.pth`\n'
        f'- `best_yolov8_fracture.pt`\n'
        f'- `best_fasterrcnn_fracture.pth`\n'
        f'- `class_info_v4.pkl`'
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_analyze, tab_pipeline, tab_about = st.tabs([
    '🔬 Analyze X-ray',
    '📊 Pipeline & Results',
    'ℹ️ About'
])


# ───────────────────────────────────────────────────────────────────────────────
#  TAB 1 — Analyze X-ray
# ───────────────────────────────────────────────────────────────────────────────
with tab_analyze:
    col_up_l, col_up_r = st.columns([2, 1])

    with col_up_l:
        st.markdown(
            '<div class="section-header">Upload an X-ray</div>',
            unsafe_allow_html=True
        )
        uploaded = st.file_uploader(
            'Drag and drop or browse — JPG / PNG · hand or leg X-rays',
            type=['jpg', 'jpeg', 'png'],
            label_visibility='collapsed'
        )

    with col_up_r:
        st.markdown(
            '<div class="section-header">While You Wait</div>',
            unsafe_allow_html=True
        )
        if 'tab1_fact' not in st.session_state:
            st.session_state.tab1_fact = random.choice(FUN_FACTS)
        st.markdown(
            f'<div class="fun-fact">'
            f'<span class="fun-fact-icon">▸</span> {st.session_state.tab1_fact}'
            f'</div>',
            unsafe_allow_html=True
        )

    if uploaded is None:
        st.markdown(
            '<div class="card">'
            '<div class="card-title">▸ Awaiting Input</div>'
            '<div style="color:#94a3b8; line-height:1.6;">'
            'Upload an X-ray image above to start the 3-stage analysis. '
            'The pipeline will run automatically:<br><br>'
            '<span style="color:#3b82f6; font-family: Space Mono, monospace;">'
            '01</span> &nbsp;EfficientNet-B3 classifies body region and fracture status<br>'
            '<span style="color:#3b82f6; font-family: Space Mono, monospace;">'
            '02</span> &nbsp;YOLOv8 or Faster R-CNN locates the fracture (if any)<br>'
            '<span style="color:#3b82f6; font-family: Space Mono, monospace;">'
            '03</span> &nbsp;Grad-CAM visualizes what the classifier focused on'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        file_bytes = uploaded.read()
        with st.spinner('Running 3-stage pipeline…'):
            result = get_or_cache_result(file_bytes)

        pil_img = result['pil_img']
        W, H = pil_img.size

        # ─── Diagnosis banner ──────────────────────────────────────────────
        if result['is_fractured']:
            banner_cls = 'banner-fracture'
            icon = '⚠️'
            title = f'FRACTURE DETECTED — {result["body_part"].upper()}'
        else:
            banner_cls = 'banner-normal'
            icon = '✅'
            title = f'NORMAL — {result["body_part"].upper()}'

        total_ms = (result['t_clf'] + result['t_cam']
                    + result['t_yolo'] + result['t_frcnn'])
        st.markdown(f'''
        <div class="banner {banner_cls}">
            <div class="banner-icon">{icon}</div>
            <div class="banner-title">{title}</div>
            <div class="banner-meta">
                CONFIDENCE: <b>{result["confidence"]*100:.1f}%</b>
                &nbsp;·&nbsp;
                TOTAL INFERENCE: <b>{total_ms:.0f}&nbsp;ms</b>
                &nbsp;·&nbsp;
                FILE: <b>{uploaded.name}</b>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # ─── 4-panel result display ───────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        # Panel 1: Original
        with col1:
            st.markdown(
                '<div class="card-title">▸ Original X-ray</div>',
                unsafe_allow_html=True
            )
            st.image(pil_img, use_container_width=True)
            st.markdown(
                f'<div style="color:#64748b; font-family: Space Mono, monospace; '
                f'font-size: 0.75rem; margin-top: 4px;">'
                f'{uploaded.name} · {W}×{H}px</div>',
                unsafe_allow_html=True
            )

        # Panel 2: Grad-CAM
        with col2:
            st.markdown(
                '<div class="card-title">▸ Grad-CAM Attention</div>',
                unsafe_allow_html=True
            )
            st.image(result['cam_image'], use_container_width=True)
            st.markdown(
                '<div style="color:#64748b; font-family: Space Mono, monospace; '
                'font-size: 0.75rem; margin-top: 4px;">'
                'Warm colors = high attention</div>',
                unsafe_allow_html=True
            )

        # Panel 3: Detection (toggle YOLO/FRCNN)
        with col3:
            st.markdown(
                '<div class="card-title">▸ Detection</div>',
                unsafe_allow_html=True
            )
            detector = st.radio(
                'Detector',
                ['YOLOv8s', 'Faster R-CNN'],
                horizontal=True,
                key='detector_tab1',
                label_visibility='collapsed'
            )
            if detector == 'YOLOv8s':
                boxes = result['yolo_boxes']
                color = '#10b981'
                t_ms = result['t_yolo']
            else:
                boxes = result['frcnn_boxes']
                color = '#3b82f6'
                t_ms = result['t_frcnn']

            if not result['is_fractured']:
                st.image(pil_img, use_container_width=True)
                st.markdown(
                    '<div style="color:#10b981; font-family: Space Mono, monospace; '
                    'font-size: 0.78rem; margin-top: 4px;">'
                    'Normal — detector not triggered</div>',
                    unsafe_allow_html=True
                )
            elif not boxes:
                st.image(pil_img, use_container_width=True)
                st.markdown(
                    f'<div style="color:#f59e0b; font-family: Space Mono, monospace; '
                    f'font-size: 0.78rem; margin-top: 4px;">'
                    f'No high-confidence detection ({detector})</div>',
                    unsafe_allow_html=True
                )
            else:
                fig = fig_with_boxes(pil_img, boxes, color)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown(
                    f'<div style="color:#64748b; font-family: Space Mono, monospace; '
                    f'font-size: 0.75rem; margin-top: 4px;">'
                    f'{len(boxes)} box(es) · {t_ms:.0f}ms</div>',
                    unsafe_allow_html=True
                )

        # Panel 4: Confidence bars
        with col4:
            st.markdown(
                '<div class="card-title">▸ Stage 1 Confidence</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                render_confidence_bars(
                    result['probs'], result['classes'], result['pred_idx']
                ),
                unsafe_allow_html=True
            )

        # ─── Disclaimer ───────────────────────────────────────────────────
        st.markdown('''
        <div class="disclaimer">
            <strong>⚕ CLINICAL DISCLAIMER —</strong>
            This is an academic research tool, not a medical device. Predictions
            are not a substitute for professional radiological interpretation.
            All clinical decisions must be made by qualified healthcare professionals.
        </div>
        ''', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
#  TAB 2 — Pipeline & Results
# ───────────────────────────────────────────────────────────────────────────────
with tab_pipeline:
    col_left, col_right = st.columns([1, 1])

    # ─── LEFT: Pipeline Architecture ──────────────────────────────────────
    with col_left:
        st.markdown(
            '<div class="section-header">Pipeline Architecture</div>',
            unsafe_allow_html=True
        )

        st.markdown('''
        <div class="pipeline-step">
            <div class="step-number">STEP 01</div>
            <div class="step-title">EfficientNet-B3 · Classification</div>
            <div class="step-desc">
                ImageNet-pretrained CNN fine-tuned in two phases on FracAtlas.
                Predicts <b>body region</b> (hand or leg) and <b>fracture status</b>
                (fractured or normal) as a single 4-way decision.
            </div>
            <div class="step-meta">
                ▸ 10.7M parameters &nbsp; · &nbsp; 300×300 input &nbsp; · &nbsp; threshold-tuned softmax
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Detector toggle for Step 02
        step2_choice = st.radio(
            'Stage 2 detector',
            ['YOLOv8s', 'Faster R-CNN'],
            horizontal=True,
            key='step2_choice',
            label_visibility='collapsed'
        )
        if step2_choice == 'YOLOv8s':
            step2_html = '''
            <div class="pipeline-step">
                <div class="step-number">STEP 02 · YOLOv8s</div>
                <div class="step-title">⚡ Single-Shot Detection</div>
                <div class="step-desc">
                    1-stage anchor-free detector. Predicts boxes directly from
                    multi-scale feature maps in a single forward pass — optimized
                    for low-latency clinical screening.
                </div>
                <div class="step-meta">
                    ▸ 11.1M params &nbsp; · &nbsp; 640×640 input &nbsp; · &nbsp; ~32 ms / image on T4
                </div>
            </div>
            '''
        else:
            step2_html = '''
            <div class="pipeline-step">
                <div class="step-number">STEP 02 · FASTER R-CNN</div>
                <div class="step-title">🎯 Two-Stage Detection</div>
                <div class="step-desc">
                    2-stage detector with ResNet-50 FPN backbone. A Region Proposal
                    Network generates candidate regions, then a head classifies
                    and refines each box — more thorough, higher recall.
                </div>
                <div class="step-meta">
                    ▸ ~41.8M params &nbsp; · &nbsp; ResNet-50 FPN &nbsp; · &nbsp; ~126 ms / image on T4
                </div>
            </div>
            '''
        st.markdown(step2_html, unsafe_allow_html=True)

        st.markdown('''
        <div class="pipeline-step">
            <div class="step-number">STEP 03</div>
            <div class="step-title">Grad-CAM · Explainability</div>
            <div class="step-desc">
                Gradient-weighted Class Activation Mapping highlights the
                regions of the X-ray that drove the classifier's decision.
                Targets the last MBConv block of EfficientNet-B3.
            </div>
            <div class="step-meta">
                ▸ Zero training overhead &nbsp; · &nbsp; ICCV 2017 &nbsp; · &nbsp; visual sanity check
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Key design decisions
        st.markdown(
            '<div class="section-header">Key Design Decisions</div>',
            unsafe_allow_html=True
        )

        decisions = [
            ('No ColorJitter',
             'X-ray pixel intensity encodes bone density. Color/contrast jitter '
             'corrupts diagnostic information.'),
            ('Grayscale → 3 channels',
             'Some FracAtlas images are L-mode. Replicate to 3 channels so '
             'ImageNet-pretrained backbone receives expected input.'),
            ('Threshold tuning (+10pp F1)',
             'Per-class confidence thresholds found on val set: [0.5, 0.32, 0.5, 0.1]. '
             'Macro F1 improved 0.69 → 0.76 with zero retraining.'),
            ('Two-phase fine-tuning',
             'Phase 1 (15 epochs): backbone frozen, head trains at LR=3e-3. '
             'Phase 2: full fine-tuning with backbone LR=1e-5. Prevents '
             'catastrophic forgetting of ImageNet features.'),
            ('WeightedRandomSampler ON',
             '9× imbalance between leg_normal and leg_fractured. Sampler + '
             'class weights in loss together combat majority bias.'),
        ]
        for title, desc in decisions:
            st.markdown(
                f'<div class="decision-card">'
                f'<div class="decision-title">▸ {title}</div>'
                f'<div class="decision-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ─── RIGHT: Final Results ─────────────────────────────────────────────
    with col_right:
        st.markdown(
            '<div class="section-header">Final Results</div>',
            unsafe_allow_html=True
        )

        # EfficientNet metric card
        st.markdown(f'''
        <div class="card">
            <div class="card-title">▸ EfficientNet-B3 · Stage 1 · Test Set</div>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div class="metric-tile">
                    <div class="metric-label">Test Accuracy</div>
                    <div class="metric-value">{METRICS["clf_acc"]*100:.2f}%</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-label">Macro F1</div>
                    <div class="metric-value">{METRICS["clf_f1"]:.4f}</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Per-class F1 chart
        st.markdown(
            '<div class="card-title">▸ Per-Class F1 (Test Set)</div>',
            unsafe_allow_html=True
        )
        fig_f1 = render_per_class_f1_fig()
        st.pyplot(fig_f1, use_container_width=True)
        plt.close(fig_f1)
        st.markdown(
            '<div style="color:#64748b; font-family: Space Mono, monospace; '
            'font-size: 0.75rem; margin-top: 4px;">'
            'Green ≥ 0.75 &nbsp;|&nbsp; Yellow 0.60–0.75 &nbsp;|&nbsp; Red &lt; 0.60'
            '</div>',
            unsafe_allow_html=True
        )

        # Comparison table
        st.markdown(
            '<div class="section-header" style="margin-top: 26px;">'
            'YOLOv8 vs Faster R-CNN'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(render_comparison_table(), unsafe_allow_html=True)

        st.markdown('''
        <div style="color:#94a3b8; font-size: 0.85rem; line-height: 1.55; margin-top: 8px;">
            ▸ Hand + leg test images only — consistent with Stage 1 scope.<br>
            ▸ Clinical winner: <span style="color:#10b981; font-weight:700;">
            Faster R-CNN</span> (higher recall — missing a fracture is worse
            than a false alarm).<br>
            ▸ Speed winner: <span style="color:#10b981; font-weight:700;">
            YOLOv8s</span> (~4× faster — better for real-time screening).
        </div>
        ''', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
#  TAB 3 — About
# ───────────────────────────────────────────────────────────────────────────────
with tab_about:
    col_l, col_r = st.columns([1, 1])

    # ─── LEFT: Team & References ──────────────────────────────────────────
    with col_l:
        st.markdown(
            '<div class="section-header">Project Information</div>',
            unsafe_allow_html=True
        )
        st.markdown('''
        <div class="card">
            <div style="display: grid; grid-template-columns: auto 1fr; gap: 14px 16px; align-items: center;">
                <div style="color:#94a3b8; font-family: Space Mono, monospace; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px;">University</div>
                <div style="color:#f1f5f9; font-weight: 700;">Saint Joseph University · Beirut</div>
                <div style="color:#94a3b8; font-family: Space Mono, monospace; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px;">Course</div>
                <div style="color:#f1f5f9; font-weight: 700;">Computer Vision · Spring 2026</div>
                <div style="color:#94a3b8; font-family: Space Mono, monospace; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px;">Task</div>
                <div style="color:#f1f5f9; font-weight: 700;">3-stage fracture detection pipeline</div>
                <div style="color:#94a3b8; font-family: Space Mono, monospace; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px;">Dataset</div>
                <div style="color:#f1f5f9; font-weight: 700;">FracAtlas — 4,083 X-rays · 3 hospitals</div>
                <div style="color:#94a3b8; font-family: Space Mono, monospace; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px;">Annotators</div>
                <div style="color:#f1f5f9; font-weight: 700;">2 radiologists + orthopedic surgeon</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-header">References</div>',
            unsafe_allow_html=True
        )
        refs = [
            ('FracAtlas',
             'A Dataset for Fracture Classification, Localization and Segmentation',
             'Islam et al. · Scientific Data (Nature) · 2023'),
            ('EfficientNet',
             'Rethinking Model Scaling for Convolutional Neural Networks',
             'Tan & Le · ICML · 2019'),
            ('YOLOv8',
             'Real-Time Object Detection',
             'Ultralytics · 2023'),
            ('Faster R-CNN',
             'Towards Real-Time Object Detection with Region Proposal Networks',
             'Ren et al. · NeurIPS · 2015'),
            ('Grad-CAM',
             'Visual Explanations from Deep Networks via Gradient-based Localization',
             'Selvaraju et al. · ICCV · 2017'),
        ]
        for short, full, venue in refs:
            st.markdown(f'''
            <div class="ref-item">
                <div class="ref-title">{short}</div>
                <div style="color:#cbd5e1; font-size: 0.85rem; margin-top: 2px;">{full}</div>
                <div class="ref-venue">{venue}</div>
            </div>
            ''', unsafe_allow_html=True)

    # ─── RIGHT: Tech Stack & Architecture ─────────────────────────────────
    with col_r:
        st.markdown(
            '<div class="section-header">Technical Stack</div>',
            unsafe_allow_html=True
        )
        st.markdown('''
        <div class="card">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px 18px; color:#cbd5e1;">
                <div>▸ Python 3.12</div>
                <div>▸ PyTorch 2.x</div>
                <div>▸ torchvision</div>
                <div>▸ Ultralytics YOLOv8</div>
                <div>▸ scikit-learn</div>
                <div>▸ pytorch-grad-cam</div>
                <div>▸ Google Colab · T4 GPU</div>
                <div>▸ Streamlit</div>
                <div>▸ matplotlib / seaborn</div>
                <div>▸ PIL / NumPy / pandas</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-header">Model Architecture</div>',
            unsafe_allow_html=True
        )
        st.markdown('''
<div class="code-block">
<span class="com"># Stage 1 — EfficientNet-B3 (4-class classifier)</span>
backbone     = <span class="kw">efficientnet_b3</span>(weights=ImageNet1k)
classifier   = <span class="kw">nn.Sequential</span>(
    <span class="kw">nn.Dropout</span>(p=<span class="str">0.4</span>),
    <span class="kw">nn.Linear</span>(<span class="str">1536</span>, <span class="str">4</span>)
)

<span class="com"># Two-phase fine-tuning</span>
phase_1 = freeze_backbone(), lr_head=<span class="str">3e-3</span>, epochs=<span class="str">15</span>
phase_2 = unfreeze_all(),    lr_bb=<span class="str">1e-5</span>, lr_head=<span class="str">1e-4</span>

<span class="com"># Stage 2 — detector ensemble</span>
yolov8s      = <span class="kw">YOLO</span>(<span class="str">'yolov8s.pt'</span>)        <span class="com"># 1-stage</span>
faster_rcnn  = <span class="kw">fasterrcnn_resnet50_fpn</span>()  <span class="com"># 2-stage</span>

<span class="com"># Stage 3 — Grad-CAM (target last MBConv)</span>
cam = <span class="kw">GradCAM</span>(model, target_layers=[model.features[-<span class="str">1</span>]])
</div>
        ''', unsafe_allow_html=True)

    # ─── Bottom: Did You Know cards ───────────────────────────────────────
    st.markdown(
        '<div class="section-header" style="margin-top: 30px;">'
        '🦴 Did You Know?'
        '</div>',
        unsafe_allow_html=True
    )

    if 'about_facts' not in st.session_state:
        st.session_state.about_facts = random.sample(FUN_FACTS, 4)
    facts = st.session_state.about_facts
    fc1, fc2, fc3, fc4 = st.columns(4)
    for col, fact in zip([fc1, fc2, fc3, fc4], facts):
        with col:
            st.markdown(
                f'<div class="fact-card">{fact}</div>',
                unsafe_allow_html=True
            )
