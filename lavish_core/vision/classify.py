
from __future__ import annotations
from typing import List, Dict
import torch
from torchvision import models, transforms
from PIL import Image

# Lazy-load global model once
_model = None
_labels = None

def _load_model():
    global _model, _labels
    if _model is None:
        _model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        _model.eval()
        _labels = models.ResNet50_Weights.DEFAULT.meta["categories"]

def classify_topk(path: str, k: int = 5) -> List[Dict]:
    _load_model()
    preprocess = models.ResNet50_Weights.DEFAULT.transforms()
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, k=k)
    out = []
    for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
        out.append({"label": _labels[idx], "prob": float(prob)})
    return out