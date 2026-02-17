"""
Adversarial Attack — Web Application
=====================================
Flask web app: upload two images, make one "become" the other
in the eyes of a neural network while preserving visual proportions.

Usage:
  python webapp.py
  Then open http://localhost:5000 in your browser.
"""

import base64
import io
import json

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import models, transforms

from adversarial_attack import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_imagenet_labels,
    normalise,
    pgd_targeted,
    fgsm_targeted,
)

app = Flask(__name__)

# ── Load model & labels once at startup ──────────────────────────────────────

print("[*] Loading ResNet-50 …")
MODEL = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
MODEL.eval()

print("[*] Loading ImageNet labels …")
LABELS = load_imagenet_labels()

print("[✓] Ready — open http://localhost:5000")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_image_preserve_aspect(file_storage, size: int = 224) -> torch.Tensor:
    """Load uploaded image using center-crop to preserve aspect ratio."""
    img = Image.open(file_storage).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),       # shortest side → 224
        transforms.CenterCrop(size),   # crop center square (no stretching)
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # 1×3×224×224


def classify(img_tensor: torch.Tensor) -> list[dict]:
    """Return top-5 predictions as list of {idx, label, confidence}."""
    with torch.no_grad():
        logits = MODEL(normalise(img_tensor))
    probs = F.softmax(logits, dim=1)
    top5_conf, top5_idx = torch.topk(probs, 5, dim=1)

    results = []
    for i in range(5):
        idx = top5_idx[0, i].item()
        results.append({
            "idx": idx,
            "label": LABELS[idx],
            "confidence": round(top5_conf[0, i].item() * 100, 1),
        })
    return results


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert 1×3×H×W [0,1] tensor to base64-encoded PNG."""
    img_np = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def noise_to_base64(original: torch.Tensor, adversarial: torch.Tensor,
                    scale: float = 10.0) -> str:
    """Amplify and encode the noise difference as a visible PNG."""
    noise = adversarial - original
    noise_vis = noise.squeeze(0).permute(1, 2, 0).numpy() * scale + 0.5
    noise_vis = np.clip(noise_vis, 0, 1)
    noise_vis = (noise_vis * 255).astype(np.uint8)
    img = Image.fromarray(noise_vis)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transform", methods=["POST"])
def transform():
    """
    Accept two uploaded images + settings.
    1) Classify the target image to find its class.
    2) Attack the source image to be classified as that class.
    3) Return results as JSON with base64-encoded images.
    """
    target_file = request.files.get("target_image")
    source_file = request.files.get("source_image")

    if not target_file or not source_file:
        return jsonify({"error": "Both images are required."}), 400

    method = request.form.get("method", "pgd")
    epsilon = float(request.form.get("epsilon", "0.03"))
    steps = int(request.form.get("steps", "40"))

    # Load images (center-crop preserves proportions — no stretching)
    target_tensor = load_image_preserve_aspect(target_file)
    source_tensor = load_image_preserve_aspect(source_file)

    # Classify target image → this is the class we want to fool the network into
    target_preds = classify(target_tensor)
    target_class_idx = target_preds[0]["idx"]
    target_label = target_preds[0]["label"]
    target_conf = target_preds[0]["confidence"]

    # Classify source image before attack
    source_preds = classify(source_tensor)
    source_label = source_preds[0]["label"]
    source_conf = source_preds[0]["confidence"]

    # Run adversarial attack
    if method == "fgsm":
        adv_tensor = fgsm_targeted(MODEL, source_tensor, target_class_idx, epsilon)
    else:
        adv_tensor = pgd_targeted(
            MODEL, source_tensor, target_class_idx, epsilon, steps=steps
        )

    # Classify result
    adv_preds = classify(adv_tensor)
    adv_label = adv_preds[0]["label"]
    adv_conf = adv_preds[0]["confidence"]
    success = adv_preds[0]["idx"] == target_class_idx

    # Noise stats
    noise = adv_tensor - source_tensor
    l_inf = noise.abs().max().item()
    l_2 = noise.norm(2).item()

    return jsonify({
        "success": success,
        # Target image info
        "target_label": target_label,
        "target_confidence": target_conf,
        "target_image": tensor_to_base64(target_tensor),
        # Source image info (before attack)
        "source_label": source_label,
        "source_confidence": source_conf,
        "source_image": tensor_to_base64(source_tensor),
        # Result
        "result_label": adv_label,
        "result_confidence": adv_conf,
        "result_image": tensor_to_base64(adv_tensor),
        "noise_image": noise_to_base64(source_tensor, adv_tensor),
        # Top-5 predictions after attack
        "result_top5": adv_preds,
        # Stats
        "noise_l_inf": round(l_inf, 4),
        "noise_l_2": round(l_2, 4),
        "max_pixel_change": round(l_inf * 255, 1),
        # Settings used
        "method": method.upper(),
        "epsilon": epsilon,
        "steps": steps,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
