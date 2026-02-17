"""
Adversarial Attack — Web Application (TikTok-targeted ensemble)
================================================================
Flask web app: pick a target class, upload a source image,
make the source "become" the target in the eyes of TikTok-like classifiers.

Features:
  - 5-model ensemble matched to TikTok's vision stack
  - Real-time progress bar via threaded processing + polling
  - Full-resolution output (noise upscaled to original image size)

Usage:
  python webapp.py
  Then open http://localhost:5000 in your browser.
"""

import base64
import io
import json
import threading
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
from torchvision import models, transforms

from adversarial_attack import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_imagenet_labels,
    normalise,
    pgd_targeted,
    fgsm_targeted,
    ensemble_mi_di_ti_fgsm,
)

app = Flask(__name__)

# ── Preset target classes ────────────────────────────────────────────────────

PRESET_TARGETS = [
    {"idx": 980, "name": "Вулкан", "icon": "🌋"},
    {"idx": 970, "name": "Горы (Альпы)", "icon": "🏔️"},
    {"idx": 978, "name": "Побережье", "icon": "🏖️"},
    {"idx": 975, "name": "Озеро", "icon": "🏞️"},
    {"idx": 973, "name": "Коралловый риф", "icon": "🪸"},
    {"idx": 281, "name": "Полосатый кот", "icon": "🐱"},
    {"idx": 207, "name": "Золотистый ретривер", "icon": "🐕"},
    {"idx": 388, "name": "Панда", "icon": "🐼"},
    {"idx": 949, "name": "Клубника", "icon": "🍓"},
    {"idx": 954, "name": "Банан", "icon": "🍌"},
    {"idx": 985, "name": "Ромашка", "icon": "🌼"},
    {"idx": 309, "name": "Пчела", "icon": "🐝"},
    {"idx": 717, "name": "Церковь", "icon": "⛪"},
    {"idx": 510, "name": "Спорткар", "icon": "🏎️"},
    {"idx": 920, "name": "Маяк", "icon": "🗼"},
    {"idx": 113, "name": "Улитка", "icon": "🐌"},
]

# ── Load TikTok-matched ensemble at startup ─────────────────────────────────

MODELS = {}

print("[1/5] Loading ResNet-50 (→ TikTok moderation CNN) …")
MODELS["ResNet-50"] = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
MODELS["ResNet-50"].eval()

print("[2/5] Loading EfficientNet-V2-S (→ TikTok moderation CNN) …")
MODELS["EfficientNet-V2"] = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.DEFAULT)
MODELS["EfficientNet-V2"].eval()

print("[3/5] Loading Swin-T (→ Video Swin Transformer) …")
MODELS["Swin-T"] = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
MODELS["Swin-T"].eval()

print("[4/5] Loading ViT-B/16 (→ BEiT3 vision encoder) …")
MODELS["ViT-B/16"] = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
MODELS["ViT-B/16"].eval()

print("[5/5] Loading ConvNeXt-Small (extra diversity) …")
MODELS["ConvNeXt-S"] = models.convnext_small(
    weights=models.ConvNeXt_Small_Weights.DEFAULT)
MODELS["ConvNeXt-S"].eval()

ENSEMBLE = list(MODELS.values())

MODEL_ROLES = {
    "ResNet-50": "Модерация (CNN)",
    "EfficientNet-V2": "Модерация (CNN)",
    "Swin-T": "Video Swin Transformer",
    "ViT-B/16": "BEiT3 vision encoder",
    "ConvNeXt-S": "Доп. backbone",
}

print("[*] Loading ImageNet labels …")
LABELS = load_imagenet_labels()

print(f"[✓] {len(ENSEMBLE)} models loaded — open http://localhost:5000")

# ── Task tracking for async processing ───────────────────────────────────────

_tasks: dict = {}
_last_result_png: bytes | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_image_for_attack(img_bytes: bytes, size: int = 224) -> torch.Tensor:
    """Load image resized to model input size (stretches to square)."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def load_image_full_res(img_bytes: bytes) -> torch.Tensor:
    """Load image at original resolution."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return transforms.ToTensor()(img).unsqueeze(0)


def get_image_dimensions(img_bytes: bytes) -> tuple[int, int]:
    """Get (width, height) of image."""
    img = Image.open(io.BytesIO(img_bytes))
    return img.size  # (W, H)


def classify(img_tensor: torch.Tensor, model=None) -> list[dict]:
    """Return top-5 predictions as list of {idx, label, confidence}."""
    if model is None:
        model = ENSEMBLE[0]
    with torch.no_grad():
        logits = model(normalise(img_tensor))
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


def tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    """Convert 1×3×H×W [0,1] tensor to PNG bytes."""
    img_np = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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
    return render_template("index.html", presets=PRESET_TARGETS)


@app.route("/transform", methods=["POST"])
def transform():
    """Start adversarial attack in background thread. Returns task_id."""
    global _last_result_png

    source_file = request.files.get("source_image")
    target_class_idx = request.form.get("target_class")

    if not source_file:
        return jsonify({"error": "Загрузите исходное изображение."}), 400
    if target_class_idx is None:
        return jsonify({"error": "Выберите целевой класс."}), 400

    # Read file into memory (needed for thread)
    img_bytes = source_file.read()
    target_class_idx = int(target_class_idx)
    method = request.form.get("method", "ensemble")
    epsilon = float(request.form.get("epsilon", "0.06"))
    steps = int(request.form.get("steps", "120"))

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "progress": 0,
        "total": steps,
        "phase": "starting",
        "status": "running",
        "result": None,
    }

    def run_attack():
        global _last_result_png
        try:
            task = _tasks[task_id]
            target_label = LABELS[target_class_idx]

            # ── Phase 1: Load images ──
            task["phase"] = "loading"
            source_224 = load_image_for_attack(img_bytes, 224)
            source_full = load_image_full_res(img_bytes)
            orig_h, orig_w = source_full.shape[2], source_full.shape[3]

            # Classify source
            source_preds = classify(source_224, ENSEMBLE[0])
            source_label = source_preds[0]["label"]
            source_conf = source_preds[0]["confidence"]

            # ── Phase 2: Attack at 224x224 ──
            task["phase"] = "attacking"

            def on_progress(step, total):
                task["progress"] = step + 1
                task["total"] = total

            if method == "fgsm":
                task["total"] = 1
                adv_224 = fgsm_targeted(ENSEMBLE[0], source_224,
                                        target_class_idx, epsilon)
                task["progress"] = 1
            elif method == "pgd":
                adv_224 = pgd_targeted(
                    ENSEMBLE[0], source_224, target_class_idx,
                    epsilon, steps=steps, progress_callback=on_progress,
                )
            else:
                adv_224 = ensemble_mi_di_ti_fgsm(
                    ENSEMBLE, source_224, target_class_idx,
                    epsilon=epsilon, steps=steps,
                    progress_callback=on_progress,
                )

            # ── Phase 3: Upscale noise to full resolution ──
            task["phase"] = "upscaling"
            noise_224 = adv_224 - source_224
            noise_full = F.interpolate(
                noise_224, size=(orig_h, orig_w),
                mode='bilinear', align_corners=False,
            )
            adv_full = torch.clamp(source_full + noise_full, 0, 1)

            # ── Phase 4: Classify on all models ──
            task["phase"] = "classifying"

            models_fooled = 0
            model_results = []
            first_preds = None

            for name, model in MODELS.items():
                preds = classify(adv_224, model)
                if first_preds is None:
                    first_preds = preds
                fooled = preds[0]["idx"] == target_class_idx
                if fooled:
                    models_fooled += 1
                model_results.append({
                    "name": name,
                    "role": MODEL_ROLES[name],
                    "label": preds[0]["label"],
                    "confidence": preds[0]["confidence"],
                    "fooled": fooled,
                })

            adv_label = first_preds[0]["label"]
            adv_conf = first_preds[0]["confidence"]
            success = first_preds[0]["idx"] == target_class_idx

            # Save full-res result for download
            _last_result_png = tensor_to_png_bytes(adv_full)

            # Noise stats
            noise = adv_224 - source_224
            l_inf = noise.abs().max().item()
            l_2 = noise.norm(2).item()

            task["result"] = {
                "success": success,
                "models_fooled": models_fooled,
                "models_total": len(ENSEMBLE),
                "model_results": model_results,
                "target_label": target_label,
                "source_label": source_label,
                "source_confidence": source_conf,
                "source_image": tensor_to_base64(source_224),
                "result_label": adv_label,
                "result_confidence": adv_conf,
                "result_image": tensor_to_base64(adv_224),
                "noise_image": noise_to_base64(source_224, adv_224),
                "result_top5": first_preds,
                "noise_l_inf": round(l_inf, 4),
                "noise_l_2": round(l_2, 4),
                "max_pixel_change": round(l_inf * 255, 1),
                "method": method.upper(),
                "epsilon": epsilon,
                "steps": steps,
                "output_resolution": f"{orig_w}x{orig_h}",
            }
            task["status"] = "done"
            task["phase"] = "done"

        except Exception as e:
            _tasks[task_id]["status"] = "error"
            _tasks[task_id]["error"] = str(e)

    thread = threading.Thread(target=run_attack, daemon=True)
    thread.start()

    return jsonify({"task_id": task_id})


@app.route("/progress/<task_id>")
def progress(task_id):
    """Poll progress of a running attack task."""
    task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    if task["status"] == "error":
        return jsonify({
            "status": "error",
            "error": task.get("error", "Unknown error"),
        })

    if task["status"] == "done":
        result = task["result"]
        # Clean up task
        del _tasks[task_id]
        return jsonify({
            "status": "done",
            "result": result,
        })

    return jsonify({
        "status": "running",
        "progress": task["progress"],
        "total": task["total"],
        "phase": task["phase"],
    })


@app.route("/download")
def download():
    """Download the last generated adversarial image as PNG."""
    if _last_result_png is None:
        return "No result available", 404
    buf = io.BytesIO(_last_result_png)
    return send_file(buf, mimetype="image/png", as_attachment=True,
                     download_name="adversarial_result.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adversarial Attack Web App")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    args = parser.parse_args()
    app.run(debug=True, host=args.host, port=args.port, threaded=True)
