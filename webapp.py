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
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
from torchvision import models, transforms

from adversarial_attack import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEVICE,
    load_imagenet_labels,
    normalise,
    pgd_targeted,
    fgsm_targeted,
    ensemble_mi_di_ti_fgsm,
)
try:
    from video_attack import attack_video_fast, attack_video_warmstart
    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload

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

print(f"[*] Device: {DEVICE}")

MODELS = {}

def _load_model(name, factory, role_hint):
    print(f"  Loading {name} ({role_hint}) …")
    m = factory()
    m.eval()
    m.to(DEVICE)
    return m

print("[1/5] Loading models …")
MODELS["ResNet-50"] = _load_model(
    "ResNet-50", lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    "TikTok moderation CNN")
MODELS["EfficientNet-V2"] = _load_model(
    "EfficientNet-V2", lambda: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT),
    "TikTok moderation CNN")
MODELS["Swin-T"] = _load_model(
    "Swin-T", lambda: models.swin_t(weights=models.Swin_T_Weights.DEFAULT),
    "Video Swin Transformer")
MODELS["ViT-B/16"] = _load_model(
    "ViT-B/16", lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT),
    "BEiT3 vision encoder")
MODELS["ConvNeXt-S"] = _load_model(
    "ConvNeXt-S", lambda: models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT),
    "extra backbone diversity")

ENSEMBLE = list(MODELS.values())

MODEL_ROLES = {
    "ResNet-50": "Модерация (CNN)",
    "EfficientNet-V2": "Модерация (CNN)",
    "Swin-T": "Video Swin Transformer",
    "ViT-B/16": "BEiT3 vision encoder",
    "ConvNeXt-S": "Доп. backbone",
}

# Cache labels locally so we don't depend on internet after first run
LABELS_CACHE = Path(__file__).parent / "imagenet_labels.json"

def _load_labels():
    if LABELS_CACHE.exists():
        return json.loads(LABELS_CACHE.read_text())
    labels = load_imagenet_labels()
    if labels[0] != "0":  # successfully fetched, not fallback indices
        LABELS_CACHE.write_text(json.dumps(labels, ensure_ascii=False))
    return labels

print("[*] Loading ImageNet labels …")
LABELS = _load_labels()

# ── Face detection & recognition for celebrity mode ──────────────────────────

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    print("[*] Loading face detection & recognition …")
    _face_mtcnn = MTCNN(keep_all=False, device=DEVICE)
    FACE_MODEL = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    FACE_SUPPORT = True
    print("[*] Face models loaded (celebrity mode available)")
except ImportError:
    _face_mtcnn = None
    FACE_MODEL = None
    FACE_SUPPORT = False
    print("[*] Face models unavailable — install facenet-pytorch for celebrity mode")


def detect_face_box(img_tensor_224: torch.Tensor):
    """Detect face in 1×3×224×224 tensor [0,1]. Returns (x1,y1,x2,y2) or None."""
    if _face_mtcnn is None:
        return None
    with torch.no_grad():
        img_np = (img_tensor_224.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        boxes, probs = _face_mtcnn.detect(pil_img)
        if boxes is not None and len(boxes) > 0 and probs[0] > 0.9:
            return tuple(boxes[0].tolist())
    return None


# ── OCR text detection for text/number protection ────────────────────────────

try:
    import easyocr
    print("[*] Loading OCR text detection …")
    _ocr_reader = easyocr.Reader(
        ['en', 'ru'], gpu=(str(DEVICE) != 'cpu'), verbose=False,
    )
    OCR_SUPPORT = True
    print("[*] OCR models loaded (text protection available)")
except ImportError:
    _ocr_reader = None
    OCR_SUPPORT = False
    print("[*] OCR unavailable — install easyocr for text/number protection")


def detect_text_boxes_224(frame_rgb: np.ndarray) -> list[tuple] | None:
    """Detect text regions in full-res frame, return boxes scaled to 224×224."""
    if _ocr_reader is None:
        return None
    h, w = frame_rgb.shape[:2]
    results = _ocr_reader.readtext(frame_rgb)
    if not results:
        return None
    boxes = []
    sx, sy = 224.0 / w, 224.0 / h
    for (bbox, text, conf) in results:
        if conf < 0.3:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1 = min(xs) * sx, min(ys) * sy
        x2, y2 = max(xs) * sx, max(ys) * sy
        boxes.append((x1, y1, x2, y2))
    return boxes if boxes else None


print(f"[OK] {len(ENSEMBLE)} models on {DEVICE} — open http://localhost:5000")

# ── Task tracking for async processing ───────────────────────────────────────

_tasks: dict = {}
_tasks_lock = threading.Lock()
_results_lock = threading.Lock()
_last_result_png: bytes | None = None
_last_result_video: bytes | None = None
_TASK_MAX_AGE = 600  # auto-expire tasks older than 10 minutes


def _cleanup_tasks():
    """Remove stale tasks to prevent memory leaks."""
    now = time.time()
    with _tasks_lock:
        stale = [tid for tid, t in _tasks.items()
                 if now - t.get("created", 0) > _TASK_MAX_AGE]
        for tid in stale:
            del _tasks[tid]


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
    dev = next(model.parameters()).device
    img_d = img_tensor.to(dev)
    with torch.no_grad():
        logits = model(normalise(img_d))
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


@app.route("/device")
def device_info():
    """Return current compute device info."""
    return jsonify({
        "device": str(DEVICE),
        "face_support": FACE_SUPPORT,
        "ocr_support": OCR_SUPPORT,
    })


@app.route("/transform", methods=["POST"])
def transform():
    """Start adversarial attack in background thread. Returns task_id."""
    global _last_result_png
    _cleanup_tasks()

    source_file = request.files.get("source_image")
    target_class_idx = request.form.get("target_class")

    if not source_file:
        return jsonify({"error": "Загрузите исходное изображение."}), 400
    if target_class_idx is None:
        return jsonify({"error": "Выберите целевой класс."}), 400

    img_bytes = source_file.read()
    if len(img_bytes) > 20 * 1024 * 1024:
        return jsonify({"error": "Файл слишком большой (макс. 20 МБ)"}), 400

    target_class_idx = int(target_class_idx)
    method = request.form.get("method", "ensemble")
    epsilon = float(request.form.get("epsilon", "0.06"))
    steps = int(request.form.get("steps", "120"))
    robust = request.form.get("robust", "false") == "true"
    celebrity = request.form.get("celebrity", "false") == "true"
    ocr_protect = request.form.get("ocr_protect", "false") == "true"

    cancel_flag = [False]
    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {
            "progress": 0,
            "total": steps,
            "phase": "starting",
            "status": "running",
            "result": None,
            "created": time.time(),
            "started": time.time(),
            "cancel_flag": cancel_flag,
        }

    def run_attack():
        global _last_result_png
        try:
            task = _tasks[task_id]
            target_label = LABELS[target_class_idx]

            task["phase"] = "loading"
            source_224 = load_image_for_attack(img_bytes, 224)
            source_full = load_image_full_res(img_bytes)
            orig_h, orig_w = source_full.shape[2], source_full.shape[3]

            source_preds = classify(source_224, ENSEMBLE[0])
            source_label = source_preds[0]["label"]
            source_conf = source_preds[0]["confidence"]

            task["phase"] = "attacking"
            task["started"] = time.time()

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
                face_box = None
                if celebrity and FACE_SUPPORT:
                    face_box = detect_face_box(source_224)
                text_boxes = None
                if ocr_protect and OCR_SUPPORT:
                    orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    text_boxes = detect_text_boxes_224(np.array(orig_img))
                adv_224 = ensemble_mi_di_ti_fgsm(
                    ENSEMBLE, source_224, target_class_idx,
                    epsilon=epsilon, steps=steps,
                    progress_callback=on_progress,
                    cancel_flag=cancel_flag,
                    compression_robust=robust,
                    face_model=FACE_MODEL if face_box else None,
                    face_box=face_box,
                    text_boxes=text_boxes,
                )

            if cancel_flag[0]:
                task["status"] = "cancelled"
                return

            task["phase"] = "upscaling"
            noise_224 = adv_224 - source_224
            noise_full = F.interpolate(
                noise_224, size=(orig_h, orig_w),
                mode='bilinear', align_corners=False,
            )
            adv_full = torch.clamp(source_full + noise_full, 0, 1)

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

            with _results_lock:
                _last_result_png = tensor_to_png_bytes(adv_full)

            noise = adv_224 - source_224
            l_inf = noise.abs().max().item()
            l_2 = noise.norm(2).item()
            elapsed = round(time.time() - task["created"], 1)

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
                "elapsed_sec": elapsed,
                "device": str(DEVICE),
                "robust": robust,
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

    if task["status"] == "cancelled":
        with _tasks_lock:
            _tasks.pop(task_id, None)
        return jsonify({"status": "cancelled"})

    if task["status"] == "error":
        error = task.get("error", "Unknown error")
        with _tasks_lock:
            _tasks.pop(task_id, None)
        return jsonify({"status": "error", "error": error})

    if task["status"] == "done":
        result = task["result"]
        with _tasks_lock:
            _tasks.pop(task_id, None)
        return jsonify({"status": "done", "result": result})

    # Running — compute ETA
    elapsed = time.time() - task.get("started", task.get("created", time.time()))
    prog = task["progress"]
    total = task["total"]
    eta = None
    if prog > 0 and total > 0:
        rate = elapsed / prog
        eta = round(rate * (total - prog), 1)

    resp = {
        "status": "running",
        "progress": prog,
        "total": total,
        "phase": task["phase"],
        "elapsed": round(elapsed, 1),
        "eta": eta,
    }
    if task.get("extra"):
        resp["extra"] = task["extra"]
    if task.get("media_type"):
        resp["media_type"] = task["media_type"]
    return jsonify(resp)


@app.route("/cancel/<task_id>", methods=["POST"])
def cancel_task(task_id):
    """Cancel a running attack."""
    task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    flag = task.get("cancel_flag")
    if flag:
        flag[0] = True
    return jsonify({"ok": True})


@app.route("/download")
def download():
    """Download the last generated adversarial image as PNG."""
    with _results_lock:
        data = _last_result_png
    if data is None:
        return "No result available", 404
    buf = io.BytesIO(data)
    return send_file(buf, mimetype="image/png", as_attachment=True,
                     download_name="adversarial_result.png")


@app.route("/transform_video", methods=["POST"])
def transform_video():
    """Start video adversarial attack in background thread. Returns task_id."""
    global _last_result_video
    _cleanup_tasks()

    if not VIDEO_SUPPORT:
        return jsonify({"error": "Видео не поддерживается: pip install opencv-python-headless"}), 400

    source_file = request.files.get("source_video")
    target_class_idx = request.form.get("target_class")

    if not source_file:
        return jsonify({"error": "Загрузите видео."}), 400
    if target_class_idx is None:
        return jsonify({"error": "Выберите целевой класс."}), 400

    video_bytes = source_file.read()
    if len(video_bytes) > 150 * 1024 * 1024:
        return jsonify({"error": "Видео слишком большое (макс. 150 МБ)"}), 400

    target_class_idx = int(target_class_idx)
    mode = request.form.get("video_mode", "fast")
    epsilon = float(request.form.get("epsilon", "0.06"))
    steps = int(request.form.get("steps", "120"))
    robust = request.form.get("robust", "true") == "true"  # default ON for video
    celebrity = request.form.get("celebrity", "false") == "true"
    ocr_protect = request.form.get("ocr_protect", "false") == "true"

    cancel_flag = [False]
    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {
            "progress": 0,
            "total": steps,
            "phase": "starting",
            "status": "running",
            "result": None,
            "media_type": "video",
            "created": time.time(),
            "started": time.time(),
            "cancel_flag": cancel_flag,
        }

    def run_video_attack():
        global _last_result_video
        try:
            task = _tasks[task_id]
            target_label = LABELS[target_class_idx]
            task["started"] = time.time()

            def on_progress(phase, frame_idx, total_frames, step, total_steps):
                task["phase"] = phase
                if phase == "attacking":
                    task["progress"] = step
                    task["total"] = total_steps
                    task["extra"] = f"Кадр {frame_idx}/{total_frames}" if total_frames > 1 else ""
                elif phase == "applying":
                    task["progress"] = frame_idx
                    task["total"] = total_frames
                    task["extra"] = ""
                elif phase == "interpolating":
                    task["progress"] = 0
                    task["total"] = total_frames
                    task["extra"] = ""
                elif phase in ("extracting", "saving"):
                    task["extra"] = ""

            _fm = FACE_MODEL if (celebrity and FACE_SUPPORT) else None
            _fd = detect_face_box if (celebrity and FACE_SUPPORT) else None
            _td = detect_text_boxes_224 if (ocr_protect and OCR_SUPPORT) else None

            if mode == "quality":
                result_bytes = attack_video_warmstart(
                    ENSEMBLE, video_bytes, target_class_idx,
                    epsilon=epsilon,
                    steps_first=steps,
                    steps_warm=max(20, steps // 4),
                    keyframe_fps=2.0,
                    progress_callback=on_progress,
                    compression_robust=robust,
                    face_model=_fm,
                    face_detector=_fd,
                    text_detector=_td,
                )
            else:
                result_bytes = attack_video_fast(
                    ENSEMBLE, video_bytes, target_class_idx,
                    epsilon=epsilon,
                    steps=steps,
                    progress_callback=on_progress,
                    compression_robust=robust,
                    face_model=_fm,
                    face_detector=_fd,
                    text_detector=_td,
                )

            elapsed = round(time.time() - task["created"], 1)

            with _results_lock:
                _last_result_video = result_bytes

            task["result"] = {
                "success": True,
                "target_label": target_label,
                "mode": mode,
                "epsilon": epsilon,
                "steps": steps,
                "video_size_kb": round(len(result_bytes) / 1024, 1),
                "elapsed_sec": elapsed,
                "device": str(DEVICE),
            }
            task["status"] = "done"
            task["phase"] = "done"

        except Exception as e:
            _tasks[task_id]["status"] = "error"
            _tasks[task_id]["error"] = str(e)

    thread = threading.Thread(target=run_video_attack, daemon=True)
    thread.start()

    return jsonify({"task_id": task_id})


@app.route("/download_video")
def download_video():
    """Download the last generated adversarial video as MP4."""
    with _results_lock:
        data = _last_result_video
    if data is None:
        return "No video result available", 404
    buf = io.BytesIO(data)
    return send_file(buf, mimetype="video/mp4", as_attachment=True,
                     download_name="adversarial_video.mp4")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adversarial Attack Web App")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    args = parser.parse_args()
    app.run(debug=True, host=args.host, port=args.port, threaded=True)
